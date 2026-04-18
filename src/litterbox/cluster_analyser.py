"""
cluster_analyser.py — GMM + BIC cluster analysis on expansion coefficients
============================================================================

This plugin implements Layer 2 anomaly detection: **in-subspace anomaly
detection** via Gaussian Mixture Model clustering on the N-dimensional
expansion coefficient vectors produced by the EigenAnalyser.

Even when a visit's explained variance is high (Layer 1 passes), the
coefficient vector may place the visit in an unusual region of the cat's
coefficient space.  This module discovers the cluster structure via BIC
model selection and scores each new visit by log-likelihood against the
fitted mixture.

Workflow per visit
------------------
1. Read the coefficient vector from ``eigen_waveforms`` (written by
   EigenAnalyser, which runs first in the pipeline).
2. Load all scored coefficient vectors for this cat.
3. If enough data exists, fit GMMs for k=1..k_max, select k* via BIC.
4. Score the new visit: compute log-likelihood and z-score against
   the training distribution of log-likelihoods.
5. Store the cluster model and per-visit scores.

Requires ``scikit-learn`` for ``GaussianMixture``.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np

from litterbox.analyser_pipeline import AnalysisResult, BaseAnalyser
from litterbox.db import get_conn, init_db
from litterbox.visit_analyser import TdVisitRecord

logger = logging.getLogger(__name__)


class ClusterAnalyser(BaseAnalyser):
    """GMM + BIC cluster analysis plugin.

    Parameters
    ----------
    config:
        Parsed ``td_config.json`` dict.  The ``"cluster"`` and ``"eigen"``
        sections are read for parameters.
    """

    def __init__(self, config: dict) -> None:
        cluster_cfg = config.get("cluster", {})
        eigen_cfg = config.get("eigen", {})

        self._min_samples: int = int(
            cluster_cfg.get("min_samples_for_clustering", 20)
        )
        self._min_per_cluster: int = int(
            cluster_cfg.get("min_samples_per_cluster", 15)
        )
        self._max_clusters: int = int(
            cluster_cfg.get("max_clusters", 5)
        )
        self._n_init: int = int(cluster_cfg.get("n_init", 5))

        thresholds = cluster_cfg.get("z_score_thresholds", {})
        self._z_mild: float = float(thresholds.get("mild", -2.0))
        self._z_significant: float = float(thresholds.get("significant", -3.0))
        self._z_major: float = float(thresholds.get("major", -4.0))

        self._uniform_n: int = int(eigen_cfg.get("uniform_n", 4))

    @property
    def name(self) -> str:
        return "cluster"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyse(
        self,
        waveform: np.ndarray,
        visit_record: TdVisitRecord,
        channel: str,
    ) -> AnalysisResult:
        """Score a visit's expansion coefficients via GMM clustering."""
        init_db()

        cat_id = (
            visit_record.confirmed_cat_id or visit_record.tentative_cat_id
        )
        td_visit_id = visit_record.td_visit_id

        # --- Load this visit's coefficients from eigen_waveforms ---
        coeffs = self._load_visit_coefficients(td_visit_id, channel)
        if coeffs is None:
            return AnalysisResult(
                plugin_name="cluster",
                anomaly_score=0.0,
                anomaly_level="insufficient_data",
                details={"reason": "no_coefficients"},
            )

        # Take first N (uniform_n) coefficients.
        x = np.array(coeffs[: self._uniform_n])

        if cat_id is None:
            return AnalysisResult(
                plugin_name="cluster",
                anomaly_score=0.0,
                anomaly_level="insufficient_data",
                details={"reason": "no_cat_id"},
            )

        # --- Load all scored coefficient vectors for this cat ---
        # Exclude the current visit so the model doesn't train on the
        # point it's about to score.
        all_coeffs = self._load_cat_coefficients(
            cat_id, channel, exclude_td_visit_id=td_visit_id,
        )
        K = len(all_coeffs)

        if K < self._min_samples:
            return AnalysisResult(
                plugin_name="cluster",
                anomaly_score=0.0,
                anomaly_level="insufficient_data",
                details={
                    "reason": "not_enough_samples",
                    "k_samples": K,
                    "min_required": self._min_samples,
                },
            )

        X = np.array(all_coeffs)  # (K, N)

        # --- GMM + BIC model selection ---
        k_max = min(self._max_clusters, K // self._min_per_cluster)
        k_max = max(k_max, 1)

        from sklearn.mixture import GaussianMixture

        bic_values = []
        gmms = []
        for k in range(1, k_max + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=self._n_init,
                random_state=42,
            )
            gmm.fit(X)
            bic_values.append(gmm.bic(X))
            gmms.append(gmm)

        best_idx = int(np.argmin(bic_values))
        best_gmm = gmms[best_idx]
        k_star = best_idx + 1

        # --- Score the new visit ---
        x_2d = x.reshape(1, -1)
        log_likelihood = float(best_gmm.score_samples(x_2d)[0])
        cluster_assignment = int(best_gmm.predict(x_2d)[0])

        # --- Compute z-score against training log-likelihoods ---
        train_ll = best_gmm.score_samples(X)  # (K,)
        mean_ll = float(np.mean(train_ll))
        std_ll = float(np.std(train_ll))

        if std_ll > 1e-10:
            z_score = (log_likelihood - mean_ll) / std_ll
        else:
            z_score = 0.0

        # --- Classify ---
        anomaly_level, anomaly_score = self._classify(z_score)

        # --- Save cluster model ---
        cluster_model_id = self._save_model(
            cat_id, channel, best_gmm, k_star, bic_values, K,
        )

        # --- Update eigen_waveforms row ---
        self._update_waveform_cluster(
            td_visit_id, channel, log_likelihood, z_score,
            cluster_assignment, cluster_model_id,
        )

        return AnalysisResult(
            plugin_name="cluster",
            anomaly_score=anomaly_score,
            anomaly_level=anomaly_level,
            details={
                "k_clusters": k_star,
                "log_likelihood": log_likelihood,
                "z_score": z_score,
                "cluster_assignment": cluster_assignment,
                "bic_values": bic_values,
                "k_max_tried": k_max,
                "mean_ll": mean_ll,
                "std_ll": std_ll,
                "cluster_model_id": cluster_model_id,
            },
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self, z_score: float) -> tuple[str, float]:
        """Map z-score to (anomaly_level, anomaly_score)."""
        if z_score >= self._z_mild:
            return "normal", 0.0
        elif z_score >= self._z_significant:
            frac = (self._z_mild - z_score) / (self._z_mild - self._z_significant)
            return "mild", round(frac * 0.33, 4)
        elif z_score >= self._z_major:
            frac = (self._z_significant - z_score) / (
                self._z_significant - self._z_major
            )
            return "significant", round(0.33 + frac * 0.34, 4)
        else:
            return "major", 1.0

    # ------------------------------------------------------------------
    # DB reads
    # ------------------------------------------------------------------

    def _load_visit_coefficients(
        self, td_visit_id: int, channel: str,
    ) -> Optional[list[float]]:
        """Load coefficients for a specific visit from eigen_waveforms."""
        with get_conn() as conn:
            row = conn.execute(
                """SELECT coefficients_json FROM eigen_waveforms
                   WHERE td_visit_id = ? AND channel = ?
                   ORDER BY waveform_id DESC LIMIT 1""",
                (td_visit_id, channel),
            ).fetchone()

        if row is None or row["coefficients_json"] is None:
            return None
        return json.loads(row["coefficients_json"])

    def _load_cat_coefficients(
        self,
        cat_id: int,
        channel: str,
        exclude_td_visit_id: Optional[int] = None,
    ) -> list[list[float]]:
        """Load all scored N-dimensional coefficient vectors for a cat."""
        with get_conn() as conn:
            if exclude_td_visit_id is not None:
                rows = conn.execute(
                    """SELECT coefficients_json FROM eigen_waveforms
                       WHERE cat_id = ? AND channel = ?
                         AND coefficients_json IS NOT NULL
                         AND td_visit_id != ?""",
                    (cat_id, channel, exclude_td_visit_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT coefficients_json FROM eigen_waveforms
                       WHERE cat_id = ? AND channel = ? AND coefficients_json IS NOT NULL""",
                    (cat_id, channel),
                ).fetchall()

        result = []
        for r in rows:
            coeffs = json.loads(r["coefficients_json"])
            result.append(coeffs[: self._uniform_n])
        return result

    # ------------------------------------------------------------------
    # DB writes
    # ------------------------------------------------------------------

    def _save_model(
        self,
        cat_id: int,
        channel: str,
        gmm,
        k_clusters: int,
        bic_values: list[float],
        n_samples: int,
    ) -> int:
        """Insert a cluster_models row. Returns cluster_model_id."""
        with get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO cluster_models
                   (cat_id, channel, k_clusters, means_json, covariances_json,
                    weights_json, bic_values_json, n_samples, uniform_n)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cat_id,
                    channel,
                    k_clusters,
                    json.dumps(gmm.means_.tolist()),
                    json.dumps(gmm.covariances_.tolist()),
                    json.dumps(gmm.weights_.tolist()),
                    json.dumps(bic_values),
                    n_samples,
                    self._uniform_n,
                ),
            )
            return cur.lastrowid

    def _update_waveform_cluster(
        self,
        td_visit_id: int,
        channel: str,
        log_likelihood: float,
        z_score: float,
        cluster_assignment: int,
        cluster_model_id: int,
    ) -> None:
        """Update the eigen_waveforms row with cluster scores."""
        with get_conn() as conn:
            conn.execute(
                """UPDATE eigen_waveforms
                   SET cluster_log_likelihood = ?,
                       cluster_z_score = ?,
                       cluster_assignment = ?,
                       cluster_model_id = ?
                   WHERE td_visit_id = ? AND channel = ?""",
                (
                    log_likelihood,
                    z_score,
                    cluster_assignment,
                    cluster_model_id,
                    td_visit_id,
                    channel,
                ),
            )
