"""
eigen_analyser.py — Step 5b of the Time-Domain Measurement System
==================================================================

Eigendecomposition-based waveform anomaly detection plugin.

This module implements ``BaseAnalyser`` from ``analyser_pipeline.py``.
For each visit it:

1. Mean-subtracts the resampled waveform and stores the DC term.
2. Archives the zero-mean waveform in ``eigen_waveforms``.
3. Selects the appropriate model (per-cat or pooled) based on data volume.
4. Recomputes the autocovariance matrix and eigendecomposition.
5. Projects the new waveform, computes explained variance, and stores the
   full expansion coefficient vector.
6. Classifies the anomaly level and returns an ``AnalysisResult``.

Two-layer anomaly detection
---------------------------
**Layer 1 — Explained Variance (out-of-subspace):**
    If the waveform has significant energy in directions the cat's history
    has never seen, EV drops.  This catches sensor failures and dramatic
    behavioural anomalies.

**Layer 2 — Coefficient Clustering (in-subspace):**
    Even if EV is high, the expansion coefficients may place the visit in
    an unusual region of the signal subspace.  This layer is deferred to a
    future ML plugin that consumes the stored coefficients.  The current
    module stores the full coefficient vector to enable that future work.

Regularization
--------------
When operating in per-cat mode with K < L, the covariance matrix is
rank-deficient.  Tikhonov regularization (C + αI) lifts the zero
eigenvalues so the noise subspace has small uniform variance rather than
zero.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from litterbox.analyser_pipeline import AnalysisResult, BaseAnalyser
from litterbox.db import get_conn, init_db
from litterbox.visit_analyser import TdVisitRecord

logger = logging.getLogger(__name__)


class EigenAnalyser(BaseAnalyser):
    """Eigendecomposition plugin for time-domain waveform anomaly detection.

    Parameters
    ----------
    config:
        Parsed ``td_config.json`` dict.  The ``"eigen"`` section is read
        for all tunable parameters; sensible defaults are used if absent.
    """

    def __init__(self, config: dict) -> None:
        eigen_cfg = config.get("eigen", {})
        self._L: int = int(eigen_cfg.get("resample_length", 64))
        self._ev_threshold: float = float(
            eigen_cfg.get("explained_variance_threshold", 0.95)
        )
        self._pooled_min: int = int(eigen_cfg.get("pooled_minimum", 128))
        self._per_cat_min: int = int(eigen_cfg.get("per_cat_minimum", 32))
        self._reg_eps: float = float(
            eigen_cfg.get("regularization_epsilon", 0.01)
        )
        self._max_nan: float = float(
            eigen_cfg.get("max_nan_fraction", 0.25)
        )
        thresholds = eigen_cfg.get("anomaly_thresholds", {})
        self._thresh_normal: float = float(thresholds.get("normal", 0.90))
        self._thresh_mild: float = float(thresholds.get("mild", 0.70))
        self._thresh_significant: float = float(
            thresholds.get("significant", 0.40)
        )

    @property
    def name(self) -> str:
        return "eigen"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyse(
        self,
        waveform: np.ndarray,
        visit_record: TdVisitRecord,
        channel: str,
    ) -> AnalysisResult:
        """Analyse a resampled waveform via eigendecomposition.

        Parameters
        ----------
        waveform:
            1-D numpy array of length L, resampled by the pipeline.
            Not mean-subtracted.
        visit_record:
            The persisted ``TdVisitRecord`` for this visit.
        channel:
            Sensor channel name (e.g. ``"weight_g"``).

        Returns
        -------
        AnalysisResult
        """
        init_db()

        cat_id = (
            visit_record.confirmed_cat_id or visit_record.tentative_cat_id
        )
        td_visit_id = visit_record.td_visit_id

        # --- NaN fraction check ---
        nan_fraction = float(np.isnan(waveform).sum() / len(waveform))
        raw_length = len(waveform)

        if nan_fraction > self._max_nan:
            waveform_clean = np.where(np.isnan(waveform), 0.0, waveform)
            dc_term = float(np.nanmean(waveform)) if not np.isnan(waveform).all() else 0.0
            x = waveform_clean - dc_term
            self._store_waveform(
                td_visit_id, cat_id, channel, x, dc_term,
                raw_length, nan_fraction,
            )
            return AnalysisResult(
                plugin_name="eigen",
                anomaly_score=0.0,
                anomaly_level="insufficient_data",
                details={
                    "reason": "nan_fraction_exceeded",
                    "nan_fraction": nan_fraction,
                    "dc_term": dc_term,
                },
            )

        # --- DC subtraction ---
        dc_term = float(np.nanmean(waveform))
        x = waveform - dc_term
        # Replace any remaining NaN with 0.0 for linear algebra.
        x = np.where(np.isnan(x), 0.0, x)

        # --- Store waveform ---
        waveform_id = self._store_waveform(
            td_visit_id, cat_id, channel, x, dc_term,
            raw_length, nan_fraction,
        )

        # --- Model selection ---
        k_cat = self._count_waveforms(channel, cat_id=cat_id) if cat_id else 0
        k_all = self._count_waveforms(channel, cat_id=None)

        if cat_id and k_cat >= self._per_cat_min:
            model_type = "per_cat"
            vectors = self._load_vectors(channel, cat_id=cat_id)
        elif k_all >= self._pooled_min:
            model_type = "pooled"
            vectors = self._load_vectors(channel, cat_id=None)
        else:
            return AnalysisResult(
                plugin_name="eigen",
                anomaly_score=0.0,
                anomaly_level="insufficient_data",
                details={
                    "reason": "not_enough_waveforms",
                    "k_cat": k_cat,
                    "k_all": k_all,
                    "dc_term": dc_term,
                },
            )

        # --- Covariance matrix ---
        X = np.array(vectors)  # (K, L)
        K = X.shape[0]
        C = X.T @ X / (K - 1)  # (L, L)

        # --- Regularization ---
        regularized = False
        if model_type == "per_cat" and K < self._L:
            alpha = self._reg_eps * np.trace(C) / self._L
            C += alpha * np.eye(self._L)
            regularized = True

        # --- Eigendecomposition ---
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        # eigh returns ascending order; reverse to descending.
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clamp any tiny negative eigenvalues from numerical noise.
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # --- Select N components ---
        total_variance = eigenvalues.sum()
        if total_variance == 0:
            n_components = self._L
            cum_ratio = np.ones(self._L)
        else:
            cum_ratio = np.cumsum(eigenvalues) / total_variance
            n_components = int(np.searchsorted(cum_ratio, self._ev_threshold) + 1)
            n_components = min(n_components, self._L)

        # --- Save model ---
        model_cat_id = cat_id if model_type == "per_cat" else None
        model_id = self._save_model(
            model_cat_id, channel, eigenvalues, eigenvectors,
            n_components, K, regularized,
        )

        # --- Project and score ---
        V_N = eigenvectors[:, :n_components]       # (L, N)
        coefficients = eigenvectors.T @ x           # full L coefficients
        x_hat = V_N @ coefficients[:n_components]   # reconstruction

        x_norm_sq = float(np.dot(x, x))
        residual_sq = float(np.dot(x - x_hat, x - x_hat))

        if x_norm_sq == 0:
            ev = 1.0
        else:
            ev = 1.0 - residual_sq / x_norm_sq

        residual = float(np.sqrt(residual_sq))

        # --- Update waveform row with scoring results ---
        self._update_waveform_scores(
            waveform_id, coefficients, ev, residual, model_id,
        )

        # --- Classify ---
        anomaly_level, anomaly_score = self._classify(ev)

        return AnalysisResult(
            plugin_name="eigen",
            anomaly_score=anomaly_score,
            anomaly_level=anomaly_level,
            details={
                "ev": ev,
                "residual": residual,
                "dc_term": dc_term,
                "n_components": n_components,
                "model_type": model_type,
                "model_id": model_id,
                "k_waveforms": K,
                "regularized": regularized,
                "coefficients": coefficients.tolist(),
                "nan_fraction": nan_fraction,
            },
        )

    # ------------------------------------------------------------------
    # Anomaly classification
    # ------------------------------------------------------------------

    def _classify(self, ev: float) -> tuple[str, float]:
        """Map explained variance to (anomaly_level, anomaly_score)."""
        if ev >= self._thresh_normal:
            return "normal", 0.0
        elif ev >= self._thresh_mild:
            # Linear interpolation in [mild, normal] → score [0.33, 0]
            frac = (self._thresh_normal - ev) / (
                self._thresh_normal - self._thresh_mild
            )
            return "mild", round(frac * 0.33, 4)
        elif ev >= self._thresh_significant:
            # Linear interpolation in [significant, mild] → score [0.67, 0.33]
            frac = (self._thresh_mild - ev) / (
                self._thresh_mild - self._thresh_significant
            )
            return "significant", round(0.33 + frac * 0.34, 4)
        else:
            # Below significant threshold → major.
            return "major", 1.0

    # ------------------------------------------------------------------
    # DB operations
    # ------------------------------------------------------------------

    def _store_waveform(
        self,
        td_visit_id: Optional[int],
        cat_id: Optional[int],
        channel: str,
        vector: np.ndarray,
        dc_term: float,
        raw_length: int,
        nan_fraction: float,
    ) -> int:
        """Insert a row into eigen_waveforms. Returns waveform_id."""
        with get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO eigen_waveforms
                   (td_visit_id, cat_id, channel, vector_json, dc_term,
                    raw_length, nan_fraction)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    td_visit_id,
                    cat_id,
                    channel,
                    json.dumps(vector.tolist()),
                    dc_term,
                    raw_length,
                    nan_fraction,
                ),
            )
            return cur.lastrowid

    def _update_waveform_scores(
        self,
        waveform_id: int,
        coefficients: np.ndarray,
        ev: float,
        residual: float,
        model_id: int,
    ) -> None:
        """Update an eigen_waveforms row with scoring results."""
        with get_conn() as conn:
            conn.execute(
                """UPDATE eigen_waveforms
                   SET coefficients_json = ?, eigen_ev = ?, eigen_residual = ?,
                       model_id = ?
                   WHERE waveform_id = ?""",
                (
                    json.dumps(coefficients.tolist()),
                    ev,
                    residual,
                    model_id,
                    waveform_id,
                ),
            )

    def _save_model(
        self,
        cat_id: Optional[int],
        channel: str,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        n_components: int,
        n_waveforms: int,
        regularized: bool,
    ) -> int:
        """Insert a new eigen_models row. Returns model_id."""
        with get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO eigen_models
                   (cat_id, channel, eigenvalues_json, eigenvectors_json,
                    n_components, n_waveforms, explained_variance_threshold,
                    regularized)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cat_id,
                    channel,
                    json.dumps(eigenvalues.tolist()),
                    json.dumps(eigenvectors.tolist()),
                    n_components,
                    n_waveforms,
                    self._ev_threshold,
                    regularized,
                ),
            )
            return cur.lastrowid

    def _count_waveforms(
        self, channel: str, cat_id: Optional[int],
    ) -> int:
        """Count stored waveforms for a channel, optionally filtered by cat."""
        with get_conn() as conn:
            if cat_id is not None:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM eigen_waveforms "
                    "WHERE channel = ? AND cat_id = ?",
                    (channel, cat_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM eigen_waveforms "
                    "WHERE channel = ?",
                    (channel,),
                ).fetchone()
        return row["cnt"]

    def _load_vectors(
        self, channel: str, cat_id: Optional[int],
    ) -> list[list[float]]:
        """Load all stored zero-mean vectors for a channel/cat combination."""
        with get_conn() as conn:
            if cat_id is not None:
                rows = conn.execute(
                    "SELECT vector_json FROM eigen_waveforms "
                    "WHERE channel = ? AND cat_id = ?",
                    (channel, cat_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT vector_json FROM eigen_waveforms "
                    "WHERE channel = ?",
                    (channel,),
                ).fetchall()
        return [json.loads(r["vector_json"]) for r in rows]
