"""
Tests for cluster_analyser.py — GMM + BIC cluster analysis on coefficients
============================================================================

Covers:
- Insufficient data handling (no coefficients, too few samples)
- Single-cluster BIC selection for unimodal data
- Multi-cluster BIC selection for bimodal data
- Log-likelihood scoring and z-score classification
- Cluster model DB storage
- Waveform row update with cluster scores
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pytest

from litterbox.cluster_analyser import ClusterAnalyser
from litterbox.db import get_conn, init_db
from litterbox.visit_analyser import TdVisitRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 4, 4, 12, 0, 0, tzinfo=timezone.utc)


def _ts(offset_seconds: float = 0.0) -> datetime:
    return _BASE + timedelta(seconds=offset_seconds)


def _make_config(**overrides) -> dict:
    """Build a config dict with cluster and eigen sections."""
    cluster = {
        "min_samples_for_clustering": 20,
        "min_samples_per_cluster": 15,
        "max_clusters": 5,
        "n_init": 3,
        "z_score_thresholds": {
            "mild": -2.0,
            "significant": -3.0,
            "major": -4.0,
        },
    }
    cluster.update(overrides)
    return {
        "cluster": cluster,
        "eigen": {"uniform_n": 4},
    }


def _insert_cat(name: str) -> int:
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO cats (name) VALUES (?)", (name,))
        return cur.lastrowid


def _insert_td_visit() -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO td_visits (entry_time, exit_time, snapshot_json)
               VALUES (?, ?, ?)""",
            (_ts(0).isoformat(), _ts(50).isoformat(), "[]"),
        )
        return cur.lastrowid


def _make_record(
    cat_id: Optional[int] = None,
    td_visit_id: int = 1,
) -> TdVisitRecord:
    return TdVisitRecord(
        entry_time=_ts(0),
        exit_time=_ts(50),
        snapshot_json="[]",
        td_visit_id=td_visit_id,
        tentative_cat_id=cat_id,
    )


def _insert_scored_waveform(
    td_visit_id: int,
    cat_id: int,
    coefficients: list[float],
    channel: str = "weight_g",
) -> int:
    """Insert an eigen_waveforms row with coefficients (simulating EigenAnalyser output)."""
    vec = [0.0] * 64
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO eigen_waveforms
               (td_visit_id, cat_id, channel, vector_json, dc_term,
                coefficients_json, eigen_ev, eigen_residual,
                raw_length, nan_fraction)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                td_visit_id, cat_id, channel,
                json.dumps(vec), 5000.0,
                json.dumps(coefficients), 0.98, 1.0,
                64, 0.0,
            ),
        )
        return cur.lastrowid


def _populate_unimodal(cat_id: int, n: int = 30, seed: int = 42) -> None:
    """Insert n scored waveforms with unimodal 4-D coefficient vectors."""
    rng = np.random.RandomState(seed)
    center = np.array([10.0, -3.0, 1.5, 0.5])
    for i in range(n):
        td_visit_id = _insert_td_visit()
        coeffs = (center + rng.randn(4) * 0.5).tolist()
        # Pad to 64 total coefficients.
        full_coeffs = coeffs + [0.0] * 60
        _insert_scored_waveform(td_visit_id, cat_id, full_coeffs)


def _populate_bimodal(cat_id: int, n: int = 40, seed: int = 42) -> None:
    """Insert n scored waveforms with bimodal 4-D coefficient vectors."""
    rng = np.random.RandomState(seed)
    center_a = np.array([10.0, -3.0, 1.5, 0.5])
    center_b = np.array([-8.0, 5.0, -2.0, 3.0])
    for i in range(n):
        td_visit_id = _insert_td_visit()
        center = center_a if i % 2 == 0 else center_b
        coeffs = (center + rng.randn(4) * 0.3).tolist()
        full_coeffs = coeffs + [0.0] * 60
        _insert_scored_waveform(td_visit_id, cat_id, full_coeffs)


# ---------------------------------------------------------------------------
# TestInsufficientData
# ---------------------------------------------------------------------------


class TestInsufficientData:
    """Insufficient data returns early without clustering."""

    def test_no_coefficients(self):
        """Visit with no coefficients_json → insufficient_data."""
        cat_id = _insert_cat("NoCo")
        td_visit_id = _insert_td_visit()

        # Insert waveform without coefficients.
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO eigen_waveforms
                   (td_visit_id, cat_id, channel, vector_json, dc_term,
                    raw_length, nan_fraction)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (td_visit_id, cat_id, "weight_g", "[]", 5000.0, 64, 0.0),
            )

        config = _make_config()
        analyser = ClusterAnalyser(config)
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        result = analyser.analyse(np.zeros(64), record, "weight_g")

        assert result.anomaly_level == "insufficient_data"
        assert result.details["reason"] == "no_coefficients"

    def test_too_few_samples(self):
        """Fewer than min_samples_for_clustering → insufficient_data."""
        cat_id = _insert_cat("FewSamples")

        # Insert only 5 waveforms (min is 20).
        for i in range(5):
            td_visit_id = _insert_td_visit()
            _insert_scored_waveform(td_visit_id, cat_id, [1.0, 2.0, 3.0, 4.0] + [0.0] * 60)

        # New visit to analyse.
        new_td = _insert_td_visit()
        _insert_scored_waveform(new_td, cat_id, [1.1, 2.1, 3.1, 4.1] + [0.0] * 60)

        config = _make_config(min_samples_for_clustering=20)
        analyser = ClusterAnalyser(config)
        record = _make_record(cat_id=cat_id, td_visit_id=new_td)
        result = analyser.analyse(np.zeros(64), record, "weight_g")

        assert result.anomaly_level == "insufficient_data"
        assert result.details["reason"] == "not_enough_samples"

    def test_no_cat_id(self):
        """Visit with no cat_id → insufficient_data."""
        td_visit_id = _insert_td_visit()
        _insert_scored_waveform(td_visit_id, None, [1.0, 2.0, 3.0, 4.0] + [0.0] * 60)

        config = _make_config()
        analyser = ClusterAnalyser(config)
        record = _make_record(cat_id=None, td_visit_id=td_visit_id)
        result = analyser.analyse(np.zeros(64), record, "weight_g")

        assert result.anomaly_level == "insufficient_data"


# ---------------------------------------------------------------------------
# TestSingleCluster
# ---------------------------------------------------------------------------


class TestSingleCluster:
    """Unimodal data should produce k*=1."""

    def test_bic_selects_k1(self):
        cat_id = _insert_cat("UniCat")
        _populate_unimodal(cat_id, n=30)

        # New in-distribution visit.
        new_td = _insert_td_visit()
        rng = np.random.RandomState(999)
        coeffs = (np.array([10.0, -3.0, 1.5, 0.5]) + rng.randn(4) * 0.5).tolist()
        _insert_scored_waveform(new_td, cat_id, coeffs + [0.0] * 60)

        config = _make_config(min_samples_for_clustering=10)
        analyser = ClusterAnalyser(config)
        record = _make_record(cat_id=cat_id, td_visit_id=new_td)
        result = analyser.analyse(np.zeros(64), record, "weight_g")

        assert result.anomaly_level != "insufficient_data"
        assert result.details["k_clusters"] == 1

    def test_in_distribution_normal_z_score(self):
        """In-distribution point should have z-score near 0."""
        cat_id = _insert_cat("NormZ")
        _populate_unimodal(cat_id, n=30, seed=10)

        new_td = _insert_td_visit()
        coeffs = [10.0, -3.0, 1.5, 0.5]  # right at the center
        _insert_scored_waveform(new_td, cat_id, coeffs + [0.0] * 60)

        config = _make_config(min_samples_for_clustering=10)
        analyser = ClusterAnalyser(config)
        record = _make_record(cat_id=cat_id, td_visit_id=new_td)
        result = analyser.analyse(np.zeros(64), record, "weight_g")

        assert result.details["z_score"] > -2.0
        assert result.anomaly_level == "normal"


# ---------------------------------------------------------------------------
# TestMultipleClusters
# ---------------------------------------------------------------------------


class TestMultipleClusters:
    """Bimodal data should produce k*=2."""

    def test_bic_selects_k2(self):
        cat_id = _insert_cat("BiCat")
        _populate_bimodal(cat_id, n=40, seed=42)

        new_td = _insert_td_visit()
        coeffs = [10.0, -3.0, 1.5, 0.5]  # near cluster A
        _insert_scored_waveform(new_td, cat_id, coeffs + [0.0] * 60)

        config = _make_config(min_samples_for_clustering=10, min_samples_per_cluster=10)
        analyser = ClusterAnalyser(config)
        record = _make_record(cat_id=cat_id, td_visit_id=new_td)
        result = analyser.analyse(np.zeros(64), record, "weight_g")

        assert result.details["k_clusters"] == 2


# ---------------------------------------------------------------------------
# TestScoring
# ---------------------------------------------------------------------------


class TestScoring:
    """Log-likelihood scoring and outlier detection."""

    def test_outlier_low_z_score(self):
        """A point far from the cluster should have low z-score."""
        cat_id = _insert_cat("OutlierCat")
        _populate_unimodal(cat_id, n=30, seed=42)

        # Outlier: very far from center [10, -3, 1.5, 0.5].
        new_td = _insert_td_visit()
        coeffs = [100.0, 50.0, -80.0, 60.0]
        _insert_scored_waveform(new_td, cat_id, coeffs + [0.0] * 60)

        config = _make_config(min_samples_for_clustering=10)
        analyser = ClusterAnalyser(config)
        record = _make_record(cat_id=cat_id, td_visit_id=new_td)
        result = analyser.analyse(np.zeros(64), record, "weight_g")

        assert result.details["z_score"] < -2.0
        assert result.anomaly_level in ("mild", "significant", "major")


# ---------------------------------------------------------------------------
# TestModelStorage
# ---------------------------------------------------------------------------


class TestModelStorage:
    """Cluster model is stored in cluster_models table."""

    def test_model_row_created(self):
        cat_id = _insert_cat("ModelStoreCat")
        _populate_unimodal(cat_id, n=25, seed=42)

        new_td = _insert_td_visit()
        coeffs = [10.0, -3.0, 1.5, 0.5]
        _insert_scored_waveform(new_td, cat_id, coeffs + [0.0] * 60)

        config = _make_config(min_samples_for_clustering=10)
        analyser = ClusterAnalyser(config)
        record = _make_record(cat_id=cat_id, td_visit_id=new_td)
        result = analyser.analyse(np.zeros(64), record, "weight_g")

        model_id = result.details["cluster_model_id"]
        with get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM cluster_models WHERE cluster_model_id = ?",
                (model_id,),
            ).fetchone()

        assert row is not None
        assert row["cat_id"] == cat_id
        assert row["k_clusters"] >= 1
        assert row["uniform_n"] == 4
        means = json.loads(row["means_json"])
        assert len(means) == row["k_clusters"]
        assert len(means[0]) == 4
        covariances = json.loads(row["covariances_json"])
        assert len(covariances) == row["k_clusters"]
        assert len(covariances[0]) == 4
        assert len(covariances[0][0]) == 4


# ---------------------------------------------------------------------------
# TestWaveformUpdate
# ---------------------------------------------------------------------------


class TestWaveformUpdate:
    """eigen_waveforms row is updated with cluster scores."""

    def test_cluster_fields_populated(self):
        cat_id = _insert_cat("UpdateCat")
        _populate_unimodal(cat_id, n=25, seed=42)

        new_td = _insert_td_visit()
        coeffs = [10.0, -3.0, 1.5, 0.5]
        _insert_scored_waveform(new_td, cat_id, coeffs + [0.0] * 60)

        config = _make_config(min_samples_for_clustering=10)
        analyser = ClusterAnalyser(config)
        record = _make_record(cat_id=cat_id, td_visit_id=new_td)
        analyser.analyse(np.zeros(64), record, "weight_g")

        with get_conn() as conn:
            row = conn.execute(
                """SELECT cluster_log_likelihood, cluster_z_score,
                          cluster_assignment, cluster_model_id
                   FROM eigen_waveforms
                   WHERE td_visit_id = ? AND channel = ?""",
                (new_td, "weight_g"),
            ).fetchone()

        assert row["cluster_log_likelihood"] is not None
        assert row["cluster_z_score"] is not None
        assert row["cluster_assignment"] is not None
        assert row["cluster_model_id"] is not None


# ---------------------------------------------------------------------------
# TestClassification
# ---------------------------------------------------------------------------


class TestClassification:
    """Z-score thresholds map to correct anomaly levels."""

    def test_classify_normal(self):
        analyser = ClusterAnalyser(_make_config())
        level, score = analyser._classify(-0.5)
        assert level == "normal"
        assert score == 0.0

    def test_classify_mild(self):
        analyser = ClusterAnalyser(_make_config())
        level, score = analyser._classify(-2.5)
        assert level == "mild"
        assert 0.0 < score < 0.34

    def test_classify_significant(self):
        analyser = ClusterAnalyser(_make_config())
        level, score = analyser._classify(-3.5)
        assert level == "significant"
        assert 0.33 <= score < 0.67

    def test_classify_major(self):
        analyser = ClusterAnalyser(_make_config())
        level, score = analyser._classify(-5.0)
        assert level == "major"
        assert score == 1.0

    def test_boundary_at_mild(self):
        analyser = ClusterAnalyser(_make_config())
        level, _ = analyser._classify(-2.0)
        assert level == "normal"  # >= -2.0 is normal

    def test_boundary_below_mild(self):
        analyser = ClusterAnalyser(_make_config())
        level, _ = analyser._classify(-2.01)
        assert level == "mild"
