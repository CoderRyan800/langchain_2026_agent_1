"""
Tests for eigen_analyser.py — Step 5b of the Time-Domain Measurement System
=============================================================================

Covers:
- NaN handling and waveform flagging
- DC subtraction correctness
- Waveform storage and round-trip
- Model selection (per-cat vs. pooled vs. insufficient)
- Covariance and eigendecomposition correctness
- Tikhonov regularization for rank-deficient cases
- Explained variance calculation
- Expansion coefficient storage
- Anomaly classification thresholds
- Full round-trip: synthetic subspace waveforms vs. noise
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pytest

from litterbox.db import get_conn
from litterbox.eigen_analyser import EigenAnalyser
from litterbox.visit_analyser import TdVisitRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 4, 4, 12, 0, 0, tzinfo=timezone.utc)


def _ts(offset_seconds: float = 0.0) -> datetime:
    return _BASE + timedelta(seconds=offset_seconds)


def _make_config(**overrides) -> dict:
    """Build a minimal config dict with eigen section."""
    eigen = {
        "resample_length": 64,
        "explained_variance_threshold": 0.95,
        "pooled_minimum": 128,
        "per_cat_minimum": 32,
        "regularization_epsilon": 0.01,
        "max_nan_fraction": 0.25,
        "anomaly_thresholds": {
            "normal": 0.90,
            "mild": 0.70,
            "significant": 0.40,
        },
    }
    eigen.update(overrides)
    return {"eigen": eigen}


def _make_record(
    cat_id: Optional[int] = None,
    td_visit_id: int = 1,
) -> TdVisitRecord:
    """Create a minimal TdVisitRecord for tests."""
    return TdVisitRecord(
        entry_time=_ts(0),
        exit_time=_ts(50),
        snapshot_json="[]",
        td_visit_id=td_visit_id,
        tentative_cat_id=cat_id,
    )


def _insert_cat(name: str) -> int:
    """Insert a cat and return cat_id."""
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO cats (name) VALUES (?)", (name,))
        return cur.lastrowid


def _insert_td_visit() -> int:
    """Insert a minimal td_visits row and return td_visit_id."""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO td_visits (entry_time, exit_time, snapshot_json)
               VALUES (?, ?, ?)""",
            (_ts(0).isoformat(), _ts(50).isoformat(), "[]"),
        )
        return cur.lastrowid


def _insert_waveforms(
    channel: str,
    cat_id: Optional[int],
    vectors: list[np.ndarray],
) -> list[int]:
    """Insert pre-computed waveforms directly into eigen_waveforms.

    Returns list of waveform_ids.
    """
    ids = []
    td_visit_id = _insert_td_visit()
    with get_conn() as conn:
        for vec in vectors:
            cur = conn.execute(
                """INSERT INTO eigen_waveforms
                   (td_visit_id, cat_id, channel, vector_json, dc_term,
                    raw_length, nan_fraction)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    td_visit_id,
                    cat_id,
                    channel,
                    json.dumps(vec.tolist()),
                    0.0,
                    len(vec),
                    0.0,
                ),
            )
            ids.append(cur.lastrowid)
    return ids


def _make_subspace_waveforms(
    n_waveforms: int,
    L: int = 64,
    n_basis: int = 3,
    noise_sigma: float = 0.01,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate waveforms that live in a low-dimensional subspace.

    Each waveform is a random linear combination of n_basis sinusoidal
    basis vectors plus small noise.  All waveforms are zero-mean.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, L)
    basis = [np.sin((k + 1) * t) for k in range(n_basis)]

    waveforms = []
    for _ in range(n_waveforms):
        coeffs = rng.randn(n_basis)
        wf = sum(c * b for c, b in zip(coeffs, basis))
        wf += rng.randn(L) * noise_sigma
        wf -= wf.mean()  # zero-mean
        waveforms.append(wf)

    return waveforms


# ---------------------------------------------------------------------------
# TestNanHandling
# ---------------------------------------------------------------------------


class TestNanHandling:
    """NaN fraction check and flagging."""

    def test_high_nan_fraction_returns_insufficient_data(self):
        """Waveform with >25% NaN → insufficient_data, but waveform is stored."""
        td_visit_id = _insert_td_visit()
        config = _make_config()
        analyser = EigenAnalyser(config)

        waveform = np.full(64, np.nan)
        waveform[:10] = np.linspace(1, 10, 10)  # 10/64 = 15.6% valid
        # 54/64 ≈ 84% NaN > 25%

        record = _make_record(td_visit_id=td_visit_id)
        result = analyser.analyse(waveform, record, "weight_g")

        assert result.anomaly_level == "insufficient_data"
        assert result.details["reason"] == "nan_fraction_exceeded"

        # Waveform should still be stored.
        with get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM eigen_waveforms WHERE td_visit_id = ?",
                (td_visit_id,),
            ).fetchone()
        assert row["cnt"] == 1

    def test_acceptable_nan_fraction_proceeds(self):
        """Waveform with ≤25% NaN proceeds to analysis (or insufficient_data for model)."""
        td_visit_id = _insert_td_visit()
        config = _make_config()
        analyser = EigenAnalyser(config)

        waveform = np.linspace(100, 200, 64)
        waveform[0:10] = np.nan  # 10/64 = 15.6% NaN ≤ 25%

        record = _make_record(td_visit_id=td_visit_id)
        result = analyser.analyse(waveform, record, "weight_g")

        # Should not be "nan_fraction_exceeded" — it proceeds to model selection,
        # which will return insufficient_data because no waveforms are stored yet.
        assert result.details.get("reason") != "nan_fraction_exceeded"

    def test_zero_nan_fraction(self):
        """Clean waveform with no NaN."""
        td_visit_id = _insert_td_visit()
        config = _make_config()
        analyser = EigenAnalyser(config)

        waveform = np.linspace(100, 200, 64)
        record = _make_record(td_visit_id=td_visit_id)
        result = analyser.analyse(waveform, record, "weight_g")

        assert result.details.get("nan_fraction", 0.0) == 0.0


# ---------------------------------------------------------------------------
# TestDcSubtraction
# ---------------------------------------------------------------------------


class TestDcSubtraction:
    """DC term (mean) is correctly removed and stored."""

    def test_dc_term_is_correct_mean(self):
        """dc_term should equal the mean of the waveform."""
        td_visit_id = _insert_td_visit()
        config = _make_config()
        analyser = EigenAnalyser(config)

        waveform = np.array([100.0] * 32 + [200.0] * 32)
        expected_dc = 150.0

        record = _make_record(td_visit_id=td_visit_id)
        result = analyser.analyse(waveform, record, "weight_g")

        assert result.details["dc_term"] == pytest.approx(expected_dc)

    def test_stored_vector_is_zero_mean(self):
        """The vector stored in eigen_waveforms should sum to approximately 0."""
        td_visit_id = _insert_td_visit()
        config = _make_config()
        analyser = EigenAnalyser(config)

        waveform = np.linspace(1000, 2000, 64)
        record = _make_record(td_visit_id=td_visit_id)
        analyser.analyse(waveform, record, "weight_g")

        with get_conn() as conn:
            row = conn.execute(
                "SELECT vector_json FROM eigen_waveforms WHERE td_visit_id = ?",
                (td_visit_id,),
            ).fetchone()

        vec = np.array(json.loads(row["vector_json"]))
        assert abs(vec.sum()) < 1e-10


# ---------------------------------------------------------------------------
# TestWaveformStorage
# ---------------------------------------------------------------------------


class TestWaveformStorage:
    """Waveform insertion and JSON round-trip."""

    def test_waveform_row_created(self):
        """analyse() creates an eigen_waveforms row."""
        td_visit_id = _insert_td_visit()
        config = _make_config()
        analyser = EigenAnalyser(config)

        waveform = np.linspace(0, 100, 64)
        record = _make_record(td_visit_id=td_visit_id)
        analyser.analyse(waveform, record, "weight_g")

        with get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM eigen_waveforms WHERE td_visit_id = ?",
                (td_visit_id,),
            ).fetchone()

        assert row is not None
        assert row["channel"] == "weight_g"
        assert row["raw_length"] == 64
        assert row["nan_fraction"] == pytest.approx(0.0)

    def test_vector_json_round_trip(self):
        """Stored vector_json deserializes back to the correct array."""
        td_visit_id = _insert_td_visit()
        config = _make_config()
        analyser = EigenAnalyser(config)

        waveform = np.linspace(0, 100, 64)
        dc = waveform.mean()
        expected_vec = waveform - dc

        record = _make_record(td_visit_id=td_visit_id)
        analyser.analyse(waveform, record, "weight_g")

        with get_conn() as conn:
            row = conn.execute(
                "SELECT vector_json FROM eigen_waveforms WHERE td_visit_id = ?",
                (td_visit_id,),
            ).fetchone()

        restored = np.array(json.loads(row["vector_json"]))
        np.testing.assert_array_almost_equal(restored, expected_vec)


# ---------------------------------------------------------------------------
# TestModelSelection
# ---------------------------------------------------------------------------


class TestModelSelection:
    """Model selection logic: per-cat, pooled, or insufficient."""

    def test_insufficient_data(self):
        """No stored waveforms → insufficient_data."""
        td_visit_id = _insert_td_visit()
        config = _make_config(pooled_minimum=5, per_cat_minimum=3)
        analyser = EigenAnalyser(config)

        waveform = np.linspace(0, 100, 64)
        record = _make_record(td_visit_id=td_visit_id)
        result = analyser.analyse(waveform, record, "weight_g")

        assert result.anomaly_level == "insufficient_data"
        assert result.details["reason"] == "not_enough_waveforms"

    def test_pooled_model_used(self):
        """When K_all ≥ pooled_min but K_cat < per_cat_min → pooled model."""
        cat_id = _insert_cat("TestCat")
        other_cat_id = _insert_cat("OtherCat")

        # Use small thresholds for testing.
        config = _make_config(pooled_minimum=10, per_cat_minimum=8)
        analyser = EigenAnalyser(config)

        # Insert 6 waveforms for each cat (total 12 ≥ 10 pooled min).
        vecs = _make_subspace_waveforms(6, L=64)
        _insert_waveforms("weight_g", cat_id, vecs)
        _insert_waveforms("weight_g", other_cat_id, vecs)

        td_visit_id = _insert_td_visit()
        waveform = vecs[0].copy()  # in-distribution
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        result = analyser.analyse(waveform, record, "weight_g")

        assert result.anomaly_level != "insufficient_data"
        assert result.details["model_type"] == "pooled"

    def test_per_cat_model_used(self):
        """When K_cat ≥ per_cat_min → per-cat model."""
        cat_id = _insert_cat("TestCat")

        config = _make_config(pooled_minimum=128, per_cat_minimum=5)
        analyser = EigenAnalyser(config)

        vecs = _make_subspace_waveforms(6, L=64)
        _insert_waveforms("weight_g", cat_id, vecs)

        td_visit_id = _insert_td_visit()
        waveform = vecs[0].copy()
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        result = analyser.analyse(waveform, record, "weight_g")

        assert result.anomaly_level != "insufficient_data"
        assert result.details["model_type"] == "per_cat"


# ---------------------------------------------------------------------------
# TestCovarianceAndEigen
# ---------------------------------------------------------------------------


class TestCovarianceAndEigen:
    """Covariance matrix and eigendecomposition correctness."""

    def test_eigenvalues_match_numpy_cov(self):
        """Eigenvalues from the analyser should match numpy's covariance computation.

        Since K < L here (21 waveforms, L=64), the model is regularized.
        We replicate that regularization in the reference to get a match.
        """
        cat_id = _insert_cat("EigenCat")
        L = 64
        eps = 0.01  # regularization_epsilon default

        vecs = _make_subspace_waveforms(20, L=L, n_basis=3, seed=123)
        _insert_waveforms("weight_g", cat_id, vecs)

        config = _make_config(per_cat_minimum=5, regularization_epsilon=eps)
        analyser = EigenAnalyser(config)

        td_visit_id = _insert_td_visit()
        waveform = vecs[0].copy()
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        result = analyser.analyse(waveform, record, "weight_g")

        # Load the stored model.
        model_id = result.details["model_id"]
        with get_conn() as conn:
            row = conn.execute(
                "SELECT eigenvalues_json FROM eigen_models WHERE model_id = ?",
                (model_id,),
            ).fetchone()

        stored_eigenvalues = np.array(json.loads(row["eigenvalues_json"]))

        # Compute reference including the new waveform (already zero-mean).
        all_vecs = vecs + [waveform]
        X = np.array(all_vecs)
        K = X.shape[0]
        C_ref = X.T @ X / (K - 1)

        # Apply the same regularization the analyser uses.
        alpha = eps * np.trace(C_ref) / L
        C_ref += alpha * np.eye(L)

        ref_eigenvalues = np.sort(np.linalg.eigvalsh(C_ref))[::-1]
        ref_eigenvalues = np.maximum(ref_eigenvalues, 0.0)

        # Top eigenvalues should match closely.
        np.testing.assert_array_almost_equal(
            stored_eigenvalues[:5], ref_eigenvalues[:5], decimal=4
        )


# ---------------------------------------------------------------------------
# TestRegularization
# ---------------------------------------------------------------------------


class TestRegularization:
    """Tikhonov regularization for rank-deficient cases."""

    def test_regularization_applied_when_k_less_than_l(self):
        """Per-cat model with K < L → regularized flag is True."""
        cat_id = _insert_cat("RegCat")

        # 10 waveforms, L=64 → K=11 (including the new one) < L=64.
        config = _make_config(per_cat_minimum=5)
        analyser = EigenAnalyser(config)

        vecs = _make_subspace_waveforms(10, L=64, n_basis=3)
        _insert_waveforms("weight_g", cat_id, vecs)

        td_visit_id = _insert_td_visit()
        waveform = vecs[0].copy()
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        result = analyser.analyse(waveform, record, "weight_g")

        assert result.details["regularized"] is True

    def test_all_eigenvalues_positive_after_regularization(self):
        """After regularization, all eigenvalues should be strictly positive."""
        cat_id = _insert_cat("RegCat2")

        config = _make_config(per_cat_minimum=5)
        analyser = EigenAnalyser(config)

        vecs = _make_subspace_waveforms(10, L=64, n_basis=3)
        _insert_waveforms("weight_g", cat_id, vecs)

        td_visit_id = _insert_td_visit()
        waveform = vecs[0].copy()
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        result = analyser.analyse(waveform, record, "weight_g")

        model_id = result.details["model_id"]
        with get_conn() as conn:
            row = conn.execute(
                "SELECT eigenvalues_json FROM eigen_models WHERE model_id = ?",
                (model_id,),
            ).fetchone()

        eigenvalues = np.array(json.loads(row["eigenvalues_json"]))
        assert (eigenvalues > 0).all()


# ---------------------------------------------------------------------------
# TestExplainedVariance
# ---------------------------------------------------------------------------


class TestExplainedVariance:
    """Explained variance scoring."""

    def test_in_distribution_high_ev(self):
        """A waveform from the same distribution as training data → high EV."""
        cat_id = _insert_cat("EVCat")

        config = _make_config(per_cat_minimum=5)
        analyser = EigenAnalyser(config)

        vecs = _make_subspace_waveforms(20, L=64, n_basis=3, noise_sigma=0.01)
        _insert_waveforms("weight_g", cat_id, vecs)

        # New waveform from the SAME distribution.
        new_vec = _make_subspace_waveforms(1, L=64, n_basis=3, noise_sigma=0.01, seed=999)[0]

        td_visit_id = _insert_td_visit()
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        result = analyser.analyse(new_vec, record, "weight_g")

        assert result.details["ev"] > 0.80

    def test_random_noise_low_ev(self):
        """Pure random noise waveform → low EV against a structured subspace."""
        cat_id = _insert_cat("NoiseCat")

        config = _make_config(per_cat_minimum=5)
        analyser = EigenAnalyser(config)

        vecs = _make_subspace_waveforms(20, L=64, n_basis=3, noise_sigma=0.01)
        _insert_waveforms("weight_g", cat_id, vecs)

        # Random noise — NOT in the subspace.
        rng = np.random.RandomState(777)
        noise = rng.randn(64)
        noise -= noise.mean()

        td_visit_id = _insert_td_visit()
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        result = analyser.analyse(noise, record, "weight_g")

        assert result.details["ev"] < 0.50


# ---------------------------------------------------------------------------
# TestCoefficientStorage
# ---------------------------------------------------------------------------


class TestCoefficientStorage:
    """Expansion coefficients are stored per visit."""

    def test_coefficients_populated(self):
        """After scoring, coefficients_json is not NULL in eigen_waveforms."""
        cat_id = _insert_cat("CoeffCat")

        config = _make_config(per_cat_minimum=5)
        analyser = EigenAnalyser(config)

        vecs = _make_subspace_waveforms(10, L=64, n_basis=3)
        _insert_waveforms("weight_g", cat_id, vecs)

        td_visit_id = _insert_td_visit()
        new_vec = vecs[0].copy()
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        result = analyser.analyse(new_vec, record, "weight_g")

        # The newest waveform should have coefficients_json populated.
        with get_conn() as conn:
            row = conn.execute(
                """SELECT coefficients_json, eigen_ev, eigen_residual, model_id
                   FROM eigen_waveforms
                   WHERE td_visit_id = ? AND channel = ?
                   ORDER BY waveform_id DESC LIMIT 1""",
                (td_visit_id, "weight_g"),
            ).fetchone()

        assert row["coefficients_json"] is not None
        coeffs = json.loads(row["coefficients_json"])
        assert len(coeffs) == 64  # full L coefficients
        assert row["eigen_ev"] is not None
        assert row["eigen_residual"] is not None
        assert row["model_id"] is not None

    def test_model_id_references_valid_model(self):
        """model_id FK points to a valid eigen_models row."""
        cat_id = _insert_cat("FKCat")

        config = _make_config(per_cat_minimum=5)
        analyser = EigenAnalyser(config)

        vecs = _make_subspace_waveforms(10, L=64, n_basis=3)
        _insert_waveforms("weight_g", cat_id, vecs)

        td_visit_id = _insert_td_visit()
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        result = analyser.analyse(vecs[0].copy(), record, "weight_g")

        model_id = result.details["model_id"]
        with get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM eigen_models WHERE model_id = ?",
                (model_id,),
            ).fetchone()

        assert row is not None
        assert row["channel"] == "weight_g"
        assert row["n_components"] > 0
        assert row["n_waveforms"] > 0


# ---------------------------------------------------------------------------
# TestAnomalyClassification
# ---------------------------------------------------------------------------


class TestAnomalyClassification:
    """Anomaly level classification from EV values."""

    def test_classify_normal(self):
        analyser = EigenAnalyser(_make_config())
        level, score = analyser._classify(0.95)
        assert level == "normal"
        assert score == 0.0

    def test_classify_mild(self):
        analyser = EigenAnalyser(_make_config())
        level, score = analyser._classify(0.80)
        assert level == "mild"
        assert 0.0 < score < 0.34

    def test_classify_significant(self):
        analyser = EigenAnalyser(_make_config())
        level, score = analyser._classify(0.55)
        assert level == "significant"
        assert 0.33 <= score < 0.67

    def test_classify_major(self):
        analyser = EigenAnalyser(_make_config())
        level, score = analyser._classify(0.30)
        assert level == "major"
        assert score == 1.0

    def test_boundary_at_normal(self):
        analyser = EigenAnalyser(_make_config())
        level, _ = analyser._classify(0.90)
        assert level == "normal"

    def test_boundary_at_mild(self):
        analyser = EigenAnalyser(_make_config())
        level, _ = analyser._classify(0.70)
        assert level == "mild"

    def test_boundary_at_significant(self):
        analyser = EigenAnalyser(_make_config())
        level, _ = analyser._classify(0.40)
        assert level == "significant"


# ---------------------------------------------------------------------------
# TestEigenModelStorage
# ---------------------------------------------------------------------------


class TestEigenModelStorage:
    """eigen_models table storage and round-trip."""

    def test_model_stored_after_analysis(self):
        """Analysis with sufficient waveforms creates an eigen_models row."""
        cat_id = _insert_cat("ModelCat")

        config = _make_config(per_cat_minimum=5)
        analyser = EigenAnalyser(config)

        vecs = _make_subspace_waveforms(10, L=64, n_basis=3)
        _insert_waveforms("weight_g", cat_id, vecs)

        td_visit_id = _insert_td_visit()
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        analyser.analyse(vecs[0].copy(), record, "weight_g")

        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM eigen_models WHERE cat_id = ? AND channel = ?",
                (cat_id, "weight_g"),
            ).fetchall()

        assert len(rows) >= 1
        row = rows[-1]
        eigenvalues = json.loads(row["eigenvalues_json"])
        eigenvectors = json.loads(row["eigenvectors_json"])
        assert len(eigenvalues) == 64
        assert len(eigenvectors) == 64
        assert len(eigenvectors[0]) == 64
        assert row["n_components"] > 0
        assert row["n_components"] <= 64

    def test_eigenvectors_round_trip(self):
        """Stored eigenvectors deserialize to an orthogonal matrix."""
        cat_id = _insert_cat("OrthoCheck")

        config = _make_config(per_cat_minimum=5)
        analyser = EigenAnalyser(config)

        vecs = _make_subspace_waveforms(20, L=64, n_basis=3)
        _insert_waveforms("weight_g", cat_id, vecs)

        td_visit_id = _insert_td_visit()
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        result = analyser.analyse(vecs[0].copy(), record, "weight_g")

        model_id = result.details["model_id"]
        with get_conn() as conn:
            row = conn.execute(
                "SELECT eigenvectors_json FROM eigen_models WHERE model_id = ?",
                (model_id,),
            ).fetchone()

        V = np.array(json.loads(row["eigenvectors_json"]))
        # V^T V should be approximately identity.
        product = V.T @ V
        np.testing.assert_array_almost_equal(
            product, np.eye(64), decimal=10
        )


# ---------------------------------------------------------------------------
# TestRoundTrip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """End-to-end: synthetic subspace waveforms vs. random noise."""

    def test_subspace_waveform_scores_high(self):
        """40 training waveforms from a 3-sinusoid subspace; new in-distribution → high EV."""
        cat_id = _insert_cat("RoundTripCat")

        config = _make_config(per_cat_minimum=5)
        analyser = EigenAnalyser(config)

        training = _make_subspace_waveforms(40, L=64, n_basis=3, noise_sigma=0.01, seed=42)
        _insert_waveforms("weight_g", cat_id, training)

        # New waveform from the same distribution but different seed.
        test_vec = _make_subspace_waveforms(1, L=64, n_basis=3, noise_sigma=0.01, seed=9999)[0]

        td_visit_id = _insert_td_visit()
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        result = analyser.analyse(test_vec, record, "weight_g")

        assert result.details["ev"] > 0.85
        assert result.anomaly_level in ("normal", "mild")

    def test_noise_waveform_scores_low(self):
        """40 training waveforms from a 3-sinusoid subspace; random noise → low EV."""
        cat_id = _insert_cat("NoiseTripCat")

        config = _make_config(per_cat_minimum=5)
        analyser = EigenAnalyser(config)

        training = _make_subspace_waveforms(40, L=64, n_basis=3, noise_sigma=0.01, seed=42)
        _insert_waveforms("weight_g", cat_id, training)

        # Random noise waveform.
        rng = np.random.RandomState(12345)
        noise = rng.randn(64)
        noise -= noise.mean()

        td_visit_id = _insert_td_visit()
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        result = analyser.analyse(noise, record, "weight_g")

        assert result.details["ev"] < 0.50
        assert result.anomaly_level in ("significant", "major")

    def test_full_coefficient_vector_length(self):
        """Expansion coefficients have length L=64."""
        cat_id = _insert_cat("CoeffLenCat")

        config = _make_config(per_cat_minimum=5)
        analyser = EigenAnalyser(config)

        training = _make_subspace_waveforms(20, L=64, n_basis=3, seed=42)
        _insert_waveforms("weight_g", cat_id, training)

        test_vec = training[0].copy()
        td_visit_id = _insert_td_visit()
        record = _make_record(cat_id=cat_id, td_visit_id=td_visit_id)
        result = analyser.analyse(test_vec, record, "weight_g")

        coefficients = result.details["coefficients"]
        assert len(coefficients) == 64
