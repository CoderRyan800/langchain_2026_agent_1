"""
Tests for the data-driven gas anomaly detector.

Covers internal helpers (log-Gaussian fit, z-score, tier mapping) and the
public ``score_gas_visit`` API end-to-end against an isolated SQLite DB.
No LLM calls.
"""

from __future__ import annotations

import math

import pytest

from litterbox.gas_anomaly import (
    score_gas_visit,
    ALARM_TIERS,
    _fit_log_gaussian,
    _z_score,
    _tier_for_z,
    _combine_tiers,
    _DEFAULT_THRESHOLDS,
)
from litterbox.db import get_conn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestFitLogGaussian:
    def test_too_few_samples_returns_none(self):
        assert _fit_log_gaussian([]) is None
        assert _fit_log_gaussian([10.0]) is None

    def test_zero_variance_returns_none(self):
        # All identical → MAD=0 → sigma=0 → degenerate.
        assert _fit_log_gaussian([5.0, 5.0, 5.0]) is None

    def test_returns_median_and_mad_sigma(self):
        # log1p(0)=0, log1p(e-1)=1, log1p(e^2-1)=2 → median=1.
        # |dev| = [1, 0, 1] → MAD = 1 → sigma = 1.4826.
        values = [0.0, math.e - 1, math.e ** 2 - 1]
        result = _fit_log_gaussian(values)
        assert result is not None
        location, sigma = result
        assert location == pytest.approx(1.0, abs=1e-9)
        assert sigma == pytest.approx(1.4826, abs=1e-4)

    def test_robust_to_outlier_contamination(self):
        # Five clean values around log1p ≈ 1.0 plus one wild outlier.
        # Mean would be pulled high; median ignores the outlier.
        clean = [math.exp(0.95) - 1, math.exp(1.0) - 1, math.exp(1.05) - 1,
                 math.exp(1.0) - 1, math.exp(1.0) - 1]
        outlier = math.exp(10.0) - 1   # log1p ≈ 10
        result = _fit_log_gaussian(clean + [outlier])
        assert result is not None
        location, sigma = result
        # Median of 6 values is mean of two middle ones — both ~1.0.
        assert location == pytest.approx(1.0, abs=0.05)
        # MAD must be small (clean data) — outlier doesn't expand it.
        # Without robustness, std would be ~3.4; here we expect sigma << 1.
        assert sigma < 0.5

    def test_signed_z_uses_log_median(self):
        # Six readings spanning log1p ≈ 0..5. Median in log-space is 2.5.
        values = [math.exp(0.0) - 1, math.exp(1.0) - 1, math.exp(2.0) - 1,
                  math.exp(3.0) - 1, math.exp(4.0) - 1, math.exp(5.0) - 1]
        result = _fit_log_gaussian(values)
        assert result is not None
        location, sigma = result
        assert location == pytest.approx(2.5, abs=1e-9)
        assert sigma > 0.0


class TestZScore:
    def test_none_reading_returns_none(self):
        assert _z_score(None, (1.0, 1.0)) is None

    def test_none_model_returns_none(self):
        assert _z_score(50.0, None) is None

    def test_correct_signed_value(self):
        # mu=1, sigma=2, reading log1p=5 → z = (5-1)/2 = 2.0
        # log1p(x) = 5 → x = e^5 - 1
        z = _z_score(math.e ** 5 - 1, (1.0, 2.0))
        assert z == pytest.approx(2.0, abs=1e-9)

    def test_negative_z_for_low_reading(self):
        # mu=3, sigma=1, reading log1p=1 → z = -2
        z = _z_score(math.e - 1, (3.0, 1.0))
        assert z == pytest.approx(-2.0, abs=1e-9)


class TestTierForZ:
    def test_none_is_insufficient(self):
        assert _tier_for_z(None, _DEFAULT_THRESHOLDS) == "insufficient_data"

    def test_nan_is_insufficient(self):
        assert _tier_for_z(float("nan"), _DEFAULT_THRESHOLDS) == "insufficient_data"

    def test_negative_z_is_normal(self):
        assert _tier_for_z(-3.0, _DEFAULT_THRESHOLDS) == "normal"

    def test_below_mild_is_normal(self):
        assert _tier_for_z(1.99, _DEFAULT_THRESHOLDS) == "normal"

    def test_at_mild_threshold_is_mild(self):
        assert _tier_for_z(2.0, _DEFAULT_THRESHOLDS) == "mild"

    def test_at_significant_threshold_is_significant(self):
        assert _tier_for_z(3.0, _DEFAULT_THRESHOLDS) == "significant"

    def test_at_severe_threshold_is_severe(self):
        assert _tier_for_z(5.0, _DEFAULT_THRESHOLDS) == "severe"

    def test_above_severe_is_still_severe(self):
        assert _tier_for_z(99.0, _DEFAULT_THRESHOLDS) == "severe"

    def test_thresholds_are_picked_highest_first(self):
        # z=10 must land at severe (the highest), not just mild.
        assert _tier_for_z(10.0, _DEFAULT_THRESHOLDS) == "severe"


class TestCombineTiers:
    def test_severe_dominates(self):
        assert _combine_tiers("severe", "normal") == "severe"
        assert _combine_tiers("normal", "severe") == "severe"

    def test_significant_beats_mild(self):
        assert _combine_tiers("significant", "mild") == "significant"

    def test_normal_beats_insufficient(self):
        # A confirmed-normal channel is more informative than missing data.
        assert _combine_tiers("normal", "insufficient_data") == "normal"

    def test_two_normals_yield_normal(self):
        assert _combine_tiers("normal", "normal") == "normal"


# ---------------------------------------------------------------------------
# score_gas_visit — DB integration
# ---------------------------------------------------------------------------

CONFIG = {
    "gas_anomaly": {
        "min_visits_per_cat": 5,
        "min_visits_pooled":  10,
        "z_score_thresholds": _DEFAULT_THRESHOLDS,
    }
}


def _seed_visits(conn, cat_id, ammonia_values, methane_values):
    """Insert one visit per (NH3, CH4) pair under the given cat_id."""
    for nh3, ch4 in zip(ammonia_values, methane_values):
        conn.execute(
            """INSERT INTO visits
               (entry_time, exit_time, tentative_cat_id, ammonia_peak_ppb, methane_peak_ppb)
               VALUES (?, ?, ?, ?, ?)""",
            ("2026-01-01T00:00:00", "2026-01-01T00:05:00", cat_id, nh3, ch4),
        )


@pytest.fixture()
def two_cats():
    """Insert two cats; return (cat_a_id, cat_b_id)."""
    with get_conn() as conn:
        a = conn.execute("INSERT INTO cats (name) VALUES ('A')").lastrowid
        b = conn.execute("INSERT INTO cats (name) VALUES ('B')").lastrowid
    return a, b


class TestScoreInsufficientData:
    def test_empty_db_returns_insufficient(self, registered_cat):
        cat_id, _ = registered_cat
        with get_conn() as conn:
            score = score_gas_visit(conn, cat_id, 50.0, 30.0, config=CONFIG)
        assert score["model_used"] == "insufficient_data"
        assert score["overall_tier"] == "insufficient_data"
        assert score["ammonia_z"] is None
        assert score["methane_z"] is None
        assert score["n_samples"] == 0

    def test_below_per_cat_below_pooled_returns_insufficient(self, registered_cat):
        cat_id, _ = registered_cat
        with get_conn() as conn:
            _seed_visits(conn, cat_id, [10.0, 12.0, 11.0], [5.0, 6.0, 5.5])
            score = score_gas_visit(conn, cat_id, 50.0, 30.0, config=CONFIG)
        assert score["model_used"] == "insufficient_data"


class TestScorePerCatModel:
    def test_per_cat_model_when_threshold_met(self, registered_cat):
        cat_id, _ = registered_cat
        with get_conn() as conn:
            _seed_visits(
                conn, cat_id,
                ammonia_values=[10.0, 12.0, 11.0, 13.0, 9.0, 10.5],
                methane_values=[5.0, 6.0, 5.5, 6.5, 4.5, 5.2],
            )
            score = score_gas_visit(conn, cat_id, 11.0, 5.5, config=CONFIG)
        assert score["model_used"] == "per_cat"
        assert score["n_samples"] == 6
        assert score["overall_tier"] == "normal"

    def test_high_ammonia_triggers_severe(self, registered_cat):
        cat_id, _ = registered_cat
        with get_conn() as conn:
            # Tight cluster around log1p≈log(11)≈2.4
            _seed_visits(
                conn, cat_id,
                ammonia_values=[10.0, 11.0, 12.0, 10.5, 11.5, 10.8],
                methane_values=[5.0, 6.0, 5.5, 6.5, 4.5, 5.2],
            )
            score = score_gas_visit(conn, cat_id, 1000.0, 5.5, config=CONFIG)
        assert score["model_used"] == "per_cat"
        assert score["ammonia_z"] is not None and score["ammonia_z"] > 5.0
        assert score["ammonia_tier"] == "severe"
        assert score["overall_tier"] == "severe"

    def test_low_reading_does_not_alarm(self, registered_cat):
        cat_id, _ = registered_cat
        with get_conn() as conn:
            _seed_visits(
                conn, cat_id,
                ammonia_values=[50.0, 55.0, 52.0, 48.0, 51.0, 53.0],
                methane_values=[20.0] * 6,
            )
            score = score_gas_visit(conn, cat_id, 5.0, 20.0, config=CONFIG)
        # Low reading → strongly negative z, but tier must stay normal.
        assert score["ammonia_z"] is not None and score["ammonia_z"] < 0
        assert score["ammonia_tier"] == "normal"


class TestScorePooledFallback:
    def test_pooled_used_when_per_cat_sparse_but_pool_full(self, two_cats, registered_cat):
        cat_id, _ = registered_cat
        a_id, b_id = two_cats
        with get_conn() as conn:
            # cat under test has only 2 readings — below per-cat threshold.
            _seed_visits(conn, cat_id, [11.0, 12.0], [5.0, 6.0])
            # Pool gets enough data from the other two cats.
            _seed_visits(conn, a_id, [10.0] * 6, [5.0] * 6)
            _seed_visits(conn, b_id, [13.0] * 6, [6.0] * 6)
            score = score_gas_visit(conn, cat_id, 12.0, 5.5, config=CONFIG)
        assert score["model_used"] == "pooled"
        assert score["n_samples"] >= 10

    def test_unidentified_visit_uses_pooled(self, two_cats):
        a_id, b_id = two_cats
        with get_conn() as conn:
            _seed_visits(conn, a_id, [10.0] * 6, [5.0] * 6)
            _seed_visits(conn, b_id, [13.0] * 6, [6.0] * 6)
            score = score_gas_visit(conn, None, 12.0, 5.5, config=CONFIG)
        assert score["model_used"] == "pooled"


class TestScoreNullSensors:
    def test_null_ammonia_returns_none_z_for_that_channel(self, registered_cat):
        cat_id, _ = registered_cat
        with get_conn() as conn:
            _seed_visits(
                conn, cat_id,
                ammonia_values=[10.0] * 6,
                methane_values=[5.0] * 6,
            )
            score = score_gas_visit(conn, cat_id, None, 5.0, config=CONFIG)
        assert score["ammonia_z"] is None
        assert score["ammonia_tier"] == "insufficient_data"

    def test_both_null_yields_insufficient_overall(self, registered_cat):
        cat_id, _ = registered_cat
        with get_conn() as conn:
            _seed_visits(
                conn, cat_id,
                ammonia_values=[10.0] * 6,
                methane_values=[5.0] * 6,
            )
            score = score_gas_visit(conn, cat_id, None, None, config=CONFIG)
        assert score["overall_tier"] == "insufficient_data"


class TestExcludeVisitId:
    def test_current_visit_excluded_from_fit(self, registered_cat):
        cat_id, _ = registered_cat
        with get_conn() as conn:
            # Five normal historical visits...
            _seed_visits(
                conn, cat_id,
                ammonia_values=[10.0, 11.0, 12.0, 10.5, 11.5],
                methane_values=[5.0, 5.5, 6.0, 5.2, 5.8],
            )
            # ...plus the current visit row, already inserted with a wild value.
            cur = conn.execute(
                """INSERT INTO visits
                   (entry_time, tentative_cat_id, ammonia_peak_ppb, methane_peak_ppb)
                   VALUES (?, ?, ?, ?)""",
                ("2026-02-01T00:00:00", cat_id, 9999.0, 9999.0),
            )
            current_id = cur.lastrowid

            # Without exclusion, the wild value would inflate sigma and
            # suppress its own z-score. With exclusion the model fits the
            # tight historical cluster and the wild value scores severe.
            score = score_gas_visit(
                conn, cat_id,
                ammonia_peak_ppb=9999.0, methane_peak_ppb=9999.0,
                exclude_visit_id=current_id, config=CONFIG,
            )
        assert score["model_used"] == "per_cat"
        assert score["ammonia_tier"] == "severe"
        assert score["methane_tier"] == "severe"


class TestAlarmTiers:
    def test_alarm_tiers_constant_matches_design(self):
        assert ALARM_TIERS == frozenset({"mild", "significant", "severe"})

    def test_normal_not_in_alarm_tiers(self):
        assert "normal" not in ALARM_TIERS
        assert "insufficient_data" not in ALARM_TIERS
