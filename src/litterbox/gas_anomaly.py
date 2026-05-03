"""
gas_anomaly.py — Per-cat data-driven gas anomaly detector
==========================================================

Scores each visit's NH3 and CH4 peak readings as signed z-scores against the
distribution of that cat's prior non-null readings. Falls back to a pooled
distribution across all cats when the per-cat sample is too small. Returns
``"insufficient_data"`` and contributes no signal when even the pooled sample
is too small.

Design rationale
----------------
- Absolute gas readings (ppb) are deployment-dependent: sensor placement,
  ventilation, ambient conditions, and chip-to-chip calibration drift mean a
  fixed numeric threshold has no portable meaning. The detector judges each
  visit relative to the cat's own history, not against a hardcoded number.
- Gas concentrations are right-skewed and bounded below by zero, so values
  are ``log1p``-transformed before fitting a univariate Gaussian. ``log1p``
  is well-defined at zero, unlike plain ``log``.
- Only the high-side tail is alarming: a visit with unusually low ammonia is
  not a health concern. We compute signed z-scores and tier on ``z >= t``.
- The detector mirrors the cluster-analyser's contract: it fits on demand
  from the visits table, no separate persisted model, no separate fit step.
  At the dataset sizes we care about (tens to hundreds of visits per cat)
  the fit is microseconds.
- The current visit is excluded from its own fit. Otherwise the new datum
  inflates variance and makes itself look less anomalous — a self-licking
  ice cream cone we want to avoid.
"""

from __future__ import annotations

import math
import sqlite3
from typing import Optional

from litterbox.time_buffer import load_td_config


# ---------------------------------------------------------------------------
# Tier classification — only the positive z-tail triggers an alarm.
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS = {"mild": 2.0, "significant": 3.0, "severe": 5.0}
_TIERS_BY_SEVERITY = ("severe", "significant", "mild")  # high → low


def _tier_for_z(z: Optional[float], thresholds: dict) -> str:
    """Map a signed z-score to a tier label.

    Negative z (low reading) → ``"normal"``. Positive z is bucketed into the
    highest tier whose threshold it meets or exceeds.
    """
    if z is None or not math.isfinite(z):
        return "insufficient_data"
    for tier in _TIERS_BY_SEVERITY:
        if z >= thresholds[tier]:
            return tier
    return "normal"


def _combine_tiers(tier_a: str, tier_b: str) -> str:
    """Per-visit tier is the most severe across both gas channels."""
    severity = {"insufficient_data": -1, "normal": 0, "mild": 1, "significant": 2, "severe": 3}
    return tier_a if severity.get(tier_a, -1) >= severity.get(tier_b, -1) else tier_b


# ---------------------------------------------------------------------------
# Distribution fit — log1p, mean+std, return None when sample is too small.
# ---------------------------------------------------------------------------

_DEGENERATE_SIGMA = 1e-9   # log-space scale below this is treated as "constant data"
_MAD_TO_SIGMA = 1.4826     # consistency factor: 1.4826 × MAD ≈ σ for Gaussian data


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _fit_log_gaussian(values: list[float]) -> Optional[tuple[float, float]]:
    """Fit a *robust* Gaussian-like model on ``log1p(values)``.

    Returns ``(median, sigma_from_MAD)`` or ``None``.

    Why robust statistics: at the dataset sizes we work with, the historical
    pool can easily contain 15-30% prior anomalies (a chronically-monitored
    cat may rack up several flagged visits). Mean + std would let those
    outliers pull both location and scale toward themselves, suppressing
    future z-scores for new anomalies of the same kind. Median has a 50%
    breakdown point, and MAD-based sigma (1.4826 × MAD) is the standard
    Gaussian-consistent robust scale estimator.

    Returns ``None`` when there are fewer than 2 samples or when the scale
    collapses to (effectively) zero — typically a sensor stuck at one value,
    where any deviation would produce inflated z-scores with no real meaning.
    """
    if len(values) < 2:
        return None
    log_vals = [math.log1p(v) for v in values]
    location = _median(log_vals)
    abs_dev = [abs(x - location) for x in log_vals]
    mad = _median(abs_dev)
    sigma = _MAD_TO_SIGMA * mad
    if sigma < _DEGENERATE_SIGMA:
        return None
    return location, sigma


def _z_score(reading: Optional[float], model: Optional[tuple[float, float]]) -> Optional[float]:
    """Signed z-score of a single reading under a log-Gaussian model."""
    if reading is None or model is None:
        return None
    mu, sigma = model
    return (math.log1p(reading) - mu) / sigma


# ---------------------------------------------------------------------------
# Data fetch — historical readings for one channel.
# ---------------------------------------------------------------------------

def _fetch_history(
    conn: sqlite3.Connection,
    column: str,
    cat_id: Optional[int] = None,
    exclude_visit_id: Optional[int] = None,
) -> list[float]:
    """Fetch all non-null prior readings for a single gas channel.

    When ``cat_id`` is given, restrict to that cat's identified visits
    (tentative or confirmed). When ``exclude_visit_id`` is given, drop that
    visit so a row in mid-update doesn't bias its own fit.

    Column whitelist guards against accidental injection — ``column`` is
    interpolated into the SQL because parameter binding doesn't work for
    column names. Only the two known gas columns are accepted.
    """
    if column not in ("ammonia_peak_ppb", "methane_peak_ppb"):
        raise ValueError(f"unknown gas column: {column!r}")

    where_clauses = [f"{column} IS NOT NULL"]
    params: list = []
    if cat_id is not None:
        where_clauses.append(
            "(confirmed_cat_id = ? OR (confirmed_cat_id IS NULL AND tentative_cat_id = ?))"
        )
        params.extend([cat_id, cat_id])
    if exclude_visit_id is not None:
        where_clauses.append("visit_id != ?")
        params.append(exclude_visit_id)

    sql = f"SELECT {column} FROM visits WHERE " + " AND ".join(where_clauses)
    return [float(row[0]) for row in conn.execute(sql, params)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_gas_visit(
    conn: sqlite3.Connection,
    cat_id: Optional[int],
    ammonia_peak_ppb: Optional[float],
    methane_peak_ppb: Optional[float],
    *,
    exclude_visit_id: Optional[int] = None,
    config: Optional[dict] = None,
) -> dict:
    """Score one visit's gas readings against the cat's history.

    Parameters
    ----------
    conn:
        Open SQLite connection. Caller owns the lifecycle.
    cat_id:
        The cat being scored, or ``None`` if the visit is unidentified. When
        ``None``, the per-cat model is skipped and only the pooled fallback
        is attempted.
    ammonia_peak_ppb, methane_peak_ppb:
        The current visit's peak readings. ``None`` for missing sensor.
    exclude_visit_id:
        Visit row to omit from the historical fit (typically the current
        visit, whose row has already been UPDATEd by the time we score).
    config:
        Parsed ``td_config.json`` dict. If ``None`` the file is loaded from
        its default location.

    Returns
    -------
    dict
        ``ammonia_z``, ``methane_z`` — signed z-scores or ``None``.
        ``ammonia_tier``, ``methane_tier`` — per-channel tier label.
        ``overall_tier`` — most severe of the two.
        ``model_used`` — ``"per_cat"``, ``"pooled"``, or ``"insufficient_data"``.
        ``n_samples`` — number of historical readings the model was fit on
        (max of ammonia and methane sample counts).
    """
    cfg = config if config is not None else load_td_config()
    ga_cfg = cfg.get("gas_anomaly", {})
    min_per_cat = int(ga_cfg.get("min_visits_per_cat", 10))
    min_pooled  = int(ga_cfg.get("min_visits_pooled", 30))
    thresholds  = {**_DEFAULT_THRESHOLDS, **ga_cfg.get("z_score_thresholds", {})}

    # Try per-cat fit first.
    nh3_model: Optional[tuple[float, float]] = None
    ch4_model: Optional[tuple[float, float]] = None
    n_samples = 0
    model_used = "insufficient_data"

    if cat_id is not None:
        nh3_hist = _fetch_history(conn, "ammonia_peak_ppb",
                                  cat_id=cat_id, exclude_visit_id=exclude_visit_id)
        ch4_hist = _fetch_history(conn, "methane_peak_ppb",
                                  cat_id=cat_id, exclude_visit_id=exclude_visit_id)
        if len(nh3_hist) >= min_per_cat or len(ch4_hist) >= min_per_cat:
            nh3_model = _fit_log_gaussian(nh3_hist) if len(nh3_hist) >= min_per_cat else None
            ch4_model = _fit_log_gaussian(ch4_hist) if len(ch4_hist) >= min_per_cat else None
            if nh3_model is not None or ch4_model is not None:
                model_used = "per_cat"
                n_samples = max(len(nh3_hist), len(ch4_hist))

    # Pooled fallback for any channel that didn't get a per-cat model.
    if nh3_model is None or ch4_model is None:
        nh3_pool = _fetch_history(conn, "ammonia_peak_ppb",
                                  exclude_visit_id=exclude_visit_id)
        ch4_pool = _fetch_history(conn, "methane_peak_ppb",
                                  exclude_visit_id=exclude_visit_id)
        if nh3_model is None and len(nh3_pool) >= min_pooled:
            nh3_model = _fit_log_gaussian(nh3_pool)
            if nh3_model is not None:
                if model_used == "insufficient_data":
                    model_used = "pooled"
                n_samples = max(n_samples, len(nh3_pool))
        if ch4_model is None and len(ch4_pool) >= min_pooled:
            ch4_model = _fit_log_gaussian(ch4_pool)
            if ch4_model is not None:
                if model_used == "insufficient_data":
                    model_used = "pooled"
                n_samples = max(n_samples, len(ch4_pool))

    nh3_z = _z_score(ammonia_peak_ppb, nh3_model)
    ch4_z = _z_score(methane_peak_ppb, ch4_model)

    # If both readings are present but both models are missing, the visit is
    # un-scoreable. If a reading is None its tier is naturally insufficient.
    nh3_tier = _tier_for_z(nh3_z, thresholds) if ammonia_peak_ppb is not None else "insufficient_data"
    ch4_tier = _tier_for_z(ch4_z, thresholds) if methane_peak_ppb is not None else "insufficient_data"
    overall_tier = _combine_tiers(nh3_tier, ch4_tier)

    return {
        "ammonia_z":     nh3_z,
        "methane_z":     ch4_z,
        "ammonia_tier":  nh3_tier,
        "methane_tier":  ch4_tier,
        "overall_tier":  overall_tier,
        "model_used":    model_used,
        "n_samples":     n_samples,
    }


# Tiers that should trigger is_anomalous=True regardless of LLM verdict.
ALARM_TIERS = frozenset({"mild", "significant", "severe"})
