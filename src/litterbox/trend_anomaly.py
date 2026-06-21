"""
trend_anomaly.py — Long-term trend detector for body weight, waste output,
and gas readings.

The per-visit detectors (gas_anomaly.py, eigen_analyser.py,
cluster_analyser.py) each ask "is this single visit unusual against the
cat's history?" They are deliberately blind to slow drifts: gas_anomaly
fits over the entire history, so a multi-month upward creep gets absorbed
into the new baseline. Body weight has no per-visit detector at all.

This module fills that gap. For each cat, it splits the recent history
into two windows — `recent` (last 14 days by default) and `baseline` (the
75 days before that) — and asks "has the cat's typical reading shifted?"
The signal is a mean-shift z-score with MAD-based σ (same robust
estimator the gas detector uses), and tiers are the same shape:
mild ≥ 2σ, significant ≥ 3σ, severe ≥ 5σ.

Three direction-aware overlays match clinical reality:

* **Weight** — alarm in either direction, but additionally apply
  veterinary % thresholds (5/10/15% body weight) and take the worse of
  the two tiers. Either direction matters.
* **NH₃ / CH₄** — alarm only on the high side (z > 0). Low gas is good
  news.
* **Waste** — alarm only on the low side AND only when a constipation
  pattern is also present: a high fraction of "no-waste" visits in the
  recent window relative to baseline. Single low-waste visits are
  meaningless because cats often pee-only.

Configuration lives in ``td_config.json`` under ``trend_anomaly``.

Returns plain dicts; the caller decides how to render or persist.
Mirror of ``gas_anomaly.score_gas_visit`` in spirit — pure function on
top of an open sqlite connection, no module-level state.
"""

from __future__ import annotations

import math
import sqlite3
import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional

from litterbox.time_buffer import load_td_config


# ===========================================================================
# Constants
# ===========================================================================

# Same MAD → Gaussian-σ scale factor used by gas_anomaly.py — keeps tier
# semantics consistent across the system.
_MAD_TO_SIGMA = 1.4826
_DEGENERATE_SIGMA = 1e-6

# Channels and the visit columns they live in. Order matters for report layout.
_CHANNELS = (
    ("cat_weight_g",     "both"),
    ("waste_weight_g",   "low"),    # alarm only on low side (constipation)
    ("ammonia_peak_ppb", "high"),   # alarm only on high side
    ("methane_peak_ppb", "high"),
)

# Tier ranking for "worse-of-two" combination.
_TIER_RANK = {
    "insufficient_data": 0,
    "normal": 0,
    "mild": 1,
    "significant": 2,
    "severe": 3,
}
_RANK_TO_TIER = {0: "normal", 1: "mild", 2: "significant", 3: "severe"}


def _worse_tier(a: str, b: str) -> str:
    return _RANK_TO_TIER[max(_TIER_RANK[a], _TIER_RANK[b])]


# ===========================================================================
# Robust statistics
# ===========================================================================

def _median(xs: list[float]) -> float:
    """Plain median."""
    return statistics.median(xs)


def _mad_sigma(xs: list[float]) -> Optional[float]:
    """Robust σ estimate via MAD × 1.4826. None if degenerate."""
    if len(xs) < 2:
        return None
    m = _median(xs)
    mad = _median([abs(x - m) for x in xs])
    sigma = _MAD_TO_SIGMA * mad
    if sigma < _DEGENERATE_SIGMA:
        return None
    return sigma


def _window_stats(xs: list[float]) -> Optional[dict]:
    """Mean, robust σ, min, max, count. Returns None if fewer than 2 samples."""
    if len(xs) < 2:
        return None
    return {
        "mean":  float(statistics.mean(xs)),
        "std":   _mad_sigma(xs),
        "min":   float(min(xs)),
        "max":   float(max(xs)),
        "n":     len(xs),
    }


def _mean_shift_z(
    recent: dict, baseline: dict,
) -> Optional[float]:
    """Welch-style standardised mean shift, using the baseline's robust σ as
    the variance estimate (recent window may be too small to estimate its
    own σ reliably)."""
    if recent is None or baseline is None:
        return None
    sigma_b = baseline["std"]
    if sigma_b is None or sigma_b < _DEGENERATE_SIGMA:
        return None
    pooled_se = math.sqrt(sigma_b**2 / recent["n"] + sigma_b**2 / baseline["n"])
    if pooled_se < _DEGENERATE_SIGMA:
        return None
    return (recent["mean"] - baseline["mean"]) / pooled_se


def _tier_from_z(z: Optional[float], thresholds: dict, direction: str) -> str:
    """Map a z-score to a tier. ``direction`` ∈ {"both","high","low"} controls
    which tail counts as anomalous.

    high → only positive z alarms. low → only negative z alarms.
    both → |z| is compared against each threshold."""
    if z is None or not math.isfinite(z):
        return "normal"
    if direction == "high":
        signal = z
    elif direction == "low":
        signal = -z
    else:
        signal = abs(z)
    if signal >= float(thresholds["severe"]):
        return "severe"
    if signal >= float(thresholds["significant"]):
        return "significant"
    if signal >= float(thresholds["mild"]):
        return "mild"
    return "normal"


def _tier_from_pct(
    pct_change: Optional[float], pct_thresholds: dict,
) -> str:
    """Tier from |pct_change|. Used for the body-weight % overlay."""
    if pct_change is None or not math.isfinite(pct_change):
        return "normal"
    p = abs(pct_change)
    if p >= float(pct_thresholds["severe"]):
        return "severe"
    if p >= float(pct_thresholds["significant"]):
        return "significant"
    if p >= float(pct_thresholds["mild"]):
        return "mild"
    return "normal"


# ===========================================================================
# Data fetch
# ===========================================================================

# Whitelist guards against accidental SQL injection — column comes from the
# _CHANNELS tuple but be defensive.
_VALID_COLUMNS = {c for c, _ in _CHANNELS}


def _fetch_visits_in_window(
    conn: sqlite3.Connection,
    cat_id: int,
    column: str,
    window_start_iso: str,
    window_end_iso: str,
) -> list[float]:
    """Pull non-null `column` values for the cat between two ISO timestamps."""
    if column not in _VALID_COLUMNS:
        raise ValueError(f"unknown trend column: {column!r}")
    sql = (
        f"SELECT {column} FROM visits "
        f"WHERE (confirmed_cat_id = ? "
        f"       OR (confirmed_cat_id IS NULL AND tentative_cat_id = ?)) "
        f"  AND {column} IS NOT NULL "
        f"  AND entry_time >= ? "
        f"  AND entry_time <  ?"
    )
    rows = conn.execute(sql, (cat_id, cat_id, window_start_iso, window_end_iso))
    return [float(r[0]) for r in rows]


def _fetch_visit_count_in_window(
    conn: sqlite3.Connection,
    cat_id: int,
    window_start_iso: str,
    window_end_iso: str,
) -> int:
    """Total visits for the cat in the window (regardless of which sensors fired)."""
    sql = (
        "SELECT COUNT(*) FROM visits "
        "WHERE (confirmed_cat_id = ? "
        "       OR (confirmed_cat_id IS NULL AND tentative_cat_id = ?)) "
        "  AND entry_time >= ? "
        "  AND entry_time <  ?"
    )
    return int(conn.execute(sql, (cat_id, cat_id, window_start_iso, window_end_iso)).fetchone()[0])


# ===========================================================================
# Constipation rule
# ===========================================================================

def _constipation_check(
    recent_waste: list[float],
    baseline_waste: list[float],
    waste_z: Optional[float],
    cfg: dict,
) -> dict:
    """Two-part constipation rule: frequency of no-waste visits AND mean
    waste z-score below threshold. Returns a dict with the verdict and the
    component metrics so the report can show why.

    All three conditions must fire for ``flagged = True``:
      1. recent no-waste rate ≥ min_no_waste_rate
      2. recent rate ≥ min_no_waste_ratio × baseline rate
      3. mean waste z ≤ min_waste_z_score (recent significantly lower)
    """
    cutoff   = float(cfg["no_waste_g_cutoff"])
    min_rate = float(cfg["min_no_waste_rate"])
    min_ratio = float(cfg["min_no_waste_ratio"])
    min_z    = float(cfg["min_waste_z_score"])

    n_recent   = len(recent_waste)
    n_baseline = len(baseline_waste)
    if n_recent == 0 or n_baseline == 0:
        return {
            "flagged": False,
            "reason":  "insufficient_data",
            "recent_no_waste_rate":   None,
            "baseline_no_waste_rate": None,
            "ratio":                  None,
            "z":                      waste_z,
        }
    recent_rate   = sum(1 for w in recent_waste   if w < cutoff) / n_recent
    baseline_rate = sum(1 for w in baseline_waste if w < cutoff) / n_baseline
    # Avoid div-by-zero when baseline is all-waste; treat as if baseline rate
    # were a tiny epsilon so any nonzero recent rate looks like infinity-x.
    safe_baseline_rate = max(baseline_rate, 1.0 / max(n_baseline, 1))
    ratio = recent_rate / safe_baseline_rate

    cond_rate  = recent_rate  >= min_rate
    cond_ratio = ratio        >= min_ratio
    cond_z     = (waste_z is not None) and (waste_z <= min_z)

    flagged = cond_rate and cond_ratio and cond_z

    if not flagged:
        missed = []
        if not cond_rate:  missed.append("no_waste_rate_too_low")
        if not cond_ratio: missed.append("ratio_to_baseline_too_low")
        if not cond_z:     missed.append("waste_z_above_threshold")
        reason = "ok" if not missed else ",".join(missed)
    else:
        reason = "constipation_pattern_detected"

    return {
        "flagged":                flagged,
        "reason":                 reason,
        "recent_no_waste_rate":   recent_rate,
        "baseline_no_waste_rate": baseline_rate,
        "ratio":                  ratio,
        "z":                      waste_z,
    }


# ===========================================================================
# Public API
# ===========================================================================

def score_trends(
    conn: sqlite3.Connection,
    cat_id: int,
    *,
    days_recent: Optional[int] = None,
    days_baseline: Optional[int] = None,
    now: Optional[datetime] = None,
    config: Optional[dict] = None,
) -> dict:
    """Score one cat's long-term trends across all four metrics.

    Parameters
    ----------
    conn:
        Open sqlite connection. Caller owns the lifecycle.
    cat_id:
        Cat to score.
    days_recent, days_baseline:
        Override the windows from the config. ``recent`` is the most
        recent N days; ``baseline`` is the prior M days right before
        ``recent``. The full lookback is therefore (recent + baseline).
    now:
        Reference "now" for the windows, mostly for tests. Defaults to
        ``datetime.now(timezone.utc)``.
    config:
        Parsed td_config dict. Loads from default location if None.

    Returns
    -------
    dict with the per-channel verdicts plus an overall worst tier.
    """
    cfg_full = config if config is not None else load_td_config()
    cfg = cfg_full.get("trend_anomaly", {})
    if days_recent is None:
        days_recent = int(cfg.get("days_recent", 14))
    if days_baseline is None:
        days_baseline = int(cfg.get("days_baseline", 75))
    z_thresh   = cfg.get("z_score_thresholds", {"mild": 2.0, "significant": 3.0, "severe": 5.0})
    pct_thresh = cfg.get("weight_pct_thresholds", {"mild": 0.05, "significant": 0.10, "severe": 0.15})
    constip_cfg = cfg.get("constipation", {
        "no_waste_g_cutoff":  5.0,
        "min_no_waste_rate":  0.5,
        "min_no_waste_ratio": 2.0,
        "min_waste_z_score": -2.0,
    })
    min_visits_recent   = int(cfg.get("min_visits_recent",   5))
    min_visits_baseline = int(cfg.get("min_visits_baseline", 10))

    if now is None:
        now = datetime.now(timezone.utc)
    recent_start   = now - timedelta(days=days_recent)
    baseline_start = recent_start - timedelta(days=days_baseline)

    recent_iso_start   = recent_start.isoformat()
    recent_iso_end     = now.isoformat()
    baseline_iso_start = baseline_start.isoformat()
    baseline_iso_end   = recent_iso_start

    visits_recent   = _fetch_visit_count_in_window(conn, cat_id, recent_iso_start, recent_iso_end)
    visits_baseline = _fetch_visit_count_in_window(conn, cat_id, baseline_iso_start, baseline_iso_end)

    overall_tier = "normal"
    channels_out: dict[str, dict] = {}
    overall_insufficient = (
        visits_recent < min_visits_recent or visits_baseline < min_visits_baseline
    )

    for column, direction in _CHANNELS:
        recent_vals   = _fetch_visits_in_window(conn, cat_id, column, recent_iso_start, recent_iso_end)
        baseline_vals = _fetch_visits_in_window(conn, cat_id, column, baseline_iso_start, baseline_iso_end)
        # Per-channel sample gates: a cat may meet the overall visit count
        # while a specific sensor (NH₃, CH₄, weight, waste) is sparsely
        # populated due to malfunction or being optional. The same
        # min_visits_* gates apply per channel — without this, a channel
        # with only 2 non-null readings could trip mild/significant/severe
        # off a statistically meaningless sample.
        channel_insufficient = (
            len(recent_vals)   < min_visits_recent or
            len(baseline_vals) < min_visits_baseline
        )
        recent_stats   = _window_stats(recent_vals)
        baseline_stats = _window_stats(baseline_vals)
        z = _mean_shift_z(recent_stats, baseline_stats)

        if (overall_insufficient or channel_insufficient
                or recent_stats is None or baseline_stats is None
                or z is None):
            channels_out[column] = {
                "recent":   recent_stats,
                "baseline": baseline_stats,
                "z_score":  z,
                "tier":     "insufficient_data",
                "direction": direction,
            }
            continue

        tier = _tier_from_z(z, z_thresh, direction)

        # Direction-aware overlays.
        if column == "cat_weight_g":
            pct_change = (recent_stats["mean"] - baseline_stats["mean"]) / baseline_stats["mean"]
            pct_tier = _tier_from_pct(pct_change, pct_thresh)
            tier_combined = _worse_tier(tier, pct_tier)
            channels_out[column] = {
                "recent":         recent_stats,
                "baseline":       baseline_stats,
                "z_score":        z,
                "pct_change":     pct_change,
                "tier_z":         tier,
                "tier_pct":       pct_tier,
                "tier":           tier_combined,
                "direction":      direction,
            }
            tier = tier_combined

        elif column == "waste_weight_g":
            constip = _constipation_check(recent_vals, baseline_vals, z, constip_cfg)
            channels_out[column] = {
                "recent":         recent_stats,
                "baseline":       baseline_stats,
                "z_score":        z,
                "tier":           tier,
                "direction":      direction,
                "constipation":   constip,
            }
            # Constipation is a separate alarm overlay. If flagged, escalate
            # the channel tier to at least "significant" — pattern is more
            # important than the z-score alone for this channel.
            if constip["flagged"] and _TIER_RANK[tier] < _TIER_RANK["significant"]:
                channels_out[column]["tier"] = "significant"
                tier = "significant"
        else:
            channels_out[column] = {
                "recent":   recent_stats,
                "baseline": baseline_stats,
                "z_score":  z,
                "tier":     tier,
                "direction": direction,
            }

        if _TIER_RANK[tier] > _TIER_RANK[overall_tier]:
            overall_tier = tier

    if overall_insufficient:
        overall_tier = "insufficient_data"

    return {
        "cat_id":             cat_id,
        "now":                now.isoformat(),
        "recent_window":      {"start": recent_iso_start, "end": recent_iso_end,
                                "n_visits": visits_recent},
        "baseline_window":    {"start": baseline_iso_start, "end": baseline_iso_end,
                                "n_visits": visits_baseline},
        "channels":           channels_out,
        "overall_tier":       overall_tier,
        "min_visits_recent":   min_visits_recent,
        "min_visits_baseline": min_visits_baseline,
    }


# Tiers that should escalate is_anomalous / draw user attention.
ALARM_TIERS = frozenset({"mild", "significant", "severe"})
