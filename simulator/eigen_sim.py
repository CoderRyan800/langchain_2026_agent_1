#!/usr/bin/env python3
"""
eigen_sim.py — Eigenanalysis end-to-end simulator
====================================================

Generates synthetic litter-box visit waveforms for four cats, feeds them
through the full VisitAnalyser → AnalyserPipeline → EigenAnalyser pipeline,
and produces:

- Per-cat HTML reports with waveform plots and data tables
- Ground truth JSON with per-visit metadata
- Markdown summary with detection statistics

Each cat has a characteristic parametric waveform shape.  ~5% of visits
inject a *different* cat's waveform shape to test anomaly detection via
low explained variance.

Usage::

    python simulator/eigen_sim.py                   # 200 visits, seed=42
    python simulator/eigen_sim.py --seed 123        # different seed
    python simulator/eigen_sim.py --visits 50       # fewer visits
    python simulator/eigen_sim.py --report-only     # regenerate reports only
    python simulator/eigen_sim.py --clean           # wipe eigen tables first
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# Ensure src/ is importable when running from repo root.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eigen_sim_config import (
    BOX_BASELINE_G,
    CATS,
    DC_JITTER_G,
    DEFAULT_SEED,
    GROUND_TRUTH_PATH,
    NOISE_FRACTION,
    SAMPLE_INTERVAL_S,
    SIM_OUTPUT_DIR,
    SUMMARY_PATH,
    SWAP_PROBABILITY,
    TOTAL_VISITS,
    VISIT_DURATION_MAX_S,
    VISIT_DURATION_MIN_S,
    WASTE_MAX_G,
    WASTE_MIN_G,
)

from litterbox.db import get_conn, init_db
from litterbox.time_buffer import load_td_config
from litterbox.visit_analyser import VisitAnalyser
from litterbox.analyser_pipeline import AnalyserPipeline
from litterbox.eigen_analyser import EigenAnalyser
from litterbox.eigen_query import generate_report


# ===========================================================================
# Baseline waveform generation
# ===========================================================================

def _parametric_waveform(t: np.ndarray, cat_cfg: dict) -> np.ndarray:
    """Evaluate a cat's characteristic waveform at normalised times t ∈ [0, 1].

    Returns weight contribution in grams (above box baseline).
    The shape is:
        [pre-entry flat] → [ramp up] → [plateau with character] → [ramp down] → [post-exit flat + waste]

    Parameters
    ----------
    t : np.ndarray
        Normalised time array, values in [0, 1].
    cat_cfg : dict
        Per-cat config from CATS dict.
    """
    weight = float(cat_cfg["true_weight_g"])
    ramp_frac = cat_cfg["ramp_fraction"]
    plateau_type = cat_cfg["plateau_type"]
    plateau_param = cat_cfg["plateau_param"]

    # Phase boundaries (normalised)
    t_pre_end = 0.10               # 10% pre-entry baseline
    t_ramp_up_end = t_pre_end + ramp_frac
    t_ramp_down_start = 0.85 - ramp_frac
    t_ramp_down_end = 0.85
    # 0.85 → 1.0 is post-exit baseline

    out = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti < t_pre_end:
            # Pre-entry: just box baseline (0 contribution)
            out[i] = 0.0

        elif ti < t_ramp_up_end:
            # Ramp up: smooth sigmoid-like transition
            frac = (ti - t_pre_end) / (t_ramp_up_end - t_pre_end)
            out[i] = weight * _smooth_step(frac)

        elif ti < t_ramp_down_start:
            # Plateau with character
            plateau_frac = (ti - t_ramp_up_end) / (t_ramp_down_start - t_ramp_up_end)
            out[i] = weight + _plateau_character(
                plateau_frac, weight, plateau_type, plateau_param
            )

        elif ti < t_ramp_down_end:
            # Ramp down
            frac = (ti - t_ramp_down_start) / (t_ramp_down_end - t_ramp_down_start)
            out[i] = weight * (1.0 - _smooth_step(frac))

        else:
            # Post-exit: small waste residual
            out[i] = 0.0

    return out


def _smooth_step(x: float) -> float:
    """Smooth step function (3x² - 2x³) for natural ramp transitions."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _plateau_character(
    frac: float, weight: float, ptype: str, param: float,
) -> float:
    """Return the plateau deviation from flat weight.

    frac : normalised position within the plateau [0, 1]
    weight : cat weight in grams
    ptype : "ripple", "sag", "bumps", "parabola"
    param : amplitude as fraction of weight
    """
    amp = weight * param

    if ptype == "ripple":
        # Small sinusoidal ripple — ~2 cycles
        return amp * np.sin(frac * 4 * np.pi)

    elif ptype == "sag":
        # Dips in the middle (inverted parabola)
        return -amp * 4 * frac * (1 - frac)

    elif ptype == "bumps":
        # Two Gaussian bumps at 1/3 and 2/3 through the plateau
        bump1 = amp * np.exp(-((frac - 0.33) ** 2) / 0.01)
        bump2 = amp * np.exp(-((frac - 0.67) ** 2) / 0.01)
        return bump1 + bump2

    elif ptype == "parabola":
        # Smooth upward curve — peaks at center
        return amp * 4 * frac * (1 - frac)

    return 0.0


# ===========================================================================
# Visit synthesis
# ===========================================================================

def _generate_visit(
    cat_name: str,
    source_cat_name: str,
    visit_index: int,
    rng: random.Random,
    base_time: datetime,
) -> dict:
    """Generate one synthetic visit.

    Returns a dict with:
        snapshot: list[dict]  — synthetic buffer snapshot
        entry_time, exit_time: datetime
        cat_name: str         — the cat the chip says it is
        source_baseline: str  — which cat's waveform shape was actually used
        is_swapped: bool
        dc_term_true: float   — the absolute weight used
        noise_sigma: float    — noise standard deviation applied
    """
    cat_cfg = CATS[cat_name]
    source_cfg = CATS[source_cat_name]

    # Random visit duration → raw sample count.
    duration_s = rng.uniform(VISIT_DURATION_MIN_S, VISIT_DURATION_MAX_S)
    n_samples = max(3, int(duration_s / SAMPLE_INTERVAL_S))

    # Normalised time array for this visit.
    t = np.linspace(0.0, 1.0, n_samples)

    # Generate the waveform from the SOURCE cat's shape.
    shape = _parametric_waveform(t, source_cfg)

    # DC term: box baseline + cat weight + jitter.
    dc_offset = BOX_BASELINE_G + cat_cfg["true_weight_g"] + rng.gauss(0, DC_JITTER_G)

    # Add waste weight to post-exit portion.
    waste_g = rng.uniform(WASTE_MIN_G, WASTE_MAX_G)
    # Find where shape has returned to near zero (post-exit).
    for i in range(len(shape)):
        if t[i] > 0.85:
            shape[i] += waste_g

    # Full waveform with DC.
    raw_waveform = shape + dc_offset

    # Add Gaussian noise.
    rms = float(np.sqrt(np.mean(shape ** 2))) if np.any(shape != 0) else 1.0
    noise_sigma = NOISE_FRACTION * rms
    noise = np.array([rng.gauss(0, noise_sigma) for _ in range(n_samples)])
    raw_waveform = raw_waveform + noise

    # Build the snapshot.
    # Timestamps: spread over the visit duration, offset by visit index.
    visit_start = base_time + timedelta(seconds=visit_index * 600)
    entry_offset = int(n_samples * 0.10)  # entry at ~10% into waveform
    exit_offset = int(n_samples * 0.85)   # exit at ~85%

    entry_time = visit_start + timedelta(seconds=entry_offset * SAMPLE_INTERVAL_S)
    exit_time = visit_start + timedelta(seconds=exit_offset * SAMPLE_INTERVAL_S)

    snapshot = []
    for i in range(n_samples):
        ts = visit_start + timedelta(seconds=i * SAMPLE_INTERVAL_S)
        values = {"weight_g": float(raw_waveform[i])}
        # Chip ID present during the visit window.
        if entry_time <= ts <= exit_time:
            values["chip_id"] = cat_name
        snapshot.append({"timestamp": ts, "values": values})

    return {
        "snapshot": snapshot,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "cat_name": cat_name,
        "source_baseline": source_cat_name,
        "is_swapped": cat_name != source_cat_name,
        "dc_term_true": dc_offset,
        "noise_sigma": noise_sigma,
        "n_samples_raw": n_samples,
    }


# ===========================================================================
# DB cleanup
# ===========================================================================

def _clean_eigen_tables() -> None:
    """Wipe eigen_waveforms, eigen_models, and td_visits tables."""
    init_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM eigen_waveforms")
        conn.execute("DELETE FROM eigen_models")
        conn.execute("DELETE FROM td_visits")
    print("Cleaned eigen_waveforms, eigen_models, td_visits tables.")


def _ensure_cats_registered() -> None:
    """Ensure all simulation cats exist in the cats table."""
    init_db()
    with get_conn() as conn:
        for name in CATS:
            existing = conn.execute(
                "SELECT cat_id FROM cats WHERE name = ?", (name,)
            ).fetchone()
            if not existing:
                conn.execute("INSERT INTO cats (name) VALUES (?)", (name,))
                print(f"  Registered cat: {name}")


# ===========================================================================
# Summary report
# ===========================================================================

def _write_summary(ground_truth: list[dict]) -> None:
    """Write a Markdown summary report."""
    lines = [
        "# Eigenanalysis Simulation Summary",
        "",
        f"**Seed:** {ground_truth[0].get('seed', '?') if ground_truth else '?'}",
        f"**Total visits:** {len(ground_truth)}",
        "",
    ]

    # Per-cat stats.
    cat_names = sorted(CATS.keys())
    lines.append("## Per-Cat Results")
    lines.append("")
    lines.append("| Cat | Visits | Scored | Mean EV | Min EV | Anomalies Detected |")
    lines.append("|-----|--------|--------|---------|--------|--------------------|")

    for cat in cat_names:
        cat_visits = [v for v in ground_truth if v["cat_name"] == cat]
        scored = [v for v in cat_visits if v.get("eigen_ev") is not None]
        evs = [v["eigen_ev"] for v in scored]
        mean_ev = f"{np.mean(evs):.4f}" if evs else "—"
        min_ev = f"{np.min(evs):.4f}" if evs else "—"
        anomalies = sum(1 for v in scored if v.get("anomaly_level") in ("significant", "major"))
        lines.append(f"| {cat} | {len(cat_visits)} | {len(scored)} | {mean_ev} | {min_ev} | {anomalies} |")

    # Swap detection.
    lines.append("")
    lines.append("## Swapped Visit Detection")
    lines.append("")
    swapped = [v for v in ground_truth if v.get("is_swapped")]
    swapped_scored = [v for v in swapped if v.get("eigen_ev") is not None]
    swapped_detected = [v for v in swapped_scored if v.get("anomaly_level") in ("mild", "significant", "major")]

    lines.append(f"- **Total swapped visits:** {len(swapped)}")
    lines.append(f"- **Scored (had enough data for model):** {len(swapped_scored)}")
    lines.append(f"- **Detected (EV flagged):** {len(swapped_detected)}")
    if swapped_scored:
        rate = len(swapped_detected) / len(swapped_scored) * 100
        lines.append(f"- **Detection rate:** {rate:.0f}%")

    lines.append("")
    lines.append("## Swapped Visit Details")
    lines.append("")
    lines.append("| Visit | Cat (chip) | Source Shape | EV | Level |")
    lines.append("|-------|-----------|-------------|-----|-------|")
    for v in swapped:
        ev_str = f"{v['eigen_ev']:.4f}" if v.get("eigen_ev") is not None else "unscored"
        level = v.get("anomaly_level", "unscored")
        lines.append(f"| {v['visit_index']} | {v['cat_name']} | {v['source_baseline']} | {ev_str} | {level} |")

    # Normal visit stats.
    lines.append("")
    lines.append("## Normal Visit Statistics")
    lines.append("")
    normal_scored = [v for v in ground_truth if not v.get("is_swapped") and v.get("eigen_ev") is not None]
    if normal_scored:
        normal_evs = [v["eigen_ev"] for v in normal_scored]
        lines.append(f"- **Scored normal visits:** {len(normal_scored)}")
        lines.append(f"- **Mean EV:** {np.mean(normal_evs):.4f}")
        lines.append(f"- **Std EV:** {np.std(normal_evs):.4f}")
        lines.append(f"- **Min EV:** {np.min(normal_evs):.4f}")
        false_positives = sum(1 for v in normal_scored if v.get("anomaly_level") in ("significant", "major"))
        lines.append(f"- **False positives (significant/major):** {false_positives}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to: {SUMMARY_PATH}")


# ===========================================================================
# Main simulation
# ===========================================================================

def run_simulation(
    seed: int = DEFAULT_SEED,
    total_visits: int = TOTAL_VISITS,
    clean: bool = False,
) -> list[dict]:
    """Run the full eigenanalysis simulation.

    Returns the ground truth list (also written to JSON).
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    print(f"Eigenanalysis Simulator — seed={seed}, visits={total_visits}")
    print()

    # --- Setup ---
    init_db()
    if clean:
        _clean_eigen_tables()
    _ensure_cats_registered()

    config = load_td_config()
    analyser = VisitAnalyser(config)
    eigen = EigenAnalyser(config)
    pipeline = AnalyserPipeline([eigen], config)

    # --- Build visit schedule ---
    cat_names = list(CATS.keys())
    schedule = []
    for cat_name in cat_names:
        n = CATS[cat_name]["visits"]
        # Scale to requested total.
        n_scaled = max(1, int(n * total_visits / TOTAL_VISITS))
        schedule.extend([cat_name] * n_scaled)
    rng.shuffle(schedule)
    schedule = schedule[:total_visits]

    base_time = datetime(2026, 4, 1, 8, 0, 0, tzinfo=timezone.utc)
    ground_truth = []

    print(f"Simulating {len(schedule)} visits...")
    print()

    for idx, cat_name in enumerate(schedule):
        # Decide if this visit is swapped.
        if rng.random() < SWAP_PROBABILITY:
            other_cats = [c for c in cat_names if c != cat_name]
            source_cat = rng.choice(other_cats)
        else:
            source_cat = cat_name

        # Generate the visit.
        visit = _generate_visit(cat_name, source_cat, idx, rng, base_time)

        # Run through the pipeline.
        record = analyser.analyse(
            visit["snapshot"], visit["entry_time"], visit["exit_time"],
        )
        analyser.save(record)

        results = pipeline.run(
            record, visit["snapshot"],
            visit["entry_time"], visit["exit_time"],
        )

        # Extract eigenanalysis result.
        eigen_result = results[0] if results else None
        ev = eigen_result.details.get("ev") if eigen_result else None
        anomaly_level = eigen_result.anomaly_level if eigen_result else "unscored"
        n_comp = eigen_result.details.get("n_components") if eigen_result else None

        # Progress indicator.
        swap_marker = " [SWAPPED]" if visit["is_swapped"] else ""
        ev_str = f"EV={ev:.4f}" if ev is not None else "accumulating"
        status = f"  [{idx + 1:3d}/{len(schedule)}] {cat_name:8s} → {ev_str:20s} {anomaly_level:18s}{swap_marker}"
        print(status)

        gt_entry = {
            "seed": seed,
            "visit_index": idx,
            "cat_name": cat_name,
            "source_baseline": source_cat,
            "is_swapped": visit["is_swapped"],
            "dc_term_true": round(visit["dc_term_true"], 1),
            "noise_sigma": round(visit["noise_sigma"], 4),
            "n_samples_raw": visit["n_samples_raw"],
            "td_visit_id": record.td_visit_id,
            "eigen_ev": round(ev, 6) if ev is not None else None,
            "anomaly_level": anomaly_level,
            "n_components": n_comp,
        }
        ground_truth.append(gt_entry)

    # --- Write ground truth ---
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_PATH.write_text(
        json.dumps(ground_truth, indent=2), encoding="utf-8"
    )
    print(f"\nGround truth written to: {GROUND_TRUTH_PATH}")

    # --- Generate reports ---
    _generate_all_reports(cat_names, ground_truth)

    return ground_truth


def _generate_all_reports(cat_names: list[str], ground_truth: list[dict]) -> None:
    """Generate per-cat HTML reports and the summary."""
    SIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating reports...")
    for cat_name in cat_names:
        output_path = SIM_OUTPUT_DIR / f"eigen_{cat_name.lower()}.html"
        generate_report(cat_name, output_path=output_path)
        print(f"  {cat_name}: {output_path}")

    _write_summary(ground_truth)


def regenerate_reports() -> None:
    """Regenerate reports from existing ground truth JSON."""
    if not GROUND_TRUTH_PATH.exists():
        print(f"No ground truth found at {GROUND_TRUTH_PATH}. Run the simulation first.")
        return

    gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    cat_names = sorted(CATS.keys())
    _generate_all_reports(cat_names, gt)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Eigenanalysis end-to-end simulator"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--visits", type=int, default=TOTAL_VISITS,
                        help=f"Number of visits to simulate (default: {TOTAL_VISITS})")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate reports from existing DB/ground truth")
    parser.add_argument("--clean", action="store_true",
                        help="Wipe eigen tables before running")

    args = parser.parse_args()

    if args.report_only:
        regenerate_reports()
    else:
        run_simulation(seed=args.seed, total_visits=args.visits, clean=args.clean)


if __name__ == "__main__":
    main()
