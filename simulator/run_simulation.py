#!/usr/bin/env python
"""Main simulation entry point.

Usage:
    python simulator/run_simulation.py [--seed N] [--no-register] [--report-only]

Options:
    --seed N        Override the random seed (default: 42).
    --no-register   Skip cat registration (useful when re-running against an
                    existing database that already has the cats registered).
    --report-only   Skip the simulation; regenerate the Markdown report from
                    the existing sim_ground_truth.json.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup — must happen before any local imports
# ---------------------------------------------------------------------------
SIM_DIR      = Path(__file__).parent
PROJECT_ROOT = SIM_DIR.parent

sys.path.insert(0, str(SIM_DIR))          # simulator/ modules
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # src/ (litterbox package)

from PIL import Image  # noqa: E402

from sim_config import (  # noqa: E402
    CATS, TEST_DATA_DIR, ASSETS_DIR,
    GROUND_TRUTH_PATH, REPORT_PATH, RANDOM_SEED,
    CAT_ACQUISITION_DATE,
    CLEAN_BOX_COLOR, USED_BOX_COLOR,
    PLACEHOLDER_WIDTH, PLACEHOLDER_HEIGHT,
)
from sensor_model import generate_readings  # noqa: E402
from schedule_generator import build_schedule  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_placeholder_images() -> tuple[str, str]:
    """Create clean_box.jpg and used_box.jpg in simulator/assets/ if needed."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = ASSETS_DIR / "clean_box.jpg"
    used_path  = ASSETS_DIR / "used_box.jpg"
    if not clean_path.exists():
        Image.new("RGB", (PLACEHOLDER_WIDTH, PLACEHOLDER_HEIGHT), CLEAN_BOX_COLOR).save(str(clean_path))
        print(f"  Created {clean_path.relative_to(PROJECT_ROOT)}")
    if not used_path.exists():
        Image.new("RGB", (PLACEHOLDER_WIDTH, PLACEHOLDER_HEIGHT), USED_BOX_COLOR).save(str(used_path))
        print(f"  Created {used_path.relative_to(PROJECT_ROOT)}")
    return str(clean_path), str(used_path)


def register_cats() -> None:
    """Register the first (reference) photo for each cat."""
    from litterbox.tools import register_cat_image  # noqa: PLC0415

    print("\n--- Registering cats ---")
    for cat_name in CATS:
        cat_dir = TEST_DATA_DIR / cat_name
        photos  = sorted(p for p in cat_dir.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not photos:
            print(f"  WARNING: no photos found for {cat_name} — skipping.")
            continue
        reference = photos[0]
        result = register_cat_image.invoke({
            "image_path": str(reference.resolve()),
            "cat_name":   cat_name,
        })
        print(f"  {cat_name}: {result}")


def run_entry(entry_image: str, readings, _clean_box_path: str) -> Optional[int]:
    """Call record_entry; return the visit_id extracted from the result string."""
    from litterbox.tools import record_entry  # noqa: PLC0415

    kwargs: dict = {
        "image_path":    entry_image,
        "weight_pre_g":  float(readings.weight_pre_g),
        "weight_entry_g": float(readings.weight_entry_g),
    }
    if readings.ammonia_peak_ppb is not None:
        kwargs["ammonia_peak_ppb"] = readings.ammonia_peak_ppb
    if readings.methane_peak_ppb is not None:
        kwargs["methane_peak_ppb"] = readings.methane_peak_ppb

    result = record_entry.invoke(kwargs)
    snippet = result[:160].replace("\n", " ")
    print(f"    ENTRY: {snippet}")

    m = re.search(r"Visit #(\d+) opened", result)
    if m:
        return int(m.group(1))
    print("    WARNING: could not parse visit_id from entry result.")
    return None


def run_exit(used_box_path: str, readings) -> str:
    """Call record_exit; return the result string."""
    from litterbox.tools import record_exit  # noqa: PLC0415

    kwargs: dict = {"image_path": used_box_path}
    if readings.weight_exit_g is not None:
        kwargs["weight_exit_g"] = float(readings.weight_exit_g)
    if readings.ammonia_peak_ppb is not None:
        kwargs["ammonia_peak_ppb"] = readings.ammonia_peak_ppb
    if readings.methane_peak_ppb is not None:
        kwargs["methane_peak_ppb"] = readings.methane_peak_ppb

    result = record_exit.invoke(kwargs)
    snippet = result[:160].replace("\n", " ")
    print(f"    EXIT:  {snippet}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Litter Box Monitor Simulator")
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED,
                        help="Random seed (default: %(default)s)")
    parser.add_argument("--no-register", action="store_true",
                        help="Skip cat registration step")
    parser.add_argument("--report-only", action="store_true",
                        help="Re-generate report from existing ground-truth JSON")
    args = parser.parse_args()

    if args.report_only:
        import sim_report  # noqa: PLC0415
        sim_report.generate(GROUND_TRUTH_PATH, REPORT_PATH)
        print(f"Report written to {REPORT_PATH}")
        return

    print("=" * 60)
    print("  Litter Box Monitor — Simulation")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    clean_box, used_box = generate_placeholder_images()

    if not args.no_register:
        register_cats()

    schedule = build_schedule(args.seed)
    rng      = random.Random(args.seed)

    print(f"\n--- Replaying {len(schedule)} simulated visits ---")

    ground_truth: list[dict] = []

    for event in schedule:
        readings = generate_readings(event.true_weight_g, event.is_anomalous, rng)

        tag = "ANOMALY" if event.is_anomalous else "normal"
        print(
            f"\n[Event {event.event_index:02d}/{len(schedule)}]"
            f"  {event.simulated_time[:16]}  |  {event.cat_name}  |  {tag}"
        )
        print(
            f"  Weights — pre: {readings.weight_pre_g} g  "
            f"entry: {readings.weight_entry_g} g  "
            f"exit: {readings.weight_exit_g} g  "
            f"waste(true): {readings.waste_g_true} g"
        )
        nh3 = f"{readings.ammonia_peak_ppb} ppb" if readings.ammonia_peak_ppb is not None else "null"
        ch4 = f"{readings.methane_peak_ppb} ppb" if readings.methane_peak_ppb is not None else "null"
        print(f"  Gas    — NH₃: {nh3}  CH₄: {ch4}")

        visit_id = run_entry(event.entry_image, readings, clean_box)
        run_exit(used_box, readings)

        ground_truth.append({
            "event_index":     event.event_index,
            "simulated_time":  event.simulated_time,
            "cat_name":        event.cat_name,
            "cat_true_weight_g": event.true_weight_g,
            "weight_pre_g":    readings.weight_pre_g,
            "weight_entry_g":  readings.weight_entry_g,
            "weight_exit_g":   readings.weight_exit_g,
            "waste_g_true":    readings.waste_g_true,
            "ammonia_peak_ppb": readings.ammonia_peak_ppb,
            "methane_peak_ppb": readings.methane_peak_ppb,
            "is_anomalous_seed": event.is_anomalous,
            "entry_image":     event.entry_image,
            "visit_id":        visit_id,
        })

    # Persist ground truth
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"\nGround truth written to {GROUND_TRUTH_PATH.relative_to(PROJECT_ROOT)}")

    # Generate Markdown report
    import sim_report  # noqa: PLC0415
    sim_report.generate(GROUND_TRUTH_PATH, REPORT_PATH)
    print(f"Report written to {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
