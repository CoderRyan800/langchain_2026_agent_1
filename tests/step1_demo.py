"""
step1_demo.py — Step 1 demonstration script
=============================================

This script exercises the complete Step 1 implementation:

  1. ``RollingBuffer`` — fills it with synthetic sensor data covering a
     realistic 10-minute litter-box visit.
  2. ``load_td_config`` — reads the real td_config.json shipped with the
     package.
  3. ``to_dataframe`` — converts the buffer to a pandas DataFrame and
     demonstrates prefix stripping and NaN handling.
  4. ``get_plot_backend("bokeh")`` — generates two standalone HTML plots:
        * ``output/step1_channels.html``  — weight, ammonia, methane channels
        * ``output/step1_similarity.html`` — per-cat CLIP similarity scores

Run this script from the repository root::

    python tests/step1_demo.py

Output files land in ``output/`` (created if absent).  Open the HTML files in
any browser — no internet connection required (Bokeh JS is inlined).

Design notes
------------
* The synthetic data model mimics a real visit: baseline → gradual weight rise
  (cat enters) → plateau → weight drop (cat exits) → return to baseline.
* Gas readings spike mid-visit to simulate ammonia and methane production.
* Five cats are registered with slowly-varying CLIP similarity scores; one cat
  (``luna``) has a clearly dominant score during the plateau phase.
* Every value is generated deterministically with ``random.seed(42)`` so the
  output is reproducible between runs.
"""

from __future__ import annotations

import math
import random
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — add src/ to sys.path so we can import litterbox without install
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from litterbox.time_buffer import RollingBuffer, load_td_config  # noqa: E402
from litterbox.td_plot import get_plot_backend                   # noqa: E402

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducible random source
# ---------------------------------------------------------------------------
rng = random.Random(42)

# ---------------------------------------------------------------------------
# Parameters — mimic td_config.json defaults
# ---------------------------------------------------------------------------
WINDOW_MINUTES: int = 10
SAMPLES_PER_MINUTE: int = 12          # one sample every 5 seconds
INTERVAL_SECONDS: float = 60.0 / SAMPLES_PER_MINUTE   # 5.0 s

TOTAL_SAMPLES: int = WINDOW_MINUTES * SAMPLES_PER_MINUTE  # 120

# Visit timing (sample indices into the 120-sample window)
VISIT_START: int = 20   # cat enters ~100 s after recording starts
VISIT_END: int   = 90   # cat exits ~450 s later (5.5-minute visit)

# Physical constants for synthetic readings
BASELINE_WEIGHT_G: float  = 5_400.0   # empty box + litter
CAT_WEIGHT_G: float       = 5_500.0   # luna ~5.5 kg
BASELINE_AMMONIA_PPB: float = 8.0
PEAK_AMMONIA_PPB: float     = 65.0
BASELINE_METHANE_PPB: float = 5.0
PEAK_METHANE_PPB: float     = 35.0

# CLIP similarity profiles — five cats, scores in [0, 1]
# luna is the visitor; the others have low, slowly-varying scores
CAT_NAMES: list[str] = ["anna", "luna", "marina", "natasha", "whiskers"]

def _similarity_profile(cat: str, sample_index: int) -> float:
    """Return a deterministic CLIP similarity score for a given cat and sample.

    luna dominates during the visit window (VISIT_START..VISIT_END).
    All other cats have low background noise in [0.10, 0.35].
    """
    t = sample_index / TOTAL_SAMPLES   # normalised time 0..1

    if cat == "luna":
        if VISIT_START <= sample_index <= VISIT_END:
            # Bell-curve peak centred on the visit midpoint
            mid = (VISIT_START + VISIT_END) / 2.0 / TOTAL_SAMPLES
            score = 0.90 * math.exp(-50 * (t - mid) ** 2) + 0.60
            score = min(score, 0.97)
            # Add a small amount of realistic noise
            score += rng.gauss(0, 0.015)
            return max(0.0, min(1.0, score))
        else:
            # Luna out of frame: NaN — return sentinel that the caller converts
            return float("nan")
    else:
        # Non-visiting cats: random low-level noise, never present (NaN)
        # during the actual visit to keep the demo clean.
        if VISIT_START <= sample_index <= VISIT_END:
            return float("nan")
        base = {"anna": 0.18, "marina": 0.22, "natasha": 0.15, "whiskers": 0.20}
        b = base.get(cat, 0.15)
        return max(0.0, min(1.0, b + rng.gauss(0, 0.03)))


# ---------------------------------------------------------------------------
# Build the synthetic buffer
# ---------------------------------------------------------------------------

def build_demo_buffer() -> RollingBuffer:
    """Create a RollingBuffer pre-filled with 120 samples of synthetic data.

    Returns
    -------
    RollingBuffer
        A full 10-minute buffer ready for plotting and DataFrame conversion.
    """
    buf = RollingBuffer(
        window_minutes=WINDOW_MINUTES,
        samples_per_minute=SAMPLES_PER_MINUTE,
    )

    # Start timestamp: a fixed UTC instant for reproducibility
    t0 = datetime(2026, 4, 4, 14, 0, 0, tzinfo=timezone.utc)

    for i in range(TOTAL_SAMPLES):
        ts = t0 + timedelta(seconds=i * INTERVAL_SECONDS)
        values: dict = {}

        # ----------------------------------------------------------------
        # Weight channel — S-curve transition on entry, mirror on exit
        # ----------------------------------------------------------------
        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        entry_ramp = sigmoid((i - VISIT_START) * 0.5)   # rises at VISIT_START
        exit_ramp  = sigmoid((i - VISIT_END)   * 0.5)   # rises at VISIT_END

        weight = (BASELINE_WEIGHT_G
                  + CAT_WEIGHT_G * entry_ramp
                  - CAT_WEIGHT_G * exit_ramp
                  + rng.gauss(0, 25.0))   # ±25 g sensor noise
        values["weight_g"] = round(weight, 1)

        # ----------------------------------------------------------------
        # Gas channels — bell-curve peaks during the visit
        # ----------------------------------------------------------------
        if VISIT_START <= i <= VISIT_END:
            progress = (i - VISIT_START) / max(1, VISIT_END - VISIT_START)
            gas_factor = math.sin(math.pi * progress)   # 0 → 1 → 0 over visit

            ammonia = (BASELINE_AMMONIA_PPB
                       + (PEAK_AMMONIA_PPB - BASELINE_AMMONIA_PPB) * gas_factor
                       + rng.gauss(0, 2.0))
            methane = (BASELINE_METHANE_PPB
                       + (PEAK_METHANE_PPB - BASELINE_METHANE_PPB) * gas_factor
                       + rng.gauss(0, 1.5))
        else:
            ammonia = BASELINE_AMMONIA_PPB + rng.gauss(0, 1.0)
            methane = BASELINE_METHANE_PPB + rng.gauss(0, 0.8)

        values["ammonia_ppb"] = round(max(0.0, ammonia), 2)
        values["methane_ppb"] = round(max(0.0, methane), 2)

        # ----------------------------------------------------------------
        # CLIP similarity channels — one key per registered cat.
        # NaN frames are simply omitted from the values dict so that
        # to_dataframe() correctly fills them with NaN (not 0.0).
        # ----------------------------------------------------------------
        for cat in CAT_NAMES:
            score = _similarity_profile(cat, i)
            if not math.isnan(score):
                values[f"similarity_{cat}"] = round(score, 4)
            # If score is NaN the key is absent from values — correct behaviour.

        buf.append(ts, values)

    return buf


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Step 1 Demo — RollingBuffer + Bokeh Plotter")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load and display the real td_config.json
    # ------------------------------------------------------------------
    print("\n[1] Loading td_config.json ...")
    cfg = load_td_config()
    print(f"    window_minutes     : {cfg['window_minutes']}")
    print(f"    samples_per_minute : {cfg['samples_per_minute']}")
    print(f"    channels           : {[c['name'] for c in cfg['channels']]}")
    print(f"    image_retention_days: {cfg['image_retention_days']}")

    # ------------------------------------------------------------------
    # 2. Build the buffer and print summary stats
    # ------------------------------------------------------------------
    print("\n[2] Building synthetic RollingBuffer ...")
    buf = build_demo_buffer()
    print(f"    {buf}")
    print(f"    window span : {buf.window_span_seconds():.1f} s")
    print(f"    is_full     : {buf.is_full()}")

    # Quick channel stats
    weights = [v for v in buf.get_channel("weight_g") if v is not None]
    print(f"    weight_g    : min={min(weights):.0f} g  max={max(weights):.0f} g")

    ammonia = [v for v in buf.get_channel("ammonia_ppb") if v is not None]
    print(f"    ammonia_ppb : min={min(ammonia):.1f}  max={max(ammonia):.1f}")

    # ------------------------------------------------------------------
    # 3. DataFrame conversion — similarity channels
    # ------------------------------------------------------------------
    print("\n[3] Converting similarity channels to DataFrame ...")
    sim_df = buf.to_dataframe(channel_prefix="similarity_")
    print(f"    shape  : {sim_df.shape}  (rows × cats)")
    print(f"    columns: {list(sim_df.columns)}")
    non_nan = sim_df.count()
    print("    non-NaN counts per cat:")
    for cat_col in sim_df.columns:
        print(f"      {cat_col:10s}: {non_nan[cat_col]} / {len(sim_df)}")

    # Per-cat mean (NaN-skipping) — luna should dominate
    means = sim_df.mean(skipna=True)
    print("    mean similarity per cat (NaN-skipped):")
    for cat_col in sim_df.columns:
        print(f"      {cat_col:10s}: {means[cat_col]:.4f}")

    # ------------------------------------------------------------------
    # 4. Generate Bokeh plots
    # ------------------------------------------------------------------
    print("\n[4] Generating Bokeh HTML plots ...")
    backend = get_plot_backend("bokeh")

    # --- Scalar channels plot ---
    channels_path = OUTPUT_DIR / "step1_channels.html"
    backend.plot_channels(
        timestamps  = buf.get_timestamps(),
        channels    = {
            "weight_g"   : buf.get_channel("weight_g"),
            "ammonia_ppb": buf.get_channel("ammonia_ppb"),
            "methane_ppb": buf.get_channel("methane_ppb"),
        },
        title       = "Step 1 Demo — Sensor Channels (synthetic)",
        output_path = channels_path,
    )
    print(f"    Saved: {channels_path.relative_to(REPO_ROOT)}")

    # --- Similarity DataFrame plot ---
    sim_path = OUTPUT_DIR / "step1_similarity.html"
    backend.plot_similarity_dataframe(
        df          = sim_df,
        title       = "Step 1 Demo — Cat Similarity Scores (luna visiting)",
        output_path = sim_path,
        threshold   = cfg["trigger"]["similarity_entry_threshold"],
    )
    print(f"    Saved: {sim_path.relative_to(REPO_ROOT)}")

    # ------------------------------------------------------------------
    # 5. Full-window DataFrame (all channels, no prefix filter)
    # ------------------------------------------------------------------
    print("\n[5] Full-window DataFrame (all channels) ...")
    full_df = buf.to_dataframe()
    print(f"    shape  : {full_df.shape}")
    print(f"    columns: {list(full_df.columns)}")

    # ------------------------------------------------------------------
    # 6. Time-range filtering
    # ------------------------------------------------------------------
    print("\n[6] Time-range filtering — visit window only ...")
    timestamps = buf.get_timestamps()
    visit_start_ts = timestamps[VISIT_START]
    visit_end_ts   = timestamps[VISIT_END]
    visit_df = buf.to_dataframe(
        channel_prefix="similarity_",
        start=visit_start_ts,
        end=visit_end_ts,
    )
    print(f"    visit window rows : {len(visit_df)}  "
          f"(expected ~{VISIT_END - VISIT_START + 1})")
    print(f"    luna mean in window: {visit_df['luna'].mean(skipna=True):.4f}")

    print("\n" + "=" * 60)
    print("Demo complete.  Open the HTML files in your browser:")
    print(f"  {channels_path}")
    print(f"  {sim_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
