"""
eigen_sim_config.py — Configuration for the eigenanalysis simulator
====================================================================

Defines per-cat baseline waveform parameters, noise levels, visit counts,
and anomaly injection settings.

Each cat has a characteristic weight waveform shape defined as a parametric
function over normalised time t ∈ [0, 1]:

    [baseline] → [ramp up] → [plateau with character] → [ramp down] → [baseline + waste]

The parametric function returns the cat's weight contribution (grams above
box baseline).  The DC term (absolute weight = box + cat) is added separately.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
SIM_OUTPUT_DIR = Path(__file__).parent / "eigen_sim_reports"
GROUND_TRUTH_PATH = Path(__file__).parent / "eigen_sim_ground_truth.json"
SUMMARY_PATH = Path(__file__).parent / "eigen_sim_summary.md"

# ---------------------------------------------------------------------------
# Global simulation parameters
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
TOTAL_VISITS = 400
SWAP_PROBABILITY = 0.05       # 5% of visits use wrong cat's baseline

BOX_BASELINE_G = 2000.0       # empty box + litter weight
NOISE_FRACTION = 0.02         # noise sigma = this × baseline RMS

# Visit duration range (seconds) — determines raw sample count at 5s intervals
VISIT_DURATION_MIN_S = 50
VISIT_DURATION_MAX_S = 150
SAMPLE_INTERVAL_S = 5

# Weight variation: DC offset per visit ~ N(cat_weight, DC_JITTER_G)
DC_JITTER_G = 50.0

# Waste weight added after cat leaves (grams)
WASTE_MIN_G = 30.0
WASTE_MAX_G = 120.0

# ---------------------------------------------------------------------------
# Per-cat configuration
# ---------------------------------------------------------------------------

CATS = {
    "Anna": {
        "true_weight_g": 3200,
        "visits": 100,
        # Quick, decisive — very steep ramp, high-frequency ripple on plateau
        "ramp_fraction": 0.05,     # 5% — snappy transitions
        "plateau_type": "ripple",  # sinusoidal ripple (3 cycles)
        "plateau_param": 0.12,     # 12% ripple — ~384g oscillation
    },
    "Luna": {
        "true_weight_g": 5000,
        "visits": 100,
        # Settles slowly — very gradual ramp, deep sag in middle
        "ramp_fraction": 0.20,     # 20% — very gradual
        "plateau_type": "sag",     # dips deeply in the middle
        "plateau_param": 0.15,     # 15% sag — ~750g dip
    },
    "Marina": {
        "true_weight_g": 4000,
        "visits": 100,
        # Fidgety — medium ramp, pronounced double bumps
        "ramp_fraction": 0.10,
        "plateau_type": "bumps",   # two sharp bumps (repositions)
        "plateau_param": 0.18,     # 18% bumps — ~720g spikes
    },
    "Natasha": {
        "true_weight_g": 5500,
        "visits": 100,
        # Calm, heavy — steep ramp, broad parabolic rise
        "ramp_fraction": 0.06,
        "plateau_type": "parabola",  # smooth upward curve
        "plateau_param": 0.10,      # 10% rise — ~550g peak
    },
}
