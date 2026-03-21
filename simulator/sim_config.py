"""Simulator configuration: cat definitions, noise parameters, schedule parameters."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
TEST_DATA_DIR = PROJECT_ROOT / "simulator" / "cat_pictures"
ASSETS_DIR    = PROJECT_ROOT / "simulator" / "assets"

GROUND_TRUTH_PATH = PROJECT_ROOT / "simulator" / "sim_ground_truth.json"
REPORT_PATH       = PROJECT_ROOT / "simulator" / "simulation_report.md"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Cat registry
# ---------------------------------------------------------------------------
# true_weight_g: actual cat weight in grams (ground truth only)
CATS = {
    "Anna":    {"true_weight_g": 3200},
    "Marina":  {"true_weight_g": 4000},
    "Luna":    {"true_weight_g": 5000},
    "Natasha": {"true_weight_g": 5500},
}

# Visit quotas — must sum to TOTAL_VISITS
VISIT_QUOTAS = {
    "Anna":    4,
    "Marina":  5,
    "Luna":    6,
    "Natasha": 5,
}

CAT_ACQUISITION_DATE = "2026-03-01"

# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
SIM_START_DATE = "2026-03-14"
SIM_DAYS       = 7
TOTAL_VISITS   = 20
NUM_ANOMALIES  = 3       # events randomly seeded with elevated gas readings

# ---------------------------------------------------------------------------
# Weight sensor noise model
# ---------------------------------------------------------------------------
BOX_BASELINE_G       = 2000    # empty box + litter baseline (g)
WEIGHT_PRE_NOISE_G   = 20      # Gaussian std dev for pre-entry reading
WEIGHT_ENTRY_NOISE_G = 30      # Gaussian std dev for entry (cat on scale)
WEIGHT_EXIT_NOISE_G  = 20      # Gaussian std dev for post-exit reading
WASTE_MIN_G          = 30      # minimum waste deposit (uniform draw)
WASTE_MAX_G          = 120     # maximum waste deposit

# ---------------------------------------------------------------------------
# Gas sensor noise model (ppb)
# ---------------------------------------------------------------------------
AMMONIA_NORMAL  = (5.0,   60.0)    # (min, max) uniform for normal visit
AMMONIA_ANOMALY = (150.0, 300.0)   # elevated range for seeded anomalies
AMMONIA_NULL_PROB = 0.10           # probability of sensor dropout

METHANE_NORMAL  = (0.0,  40.0)
METHANE_ANOMALY = (80.0, 180.0)
METHANE_NULL_PROB = 0.15

# ---------------------------------------------------------------------------
# Placeholder litter-box images
# ---------------------------------------------------------------------------
PLACEHOLDER_WIDTH  = 640
PLACEHOLDER_HEIGHT = 480
CLEAN_BOX_COLOR = (210, 200, 170)   # beige — empty/clean box
USED_BOX_COLOR  = (170, 160, 130)   # darker beige — box after visit
