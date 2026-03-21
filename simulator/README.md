# Litter Box Monitor — Simulator

A self-contained simulation harness that drives the production litter-box
monitoring agent with realistic sensor data and real cat photographs.

---

## Purpose

- Validates cat identification accuracy using actual photos of Anna, Marina,
  Luna, and Natasha.
- Exercises the full sensor data pipeline (weight scale, ammonia, methane)
  with realistic Gaussian noise and random sensor dropouts.
- Seeds a small number of anomalous events (elevated gas readings) to test
  anomaly detection.
- Produces a structured Markdown report comparing agent outputs against
  ground truth.

---

## Cat weights

| Cat | True weight |
|-----|-------------|
| Anna | 3,200 g |
| Marina | 4,000 g |
| Luna | 5,000 g |
| Natasha | 5,500 g |

---

## Running the simulator

```bash
conda activate langchain_env_2026_1

# Full run (register cats + 20 visits + report)
python simulator/run_simulation.py

# Override the random seed
python simulator/run_simulation.py --seed 123

# Re-run events without re-registering (DB already has the cats)
python simulator/run_simulation.py --no-register

# Regenerate the report without re-running the simulation
python simulator/run_simulation.py --report-only
```

The first run downloads the CLIP model (~350 MB, cached afterwards).
Each of the 20 visits makes two GPT-4o API calls (identification + health
analysis), so a full run costs approximately $0.50–$1.50.

---

## Outputs

| File | Description |
|------|-------------|
| `simulator/assets/clean_box.jpg` | Beige placeholder — clean box image |
| `simulator/assets/used_box.jpg` | Darker beige placeholder — used box image |
| `simulator/sim_ground_truth.json` | Per-event ground-truth log (sensor values, true cat, visit_id) |
| `simulator/simulation_report.md` | Markdown report with identity, weight, and anomaly metrics |

---

## File structure

```
simulator/
├── README.md               # This file
├── sim_config.py           # Cat definitions, noise params, schedule params
├── sensor_model.py         # Weight and gas sensor noise generation
├── schedule_generator.py   # Reproducible visit schedule builder
├── run_simulation.py       # Main entry point
├── sim_report.py           # Report generator (also runnable standalone)
├── assets/                 # Placeholder litter-box images (created on first run)
└── cat_pictures/           # Reference and visit photos per cat
    ├── Anna/
    ├── Luna/
    ├── Marina/
    └── Natasha/
```

---

## Sensor noise model

**Weight scale (Gaussian noise)**

| Reading | Formula | Std dev |
|---------|---------|---------|
| `weight_pre_g` | box baseline 2,000 g + noise | ±20 g |
| `weight_entry_g` | pre + true cat weight + noise | ±30 g |
| `weight_exit_g` | pre + waste deposited + noise | ±20 g |
| waste deposit | uniform draw | 30–120 g |

**Gas sensors**

| Sensor | Normal range | Anomalous range | Null probability |
|--------|-------------|-----------------|-----------------|
| NH₃ (ammonia) | 5–60 ppb | 150–300 ppb | 10% |
| CH₄ (methane) | 0–40 ppb | 80–180 ppb | 15% |

Three events per run are randomly seeded as anomalous (both gas sensors
pushed into the elevated range when non-null).

---

## Reference photos

Cat photos live in `simulator/cat_pictures/<CatName>/`.
The first photo (alphabetically) is used as the registration reference;
all photos (including the reference) are available for visit cycles and
may be reused.

The simulator does **not** modify or resize the originals.
