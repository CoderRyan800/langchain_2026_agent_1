# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two AI agents built with LangChain and LangGraph:

- **Bob** (`src/basic_agent.py`) — General-purpose conversational assistant with web search and multimodal file upload
- **Litter Box Monitor** (`src/litterbox_agent.py`) — Automated cat health monitoring via litter box camera images, with 11 LangChain tools, two-stage CLIP+GPT-4o identity pipeline, sensor data ingestion (weight scale, gas sensors), and health analysis

## Environment Setup

```bash
conda create -n langchain_env_2026_1 python=3.11 -y
conda activate langchain_env_2026_1
pip install -r requirements.txt
```

Required `.env` file:
```
OPENAI_API_KEY=...
TAVILY_API_KEY=...   # Bob's web search only
```

## Running the Agents

```bash
# Bob (interactive)
python src/basic_agent.py

# Litter box (interactive)
python src/litterbox_agent.py

# Litter box (sensor-triggered — no user interaction)
python src/litterbox_agent.py --event entry --image /path/to/image.jpg
python src/litterbox_agent.py --event exit  --image /path/to/image.jpg

# Litter box (sensor-triggered with weight and gas readings)
python src/litterbox_agent.py --event entry --image /path/to/image.jpg \
    --weight-pre 5400 --weight-entry 5900 --ammonia-peak 45 --methane-peak 30
python src/litterbox_agent.py --event exit --image /path/to/image.jpg \
    --weight-exit 5680 --ammonia-peak 62 --methane-peak 41
```

Bob supports `/UPLOAD /path/to/file` for images/audio and `/STOP` to quit.

## Tests

```bash
# Manual integration test runner (uses real LLM calls)
python tests/run_manual_test.py                    # all 8 phases (~$0.25–0.50)
python tests/run_manual_test.py --phase 1          # phase 1 only (free, no LLM calls)
python tests/run_manual_test.py --phase 1 2 4      # multiple phases
python tests/run_manual_test.py --no-cleanup       # keep test artifacts

# Automated pytest suite (no LLM calls except slow CLIP tests)
pytest -m "not slow"    # 180 tests, ~10 s
pytest -m slow          # CLIP embedding tests (~350 MB model download on first run)
pytest                  # all 200 tests
```

Manual test phases: 1=storage/schema, 2=CLIP embeddings, 3=health analysis, 4=identity confirmation, 5=sensor CLI, 6=reset, 7=retroactive recognition, 8=sensor data ingestion. Phases 1, 4, 6 are free. The test suite uses isolated paths (`tests/test_data/`) to avoid touching production data.

Pytest test files: `test_db.py` (schema/migration), `test_health.py` (prompt builder/parser), `test_tools_core.py` (query tools), `test_tools_sensor.py` (record_entry/record_exit with sensors), `test_integration.py` (full lifecycle), `test_embeddings.py` (CLIP — slow).

## Architecture

### Bob (`src/basic_agent.py`)
- Model: `gpt-4o`; memory: `agent_memory.db` (SQLite via `SqliteSaver`)
- `SummarizationMiddleware`: after 10 messages, summarizes messages 1–7, keeps last 3
- Tool: Tavily web search

### Litter Box Agent (`src/litterbox_agent.py` + `src/litterbox/`)
- Model: `gpt-4o` (vision required)
- Memory: `data/agent_litterbox_memory.db`; thread IDs `"sensor"` vs `"interactive"` maintain separate histories
- **`src/litterbox/db.py`** — SQLite schema (cats, cat_images, visits, visit_sensor_events); `init_db()` is idempotent with automatic migration for old DBs missing sensor columns
- **`src/litterbox/embeddings.py`** — CLIP (`clip-ViT-B-32`, downloads ~350 MB on first run) + Chroma vector search; `ID_THRESHOLD = 0.82`
- **`src/litterbox/health.py`** — `build_health_prompt(**sensor_kwargs)` assembles the GPT-4o prompt with optional sensor readings; `HEALTH_PROMPT` constant preserved for backward compatibility; `parse_health_response()` unchanged
- **`src/litterbox/tools.py`** — 11 `@tool` functions; docstrings auto-generate LangChain tool descriptions

### Two-Stage Cat ID Pipeline
1. **CLIP** (local, free): embeds entry image, queries Chroma for top candidates
2. **GPT-4o** (API cost): visual side-by-side comparison for candidates above threshold (0.82)

If no candidate passes both stages, the visit is recorded unconfirmed for human review via `confirm_identity` tool.

### Sensor Data Ingestion
Each visit can capture readings from a weight scale and gas sensors. Data flows two ways:

- **Summary columns on `visits`**: `weight_pre_g`, `weight_entry_g`, `weight_exit_g`, `cat_weight_g` (derived: entry − pre), `waste_weight_g` (derived: exit − pre), `ammonia_peak_ppb`, `methane_peak_ppb` (peak = MAX of entry and exit readings)
- **Time-series log in `visit_sensor_events`**: one row per reading with `phase` (pre_entry / entry / exit), `sensor_type`, `value_numeric`, and `unit`

CLI flags: `--weight-pre G`, `--weight-entry G`, `--weight-exit G`, `--ammonia-peak PPB`, `--methane-peak PPB`. All are optional — omit any that are unavailable or malfunctioning. The flags are serialised into the sensor event prompt string; the LLM extracts the named values and passes them to the tool. See `docs/USER_GUIDE.md` §6 for full CLI examples and derivation rules.

### Health Analysis
`record_exit()` sends entry+exit images to GPT-4o; sensor readings are included in the prompt when available. Response stored in `visits.health_notes` with `is_anomalous` flag. Response always includes a mandatory veterinary disclaimer.

### Runtime Data (gitignored)
- `data/litterbox.db` — SQLite metadata
- `data/chroma/` — CLIP vector index
- `images/cats/`, `images/visits/`, `images/captures/` — image storage

### Simulator (`simulator/`)
A self-contained simulation harness that drives the production agent with realistic sensor data and real cat photographs (Anna, Luna, Marina, Natasha).

```bash
python simulator/run_simulation.py               # full run: register + 20 visits + report
python simulator/run_simulation.py --seed 123    # different random seed
python simulator/run_simulation.py --no-register # skip registration (cats already in DB)
python simulator/run_simulation.py --report-only # regenerate report from existing JSON
```

- **`simulator/sim_config.py`** — cat weights, noise params, visit quotas, random seed
- **`simulator/sensor_model.py`** — Gaussian weight noise + uniform gas readings with null dropout
- **`simulator/schedule_generator.py`** — reproducible 20-visit schedule; 3 random anomalies seeded
- **`simulator/run_simulation.py`** — registers cats via `register_cat_image`, replays events via direct `record_entry`/`record_exit` tool calls, writes `sim_ground_truth.json`
- **`simulator/sim_report.py`** — joins ground truth with live DB; produces `simulation_report.md`
- **`simulator/cat_pictures/`** — 19 real photos (4–6 per cat); photo[0] per cat is the reference
- **`simulator/assets/`** — generated placeholder box images (beige solid-colour JPEGs)

Outputs: `simulator/sim_ground_truth.json` (per-event ground truth), `simulator/simulation_report.md` (identity accuracy, weight accuracy, sensor coverage, anomaly detection).

Baseline result (seed=42): 70% identity accuracy (Anna 100%, Marina 80%, Luna 83%, Natasha 20%); weight error <33 g across all cats; sensor coverage 90%/85% NH₃/CH₄.

### Key Implementation Details
- `_abs()` in `tools.py` handles both absolute and project-relative paths
- Orphan exits (no open visit): `record_exit()` creates a flagged orphan record rather than failing; sensor data is stored on orphan rows too
- `retroactive_recognition()` scans unknown visits since a given date and re-runs the full CLIP+GPT-4o pipeline; called automatically by the agent after every new cat registration
- Peak gas reconciliation: `record_exit()` fetches the entry-phase gas readings and stores `MAX(entry, exit)` in the summary columns
- `waste_weight_g` is derived at exit time by reading `weight_pre_g` from the open visit before the UPDATE
- Pytest fixtures patch `DB_PATH`, `CHROMA_PATH`, `IMAGES_DIR`, and `PROJECT_ROOT` to `tmp_path` for full isolation; `_identify_cat` and `_run_gpt4o_vision` are stubbed in all non-slow tests
- Phase 7 uses a dedicated Chroma subdirectory (`chroma_phase7/`) to avoid file-lock conflicts with Phase 6's wipe

---

## Time-Domain Measurement System (branch: feature/time_domain_measurements)

This system adds continuous, time-series data capture to the litter box
monitor.  Instead of recording only per-visit snapshots at entry and exit
events, it maintains a **rolling buffer** of every sensor channel at a fixed
sample rate.  When a visit is detected the full M-minute window of vector data
is captured, analysed, and stored alongside the existing visit record.

The implementation is broken into four steps executed one at a time.  Each
step is committed and evaluated before the next begins.

---

### Configuration — `src/litterbox/td_config.json`

A single JSON file controls everything.  It is loaded once at startup.
All time-domain modules read from this file; nothing is hard-coded.

```jsonc
{
  "window_minutes": 10,        // M — rolling buffer length in minutes
  "samples_per_minute": 12,    // N — sample rate (12 = one every 5 s)

  "channels": [
    { "name": "weight_g",        "type": "weight",     "enabled": true  },
    { "name": "ammonia_ppb",     "type": "ammonia",    "enabled": true  },
    { "name": "methane_ppb",     "type": "methane",    "enabled": true  },
    { "name": "chip_id",         "type": "chip_id",    "enabled": true  },
    { "name": "similarity",      "type": "similarity", "enabled": true  }
  ],

  "trigger": {
    "weight_entry_delta_g":        300,  // min weight rise to declare kitty_present
    "weight_exit_delta_g":         200,  // min weight drop to declare kitty_absent
    "chip_absent_consecutive":       3,  // null chip readings in a row → kitty_absent
    "similarity_entry_threshold":  0.70, // min CLIP score to declare kitty_present
    "similarity_exit_threshold":   0.50, // score must fall below this → kitty_absent
    "similarity_sustained_peak_samples": 3  // P — top cat must exceed entry_threshold
                                            //     for at least P consecutive samples
                                            //     before ID is accepted (guards against
                                            //     a brief spike from a wrong cat)
  },

  "image_retention_days": 7    // camera images from visits deleted after this many days
}
```

Key design notes:
- `window_minutes` × `samples_per_minute` = total buffer depth (default 120 samples).
- The `channels` list is the **only** place where enabled sensors are defined.
  A box without a camera omits `similarity`; a box without a chip reader
  omits `chip_id`.  The rest of the system adapts automatically.
- The `similarity` channel is **dynamic**: the buffer stores one similarity
  score per registered cat at each sample tick.  The column names are derived
  at runtime from the registered cat list (e.g. `similarity_whiskers`,
  `similarity_anna`).  K varies with the number of registered cats.
- The `trigger` block holds all state-machine thresholds.  Any threshold
  can be tuned without touching code.

---

### Step 1 — Generic Rolling Buffer (`src/litterbox/time_buffer.py`)

**Goal:** A self-contained, hardware-agnostic data structure that holds
M × N time-stamped measurement dicts in a circular sliding window.

**Key classes / functions:**

```
RollingBuffer
  __init__(window_minutes, samples_per_minute)
      Computes max_len = window_minutes × samples_per_minute.
      Allocates a collections.deque(maxlen=max_len).
      Creates a threading.Lock for thread safety.

  append(timestamp: datetime, values: dict[str, float | str | None]) → None
      Acquires lock, appends (timestamp, values) to the deque.
      The deque automatically discards the oldest entry when full.

  snapshot() → list[dict]
      Returns a shallow copy of the entire buffer as a list of
      {"timestamp": ..., "values": {...}} dicts, oldest first.

  get_channel(name: str) → list[float | str | None]
      Returns a time-ordered list of values for one named channel.
      Missing keys in a sample produce None in the output list.

  get_timestamps() → list[datetime]
      Returns the timestamp column only.

  window_span_seconds() → float
      Returns (newest_ts - oldest_ts).total_seconds() or 0.0 if < 2 entries.

  clear() → None
      Empties the buffer (used by tests and simulator).
```

**Configuration loading:**

```
load_td_config(path: str | Path | None = None) → dict
    Reads td_config.json (default: src/litterbox/td_config.json).
    Returns the parsed dict.
    Raises FileNotFoundError with a clear message if missing.
    Validates required top-level keys (window_minutes, samples_per_minute,
    channels, trigger, image_retention_days) and raises ValueError if any
    are absent or have wrong types.
```

**What Step 1 does NOT include:** hardware drivers, sample scheduling,
state machine, DB writes — those come in later steps.

**Files created in Step 1:**
- `src/litterbox/time_buffer.py` — `RollingBuffer` class + `load_td_config()`
- `src/litterbox/td_config.json` — default configuration file
- `tests/test_time_buffer.py` — pytest unit tests (no LLM, no hardware)

**Tests for Step 1:**
- Buffer respects `maxlen`: inserting 130 samples into a 120-sample buffer
  leaves exactly 120 entries (oldest 10 gone).
- `get_channel()` returns correct values and None for missing keys.
- `window_span_seconds()` returns correct elapsed time.
- `snapshot()` returns a copy (mutating it does not affect the buffer).
- `load_td_config()` raises on missing file and on missing required keys.
- Config round-trip: load → check `window_minutes`, `samples_per_minute`,
  `channels` length, `trigger` keys.

**Step 1 status: COMPLETE** — 42/42 tests pass. See `docs/step1_test_report.md`.

---

### Step 2 — Sensor Collector (`src/litterbox/sensor_collector.py`)

**Goal:** Apply the `RollingBuffer` to the actual sensor channels defined in
the config.  Hardware is accessed through pluggable driver objects so the
collector can run with mock drivers during tests and with real hardware in
production.

**Design — hardware driver interface:**

Each channel type is served by a driver that implements a single method:

```
class BaseDriver:
    def read(self) -> float | str | None:
        """Return the current reading, or None if unavailable."""
```

Concrete drivers (one per channel type):

| Driver class | Channel type | Production source | Mock behaviour |
|---|---|---|---|
| `WeightDriver` | `weight` | scale hardware via serial/I²C | Returns configurable static value + Gaussian noise |
| `AmmoniaDriver` | `ammonia` | MQ-135 / ENS160 ADC | Returns configurable static value + noise |
| `MethaneDriver` | `methane` | MQ-4 / MQ-9 ADC | Returns configurable static value + noise |
| `ChipIdDriver` | `chip_id` | RFID/NFC reader | Returns configurable cat name or None |
| `SimilarityDriver` | `similarity` | CLIP embedder + current camera frame | Returns dict `{cat_name: score}` for all registered cats |

`SimilarityDriver` is the only driver whose `read()` returns a `dict` rather
than a scalar.  The collector expands it into per-cat channels named
`similarity_<catname>` before writing to the buffer.

**Important: CLIP-only on the continuous stream.**  `SimilarityDriver` uses
the local CLIP embedder exclusively — no GPT-4o calls during continuous
monitoring.  GPT-4o confirmation is reserved for post-visit analysis in
Step 4, applied to a small number of selected frames only.  This keeps the
per-tick cost at zero beyond local inference.

**Missing frames.**  If the camera cannot capture a usable frame at a given
tick (motion blur, lid closed, cat's back to lens, etc.), `SimilarityDriver`
returns `None` for every cat rather than 0.0.  The collector writes these as
absent keys in the values dict.  The buffer therefore stores genuine absences,
not spurious zero scores.  Downstream code must treat missing keys as `NaN`,
not zero.

**`SensorCollector` class:**

```
SensorCollector
  __init__(config: dict, drivers: dict[str, BaseDriver], buffer: RollingBuffer)
      Reads the enabled channel list from config.
      Stores the driver map and the shared RollingBuffer.

  _sample_once() → None
      Calls driver.read() for each enabled channel.
      Expands similarity dict into per-cat keys.
      Calls buffer.append(datetime.utcnow(), values).

  start() → None
      Launches a daemon thread that calls _sample_once() every
      (60 / samples_per_minute) seconds.

  stop() → None
      Signals the daemon thread to exit cleanly.
```

**Files created in Step 2:**
- `src/litterbox/sensor_collector.py` — `BaseDriver`, all driver classes,
  `SensorCollector`
- `tests/test_sensor_collector.py` — pytest unit tests with mock drivers

**Tests for Step 2:**
- `_sample_once()` with mock drivers populates the buffer correctly.
- Similarity dict is correctly expanded to `similarity_anna`, `similarity_luna`, etc.
- Disabled channels (enabled: false) are not sampled.
- SensorCollector with mock drivers runs for 3 ticks and produces 3 buffer
  entries (using a short tick interval in test).
- `stop()` terminates the background thread within 2 seconds.

**Step 2 status: NOT STARTED**

---

### Step 3 — State Machine and Visit Trigger (`src/litterbox/visit_trigger.py`)

**Goal:** Watch the rolling buffer, detect kitty-present and kitty-absent
transitions, and — on each kitty-absent transition — capture the M-minute
buffer snapshot and initiate the visit analysis pipeline.

**State machine:**

```
States:   KITTY_ABSENT  ←──────────────┐
               │                        │
               │  (first-stage trigger) │  (second-stage trigger)
               ▼                        │
          KITTY_PRESENT ────────────────┘
               │
               │  (on PRESENT → ABSENT transition)
               ▼
          capture_visit_snapshot()
```

**First-stage trigger (ABSENT → PRESENT):** any one of:
- Weight reading exceeds `(baseline_weight + weight_entry_delta_g)` where
  `baseline_weight` is the rolling median of the last N weight readings before
  the transition.
- `chip_id` channel returns a non-null value.
- Any `similarity_<cat>` channel exceeds `similarity_entry_threshold`.

**Second-stage trigger (PRESENT → ABSENT):** any one of:
- Weight reading falls below `(baseline_weight + weight_exit_delta_g)` after
  having been elevated.
- `chip_id` channel has returned None for `chip_absent_consecutive` consecutive
  samples.
- All `similarity_<cat>` channels have fallen below `similarity_exit_threshold`
  for `chip_absent_consecutive` consecutive samples.

**`VisitTrigger` class:**

```
VisitTrigger
  __init__(config: dict, buffer: RollingBuffer, on_visit_complete: Callable)
      Stores config thresholds and the callback.
      Initialises state = KITTY_ABSENT.
      Computes baseline_weight from the first N samples in the buffer.

  check(latest_values: dict) → None
      Called by SensorCollector after each _sample_once().
      Evaluates first-stage conditions if state == KITTY_ABSENT.
      Evaluates second-stage conditions if state == KITTY_PRESENT.
      Calls _on_entry() or _on_exit() as appropriate.

  _on_entry() → None
      Logs the transition with timestamp.
      Sets state = KITTY_PRESENT.
      Records entry_time.

  _on_exit() → None
      Logs the transition with timestamp.
      Sets state = KITTY_ABSENT.
      Calls on_visit_complete(snapshot, entry_time, exit_time).

  reset() → None
      Forces state back to KITTY_ABSENT (used by tests and simulator).
```

The `on_visit_complete` callback receives:
```python
on_visit_complete(
    snapshot:    list[dict],   # full M-minute buffer at moment of exit
    entry_time:  datetime,
    exit_time:   datetime,
)
```

**Files created in Step 3:**
- `src/litterbox/visit_trigger.py` — `VisitTrigger`, state constants
- `tests/test_visit_trigger.py` — pytest unit tests

**Tests for Step 3:**
- Feed a synthetic weight ramp: flat → sharp rise → sharp fall.
  Assert callback fires exactly once, entry_time and exit_time are correct.
- Chip-ID trigger: None × 5, non-null × 3, None × 4 consecutive → fires.
- Similarity trigger: all below threshold, one rises above, all fall below → fires.
- Multiple cats active simultaneously (two similarity scores elevated): still
  fires exactly once on exit.
- No spurious trigger when weight oscillates around threshold without a clean
  rise-then-fall.
- `reset()` returns to KITTY_ABSENT mid-visit without firing the callback.

**Step 3 status: NOT STARTED**

---

### Step 4 — Visit Capture, Analysis, and Storage

**Goal:** When the `on_visit_complete` callback fires, analyse the buffer
snapshot to produce a cat identification, store the visit record (including the
full vector snapshot), save camera images from the visit window, and integrate
with the existing `confirm_identity` / tentative-ID flow.

**Similarity DataFrame — the core data structure for camera-based ID:**

For a visit bounded by `entry_time` and `exit_time`, the analyser extracts
the relevant rows from the buffer snapshot and constructs a pandas DataFrame:

```
                        anna    luna   marina   natasha   whiskers
2026-04-04 22:10:00    0.91    0.23     0.18      0.21       0.19
2026-04-04 22:10:05    0.89    0.25     0.20      0.19       0.22
2026-04-04 22:10:10     NaN     NaN      NaN       NaN        NaN   ← no frame
2026-04-04 22:10:15    0.93    0.22     0.17      0.20       0.18
...
```

Design rules for the DataFrame:

- **Index**: `datetime` timestamps from the buffer — not integer row numbers.
  This enables natural time-slicing (`df.loc[entry_time:exit_time]`) and
  correlation with weight/gas channels.
- **Columns**: one per registered cat, named by cat name (not `similarity_anna`
  — the prefix is stripped when building the DataFrame).  K columns; K is
  determined at analysis time from the registered cat list and may vary between
  visits if cats are registered or removed.
- **Values**: CLIP cosine similarity in [0, 1].  Missing camera frames are
  stored as `NaN` (not 0.0).  Using `skipna=True` in all aggregations ensures
  missing frames do not bias the mean downward.
- **Shape**: rows = number of buffer samples between `entry_time` and
  `exit_time` (variable across visits); columns = K (variable across
  registrations).
- **Two windows**:
  - *Full window* — all M × N samples in the buffer at moment of exit.
    Serialised to JSON and stored in `td_visits.snapshot_json` for owner
    review and future re-analysis.  Captures pre-entry and post-exit context.
  - *Visit window* — rows trimmed to `[entry_time, exit_time]`.  Used for ID
    computation only.  Not stored separately.
- **Construction**: the `RollingBuffer` stores flat dicts
  (`{"similarity_anna": 0.91, "similarity_luna": 0.23, ...}`).  The
  `VisitAnalyser` is solely responsible for pivoting these into a DataFrame.
  The buffer itself has no knowledge of DataFrames.
- **Serialisation**: `df.to_json(orient="split")` stores index, columns, and
  data efficiently.  `pd.read_json(..., orient="split")` reconstructs it
  exactly, preserving the timestamp index and NaN values.

**Cat identification from the snapshot — priority order:**

1. **Chip ID** — if any sample in the visit window has a non-null `chip_id`,
   use the most frequent non-null value.  Mark `is_confirmed = True`.
   No DataFrame analysis needed.

2. **Similarity DataFrame** — applicable only when the `similarity` channel
   is enabled.  Steps:
   a. Build the visit-window DataFrame from buffer rows in `[entry_time, exit_time]`.
   b. Compute `col_means = df.mean(skipna=True)` — per-cat mean over the visit.
   c. Winning cat = `col_means.idxmax()`.
   d. **Sustained-peak gate**: verify that the winning cat's score exceeded
      `similarity_entry_threshold` for at least `similarity_sustained_peak_samples`
      (P) consecutive samples in the visit window.  This guards against a
      brief spike from a passing wrong cat.
   e. If the winning mean exceeds `similarity_entry_threshold` **and** the
      sustained-peak gate passes → record as tentative ID (mirrors existing
      CLIP pipeline behaviour).
   f. Otherwise → Unknown.

3. **Unknown** — if neither chip ID nor similarity produces a result, record
   as unknown and flag for human review.

**Camera image handling:**

- The `similarity` channel driver holds a reference to the camera.  At each
  sample tick it also saves a frame to a temporary ring buffer (in memory or
  on disk, configurable).
- On visit completion, frames captured between `entry_time` and `exit_time`
  are written to `images/visits/YYYY-MM-DD/<visit_uuid>/frame_NNNN.jpg`.
- A background task runs daily and deletes visit image directories older than
  `image_retention_days` (default 7).

**New DB table — `td_visits`:**

```sql
CREATE TABLE td_visits (
    td_visit_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_time        TIMESTAMP NOT NULL,
    exit_time         TIMESTAMP NOT NULL,
    chip_id           TEXT,                    -- raw chip ID string, if available
    tentative_cat_id  INTEGER REFERENCES cats(cat_id),
    confirmed_cat_id  INTEGER REFERENCES cats(cat_id),
    is_confirmed      BOOLEAN DEFAULT FALSE,
    id_method         TEXT,                    -- 'chip', 'similarity', 'unknown'
    top_similarity    REAL,                    -- highest mean score across cats
    snapshot_json     TEXT NOT NULL,           -- full M-minute buffer as JSON
    images_dir        TEXT,                    -- relative path to visit image folder
    health_notes      TEXT,
    is_anomalous      BOOLEAN DEFAULT FALSE,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**`VisitAnalyser` class (`src/litterbox/visit_analyser.py`):**

```
VisitAnalyser
  __init__(config: dict)

  analyse(snapshot, entry_time, exit_time) → TdVisitRecord
      Runs the ID priority logic above.
      Returns a dataclass with all fields needed for DB insertion.

  save(record: TdVisitRecord, images: list[Path]) → int
      Writes to td_visits, copies images to permanent storage.
      Returns the new td_visit_id.
      Triggers image deletion sweep for entries older than retention policy.
```

**Integration with existing tools:**

- `confirm_identity` tool gains an optional `td_visit_id` parameter so owners
  can confirm time-domain visits the same way they confirm snapshot visits.
- `get_unconfirmed_visits` returns both `visits` and `td_visits` rows.
- Health analysis: the same `build_health_prompt` / `parse_health_response`
  pipeline is run against the entry and exit frames from the image folder if
  a camera is present.

**Files created in Step 4:**
- `src/litterbox/visit_analyser.py` — `VisitAnalyser`, `TdVisitRecord`
- `src/litterbox/image_retention.py` — deletion sweep utility
- `src/litterbox/db.py` — `td_visits` table added to `init_db()` with migration

**Tests for Step 4:**
- `analyse()` returns chip-ID result (is_confirmed=True) when chip column is
  populated; DataFrame analysis is skipped entirely.
- `analyse()` returns correct tentative cat when similarity columns are above
  threshold, sustained-peak gate passes, and chip column is absent.
- `analyse()` returns Unknown when all column means are below threshold.
- `analyse()` returns Unknown when the winning cat's mean exceeds threshold
  but the sustained-peak gate fails (spike shorter than P samples).
- DataFrame construction: buffer with 3 cats and 5 ticks (including 1 NaN
  tick) produces a 5×3 DataFrame with correct NaN placement and timestamp index.
- `df.mean(skipna=True)` gives correct per-cat means that exclude NaN rows;
  verify that a spurious 0.0 would have incorrectly lowered the mean (NaN
  correctness justification test).
- Serialisation round-trip: `to_json(orient="split")` → `read_json` restores
  the same index, columns, and NaN positions.
- `save()` inserts a row and returns a valid `td_visit_id`.
- Deletion sweep removes image directories older than retention window and
  leaves newer ones intact.
- `confirm_identity` with `td_visit_id` updates the correct table.

**Step 4 status: NOT STARTED**

---

### Time-Domain Module — File Map

```
src/litterbox/
├── td_config.json          # configuration (window, sample rate, channels, thresholds)
├── time_buffer.py          # Step 1 — RollingBuffer + load_td_config()
├── sensor_collector.py     # Step 2 — BaseDriver, all drivers, SensorCollector
├── visit_trigger.py        # Step 3 — VisitTrigger state machine
├── visit_analyser.py       # Step 4 — VisitAnalyser, TdVisitRecord
└── image_retention.py      # Step 4 — visit image deletion sweep

tests/
├── test_time_buffer.py     # Step 1 tests
├── test_sensor_collector.py# Step 2 tests
├── test_visit_trigger.py   # Step 3 tests
└── test_visit_analyser.py  # Step 4 tests
```

### Time-Domain Module — Step Status Tracker

| Step | Description | Status |
|------|-------------|--------|
| 1 | `RollingBuffer` + config loading | NOT STARTED |
| 2 | `SensorCollector` + hardware driver interface | NOT STARTED |
| 3 | `VisitTrigger` state machine | NOT STARTED |
| 4 | `VisitAnalyser`, DB storage, image retention | NOT STARTED |
