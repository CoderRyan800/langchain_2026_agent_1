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
