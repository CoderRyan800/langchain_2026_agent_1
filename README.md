# LangChain 2026 Agent 1

Two AI agents built with LangChain and LangGraph:

| Agent | Entry point | Purpose |
|---|---|---|
| **Bob** | `src/basic_agent.py` | General-purpose conversational assistant with web search and multimodal file upload |
| **Litter Box Monitor** | `src/litterbox_agent.py` | Automated cat health monitoring via litter box camera images |

---

## Documentation

- **[User Guide](docs/USER_GUIDE.md)** — setup, day-to-day usage, sensor integration, troubleshooting
- **[Testing](docs/TESTING.md)** — test architecture, how to run the suite, baseline results

---

## Quick start

### 1. Create `.env`

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key      # Bob only
```

### 2. Install dependencies

```bash
# Conda (recommended)
conda create -n langchain_env_2026_1 python=3.11 -y
conda activate langchain_env_2026_1
pip install -r requirements.txt

# Python venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run

```bash
# Bob
python src/basic_agent.py

# Litter box agent — interactive
python src/litterbox_agent.py

# Litter box agent — sensor-triggered (image only)
python src/litterbox_agent.py --event entry --image images/captures/entry.jpg
python src/litterbox_agent.py --event exit  --image images/captures/exit.jpg

# Litter box agent — sensor-triggered with weight and gas readings
python src/litterbox_agent.py --event entry --image images/captures/entry.jpg \
    --weight-pre 5400 --weight-entry 5900 --ammonia-peak 45 --methane-peak 30
python src/litterbox_agent.py --event exit --image images/captures/exit.jpg \
    --weight-exit 5680 --ammonia-peak 62 --methane-peak 41
```

### 4. Run the test suite

```bash
# Automated pytest (no LLM calls — fast)
pytest -m "not slow"                          # 180 tests, ~10 s
pytest -m slow                                # CLIP embedding tests (downloads ~350 MB model once)

# Manual integration test runner (uses real LLM API calls)
python tests/run_manual_test.py               # all 8 phases (~$0.25–0.50)
python tests/run_manual_test.py --phase 1     # storage/schema only — free
python tests/run_manual_test.py --phase 8     # sensor data ingestion only — free
```

---

## How the agents work

### Bob

Bob is a persistent conversational assistant. He uses `gpt-4o`, Tavily web search, and SQLite-backed memory that survives restarts. Use `/UPLOAD <path>` to share an image or audio file for analysis, and `/STOP` to quit.

### Litter Box Monitor

The litter box agent tracks when cats use the litter box and flags potential health concerns. It operates in two modes:

**Interactive** — register cats, confirm identities, query records:
```
You: /UPLOAD /path/to/whiskers.jpg
You: Whiskers
Agent: When did you get Whiskers? (YYYY-MM-DD)
You: 2026-01-15
Agent: [scans old unknown visits and confirms any that match]
You: Show me all anomalous visits this month
You: Visit 7 is confirmed as Whiskers
```

**Sensor-triggered** — called automatically by the camera system on entry and exit events. The agent runs without human input: it identifies the cat using a two-stage pipeline (local CLIP embeddings + GPT-4o visual confirmation), stores the visit record, and on exit runs two parallel checks for health concerns: a GPT-4o visual analysis of the entry/exit images, and a per-cat data-driven anomaly score on the NH₃/CH₄ peak readings (signed z-score against the cat's own history; see `docs/USER_GUIDE.md` §8). Either check flagging the visit marks it as anomalous.

Optional sensor data (weight scale and ammonia/methane gas sensors) can be passed via CLI flags:

| Flag | Event | Description |
|------|-------|-------------|
| `--weight-pre G` | entry | Box + litter baseline before cat enters (g) |
| `--weight-entry G` | entry | Box + litter + cat at entry (g) |
| `--weight-exit G` | exit | Box + litter + waste after cat leaves (g) |
| `--ammonia-peak PPB` | entry or exit | Peak NH₃ reading (ppb) |
| `--methane-peak PPB` | entry or exit | Peak CH₄ reading (ppb) |

All flags are optional — omit any that are unavailable or malfunctioning. Cat weight (`weight_entry_g − weight_pre_g`), waste weight, and peak gas reconciliation are derived automatically. See [User Guide §6](docs/USER_GUIDE.md#6-sensor-integration) for full examples.

All health findings include a mandatory disclaimer and require veterinary review.

---

## Simulator

A self-contained harness that drives the production agent with realistic sensor noise and real cat photos, then produces a Markdown accuracy report.

```bash
conda activate langchain_env_2026_1
python simulator/run_simulation.py               # full run (~$0.50–1.50 in API calls)
python simulator/run_simulation.py --no-register # re-run without re-registering cats
python simulator/run_simulation.py --report-only # regenerate report only
```

Outputs written to `simulator/sim_ground_truth.json` and `simulator/simulation_report.md`. See [`simulator/README.md`](simulator/README.md) for full details.

**Baseline results (seed=42, 20 visits):**

| Metric | Result |
|--------|--------|
| Identity accuracy | 70% (14/20) |
| Anna | 4/4 correct |
| Marina | 4/5 correct |
| Luna | 5/6 correct |
| Natasha | 1/5 correct — needs better reference photo |
| Weight error | <33 g mean across all cats |
| Sensor coverage | 90% NH₃, 85% CH₄ (null dropout working) |

---

## How memory works

Both agents use LangGraph's `SqliteSaver` checkpointer to persist the full conversation state after every invocation. On restart, the latest checkpoint for the configured `thread_id` is loaded automatically.

`SummarizationMiddleware` keeps the active context window lean: when the message count reaches 10, messages 1–7 are replaced with a rolling summary and the 3 most recent messages are retained. The full raw history remains in the SQLite checkpoint store for audit purposes but does not affect the running agent.

Bob writes to `agent_memory.db`. The litter box agent writes to `data/agent_litterbox_memory.db`. Both can be migrated to Postgres by swapping `SqliteSaver` for LangGraph's `PostgresSaver`.

---

## Project structure

```
.
├── src/
│   ├── basic_agent.py           # Bob — general purpose agent
│   ├── litterbox_agent.py       # Litter box monitoring agent
│   └── litterbox/
│       ├── db.py                # SQLite schema and query helpers
│       ├── embeddings.py        # Local CLIP embeddings + Chroma vector search
│       ├── health.py            # GPT-4o health prompt, parser, refusal sanitiser
│       ├── gas_anomaly.py       # Per-cat data-driven NH₃/CH₄ detector (median + MAD)
│       ├── history_plot.py      # Per-cat Bokeh history plots (weight, NH₃, CH₄)
│       ├── rescore.py           # Re-score historical visits against current detector
│       └── tools.py             # All 13 LangChain tools (record_entry/exit + queries + plots)
├── tests/
│   ├── run_manual_test.py       # Manual integration test runner (8 phases, 80+ checks)
│   ├── conftest.py              # Shared pytest fixtures and isolation helpers
│   ├── test_db.py               # Schema, migration, and constraint tests
│   ├── test_health.py           # Health prompt builder, parser, refusal sanitiser
│   ├── test_gas_anomaly.py      # Per-cat data-driven gas anomaly detector tests
│   ├── test_history_plot.py     # Per-cat Bokeh history plot tests
│   ├── test_rescore.py          # Rescore-historical-visits utility tests
│   ├── test_tools_core.py       # Query and management tool tests
│   ├── test_tools_sensor.py     # record_entry / record_exit sensor tests
│   ├── test_integration.py      # Full visit lifecycle integration tests
│   └── test_embeddings.py       # CLIP embedding tests (slow — loads model)
├── simulator/
│   ├── README.md                # Simulator usage guide
│   ├── sim_config.py            # Cat weights, noise params, schedule config
│   ├── sensor_model.py          # Gaussian weight + gas sensor noise model
│   ├── schedule_generator.py    # Reproducible visit schedule builder
│   ├── run_simulation.py        # Main entry point
│   ├── sim_report.py            # Markdown report generator
│   ├── cat_pictures/            # Real photos: Anna, Luna, Marina, Natasha
│   ├── assets/                  # Generated placeholder box images
│   ├── sim_ground_truth.json    # Per-event ground truth from last run (gitignored)
│   └── simulation_report.md     # Accuracy report from last run (gitignored)
├── docs/
│   ├── USER_GUIDE.md            # Full user guide
│   └── TESTING.md               # Test procedure and baseline results
├── data/                        # Gitignored — created on first run
│   ├── litterbox.db             # Cat and visit metadata (SQLite)
│   ├── chroma/                  # CLIP vector index
│   └── agent_litterbox_memory.db
├── images/                      # Gitignored — created on first run
│   ├── cats/                    # Reference photos per cat
│   ├── visits/                  # Entry/exit captures organised by date
│   └── captures/                # Drop zone for incoming sensor images
├── agent_memory.db              # Bob's conversation store (gitignored)
├── .env                         # API keys (gitignored)
├── .gitignore
├── pytest.ini                   # Test discovery config and slow marker
├── requirements.txt
└── requirements-dev.txt         # Adds pytest and pytest-mock
```
