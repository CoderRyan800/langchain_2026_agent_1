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

**Sensor-triggered** — called automatically by the camera system on entry and exit events. The agent runs without human input: it identifies the cat using a two-stage pipeline (local CLIP embeddings + GPT-4o visual confirmation), stores the visit record, and on exit runs a GPT-4o health analysis comparing the before and after images of the litter box.

Optional sensor data (weight scale and ammonia/methane gas sensors) can be passed via CLI flags. When present, weight and gas readings are stored in the `visits` table as summary columns and in the `visit_sensor_events` time-series log. Cat weight and waste weight are derived automatically, peak gas readings are reconciled across entry and exit, and the health analysis prompt is enriched with all available sensor data.

All health findings include a mandatory disclaimer and require veterinary review.

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
│       ├── health.py            # GPT-4o health analysis prompt and parser
│       └── tools.py             # All 11 LangChain tools (record_entry/exit accept sensor data)
├── tests/
│   ├── run_manual_test.py       # Manual integration test runner (8 phases, 80+ checks)
│   ├── conftest.py              # Shared pytest fixtures and isolation helpers
│   ├── test_db.py               # Schema, migration, and constraint tests
│   ├── test_health.py           # Health prompt builder and response parser tests
│   ├── test_tools_core.py       # Query and management tool tests
│   ├── test_tools_sensor.py     # record_entry / record_exit sensor tests
│   ├── test_integration.py      # Full visit lifecycle integration tests
│   └── test_embeddings.py       # CLIP embedding tests (slow — loads model)
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
