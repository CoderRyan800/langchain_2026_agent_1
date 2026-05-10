# Developer Introduction

For software engineers who can read Python but **don't have a machine
learning background**. This document gets you oriented in the codebase,
explains where to look when things break, and walks you through common
extension tasks. The math is mentioned only when needed for context;
when you want to dig into the algorithms themselves, see `ML_TUTORIAL.md`.

---

## What this project actually is

Two LangChain/LangGraph agents:

1. **Bob** (`src/basic_agent.py`) — a general-purpose chat assistant with
   web search and file upload. Mostly an example/baseline. You can
   ignore Bob if you're working on the litter box system.

2. **Litter Box Monitor** (`src/litterbox_agent.py`) — the main project.
   A monitoring system for cat health that combines:
   - A camera that captures photos of litter box visits
   - Optional weight + gas sensors
   - A vision model (GPT-4o) that analyses the photos
   - Several **anomaly detectors** that score each visit and the cat's
     long-term trends
   - A SQLite database that stores everything
   - A LangGraph agent that wraps it all behind plain-English chat

The agent has 16 tools (Python functions decorated with `@tool`) that it
can call when the user asks something. Most of the codebase is the
machinery behind those tools — detectors, database, image storage,
report generators.

---

## High-level architecture

```
                  ┌──────────────────────────┐
   user types →   │  src/litterbox_agent.py  │  ← LangChain agent loop
                  │  (interactive CLI)       │      (GPT-4o + tools)
                  └──────────┬───────────────┘
                             │
                             ▼  agent calls one of 16 tools
                  ┌──────────────────────────┐
                  │  src/litterbox/tools.py  │  ← @tool functions
                  └──────────┬───────────────┘
                             │
            ┌────────────────┼─────────────────┬──────────────────┐
            ▼                ▼                 ▼                  ▼
   src/litterbox/db.py  embeddings.py    health.py         (one of the
   (sqlite schema +    (CLIP image-      (GPT-4o vision    detectors —
    queries)            similarity)       prompt + parser)  see below)
```

Three things to know up-front:

- **The agent is just GPT-4o + a tool list.** When you ask "Show me
  Luna's visits", GPT-4o decides to call `get_visits_by_cat("Luna")`,
  the tool runs, the result comes back as text, GPT-4o formats it for
  you. There's no special "intent recognition" code — the system prompt
  in `src/litterbox_agent.py` and the tool docstrings are what teach
  GPT-4o which tool to call.

- **Most of the heavy lifting is in plain Python, not the agent.** The
  tools call into ordinary functions in `src/litterbox/`. You can call
  any of those functions directly without involving the agent at all.
  See `src/litterbox/api.py` for a `LitterboxAgent` class that exposes
  every tool as a regular method.

- **No streaming, no async.** Every operation is synchronous. The
  agent loop is request-response.

---

## The five anomaly detectors (in plain English)

The system has **five** independent ways to look for something wrong.
Each looks at a different kind of evidence. They run in parallel and
their verdicts are combined.

| Detector | Asks | When it runs | Code |
|---|---|---|---|
| **CLIP cat ID** | "Whose photo is this?" | At entry of every visit | `embeddings.py` |
| **GPT-4o vision** | "Is anything visually concerning in this photo?" | At exit of every visit | `health.py` |
| **Gas anomaly** | "Is this visit's NH₃/CH₄ spike unusually high for this cat?" | At exit of every visit | `gas_anomaly.py` |
| **Eigen + cluster** | "Does this visit's WAVEFORM look like the cat's typical pattern?" | After each visit, batch | `eigen_analyser.py` + `cluster_analyser.py` |
| **Long-term trend** | "Has this cat been gradually drifting over weeks?" | On demand (and auto-surfaced by `get_anomalous_visits`) | `trend_anomaly.py` |

The tutorial document (`ML_TUTORIAL.md`) explains *how* each one works.
This document just tells you where they live and what they do.

A key design rule: **the detectors are pure functions on a database
connection.** None of them keep state between calls. If you want to
debug a detector's verdict, you can construct a tiny in-memory database,
populate it with synthetic visits, and call the detector directly. The
test files (`tests/test_*.py`) all do this.

---

## File map — what lives where

```
src/
├── basic_agent.py            ← Bob (ignore unless working on Bob)
├── litterbox_agent.py        ← The main agent's interactive loop + system prompt
└── litterbox/
    ├── _cli.py               ← Console-script entry point (alternate runner)
    ├── api.py                ← LitterboxAgent class — Python API
    │
    ├── db.py                 ← SQLite schema + connection helpers
    ├── embeddings.py         ← CLIP image embedder + Chroma vector index
    ├── health.py             ← GPT-4o vision prompt + response parser
    │
    ├── tools.py              ← All 16 LangChain @tool functions
    │
    ├── gas_anomaly.py        ← Per-visit NH₃/CH₄ detector
    ├── trend_anomaly.py      ← Long-term per-cat drift detector
    ├── rescore.py            ← Maintenance: recompute gas verdicts
    │
    ├── time_buffer.py        ← Rolling sample buffer (time-domain Step 1)
    ├── sensor_collector.py   ← Hardware driver interface (Step 2)
    ├── visit_trigger.py      ← Entry/exit state machine (Step 3)
    ├── visit_analyser.py     ← Identifies the cat from a buffer snapshot (Step 4)
    ├── image_retention.py    ← Deletes old visit images (Step 4)
    │
    ├── eigen_analyser.py     ← Waveform shape detector — Layer 1 (Step 5a)
    ├── eigen_query.py        ← Read-side queries + HTML report
    ├── cluster_analyser.py   ← Coefficient-cluster detector — Layer 2 (Step 5c)
    ├── analyser_pipeline.py  ← Plugin orchestrator that chains 5a → 5c
    │
    ├── history_plot.py       ← Bokeh plot: weight + gas over time (per cat)
    ├── td_plot.py            ← Abstract time-domain plot interface
    ├── td_plot_bokeh.py      ← Bokeh implementation
    │
    └── td_config.json        ← All tunable thresholds and windows

tests/
├── conftest.py               ← Fixtures: tmp dirs, isolated DBs, image stubs
└── test_*.py                 ← One file per module above
```

A few naming conventions:

- Files named `*_anomaly.py` are detectors. Each has a `score_X()` function
  that takes an open SQLite connection and returns a verdict dict.
- Files named `td_*` belong to the time-domain (waveform) measurement
  system. If you're not working on continuous sample buffers or eigen
  analysis, you can ignore these.
- The Step N comments (Step 1, Step 2, etc.) refer to a build sequence
  documented in CLAUDE.md. Step 5 is the eigen + cluster system; Step 4
  is the visit analyser; Step 1–3 are the time-domain plumbing.

---

## Database schema (the parts that matter)

You'll touch these three tables most often:

### `cats`
```sql
cat_id        INTEGER PRIMARY KEY
name          TEXT UNIQUE NOT NULL
created_at    TIMESTAMP
```

One row per registered cat.

### `cat_images`
```sql
image_id      INTEGER PRIMARY KEY
cat_id        INTEGER REFERENCES cats(cat_id)
image_path    TEXT
chroma_id     TEXT     -- foreign key into the Chroma vector index
```

Reference photos used by CLIP to identify visiting cats.

### `visits`
```sql
visit_id              INTEGER PRIMARY KEY
entry_time            TIMESTAMP
exit_time             TIMESTAMP
entry_image_path      TEXT
exit_image_path       TEXT

-- Identity (pick one)
tentative_cat_id      INTEGER  -- best CLIP guess
confirmed_cat_id      INTEGER  -- human-confirmed
is_confirmed          BOOLEAN
similarity_score      REAL     -- CLIP cosine similarity, 0..1

-- Sensors
weight_pre_g          REAL
weight_entry_g        REAL
weight_exit_g         REAL
cat_weight_g          REAL     -- derived: entry - pre
waste_weight_g        REAL     -- derived: exit - pre
ammonia_peak_ppb      REAL
methane_peak_ppb      REAL

-- Gas anomaly verdict (written by record_exit, rewritten by rescore.py)
ammonia_z_score          REAL
methane_z_score          REAL
gas_anomaly_tier         TEXT   -- normal | mild | significant | severe | insufficient_data
gas_anomaly_n_samples    INTEGER
gas_anomaly_model_used   TEXT   -- per_cat | pooled | insufficient_data
gas_anomaly_rescored_at  TIMESTAMP

-- Health analysis verdict
health_notes             TEXT   -- GPT-4o response (or sanitised placeholder)
is_anomalous             BOOLEAN  -- LLM verdict OR gas tier in {mild, significant, severe}
is_orphan_exit           BOOLEAN
```

The trend detector reads from this table directly (no separate storage)
and computes its verdict on demand.

There's also `td_visits`, `eigen_waveforms`, `eigen_models`, and
`cluster_models` — these store the time-domain (waveform) data. You'll
only touch them if you're working on the eigen/cluster system.

The full schema is in `src/litterbox/db.py`. `init_db()` is idempotent
and includes migration logic for older DBs.

---

## How a typical visit flows through the system

When the camera triggers an entry event, this happens:

```python
# 1. Camera saves entry photo, calls record_entry tool
record_entry(
    image_path="/path/to/entry.jpg",
    weight_pre_g=5400, weight_entry_g=8600,
    ammonia_peak_ppb=42, methane_peak_ppb=18,
)
```

Inside `record_entry` (in `src/litterbox/tools.py`):

1. Copy the image to permanent storage under `images/visits/YYYY-MM-DD/`.
2. Call `_identify_cat()`:
   a. CLIP embeds the image (`embeddings.py`)
   b. Query Chroma for the top candidates among registered reference photos
   c. If the best match exceeds `ID_THRESHOLD = 0.82`, call GPT-4o to
      visually confirm by side-by-side comparison
3. Insert a new row into `visits` with the entry data and tentative cat ID.
4. Return a text summary string to the agent.

Then the cat leaves and the camera triggers `record_exit`:

```python
record_exit(
    image_path="/path/to/exit.jpg",
    weight_exit_g=5485, ammonia_peak_ppb=58, methane_peak_ppb=22,
)
```

Inside `record_exit`:

1. Copy exit image, find the open visit (most recent `exit_time IS NULL`)
2. Update the visit row with exit data, derive `waste_weight_g`, take peak
   gas readings (max of entry and exit values)
3. **Score the gas anomaly** (`gas_anomaly.score_gas_visit`) — this is
   pure math on the visit's history, returns a tier
4. **Build the health prompt** including the gas tier as context
5. **Call GPT-4o vision** with entry + exit images plus the prompt
6. Parse the response, OR the LLM verdict with the gas alarm tier into
   `is_anomalous`, store everything

That single function call covers identification, sensor analysis, vision
analysis, and final verdict. The agent (GPT-4o-the-LLM) just sees a text
summary back.

The eigen + cluster detectors run separately in the time-domain (waveform)
pipeline, which is fed by the `SensorCollector` in continuous mode rather
than the per-visit camera trigger.

The trend detector runs lazily — on demand whenever `get_trend_summary`
or `get_anomalous_visits` is called.

---

## Common tasks

### Add a new tool the agent can call

1. Pick a name and signature. Tools must take simple typed args
   (str, int, float, bool — no dicts or objects from the agent side).
2. Write a function in `src/litterbox/tools.py` decorated with `@tool`.
   The docstring is what GPT-4o sees as the tool description, so write
   it clearly:

   ```python
   @tool
   def my_new_tool(cat_name: str, days: int = 30) -> str:
       """One-line description of what the tool does.

       Use this whenever the user asks about [thing]. Returns [shape of result].

       cat_name: the cat to query.
       days:     how many days back (default 30).
       """
       init_db()
       with get_conn() as conn:
           # ... do the work ...
       return "human-readable summary"
   ```

3. Add the function to `ALL_TOOLS` at the bottom of `tools.py`.
4. Optionally add a wrapper method to `LitterboxAgent` in `src/litterbox/api.py`
   so it's callable without going through the agent.
5. Update the system prompt in `src/litterbox_agent.py` if the tool is
   for a use case that the existing prompt doesn't already route to.
6. Write a test in `tests/test_tools_core.py` (or a dedicated file).

### Change a detector threshold

All thresholds for the time-domain detectors live in
`src/litterbox/td_config.json`:

```json
{
  "gas_anomaly":   { "z_score_thresholds": { ... } },
  "trend_anomaly": { "z_score_thresholds": { ... }, "weight_pct_thresholds": { ... } },
  "eigen":         { "anomaly_thresholds": { ... } },
  "cluster":       { "z_score_thresholds": { ... } }
}
```

Change a number, save the file. The next call to a detector reads the new
value (no restart needed for stateless detectors; restart for the long-
running agent).

If you change a threshold AND want existing visit verdicts updated
to match, run:

```bash
python -m litterbox.rescore           # rescore all gas verdicts
```

The trend detector is computed on demand so its verdicts auto-update.
The eigen/cluster verdicts are one-shot per visit; the existing rescore
utility doesn't cover those (gap — could be added).

### Debug a wrong verdict

Every detector is a pure function. To reproduce a verdict offline:

```python
import sqlite3
from datetime import datetime, timezone
from litterbox.gas_anomaly import score_gas_visit

conn = sqlite3.connect("data/litterbox.db")  # the live DB
conn.row_factory = sqlite3.Row

# Score a specific visit's gas readings against history
result = score_gas_visit(
    conn,
    cat_id=4,                 # Luna
    ammonia_peak_ppb=190.0,
    methane_peak_ppb=91.2,
    exclude_visit_id=110,     # exclude this row from the historical fit
)
print(result)
# {'ammonia_z': 6.55, 'methane_z': 0.67,
#  'overall_tier': 'severe', 'model_used': 'per_cat', 'n_samples': 22, ...}
```

Same pattern for `trend_anomaly.score_trends`, `eigen_analyser.EigenAnalyser`,
etc.

### Add a new detector

1. Create `src/litterbox/your_detector.py` with a `score_X(conn, ...)`
   function. Keep it pure (no module-level state, no side effects beyond
   what the caller asks for).
2. If it should run automatically, decide where to plug in — for per-visit
   detectors, hook into `record_exit` in `tools.py`; for cross-visit,
   wire it into a tool the agent calls.
3. Add a config block to `td_config.json` and a constants section to
   your module that loads from it.
4. Add tests in `tests/test_your_detector.py`. Use the patterns in
   `test_gas_anomaly.py` or `test_trend_anomaly.py` as templates.
5. Add an OC characterization to `simulator/oc_study.py` so the
   detector's false-positive and detection rates are measured before
   it's trusted in production. This is non-negotiable for any new
   detector — calibration matters.

### Run the tests

```bash
pytest -m "not slow"     # 629 tests, ~22 seconds
pytest -m slow           # CLIP embedding tests, ~minute (downloads model first time)
pytest                   # all 649 tests
```

Tests use isolated tmp directories — they cannot touch your live
database. Each test gets a fresh schema.

### Run the simulator

The simulator generates synthetic visits and runs them through the live
detector code. Useful for end-to-end smoke testing or for measuring
detector behaviour at scale.

```bash
python simulator/run_simulation.py --no-register     # 20 visits, real GPT-4o calls (~$0.50)
python simulator/oc_study.py --quick                 # synthetic OC study, ~1 min, no API calls
python simulator/oc_study.py                         # full OC study, ~9 min
python simulator/eigen_sim.py                        # 200 waveform visits through the eigen pipeline
```

`oc_study.py` is the operating-characteristic study that measures FPR
and TPR for every detector at the documented thresholds. The output is
in `simulator/oc_report.md` (gitignored).

---

## Configuration that matters

### `.env` (gitignored — keep your keys here)

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...           # for Bob's web search; optional
LANGSMITH_TRACING=false           # silence trace upload errors
LANGCHAIN_TRACING_V2=false        # legacy var name, also disable
```

The agent code calls `load_dotenv(override=True)` at startup so these
beat any conflicting shell environment variables.

### `td_config.json`

All algorithmic thresholds and windows. See in-line `_comment` fields
for what each block does. Safe to edit and hot-reload (next call to a
stateless detector picks up the change).

### `agent_litterbox_memory.db`

LangGraph's conversation checkpointer. Each interactive run gets a fresh
thread by default (timestamped). Old threads accumulate but don't affect
new sessions. To wipe: `rm data/agent_litterbox_memory.db`.

---

## Things that bite

A few non-obvious gotchas worth knowing about:

- **The agent's "interactive" thread used to persist across runs and got
  poisoned with stale context.** This was fixed — every process now gets
  a fresh thread (timestamped) by default. To resume a specific
  conversation, set `LITTERBOX_THREAD=foo` before launching.

- **OpenAI's content filter sometimes refuses litter box images.** The
  health prompt has anti-refusal preamble language but it still happens
  occasionally. `safe_health_notes()` in `health.py` writes a clean
  placeholder to the database when the LLM refuses, so the visit isn't
  contaminated with garbage text. The data-driven gas detector keeps
  working regardless.

- **CLIP needs ~350 MB downloaded on first run.** First call to
  `embed_image` will hang for a minute or two while sentence-transformers
  pulls the model. After that it's cached.

- **`init_db()` is idempotent but it does run a migration check.** Don't
  call it inside hot loops. Call it once at startup or once per tool
  invocation, not per row.

- **Each `record_exit` triggers a real GPT-4o vision call** (1–3 cents).
  The simulator's `run_simulation.py` calls `record_exit` 20 times so
  expect $0.50–1.00 per simulator run. The OC study (`oc_study.py`) does
  NOT call any external APIs — it's pure math on synthetic data.

- **The gas anomaly detector adapts to slow drift.** Because
  `_fetch_history` pulls all visits with no time window, a cat that's
  gradually getting worse will see the detector adapt to the new normal
  and stop alarming. The trend detector is what catches that.

- **`uniform_n=4` in the eigen config was empirically calibrated** so
  that ~95% of normal waveforms are well-represented in the first 4
  principal components. Don't change it without recalibrating — see
  `simulator/eigen_sim.py`'s calibration step.

---

## Where to go next

- **`docs/USER_GUIDE.md`** — comprehensive end-user reference, useful
  even for developers who need to understand the user-facing behaviour.
- **`docs/PYTHON_API.md`** — the `LitterboxAgent` class methods
  documented one by one.
- **`docs/ML_TUTORIAL.md`** — what each detector's algorithm actually
  does, from first principles. Read this when you need to understand
  *why* a verdict came out a certain way, not just what the code does.
- **`CLAUDE.md`** — top-level architectural overview, kept current as
  the system evolves. This is the canonical "how the system fits
  together" document.
- **`simulator/oc_report.md`** — the operating-characteristic study
  (regenerated on demand). Tells you what false-positive rates and
  detection rates each detector actually delivers in synthetic tests.
