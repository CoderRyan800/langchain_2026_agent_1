# User Guide

This guide covers day-to-day use of both agents in this project.

---

## Contents

1. [Initial Setup](#1-initial-setup)
2. [Bob — General Purpose Agent](#2-bob--general-purpose-agent)
3. [Litter Box Agent — First-Time Setup](#3-litter-box-agent--first-time-setup)
4. [Litter Box Agent — Daily Workflow](#4-litter-box-agent--daily-workflow)
5. [Litter Box Agent — Querying Records](#5-litter-box-agent--querying-records)
6. [Sensor Integration (CLI)](#6-sensor-integration-cli)
7. [Python API](#7-python-api)
8. [Understanding Health Alerts](#8-understanding-health-alerts)
9. [Direct Database Access (SQLite)](#9-direct-database-access-sqlite)
10. [Data and Storage](#10-data-and-storage)
11. [Troubleshooting](#11-troubleshooting)
12. [Time-Domain Measurement System](#12-time-domain-measurement-system)
    - [12.10 Visit Analyser](#1210-step-4--visit-analyser-visit_analyserpy)
    - [12.11 Analyser Pipeline](#1211-step-5a--analyser-pipeline-analyser_pipelinepy)
    - [12.12 Eigenanalysis](#1212-step-5b--eigenanalysis-eigen_analyserpy)
    - [12.13 Cluster Analysis](#1213-step-5c--cluster-analysis-cluster_analyserpy)
    - [12.14 HTML Reports](#1214-html-reports-and-the-eigen_report-tool)
    - [12.15 Simulation](#1215-simulation)

---

## 1. Initial Setup

### API keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Both keys are required to use Bob. The litter box agent only needs `OPENAI_API_KEY`.

### Install dependencies

**Conda (recommended):**
```bash
conda create -n langchain_env_2026_1 python=3.11 -y
conda activate langchain_env_2026_1
pip install -r requirements.txt
```

**Python venv:**
```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Install the package (required for the Python API)

```bash
pip install -e .
```

The `-e` flag installs in *editable* mode so source changes in `src/` take
effect immediately without reinstalling.

### First-run note — CLIP model download

The litter box agent uses a local vision model (`clip-ViT-B-32`) for cat
identification. On the first run it downloads approximately 350 MB from
HuggingFace and caches it. Subsequent starts load the cached model in about
one second. No API key is required for CLIP — it runs entirely on your machine.

---

## 2. Bob — General Purpose Agent

Bob is a conversational assistant for web searches, answering questions, and
analysing images or audio files you upload.

### Starting Bob

```bash
python src/basic_agent.py
```

### Talking to Bob

Type anything at the `You:` prompt and press Enter. Bob remembers the full
conversation across restarts — his history is stored in `agent_memory.db`.

```
You: What's the weather like in Tokyo today?
You: Summarise the top AI news from this week
You: What did we talk about last time?
```

### Uploading files

Use `/UPLOAD` followed by an absolute path to share an image or audio file:

```
You: /UPLOAD /Users/yourname/Desktop/xray.jpg
You: What do you see in that image?

You: /UPLOAD /tmp/voice_memo.mp3
You: Can you transcribe that?
```

**Supported image formats:** `.jpg` `.jpeg` `.png` `.gif` `.webp`

**Supported audio formats:** `.mp3` `.wav` `.ogg` `.m4a` `.flac` `.opus`

> **Audio note:** Audio input requires changing `MODEL = "gpt-4o-audio-preview"`
> at the top of `src/basic_agent.py`. The default `gpt-4o` supports images only.

### Quitting

```
You: /STOP
```

### Bob and the litter box data

Bob can query the litter box database using plain English. Since
`data/litterbox.db` is a standard SQLite file, you can ask Bob to run
read-only SQL queries against it:

```
You: How many times did Whiskers use the litter box this week?
You: Show me all visits that were flagged as anomalous in the last 30 days
You: Which cat visited the most in March?
```

---

## 3. Litter Box Agent — First-Time Setup

Before the automated monitoring system can identify cats, you must register at
least one reference photo per cat.

### Starting the agent in interactive mode

```bash
python src/litterbox_agent.py
```

### Registering a cat

Use `/UPLOAD` followed by the path to a clear photo of your cat:

```
You: /UPLOAD /Users/yourname/Photos/whiskers_sofa.jpg
Assistant: I want to register this cat reference image. What is this cat's name?
You: Whiskers
Assistant: Registered reference image #1 for 'Whiskers'.
           Stored at images/cats/whiskers/001_3a2f1c.jpg
```

You can include the name in the same message if you prefer:

```
You: /UPLOAD /path/to/photo.jpg — this is my cat Marmalade
```

### Registering multiple photos per cat

More reference images improve identification accuracy, especially across
different lighting conditions and angles:

```
You: /UPLOAD /path/to/whiskers_window.jpg
Assistant: What is this cat's name?
You: Whiskers
Assistant: Registered reference image #2 for 'Whiskers'.
```

> **Tip:** Register photos taken from the angle and distance your litter box
> camera will see. Photos taken at ground level looking slightly upward give
> the best identification performance.

### Checking your registered cats

```
You: Which cats are registered?
Assistant: Registered cats:
             Marmalade: 3 reference image(s) (registered 2026-03-14)
             Whiskers: 2 reference image(s) (registered 2026-03-14)
```

---

## 4. Litter Box Agent — Daily Workflow

Once cats are registered and sensors are connected, the system runs
automatically. Day-to-day interaction mostly involves **confirming tentative
identities** and **reviewing health alerts**.

### Reviewing unconfirmed visits

```
You: Show me visits that need confirmation
Assistant: 4 unconfirmed visit(s):
             #12: ~Whiskers (sim=0.94) at 2026-03-14 07:22:11
             #11: ~Marmalade (sim=0.88) at 2026-03-14 06:55:02
             #10: Unknown at 2026-03-13 23:10:44
             #9:  ~Whiskers (sim=0.91) ⚠️ at 2026-03-13 20:15:33
```

The `~` prefix means tentative (not yet confirmed). The similarity score shows
how confident the system was. A `⚠️` means that visit has a health flag.

### Confirming an identity

```
You: Visit 12 is confirmed as Whiskers
Assistant: Visit #12 confirmed: cat is 'Whiskers'.

You: Confirm visit 11 as Marmalade
Assistant: Visit #11 confirmed: cat is 'Marmalade'.
```

### Handling an Unknown visit

When the system could not identify the cat:

```
You: Show me the images for visit 10
Assistant: Images for visit #10:
             Entry: images/visits/2026-03-13/a4b5c6d7_entry.jpg
             Exit:  images/visits/2026-03-13/a4b5c6d7_exit.jpg

You: That's Whiskers — confirm visit 10 as Whiskers
Assistant: Visit #10 confirmed: cat is 'Whiskers'.
```

Open the image files from the paths the agent returns to inspect them visually
before confirming.

---

## 5. Litter Box Agent — Querying Records

All queries work in natural language in the interactive session.

### By date

```
You: Show me all visits from today
You: What happened on 14th March?
You: Show me visits from 2026-03-13
```

### By cat

```
You: How many times did Whiskers visit this week?
You: Show me all of Marmalade's visits
You: When did Whiskers last use the litter box?
```

### Health flags

```
You: Show me all flagged health events
You: Are there any anomalous visits this month?
```

The list view shows a one-line gas-anomaly summary per row (`tier=severe,
NH₃ z=+1.08, CH₄ z=+6.11, n=32, model=per_cat`) plus a snippet of the
GPT-4o health notes — enough to triage.

### Specific visit details

When you want the full report for one visit (sensor readings, gas-anomaly
score, GPT-4o health text, image paths, all in one place):

```
You: Explain visit 115
You: Why was visit 70 flagged?
You: What were the readings on visit 42?
```

The agent uses `get_visit_details(visit_id)` for these. The reply includes
the cat's identity (with confirmation status and CLIP similarity), all
weight and gas readings, the data-driven gas anomaly block (tier, signed
z-scores, sample count, model used — `per_cat` / `pooled` / `insufficient_data`),
the GPT-4o health analysis, and image paths.

### History plots

```
You: Show me a chart of Anna's history
You: Plot the last 30 days for Whiskers
```

Generates a self-contained HTML file with three stacked time-series
sub-plots: cat weight (linear), NH₃ peak (log scale), and CH₄ peak (log
scale). Anomalous visits are red ✕ markers; normal visits are blue dots.
Hover any point for visit ID, raw reading, z-score, tier, and model.
A solid green line marks the cat's robust median; orange and red dashed
lines mark the mild (z=2σ) and significant (z=3σ) alarm thresholds —
back-projected from log space to ppb so you can see how close the data
is to alarming. The HTML opens in any browser without an internet
connection (Bokeh JavaScript is inlined).

Default window is 90 days; specify `days` in natural language to override.
Output writes to `output/cat_history_<name>.html`.

### Images

```
You: Show me the images for visit 9
```

The agent returns relative file paths — open them in any image viewer.

---

## 6. Sensor Integration (CLI)

### How sensor events are triggered

The sensor system calls the agent directly from the command line when a camera
detects motion at the litter box:

```bash
# Cat detected entering (camera only — no scale or gas sensors)
python src/litterbox_agent.py --event entry --image images/captures/entry_001.jpg

# Cat detected exiting (camera only)
python src/litterbox_agent.py --event exit --image images/captures/exit_001.jpg
```

The agent processes the image, runs the identification pipeline, writes the
visit record to the database, and exits. No human input is required.

### Passing weight and gas sensor data

If your hardware includes a weight scale and/or gas sensors, pass their
readings as additional flags. **All sensor flags are optional** — omit any
that your hardware does not support or that produced a failed reading.

**Entry event with full sensor data:**
```bash
python src/litterbox_agent.py \
    --event entry \
    --image images/captures/entry_001.jpg \
    --weight-pre   5412 \     # box + litter weight before cat entered (grams)
    --weight-entry 8634 \     # box + litter + cat weight at entry (grams)
    --ammonia-peak 38 \       # peak NH₃ reading during entry phase (ppb)
    --methane-peak 12         # peak CH₄ reading during entry phase (ppb)
```

**Exit event with full sensor data:**
```bash
python src/litterbox_agent.py \
    --event exit \
    --image images/captures/exit_001.jpg \
    --weight-exit  5489 \     # box + litter + waste weight after cat left (grams)
    --ammonia-peak 62 \       # peak NH₃ reading during/after exit (ppb)
    --methane-peak 29         # peak CH₄ reading during/after exit (ppb)
```

**Mixed — scale present, gas sensors absent:**
```bash
python src/litterbox_agent.py \
    --event entry \
    --image images/captures/entry_001.jpg \
    --weight-pre 5412 \
    --weight-entry 8634
    # --ammonia-peak and --methane-peak simply omitted
```

### Console script shorthand (after `pip install -e .`)

```bash
litterbox-agent --event entry --image /path/to/entry.jpg \
    --weight-pre 5412 --weight-entry 8634 --ammonia-peak 38

litterbox-agent --event exit  --image /path/to/exit.jpg \
    --weight-exit 5489 --ammonia-peak 62

# Custom data directories
litterbox-agent --data-dir /srv/litterbox/data \
                --images-dir /srv/litterbox/images \
                --event entry --image /path/to/entry.jpg
```

### What the system derives from sensor data

The agent computes the following values automatically:

| Derived value | Formula | Stored in |
|---|---|---|
| `cat_weight_g` | `weight_entry_g − weight_pre_g` | `visits` table |
| `waste_weight_g` | `weight_exit_g − weight_pre_g` | `visits` table |
| Peak gas (final) | `MAX(entry_reading, exit_reading)` | `visits` table |

Both the summary values and every individual raw reading are stored. The raw
readings go into `visit_sensor_events` with a `phase` tag (`pre_entry`,
`entry`, or `exit`) so you have the full time-series if you need it.

### How sensor values reach the tools

Internally, the CLI flags are serialised into a plain-English message that the
LangGraph agent reads:

```
SENSOR EVENT: A cat has entered the litter box.
Entry image path: images/captures/entry_001.jpg
Sensor readings: weight_pre_g=5412, weight_entry_g=8634, ammonia_peak_ppb=38, methane_peak_ppb=12.
```

The agent extracts the named values from that message and passes them as
parameters to `record_entry()`. This is why the system prompt explicitly
instructs the agent to pass sensor readings from the event message to the tool.

### Exit event association

When an exit event arrives, the agent automatically associates it with the
**most recent open visit** (a visit with an entry image but no exit image yet).

### Orphan exit records

If an exit event is received with no corresponding open visit, an **orphan
exit record** is created:

```
⚠️  WARNING: No open visit found. Orphan exit record #18 created — human review required.
    Exit image stored at: images/visits/2026-03-14/c3d4e5f6_exit.jpg
```

Orphan records are visible when querying visits and are flagged for review.

---

## 7. Python API

The `LitterboxAgent` class provides a fully programmatic interface to the
litter box monitoring system — no command line, no LLM overhead for
sensor-event methods.

### Installation

```bash
pip install -e .                # from the project root
```

```python
from litterbox import LitterboxAgent
```

### Constructor

```python
LitterboxAgent(
    data_dir: str | None = None,       # default: ~/.litterbox_monitor/data
    images_dir: str | None = None,     # default: ~/.litterbox_monitor/images
    openai_api_key: str | None = None, # default: reads OPENAI_API_KEY from env
)
```

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | `~/.litterbox_monitor/data` | SQLite databases and Chroma vector index |
| `images_dir` | `~/.litterbox_monitor/images` | Cat reference photos and visit captures |
| `openai_api_key` | `$OPENAI_API_KEY` | Pass an API key without touching the environment |

#### Custom paths

```python
agent = LitterboxAgent(
    data_dir="/srv/litterbox/data",
    images_dir="/srv/litterbox/images",
)
```

#### Key-in-code

```python
import os
agent = LitterboxAgent(openai_api_key=os.environ["MY_OPENAI_KEY"])
```

---

### Quick start

```python
from litterbox import LitterboxAgent

agent = LitterboxAgent()

# Register a reference photo
print(agent.register_cat("/photos/whiskers_sitting.jpg", "Whiskers"))

# Sensor-triggered entry event
print(agent.record_entry(
    "/captures/entry_001.jpg",
    weight_pre_g=5412,
    weight_entry_g=8634,
    ammonia_peak_ppb=38,
    methane_peak_ppb=12,
))

# Corresponding exit event
print(agent.record_exit(
    "/captures/exit_001.jpg",
    weight_exit_g=5489,
    ammonia_peak_ppb=62,
    methane_peak_ppb=29,
))

# Query
print(agent.get_anomalous_visits())
print(agent.list_cats())

agent.close()
```

---

### Sensor event methods

These call the underlying tools **directly** — no LLM round-trip, no GPT-4o
cost beyond the identification and health analysis steps that are always
required.

#### `record_entry`

```python
result: str = agent.record_entry(
    image_path: str,
    weight_pre_g: float | None = None,
    weight_entry_g: float | None = None,
    ammonia_peak_ppb: float | None = None,
    methane_peak_ppb: float | None = None,
)
```

Runs the two-stage CLIP + GPT-4o cat identification pipeline and writes a new
visit record.  All sensor parameters are optional.

**Returns** a plain-text summary:
```
Visit #7 opened at 2026-03-28 14:22:05 UTC.
Entry image: images/visits/2026-03-28/a1b2c3d4_entry.jpg
Tentative ID: Whiskers (similarity 0.94).
Sensors: cat weight 3222 g, NH₃ 38 ppb.
```

#### `record_exit`

```python
result: str = agent.record_exit(
    image_path: str,
    weight_exit_g: float | None = None,
    ammonia_peak_ppb: float | None = None,
    methane_peak_ppb: float | None = None,
)
```

Associates the exit with the most recent open visit, runs the GPT-4o health
analysis, and writes the result.  If no open visit exists, an orphan record
is created.

**Returns** a plain-text summary including health notes:
```
Visit #7 closed (tentative cat: Whiskers).
Exit image: images/visits/2026-03-28/a1b2c3d4_exit.jpg
Sensors: cat weight 3222 g, waste 77 g, NH₃ 62 ppb.
Health: No anomalies detected

CONCERNS_PRESENT: no
DESCRIPTION: The litter box appears normal after use...
```

---

### Cat registration

#### `register_cat`

```python
result: str = agent.register_cat(image_path: str, cat_name: str)
```

Registers a reference photo. Call multiple times to add more images for the
same cat — more images improve identification accuracy.

```python
agent.register_cat("/photos/whiskers_1.jpg", "Whiskers")
agent.register_cat("/photos/whiskers_2.jpg", "Whiskers")  # adds a second reference
agent.register_cat("/photos/mochi.jpg",      "Mochi")
```

#### `list_cats`

```python
result: str = agent.list_cats()
```

Returns a summary of all registered cats and their reference image counts.

---

### Identity management

#### `confirm_identity`

```python
result: str = agent.confirm_identity(visit_id: int, cat_name: str)
```

Permanently sets the confirmed cat identity for a visit.

```python
print(agent.get_unconfirmed_visits())   # see what needs reviewing
print(agent.confirm_identity(7, "Whiskers"))
```

#### `retroactive_recognition`

```python
result: str = agent.retroactive_recognition(cat_name: str, since_date: str)
```

Scans all unknown visits since `since_date` and re-runs the full CLIP +
GPT-4o pipeline for the specified cat. Useful after registering a new cat.

```python
agent.retroactive_recognition("Mochi", "2026-03-01")
```

---

### Query methods

All query methods hit the database directly — no LLM involved.

| Method | Parameter | Returns |
|---|---|---|
| `list_cats()` | — | All cats and reference image counts |
| `get_visits_by_date(date_str)` | `"YYYY-MM-DD"` | All visits on that date |
| `get_visits_by_cat(cat_name)` | cat name | All visits for that cat |
| `get_anomalous_visits()` | — | All visits flagged as anomalous |
| `get_unconfirmed_visits()` | — | Visits with tentative (unconfirmed) IDs |
| `get_visit_images(visit_id)` | visit number | Entry and exit image paths |

```python
print(agent.get_visits_by_date("2026-03-28"))
print(agent.get_visits_by_cat("Whiskers"))
print(agent.get_anomalous_visits())
print(agent.get_unconfirmed_visits())
print(agent.get_visit_images(7))
```

---

### Natural language queries

```python
response: str = agent.query(message: str, thread_id: str = "api")
```

Sends any plain-English message to the full LangGraph agent (all 11 tools,
conversation history preserved within `thread_id`). Uses GPT-4o for every
call — use the structured methods above when cost matters.

```python
# Simple queries
print(agent.query("How many times did Whiskers visit this week?"))
print(agent.query("Show me all anomalous visits from March 2026"))

# Multi-turn conversation (same thread_id preserves history)
agent.query("Tell me about visit 5",   thread_id="session-1")
agent.query("Confirm that visit as Mochi", thread_id="session-1")

# Natural language registration
agent.query("Register this cat. File path: /photos/mochi.jpg",
            thread_id="registration")
```

---

### Context manager

```python
with LitterboxAgent() as agent:
    agent.record_entry("/captures/entry.jpg")
    agent.record_exit("/captures/exit.jpg")
    print(agent.get_anomalous_visits())
# SQLite checkpointer closed automatically on exit
```

Alternatively call `agent.close()` explicitly. An `atexit` handler is
registered automatically as a safety net.

---

### Integration example — Raspberry Pi sensor daemon

```python
#!/usr/bin/env python3
"""litterbox_sensor_daemon.py — called by camera motion trigger."""

import sys
from litterbox import LitterboxAgent

# Your sensor hardware libraries
from scale import read_weight
from gas_sensor import read_ammonia, read_methane


def on_entry(image_path: str) -> None:
    with LitterboxAgent() as agent:
        result = agent.record_entry(
            image_path,
            weight_pre_g=read_weight(),
            weight_entry_g=read_weight(),
            ammonia_peak_ppb=read_ammonia(),   # None if sensor offline
            methane_peak_ppb=read_methane(),
        )
    print(result)


def on_exit(image_path: str) -> None:
    with LitterboxAgent() as agent:
        result = agent.record_exit(
            image_path,
            weight_exit_g=read_weight(),
            ammonia_peak_ppb=read_ammonia(),
            methane_peak_ppb=read_methane(),
        )
    print(result)


if __name__ == "__main__":
    event, image = sys.argv[1], sys.argv[2]   # "entry"/"exit", path
    if event == "entry":
        on_entry(image)
    elif event == "exit":
        on_exit(image)
```

---

## 8. Understanding Health Alerts

### Two independent checks, OR'd together

Every exit event is evaluated by **two** detectors. A visit is flagged as
anomalous (`is_anomalous = TRUE`) if **either** detector says so.

**1. GPT-4o visual analysis.** Compares the entry and exit images for visual
differences that may indicate a health concern:

- Blood in urine (pink, red, or dark discolouration of urine clumps)
- Blood in stool
- Unusual stool colour, consistency, or quantity
- Evidence of diarrhoea or mucus
- Abnormal deposits or clumping patterns
- Any other unexpected visual changes

When sensor readings are available they are included in the prompt; when the
gas detector below has flagged a tier, that tier is also passed in so the
LLM's `DESCRIPTION` and `OWNER_SUMMARY` can ground the explanation in the
statistical anomaly rather than guessing from raw numbers.

The prompt opens with explicit anti-refusal language: it tells the model
the images contain no people, no faces, and no identifiable individuals,
and that any pareidolic shapes in the litter are litter material, not
faces. This suppresses OpenAI's content classifier from refusing on
litter shadows or clumps. When a refusal still happens — or any other
unstructured response — `health_notes` stores a clean placeholder
(`"Health analysis unavailable — GPT-4o did not return a structured
response..."`) instead of the refusal text, and points the reader at
the gas-anomaly columns for the data-driven verdict.

**2. Data-driven gas anomaly detector.** Per-cat statistical check on the
NH₃ and CH₄ peak readings. The detector judges each visit relative to the
cat's *own* history — there are no fixed ppb thresholds, because absolute
gas readings depend on sensor placement, ventilation, ambient conditions,
and chip-to-chip calibration drift, and have no portable meaning.

How it works:

- Fit a robust log-Gaussian (median + MAD-based sigma) on the cat's prior
  non-null peak readings for each channel. Robust statistics so historical
  contamination by past anomalies doesn't suppress new ones (50% breakdown
  point — the detector keeps working even if up to half the cat's history
  is anomalous).
- Compute a signed z-score for the current visit's reading. Only the
  positive tail (high readings) raises an alarm — low gas is not a concern.
- Tier the result: `mild` (z ≥ 2σ), `significant` (z ≥ 3σ), `severe` (z ≥ 5σ).
- If a cat has fewer than `min_visits_per_cat` prior readings (default 10),
  fall back to a pooled distribution across all cats. If even the pool is
  too small (default minimum 30), tier is `insufficient_data` and the
  detector contributes no signal.
- All thresholds live in `src/litterbox/td_config.json` under `gas_anomaly`
  and can be tuned per deployment without code changes.

The detector's z-scores, tier, sample count, and model type (`per_cat`,
`pooled`, or `insufficient_data`) are persisted on the `visits` row for
later SQL queries and reports — see §9 for the schema.

**Caveat:** if a cat genuinely transitions to a chronic high-NH₃ state and
accumulates many readings there, future high readings will eventually look
normal to the detector. That's medically defensible (a stable chronic
state is the cat's new normal, not an acute event), but if you want to
catch slow drift, watch the trend in the `*_z_score` columns over time
rather than just the tier.

### What a flagged alert looks like

```
Health: ⚠️  ANOMALY FLAGGED — veterinary review recommended

CONCERNS_PRESENT: yes
DESCRIPTION: The exit image shows a pink-tinged clump in the lower left
that was not present in the entry image.
OWNER_SUMMARY: There appears to be discolouration consistent with blood in
the urine. This warrants prompt veterinary attention.

⚠️ This analysis is preliminary and must be reviewed by a licensed
veterinarian before any medical decisions are made.
```

### Viewing all flagged visits

```
You: Show me all anomalous visits       # interactive agent
print(agent.get_anomalous_visits())     # Python API
```

### Critical disclaimer

**This system is a monitoring aid, not a medical device.** The AI analysis is:

- **Preliminary** — it cannot replace a veterinarian's examination
- **Not always accurate** — lighting, camera angle, and litter type all affect
  image quality and therefore the reliability of the analysis
- **Not a substitute for regular vet check-ups**

If a visit is flagged as anomalous, or if you observe any change in your cat's
behaviour or litter box habits, consult a licensed veterinarian promptly.

### Rescoring historical visits

The `is_anomalous` flag and the `gas_anomaly_*` columns are written exactly
once at the moment a visit's exit is recorded. The system never recomputes
these for old visits during normal operation. So if the detector logic
changes — a new threshold, a new statistical estimator, a new prompt —
older visits keep the verdict that was true at the time they were scored.
Reports and plots that span dates can show the same physical readings with
different verdicts purely because they were processed by different detector
versions.

To bring a database to a single consistent verdict policy under the
*current* detector, run:

```bash
python -m litterbox.rescore --dry-run    # preview what would change
python -m litterbox.rescore              # apply the changes
```

Mechanics:
- For each visit with at least one non-null gas reading, the rescorer
  re-runs `score_gas_visit` against the current `visits` table (excluding
  the visit itself from the fit, same as the live detector).
- The LLM-side verdict is recovered from `health_notes`: `CONCERNS_PRESENT:
  yes` → True; refusal text or the placeholder → False.
- Final `is_anomalous = LLM_yes OR (gas_tier in {mild, significant, severe})`
  — exactly what `record_exit` computes today.
- The `gas_anomaly_rescored_at` column is set on every changed row so the
  migration is auditable.

The utility is idempotent. `--dry-run` writes nothing — it only reports
what *would* change. The `--show N` flag controls how many sample changes
are printed.

---

## 9. Direct Database Access (SQLite)

The litter box metadata is stored in a standard SQLite file. You can open it
directly with the `sqlite3` command-line shell, any SQLite GUI, or Python's
built-in `sqlite3` module — no special drivers required.

### Default database location

| Interface | Database path |
|---|---|
| CLI (`python src/litterbox_agent.py`) | `data/litterbox.db` |
| Python API (default) | `~/.litterbox_monitor/data/litterbox.db` |
| Python API (custom `data_dir`) | `<data_dir>/litterbox.db` |

---

### Opening the database from the terminal

```bash
# Standard production database (CLI / project-root usage)
sqlite3 data/litterbox.db

# Default Python API location
sqlite3 ~/.litterbox_monitor/data/litterbox.db
```

Once inside the `sqlite3` shell you will see a `sqlite>` prompt.

#### Useful shell meta-commands

```sql
.tables               -- list all tables
.schema               -- show CREATE statements for all tables
.schema visits        -- show schema for one table
.headers on           -- show column names in query output
.mode column          -- align output in columns
.mode table           -- pretty-print as a table (sqlite3 ≥ 3.33)
.width 20 20 10       -- set column widths (for .mode column)
.quit                 -- exit the shell
```

Run `.headers on` and `.mode column` first — they make output much easier
to read:

```sql
sqlite> .headers on
sqlite> .mode column
```

---

### Database schema

The database has four tables:

#### `cats` — registered cat profiles

```sql
CREATE TABLE cats (
    cat_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `cat_images` — reference photos per cat

```sql
CREATE TABLE cat_images (
    image_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    cat_id     INTEGER NOT NULL REFERENCES cats(cat_id),
    file_path  TEXT NOT NULL,      -- relative path under images_dir
    chroma_id  TEXT NOT NULL,      -- ID in the Chroma CLIP vector index
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `visits` — litter box visit records

```sql
CREATE TABLE visits (
    visit_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_time        TIMESTAMP,
    exit_time         TIMESTAMP,
    entry_image_path  TEXT,
    exit_image_path   TEXT,
    tentative_cat_id  INTEGER REFERENCES cats(cat_id),
    confirmed_cat_id  INTEGER REFERENCES cats(cat_id),
    is_confirmed      BOOLEAN DEFAULT FALSE,
    similarity_score  REAL,            -- CLIP cosine similarity (0–1)
    health_notes      TEXT,            -- full GPT-4o response
    is_anomalous      BOOLEAN DEFAULT FALSE,
    is_orphan_exit    BOOLEAN DEFAULT FALSE,
    weight_pre_g      REAL,            -- box + litter before entry
    weight_entry_g    REAL,            -- box + litter + cat at entry
    weight_exit_g     REAL,            -- box + litter + waste after exit
    cat_weight_g      REAL,            -- derived: entry − pre
    waste_weight_g    REAL,            -- derived: exit − pre
    ammonia_peak_ppb  REAL,            -- peak NH₃ (MAX of entry & exit)
    methane_peak_ppb  REAL,            -- peak CH₄ (MAX of entry & exit)
    -- Gas anomaly detector output (see §8 — Understanding Health Alerts)
    ammonia_z_score        REAL,       -- signed z-score vs cat's history
    methane_z_score        REAL,
    gas_anomaly_tier       TEXT,       -- normal | mild | significant | severe | insufficient_data
    gas_anomaly_n_samples  INTEGER,    -- size of the historical fit
    gas_anomaly_model_used TEXT,       -- per_cat | pooled | insufficient_data
    gas_anomaly_rescored_at TIMESTAMP, -- set by python -m litterbox.rescore (NULL otherwise)
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `visit_sensor_events` — raw per-phase sensor log

```sql
CREATE TABLE visit_sensor_events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    visit_id      INTEGER NOT NULL REFERENCES visits(visit_id),
    recorded_at   TEXT NOT NULL,
    phase         TEXT,          -- 'pre_entry', 'entry', or 'exit'
    sensor_type   TEXT NOT NULL, -- 'weight', 'ammonia', or 'methane'
    value_numeric REAL,
    value_text    TEXT,
    unit          TEXT           -- 'g' or 'ppb'
);
```

---

### Useful queries

Paste these directly into the `sqlite3` shell (remember `.headers on` and
`.mode column` first).

#### List all registered cats

```sql
SELECT cat_id, name, created_at FROM cats ORDER BY name;
```

#### Count reference images per cat

```sql
SELECT c.name, COUNT(ci.image_id) AS ref_images
FROM cats c
LEFT JOIN cat_images ci ON c.cat_id = ci.cat_id
GROUP BY c.cat_id
ORDER BY c.name;
```

#### Recent visits (last 20)

```sql
SELECT v.visit_id,
       c.name        AS cat,
       v.entry_time,
       v.exit_time,
       v.is_confirmed,
       v.similarity_score,
       v.is_anomalous
FROM visits v
LEFT JOIN cats c ON coalesce(v.confirmed_cat_id, v.tentative_cat_id) = c.cat_id
ORDER BY v.visit_id DESC
LIMIT 20;
```

#### All anomalous visits

```sql
SELECT v.visit_id,
       c.name AS cat,
       v.entry_time,
       v.ammonia_peak_ppb,
       v.methane_peak_ppb,
       v.health_notes
FROM visits v
LEFT JOIN cats c ON coalesce(v.confirmed_cat_id, v.tentative_cat_id) = c.cat_id
WHERE v.is_anomalous = 1
ORDER BY v.entry_time DESC;
```

#### Unconfirmed visits

```sql
SELECT v.visit_id,
       c.name          AS tentative_cat,
       v.similarity_score,
       v.entry_time,
       v.is_anomalous
FROM visits v
LEFT JOIN cats c ON v.tentative_cat_id = c.cat_id
WHERE v.is_confirmed = 0
ORDER BY v.visit_id DESC;
```

#### Sensor readings for a specific visit

```sql
-- Replace 7 with the visit_id you want
SELECT phase, sensor_type, value_numeric, unit, recorded_at
FROM visit_sensor_events
WHERE visit_id = 7
ORDER BY event_id;
```

#### Weight and gas summary per cat

```sql
SELECT c.name,
       COUNT(*)                    AS visits,
       ROUND(AVG(v.cat_weight_g))  AS avg_cat_weight_g,
       ROUND(AVG(v.waste_weight_g))AS avg_waste_g,
       ROUND(AVG(v.ammonia_peak_ppb), 1) AS avg_nh3_ppb,
       ROUND(AVG(v.methane_peak_ppb), 1) AS avg_ch4_ppb
FROM visits v
JOIN cats c ON coalesce(v.confirmed_cat_id, v.tentative_cat_id) = c.cat_id
WHERE v.cat_weight_g IS NOT NULL
GROUP BY c.cat_id
ORDER BY c.name;
```

#### Visits on a specific date

```sql
SELECT v.visit_id, c.name AS cat, v.entry_time, v.exit_time, v.is_anomalous
FROM visits v
LEFT JOIN cats c ON coalesce(v.confirmed_cat_id, v.tentative_cat_id) = c.cat_id
WHERE date(v.entry_time) = '2026-03-28'
ORDER BY v.entry_time;
```

#### Full health notes for one visit

```sql
-- Replace 9 with the visit_id you want to inspect
SELECT visit_id, entry_time, health_notes FROM visits WHERE visit_id = 9;
```

---

### Reading the database from Python

You can also read the database directly from a Python script without using the
agent at all:

```python
import sqlite3
from pathlib import Path

# Adjust path to match your setup
db = Path.home() / ".litterbox_monitor" / "data" / "litterbox.db"
# Or for the project-root CLI setup:
# db = Path("data/litterbox.db")

conn = sqlite3.connect(str(db))
conn.row_factory = sqlite3.Row   # access columns by name

cursor = conn.execute("""
    SELECT v.visit_id,
           c.name        AS cat,
           v.entry_time,
           v.is_anomalous,
           v.cat_weight_g,
           v.ammonia_peak_ppb
    FROM visits v
    LEFT JOIN cats c
          ON coalesce(v.confirmed_cat_id, v.tentative_cat_id) = c.cat_id
    ORDER BY v.visit_id DESC
    LIMIT 10
""")

for row in cursor:
    print(dict(row))

conn.close()
```

---

### Recommended SQLite GUI tools

If you prefer a graphical interface:

| Tool | Platform | Notes |
|---|---|---|
| [DB Browser for SQLite](https://sqlitebrowser.org/) | macOS / Windows / Linux | Free, open-source, most popular |
| [TablePlus](https://tableplus.com/) | macOS / Windows / Linux | Polished UI; free tier available |
| [DBeaver](https://dbeaver.io/) | macOS / Windows / Linux | Full-featured; free community edition |
| VS Code SQLite extension | macOS / Windows / Linux | Open `.db` files directly in the editor |

Open the `litterbox.db` file with any of these tools to browse tables, run
queries, and export data without touching the command line.

---

## 10. Data and Storage

### Where everything lives

#### CLI / project-root usage

| What | Where | Gitignored |
|---|---|---|
| Cat reference photos | `images/cats/<catname>/` | Yes |
| Visit entry/exit images | `images/visits/YYYY-MM-DD/` | Yes |
| Sensor capture drop zone | `images/captures/` | Yes |
| Cat and visit metadata | `data/litterbox.db` | Yes |
| CLIP vector index | `data/chroma/` | Yes |
| Litter box agent conversation history | `data/agent_litterbox_memory.db` | Yes |
| Bob's conversation history | `agent_memory.db` | Yes |

#### Python API (default paths)

| What | Where |
|---|---|
| Cat reference photos | `~/.litterbox_monitor/images/cats/<catname>/` |
| Visit entry/exit images | `~/.litterbox_monitor/images/visits/YYYY-MM-DD/` |
| Cat and visit metadata | `~/.litterbox_monitor/data/litterbox.db` |
| CLIP vector index | `~/.litterbox_monitor/data/chroma/` |
| Agent conversation history | `~/.litterbox_monitor/data/agent_memory.db` |

Override any of these by passing `data_dir` and/or `images_dir` to the
constructor.

### Backing up your data

```bash
# CLI setup
tar -czf litterbox_backup_$(date +%F).tar.gz data/ images/

# Python API default location
tar -czf litterbox_backup_$(date +%F).tar.gz ~/.litterbox_monitor/
```

### Starting fresh

```bash
# CLI setup
rm -rf data/ images/

# Python API default location
rm -rf ~/.litterbox_monitor/

# Bob's conversation history (separate from litter box data)
rm agent_memory.db
```

### Starting a new conversation thread

All history is keyed by `thread_id`. To start a fresh conversation without
deleting the database, increment the thread ID:

```python
# In src/basic_agent.py or src/litterbox_agent.py
config = {"configurable": {"thread_id": "2"}}

# In Python API natural language queries
agent.query("Hello", thread_id="session-2")
```

---

## 11. Troubleshooting

### "No module named 'langgraph.checkpoint.sqlite'"

```bash
pip install langgraph-checkpoint-sqlite
```

### CLIP model warnings on startup

```
Key vision_model.embeddings.position_ids | UNEXPECTED
```

These are harmless. The `sentence-transformers` library reports unused keys
when loading a CLIP model from the HuggingFace cache. They do not affect
embedding quality or identification accuracy.

### GPT-4o refuses to analyse images

The health analysis prompt includes veterinary terminology that can occasionally
trigger OpenAI's content safety filter, particularly when images do not clearly
resemble a litter box (e.g. test images, very dark captures, extreme
close-ups). In production this is rare with genuine camera footage. If it
occurs regularly, check that your camera is correctly framed to show the full
litter box interior.

### Cat identified as "Unknown"

This means either:
1. The cat's photo was not registered — run `/UPLOAD` with a clear reference
   photo, or call `agent.register_cat()` via the Python API
2. The similarity score was below the 0.82 threshold — add more reference
   images taken from the camera's viewing angle
3. GPT-4o did not confirm the CLIP match — this can happen in poor lighting.
   Improve camera lighting or add reference images taken in similar conditions.

### Sensor data not appearing in the database

If `cat_weight_g`, `waste_weight_g`, or gas readings are NULL in the database
after a CLI run:

1. **Check flag spelling** — CLI flags use hyphens: `--weight-pre`,
   `--weight-entry`, `--weight-exit`, `--ammonia-peak`, `--methane-peak`.
   Underscores are silently ignored by argparse.
2. **Check which event gets which flags** — `--weight-pre` and
   `--weight-entry` belong on the **entry** event; `--weight-exit` belongs on
   the **exit** event.
3. **Confirm the agent received the values** — run a single event manually and
   check the printed output. The tool result will include `cat weight X g` if
   the weight was received correctly.
4. **NULL is valid** — if a sensor malfunctions, simply omit its flag. The
   system records NULL and continues; derived values that depend on a missing
   reading will also be NULL.

For the Python API, pass `None` explicitly or omit the keyword argument:

```python
agent.record_entry("/captures/entry.jpg",
                   weight_pre_g=5412,
                   weight_entry_g=8634)
                   # ammonia_peak_ppb and methane_peak_ppb default to None
```

### Orphan exit records accumulating

This indicates the entry sensor is misfiring or the agent was not running when
the cat entered. Check:

- That the entry sensor has a reliable power supply and network connection
- That the agent process is running continuously (consider a `systemd` service
  or `launchd` plist on macOS)
- The camera trigger threshold — if it is too sensitive it may miss slow-moving
  entries

### Agent loses conversation history

History is tied to `thread_id`. If the thread ID in the config was changed or
the database was deleted, prior history will not be available. Check that
`data/agent_litterbox_memory.db` (or `~/.litterbox_monitor/data/agent_memory.db`
for the Python API) exists and is non-empty.

### Python API: changes not taking effect

If you installed with `pip install -e .` (editable mode), changes to source
files in `src/` take effect immediately. If you installed without `-e`,
reinstall after making changes:

```bash
pip install -e .
```

### Inspecting the database after an unexpected result

The fastest way to confirm what was actually written to the database is the
`sqlite3` shell:

```bash
sqlite3 data/litterbox.db
sqlite> .headers on
sqlite> .mode column
sqlite> SELECT visit_id, entry_time, cat_weight_g, waste_weight_g,
   ...>        ammonia_peak_ppb, is_anomalous
   ...> FROM visits ORDER BY visit_id DESC LIMIT 5;
```

This lets you verify sensor values, health flags, and cat IDs without going
through the agent layer.

---

## 12. Time-Domain Measurement System

> **Branch:** `feature/time_domain_measurements`
>
> This section describes the continuous time-series layer that supplements the
> per-visit snapshot system described in earlier sections.  It is implemented
> in three steps and uses no LLM calls during normal monitoring — all real-time
> processing is local.

### 12.1 Overview

The per-visit system (Sections 3–9) captures a single weight reading and a
single image at entry and exit.  The time-domain system adds **continuous
sampling**: it maintains a rolling 10-minute buffer of every sensor channel at
a 5-second interval, detects visit boundaries automatically from the data
stream, and fires a callback when a complete visit is detected.

```
Physical sensors / camera
        │
        ▼
  SensorCollector          ← one sample every 5 s
        │ buffer.append()
        ▼
   RollingBuffer            ← circular, 120 samples × 10 minutes
        │ buffer.snapshot() (on visit complete)
        ▼
   VisitTrigger             ← state machine: KITTY_ABSENT ↔ KITTY_PRESENT
        │ on_visit_complete callback
        ▼
  (Step 4 — VisitAnalyser)  ← cat ID from the full time-series window
```

### 12.2 Configuration — `td_config.json`

All time-domain settings live in one file:

```
src/litterbox/td_config.json
```

```jsonc
{
  "window_minutes":     10,   // rolling buffer length
  "samples_per_minute": 12,   // one sample every 5 seconds

  "channels": [
    { "name": "weight_g",    "type": "weight",     "enabled": true  },
    { "name": "ammonia_ppb", "type": "ammonia",    "enabled": true  },
    { "name": "methane_ppb", "type": "methane",    "enabled": true  },
    { "name": "chip_id",     "type": "chip_id",    "enabled": true  },
    { "name": "similarity",  "type": "similarity", "enabled": true  }
  ],

  "trigger": {
    "weight_entry_delta_g":            300,  // g above baseline to declare entry
    "weight_exit_delta_g":             200,  // g above baseline to declare exit
    "chip_absent_consecutive":           3,  // null chip readings in a row → exit
    "similarity_entry_threshold":      0.70, // CLIP score to declare entry
    "similarity_exit_threshold":       0.50, // score must fall below this → exit
    "similarity_sustained_peak_samples": 3   // reserved for Step 4 analysis
  },

  "image_retention_days": 7
}
```

**Disabling sensors:** set `"enabled": false` for any channel whose hardware
is not installed.  The collector skips that channel entirely; the rest of the
system adapts automatically.

**Tuning thresholds:** edit the `trigger` block without touching any code.
Increase `weight_entry_delta_g` if sensor noise causes false entries; decrease
`similarity_entry_threshold` if the camera angle produces lower scores.

---

### 12.3 Step 1 — Rolling Buffer (`time_buffer.py`)

The `RollingBuffer` is the data backbone of the entire system.  It is a
thread-safe circular buffer that holds time-stamped measurement dictionaries.

```python
from litterbox.time_buffer import RollingBuffer, load_td_config
from datetime import datetime, timezone

cfg = load_td_config()          # reads td_config.json
buf = RollingBuffer(
    window_minutes     = cfg["window_minutes"],
    samples_per_minute = cfg["samples_per_minute"],
)
# capacity = 10 × 12 = 120 samples (10 minutes at one every 5 s)

# Append a measurement
buf.append(
    datetime.now(timezone.utc),
    {"weight_g": 5412.3, "ammonia_ppb": 9.1, "chip_id": None}
)

# Read back
print(buf.get_channel("weight_g"))     # [5412.3]
print(buf.get_channel("chip_id"))      # [None]
print(buf.window_span_seconds())       # 0.0  (only one entry)
print(len(buf))                        # 1
print(buf.is_full())                   # False
```

**Key behaviours:**

- When the buffer reaches capacity (120 entries), the oldest entry is
  automatically discarded on each new `append()`.
- All public methods are protected by a `threading.Lock`.  A background
  sampling thread and the main thread can read/write concurrently.
- Missing keys in a values dict are stored as absent (not zero).  Calling
  `get_channel("ammonia_ppb")` when some entries lacked that key returns
  `None` for those positions.

**Converting to a pandas DataFrame:**

```python
# All channels
df_all = buf.to_dataframe()

# Similarity channels only — prefix is stripped from column names
df_sim = buf.to_dataframe(channel_prefix="similarity_")
# columns: ["anna", "luna", "marina", ...]
# index:   datetime timestamps
# values:  CLIP scores, NaN where the camera had no frame

# Time-slice to a visit window
df_visit = buf.to_dataframe(
    channel_prefix = "similarity_",
    start          = entry_time,
    end            = exit_time,
)
```

**Why NaN, not 0?**  A missing camera frame is not a score of zero.  If the
middle frame of a 3-frame visit is missing and stored as 0, the mean drops
from 0.90 to 0.60 — below the identification threshold, causing a false
"Unknown" result.  NaN with `skipna=True` gives the correct 0.90.

---

### 12.4 Step 2 — Sensor Collector (`sensor_collector.py`)

The `SensorCollector` drives the buffer from actual (or mock) hardware.

#### Driver interface

Every sensor is accessed through a `BaseDriver` subclass that implements one
method:

```python
class BaseDriver(ABC):
    def read(self) -> float | str | dict | None:
        """Return the current reading, or None if unavailable."""
```

| Driver class | Channel type | Returns |
|---|---|---|
| `WeightDriver` | `weight` | `float` (grams) or `None` |
| `AmmoniaDriver` | `ammonia` | `float` (ppb) or `None` |
| `MethaneDriver` | `methane` | `float` (ppb) or `None` |
| `ChipIdDriver` | `chip_id` | `str` (cat name) or `None` |
| `SimilarityDriver` | `similarity` | `dict[cat_name, score]` or `None` |

All five classes act as **mocks** in tests and simulation.  In production,
subclass the appropriate driver and override `read()` with real hardware I/O.

#### Constructing and running the collector

```python
from litterbox.sensor_collector import (
    SensorCollector, WeightDriver, AmmoniaDriver,
    MethaneDriver, ChipIdDriver, SimilarityDriver,
)

drivers = {
    "weight":     WeightDriver(base_value=5400.0, noise_sigma=20.0),
    "ammonia":    AmmoniaDriver(base_value=8.0,   noise_sigma=1.0),
    "methane":    MethaneDriver(base_value=5.0,   noise_sigma=0.8),
    "chip_id":    ChipIdDriver(cat_name=None),       # no chip reader
    "similarity": SimilarityDriver(cat_scores=None), # no camera yet
}

collector = SensorCollector(
    config   = cfg,
    drivers  = drivers,
    buffer   = buf,
    on_sample = None,   # wire VisitTrigger.check here (see §12.5)
)

collector.start()   # launches daemon thread
# ... runs in background at 5-second intervals ...
collector.stop()    # clean shutdown; wait for thread to exit
```

**Similarity expansion:** when the `SimilarityDriver` returns
`{"anna": 0.18, "luna": 0.85}`, the collector writes
`{"similarity_anna": 0.18, "similarity_luna": 0.85}` into the buffer — one
key per cat, with the prefix added automatically.

**Missing frames:** if the camera cannot produce a usable frame this tick,
`SimilarityDriver` returns `None`.  The collector writes nothing for that
tick's similarity keys.  The buffer entry simply lacks them, which the
DataFrame converts to `NaN`.

**CLIP-only on the continuous stream:** the `SimilarityDriver` runs the local
CLIP model on every tick.  GPT-4o is never called during continuous monitoring
— it is reserved for post-visit confirmation in Step 4.  This keeps the
per-tick cost at zero beyond local CPU/GPU inference.

---

### 12.5 Step 3 — Visit Trigger (`visit_trigger.py`)

The `VisitTrigger` watches each tick's values dict and transitions between
two states:

```
KITTY_ABSENT  ──── first-stage trigger ───► KITTY_PRESENT
                                                  │
              ◄─── second-stage trigger ──────────┘
                   (fires on_visit_complete)
```

#### Entry triggers (ABSENT → PRESENT)

Any **one** of the following fires the transition:

| Condition | Default threshold |
|-----------|-------------------|
| `weight_g > baseline + weight_entry_delta_g` | +300 g above rolling median |
| `chip_id` is non-null | any chip detected |
| Any `similarity_<cat>` > `similarity_entry_threshold` | 0.70 |

The baseline weight is the **median** of all weight readings currently in the
buffer.  Median is used instead of mean to be robust against outliers.

#### Exit triggers (PRESENT → ABSENT)

Any **one** of the following fires the callback:

| Condition | Default threshold |
|-----------|-------------------|
| `weight_g < baseline + weight_exit_delta_g` | +200 g (100 g hysteresis below entry) |
| `chip_id` has been `None` for N consecutive ticks | N = 3 |
| All `similarity_<cat>` have been < `similarity_exit_threshold` for N consecutive ticks | 0.50, N = 3 |

The 100 g hysteresis band between the entry (300 g) and exit (200 g) thresholds
prevents rapid oscillation around the boundary from producing spurious entry/exit pairs.

#### Wiring trigger to collector

```python
from litterbox.visit_trigger import VisitTrigger, KITTY_ABSENT, KITTY_PRESENT

visits_completed = []

def handle_visit(snapshot, entry_time, exit_time):
    """Called once when a complete visit is detected."""
    print(f"Visit: {entry_time} → {exit_time}  ({len(snapshot)} buffer samples)")
    visits_completed.append((snapshot, entry_time, exit_time))

trigger = VisitTrigger(
    config             = cfg,
    buffer             = buf,
    on_visit_complete  = handle_visit,
)

# Wire trigger into collector: check() is called after every sample tick
collector = SensorCollector(
    config    = cfg,
    drivers   = drivers,
    buffer    = buf,
    on_sample = trigger.check,   # ← the integration point
)

collector.start()
# trigger.check() is now called automatically on every 5-second tick.
# When the state machine detects a complete visit, handle_visit() fires.
```

#### Checking state

```python
print(trigger.state)   # "kitty_absent" or "kitty_present"
```

#### Resetting mid-visit

```python
trigger.reset()
# Forces state back to KITTY_ABSENT without firing the callback.
# Useful in the simulator and for recovery from sensor glitches.
```

---

### 12.6 Complete wiring example

```python
from litterbox.time_buffer import RollingBuffer, load_td_config
from litterbox.sensor_collector import (
    SensorCollector, WeightDriver, AmmoniaDriver,
    MethaneDriver, ChipIdDriver, SimilarityDriver,
)
from litterbox.visit_trigger import VisitTrigger

# ── 1. Load configuration ──────────────────────────────────────────────
cfg = load_td_config()   # reads src/litterbox/td_config.json

# ── 2. Create the rolling buffer ───────────────────────────────────────
buf = RollingBuffer(
    window_minutes     = cfg["window_minutes"],
    samples_per_minute = cfg["samples_per_minute"],
)

# ── 3. Define hardware drivers (swap for real hardware subclasses) ─────
drivers = {
    "weight":     WeightDriver(base_value=5400.0, noise_sigma=25.0),
    "ammonia":    AmmoniaDriver(base_value=8.0,   noise_sigma=1.2),
    "methane":    MethaneDriver(base_value=5.0,   noise_sigma=0.9),
    "chip_id":    ChipIdDriver(cat_name=None),
    "similarity": SimilarityDriver(cat_scores=None),
}

# ── 4. Define the visit-complete handler ───────────────────────────────
def on_visit_complete(snapshot, entry_time, exit_time):
    duration = (exit_time - entry_time).total_seconds()
    print(f"Visit detected: {entry_time.isoformat()} → {exit_time.isoformat()}")
    print(f"  Duration: {duration:.0f} s")
    print(f"  Buffer samples in snapshot: {len(snapshot)}")
    # Step 4 (VisitAnalyser) will process snapshot here.

# ── 5. Create trigger and collector ────────────────────────────────────
trigger = VisitTrigger(
    config            = cfg,
    buffer            = buf,
    on_visit_complete = on_visit_complete,
)

collector = SensorCollector(
    config    = cfg,
    drivers   = drivers,
    buffer    = buf,
    on_sample = trigger.check,
)

# ── 6. Start sampling ──────────────────────────────────────────────────
collector.start()
print("Monitoring started.  Press Ctrl-C to stop.")

try:
    import time
    while True:
        time.sleep(10)
        print(f"  Buffer: {len(buf)} samples  State: {trigger.state}")
except KeyboardInterrupt:
    pass
finally:
    collector.stop()
    print("Monitoring stopped.")
```

---

### 12.7 Plotting time-domain data

The `td_plot` module provides a swappable plotting interface backed by Bokeh.

```python
from litterbox.td_plot import get_plot_backend
from pathlib import Path

backend = get_plot_backend("bokeh")   # only Bokeh is implemented so far

# Plot scalar channels
backend.plot_channels(
    timestamps  = buf.get_timestamps(),
    channels    = {
        "weight_g":    buf.get_channel("weight_g"),
        "ammonia_ppb": buf.get_channel("ammonia_ppb"),
        "methane_ppb": buf.get_channel("methane_ppb"),
    },
    title       = "Live sensor feed",
    output_path = Path("output/channels.html"),
)

# Plot similarity scores for all registered cats
df_sim = buf.to_dataframe(channel_prefix="similarity_")
backend.plot_similarity_dataframe(
    df          = df_sim,
    title       = "Cat similarity scores",
    output_path = Path("output/similarity.html"),
    threshold   = cfg["trigger"]["similarity_entry_threshold"],
)
```

Both methods produce a **self-contained HTML file** (Bokeh JS inlined) that
can be opened in any browser without an internet connection.

Pass `output_path=None` to open the plot in the default browser instead of
saving.

**Swapping the backend:** create `td_plot_plotly.py` (or any other name) with
a class `Backend` that inherits from `PlotBackend` and implements
`plot_channels()` and `plot_similarity_dataframe()`.  Change the
`get_plot_backend("bokeh")` call to `get_plot_backend("plotly")`.  No other
file changes are required.

---

### 12.8 Step status

| Step | Module | Status | Tests |
|------|--------|--------|-------|
| 1 | `time_buffer.py` — `RollingBuffer` + `load_td_config()` | **COMPLETE** | 42/42 |
| 2 | `sensor_collector.py` — `SensorCollector` + driver interface | **COMPLETE** | 38/38 |
| 3 | `visit_trigger.py` — `VisitTrigger` state machine | **COMPLETE** | 34/34 |
| 4 | `visit_analyser.py` — cat ID from time-series, DB storage | **COMPLETE** | 23/23 |
| 5a | `analyser_pipeline.py` — plugin framework + resampling | **COMPLETE** | 20/20 |
| 5b | `eigen_analyser.py` — eigendecomposition anomaly detection | **COMPLETE** | 35/35 |
| 5c | `cluster_analyser.py` — GMM+BIC cluster analysis | **COMPLETE** | 15/15 |
| — | `eigen_query.py` — query functions + HTML reports | **COMPLETE** | 15/15 |

---

### 12.9 Troubleshooting the time-domain system

**"Module not found: bokeh" or "Module not found: pandas"**

```bash
pip install "bokeh>=3.0.0" "pandas>=2.0.0"
```

Both are in `requirements.txt`; run `pip install -r requirements.txt` to
install everything at once.

**Trigger fires immediately on startup**

The trigger computes the baseline weight from whatever is already in the buffer.
If the buffer is empty when the first samples arrive, the baseline is computed
from the first reading — which may be elevated if the box is not empty.
Let the system run for at least one minute (12 samples) before the first visit
so the buffer builds up a representative baseline.

**Trigger never fires (weight path)**

Check that:
1. The weight driver is returning readings above zero.
2. The baseline is stable (inspect via `buf.get_channel("weight_g")`).
3. `weight_entry_delta_g` in `td_config.json` is not set higher than the
   cat's weight.  For a 4 kg cat the default 300 g threshold is fine; an
   entry delta of 5 000 g would never trigger.

**Trigger never exits (similarity path)**

The similarity exit requires **all** registered cats to have scores below
`similarity_exit_threshold` for `chip_absent_consecutive` (default 3)
consecutive ticks.  If even one cat's score stays above 0.50, the counter
resets.  Check:
- That the camera is not picking up a reflection or static object scoring
  above the threshold.
- That `similarity_exit_threshold` is above 0.0 (default 0.50 is correct
  for most setups).

**DataFrame `to_dataframe()` returns an empty DataFrame**

The buffer is empty.  Start the collector, wait for at least one tick
(5 seconds with the default config), and then call `to_dataframe()`.

**Bokeh plot shows no data points (gaps only)**

All values in the channel are `None`.  This means either the driver returned
`None` every tick (sensor unavailable) or the channel was disabled in
`td_config.json`.  Check `buf.get_channel("weight_g")` to confirm data
is being written.

---

### 12.10 Step 4 — Visit Analyser (`visit_analyser.py`)

When the `VisitTrigger` fires `on_visit_complete`, the `VisitAnalyser`
identifies which cat was in the box and persists the visit to the `td_visits`
table.

#### Cat identification priority

1. **Chip ID** — if any sample in the visit window has a non-null `chip_id`,
   the most frequent value wins.  The visit is marked `is_confirmed = True`.
2. **Similarity DataFrame** — the analyser builds a per-cat DataFrame from
   `similarity_*` channels, computes column means (`skipna=True`), and checks
   that the winning cat exceeds `similarity_entry_threshold` for at least
   `similarity_sustained_peak_samples` (P) consecutive non-NaN samples.
   Missing frames (NaN) are skipped — they neither count toward nor reset
   the consecutive run.  The visit is marked as a tentative (unconfirmed) ID.
3. **Unknown** — if neither method produces a result, `id_method = "unknown"`.

#### Image retention

The `sweep_old_visit_images()` utility in `image_retention.py` deletes visit
image directories older than `image_retention_days` (default 7).  It takes the
image base path as an explicit parameter so tests can run without monkeypatching.

---

### 12.11 Step 5a — Analyser Pipeline (`analyser_pipeline.py`)

The `AnalyserPipeline` is the plugin framework that runs one or more
time-domain analysers on each completed visit.

```
VisitAnalyser.save()  →  AnalyserPipeline.run()
                              │
                      ┌───────┼──────────┐
                      ▼       ▼          ▼
                  EigenPlugin  ClusterPlugin  (future plugins)
```

#### Key components

- **`BaseAnalyser`** — abstract base class.  Plugins implement `name` and
  `analyse(waveform, visit_record, channel) → AnalysisResult`.
- **`AnalysisResult`** — dataclass with `plugin_name`, `anomaly_score` (0–1),
  `anomaly_level` (normal/mild/significant/major), and a `details` dict.
- **`resample_to_length(raw, target_length=64)`** — NaN-gap-filling + linear
  interpolation utility.  The pipeline resamples every visit's weight channel
  to L=64 samples before passing to plugins.
- **Fault isolation** — if a plugin raises an exception, the pipeline logs it,
  records an error result, and continues to the next plugin.

#### Plugin contract

Each plugin receives a **resampled** waveform of length L (default 64).  The
waveform is **not** mean-subtracted — each plugin owns its own preprocessing.
This allows different plugins to use different normalisation strategies.

---

### 12.12 Step 5b — Eigenanalysis (`eigen_analyser.py`)

This plugin decomposes each visit's weight waveform into an eigenvector basis
derived from historical visits.  It implements **Layer 1** (out-of-subspace)
anomaly detection.

#### How it works

Each visit's weight waveform is a curve with a characteristic shape:

```
[baseline] → [ramp up] → [plateau] → [ramp down] → [baseline + waste]
```

The key insight: this shape isn't random.  If you collect hundreds of
waveforms for a cat and compute the autocovariance matrix, a small number of
eigenvectors (N) will explain nearly all the variance.  A typical visit's
waveform is a linear combination of these N eigenvectors — the rest is noise.

#### Pipeline per visit

1. **Mean-subtract** the resampled L=64 waveform.  The DC term (mean weight)
   is stored separately — it's useful metadata but would dominate the
   covariance matrix if left in.

2. **Store** the zero-mean waveform and DC term in `eigen_waveforms`.

3. **Select model** — per-cat if the cat has ≥ L/2 (32) stored waveforms,
   pooled (all cats combined) if there are ≥ 2L (128) total, or skip if
   neither threshold is met.

4. **Compute the autocovariance matrix** C = XᵀX / (K-1) from all relevant
   stored waveforms.  Apply Tikhonov regularization (C + αI) when K < L to
   handle rank deficiency.

5. **Eigendecompose** via `numpy.linalg.eigh` (exploits symmetry, returns
   real eigenvalues).  Sort by descending eigenvalue.

6. **Select N** — the smallest number of eigenvectors whose cumulative
   variance exceeds 95%.  When `uniform_n` is set in config (default: 4),
   all models use that fixed N instead, producing fixed-length coefficient
   vectors suitable for ML.

7. **Project and score** — compute expansion coefficients c = Vᵀx (full
   L-vector) and reconstruction x̂ = V_N c[:N].  Explained variance:
   `EV = 1 - ‖x - x̂‖² / ‖x‖²`.

8. **Classify** by EV threshold:

   | EV range | Level | Meaning |
   |----------|-------|---------|
   | ≥ 0.90 | normal | Waveform well-explained by historical patterns |
   | 0.70 – 0.90 | mild | Unusual shape — review recommended |
   | 0.40 – 0.70 | significant | Major shape anomaly — alert owner |
   | < 0.40 | major | Extreme anomaly — likely sensor failure or acute issue |

#### What to look for in eigenanalysis results

- **EV consistently > 0.95:** Normal.  The cat's visit patterns are stable.
- **EV drops to 0.85–0.90:** Mild.  Could be a posture change, a different
  entry/exit pattern, or the cat fidgeting more than usual.  Worth noting
  but not alarming.
- **EV drops below 0.70:** Something is genuinely different.  Either the
  sensor is malfunctioning (check the waveform plot — does it look like
  noise?) or the cat's behavior has changed meaningfully (straining,
  repeated repositioning, prolonged visit).
- **EV below 0.40:** Almost certainly a sensor failure (check raw data) or
  an extreme behavioral anomaly.  Escalate immediately.
- **DC term trending:** The DC term (mean weight) is tracked separately.
  A gradual increase over weeks could indicate weight gain; a sudden change
  might indicate fluid retention.  This is independent of the shape analysis.

#### Uniform N and expansion coefficients

When `uniform_n` is set (calibrated to 4 for this system), every visit
produces a fixed-length coefficient vector [c₁, c₂, c₃, c₄].  These
represent the "fingerprint" of the visit's waveform shape in the eigenbasis.

The `calibrate_uniform_n()` method finds the smallest N where ≥ 95% of stored
waveforms achieve ≥ 95% explained variance.  Run it once after accumulating
enough data (≥ 128 visits):

```python
from litterbox.eigen_analyser import EigenAnalyser
from litterbox.time_buffer import load_td_config

config = load_td_config()
eigen = EigenAnalyser(config)
result = eigen.calibrate_uniform_n("weight_g")
print(f"N = {result['uniform_n']}, coverage = {result['actual_coverage']:.1%}")
# Then set "uniform_n": 4 in td_config.json
```

---

### 12.13 Step 5c — Cluster Analysis (`cluster_analyser.py`)

This plugin implements **Layer 2** (in-subspace) anomaly detection via
Gaussian Mixture Model (GMM) clustering on the N-dimensional expansion
coefficients.

#### Why two layers?

Layer 1 (EV) catches visits whose waveform shape can't be represented at all
by the learned eigenvectors — the signal has energy in unknown directions.

But a waveform can score high EV and still be anomalous.  If a cat normally
produces coefficient vectors near [14, -3, 1, 0.5] and a visit produces
[14, -12, 8, -5], the EV may be 0.97 (the energy is in the right directions)
but the *combination* is unlike anything the cat has done before.  Layer 2
catches this.

#### How it works

1. **Collect** all N-dimensional coefficient vectors for a cat.
2. **Fit GMMs** for k = 1, 2, ..., k_max components with full covariance.
3. **Select k\*** via BIC (Bayesian Information Criterion) — the model that
   best balances fit and complexity.  BIC naturally prefers k=1 for unimodal
   data and only adds clusters when the data justifies it.
4. **Score** the new visit by computing its log-likelihood under the fitted
   GMM.
5. **Z-score** the log-likelihood against the distribution of training
   log-likelihoods: `z = (ll - mean_ll) / std_ll`.

#### What to look for in cluster results

- **z-score near 0:** Normal — the visit's coefficients are in a
  typical region of the cat's coefficient space.
- **z-score < -2 (mild):** The coefficient vector is in a low-probability
  region.  The visit's waveform shape, while representable by the eigenbasis,
  is an unusual combination for this cat.
- **z-score < -3 (significant):** Statistically very unlikely under the
  learned distribution.  Warrants investigation.
- **z-score < -4 (major):** Essentially zero probability under the model.
  Either a sensor issue or a genuinely unprecedented behavioral pattern.
- **k_clusters > 1:** The cat has multiple distinct behavioral modes (e.g.,
  different postures for urination vs. defecation, or morning vs. evening
  patterns).  This is not inherently concerning — it means the GMM has
  learned that the cat has multiple "normal" patterns.

#### Combined Layer 1 + Layer 2 assessment

| Layer 1 (EV) | Layer 2 (z-score) | Interpretation |
|---|---|---|
| normal | normal | Fully normal visit |
| normal | flagged | In-subspace anomaly — unusual coefficient combination |
| flagged | normal | Out-of-subspace anomaly — unfamiliar waveform direction |
| flagged | flagged | Serious anomaly — both shape and combination are unusual |

In simulation testing (400 visits, 5% swap injection), Layer 1 alone detected
0% of swapped visits (EVs all > 0.90).  Layer 2 alone detected **100%** of
scored swapped visits (z-scores from -3.67 to -104.90) with zero false
positives.

---

### 12.14 HTML Reports and the `eigen_report` Tool

#### Generating reports

Ask the agent to generate a report:

```
Show me Luna's eigenanalysis report.
```

The agent calls the `eigen_report` tool, which produces an HTML file at
`output/eigen_luna.html` and returns a text summary.

You can also generate reports programmatically:

```python
from litterbox.eigen_query import generate_report

generate_report("Luna", output_path="output/eigen_luna.html")
```

#### Reading the report

Each HTML report contains:

1. **Waveform overlay plot** — all stored zero-mean waveforms for the cat
   plotted on the same axes, color-coded by visit.  Look for:
   - Consistent shapes: the waveforms should cluster visually.
   - Outlier shapes: a waveform that looks very different from the rest.
   - Shape evolution: gradual changes over time could indicate behavioral
     shifts.

2. **Data table** — one row per visit with:

   | Column | What to look for |
   |--------|-----------------|
   | DC (mean g) | Absolute weight trend — gradual increase may indicate weight gain |
   | Explained Var | Should be > 0.90 for normal visits; drops indicate shape anomaly |
   | Residual | Reconstruction error — high values correlate with low EV |
   | N | Number of eigenvectors used (fixed at 4 with uniform_n) |
   | EV Anomaly | Layer 1 classification: normal/mild/significant/major |
   | k | Number of GMM clusters found for this cat |
   | Cluster | Which cluster this visit was assigned to (0, 1, 2...) |
   | Z-score | Layer 2 score — negative values below -2 indicate anomaly |
   | Signal Coefficients | The N expansion coefficients [c₁, ..., c_N] |

#### Query functions

```python
from litterbox.eigen_query import get_visit_summary, get_waveforms, get_model

# Per-visit summary with all fields
summaries = get_visit_summary("Luna")
for s in summaries:
    print(f"Visit {s['visit_number']}: EV={s['eigen_ev']:.4f}  "
          f"z={s['cluster_z_score']:.2f}  k={s['k_clusters']}")

# Raw waveform arrays for custom analysis
waveforms, timestamps = get_waveforms("Luna")
# waveforms: list of L=64 numpy arrays (zero-mean)

# Current eigenmodel
model = get_model("Luna")
# model["eigenvalues"], model["eigenvectors"], model["n_components"]
```

---

### 12.15 Simulation

The eigenanalysis simulator generates synthetic visits and runs them through
the full pipeline to verify the system end-to-end.

```bash
python simulator/eigen_sim.py                       # 400 visits, seed=42
python simulator/eigen_sim.py --seed 123            # different seed
python simulator/eigen_sim.py --visits 50           # fewer visits (faster)
python simulator/eigen_sim.py --report-only         # regenerate reports only
python simulator/eigen_sim.py --clean               # wipe eigen tables first
```

Each cat has a distinct parametric waveform shape:

| Cat | Shape character |
|-----|----------------|
| Anna | Steep ramp, high-frequency ripple on plateau |
| Luna | Gradual ramp, deep sag in the middle |
| Marina | Medium ramp, two sharp bumps (repositioning) |
| Natasha | Steep ramp, smooth parabolic rise |

5% of visits swap in a different cat's waveform shape to test anomaly
detection.  The chip ID still identifies the correct cat — only the shape
is wrong.

**Outputs:**

- `simulator/eigen_sim_reports/eigen_*.html` — per-cat HTML reports
- `simulator/eigen_sim_ground_truth.json` — per-visit ground truth
- `simulator/eigen_sim_summary.md` — detection statistics

**Baseline results (seed=42, 400 visits):**

- Normal visits: mean EV = 0.995, zero false positives
- Swapped visits: Layer 1 (EV) detection = 0%, Layer 2 (cluster) detection = 100%
- Calibrated uniform N = 4 (96.5% coverage at 95% EV)
