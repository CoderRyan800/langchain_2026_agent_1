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

### What is analysed

After every exit event, GPT-4o compares the entry and exit images of the
litter box. It looks for visual differences that may indicate a health concern:

- Blood in urine (pink, red, or dark discolouration of urine clumps)
- Blood in stool
- Unusual stool colour, consistency, or quantity
- Evidence of diarrhoea or mucus
- Abnormal deposits or clumping patterns
- Any other unexpected visual changes

When sensor readings are available, they are included in the prompt and improve
detection of anomalies such as elevated ammonia (potential urinary issues) or
unexpectedly large waste deposits.

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
