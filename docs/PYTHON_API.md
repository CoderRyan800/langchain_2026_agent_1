# Python API — Installation and Usage

This document covers how to install the `litterbox-monitor` package and use the
`LitterboxAgent` Python API to drive the litter box monitoring system from your
own code instead of the command line.

---

## Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Constructor Options](#3-constructor-options)
4. [Sensor Event Methods](#4-sensor-event-methods)
5. [Cat Registration](#5-cat-registration)
6. [Identity Management](#6-identity-management)
7. [Query Methods](#7-query-methods)
8. [Natural Language Queries](#8-natural-language-queries)
9. [Context Manager Usage](#9-context-manager-usage)
10. [Data Storage Locations](#10-data-storage-locations)
11. [Integration Example — Raspberry Pi Sensor Script](#11-integration-example--raspberry-pi-sensor-script)

---

## 1. Installation

### Prerequisites

- Python 3.11+
- An OpenAI API key (set in `.env` or the environment)

### Install from the project root

```bash
# Clone or navigate to the project directory
cd /path/to/langchain_2026_agent_1

# Create and activate a virtual environment (conda recommended)
conda create -n litterbox python=3.11 -y
conda activate litterbox

# Install the package in editable mode
pip install -e .
```

The `-e` flag installs in *editable* mode so changes to the source files in
`src/` take effect immediately without reinstalling.

### Environment variables

Create a `.env` file in your working directory (or set the variable in your
shell/systemd unit):

```
OPENAI_API_KEY=sk-...
```

The package loads this file automatically via `python-dotenv`.

---

## 2. Quick Start

```python
from litterbox import LitterboxAgent

# Data is stored in ~/.litterbox_monitor/ by default
agent = LitterboxAgent()

# Register a reference photo for your cat
print(agent.register_cat("/path/to/whiskers.jpg", "Whiskers"))

# Process a sensor-triggered entry event
print(agent.record_entry(
    "/path/to/entry_capture.jpg",
    weight_pre_g=5412,
    weight_entry_g=8634,
    ammonia_peak_ppb=38,
))

# Process the corresponding exit event
print(agent.record_exit(
    "/path/to/exit_capture.jpg",
    weight_exit_g=5489,
    ammonia_peak_ppb=62,
))

# Query the database
print(agent.get_anomalous_visits())
print(agent.list_cats())
```

---

## 3. Constructor Options

```python
LitterboxAgent(
    data_dir: str | None = None,
    images_dir: str | None = None,
    openai_api_key: str | None = None,
)
```

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | `~/.litterbox_monitor/data` | Directory for `litterbox.db`, `agent_memory.db`, and the Chroma vector index |
| `images_dir` | `~/.litterbox_monitor/images` | Directory tree for cat reference photos and visit captures |
| `openai_api_key` | `$OPENAI_API_KEY` | Pass the key directly instead of relying on the environment |

### Example with explicit paths

```python
agent = LitterboxAgent(
    data_dir="/srv/litterbox/data",
    images_dir="/srv/litterbox/images",
)
```

### Example with an API key in code

```python
import os
agent = LitterboxAgent(openai_api_key=os.environ["MY_OPENAI_KEY"])
```

---

## 4. Sensor Event Methods

These methods call the underlying tools **directly** — no LLM round-trip is
required, so they are fast and incur minimal API cost.

### `record_entry`

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
visit record.  All sensor parameters are optional — pass only the sensors you
have.

**Returns:** Plain-text summary, e.g.:
```
Visit #7 opened at 2026-03-28 14:22:05 UTC.
Entry image: images/visits/2026-03-28/a1b2c3d4_entry.jpg
Tentative ID: Whiskers (similarity 0.94).
Sensors: cat weight 3222 g, NH₃ 38 ppb.
```

### `record_exit`

```python
result: str = agent.record_exit(
    image_path: str,
    weight_exit_g: float | None = None,
    ammonia_peak_ppb: float | None = None,
    methane_peak_ppb: float | None = None,
)
```

Associates the exit with the most recent open visit, runs the GPT-4o health
analysis, and writes the result.  If no open visit exists, an orphan record is
created.

**Returns:** Plain-text summary including health analysis, e.g.:
```
Visit #7 closed (tentative cat: Whiskers).
Exit image: images/visits/2026-03-28/a1b2c3d4_exit.jpg
Sensors: cat weight 3222 g, waste 77 g, NH₃ 62 ppb.
Health: No anomalies detected

CONCERNS_PRESENT: no
DESCRIPTION: ...
```

---

## 5. Cat Registration

### `register_cat`

```python
result: str = agent.register_cat(image_path: str, cat_name: str)
```

Registers a reference photo for a named cat.  Multiple photos can be registered
for the same cat by calling this method repeatedly — more reference images
improve identification accuracy.

```python
agent.register_cat("/photos/whiskers_1.jpg", "Whiskers")
agent.register_cat("/photos/whiskers_2.jpg", "Whiskers")  # second reference image
agent.register_cat("/photos/mochi_1.jpg", "Mochi")
```

### `list_cats`

```python
result: str = agent.list_cats()
```

Returns a summary of all registered cats and their reference image counts.

---

## 6. Identity Management

### `confirm_identity`

```python
result: str = agent.confirm_identity(visit_id: int, cat_name: str)
```

Permanently sets the confirmed cat identity for a visit.  Use this after
reviewing unconfirmed visits.

```python
# Check what needs confirmation
print(agent.get_unconfirmed_visits())

# Confirm visit #7 is Whiskers
print(agent.confirm_identity(7, "Whiskers"))
```

### `retroactive_recognition`

```python
result: str = agent.retroactive_recognition(cat_name: str, since_date: str)
```

Scans all unknown visits since `since_date` and re-runs the full CLIP + GPT-4o
pipeline for the specified cat.  Useful after registering a new cat to identify
visits that happened before the cat was added.

```python
# Re-scan all unknown visits since Mochi arrived
agent.retroactive_recognition("Mochi", "2026-03-01")
```

---

## 7. Query Methods

All query methods call the database tools directly — no LLM is involved.

| Method | Parameters | Returns |
|---|---|---|
| `list_cats()` | — | Registered cats and reference image counts |
| `get_visits_by_date(date_str)` | `"YYYY-MM-DD"` | All visits on the given date |
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

## 8. Natural Language Queries

### `query`

```python
response: str = agent.query(message: str, thread_id: str = "api")
```

Sends any plain-English message to the full LangGraph agent, which can call all
11 tools and maintains conversation history within the given `thread_id`.

This is the highest-level interface and uses GPT-4o for every call.  Use the
structured methods above when you want predictable, cost-efficient responses.

```python
# Simple queries
print(agent.query("How many times did Whiskers use the litter box this week?"))
print(agent.query("Show me all anomalous visits from March 2026"))
print(agent.query("Which cat visited most frequently last week?"))

# Multi-turn conversation (same thread_id preserves history)
agent.query("Tell me about visit 5", thread_id="session-1")
agent.query("Confirm that visit as Mochi", thread_id="session-1")

# Register a cat via natural language
agent.query("I want to register this cat. File path: /photos/mochi.jpg",
            thread_id="registration")
```

---

## 9. Context Manager Usage

Use `LitterboxAgent` as a context manager to ensure the SQLite checkpointer is
closed cleanly when your script exits:

```python
with LitterboxAgent() as agent:
    agent.record_entry("/captures/entry.jpg")
    agent.record_exit("/captures/exit.jpg")
    print(agent.get_anomalous_visits())
```

Alternatively, call `agent.close()` explicitly when you are done.  An `atexit`
handler is registered automatically so the connection is also released if your
process exits normally without an explicit close.

---

## 10. Data Storage Locations

By default all runtime state is stored under `~/.litterbox_monitor/`:

| What | Path |
|---|---|
| Cat reference photos | `~/.litterbox_monitor/images/cats/<catname>/` |
| Visit entry/exit images | `~/.litterbox_monitor/images/visits/YYYY-MM-DD/` |
| Cat and visit metadata | `~/.litterbox_monitor/data/litterbox.db` |
| CLIP vector index | `~/.litterbox_monitor/data/chroma/` |
| Agent conversation history | `~/.litterbox_monitor/data/agent_memory.db` |

Override any of these by passing `data_dir` and/or `images_dir` to the
constructor.

### Backing up

```bash
tar -czf litterbox_backup_$(date +%F).tar.gz ~/.litterbox_monitor/
```

### Starting fresh

```bash
rm -rf ~/.litterbox_monitor/
```

---

## 11. Integration Example — Raspberry Pi Sensor Script

This shows how a sensor daemon might call the API when the camera detects motion.

```python
#!/usr/bin/env python3
"""litterbox_sensor_daemon.py — called by camera motion trigger."""

import sys
from litterbox import LitterboxAgent

# Sensor hardware libraries (example)
from scale import read_weight
from gas_sensor import read_ammonia, read_methane

def on_entry(image_path: str) -> None:
    weight_pre = read_weight()           # grams, or None if scale offline
    weight_entry = read_weight()
    ammonia = read_ammonia()             # ppb, or None if sensor offline
    methane = read_methane()

    with LitterboxAgent() as agent:
        result = agent.record_entry(
            image_path,
            weight_pre_g=weight_pre,
            weight_entry_g=weight_entry,
            ammonia_peak_ppb=ammonia,
            methane_peak_ppb=methane,
        )
    print(result)


def on_exit(image_path: str) -> None:
    weight_exit = read_weight()
    ammonia = read_ammonia()
    methane = read_methane()

    with LitterboxAgent() as agent:
        result = agent.record_exit(
            image_path,
            weight_exit_g=weight_exit,
            ammonia_peak_ppb=ammonia,
            methane_peak_ppb=methane,
        )
    print(result)


if __name__ == "__main__":
    event = sys.argv[1]     # "entry" or "exit"
    image = sys.argv[2]     # absolute path to captured image
    if event == "entry":
        on_entry(image)
    elif event == "exit":
        on_exit(image)
```

---

## Console Scripts

After installation the following commands are available in your PATH:

```bash
# Interactive litter box agent (equivalent to python src/litterbox_agent.py)
litterbox-agent

# Sensor-triggered events
litterbox-agent --event entry --image /path/to/entry.jpg
litterbox-agent --event exit  --image /path/to/exit.jpg \
    --weight-exit 5489 --ammonia-peak 62

# With custom data directories
litterbox-agent --data-dir /srv/litterbox/data --images-dir /srv/litterbox/images

# Bob general-purpose assistant
litterbox-bob
```
