# LangChain 2026 Agent 1

Two AI agents built with LangChain and LangGraph:

- **Bob** (`src/basic_agent.py`) — general-purpose conversational assistant with web search and multimodal file upload
- **Litter Box Agent** (`src/litterbox_agent.py`) — automated cat health monitoring via litter box camera images

---

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key (with access to `gpt-4o`)
- A Tavily API key (Bob only)

### `.env` file

Create a `.env` file in the project root (it is gitignored):

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Option A — Conda environment

```bash
conda create -n langchain_env_2026_1 python=3.11 -y
conda activate langchain_env_2026_1
pip install -r requirements.txt
```

### Option B — Python venv

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** The litter box agent uses a local CLIP model (`clip-ViT-B-32`, ~350 MB)
> that is downloaded automatically by `sentence-transformers` on first run and cached
> afterwards.

---

## Agent 1 — Bob (general purpose)

### Running

```bash
python src/basic_agent.py
```

### Commands

| Input | Description |
|---|---|
| Any text | Send a message to Bob |
| `/UPLOAD <filepath>` | Upload an image or audio file for Bob to analyze |
| `/STOP` | Quit |

`/UPLOAD` accepts an absolute path or a path relative to the project root.

Supported formats:
- **Images**: `.jpg` `.jpeg` `.png` `.gif` `.webp`
- **Audio**: `.mp3` `.wav` `.ogg` `.m4a` `.flac` `.opus`

> **Note on audio:** Audio input requires changing `MODEL = "gpt-4o-audio-preview"` in
> `src/basic_agent.py`. The default `gpt-4o` supports images only.

### Why `/UPLOAD` is in the chat loop, not a tool

LangChain tools are invoked by the model *after* it receives a message. For vision to
work, image data must be injected into the `HumanMessage` content block *before* the
model sees it — as an `image_url` entry containing a base64-encoded data URL. If upload
were a tool, the data would come back in a `ToolMessage`, which the OpenAI vision API
does not render. Pre-processing in the loop is the correct approach for vision to work.

### Model

Bob uses `gpt-4o` by default. Change the `MODEL` constant at the top of
`src/basic_agent.py` to switch models.

---

## Agent 2 — Litter Box Monitor

### Overview

The litter box agent tracks cats' litter box usage and monitors their health through
paired entry/exit camera images. It runs in two modes:

- **Interactive** — for cat registration, identity confirmation, and data queries
- **Sensor-triggered** — called automatically by the camera/sensor system on entry or exit events

### Running — interactive mode

```bash
python src/litterbox_agent.py
```

### Running — sensor-triggered mode

```bash
# Cat enters the litter box (called by the sensor/camera system)
python src/litterbox_agent.py --event entry --image images/captures/entry.jpg

# Cat exits the litter box
python src/litterbox_agent.py --event exit --image images/captures/exit.jpg
```

### Interactive commands

| Input | Description |
|---|---|
| Any text | Chat with the agent — ask questions, confirm identities, query records |
| `/UPLOAD <filepath>` | Register a reference photo for a named cat |
| `/STOP` | Quit |

When registering a cat via `/UPLOAD`, the agent will ask for the cat's name if you
don't provide it in the same message.

---

### The four conditions

**Condition 1 — Register a cat reference image**

Upload a known photo of a cat with their name. Multiple reference images per cat
are supported and improve identification accuracy.

```
You: /UPLOAD /path/to/whiskers.jpg
Assistant: I can see a cat in this photo. What is this cat's name?
You: Whiskers
Assistant: Registered reference image #1 for 'Whiskers'. Stored at images/cats/whiskers/001_a3f2c1.jpg
```

**Condition 2 — Cat enters the litter box (automated)**

Triggered by the sensor system. The agent runs the two-stage identification pipeline
and opens a visit record.

```bash
python src/litterbox_agent.py --event entry --image images/captures/entry.jpg
```

Output:
```
Visit #7 opened at 2026-03-14 09:14:22 UTC.
Entry image: images/visits/2026-03-14/a1b2c3d4_entry.jpg
Tentative ID: Whiskers (similarity 0.91).
```

**Condition 3 — Cat exits the litter box (automated)**

Triggered by the sensor system. The agent closes the visit record and runs a health
analysis comparing the entry and exit images.

```bash
python src/litterbox_agent.py --event exit --image images/captures/exit.jpg
```

Output:
```
Visit #7 closed (tentative cat: Whiskers).
Health: ⚠️  ANOMALY FLAGGED — veterinary review recommended

CONCERNS_PRESENT: yes
DESCRIPTION: The exit image shows pink discoloration in the urine clump not
present in the entry image.
OWNER_SUMMARY: There appears to be blood in the urine. This warrants prompt
attention.

⚠️ This analysis is preliminary and must be reviewed by a licensed veterinarian
before any medical decisions are made.
```

**Condition 4 — Owner confirms cat identity**

The owner reviews unconfirmed visits and assigns the correct cat name.

```
You: Show me unconfirmed visits
You: Visit 7 is confirmed as Whiskers
Assistant: Visit #7 confirmed: cat is 'Whiskers'.
```

---

### Cat identification pipeline

Identification runs in two stages on every entry photo:

1. **CLIP nearest-neighbor search** — the entry photo is embedded using a local
   `clip-ViT-B-32` model and compared against all stored reference image embeddings
   in Chroma. The top candidates are returned with cosine similarity scores.

2. **GPT-4o visual confirmation** — for any candidate scoring above the threshold
   (default **0.82**), GPT-4o is shown the new photo alongside the candidate's
   reference photo and asked to confirm whether they show the same cat. This catches
   cases where CLIP similarity alone is insufficient (e.g. unusual angles, lighting).

If no candidate clears the threshold, or GPT-4o does not confirm any candidate, the
visit is recorded with an "Unknown" tentative ID and flagged for human review.

All tentative IDs remain unconfirmed until the owner explicitly confirms them via
`confirm_identity`.

---

### Orphan exit records

If an exit event arrives with no corresponding open visit (e.g. the entry sensor
misfired), the agent creates an orphan exit record — a visit row with only the exit
image — and emits a warning. Orphan records are visible in query results and flagged
for human review.

---

### Available query tools

The agent responds to natural-language queries backed by these tools:

| Tool | Example question |
|---|---|
| `get_visits_by_date` | "Show me all visits from yesterday" |
| `get_visits_by_cat` | "How many times did Whiskers visit this week?" |
| `get_anomalous_visits` | "Show me all flagged health events" |
| `get_unconfirmed_visits` | "Which visits still need confirmation?" |
| `get_visit_images` | "What images do we have for visit 7?" |
| `list_cats` | "Which cats are registered?" |

---

### Health analysis and the veterinary disclaimer

Every exit event triggers a GPT-4o comparison of the entry and exit images. The model
looks for visual signs that may indicate health concerns: blood in urine or stool,
unusual stool consistency or color, mucus, abnormal deposits, or other unexpected
findings.

The `is_anomalous` flag is set when the model reports concerns present. The full
analysis text is stored in `health_notes` on the visit record.

**All health findings are preliminary.** Every analysis response ends with:

> ⚠️ This analysis is preliminary and must be reviewed by a licensed veterinarian
> before any medical decisions are made.

This system is a monitoring aid, not a diagnostic tool.

---

### Data storage

| Store | Location | Contents |
|---|---|---|
| SQLite metadata | `data/litterbox.db` | cats, cat_images, visits tables |
| Chroma vector DB | `data/chroma/` | CLIP embeddings for all reference images |
| LangGraph checkpoints | `data/agent_litterbox_memory.db` | conversation history (interactive thread + sensor thread) |
| Raw images | `images/` | reference photos under `cats/`, visit captures under `visits/` |

All of `data/` and `images/` are gitignored.

The SQLite and Chroma stores use separate `thread_id` values for the interactive
session (`"interactive"`) and sensor events (`"sensor"`), so automated entries do not
pollute the interactive conversation history.

---

## How memory works

Both agents use the same LangGraph memory architecture. Memory is handled by two
independent mechanisms operating at different layers:

1. **`SqliteSaver` (LangGraph checkpointer)** — persists the full agent state after
   every invocation. Bob writes to `agent_memory.db`; the litter box agent writes to
   `data/agent_litterbox_memory.db`.

2. **`SummarizationMiddleware`** — trims the in-memory message list passed to the
   model at inference time, replacing old messages with a rolling summary. Configured
   with `trigger=("messages", 10)` and `keep=("messages", 3)`.

### What gets loaded on restart

LangGraph loads the **latest checkpoint** for the configured `thread_id`. That
checkpoint contains the summary message (covering older history) plus the most recent
messages within the keep window — not the full raw history.

### What lives in SQLite but is not reloaded

LangGraph retains all prior checkpoint snapshots for a thread — one per invocation.
The raw messages that were replaced by the summarizer are preserved in older snapshots
for audit or replay, but have no effect on the running agent.

### Scaling to Postgres

`SqliteSaver` is a drop-in development backend. LangGraph supports `PostgresSaver`
using the same interface, making migration straightforward when larger histories or
multi-user deployments require it.

---

## Project structure

```
.
├── src/
│   ├── basic_agent.py          # Bob — general purpose agent
│   ├── litterbox_agent.py      # Litter box monitoring agent
│   └── litterbox/
│       ├── db.py               # SQLite schema and query helpers
│       ├── embeddings.py       # CLIP embeddings + Chroma vector search
│       ├── health.py           # GPT-4o health analysis prompt and parser
│       └── tools.py            # All 10 LangChain tools
├── data/                       # Gitignored — created on first run
│   ├── litterbox.db            # Cat and visit metadata
│   ├── chroma/                 # CLIP vector index
│   └── agent_litterbox_memory.db
├── images/                     # Gitignored — created on first run
│   ├── cats/                   # Reference photos per cat
│   ├── visits/                 # Entry/exit captures organised by date
│   └── captures/               # Drop zone for incoming sensor images
├── agent_memory.db             # Bob's conversation store (gitignored)
├── .env                        # API keys (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```
