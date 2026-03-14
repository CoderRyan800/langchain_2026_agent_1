# LangChain 2026 Agent 1

A conversational AI agent built with LangChain and LangGraph, featuring persistent memory via SQLite, web search, and multimodal file upload (images and audio).

---

## Features

- **Persistent memory** across restarts via SQLite checkpointing
- **Conversation summarization** to keep the active context window lean
- **Web search** via Tavily
- **Multimodal file upload** via `/UPLOAD` — supports images and audio

---

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key (with access to `gpt-4o`)
- A Tavily API key

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

---

## Running the agent

**Conda:**
```bash
conda activate langchain_env_2026_1
python src/basic_agent.py
```

**venv:**
```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python src/basic_agent.py
```

---

## Usage

At the `You:` prompt:

| Input | Description |
|---|---|
| Any text | Send a message to the agent |
| `/UPLOAD <filepath>` | Upload an image or audio file for the agent to analyze |
| `/STOP` | Quit the agent |

### `/UPLOAD` example

```
You: /UPLOAD /Users/yourname/Desktop/photo.jpg
You: /UPLOAD /tmp/recording.mp3
```

Use an absolute path or a path relative to the project root. Supported formats:

- **Images**: `.jpg` `.jpeg` `.png` `.gif` `.webp`
- **Audio**: `.mp3` `.wav` `.ogg` `.m4a` `.flac` `.opus`

> **Note on audio:** Audio input requires `MODEL = "gpt-4o-audio-preview"` in `src/basic_agent.py`. The default model `gpt-4o` supports images only.

### Why `/UPLOAD` is handled in the chat loop, not as a tool

LangChain tools are invoked by the *model* after it receives a message. But for vision to work, image data must be injected into the `HumanMessage` content block *before* the model sees it — as an `image_url` entry with a base64-encoded data URL. If upload were a tool, the image data would come back in a `ToolMessage`, which the OpenAI vision API does not render. The current design pre-processes the file in the loop and injects it as proper multimodal message content, which is the correct approach for vision to actually work.

---

## Model

The agent uses `gpt-4o` by default, which supports vision (image inputs). Change the `MODEL` constant at the top of `src/basic_agent.py` to switch models.

---

## How memory works

### Overview

Memory is handled by two independent mechanisms that operate at different layers:

1. **SqliteSaver (LangGraph checkpointer)** — persists the full agent state to `agent_memory.db` after every invocation
2. **SummarizationMiddleware** — trims the message list passed to the model at inference time, replacing old messages with a summary

### SQLite checkpointing

LangGraph's `SqliteSaver` writes a checkpoint snapshot to `agent_memory.db` after every single invocation. Each checkpoint captures the complete message state of the conversation at that point in time.

Crucially, LangGraph retains **all prior checkpoint snapshots** for a thread — not just the latest one. This means the database contains the full unabridged history of every message ever exchanged, including raw messages that predate any summarization events. The database grows over time but is never automatically pruned.

On restart, the agent loads the **latest checkpoint** for the configured `thread_id`. This is not the full raw history — it is the state as it existed at the end of the last invocation, which will typically be a lean summary + recent tail (see below).

### Conversation summarization

`SummarizationMiddleware` is configured with:

```python
trigger=("messages", 10),  # fire when message count hits 10
keep=("messages", 3)        # keep the 3 most recent messages
```

When the message count reaches 10 during an invocation, the middleware replaces messages 1–7 with a single summary message and retains the 3 most recent. This trimmed state — **summary + 3 recent messages** — is what gets written to the SQLite checkpoint and reloaded on the next restart.

### What gets loaded on restart

When you restart the agent and send your first message, LangGraph loads the latest checkpoint from SQLite. That checkpoint contains:

- The summary message (covering all history older than the last summarization event)
- The most recent messages within the keep window
- Any messages added since the last summarization trigger

The raw messages that were replaced by the summary are still present in older checkpoint snapshots in SQLite, but they are not loaded into the active agent state.

### What lives in SQLite but is not reloaded

- All prior checkpoint snapshots (one per past invocation)
- The original raw messages from before any summarization events

This data is preserved for audit, replay, or inspection but has no effect on the running agent.

### Scaling to Postgres

`SqliteSaver` is a drop-in development backend. LangGraph supports `PostgresSaver` using the same interface, making migration straightforward when larger conversation histories or multi-user deployments require it. All checkpoint history stored in SQLite can be migrated to Postgres without loss.

### Thread identity

All conversation history is keyed by `thread_id`. The current default is `"1"`. To start a fresh conversation with no prior history, change the `thread_id` in the `config` dict in `src/basic_agent.py`:

```python
config = {"configurable": {"thread_id": "2"}}
```

---

## Project structure

```
.
├── src/
│   └── basic_agent.py   # Main agent script
├── agent_memory.db      # SQLite conversation store (gitignored)
├── .env                 # API keys (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```
