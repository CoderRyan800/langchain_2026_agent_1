# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two AI agents built with LangChain and LangGraph:

- **Bob** (`src/basic_agent.py`) — General-purpose conversational assistant with web search and multimodal file upload
- **Litter Box Monitor** (`src/litterbox_agent.py`) — Automated cat health monitoring via litter box camera images, with 10 LangChain tools, two-stage CLIP+GPT-4o identity pipeline, and health analysis

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
```

Bob supports `/UPLOAD /path/to/file` for images/audio and `/STOP` to quit.

## Tests

```bash
python tests/run_manual_test.py                    # all 6 phases (~$0.25–0.50)
python tests/run_manual_test.py --phase 1          # phase 1 only (free, no LLM calls)
python tests/run_manual_test.py --phase 1 2 4      # multiple phases
python tests/run_manual_test.py --no-cleanup       # keep test artifacts
```

Phases: 1=storage/schema, 2=CLIP embeddings, 3=health analysis, 4=identity confirmation, 5=sensor CLI, 6=reset. Phases 1, 4, 6 are free. The test suite uses isolated paths (`tests/test_data/`) to avoid touching production data.

## Architecture

### Bob (`src/basic_agent.py`)
- Model: `gpt-4o`; memory: `agent_memory.db` (SQLite via `SqliteSaver`)
- `SummarizationMiddleware`: after 10 messages, summarizes messages 1–7, keeps last 3
- Tool: Tavily web search

### Litter Box Agent (`src/litterbox_agent.py` + `src/litterbox/`)
- Model: `gpt-4o` (vision required)
- Memory: `data/agent_litterbox_memory.db`; thread IDs `"sensor"` vs `"interactive"` maintain separate histories
- **`src/litterbox/db.py`** — SQLite schema (cats, cat_images, visits); `init_db()` is idempotent
- **`src/litterbox/embeddings.py`** — CLIP (`clip-ViT-B-32`, downloads ~350 MB on first run) + Chroma vector search; `ID_THRESHOLD = 0.82`
- **`src/litterbox/health.py`** — GPT-4o health analysis prompt and `parse_health_response()`
- **`src/litterbox/tools.py`** — 10 `@tool` functions; docstrings auto-generate LangChain tool descriptions

### Two-Stage Cat ID Pipeline
1. **CLIP** (local, free): embeds entry image, queries Chroma for top candidates
2. **GPT-4o** (API cost): visual side-by-side comparison for candidates above threshold (0.82)

If no candidate passes both stages, the visit is recorded unconfirmed for human review via `confirm_identity` tool.

### Health Analysis
`record_exit()` sends entry+exit images to GPT-4o; response stored in `visits.health_notes` with `is_anomalous` flag. Response always includes a mandatory veterinary disclaimer.

### Runtime Data (gitignored)
- `data/litterbox.db` — SQLite metadata
- `data/chroma/` — CLIP vector index
- `images/cats/`, `images/visits/`, `images/captures/` — image storage

### Key Implementation Details
- `_abs()` in `tools.py` handles both absolute and project-relative paths
- Orphan exits (no open visit): `record_exit()` creates a flagged orphan record rather than failing
- Test suite patches module-level constants (`DB_PATH`, `CHROMA_PATH`, `IMAGES_DIR`) for isolation
