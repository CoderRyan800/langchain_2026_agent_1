# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [0.3.0] — 2026-03-14

### Added
- **Litter box monitoring agent** (`src/litterbox_agent.py`) — dedicated agent for automated cat health monitoring via litter box camera images
- **Two invocation modes** — interactive (cat registration, confirmation, queries) and sensor-triggered (`--event entry/exit --image <path>`)
- **Cat identification pipeline** — two-stage: local CLIP (`clip-ViT-B-32`) nearest-neighbor search followed by GPT-4o visual confirmation for candidates above the 0.82 cosine similarity threshold
- **Health analysis** — GPT-4o compares litter box entry and exit images after each visit and flags potential concerns; all findings include a mandatory veterinary disclaimer
- **10 LangChain tools** — `register_cat_image`, `record_entry`, `record_exit`, `confirm_identity`, `get_visits_by_date`, `get_visits_by_cat`, `get_anomalous_visits`, `get_unconfirmed_visits`, `get_visit_images`, `list_cats`
- **Orphan exit handling** — exit events with no matching open visit create a flagged orphan record rather than failing
- **Separate data stores** — `data/litterbox.db` (SQLite metadata), `data/chroma/` (CLIP vector index), `data/agent_litterbox_memory.db` (LangGraph checkpoints)
- **Manual test suite** (`tests/run_manual_test.py`) — 6 phases, 62 checks covering storage, CLIP embeddings, health analysis, confirmation, end-to-end CLI subprocess, and reset verification
- **User Guide** (`docs/USER_GUIDE.md`) and **Testing guide** (`docs/TESTING.md`)
- `chromadb`, `sentence-transformers`, and `Pillow` added to `requirements.txt`
- `data/` and `images/` added to `.gitignore`
- MIT License

---

## [0.2.0] — 2026-03-14

### Added
- **SQLite persistence** — replaced `InMemorySaver` with `SqliteSaver` (`langgraph-checkpoint-sqlite`); Bob's conversation history now survives restarts in `agent_memory.db`
- **Multimodal file upload** — `/UPLOAD <filepath>` command in the Bob chat loop; images are base64-encoded and injected as `image_url` content blocks before the model sees them, enabling true vision support
- **Audio upload support** — `.mp3`, `.wav`, `.ogg`, `.m4a`, `.flac`, `.opus` via `input_audio` content blocks (requires `MODEL = "gpt-4o-audio-preview"`)
- **Model upgrade** — switched from `gpt-5-nano` to `gpt-4o` for reliable vision support
- `requirements.txt` with pinned minimum versions for all direct dependencies
- README documenting the memory architecture, summarization behaviour, and `/UPLOAD` design rationale

### Changed
- `print_response` strips base64 content from human messages before printing to avoid flooding the terminal

---

## [0.1.0] — 2026-03-14

### Added
- **Bob** (`src/basic_agent.py`) — general-purpose conversational agent using LangChain `create_agent` and LangGraph
- **Web search tool** via Tavily
- **Conversation summarization** via `SummarizationMiddleware` (trigger: 10 messages, keep: 3)
- In-memory conversation state via `InMemorySaver`
- Interactive chat loop with `/STOP` command
- `.env`-based API key loading via `python-dotenv`

---

[Unreleased]: https://github.com/CoderRyan800/langchain_2026_agent_1/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/CoderRyan800/langchain_2026_agent_1/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/CoderRyan800/langchain_2026_agent_1/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/CoderRyan800/langchain_2026_agent_1/releases/tag/v0.1.0
