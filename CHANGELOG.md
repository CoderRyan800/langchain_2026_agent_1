# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [0.5.0] — 2026-03-21

### Added
- **Sensor data ingestion** — each visit can now capture weight scale and gas sensor readings via new CLI flags: `--weight-pre G`, `--weight-entry G`, `--weight-exit G`, `--ammonia-peak PPB`, `--methane-peak PPB`
- **Sensor summary columns on `visits`** — seven new nullable REAL columns: `weight_pre_g`, `weight_entry_g`, `weight_exit_g`, `cat_weight_g` (derived), `waste_weight_g` (derived), `ammonia_peak_ppb`, `methane_peak_ppb`
- **`visit_sensor_events` table** — time-series log of every individual sensor reading per visit, with `phase` (pre_entry / entry / exit), `sensor_type`, `value_numeric`, `value_text`, and `unit`
- **Derived weight computation** — `cat_weight_g = weight_entry_g − weight_pre_g`; `waste_weight_g = weight_exit_g − weight_pre_g`; computed in Python at write time
- **Peak gas reconciliation** — `ammonia_peak_ppb` and `methane_peak_ppb` on `visits` hold `MAX(entry_reading, exit_reading)`; entry readings are fetched from `visit_sensor_events` before the exit UPDATE
- **Health prompt enrichment** — `build_health_prompt(**sensor_kwargs)` inserts a structured sensor section into the GPT-4o prompt when any readings are present; `HEALTH_PROMPT` constant preserved for backward compatibility
- **Idempotent DB migration** — `init_db()` uses `PRAGMA table_info(visits)` to add the seven sensor columns only when missing, enabling zero-downtime upgrades of existing deployments
- **Comprehensive pytest suite** — 200 automated tests across six files: `test_db.py`, `test_health.py`, `test_tools_core.py`, `test_tools_sensor.py`, `test_integration.py`, `test_embeddings.py` (slow); run with `pytest -m "not slow"` for 180 fast tests in ~10 s
- **`pytest.ini`** — registers the `slow` marker and sets default test path/verbosity
- **`requirements-dev.txt`** — pins `pytest>=8.0.0` and `pytest-mock>=3.12.0`
- **Phase 8 manual test** — 36 new checks covering schema verification, `build_health_prompt` with sensors, `record_entry`/`record_exit` with full sensor data, derived weights, `visit_sensor_events` logging, peak gas MAX logic, and backward compatibility

### Changed
- `record_entry()` accepts optional `weight_pre_g`, `weight_entry_g`, `ammonia_peak_ppb`, `methane_peak_ppb` parameters
- `record_exit()` accepts optional `weight_exit_g`, `ammonia_peak_ppb`, `methane_peak_ppb` parameters; fetches entry gas readings to compute peak
- `run_sensor_event()` in `litterbox_agent.py` extended with five sensor parameters; system prompt updated to pass CLI sensor values to the tools
- Manual test runner updated from 7 to 8 phases

---

## [0.4.0] — 2026-03-21

### Added
- **Retroactive recognition** (`retroactive_recognition` tool) — after a new cat is registered, the agent automatically asks for the cat's acquisition date and re-runs the full CLIP + GPT-4o identification pipeline over all unknown visits on or after that date, confirming any that match
- **Agent system prompt updated** — the litter box agent now always prompts for an acquisition date after registration and calls `retroactive_recognition` automatically; owners can also trigger it directly
- **Phase 7 test suite** — 12 new checks covering invalid date, unknown cat, empty date range, orphan-exit exclusion, tentative-ID exclusion, missing-image skip, and a live CLIP+GPT-4o retroactive match
- **Test image fallback** — `prepare_images()` falls back to a local production reference image when the Wikimedia download is rate-limited, keeping the test suite runnable offline

### Changed
- Tool count increased from 10 to 11 (`ALL_TOOLS` now includes `retroactive_recognition`)
- Test runner updated from 6 to 7 phases; total check count raised from 62 to 72

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
