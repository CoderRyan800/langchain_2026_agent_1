# Test Report — LitterboxAgent Python API

**Date:** 2026-03-28
**Environment:** `langchain_env_2026_1` (Python 3.12.12, pytest 9.0.2)
**Test file:** `tests/test_api.py`
**Component under test:** `src/litterbox/api.py` — `LitterboxAgent` class

---

## Executive Summary

| Metric | Value |
|---|---|
| Tests in `test_api.py` | **87** |
| Passed | **87** |
| Failed | **0** |
| Errors | **0** |
| Execution time | **7.95 s** |
| LLM / API calls made | **0** (all mocked) |
| Regressions in existing suite | **0** (267 / 267 passing) |

All 87 tests pass. No production data was touched; every test runs against isolated temporary directories.

---

## Scope

This report covers unit and integration tests for the new Python API layer introduced in `src/litterbox/api.py`. The API wraps the existing LangChain tool functions and LangGraph agent to provide a clean, importable Python interface in place of the previous command-line-only interface.

Tests were **not** written for:

- The underlying tool functions (`record_entry`, `record_exit`, etc.) — those are covered by the pre-existing suite (`test_tools_core.py`, `test_tools_sensor.py`, `test_integration.py`).
- The CLIP embedding model — covered by `test_embeddings.py` (marked `slow`).
- The interactive CLI loop in `src/litterbox/_cli.py` — that is a thin shim; interactive terminal sessions are not amenable to automated testing.
- Live OpenAI API calls — intentionally excluded to keep the suite free, fast, and deterministic.

---

## Test Design

### Isolation strategy

Each test creates a `LitterboxAgent` pointed at a fresh `tmp_path` sub-directory (`data/` and `images/`). This overrides the module-level path constants in `litterbox.db`, `litterbox.embeddings`, and `litterbox.tools` before any tool executes, so no test can read from or write to:

- `~/.litterbox_monitor/` (the default runtime location)
- `data/litterbox.db` or `data/chroma/` (the development database)
- Any other test's temporary directory

The conftest.py `isolated_env` autouse fixture also patches these paths at the start of every test; the `agent` fixture then overwrites the patches with its own `tmp_path`. After each test, `monkeypatch` restores the original module-level values.

### LLM stub strategy

All tests that exercise `record_entry` or `record_exit` monkey-patch two functions:

- `litterbox.tools._identify_cat` → returns `(None, None, 0.3, "No reference images")` (no-match stub)
- `litterbox.tools._run_gpt4o_vision` → returns either `_HEALTH_NORMAL` or `_HEALTH_ANOMALY` (canned strings)

The `query()` method tests monkey-patch `agent._get_agent()` to return a minimal fake agent object whose `.invoke()` returns a fixed message list. No `create_agent` call is made and no checkpointer thread is spawned.

### Database verification

Tests that assert on database state bypass the API and open a direct `sqlite3` connection to the agent's `litterbox.db` via the `_direct_db()` helper. This ensures the assertions are checking what is actually persisted, not just what the API reports.

---

## Test Results by Class

### TestLitterboxAgentInit — 9 tests, 9 passed

Verifies the constructor's side effects before any tool is called.

| Test | What it checks |
|---|---|
| `test_creates_data_dir` | `data_dir` is created if it does not exist |
| `test_creates_images_dir` | `images_dir` is created if it does not exist |
| `test_db_file_created` | `litterbox.db` is created on construction |
| `test_agent_memory_db_created` | `agent_memory.db` is created on construction |
| `test_patches_db_path` | `litterbox.db.DB_PATH` is overwritten to the agent's data dir |
| `test_patches_images_dir` | `litterbox.tools.IMAGES_DIR` is overwritten to the agent's images dir |
| `test_patches_chroma_path` | `litterbox.embeddings.CHROMA_PATH` is overwritten |
| `test_resets_chroma_collection_singleton` | `_collection` is set to `None` so Chroma re-initialises at the new path |
| `test_db_schema_initialised` | All four tables (`cats`, `cat_images`, `visits`, `visit_sensor_events`) exist after construction |

---

### TestContextManager — 4 tests, 4 passed

Verifies the `with`-statement interface and `close()` semantics.

| Test | What it checks |
|---|---|
| `test_context_manager_returns_agent` | `__enter__` returns the agent itself |
| `test_close_sets_saver_ctx_none` | `close()` nulls out `_saver_ctx` |
| `test_double_close_does_not_raise` | Calling `close()` twice is safe |
| `test_close_sets_checkpointer_none` | `close()` nulls out `_checkpointer` |

---

### TestRegisterCat — 7 tests, 7 passed

Verifies `agent.register_cat(image_path, cat_name)`.

| Test | What it checks |
|---|---|
| `test_returns_success_string` | Return value contains "Registered" |
| `test_creates_cats_row` | A row appears in the `cats` table |
| `test_copies_image_file` | The image file is physically copied into `images/cats/` |
| `test_second_image_increments_count` | A second call for the same cat name adds a second `cat_images` row |
| `test_no_duplicate_cat_row` | Two calls for the same name do not create two `cats` rows |
| `test_missing_image_returns_error` | Non-existent path returns a string containing "Error" |
| `test_two_cats_both_stored` | Two different cat names produce two separate `cats` rows |

---

### TestRecordEntry — 12 tests, 12 passed

Verifies `agent.record_entry(image_path, ...)`.

| Test | What it checks |
|---|---|
| `test_returns_string` | Return value is a `str` |
| `test_creates_visit_row` | A row is inserted into `visits` |
| `test_entry_image_stored` | A `*_entry.jpg` file appears under `images/visits/` |
| `test_result_mentions_visit_number` | Return value references the visit ID |
| `test_missing_image_returns_error` | Non-existent path returns "Error" |
| `test_weight_pre_stored` | `weight_pre_g` column is persisted correctly |
| `test_weight_entry_stored` | `weight_entry_g` column is persisted correctly |
| `test_cat_weight_derived` | `cat_weight_g = weight_entry_g − weight_pre_g` is computed and stored |
| `test_ammonia_stored` | `ammonia_peak_ppb` column is persisted correctly |
| `test_methane_stored` | `methane_peak_ppb` column is persisted correctly |
| `test_sensor_event_rows_logged` | At least 2 rows appear in `visit_sensor_events` when sensors are provided |
| `test_no_sensors_columns_null` | When no sensor args are passed, sensor columns are all NULL |

---

### TestRecordExit — 12 tests, 12 passed

Verifies `agent.record_exit(image_path, ...)`.

| Test | What it checks |
|---|---|
| `test_returns_string` | Return value is a `str` |
| `test_closes_visit` | `exit_time` is set on the open visit row |
| `test_exit_image_stored` | A `*_exit.jpg` file appears under `images/visits/` |
| `test_health_normal_in_result` | Normal health stub result is reflected in the return string |
| `test_health_anomalous_flagged` | Anomalous health stub causes "ANOMALY" to appear in the return string |
| `test_is_anomalous_flag_stored_normal` | `is_anomalous = 0` in DB after a normal exit |
| `test_is_anomalous_flag_stored_anomaly` | `is_anomalous = 1` in DB after an anomalous exit |
| `test_orphan_exit_when_no_open_visit` | Exit with no prior entry creates an orphan record with a warning |
| `test_weight_exit_stored` | `weight_exit_g` is persisted |
| `test_waste_weight_derived` | `waste_weight_g = weight_exit_g − weight_pre_g` is computed and stored |
| `test_peak_gas_reconciled` | Final `ammonia_peak_ppb` is `MAX(entry_reading, exit_reading)` |
| `test_missing_exit_image_returns_error` | Non-existent path returns "Error" |

---

### TestConfirmIdentity — 5 tests, 5 passed

Verifies `agent.confirm_identity(visit_id, cat_name)`.

| Test | What it checks |
|---|---|
| `test_success_message` | Return value contains "confirmed" |
| `test_sets_is_confirmed` | `is_confirmed = 1` is written to the DB |
| `test_unknown_cat_returns_error` | Unregistered cat name returns "Error" |
| `test_invalid_visit_id_returns_error` | Non-existent visit ID returns "Error" |
| `test_result_includes_cat_name` | Cat name appears in the return string |

---

### TestRetroactiveRecognition — 4 tests, 4 passed

Verifies `agent.retroactive_recognition(cat_name, since_date)`.

| Test | What it checks |
|---|---|
| `test_no_unknown_visits_returns_graceful` | Returns a graceful message when no unknown visits exist |
| `test_unknown_cat_returns_error` | Unregistered cat name returns "Error" |
| `test_invalid_date_returns_error` | Non-ISO date string returns an error |
| `test_summary_includes_cat_name` | Cat name appears in the summary string |

---

### TestListCats — 4 tests, 4 passed

Verifies `agent.list_cats()`.

| Test | What it checks |
|---|---|
| `test_empty_database` | "No cats" message when DB is empty |
| `test_shows_registered_cat` | Cat name appears after registration |
| `test_shows_image_count` | Reference image count is reflected in the output |
| `test_multiple_cats_all_listed` | Two registered cats both appear in the output |

---

### TestGetVisitsByDate — 4 tests, 4 passed

Verifies `agent.get_visits_by_date(date_str)`.

| Test | What it checks |
|---|---|
| `test_no_visits` | "No visits" when DB is empty for that date |
| `test_returns_visit_on_correct_date` | Visit on matching date appears in output |
| `test_wrong_date_returns_no_visits` | Visit on a different date does not appear |
| `test_anomalous_visit_flagged` | Anomalous visit includes the ⚠️ flag in output |

---

### TestGetVisitsByCat — 4 tests, 4 passed

Verifies `agent.get_visits_by_cat(cat_name)`.

| Test | What it checks |
|---|---|
| `test_unknown_cat` | "No cat / not found" for an unregistered name |
| `test_cat_with_no_visits` | "No visits" for a registered cat with no visit history |
| `test_returns_confirmed_visit` | Confirmed visit status is reflected in the output |
| `test_shows_visit_count` | Visit count is included in the output |

---

### TestGetAnomalousVisits — 4 tests, 4 passed

Verifies `agent.get_anomalous_visits()`.

| Test | What it checks |
|---|---|
| `test_no_anomalies` | "No anomalous" when nothing is flagged |
| `test_lists_anomalous_visit` | Flagged visits appear in the output |
| `test_normal_visit_not_listed` | Non-anomalous visits are excluded |
| `test_count_reflects_multiple` | Count is correct with multiple anomalies |

---

### TestGetUnconfirmedVisits — 4 tests, 4 passed

Verifies `agent.get_unconfirmed_visits()`.

| Test | What it checks |
|---|---|
| `test_empty_database` | "No unconfirmed" when DB is empty |
| `test_lists_unconfirmed` | Tentative visits are listed |
| `test_confirmed_not_listed` | A confirmed visit disappears from the list |
| `test_count_reflects_multiple` | Count is correct with three unconfirmed visits |

---

### TestGetVisitImages — 4 tests, 4 passed

Verifies `agent.get_visit_images(visit_id)`.

| Test | What it checks |
|---|---|
| `test_shows_entry_and_exit_labels` | "Entry" and "Exit" labels are present |
| `test_invalid_visit_id_not_found` | Invalid ID returns "not found" |
| `test_result_includes_visit_id` | Visit number appears in the output |
| `test_shows_exit_path_after_exit_recorded` | Exit path is non-empty after `record_exit` |

---

### TestQuery — 5 tests, 5 passed

Verifies `agent.query(message, thread_id)` — the natural-language path through the full LangGraph agent. The LangGraph agent is replaced with a minimal fake object for all tests in this class.

| Test | What it checks |
|---|---|
| `test_returns_string` | Return value is always a `str` |
| `test_returns_agent_content` | AI message content appears in the return string |
| `test_custom_thread_id_accepted` | A custom `thread_id` is accepted without error |
| `test_default_thread_id_is_api` | The default `thread_id` passed to the agent's `invoke()` is `"api"` |
| `test_tool_message_included_in_output` | Tool result messages are forwarded into the return string alongside AI messages |

---

### TestSensorRoundTrip — 5 tests, 5 passed

End-to-end cycles through the full API surface without making any LLM calls.

| Test | What it checks |
|---|---|
| `test_full_cycle_no_sensors` | Entry then exit both succeed without sensor data |
| `test_full_cycle_with_sensors` | Entry + exit with full sensor data; verifies `cat_weight_g`, `waste_weight_g`, and peak gas reconciliation in the DB |
| `test_anomalous_visit_appears_in_query` | After an anomalous exit, `get_anomalous_visits()` returns a non-empty result |
| `test_register_then_entry_then_query` | Register cat → record entry → query by date; all three steps consistent |
| `test_multiple_entries_separate_visits` | Two `record_entry` calls create two independent visit rows |

---

## Bugs Found and Fixed During Testing

Two defects in `src/litterbox/api.py` were discovered by the test suite and corrected before the final run.

### Bug 1 — StructuredTool objects not directly callable

**Symptom:** `TypeError: 'StructuredTool' object is not callable` on every method that delegated to a `@tool`-decorated function.

**Root cause:** LangChain's `@tool` decorator wraps the function as a `StructuredTool` instance, which is not callable with the original Python keyword-argument signature. The API methods were calling the tools as if they were plain functions, e.g. `register_cat_image(image_path=..., cat_name=...)`.

**Fix:** All 11 tool delegation calls were changed to use `tool.invoke({...})` with a keyword-argument dict, which is the correct LangChain API for calling a tool programmatically.

**Tests that caught it:** All methods in `TestRegisterCat`, `TestRecordEntry`, `TestRecordExit`, `TestConfirmIdentity`, `TestRetroactiveRecognition`, `TestListCats`, `TestGetVisitsByDate`, `TestGetVisitsByCat`, `TestGetAnomalousVisits`, `TestGetUnconfirmedVisits`, `TestGetVisitImages` (55 tests).

---

### Bug 2 — Wrong fixture used in database assertion

**Symptom:** `AttributeError: 'FixtureFunctionDefinition' object has no attribute '_data_path'` in `TestRecordExit.test_is_anomalous_flag_stored_anomaly`.

**Root cause:** The test used the `agent_anomaly` fixture to call `record_exit`, but then passed the bare name `agent` (a pytest fixture function reference, not an instance) to `_direct_db()` when querying the database. The two fixtures write to different `tmp_path` directories, so the query would have been against an empty database even if the reference had been valid.

**Fix:** Changed `_direct_db(agent)` to `_direct_db(agent_anomaly)` in that test.

**Tests that caught it:** `TestRecordExit::test_is_anomalous_flag_stored_anomaly`.

---

## Warnings

45 warnings were emitted; all are informational and do not affect correctness.

| Warning source | Count | Description |
|---|---|---|
| `transformers` `BPE.__init__` | 1 | Deprecated internal API in the CLIP tokenizer library. Not our code; no action required. |
| `datetime.utcnow()` | 44 | Python 3.12 deprecation of `datetime.utcnow()` in `src/litterbox/tools.py`. The function still works correctly; migration to `datetime.now(UTC)` is a future housekeeping item. |

---

## Regression Check

The full non-slow suite (`pytest -m "not slow"`) was run after completing `test_api.py` to verify no regressions were introduced by the new API module or `__init__.py` export.

```
267 passed, 20 deselected (slow), 153 warnings in 9.95 s
```

The 20 deselected tests are the CLIP embedding tests (`test_embeddings.py`, marked `slow`) which require a ~350 MB model download and are excluded from the standard CI run.

---

## How to Reproduce

```bash
conda activate langchain_env_2026_1

# API tests only
pytest tests/test_api.py -v

# Full non-slow suite (includes API tests)
pytest -m "not slow"

# Everything including CLIP
pytest
```

No `.env` file or API keys are required to run this suite.
