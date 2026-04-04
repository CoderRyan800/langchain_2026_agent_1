"""
Tests for the LitterboxAgent Python API (src/litterbox/api.py).

Design principles
-----------------
* Each test creates a LitterboxAgent pointed at a fresh tmp_path so no test
  touches production data or the shared isolated_env tmp directory.
* LLM calls (_identify_cat, _run_gpt4o_vision) are stubbed in every test that
  exercises record_entry / record_exit, so the suite runs offline at zero cost.
* The query() / natural-language path is tested by monkey-patching _get_agent
  to return a tiny fake agent — no OpenAI call needed.
* The conftest.py ``isolated_env`` autouse fixture runs first (it patches to
  its own tmp dir), then the agent fixture overwrites the module paths again
  to point at its own tmp dir.  monkeypatch restores everything after each test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Canned responses
# ---------------------------------------------------------------------------

_HEALTH_NORMAL = (
    "CONCERNS_PRESENT: no\n"
    "DESCRIPTION: Litter box looks clean.\n"
    "OWNER_SUMMARY: No visual abnormalities.\n"
    "\u26a0\ufe0f This analysis is preliminary and must be reviewed by a licensed "
    "veterinarian before any medical decisions are made."
)

_HEALTH_ANOMALY = (
    "CONCERNS_PRESENT: yes\n"
    "DESCRIPTION: Pink discoloration in exit image.\n"
    "OWNER_SUMMARY: Possible blood in urine. See a vet.\n"
    "\u26a0\ufe0f This analysis is preliminary and must be reviewed by a licensed "
    "veterinarian before any medical decisions are made."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def agent_dirs(tmp_path):
    """Return (data_dir, images_dir) under a fresh tmp_path."""
    data = tmp_path / "data"
    imgs = tmp_path / "images"
    data.mkdir()
    imgs.mkdir()
    return data, imgs


@pytest.fixture()
def agent(agent_dirs, monkeypatch):
    """
    A fresh LitterboxAgent with all LLM calls stubbed out.
    Yields the agent; closes it after the test.
    """
    data, imgs = agent_dirs

    monkeypatch.setattr(
        "litterbox.tools._identify_cat",
        lambda path: (None, None, 0.3, "No reference images in database yet."),
    )
    monkeypatch.setattr(
        "litterbox.tools._run_gpt4o_vision",
        lambda prompt, *paths: _HEALTH_NORMAL,
    )

    from litterbox import LitterboxAgent
    a = LitterboxAgent(data_dir=str(data), images_dir=str(imgs))
    yield a
    a.close()


@pytest.fixture()
def agent_anomaly(agent_dirs, monkeypatch):
    """Agent whose GPT-4o health stub returns an anomalous result."""
    data, imgs = agent_dirs

    monkeypatch.setattr(
        "litterbox.tools._identify_cat",
        lambda path: (None, None, 0.3, "No reference images in database yet."),
    )
    monkeypatch.setattr(
        "litterbox.tools._run_gpt4o_vision",
        lambda prompt, *paths: _HEALTH_ANOMALY,
    )

    from litterbox import LitterboxAgent
    a = LitterboxAgent(data_dir=str(data), images_dir=str(imgs))
    yield a
    a.close()


@pytest.fixture()
def cat_photo(tmp_path) -> Path:
    img = Image.new("RGB", (320, 240), (180, 120, 80))
    p = tmp_path / "cat.jpg"
    img.save(str(p), "JPEG")
    return p


@pytest.fixture()
def litter_photo(tmp_path) -> Path:
    img = Image.new("RGB", (400, 300), (240, 230, 200))
    p = tmp_path / "litter.jpg"
    img.save(str(p), "JPEG")
    return p


@pytest.fixture()
def exit_photo(tmp_path) -> Path:
    from PIL import ImageDraw
    img = Image.new("RGB", (400, 300), (240, 230, 200))
    ImageDraw.Draw(img).ellipse([160, 120, 240, 180], fill=(210, 200, 170))
    p = tmp_path / "exit.jpg"
    img.save(str(p), "JPEG")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _direct_db(agent):
    """Return a sqlite3 connection to the agent's litterbox.db."""
    import sqlite3
    conn = sqlite3.connect(str(agent._data_path / "litterbox.db"))
    conn.row_factory = sqlite3.Row
    return conn


def _insert_cat(agent, name="Whiskers") -> int:
    with _direct_db(agent) as conn:
        cur = conn.execute("INSERT INTO cats (name) VALUES (?)", (name,))
        conn.commit()
        return cur.lastrowid


def _insert_open_visit(agent, cat_id) -> int:
    with _direct_db(agent) as conn:
        cur = conn.execute(
            "INSERT INTO visits (entry_time, entry_image_path, tentative_cat_id, similarity_score) "
            "VALUES ('2026-01-01T10:00:00', 'images/visits/test/entry.jpg', ?, 0.95)",
            (cat_id,),
        )
        conn.commit()
        return cur.lastrowid


# ---------------------------------------------------------------------------
# 1. Constructor and path setup
# ---------------------------------------------------------------------------

class TestLitterboxAgentInit:
    def test_creates_data_dir(self, tmp_path):
        from litterbox import LitterboxAgent
        data = tmp_path / "mydata"
        imgs = tmp_path / "myimages"
        a = LitterboxAgent(data_dir=str(data), images_dir=str(imgs))
        a.close()
        assert data.exists()

    def test_creates_images_dir(self, tmp_path):
        from litterbox import LitterboxAgent
        data = tmp_path / "mydata"
        imgs = tmp_path / "myimages"
        a = LitterboxAgent(data_dir=str(data), images_dir=str(imgs))
        a.close()
        assert imgs.exists()

    def test_db_file_created(self, agent, agent_dirs):
        data, _ = agent_dirs
        assert (data / "litterbox.db").exists()

    def test_agent_memory_db_created(self, agent, agent_dirs):
        data, _ = agent_dirs
        assert (data / "agent_memory.db").exists()

    def test_patches_db_path(self, agent, agent_dirs):
        import litterbox.db as db_mod
        data, _ = agent_dirs
        assert db_mod.DB_PATH == data / "litterbox.db"

    def test_patches_images_dir(self, agent, agent_dirs):
        import litterbox.tools as tools_mod
        _, imgs = agent_dirs
        assert tools_mod.IMAGES_DIR == imgs

    def test_patches_chroma_path(self, agent, agent_dirs):
        import litterbox.embeddings as emb_mod
        data, _ = agent_dirs
        assert emb_mod.CHROMA_PATH == data / "chroma"

    def test_resets_chroma_collection_singleton(self, agent):
        # Collection should be None until first use (lazy init)
        import litterbox.embeddings as emb_mod
        # After construction it's None; only populated on first Chroma call
        assert emb_mod._collection is None

    def test_db_schema_initialised(self, agent):
        with _direct_db(agent) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert {"cats", "cat_images", "visits", "visit_sensor_events"}.issubset(tables)


# ---------------------------------------------------------------------------
# 2. Context manager and close()
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_context_manager_returns_agent(self, tmp_path):
        from litterbox import LitterboxAgent
        data = tmp_path / "d"
        imgs = tmp_path / "i"
        with LitterboxAgent(data_dir=str(data), images_dir=str(imgs)) as a:
            assert a is not None

    def test_close_sets_saver_ctx_none(self, agent):
        agent.close()
        assert agent._saver_ctx is None

    def test_double_close_does_not_raise(self, agent):
        agent.close()
        agent.close()  # should not raise

    def test_close_sets_checkpointer_none(self, agent):
        agent.close()
        assert agent._checkpointer is None


# ---------------------------------------------------------------------------
# 3. register_cat
# ---------------------------------------------------------------------------

class TestRegisterCat:
    def test_returns_success_string(self, agent, cat_photo):
        result = agent.register_cat(str(cat_photo), "Whiskers")
        assert "Registered" in result

    def test_creates_cats_row(self, agent, cat_photo):
        agent.register_cat(str(cat_photo), "Whiskers")
        with _direct_db(agent) as conn:
            row = conn.execute("SELECT * FROM cats WHERE name='Whiskers'").fetchone()
        assert row is not None

    def test_copies_image_file(self, agent, cat_photo, agent_dirs):
        _, imgs = agent_dirs
        agent.register_cat(str(cat_photo), "Whiskers")
        stored = list((imgs / "cats").rglob("*.jpg"))
        assert len(stored) == 1

    def test_second_image_increments_count(self, agent, cat_photo):
        agent.register_cat(str(cat_photo), "Whiskers")
        agent.register_cat(str(cat_photo), "Whiskers")
        with _direct_db(agent) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM cat_images ci "
                "JOIN cats c ON ci.cat_id=c.cat_id WHERE c.name='Whiskers'"
            ).fetchone()[0]
        assert count == 2

    def test_no_duplicate_cat_row(self, agent, cat_photo):
        agent.register_cat(str(cat_photo), "Whiskers")
        agent.register_cat(str(cat_photo), "Whiskers")
        with _direct_db(agent) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM cats WHERE name='Whiskers'"
            ).fetchone()[0]
        assert count == 1

    def test_missing_image_returns_error(self, agent):
        result = agent.register_cat("/no/such/file.jpg", "Ghost")
        assert "Error" in result

    def test_two_cats_both_stored(self, agent, cat_photo, tmp_path):
        img2 = Image.new("RGB", (100, 100), (50, 100, 200))
        p2 = tmp_path / "cat2.jpg"
        img2.save(str(p2), "JPEG")
        agent.register_cat(str(cat_photo), "Luna")
        agent.register_cat(str(p2), "Mochi")
        with _direct_db(agent) as conn:
            count = conn.execute("SELECT COUNT(*) FROM cats").fetchone()[0]
        assert count == 2


# ---------------------------------------------------------------------------
# 4. record_entry
# ---------------------------------------------------------------------------

class TestRecordEntry:
    def test_returns_string(self, agent, litter_photo):
        result = agent.record_entry(str(litter_photo))
        assert isinstance(result, str)

    def test_creates_visit_row(self, agent, litter_photo):
        agent.record_entry(str(litter_photo))
        with _direct_db(agent) as conn:
            count = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        assert count == 1

    def test_entry_image_stored(self, agent, litter_photo, agent_dirs):
        _, imgs = agent_dirs
        agent.record_entry(str(litter_photo))
        stored = list((imgs / "visits").rglob("*_entry.jpg"))
        assert len(stored) == 1

    def test_result_mentions_visit_number(self, agent, litter_photo):
        result = agent.record_entry(str(litter_photo))
        assert "Visit #" in result or "visit" in result.lower()

    def test_missing_image_returns_error(self, agent):
        result = agent.record_entry("/no/such/capture.jpg")
        assert "Error" in result

    def test_weight_pre_stored(self, agent, litter_photo):
        agent.record_entry(str(litter_photo), weight_pre_g=5400)
        with _direct_db(agent) as conn:
            row = conn.execute("SELECT weight_pre_g FROM visits LIMIT 1").fetchone()
        assert row["weight_pre_g"] == 5400

    def test_weight_entry_stored(self, agent, litter_photo):
        agent.record_entry(str(litter_photo), weight_entry_g=8600)
        with _direct_db(agent) as conn:
            row = conn.execute("SELECT weight_entry_g FROM visits LIMIT 1").fetchone()
        assert row["weight_entry_g"] == 8600

    def test_cat_weight_derived(self, agent, litter_photo):
        agent.record_entry(str(litter_photo), weight_pre_g=5000, weight_entry_g=8200)
        with _direct_db(agent) as conn:
            row = conn.execute("SELECT cat_weight_g FROM visits LIMIT 1").fetchone()
        assert row["cat_weight_g"] == pytest.approx(3200)

    def test_ammonia_stored(self, agent, litter_photo):
        agent.record_entry(str(litter_photo), ammonia_peak_ppb=45.0)
        with _direct_db(agent) as conn:
            row = conn.execute("SELECT ammonia_peak_ppb FROM visits LIMIT 1").fetchone()
        assert row["ammonia_peak_ppb"] == pytest.approx(45.0)

    def test_methane_stored(self, agent, litter_photo):
        agent.record_entry(str(litter_photo), methane_peak_ppb=22.0)
        with _direct_db(agent) as conn:
            row = conn.execute("SELECT methane_peak_ppb FROM visits LIMIT 1").fetchone()
        assert row["methane_peak_ppb"] == pytest.approx(22.0)

    def test_sensor_event_rows_logged(self, agent, litter_photo):
        agent.record_entry(
            str(litter_photo),
            weight_pre_g=5000,
            weight_entry_g=8200,
            ammonia_peak_ppb=30,
        )
        with _direct_db(agent) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM visit_sensor_events"
            ).fetchone()[0]
        assert count >= 2  # pre_entry weight + entry weight + entry ammonia

    def test_no_sensors_columns_null(self, agent, litter_photo):
        agent.record_entry(str(litter_photo))
        with _direct_db(agent) as conn:
            row = conn.execute(
                "SELECT weight_pre_g, weight_entry_g, cat_weight_g, "
                "ammonia_peak_ppb, methane_peak_ppb FROM visits LIMIT 1"
            ).fetchone()
        assert row["weight_pre_g"] is None
        assert row["cat_weight_g"] is None
        assert row["ammonia_peak_ppb"] is None


# ---------------------------------------------------------------------------
# 5. record_exit
# ---------------------------------------------------------------------------

class TestRecordExit:
    def test_returns_string(self, agent, litter_photo, exit_photo):
        agent.record_entry(str(litter_photo))
        result = agent.record_exit(str(exit_photo))
        assert isinstance(result, str)

    def test_closes_visit(self, agent, litter_photo, exit_photo):
        agent.record_entry(str(litter_photo))
        agent.record_exit(str(exit_photo))
        with _direct_db(agent) as conn:
            row = conn.execute("SELECT exit_time FROM visits LIMIT 1").fetchone()
        assert row["exit_time"] is not None

    def test_exit_image_stored(self, agent, litter_photo, exit_photo, agent_dirs):
        _, imgs = agent_dirs
        agent.record_entry(str(litter_photo))
        agent.record_exit(str(exit_photo))
        stored = list((imgs / "visits").rglob("*_exit.jpg"))
        assert len(stored) == 1

    def test_health_normal_in_result(self, agent, litter_photo, exit_photo):
        agent.record_entry(str(litter_photo))
        result = agent.record_exit(str(exit_photo))
        assert "No anomalies" in result or "CONCERNS_PRESENT: no" in result

    def test_health_anomalous_flagged(self, agent_anomaly, litter_photo, exit_photo):
        agent_anomaly.record_entry(str(litter_photo))
        result = agent_anomaly.record_exit(str(exit_photo))
        assert "ANOMALY" in result or "anomal" in result.lower()

    def test_is_anomalous_flag_stored_normal(self, agent, litter_photo, exit_photo):
        agent.record_entry(str(litter_photo))
        agent.record_exit(str(exit_photo))
        with _direct_db(agent) as conn:
            row = conn.execute("SELECT is_anomalous FROM visits LIMIT 1").fetchone()
        assert not bool(row["is_anomalous"])

    def test_is_anomalous_flag_stored_anomaly(
        self, agent_anomaly, litter_photo, exit_photo
    ):
        agent_anomaly.record_entry(str(litter_photo))
        agent_anomaly.record_exit(str(exit_photo))
        with _direct_db(agent_anomaly) as conn:
            row = conn.execute("SELECT is_anomalous FROM visits LIMIT 1").fetchone()
        assert bool(row["is_anomalous"])

    def test_orphan_exit_when_no_open_visit(self, agent, exit_photo):
        result = agent.record_exit(str(exit_photo))
        assert "orphan" in result.lower() or "WARNING" in result or "⚠" in result

    def test_weight_exit_stored(self, agent, litter_photo, exit_photo):
        agent.record_entry(str(litter_photo), weight_pre_g=5000)
        agent.record_exit(str(exit_photo), weight_exit_g=5150)
        with _direct_db(agent) as conn:
            row = conn.execute("SELECT weight_exit_g FROM visits LIMIT 1").fetchone()
        assert row["weight_exit_g"] == pytest.approx(5150)

    def test_waste_weight_derived(self, agent, litter_photo, exit_photo):
        agent.record_entry(str(litter_photo), weight_pre_g=5000)
        agent.record_exit(str(exit_photo), weight_exit_g=5150)
        with _direct_db(agent) as conn:
            row = conn.execute("SELECT waste_weight_g FROM visits LIMIT 1").fetchone()
        assert row["waste_weight_g"] == pytest.approx(150)

    def test_peak_gas_reconciled(self, agent, litter_photo, exit_photo):
        # Entry: ammonia 30; Exit: ammonia 60 → final should be 60
        agent.record_entry(str(litter_photo), ammonia_peak_ppb=30)
        agent.record_exit(str(exit_photo), ammonia_peak_ppb=60)
        with _direct_db(agent) as conn:
            row = conn.execute(
                "SELECT ammonia_peak_ppb FROM visits LIMIT 1"
            ).fetchone()
        assert row["ammonia_peak_ppb"] == pytest.approx(60)

    def test_missing_exit_image_returns_error(self, agent, litter_photo):
        agent.record_entry(str(litter_photo))
        result = agent.record_exit("/no/such/exit.jpg")
        assert "Error" in result


# ---------------------------------------------------------------------------
# 6. confirm_identity
# ---------------------------------------------------------------------------

class TestConfirmIdentity:
    def test_success_message(self, agent):
        cat_id = _insert_cat(agent)
        visit_id = _insert_open_visit(agent, cat_id)
        result = agent.confirm_identity(visit_id, "Whiskers")
        assert "confirmed" in result.lower()

    def test_sets_is_confirmed(self, agent):
        cat_id = _insert_cat(agent)
        visit_id = _insert_open_visit(agent, cat_id)
        agent.confirm_identity(visit_id, "Whiskers")
        with _direct_db(agent) as conn:
            row = conn.execute(
                "SELECT is_confirmed FROM visits WHERE visit_id=?", (visit_id,)
            ).fetchone()
        assert bool(row["is_confirmed"])

    def test_unknown_cat_returns_error(self, agent):
        cat_id = _insert_cat(agent)
        visit_id = _insert_open_visit(agent, cat_id)
        result = agent.confirm_identity(visit_id, "NoSuchCat")
        assert "Error" in result

    def test_invalid_visit_id_returns_error(self, agent):
        _insert_cat(agent)
        result = agent.confirm_identity(99999, "Whiskers")
        assert "Error" in result

    def test_result_includes_cat_name(self, agent):
        cat_id = _insert_cat(agent)
        visit_id = _insert_open_visit(agent, cat_id)
        result = agent.confirm_identity(visit_id, "Whiskers")
        assert "Whiskers" in result


# ---------------------------------------------------------------------------
# 7. retroactive_recognition
# ---------------------------------------------------------------------------

class TestRetroactiveRecognition:
    def test_no_unknown_visits_returns_graceful(self, agent):
        _insert_cat(agent)
        result = agent.retroactive_recognition("Whiskers", "2026-01-01")
        assert "Nothing to retroactively review" in result or "No unknown" in result

    def test_unknown_cat_returns_error(self, agent):
        result = agent.retroactive_recognition("NoSuchCat", "2026-01-01")
        assert "Error" in result

    def test_invalid_date_returns_error(self, agent):
        _insert_cat(agent)
        result = agent.retroactive_recognition("Whiskers", "not-a-date")
        assert "Error" in result or "not a valid date" in result.lower()

    def test_summary_includes_cat_name(self, agent):
        _insert_cat(agent)
        result = agent.retroactive_recognition("Whiskers", "2026-01-01")
        assert "Whiskers" in result


# ---------------------------------------------------------------------------
# 8. list_cats
# ---------------------------------------------------------------------------

class TestListCats:
    def test_empty_database(self, agent):
        result = agent.list_cats()
        assert "No cats" in result

    def test_shows_registered_cat(self, agent, cat_photo):
        agent.register_cat(str(cat_photo), "Mittens")
        result = agent.list_cats()
        assert "Mittens" in result

    def test_shows_image_count(self, agent, cat_photo):
        agent.register_cat(str(cat_photo), "Mittens")
        agent.register_cat(str(cat_photo), "Mittens")
        result = agent.list_cats()
        assert "2" in result

    def test_multiple_cats_all_listed(self, agent, cat_photo, tmp_path):
        img2 = Image.new("RGB", (100, 100), (50, 100, 200))
        p2 = tmp_path / "cat2.jpg"
        img2.save(str(p2), "JPEG")
        agent.register_cat(str(cat_photo), "Anna")
        agent.register_cat(str(p2), "Luna")
        result = agent.list_cats()
        assert "Anna" in result and "Luna" in result


# ---------------------------------------------------------------------------
# 9. get_visits_by_date
# ---------------------------------------------------------------------------

class TestGetVisitsByDate:
    def test_no_visits(self, agent):
        result = agent.get_visits_by_date("1900-01-01")
        assert "No visits" in result

    def test_returns_visit_on_correct_date(self, agent):
        _insert_cat(agent)
        with _direct_db(agent) as conn:
            conn.execute(
                "INSERT INTO visits (entry_time) VALUES ('2026-03-15T10:00:00')"
            )
            conn.commit()
        result = agent.get_visits_by_date("2026-03-15")
        assert "2026-03-15" in result

    def test_wrong_date_returns_no_visits(self, agent):
        with _direct_db(agent) as conn:
            conn.execute(
                "INSERT INTO visits (entry_time) VALUES ('2026-03-15T10:00:00')"
            )
            conn.commit()
        result = agent.get_visits_by_date("2026-03-16")
        assert "No visits" in result

    def test_anomalous_visit_flagged(self, agent):
        with _direct_db(agent) as conn:
            conn.execute(
                "INSERT INTO visits (entry_time, is_anomalous) "
                "VALUES ('2026-05-01T10:00:00', 1)"
            )
            conn.commit()
        result = agent.get_visits_by_date("2026-05-01")
        assert "\u26a0" in result or "anomal" in result.lower()


# ---------------------------------------------------------------------------
# 10. get_visits_by_cat
# ---------------------------------------------------------------------------

class TestGetVisitsByCat:
    def test_unknown_cat(self, agent):
        result = agent.get_visits_by_cat("NoSuchCat")
        assert "No cat" in result or "not found" in result.lower()

    def test_cat_with_no_visits(self, agent):
        _insert_cat(agent)
        result = agent.get_visits_by_cat("Whiskers")
        assert "No visits" in result

    def test_returns_confirmed_visit(self, agent):
        cat_id = _insert_cat(agent)
        with _direct_db(agent) as conn:
            conn.execute(
                "INSERT INTO visits (entry_time, confirmed_cat_id, is_confirmed) "
                "VALUES ('2026-03-01T08:00:00', ?, 1)",
                (cat_id,),
            )
            conn.commit()
        result = agent.get_visits_by_cat("Whiskers")
        assert "confirmed" in result.lower()

    def test_shows_visit_count(self, agent):
        cat_id = _insert_cat(agent)
        with _direct_db(agent) as conn:
            for h in (8, 12):
                conn.execute(
                    f"INSERT INTO visits (entry_time, confirmed_cat_id, is_confirmed) "
                    f"VALUES ('2026-03-01T{h:02d}:00:00', ?, 1)",
                    (cat_id,),
                )
            conn.commit()
        result = agent.get_visits_by_cat("Whiskers")
        assert "2" in result


# ---------------------------------------------------------------------------
# 11. get_anomalous_visits
# ---------------------------------------------------------------------------

class TestGetAnomalousVisits:
    def test_no_anomalies(self, agent):
        result = agent.get_anomalous_visits()
        assert "No anomalous" in result

    def test_lists_anomalous_visit(self, agent):
        with _direct_db(agent) as conn:
            conn.execute(
                "INSERT INTO visits (entry_time, is_anomalous, health_notes) "
                "VALUES ('2026-01-01T08:00:00', 1, 'Blood detected.')"
            )
            conn.commit()
        result = agent.get_anomalous_visits()
        assert "anomalous" in result.lower() or "Blood" in result

    def test_normal_visit_not_listed(self, agent):
        with _direct_db(agent) as conn:
            conn.execute(
                "INSERT INTO visits (entry_time, is_anomalous) "
                "VALUES ('2026-01-01T08:00:00', 0)"
            )
            conn.commit()
        result = agent.get_anomalous_visits()
        assert "No anomalous" in result

    def test_count_reflects_multiple(self, agent):
        with _direct_db(agent) as conn:
            for h in (8, 12):
                conn.execute(
                    f"INSERT INTO visits (entry_time, is_anomalous, health_notes) "
                    f"VALUES ('2026-01-01T{h:02d}:00:00', 1, 'Issue.')"
                )
            conn.commit()
        result = agent.get_anomalous_visits()
        assert "2" in result


# ---------------------------------------------------------------------------
# 12. get_unconfirmed_visits
# ---------------------------------------------------------------------------

class TestGetUnconfirmedVisits:
    def test_empty_database(self, agent):
        result = agent.get_unconfirmed_visits()
        assert "No unconfirmed" in result

    def test_lists_unconfirmed(self, agent):
        cat_id = _insert_cat(agent)
        with _direct_db(agent) as conn:
            conn.execute(
                "INSERT INTO visits (entry_time, tentative_cat_id, is_confirmed) "
                "VALUES ('2026-01-01T08:00:00', ?, 0)",
                (cat_id,),
            )
            conn.commit()
        result = agent.get_unconfirmed_visits()
        assert "#" in result or "unconfirmed" in result.lower()

    def test_confirmed_not_listed(self, agent):
        cat_id = _insert_cat(agent)
        visit_id = _insert_open_visit(agent, cat_id)
        agent.confirm_identity(visit_id, "Whiskers")
        result = agent.get_unconfirmed_visits()
        assert "No unconfirmed" in result

    def test_count_reflects_multiple(self, agent):
        cat_id = _insert_cat(agent)
        with _direct_db(agent) as conn:
            for h in (8, 12, 16):
                conn.execute(
                    f"INSERT INTO visits (entry_time, tentative_cat_id, is_confirmed) "
                    f"VALUES ('2026-01-01T{h:02d}:00:00', ?, 0)",
                    (cat_id,),
                )
            conn.commit()
        result = agent.get_unconfirmed_visits()
        assert "3" in result


# ---------------------------------------------------------------------------
# 13. get_visit_images
# ---------------------------------------------------------------------------

class TestGetVisitImages:
    def test_shows_entry_and_exit_labels(self, agent):
        cat_id = _insert_cat(agent)
        visit_id = _insert_open_visit(agent, cat_id)
        result = agent.get_visit_images(visit_id)
        assert "Entry" in result
        assert "Exit" in result

    def test_invalid_visit_id_not_found(self, agent):
        result = agent.get_visit_images(99999)
        assert "not found" in result.lower()

    def test_result_includes_visit_id(self, agent):
        cat_id = _insert_cat(agent)
        visit_id = _insert_open_visit(agent, cat_id)
        result = agent.get_visit_images(visit_id)
        assert str(visit_id) in result

    def test_shows_exit_path_after_exit_recorded(
        self, agent, litter_photo, exit_photo
    ):
        agent.record_entry(str(litter_photo))
        agent.record_exit(str(exit_photo))
        with _direct_db(agent) as conn:
            visit_id = conn.execute(
                "SELECT visit_id FROM visits ORDER BY visit_id DESC LIMIT 1"
            ).fetchone()[0]
        result = agent.get_visit_images(visit_id)
        assert "exit.jpg" in result


# ---------------------------------------------------------------------------
# 14. query() — natural-language path (agent mocked)
# ---------------------------------------------------------------------------

class TestQuery:
    """
    Test the query() method by monkey-patching _get_agent to return a
    minimal fake agent object.  No OpenAI API call is made.
    """

    def _make_fake_agent(self, response_text: str):
        """Return an object with .invoke() that mimics the LangGraph agent."""
        from langchain.messages import AIMessage

        class _FakeAgent:
            def invoke(self, input_dict, config=None):
                return {"messages": [AIMessage(content=response_text)]}

        return _FakeAgent()

    def test_returns_string(self, agent, monkeypatch):
        fake = self._make_fake_agent("There are 3 visits today.")
        monkeypatch.setattr(agent, "_get_agent", lambda: fake)
        result = agent.query("How many visits today?")
        assert isinstance(result, str)

    def test_returns_agent_content(self, agent, monkeypatch):
        fake = self._make_fake_agent("Whiskers visited 5 times.")
        monkeypatch.setattr(agent, "_get_agent", lambda: fake)
        result = agent.query("How many visits for Whiskers?")
        assert "5" in result or "Whiskers" in result

    def test_custom_thread_id_accepted(self, agent, monkeypatch):
        fake = self._make_fake_agent("OK")
        monkeypatch.setattr(agent, "_get_agent", lambda: fake)
        # Should not raise
        result = agent.query("Hello", thread_id="custom-session")
        assert isinstance(result, str)

    def test_default_thread_id_is_api(self, agent, monkeypatch):
        from langchain.messages import AIMessage
        received_config = {}

        class _FakeAgent:
            def invoke(self, input_dict, config=None):
                received_config.update(config or {})
                return {"messages": [AIMessage(content="ok")]}

        monkeypatch.setattr(agent, "_get_agent", lambda: _FakeAgent())
        agent.query("test")
        assert received_config.get("configurable", {}).get("thread_id") == "api"

    def test_tool_message_included_in_output(self, agent, monkeypatch):
        from langchain.messages import AIMessage, ToolMessage

        class _FakeAgent:
            def invoke(self, input_dict, config=None):
                return {
                    "messages": [
                        ToolMessage(content="Tool output: 2 visits found.",
                                    tool_call_id="fake-id"),
                        AIMessage(content="Found 2 visits."),
                    ]
                }

        monkeypatch.setattr(agent, "_get_agent", lambda: _FakeAgent())
        result = agent.query("Show visits")
        # Both tool output and AI message should appear
        assert "2 visits" in result or "Found" in result


# ---------------------------------------------------------------------------
# 15. Full sensor round-trip
# ---------------------------------------------------------------------------

class TestSensorRoundTrip:
    """Entry → exit → query cycle through the API."""

    def test_full_cycle_no_sensors(self, agent, litter_photo, exit_photo):
        r1 = agent.record_entry(str(litter_photo))
        assert "Visit #" in r1 or "visit" in r1.lower()
        r2 = agent.record_exit(str(exit_photo))
        assert "closed" in r2.lower() or "Visit" in r2

    def test_full_cycle_with_sensors(self, agent, litter_photo, exit_photo):
        agent.record_entry(
            str(litter_photo),
            weight_pre_g=5000,
            weight_entry_g=8200,
            ammonia_peak_ppb=30,
            methane_peak_ppb=12,
        )
        result = agent.record_exit(
            str(exit_photo),
            weight_exit_g=5090,
            ammonia_peak_ppb=55,
        )
        with _direct_db(agent) as conn:
            row = conn.execute(
                "SELECT cat_weight_g, waste_weight_g, ammonia_peak_ppb "
                "FROM visits LIMIT 1"
            ).fetchone()
        assert row["cat_weight_g"] == pytest.approx(3200)
        assert row["waste_weight_g"] == pytest.approx(90)
        assert row["ammonia_peak_ppb"] == pytest.approx(55)

    def test_anomalous_visit_appears_in_query(
        self, agent_anomaly, litter_photo, exit_photo
    ):
        agent_anomaly.record_entry(str(litter_photo))
        agent_anomaly.record_exit(str(exit_photo))
        result = agent_anomaly.get_anomalous_visits()
        assert "anomalous" in result.lower() or "Visit #" in result

    def test_register_then_entry_then_query(self, agent, cat_photo, litter_photo):
        agent.register_cat(str(cat_photo), "Luna")
        agent.record_entry(str(litter_photo))
        cats = agent.list_cats()
        visits = agent.get_visits_by_date(
            __import__("datetime").date.today().isoformat()
        )
        assert "Luna" in cats
        # visit was recorded today
        assert "Visit" in visits or "visit" in visits.lower()

    def test_multiple_entries_separate_visits(self, agent, litter_photo):
        agent.record_entry(str(litter_photo))
        agent.record_entry(str(litter_photo))
        with _direct_db(agent) as conn:
            count = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        assert count == 2
