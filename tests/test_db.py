"""
Tests for the database layer: schema creation, migration, and constraints.
All LLM and CLIP calls are irrelevant here — no mocking needed.
"""

import sqlite3
import pytest


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_all_tables_created(self):
        from litterbox.db import get_conn
        with get_conn() as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        assert "cats"                in tables
        assert "cat_images"          in tables
        assert "visits"              in tables
        assert "visit_sensor_events" in tables

    def test_cats_columns(self):
        from litterbox.db import get_conn
        with get_conn() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(cats)")}
        assert {"cat_id", "name", "created_at"} <= cols

    def test_cat_images_columns(self):
        from litterbox.db import get_conn
        with get_conn() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(cat_images)")}
        assert {"image_id", "cat_id", "file_path", "chroma_id", "created_at"} <= cols

    def test_visits_core_columns(self):
        from litterbox.db import get_conn
        with get_conn() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(visits)")}
        core = {
            "visit_id", "entry_time", "exit_time",
            "entry_image_path", "exit_image_path",
            "tentative_cat_id", "confirmed_cat_id",
            "is_confirmed", "similarity_score",
            "health_notes", "is_anomalous", "is_orphan_exit",
        }
        assert core <= cols

    def test_visits_sensor_columns(self):
        from litterbox.db import get_conn
        with get_conn() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(visits)")}
        sensor_cols = {
            "weight_pre_g", "weight_entry_g", "weight_exit_g",
            "cat_weight_g", "waste_weight_g",
            "ammonia_peak_ppb", "methane_peak_ppb",
        }
        assert sensor_cols <= cols

    def test_visit_sensor_events_columns(self):
        from litterbox.db import get_conn
        with get_conn() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(visit_sensor_events)")}
        assert {
            "event_id", "visit_id", "recorded_at", "phase",
            "sensor_type", "value_numeric", "value_text", "unit",
        } <= cols


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_init_db_can_be_called_multiple_times(self):
        from litterbox.db import init_db
        init_db()
        init_db()
        init_db()  # must not raise

    def test_table_count_stable_after_repeated_init(self):
        from litterbox.db import init_db, get_conn
        init_db()
        init_db()
        with get_conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
        assert count >= 4  # cats, cat_images, visits, visit_sensor_events (+ sqlite_sequence)


# ---------------------------------------------------------------------------
# Migration — new sensor columns on an old-style DB
# ---------------------------------------------------------------------------

class TestMigration:
    def test_sensor_columns_added_to_old_visits_table(self, tmp_path, monkeypatch):
        import litterbox.db as db_mod

        old_db = tmp_path / "old.db"
        monkeypatch.setattr(db_mod, "DB_PATH", old_db)

        # Create a visits table that lacks all sensor columns
        conn = sqlite3.connect(str(old_db))
        conn.executescript("""
            CREATE TABLE cats (
                cat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE cat_images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cat_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                chroma_id TEXT NOT NULL
            );
            CREATE TABLE visits (
                visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                entry_image_path TEXT,
                exit_image_path TEXT,
                tentative_cat_id INTEGER,
                confirmed_cat_id INTEGER,
                is_confirmed BOOLEAN DEFAULT FALSE,
                similarity_score REAL,
                health_notes TEXT,
                is_anomalous BOOLEAN DEFAULT FALSE,
                is_orphan_exit BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        conn.close()

        from litterbox.db import init_db, get_conn
        init_db()

        with get_conn() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(visits)")}

        for col in ("weight_pre_g", "weight_entry_g", "weight_exit_g",
                    "cat_weight_g", "waste_weight_g",
                    "ammonia_peak_ppb", "methane_peak_ppb"):
            assert col in cols, f"Migration failed to add column: {col}"

    def test_migration_does_not_duplicate_columns(self, tmp_path, monkeypatch):
        import litterbox.db as db_mod
        monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "dup.db")

        from litterbox.db import init_db, get_conn
        init_db()
        init_db()  # second call must not raise "duplicate column"

        with get_conn() as conn:
            col_names = [r[1] for r in conn.execute("PRAGMA table_info(visits)")]
        assert len(col_names) == len(set(col_names))


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_cats_name_unique(self):
        from litterbox.db import get_conn
        with get_conn() as conn:
            conn.execute("INSERT INTO cats (name) VALUES ('UniqueKitty')")
        with pytest.raises(Exception):
            with get_conn() as conn:
                conn.execute("INSERT INTO cats (name) VALUES ('UniqueKitty')")

    def test_foreign_key_visit_rejects_bad_cat_id(self):
        from litterbox.db import get_conn
        with pytest.raises(Exception):
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO visits (entry_time, tentative_cat_id) "
                    "VALUES ('2026-01-01', 99999)"
                )

    def test_foreign_key_sensor_event_rejects_bad_visit_id(self):
        from litterbox.db import get_conn
        with pytest.raises(Exception):
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO visit_sensor_events "
                    "(visit_id, recorded_at, sensor_type, value_numeric, unit) "
                    "VALUES (99999, '2026-01-01T00:00:00', 'weight', 4500.0, 'g')"
                )

    def test_row_factory_column_access(self):
        from litterbox.db import get_conn
        with get_conn() as conn:
            conn.execute("INSERT INTO cats (name) VALUES ('RowKitty')")
            row = conn.execute(
                "SELECT * FROM cats WHERE name='RowKitty'"
            ).fetchone()
        assert row["name"] == "RowKitty"
        assert row["cat_id"] is not None

    def test_sensor_columns_default_null(self):
        from litterbox.db import get_conn
        with get_conn() as conn:
            conn.execute("INSERT INTO cats (name) VALUES ('NullCat')")
            cat_id = conn.execute(
                "SELECT cat_id FROM cats WHERE name='NullCat'"
            ).fetchone()["cat_id"]
            conn.execute(
                "INSERT INTO visits (entry_time, tentative_cat_id) "
                "VALUES ('2026-01-01', ?)", (cat_id,)
            )
            row = conn.execute(
                "SELECT weight_pre_g, cat_weight_g, ammonia_peak_ppb "
                "FROM visits ORDER BY visit_id DESC LIMIT 1"
            ).fetchone()
        assert row["weight_pre_g"] is None
        assert row["cat_weight_g"] is None
        assert row["ammonia_peak_ppb"] is None
