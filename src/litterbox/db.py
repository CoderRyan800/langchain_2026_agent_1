import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "litterbox.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cats (
                cat_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS cat_images (
                image_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                cat_id     INTEGER NOT NULL REFERENCES cats(cat_id),
                file_path  TEXT NOT NULL,
                chroma_id  TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS visits (
                visit_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time        TIMESTAMP,
                exit_time         TIMESTAMP,
                entry_image_path  TEXT,
                exit_image_path   TEXT,
                tentative_cat_id  INTEGER REFERENCES cats(cat_id),
                confirmed_cat_id  INTEGER REFERENCES cats(cat_id),
                is_confirmed      BOOLEAN DEFAULT FALSE,
                similarity_score  REAL,
                health_notes      TEXT,
                is_anomalous      BOOLEAN DEFAULT FALSE,
                is_orphan_exit    BOOLEAN DEFAULT FALSE,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
