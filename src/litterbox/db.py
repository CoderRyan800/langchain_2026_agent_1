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
                weight_pre_g      REAL,
                weight_entry_g    REAL,
                weight_exit_g     REAL,
                cat_weight_g      REAL,
                waste_weight_g    REAL,
                ammonia_peak_ppb  REAL,
                methane_peak_ppb  REAL,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS visit_sensor_events (
                event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                visit_id      INTEGER NOT NULL REFERENCES visits(visit_id),
                recorded_at   TEXT NOT NULL,
                phase         TEXT,
                sensor_type   TEXT NOT NULL,
                value_numeric REAL,
                value_text    TEXT,
                unit          TEXT
            );

            CREATE TABLE IF NOT EXISTS td_visits (
                td_visit_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time        TIMESTAMP NOT NULL,
                exit_time         TIMESTAMP NOT NULL,
                chip_id           TEXT,
                tentative_cat_id  INTEGER REFERENCES cats(cat_id),
                confirmed_cat_id  INTEGER REFERENCES cats(cat_id),
                is_confirmed      BOOLEAN DEFAULT FALSE,
                id_method         TEXT,
                top_similarity    REAL,
                snapshot_json     TEXT NOT NULL,
                images_dir        TEXT,
                health_notes      TEXT,
                is_anomalous      BOOLEAN DEFAULT FALSE,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS eigen_models (
                model_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                cat_id         INTEGER REFERENCES cats(cat_id),
                channel        TEXT NOT NULL,
                eigenvalues_json   TEXT NOT NULL,
                eigenvectors_json  TEXT NOT NULL,
                n_components       INTEGER NOT NULL,
                n_waveforms        INTEGER NOT NULL,
                explained_variance_threshold REAL NOT NULL,
                regularized        BOOLEAN DEFAULT FALSE,
                computed_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS eigen_waveforms (
                waveform_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                td_visit_id    INTEGER REFERENCES td_visits(td_visit_id),
                cat_id         INTEGER REFERENCES cats(cat_id),
                channel        TEXT NOT NULL,
                vector_json    TEXT NOT NULL,
                dc_term        REAL NOT NULL,
                coefficients_json  TEXT,
                eigen_ev       REAL,
                eigen_residual REAL,
                model_id       INTEGER REFERENCES eigen_models(model_id),
                raw_length     INTEGER NOT NULL,
                nan_fraction   REAL NOT NULL,
                cluster_log_likelihood REAL,
                cluster_z_score        REAL,
                cluster_assignment     INTEGER,
                cluster_model_id       INTEGER REFERENCES cluster_models(cluster_model_id),
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS cluster_models (
                cluster_model_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                cat_id            INTEGER REFERENCES cats(cat_id),
                channel           TEXT NOT NULL,
                k_clusters        INTEGER NOT NULL,
                means_json        TEXT NOT NULL,
                covariances_json  TEXT NOT NULL,
                weights_json      TEXT NOT NULL,
                bic_values_json   TEXT NOT NULL,
                n_samples         INTEGER NOT NULL,
                uniform_n         INTEGER NOT NULL,
                computed_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Idempotent migration: add cluster columns to eigen_waveforms if older DB
        ew_cols = {row[1] for row in conn.execute("PRAGMA table_info(eigen_waveforms)")}
        cluster_cols = [
            ("cluster_log_likelihood", "REAL"),
            ("cluster_z_score",        "REAL"),
            ("cluster_assignment",     "INTEGER"),
            ("cluster_model_id",       "INTEGER REFERENCES cluster_models(cluster_model_id)"),
        ]
        for col, typ in cluster_cols:
            if col not in ew_cols:
                conn.execute(f"ALTER TABLE eigen_waveforms ADD COLUMN {col} {typ}")

        # Idempotent migration: add sensor columns to visits if this is an older DB
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(visits)")}
        new_cols = [
            ("weight_pre_g",            "REAL"),
            ("weight_entry_g",          "REAL"),
            ("weight_exit_g",           "REAL"),
            ("cat_weight_g",            "REAL"),
            ("waste_weight_g",          "REAL"),
            ("ammonia_peak_ppb",        "REAL"),
            ("methane_peak_ppb",        "REAL"),
            ("ammonia_z_score",         "REAL"),
            ("methane_z_score",         "REAL"),
            ("gas_anomaly_tier",        "TEXT"),
            ("gas_anomaly_n_samples",   "INTEGER"),
            ("gas_anomaly_model_used",  "TEXT"),
            ("gas_anomaly_rescored_at", "TIMESTAMP"),
        ]
        for col, typ in new_cols:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE visits ADD COLUMN {col} {typ}")
