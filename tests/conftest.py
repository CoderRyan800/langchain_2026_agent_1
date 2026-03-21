"""
Shared fixtures for the pytest test suite.

All tests automatically receive ``isolated_env``, which redirects every
module-level path constant to a per-test ``tmp_path`` directory so no test
can touch the production database, Chroma index, or image store.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Canned GPT-4o health responses used by both fixtures and test assertions
# ---------------------------------------------------------------------------

HEALTH_NORMAL = (
    "CONCERNS_PRESENT: no\n"
    "DESCRIPTION: Both images show a clean litter box with no visible abnormalities.\n"
    "OWNER_SUMMARY: No visual abnormalities detected.\n"
    "\u26a0\ufe0f This analysis is preliminary and must be reviewed by a licensed "
    "veterinarian before any medical decisions are made."
)

HEALTH_ANOMALY = (
    "CONCERNS_PRESENT: yes\n"
    "DESCRIPTION: Red discoloration visible in exit image, possibly blood in urine.\n"
    "OWNER_SUMMARY: There appears to be blood in the urine. Please consult a veterinarian.\n"
    "\u26a0\ufe0f This analysis is preliminary and must be reviewed by a licensed "
    "veterinarian before any medical decisions are made."
)


# ---------------------------------------------------------------------------
# Path isolation — autouse so every test gets a clean slate
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Redirect all module-level paths to tmp_path for full test isolation."""
    import litterbox.db as db_mod
    import litterbox.embeddings as emb_mod
    import litterbox.tools as tools_mod

    db_path    = tmp_path / "test.db"
    chroma_path = tmp_path / "chroma"
    images_dir  = tmp_path / "images"

    monkeypatch.setattr(db_mod,    "DB_PATH",      db_path)
    monkeypatch.setattr(emb_mod,   "CHROMA_PATH",  chroma_path)
    monkeypatch.setattr(emb_mod,   "_collection",  None)   # force re-init at new path
    monkeypatch.setattr(tools_mod, "IMAGES_DIR",   images_dir)
    monkeypatch.setattr(tools_mod, "PROJECT_ROOT", tmp_path)

    from litterbox.db import init_db
    init_db()


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cat_image(tmp_path) -> Path:
    """Brownish JPEG standing in for a real cat photo."""
    img = Image.new("RGB", (320, 240), (180, 120, 80))
    path = tmp_path / "cat.jpg"
    img.save(str(path), "JPEG")
    return path


@pytest.fixture()
def litter_image(tmp_path) -> Path:
    """Beige JPEG simulating an empty litter box."""
    img = Image.new("RGB", (400, 300), (240, 230, 200))
    path = tmp_path / "litter.jpg"
    img.save(str(path), "JPEG")
    return path


@pytest.fixture()
def exit_image(tmp_path) -> Path:
    """Litter box image with a subtle shadow — used as exit photo."""
    from PIL import ImageDraw
    img = Image.new("RGB", (400, 300), (240, 230, 200))
    ImageDraw.Draw(img).ellipse([160, 120, 240, 180], fill=(210, 200, 170))
    path = tmp_path / "exit.jpg"
    img.save(str(path), "JPEG")
    return path


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def registered_cat():
    """Insert a cat directly into the test DB (no CLIP needed); returns (cat_id, name)."""
    from litterbox.db import get_conn
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO cats (name) VALUES ('Whiskers')")
        cat_id = cur.lastrowid
    return cat_id, "Whiskers"


@pytest.fixture()
def open_visit(registered_cat):
    """Create an open (no exit_time) visit row in the test DB; returns visit_id."""
    cat_id, _ = registered_cat
    from litterbox.db import get_conn
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO visits
               (entry_time, entry_image_path, tentative_cat_id, similarity_score)
               VALUES ('2026-01-01T10:00:00', 'images/visits/test/entry.jpg', ?, 0.95)""",
            (cat_id,),
        )
        return cur.lastrowid


# ---------------------------------------------------------------------------
# LLM stubs
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_identify_cat(monkeypatch):
    """Stub _identify_cat to return no-match (avoids CLIP model load)."""
    monkeypatch.setattr(
        "litterbox.tools._identify_cat",
        lambda path: (None, None, 0.3, "No reference images in database yet."),
    )


@pytest.fixture()
def mock_gpt4o(monkeypatch):
    """Stub _run_gpt4o_vision to return a normal health response."""
    monkeypatch.setattr(
        "litterbox.tools._run_gpt4o_vision",
        lambda prompt, *paths: HEALTH_NORMAL,
    )


@pytest.fixture()
def mock_gpt4o_anomaly(monkeypatch):
    """Stub _run_gpt4o_vision to return an anomalous health response."""
    monkeypatch.setattr(
        "litterbox.tools._run_gpt4o_vision",
        lambda prompt, *paths: HEALTH_ANOMALY,
    )
