"""
Tests for the non-LLM tools: register_cat_image, list_cats, confirm_identity,
and all five query tools.  record_entry / record_exit are covered in
test_tools_sensor.py.

These tests mock _identify_cat where needed (via a class-level autouse fixture)
so the CLIP model is never loaded.
"""

import pytest
from litterbox.db import get_conn


# ---------------------------------------------------------------------------
# register_cat_image
# ---------------------------------------------------------------------------

class TestRegisterCatImage:
    def test_returns_success_message(self, cat_image):
        from litterbox.tools import register_cat_image
        result = register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Whiskers"})
        assert "Registered" in result

    def test_creates_cats_row(self, cat_image):
        from litterbox.tools import register_cat_image
        register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Whiskers"})
        with get_conn() as conn:
            row = conn.execute("SELECT * FROM cats WHERE name='Whiskers'").fetchone()
        assert row is not None

    def test_creates_cat_images_row(self, cat_image):
        from litterbox.tools import register_cat_image
        register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Whiskers"})
        with get_conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM cat_images ci "
                "JOIN cats c ON ci.cat_id = c.cat_id WHERE c.name='Whiskers'"
            ).fetchone()[0]
        assert count == 1

    def test_image_file_copied_to_store(self, cat_image):
        import litterbox.tools as tools_mod
        from litterbox.tools import register_cat_image
        register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Whiskers"})
        stored = list((tools_mod.IMAGES_DIR / "cats").rglob("*.jpg"))
        assert len(stored) == 1

    def test_second_registration_adds_image_no_duplicate_cat(self, cat_image):
        from litterbox.tools import register_cat_image
        register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Whiskers"})
        register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Whiskers"})
        with get_conn() as conn:
            cat_count = conn.execute(
                "SELECT COUNT(*) FROM cats WHERE name='Whiskers'"
            ).fetchone()[0]
            img_count = conn.execute(
                "SELECT COUNT(*) FROM cat_images ci "
                "JOIN cats c ON ci.cat_id=c.cat_id WHERE c.name='Whiskers'"
            ).fetchone()[0]
        assert cat_count == 1
        assert img_count == 2

    def test_two_different_cats_both_stored(self, cat_image, tmp_path):
        from PIL import Image
        from litterbox.tools import register_cat_image
        img2 = Image.new("RGB", (100, 100), (20, 100, 200))
        path2 = tmp_path / "cat2.jpg"
        img2.save(str(path2), "JPEG")

        register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Mittens"})
        register_cat_image.invoke({"image_path": str(path2),    "cat_name": "Oliver"})
        with get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM cats").fetchone()[0]
        assert count == 2

    def test_nonexistent_path_returns_error(self):
        from litterbox.tools import register_cat_image
        result = register_cat_image.invoke({
            "image_path": "/tmp/no_such_file_abc_xyz.jpg",
            "cat_name":   "Ghost",
        })
        assert "Error" in result

    def test_chroma_id_stored(self, cat_image):
        from litterbox.tools import register_cat_image
        register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Whiskers"})
        with get_conn() as conn:
            row = conn.execute("SELECT chroma_id FROM cat_images LIMIT 1").fetchone()
        assert row["chroma_id"] and len(row["chroma_id"]) > 0

    def test_stored_path_is_relative_to_project_root(self, cat_image):
        from litterbox.tools import register_cat_image
        register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Whiskers"})
        with get_conn() as conn:
            row = conn.execute("SELECT file_path FROM cat_images LIMIT 1").fetchone()
        # Should not start with '/' (must be relative)
        assert not row["file_path"].startswith("/")

    def test_result_includes_stored_path(self, cat_image):
        from litterbox.tools import register_cat_image
        result = register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Whiskers"})
        assert "Stored at" in result or "stored" in result.lower()


# ---------------------------------------------------------------------------
# list_cats
# ---------------------------------------------------------------------------

class TestListCats:
    def test_empty_database(self):
        from litterbox.tools import list_cats
        result = list_cats.invoke({})
        assert "No cats" in result

    def test_shows_registered_cat(self, cat_image):
        from litterbox.tools import register_cat_image, list_cats
        register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Mittens"})
        result = list_cats.invoke({})
        assert "Mittens" in result

    def test_shows_image_count(self, cat_image):
        from litterbox.tools import register_cat_image, list_cats
        register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Mittens"})
        register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Mittens"})
        result = list_cats.invoke({})
        assert "2" in result

    def test_multiple_cats_all_listed(self, cat_image, tmp_path):
        from PIL import Image
        from litterbox.tools import register_cat_image, list_cats
        img2 = Image.new("RGB", (100, 100), (50, 100, 200))
        path2 = tmp_path / "cat2.jpg"
        img2.save(str(path2), "JPEG")
        register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Mittens"})
        register_cat_image.invoke({"image_path": str(path2),    "cat_name": "Oliver"})
        result = list_cats.invoke({})
        assert "Mittens" in result
        assert "Oliver"  in result

    def test_result_includes_registration_date(self, cat_image):
        from litterbox.tools import register_cat_image, list_cats
        register_cat_image.invoke({"image_path": str(cat_image), "cat_name": "Mittens"})
        result = list_cats.invoke({})
        # Registered date is included in parentheses
        assert "registered" in result.lower() or "(" in result


# ---------------------------------------------------------------------------
# confirm_identity
# ---------------------------------------------------------------------------

class TestConfirmIdentity:
    def test_success_message(self, registered_cat, open_visit):
        from litterbox.tools import confirm_identity
        cat_id, cat_name = registered_cat
        result = confirm_identity.invoke({"visit_id": open_visit, "cat_name": cat_name})
        assert "confirmed" in result.lower()

    def test_sets_is_confirmed_true(self, registered_cat, open_visit):
        from litterbox.tools import confirm_identity
        cat_id, cat_name = registered_cat
        confirm_identity.invoke({"visit_id": open_visit, "cat_name": cat_name})
        with get_conn() as conn:
            row = conn.execute(
                "SELECT is_confirmed FROM visits WHERE visit_id=?", (open_visit,)
            ).fetchone()
        assert bool(row["is_confirmed"])

    def test_sets_confirmed_cat_id(self, registered_cat, open_visit):
        from litterbox.tools import confirm_identity
        cat_id, cat_name = registered_cat
        confirm_identity.invoke({"visit_id": open_visit, "cat_name": cat_name})
        with get_conn() as conn:
            row = conn.execute(
                "SELECT confirmed_cat_id FROM visits WHERE visit_id=?", (open_visit,)
            ).fetchone()
        assert row["confirmed_cat_id"] == cat_id

    def test_unknown_cat_returns_error(self, open_visit):
        from litterbox.tools import confirm_identity
        result = confirm_identity.invoke({"visit_id": open_visit, "cat_name": "NoSuchCat"})
        assert "Error" in result

    def test_invalid_visit_id_returns_error(self, registered_cat):
        from litterbox.tools import confirm_identity
        _, cat_name = registered_cat
        result = confirm_identity.invoke({"visit_id": 99999, "cat_name": cat_name})
        assert "Error" in result

    def test_result_includes_visit_id(self, registered_cat, open_visit):
        from litterbox.tools import confirm_identity
        _, cat_name = registered_cat
        result = confirm_identity.invoke({"visit_id": open_visit, "cat_name": cat_name})
        assert str(open_visit) in result

    def test_result_includes_cat_name(self, registered_cat, open_visit):
        from litterbox.tools import confirm_identity
        _, cat_name = registered_cat
        result = confirm_identity.invoke({"visit_id": open_visit, "cat_name": cat_name})
        assert cat_name in result


# ---------------------------------------------------------------------------
# get_visits_by_date
# ---------------------------------------------------------------------------

class TestGetVisitsByDate:
    def test_no_visits_returns_graceful(self):
        from litterbox.tools import get_visits_by_date
        result = get_visits_by_date.invoke({"date_str": "1900-01-01"})
        assert "No visits" in result

    def test_returns_visit_on_correct_date(self, registered_cat):
        from litterbox.tools import get_visits_by_date
        cat_id, _ = registered_cat
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO visits (entry_time, tentative_cat_id) "
                "VALUES ('2026-03-15T10:00:00', ?)", (cat_id,)
            )
        result = get_visits_by_date.invoke({"date_str": "2026-03-15"})
        assert "2026-03-15" in result

    def test_does_not_return_other_dates(self, registered_cat):
        from litterbox.tools import get_visits_by_date
        cat_id, _ = registered_cat
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO visits (entry_time, tentative_cat_id) "
                "VALUES ('2026-03-15T10:00:00', ?)", (cat_id,)
            )
        result = get_visits_by_date.invoke({"date_str": "2026-03-16"})
        assert "No visits" in result

    def test_shows_visit_count(self, registered_cat):
        from litterbox.tools import get_visits_by_date
        cat_id, _ = registered_cat
        with get_conn() as conn:
            for h in (8, 12, 18):
                conn.execute(
                    f"INSERT INTO visits (entry_time, tentative_cat_id) "
                    f"VALUES ('2026-04-01T{h:02d}:00:00', ?)", (cat_id,)
                )
        result = get_visits_by_date.invoke({"date_str": "2026-04-01"})
        assert "3" in result

    def test_anomalous_visit_flagged(self):
        from litterbox.tools import get_visits_by_date
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO visits (entry_time, is_anomalous) "
                "VALUES ('2026-05-01T10:00:00', TRUE)"
            )
        result = get_visits_by_date.invoke({"date_str": "2026-05-01"})
        assert "\u26a0" in result or "ANOMALY" in result.upper() or "anomal" in result.lower()


# ---------------------------------------------------------------------------
# get_visits_by_cat
# ---------------------------------------------------------------------------

class TestGetVisitsByCat:
    def test_unknown_cat_returns_not_found(self):
        from litterbox.tools import get_visits_by_cat
        result = get_visits_by_cat.invoke({"cat_name": "NoSuchCat"})
        assert "No cat" in result or "not found" in result.lower()

    def test_cat_with_no_visits(self, registered_cat):
        from litterbox.tools import get_visits_by_cat
        _, cat_name = registered_cat
        result = get_visits_by_cat.invoke({"cat_name": cat_name})
        assert "No visits" in result

    def test_returns_confirmed_visit(self, registered_cat):
        from litterbox.tools import get_visits_by_cat
        cat_id, cat_name = registered_cat
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO visits (entry_time, confirmed_cat_id, is_confirmed) "
                "VALUES ('2026-03-01T08:00:00', ?, TRUE)", (cat_id,)
            )
        result = get_visits_by_cat.invoke({"cat_name": cat_name})
        assert "confirmed" in result.lower()

    def test_returns_tentative_visit(self, registered_cat):
        from litterbox.tools import get_visits_by_cat
        cat_id, cat_name = registered_cat
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO visits (entry_time, tentative_cat_id, is_confirmed) "
                "VALUES ('2026-03-01T08:00:00', ?, FALSE)", (cat_id,)
            )
        result = get_visits_by_cat.invoke({"cat_name": cat_name})
        assert "tentative" in result.lower()

    def test_shows_visit_count(self, registered_cat):
        from litterbox.tools import get_visits_by_cat
        cat_id, cat_name = registered_cat
        with get_conn() as conn:
            for h in (8, 12):
                conn.execute(
                    f"INSERT INTO visits (entry_time, confirmed_cat_id, is_confirmed) "
                    f"VALUES ('2026-03-01T{h:02d}:00:00', ?, TRUE)", (cat_id,)
                )
        result = get_visits_by_cat.invoke({"cat_name": cat_name})
        assert "2" in result


# ---------------------------------------------------------------------------
# get_unconfirmed_visits
# ---------------------------------------------------------------------------

class TestGetUnconfirmedVisits:
    def test_empty_database(self):
        from litterbox.tools import get_unconfirmed_visits
        result = get_unconfirmed_visits.invoke({})
        assert "No unconfirmed" in result

    def test_lists_unconfirmed_visit(self, registered_cat):
        from litterbox.tools import get_unconfirmed_visits
        cat_id, _ = registered_cat
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO visits (entry_time, tentative_cat_id, is_confirmed) "
                "VALUES ('2026-01-01T08:00:00', ?, FALSE)", (cat_id,)
            )
        result = get_unconfirmed_visits.invoke({})
        assert "unconfirmed" in result.lower() or "#" in result

    def test_confirmed_visit_not_listed(self, registered_cat, open_visit):
        from litterbox.tools import confirm_identity, get_unconfirmed_visits
        cat_id, cat_name = registered_cat
        confirm_identity.invoke({"visit_id": open_visit, "cat_name": cat_name})
        result = get_unconfirmed_visits.invoke({})
        assert "No unconfirmed" in result

    def test_count_reflects_multiple_unconfirmed(self, registered_cat):
        from litterbox.tools import get_unconfirmed_visits
        cat_id, _ = registered_cat
        with get_conn() as conn:
            for h in (8, 12, 16):
                conn.execute(
                    f"INSERT INTO visits (entry_time, tentative_cat_id, is_confirmed) "
                    f"VALUES ('2026-01-01T{h:02d}:00:00', ?, FALSE)", (cat_id,)
                )
        result = get_unconfirmed_visits.invoke({})
        assert "3" in result


# ---------------------------------------------------------------------------
# get_anomalous_visits
# ---------------------------------------------------------------------------

class TestGetAnomalousVisits:
    def test_no_anomalies(self):
        from litterbox.tools import get_anomalous_visits
        result = get_anomalous_visits.invoke({})
        assert "No anomalous" in result

    def test_lists_anomalous_visit(self):
        from litterbox.tools import get_anomalous_visits
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO visits (entry_time, is_anomalous, health_notes) "
                "VALUES ('2026-01-01T08:00:00', TRUE, 'Blood detected.')"
            )
        result = get_anomalous_visits.invoke({})
        assert "anomalous" in result.lower() or "Blood" in result

    def test_non_anomalous_visit_not_listed(self, registered_cat):
        from litterbox.tools import get_anomalous_visits
        cat_id, _ = registered_cat
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO visits (entry_time, is_anomalous, tentative_cat_id) "
                "VALUES ('2026-01-01T08:00:00', FALSE, ?)", (cat_id,)
            )
        result = get_anomalous_visits.invoke({})
        assert "No anomalous" in result

    def test_count_reflects_multiple_anomalies(self):
        from litterbox.tools import get_anomalous_visits
        with get_conn() as conn:
            for h in (8, 12):
                conn.execute(
                    f"INSERT INTO visits (entry_time, is_anomalous, health_notes) "
                    f"VALUES ('2026-01-01T{h:02d}:00:00', TRUE, 'Issue found.')"
                )
        result = get_anomalous_visits.invoke({})
        assert "2" in result

    def test_includes_gas_anomaly_summary(self, registered_cat):
        from litterbox.tools import get_anomalous_visits
        cat_id, _ = registered_cat
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO visits
                   (entry_time, tentative_cat_id, is_anomalous, health_notes,
                    ammonia_z_score, methane_z_score, gas_anomaly_tier,
                    gas_anomaly_n_samples, gas_anomaly_model_used)
                   VALUES ('2026-01-01T08:00:00', ?, TRUE,
                           'GPT-4o refused.', 1.08, 6.11, 'severe', 32, 'per_cat')""",
                (cat_id,),
            )
        result = get_anomalous_visits.invoke({})
        # Tier and z-scores must surface so the agent can explain the flag.
        assert "severe" in result
        assert "+6.11" in result
        assert "per_cat" in result
        assert "32" in result

    def test_insufficient_data_summary_when_no_score(self):
        from litterbox.tools import get_anomalous_visits
        with get_conn() as conn:
            # is_anomalous=TRUE but no gas_anomaly_* columns populated.
            conn.execute(
                "INSERT INTO visits (entry_time, is_anomalous, health_notes) "
                "VALUES ('2026-01-01T08:00:00', TRUE, 'Visual concern.')"
            )
        result = get_anomalous_visits.invoke({})
        assert "insufficient data" in result.lower()


# ---------------------------------------------------------------------------
# get_visit_details
# ---------------------------------------------------------------------------

class TestGetVisitDetails:
    def test_unknown_visit_id_returns_clear_error(self):
        from litterbox.tools import get_visit_details
        result = get_visit_details.invoke({"visit_id": 99999})
        assert "No visit found" in result
        assert "99999" in result

    def test_returns_full_block_for_anomalous_visit(self, registered_cat):
        from litterbox.tools import get_visit_details
        cat_id, name = registered_cat
        with get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO visits
                   (entry_time, exit_time, tentative_cat_id, is_anomalous,
                    similarity_score, health_notes,
                    weight_pre_g, weight_entry_g, weight_exit_g,
                    cat_weight_g, waste_weight_g,
                    ammonia_peak_ppb, methane_peak_ppb,
                    ammonia_z_score, methane_z_score,
                    gas_anomaly_tier, gas_anomaly_n_samples, gas_anomaly_model_used)
                   VALUES ('2026-01-01T08:00:00', '2026-01-01T08:00:30', ?,
                           TRUE, 0.93, 'I am sorry, I cannot assist.',
                           2000, 5200, 2090, 3200, 90,
                           182.0, 153.0, 1.08, 6.11,
                           'severe', 32, 'per_cat')""",
                (cat_id,),
            )
            visit_id = cur.lastrowid
        result = get_visit_details.invoke({"visit_id": visit_id})
        # Identity
        assert name in result
        assert "0.93" in result
        # Sensor block
        assert "182.0 ppb" in result or "182.0" in result
        assert "153.0" in result
        assert "3200" in result   # cat weight
        # Gas anomaly block
        assert "severe" in result
        assert "+1.08" in result
        assert "+6.11" in result
        assert "per_cat" in result
        assert "32 prior visits" in result
        # Health notes preserved verbatim (even when LLM refused)
        assert "I cannot assist" in result
        # Anomalous flag in header
        assert "ANOMALOUS" in result

    def test_normal_visit_shows_insufficient_data_block(self, registered_cat):
        from litterbox.tools import get_visit_details
        cat_id, _ = registered_cat
        with get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO visits
                   (entry_time, tentative_cat_id, is_anomalous, similarity_score)
                   VALUES ('2026-01-01T08:00:00', ?, FALSE, 0.95)""",
                (cat_id,),
            )
            visit_id = cur.lastrowid
        result = get_visit_details.invoke({"visit_id": visit_id})
        assert "insufficient data" in result.lower()
        assert "ANOMALOUS" not in result

    def test_orphan_exit_flagged_in_header(self, registered_cat):
        from litterbox.tools import get_visit_details
        cat_id, _ = registered_cat
        with get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO visits
                   (entry_time, tentative_cat_id, is_orphan_exit)
                   VALUES ('2026-01-01T08:00:00', ?, TRUE)""",
                (cat_id,),
            )
            visit_id = cur.lastrowid
        result = get_visit_details.invoke({"visit_id": visit_id})
        assert "ORPHAN EXIT" in result


# ---------------------------------------------------------------------------
# get_visit_images
# ---------------------------------------------------------------------------

class TestGetVisitImages:
    def test_valid_visit_shows_entry_path(self, open_visit):
        from litterbox.tools import get_visit_images
        result = get_visit_images.invoke({"visit_id": open_visit})
        assert "Entry" in result

    def test_valid_visit_shows_exit_placeholder(self, open_visit):
        from litterbox.tools import get_visit_images
        result = get_visit_images.invoke({"visit_id": open_visit})
        assert "Exit" in result

    def test_invalid_visit_returns_not_found(self):
        from litterbox.tools import get_visit_images
        result = get_visit_images.invoke({"visit_id": 99999})
        assert "not found" in result.lower()

    def test_result_includes_visit_id(self, open_visit):
        from litterbox.tools import get_visit_images
        result = get_visit_images.invoke({"visit_id": open_visit})
        assert str(open_visit) in result

    def test_entry_path_shown_after_exit(self, litter_image, open_visit, mock_gpt4o):
        from litterbox.tools import record_exit, get_visit_images
        record_exit.invoke({"image_path": str(litter_image)})
        result = get_visit_images.invoke({"visit_id": open_visit})
        # Both entry and exit paths should now be non-placeholder
        assert "entry.jpg" in result
        assert "exit.jpg"  in result
