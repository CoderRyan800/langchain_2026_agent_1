"""
End-to-end integration tests.

All GPT-4o and CLIP calls are mocked so the suite runs offline at zero cost.
Tests exercise complete visit lifecycles, sensor data flow, weight-trend
queries, anomaly detection, and retroactive recognition edge cases.
"""

import pytest
from litterbox.db import get_conn

HEALTH_NORMAL = (
    "CONCERNS_PRESENT: no\n"
    "DESCRIPTION: Litter box looks clean.\n"
    "OWNER_SUMMARY: No visual abnormalities detected.\n"
    "\u26a0\ufe0f This analysis is preliminary and must be reviewed by a licensed veterinarian."
)

HEALTH_ANOMALY = (
    "CONCERNS_PRESENT: yes\n"
    "DESCRIPTION: Red discoloration in exit image.\n"
    "OWNER_SUMMARY: Possible blood in urine.\n"
    "\u26a0\ufe0f This analysis is preliminary and must be reviewed by a licensed veterinarian."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def no_match(monkeypatch):
    """_identify_cat always returns no-match."""
    monkeypatch.setattr(
        "litterbox.tools._identify_cat",
        lambda path: (None, None, 0.3, "No reference images."),
    )


@pytest.fixture()
def matched(registered_cat, monkeypatch):
    """_identify_cat always returns a match for the registered cat."""
    cat_id, cat_name = registered_cat
    monkeypatch.setattr(
        "litterbox.tools._identify_cat",
        lambda path: (cat_id, cat_name, 0.97, "Visual match confirmed."),
    )
    return cat_id, cat_name


@pytest.fixture()
def normal_health(monkeypatch):
    monkeypatch.setattr(
        "litterbox.tools._run_gpt4o_vision",
        lambda *a, **kw: HEALTH_NORMAL,
    )


@pytest.fixture()
def anomaly_health(monkeypatch):
    monkeypatch.setattr(
        "litterbox.tools._run_gpt4o_vision",
        lambda *a, **kw: HEALTH_ANOMALY,
    )


def _latest_visit():
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM visits ORDER BY visit_id DESC LIMIT 1"
        ).fetchone()


def _sensor_events(visit_id):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM visit_sensor_events WHERE visit_id=?", (visit_id,)
        ).fetchall()


# ---------------------------------------------------------------------------
# Basic entry→exit lifecycle
# ---------------------------------------------------------------------------

class TestBasicLifecycle:
    def test_entry_opens_visit(self, litter_image, no_match):
        from litterbox.tools import record_entry
        result = record_entry.invoke({"image_path": str(litter_image)})
        assert "opened" in result

    def test_exit_closes_visit(self, litter_image, no_match, normal_health):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image)})
        result = record_exit.invoke({"image_path": str(litter_image)})
        assert "closed" in result

    def test_visit_has_exit_time_after_exit(self, litter_image, no_match, normal_health):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image)})
        record_exit.invoke({"image_path": str(litter_image)})
        assert _latest_visit()["exit_time"] is not None

    def test_visit_has_health_notes_after_exit(self, litter_image, no_match, normal_health):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image)})
        record_exit.invoke({"image_path": str(litter_image)})
        assert _latest_visit()["health_notes"] is not None

    def test_identified_entry_sets_tentative_cat(self, litter_image, matched):
        from litterbox.tools import record_entry
        cat_id, _ = matched
        record_entry.invoke({"image_path": str(litter_image)})
        assert _latest_visit()["tentative_cat_id"] == cat_id

    def test_unidentified_entry_null_tentative(self, litter_image, no_match):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image)})
        assert _latest_visit()["tentative_cat_id"] is None

    def test_second_entry_creates_new_visit(self, litter_image, no_match, normal_health):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image)})
        record_exit.invoke({"image_path": str(litter_image)})
        record_entry.invoke({"image_path": str(litter_image)})
        with get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        assert count == 2

    def test_anomaly_flag_set_on_anomaly_response(self, litter_image, open_visit,
                                                   anomaly_health):
        from litterbox.tools import record_exit
        record_exit.invoke({"image_path": str(litter_image)})
        with get_conn() as conn:
            row = conn.execute(
                "SELECT is_anomalous FROM visits WHERE visit_id=?", (open_visit,)
            ).fetchone()
        assert bool(row["is_anomalous"])

    def test_no_anomaly_flag_on_normal_response(self, litter_image, open_visit,
                                                 normal_health):
        from litterbox.tools import record_exit
        record_exit.invoke({"image_path": str(litter_image)})
        with get_conn() as conn:
            row = conn.execute(
                "SELECT is_anomalous FROM visits WHERE visit_id=?", (open_visit,)
            ).fetchone()
        assert not bool(row["is_anomalous"])


# ---------------------------------------------------------------------------
# Full sensor data lifecycle
# ---------------------------------------------------------------------------

class TestSensorLifecycle:
    @pytest.fixture(autouse=True)
    def _stubs(self, no_match, normal_health):
        pass

    def test_cat_weight_stored_after_entry(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({
            "image_path":     str(litter_image),
            "weight_pre_g":   5800.0,
            "weight_entry_g": 10050.0,
        })
        assert abs(_latest_visit()["cat_weight_g"] - 4250.0) < 0.01

    def test_waste_weight_stored_after_exit(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({
            "image_path":   str(litter_image),
            "weight_pre_g": 5800.0,
        })
        record_exit.invoke({"image_path": str(litter_image), "weight_exit_g": 5868.0})
        assert abs(_latest_visit()["waste_weight_g"] - 68.0) < 0.01

    def test_peak_ammonia_is_max_of_entry_and_exit(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image), "ammonia_peak_ppb": 95.0})
        record_exit.invoke({"image_path": str(litter_image), "ammonia_peak_ppb": 310.0})
        assert abs(_latest_visit()["ammonia_peak_ppb"] - 310.0) < 0.01

    def test_full_sensor_cycle_all_values_correct(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({
            "image_path":       str(litter_image),
            "weight_pre_g":     5800.0,
            "weight_entry_g":   10050.0,
            "ammonia_peak_ppb": 95.0,
            "methane_peak_ppb": 18.0,
        })
        record_exit.invoke({
            "image_path":       str(litter_image),
            "weight_exit_g":    5868.0,
            "ammonia_peak_ppb": 310.0,
            "methane_peak_ppb": 55.0,
        })
        v = _latest_visit()
        assert abs(v["cat_weight_g"]     - 4250.0) < 0.01
        assert abs(v["waste_weight_g"]   - 68.0)   < 0.01
        assert abs(v["ammonia_peak_ppb"] - 310.0)  < 0.01
        assert abs(v["methane_peak_ppb"] - 55.0)   < 0.01

    def test_sensor_events_total_count_full_cycle(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({
            "image_path":       str(litter_image),
            "weight_pre_g":     5800.0,
            "weight_entry_g":   10050.0,
            "ammonia_peak_ppb": 95.0,
            "methane_peak_ppb": 18.0,
        })
        vid = _latest_visit()["visit_id"]
        record_exit.invoke({
            "image_path":       str(litter_image),
            "weight_exit_g":    5868.0,
            "ammonia_peak_ppb": 310.0,
            "methane_peak_ppb": 55.0,
        })
        events = _sensor_events(vid)
        # pre_entry weight, entry weight, entry ammonia, entry methane,
        # exit weight, exit ammonia, exit methane = 7
        assert len(events) == 7

    def test_sensor_free_visit_all_nulls(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image)})
        record_exit.invoke({"image_path": str(litter_image)})
        v = _latest_visit()
        for col in ("weight_pre_g", "weight_entry_g", "weight_exit_g",
                    "cat_weight_g", "waste_weight_g",
                    "ammonia_peak_ppb", "methane_peak_ppb"):
            assert v[col] is None, f"Expected {col} to be NULL for sensor-free visit"


# ---------------------------------------------------------------------------
# Weight trend analytics
# ---------------------------------------------------------------------------

class TestWeightTrendAnalytics:
    @pytest.fixture(autouse=True)
    def _stubs(self, no_match, normal_health):
        pass

    def test_multiple_visits_weight_trend_queryable(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        weights = [
            (5800.0, 10000.0),   # cat = 4200 g
            (5800.0, 10100.0),   # cat = 4300 g
            (5800.0, 9900.0),    # cat = 4100 g
        ]
        for pre, entry in weights:
            record_entry.invoke({
                "image_path":     str(litter_image),
                "weight_pre_g":   pre,
                "weight_entry_g": entry,
            })
            record_exit.invoke({"image_path": str(litter_image)})

        with get_conn() as conn:
            rows = conn.execute(
                "SELECT cat_weight_g FROM visits "
                "WHERE cat_weight_g IS NOT NULL ORDER BY entry_time"
            ).fetchall()

        actual = [r["cat_weight_g"] for r in rows]
        expected = [4200.0, 4300.0, 4100.0]
        for a, e in zip(actual, expected):
            assert abs(a - e) < 0.01

    def test_waste_output_queryable(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        for pre, exit_w in [(5800.0, 5850.0), (5800.0, 5870.0)]:
            record_entry.invoke({"image_path": str(litter_image), "weight_pre_g": pre})
            record_exit.invoke({"image_path": str(litter_image), "weight_exit_g": exit_w})

        with get_conn() as conn:
            rows = conn.execute(
                "SELECT waste_weight_g FROM visits "
                "WHERE waste_weight_g IS NOT NULL ORDER BY entry_time"
            ).fetchall()

        assert abs(rows[0]["waste_weight_g"] - 50.0) < 0.01
        assert abs(rows[1]["waste_weight_g"] - 70.0) < 0.01

    def test_elevated_ammonia_queryable(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        # Visit 1: normal ammonia
        record_entry.invoke({"image_path": str(litter_image), "ammonia_peak_ppb": 80.0})
        record_exit.invoke({"image_path": str(litter_image)})
        # Visit 2: elevated ammonia (potential health concern)
        record_entry.invoke({"image_path": str(litter_image), "ammonia_peak_ppb": 600.0})
        record_exit.invoke({"image_path": str(litter_image)})

        with get_conn() as conn:
            high = conn.execute(
                "SELECT COUNT(*) FROM visits WHERE ammonia_peak_ppb > 500"
            ).fetchone()[0]
        assert high == 1


# ---------------------------------------------------------------------------
# Retroactive recognition
# ---------------------------------------------------------------------------

class TestRetroactiveRecognition:
    def test_invalid_date_format_returns_error(self, registered_cat):
        from litterbox.tools import retroactive_recognition
        _, cat_name = registered_cat
        result = retroactive_recognition.invoke({
            "cat_name": cat_name, "since_date": "not-a-date"
        })
        assert "Error" in result

    def test_unregistered_cat_returns_error(self):
        from litterbox.tools import retroactive_recognition
        result = retroactive_recognition.invoke({
            "cat_name": "GhostCat", "since_date": "2026-01-01"
        })
        assert "Error" in result or "no cat" in result.lower()

    def test_future_date_returns_graceful_empty(self, registered_cat):
        from litterbox.tools import retroactive_recognition
        _, cat_name = registered_cat
        result = retroactive_recognition.invoke({
            "cat_name": cat_name, "since_date": "2099-01-01"
        })
        assert "No unknown visits" in result or "Nothing to retroactively" in result

    def test_orphan_exits_excluded_from_scan(self, registered_cat):
        from litterbox.tools import retroactive_recognition
        cat_id, cat_name = registered_cat
        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO visits (entry_time, is_orphan_exit, is_confirmed) "
                "VALUES ('2026-01-15T12:00:00', TRUE, FALSE)"
            )
            orphan_id = cur.lastrowid

        retroactive_recognition.invoke({"cat_name": cat_name, "since_date": "2026-01-01"})

        with get_conn() as conn:
            row = conn.execute(
                "SELECT is_confirmed FROM visits WHERE visit_id=?", (orphan_id,)
            ).fetchone()
        assert not bool(row["is_confirmed"])

    def test_visits_with_tentative_id_excluded(self, registered_cat):
        from litterbox.tools import retroactive_recognition
        cat_id, cat_name = registered_cat
        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO visits "
                "(entry_time, entry_image_path, tentative_cat_id, is_confirmed, is_orphan_exit) "
                "VALUES ('2026-01-15T13:00:00', 'images/test.jpg', ?, FALSE, FALSE)",
                (cat_id,),
            )
            tentative_id = cur.lastrowid

        retroactive_recognition.invoke({"cat_name": cat_name, "since_date": "2026-01-01"})

        with get_conn() as conn:
            row = conn.execute(
                "SELECT is_confirmed FROM visits WHERE visit_id=?", (tentative_id,)
            ).fetchone()
        assert not bool(row["is_confirmed"])

    def test_missing_image_skipped_gracefully(self, registered_cat):
        from litterbox.tools import retroactive_recognition
        _, cat_name = registered_cat
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO visits "
                "(entry_time, entry_image_path, is_confirmed, is_orphan_exit) "
                "VALUES ('2026-01-16T10:00:00', 'images/nonexistent/missing.jpg', FALSE, FALSE)"
            )
        result = retroactive_recognition.invoke({
            "cat_name": cat_name, "since_date": "2026-01-16"
        })
        assert "Skipped" in result

    def test_summary_includes_review_count_header(self, registered_cat):
        from litterbox.tools import retroactive_recognition
        _, cat_name = registered_cat
        # Insert one reviewable visit with a missing image → will be skipped
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO visits "
                "(entry_time, entry_image_path, is_confirmed, is_orphan_exit) "
                "VALUES ('2026-01-20T10:00:00', 'images/missing.jpg', FALSE, FALSE)"
            )
        result = retroactive_recognition.invoke({
            "cat_name": cat_name, "since_date": "2026-01-20"
        })
        assert "Visits reviewed" in result

    def test_retroactive_match_confirms_visit(self, registered_cat, litter_image,
                                               tmp_path, monkeypatch):
        """When CLIP+GPT-4o identify a match, the visit should be confirmed."""
        import shutil
        import litterbox.tools as tools_mod
        from litterbox.tools import register_cat_image, retroactive_recognition

        cat_id, cat_name = registered_cat

        # Stub _identify_cat to report a match
        monkeypatch.setattr(
            "litterbox.tools._identify_cat",
            lambda path: (cat_id, cat_name, 0.95, "Match confirmed."),
        )

        # Place the visit image where the tool can find it
        dest_dir = tools_mod.IMAGES_DIR / "visits" / "2026-02-01"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "old_entry.jpg"
        shutil.copy2(str(litter_image), str(dest))
        rel_path = str(dest.relative_to(tmp_path))

        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO visits "
                "(entry_time, entry_image_path, tentative_cat_id, is_confirmed, is_orphan_exit) "
                "VALUES ('2026-02-01T08:00:00', ?, NULL, FALSE, FALSE)",
                (rel_path,),
            )
            visit_id = cur.lastrowid

        result = retroactive_recognition.invoke({
            "cat_name": cat_name, "since_date": "2026-02-01"
        })

        with get_conn() as conn:
            v = conn.execute(
                "SELECT is_confirmed, confirmed_cat_id FROM visits WHERE visit_id=?",
                (visit_id,),
            ).fetchone()

        assert bool(v["is_confirmed"])
        assert v["confirmed_cat_id"] == cat_id
        assert "Confirmed match" in result or "Visits reviewed" in result


# ---------------------------------------------------------------------------
# Confirm-identity + query tools integration
# ---------------------------------------------------------------------------

class TestConfirmAndQuery:
    def test_confirmed_visit_absent_from_unconfirmed_list(self, registered_cat, open_visit):
        from litterbox.tools import confirm_identity, get_unconfirmed_visits
        _, cat_name = registered_cat
        confirm_identity.invoke({"visit_id": open_visit, "cat_name": cat_name})
        result = get_unconfirmed_visits.invoke({})
        assert "No unconfirmed" in result

    def test_anomalous_visit_appears_in_anomaly_list(self, litter_image, open_visit,
                                                      anomaly_health):
        from litterbox.tools import record_exit, get_anomalous_visits
        record_exit.invoke({"image_path": str(litter_image)})
        result = get_anomalous_visits.invoke({})
        assert "anomalous" in result.lower() or str(open_visit) in result

    def test_visit_images_shows_exit_after_close(self, litter_image, open_visit,
                                                  normal_health):
        from litterbox.tools import record_exit, get_visit_images
        record_exit.invoke({"image_path": str(litter_image)})
        result = get_visit_images.invoke({"visit_id": open_visit})
        assert "exit.jpg" in result
