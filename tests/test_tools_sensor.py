"""
Tests for record_entry and record_exit, covering:
  - visits with and without sensor data
  - derived value computation (cat_weight_g, waste_weight_g)
  - peak-gas reconciliation logic
  - visit_sensor_events logging (phase, sensor_type, unit, value)
  - health prompt enrichment
  - backward compatibility (sensor columns NULL when omitted)

All LLM and CLIP calls are mocked via class-level autouse fixtures so
these tests run offline with no API cost.
"""

import pytest
from litterbox.db import get_conn

_HEALTH_NORMAL = (
    "CONCERNS_PRESENT: no\n"
    "DESCRIPTION: Litter box looks clean.\n"
    "OWNER_SUMMARY: No visual abnormalities detected.\n"
    "\u26a0\ufe0f This analysis is preliminary and must be reviewed by a licensed veterinarian."
)

_HEALTH_ANOMALY = (
    "CONCERNS_PRESENT: yes\n"
    "DESCRIPTION: Red discoloration in exit image.\n"
    "OWNER_SUMMARY: Possible blood in urine.\n"
    "\u26a0\ufe0f This analysis is preliminary and must be reviewed by a licensed veterinarian."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_visit():
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM visits ORDER BY visit_id DESC LIMIT 1"
        ).fetchone()


def _sensor_events(visit_id):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM visit_sensor_events "
            "WHERE visit_id=? ORDER BY event_id",
            (visit_id,),
        ).fetchall()


def _events_map(visit_id):
    """Return {(phase, sensor_type): row} for easy lookup."""
    return {
        (e["phase"], e["sensor_type"]): e
        for e in _sensor_events(visit_id)
    }


# ---------------------------------------------------------------------------
# record_entry — no sensors
# ---------------------------------------------------------------------------

class TestRecordEntryNoSensors:
    @pytest.fixture(autouse=True)
    def _no_clip(self, monkeypatch):
        monkeypatch.setattr(
            "litterbox.tools._identify_cat",
            lambda path: (None, None, 0.3, "No reference images."),
        )

    def test_opens_visit(self, litter_image):
        from litterbox.tools import record_entry
        result = record_entry.invoke({"image_path": str(litter_image)})
        assert "opened" in result

    def test_visit_row_created(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image)})
        assert _latest_visit() is not None

    def test_entry_time_recorded(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image)})
        assert _latest_visit()["entry_time"] is not None

    def test_exit_time_null(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image)})
        assert _latest_visit()["exit_time"] is None

    def test_all_sensor_columns_null(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image)})
        v = _latest_visit()
        for col in ("weight_pre_g", "weight_entry_g", "cat_weight_g",
                    "ammonia_peak_ppb", "methane_peak_ppb"):
            assert v[col] is None, f"Expected {col} to be NULL"

    def test_no_sensor_events_logged(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image)})
        assert len(_sensor_events(_latest_visit()["visit_id"])) == 0

    def test_image_stored_in_images_dir(self, litter_image):
        import litterbox.tools as tools_mod
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image)})
        stored = list((tools_mod.IMAGES_DIR / "visits").rglob("*_entry.jpg"))
        assert len(stored) == 1

    def test_nonexistent_image_returns_error(self):
        from litterbox.tools import record_entry
        result = record_entry.invoke({"image_path": "/no/such/file.jpg"})
        assert "Error" in result

    def test_unidentified_cat_flagged_in_output(self, litter_image):
        from litterbox.tools import record_entry
        result = record_entry.invoke({"image_path": str(litter_image)})
        assert "not identified" in result.lower() or "flagged" in result.lower()


# ---------------------------------------------------------------------------
# record_entry — with sensor data
# ---------------------------------------------------------------------------

class TestRecordEntryWithSensors:
    @pytest.fixture(autouse=True)
    def _no_clip(self, monkeypatch):
        monkeypatch.setattr(
            "litterbox.tools._identify_cat",
            lambda path: (None, None, 0.3, "No reference images."),
        )

    def test_stores_weight_pre_g(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image), "weight_pre_g": 5800.0})
        assert abs(_latest_visit()["weight_pre_g"] - 5800.0) < 0.01

    def test_stores_weight_entry_g(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image), "weight_entry_g": 10050.0})
        assert abs(_latest_visit()["weight_entry_g"] - 10050.0) < 0.01

    def test_derives_cat_weight_when_both_weights_given(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({
            "image_path":    str(litter_image),
            "weight_pre_g":  5800.0,
            "weight_entry_g": 10050.0,
        })
        assert abs(_latest_visit()["cat_weight_g"] - 4250.0) < 0.01

    def test_cat_weight_null_with_only_pre(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image), "weight_pre_g": 5800.0})
        assert _latest_visit()["cat_weight_g"] is None

    def test_cat_weight_null_with_only_entry(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image), "weight_entry_g": 10050.0})
        assert _latest_visit()["cat_weight_g"] is None

    def test_stores_ammonia(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image), "ammonia_peak_ppb": 95.0})
        assert abs(_latest_visit()["ammonia_peak_ppb"] - 95.0) < 0.01

    def test_stores_methane(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image), "methane_peak_ppb": 18.0})
        assert abs(_latest_visit()["methane_peak_ppb"] - 18.0) < 0.01

    def test_cat_weight_shown_in_output(self, litter_image):
        from litterbox.tools import record_entry
        result = record_entry.invoke({
            "image_path":     str(litter_image),
            "weight_pre_g":   5800.0,
            "weight_entry_g": 10050.0,
        })
        assert "4250" in result or "cat weight" in result.lower()

    def test_ammonia_shown_in_output(self, litter_image):
        from litterbox.tools import record_entry
        result = record_entry.invoke({
            "image_path":       str(litter_image),
            "ammonia_peak_ppb": 95.0,
        })
        assert "95" in result or "NH" in result


# ---------------------------------------------------------------------------
# record_entry — sensor event logging
# ---------------------------------------------------------------------------

class TestRecordEntrySensorEvents:
    @pytest.fixture(autouse=True)
    def _no_clip(self, monkeypatch):
        monkeypatch.setattr(
            "litterbox.tools._identify_cat",
            lambda path: (None, None, 0.3, "No reference images."),
        )

    def test_pre_entry_weight_event_logged(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image), "weight_pre_g": 5800.0})
        em = _events_map(_latest_visit()["visit_id"])
        assert ("pre_entry", "weight") in em

    def test_entry_weight_event_logged(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image), "weight_entry_g": 10050.0})
        em = _events_map(_latest_visit()["visit_id"])
        assert ("entry", "weight") in em

    def test_entry_ammonia_event_logged(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image), "ammonia_peak_ppb": 95.0})
        em = _events_map(_latest_visit()["visit_id"])
        assert ("entry", "ammonia") in em

    def test_entry_methane_event_logged(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image), "methane_peak_ppb": 18.0})
        em = _events_map(_latest_visit()["visit_id"])
        assert ("entry", "methane") in em

    def test_weight_event_unit_is_g(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image), "weight_pre_g": 5800.0})
        em = _events_map(_latest_visit()["visit_id"])
        assert em[("pre_entry", "weight")]["unit"] == "g"

    def test_ammonia_event_unit_is_ppb(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image), "ammonia_peak_ppb": 95.0})
        em = _events_map(_latest_visit()["visit_id"])
        assert em[("entry", "ammonia")]["unit"] == "ppb"

    def test_methane_event_unit_is_ppb(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({"image_path": str(litter_image), "methane_peak_ppb": 18.0})
        em = _events_map(_latest_visit()["visit_id"])
        assert em[("entry", "methane")]["unit"] == "ppb"

    def test_event_values_correct(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({
            "image_path":       str(litter_image),
            "weight_pre_g":     5800.0,
            "weight_entry_g":   10050.0,
            "ammonia_peak_ppb": 95.0,
            "methane_peak_ppb": 18.0,
        })
        em = _events_map(_latest_visit()["visit_id"])
        assert abs(em[("pre_entry", "weight")]["value_numeric"] - 5800.0)  < 0.01
        assert abs(em[("entry",     "weight")]["value_numeric"] - 10050.0) < 0.01
        assert abs(em[("entry",     "ammonia")]["value_numeric"] - 95.0)   < 0.01
        assert abs(em[("entry",     "methane")]["value_numeric"] - 18.0)   < 0.01

    def test_full_sensor_set_produces_four_events(self, litter_image):
        from litterbox.tools import record_entry
        record_entry.invoke({
            "image_path":       str(litter_image),
            "weight_pre_g":     5800.0,
            "weight_entry_g":   10050.0,
            "ammonia_peak_ppb": 95.0,
            "methane_peak_ppb": 18.0,
        })
        # pre_entry weight + entry weight + entry ammonia + entry methane
        assert len(_sensor_events(_latest_visit()["visit_id"])) == 4


# ---------------------------------------------------------------------------
# record_exit — orphan exits
# ---------------------------------------------------------------------------

class TestRecordExitOrphan:
    @pytest.fixture(autouse=True)
    def _no_gpt4o(self, monkeypatch):
        monkeypatch.setattr(
            "litterbox.tools._run_gpt4o_vision",
            lambda *a, **kw: _HEALTH_NORMAL,
        )

    def test_warning_returned_when_no_open_visit(self, litter_image):
        from litterbox.tools import record_exit
        result = record_exit.invoke({"image_path": str(litter_image)})
        assert "WARNING" in result or "orphan" in result.lower()

    def test_orphan_row_created(self, litter_image):
        from litterbox.tools import record_exit
        record_exit.invoke({"image_path": str(litter_image)})
        assert bool(_latest_visit()["is_orphan_exit"])

    def test_orphan_stores_weight_exit(self, litter_image):
        from litterbox.tools import record_exit
        record_exit.invoke({"image_path": str(litter_image), "weight_exit_g": 5868.0})
        assert abs(_latest_visit()["weight_exit_g"] - 5868.0) < 0.01

    def test_orphan_stores_ammonia(self, litter_image):
        from litterbox.tools import record_exit
        record_exit.invoke({"image_path": str(litter_image), "ammonia_peak_ppb": 310.0})
        assert abs(_latest_visit()["ammonia_peak_ppb"] - 310.0) < 0.01

    def test_orphan_stores_methane(self, litter_image):
        from litterbox.tools import record_exit
        record_exit.invoke({"image_path": str(litter_image), "methane_peak_ppb": 55.0})
        assert abs(_latest_visit()["methane_peak_ppb"] - 55.0) < 0.01

    def test_nonexistent_exit_image_returns_error(self):
        from litterbox.tools import record_exit
        result = record_exit.invoke({"image_path": "/no/such/exit.jpg"})
        assert "Error" in result


# ---------------------------------------------------------------------------
# record_exit — normal close (no sensor data)
# ---------------------------------------------------------------------------

class TestRecordExitNormalClose:
    @pytest.fixture(autouse=True)
    def _stubs(self, monkeypatch):
        monkeypatch.setattr(
            "litterbox.tools._run_gpt4o_vision",
            lambda *a, **kw: _HEALTH_NORMAL,
        )

    def test_closes_visit(self, litter_image, open_visit):
        from litterbox.tools import record_exit
        result = record_exit.invoke({"image_path": str(litter_image)})
        assert "closed" in result

    def test_exit_time_recorded(self, litter_image, open_visit):
        from litterbox.tools import record_exit
        record_exit.invoke({"image_path": str(litter_image)})
        with get_conn() as conn:
            row = conn.execute(
                "SELECT exit_time FROM visits WHERE visit_id=?", (open_visit,)
            ).fetchone()
        assert row["exit_time"] is not None

    def test_health_notes_stored(self, litter_image, open_visit):
        from litterbox.tools import record_exit
        record_exit.invoke({"image_path": str(litter_image)})
        with get_conn() as conn:
            row = conn.execute(
                "SELECT health_notes FROM visits WHERE visit_id=?", (open_visit,)
            ).fetchone()
        assert row["health_notes"] is not None

    def test_is_anomalous_false_for_normal_response(self, litter_image, open_visit):
        from litterbox.tools import record_exit
        record_exit.invoke({"image_path": str(litter_image)})
        with get_conn() as conn:
            row = conn.execute(
                "SELECT is_anomalous FROM visits WHERE visit_id=?", (open_visit,)
            ).fetchone()
        assert not bool(row["is_anomalous"])

    def test_anomaly_flagged_for_anomaly_response(self, litter_image, open_visit,
                                                   monkeypatch):
        monkeypatch.setattr(
            "litterbox.tools._run_gpt4o_vision",
            lambda *a, **kw: _HEALTH_ANOMALY,
        )
        from litterbox.tools import record_exit
        record_exit.invoke({"image_path": str(litter_image)})
        with get_conn() as conn:
            row = conn.execute(
                "SELECT is_anomalous FROM visits WHERE visit_id=?", (open_visit,)
            ).fetchone()
        assert bool(row["is_anomalous"])

    def test_no_sensor_events_when_no_data_given(self, litter_image, open_visit):
        from litterbox.tools import record_exit
        record_exit.invoke({"image_path": str(litter_image)})
        assert len(_sensor_events(open_visit)) == 0

    def test_output_includes_visit_id(self, litter_image, open_visit):
        from litterbox.tools import record_exit
        result = record_exit.invoke({"image_path": str(litter_image)})
        assert str(open_visit) in result


# ---------------------------------------------------------------------------
# record_exit — sensor data storage and derived values
# ---------------------------------------------------------------------------

class TestRecordExitWithSensors:
    @pytest.fixture(autouse=True)
    def _stubs(self, monkeypatch):
        monkeypatch.setattr(
            "litterbox.tools._identify_cat",
            lambda path: (None, None, 0.3, "No reference images."),
        )
        monkeypatch.setattr(
            "litterbox.tools._run_gpt4o_vision",
            lambda *a, **kw: _HEALTH_NORMAL,
        )

    def test_stores_weight_exit_g(self, litter_image, open_visit):
        from litterbox.tools import record_exit
        record_exit.invoke({"image_path": str(litter_image), "weight_exit_g": 5868.0})
        with get_conn() as conn:
            row = conn.execute(
                "SELECT weight_exit_g FROM visits WHERE visit_id=?", (open_visit,)
            ).fetchone()
        assert abs(row["weight_exit_g"] - 5868.0) < 0.01

    def test_computes_waste_weight_when_pre_known(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({
            "image_path":   str(litter_image),
            "weight_pre_g": 5800.0,
        })
        record_exit.invoke({"image_path": str(litter_image), "weight_exit_g": 5868.0})
        assert abs(_latest_visit()["waste_weight_g"] - 68.0) < 0.01

    def test_waste_weight_null_without_pre(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image)})  # no weight_pre_g
        record_exit.invoke({"image_path": str(litter_image), "weight_exit_g": 5868.0})
        assert _latest_visit()["waste_weight_g"] is None

    def test_peak_gas_exit_higher_wins(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image), "ammonia_peak_ppb": 95.0})
        record_exit.invoke({"image_path": str(litter_image), "ammonia_peak_ppb": 310.0})
        assert abs(_latest_visit()["ammonia_peak_ppb"] - 310.0) < 0.01

    def test_peak_gas_entry_higher_retained(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image), "ammonia_peak_ppb": 500.0})
        record_exit.invoke({"image_path": str(litter_image), "ammonia_peak_ppb": 100.0})
        assert abs(_latest_visit()["ammonia_peak_ppb"] - 500.0) < 0.01

    def test_peak_gas_exit_only_stored_directly(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image)})  # no gas at entry
        record_exit.invoke({"image_path": str(litter_image), "methane_peak_ppb": 55.0})
        assert abs(_latest_visit()["methane_peak_ppb"] - 55.0) < 0.01

    def test_peak_gas_entry_only_retained_if_no_exit(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image), "methane_peak_ppb": 30.0})
        record_exit.invoke({"image_path": str(litter_image)})  # no methane at exit
        assert abs(_latest_visit()["methane_peak_ppb"] - 30.0) < 0.01

    def test_sensor_output_shown_in_result(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({
            "image_path":   str(litter_image),
            "weight_pre_g": 5800.0,
        })
        result = record_exit.invoke({
            "image_path":     str(litter_image),
            "weight_exit_g":  5868.0,
            "ammonia_peak_ppb": 310.0,
        })
        assert "68" in result or "waste" in result.lower()


# ---------------------------------------------------------------------------
# record_exit — visit_sensor_events logging
# ---------------------------------------------------------------------------

class TestRecordExitSensorEvents:
    @pytest.fixture(autouse=True)
    def _stubs(self, monkeypatch):
        monkeypatch.setattr(
            "litterbox.tools._identify_cat",
            lambda path: (None, None, 0.3, "No reference images."),
        )
        monkeypatch.setattr(
            "litterbox.tools._run_gpt4o_vision",
            lambda *a, **kw: _HEALTH_NORMAL,
        )

    def test_exit_weight_event_logged(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image)})
        vid = _latest_visit()["visit_id"]
        record_exit.invoke({"image_path": str(litter_image), "weight_exit_g": 5868.0})
        em = _events_map(vid)
        assert ("exit", "weight") in em

    def test_exit_ammonia_event_logged(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image)})
        vid = _latest_visit()["visit_id"]
        record_exit.invoke({"image_path": str(litter_image), "ammonia_peak_ppb": 310.0})
        em = _events_map(vid)
        assert ("exit", "ammonia") in em

    def test_exit_methane_event_logged(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image)})
        vid = _latest_visit()["visit_id"]
        record_exit.invoke({"image_path": str(litter_image), "methane_peak_ppb": 55.0})
        em = _events_map(vid)
        assert ("exit", "methane") in em

    def test_exit_event_values_correct(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({"image_path": str(litter_image)})
        vid = _latest_visit()["visit_id"]
        record_exit.invoke({
            "image_path":       str(litter_image),
            "weight_exit_g":    5868.0,
            "ammonia_peak_ppb": 310.0,
            "methane_peak_ppb": 55.0,
        })
        em = _events_map(vid)
        assert abs(em[("exit", "weight")]["value_numeric"]  - 5868.0) < 0.01
        assert abs(em[("exit", "ammonia")]["value_numeric"] - 310.0)  < 0.01
        assert abs(em[("exit", "methane")]["value_numeric"] - 55.0)   < 0.01

    def test_combined_entry_and_exit_events(self, litter_image):
        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({
            "image_path":       str(litter_image),
            "weight_pre_g":     5800.0,
            "weight_entry_g":   10050.0,
            "ammonia_peak_ppb": 95.0,
        })
        vid = _latest_visit()["visit_id"]
        record_exit.invoke({
            "image_path":       str(litter_image),
            "weight_exit_g":    5868.0,
            "ammonia_peak_ppb": 310.0,
        })
        em = _events_map(vid)
        # Entry phase: pre_entry weight + entry weight + entry ammonia
        # Exit phase: exit weight + exit ammonia
        assert ("pre_entry", "weight") in em
        assert ("entry",     "weight") in em
        assert ("entry",     "ammonia") in em
        assert ("exit",      "weight") in em
        assert ("exit",      "ammonia") in em


# ---------------------------------------------------------------------------
# record_exit — health prompt enrichment
# ---------------------------------------------------------------------------

class TestRecordExitHealthPromptEnrichment:
    """Verify that sensor readings flow into the GPT-4o prompt."""

    def test_sensor_data_appears_in_health_prompt(self, litter_image, monkeypatch):
        captured = []

        def _capture(prompt, *paths):
            captured.append(prompt)
            return (
                "CONCERNS_PRESENT: no\n"
                "DESCRIPTION: fine\n"
                "OWNER_SUMMARY: No visual abnormalities detected."
            )

        monkeypatch.setattr("litterbox.tools._identify_cat",
                            lambda path: (None, None, 0.3, "No match"))
        monkeypatch.setattr("litterbox.tools._run_gpt4o_vision", _capture)

        from litterbox.tools import record_entry, record_exit
        record_entry.invoke({
            "image_path":       str(litter_image),
            "weight_pre_g":     5800.0,
            "weight_entry_g":   10050.0,
            "ammonia_peak_ppb": 95.0,
        })
        record_exit.invoke({
            "image_path":       str(litter_image),
            "weight_exit_g":    5868.0,
            "ammonia_peak_ppb": 310.0,
        })

        assert len(captured) == 1
        prompt = captured[0]
        assert "5800"  in prompt   # weight_pre_g
        assert "4250"  in prompt   # cat_weight_g (10050 - 5800)
        assert "310"   in prompt   # peak ammonia (exit was higher)
        assert "68"    in prompt   # waste_weight_g (5868 - 5800)

    def test_prompt_without_sensors_still_valid(self, litter_image, open_visit, monkeypatch):
        captured = []

        def _capture(prompt, *paths):
            captured.append(prompt)
            return (
                "CONCERNS_PRESENT: no\n"
                "DESCRIPTION: fine\n"
                "OWNER_SUMMARY: No visual abnormalities detected."
            )

        monkeypatch.setattr("litterbox.tools._run_gpt4o_vision", _capture)

        from litterbox.tools import record_exit
        record_exit.invoke({"image_path": str(litter_image)})

        assert "CONCERNS_PRESENT" in captured[0]   # format still present
        assert "sensor data" not in captured[0].lower()
