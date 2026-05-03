"""
Tests for the rescore_all_visits utility.

Verifies that visits with stale gas-anomaly verdicts are corrected,
that the LLM-side verdict is preserved where health_notes followed the
format, that dry-run does not write, and that the function is idempotent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from litterbox.db import get_conn
from litterbox.rescore import rescore_all_visits, _llm_verdict_from_notes


def _insert_visit(
    conn,
    cat_id,
    *,
    days_ago=1,
    ammonia=None,
    methane=None,
    is_anomalous=False,
    nh3_z=None,
    ch4_z=None,
    tier=None,
    health_notes=None,
):
    ts = (datetime.now(timezone.utc).replace(tzinfo=None)
          - timedelta(days=days_ago)).isoformat()
    cur = conn.execute(
        """INSERT INTO visits
           (entry_time, tentative_cat_id, is_anomalous,
            ammonia_peak_ppb, methane_peak_ppb,
            ammonia_z_score, methane_z_score,
            gas_anomaly_tier, health_notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ts, cat_id, is_anomalous, ammonia, methane,
         nh3_z, ch4_z, tier, health_notes),
    )
    return cur.lastrowid


# ---------------------------------------------------------------------------
# _llm_verdict_from_notes
# ---------------------------------------------------------------------------

class TestLlmVerdictFromNotes:
    def test_none_returns_false(self):
        assert _llm_verdict_from_notes(None) is False

    def test_empty_returns_false(self):
        assert _llm_verdict_from_notes("") is False

    def test_yes_marker_returns_true(self):
        assert _llm_verdict_from_notes(
            "CONCERNS_PRESENT: yes\nDESCRIPTION: blood"
        ) is True

    def test_no_marker_returns_false(self):
        assert _llm_verdict_from_notes(
            "CONCERNS_PRESENT: no\nDESCRIPTION: clean"
        ) is False

    def test_refusal_text_returns_false(self):
        assert _llm_verdict_from_notes(
            "I'm sorry, I can't assist with images containing people."
        ) is False

    def test_placeholder_returns_false(self):
        # The placeholder safe_health_notes writes for unstructured replies.
        assert _llm_verdict_from_notes(
            "Health analysis unavailable — GPT-4o did not return a "
            "structured response for this visit."
        ) is False


# ---------------------------------------------------------------------------
# rescore_all_visits
# ---------------------------------------------------------------------------

class TestRescoreEmptyDb:
    def test_empty_db_zeros_everywhere(self):
        with get_conn() as conn:
            summary = rescore_all_visits(conn=conn, dry_run=False)
        assert summary["seen"] == 0
        assert summary["changed"] == 0
        assert summary["unchanged"] == 0
        assert summary["changes"] == []


class TestRescoreUpdatesStaleScores:
    def test_stale_z_scores_overwritten(self, registered_cat):
        cat_id, _ = registered_cat
        with get_conn() as conn:
            # Build a clean per-cat history (all NH₃ ≈ 20, CH₄ ≈ 10) so the
            # detector has a tight model.
            for d in range(2, 14):
                _insert_visit(conn, cat_id, days_ago=d,
                              ammonia=20.0 + (d % 3), methane=10.0 + (d % 2))
            # Insert one obviously-anomalous visit but with stale stored
            # values from a "previous detector" that thought it was normal.
            stale_id = _insert_visit(
                conn, cat_id, days_ago=1,
                ammonia=500.0, methane=200.0,
                nh3_z=0.5, ch4_z=0.4,
                tier="normal", is_anomalous=False,
                health_notes="CONCERNS_PRESENT: no\nDESCRIPTION: clean",
            )

            summary = rescore_all_visits(conn=conn, dry_run=False)

            row = conn.execute(
                "SELECT ammonia_z_score, methane_z_score, gas_anomaly_tier, "
                "is_anomalous, gas_anomaly_rescored_at FROM visits "
                "WHERE visit_id = ?", (stale_id,)
            ).fetchone()

        # The 500/200 readings sit far above the cluster around 20/10, so
        # the rescore must produce an alarm tier and flip is_anomalous.
        assert row["gas_anomaly_tier"] in {"mild", "significant", "severe"}
        assert row["ammonia_z_score"] is not None
        assert row["ammonia_z_score"] > 2.0
        assert row["is_anomalous"] == 1
        assert row["gas_anomaly_rescored_at"] is not None
        assert summary["flag_changed_to_true"] >= 1


class TestRescorePreservesLlmYes:
    def test_llm_anomaly_kept_even_when_gas_normal(self, registered_cat):
        cat_id, _ = registered_cat
        with get_conn() as conn:
            # Tight cluster — every visit's gas score will be normal.
            for d in range(2, 14):
                _insert_visit(conn, cat_id, days_ago=d,
                              ammonia=20.0 + (d % 3), methane=10.0 + (d % 2))
            # One row whose LLM verdict says yes but gas reading sits
            # right inside the normal cluster.
            llm_yes_id = _insert_visit(
                conn, cat_id, days_ago=1,
                ammonia=20.0, methane=10.0,
                tier="normal", is_anomalous=False,
                health_notes="CONCERNS_PRESENT: yes\nDESCRIPTION: blood seen",
            )

            rescore_all_visits(conn=conn, dry_run=False)

            row = conn.execute(
                "SELECT gas_anomaly_tier, is_anomalous FROM visits "
                "WHERE visit_id = ?", (llm_yes_id,)
            ).fetchone()

        # Gas tier is normal (low z) but is_anomalous stays True because
        # the LLM said yes — the OR layer must hold.
        assert row["gas_anomaly_tier"] == "normal"
        assert row["is_anomalous"] == 1

    def test_refusal_text_does_not_force_llm_anomalous(self, registered_cat):
        cat_id, _ = registered_cat
        with get_conn() as conn:
            for d in range(2, 14):
                _insert_visit(conn, cat_id, days_ago=d,
                              ammonia=20.0 + (d % 3), methane=10.0 + (d % 2))
            # Refusal text + normal-range gas → not anomalous.
            refusal_id = _insert_visit(
                conn, cat_id, days_ago=1,
                ammonia=20.0, methane=10.0,
                tier="normal", is_anomalous=True,
                health_notes="I'm sorry, I can't assist with these images.",
            )

            rescore_all_visits(conn=conn, dry_run=False)

            row = conn.execute(
                "SELECT is_anomalous FROM visits WHERE visit_id = ?",
                (refusal_id,)
            ).fetchone()

        assert row["is_anomalous"] == 0


class TestRescoreDryRun:
    def test_dry_run_does_not_write(self, registered_cat):
        cat_id, _ = registered_cat
        with get_conn() as conn:
            for d in range(2, 14):
                _insert_visit(conn, cat_id, days_ago=d,
                              ammonia=20.0 + (d % 3), methane=10.0 + (d % 2))
            stale_id = _insert_visit(
                conn, cat_id, days_ago=1,
                ammonia=500.0, methane=200.0,
                nh3_z=0.5, ch4_z=0.4,
                tier="normal", is_anomalous=False,
                health_notes="CONCERNS_PRESENT: no",
            )

            summary = rescore_all_visits(conn=conn, dry_run=True)
            row = conn.execute(
                "SELECT ammonia_z_score, gas_anomaly_tier, is_anomalous, "
                "gas_anomaly_rescored_at FROM visits WHERE visit_id = ?",
                (stale_id,),
            ).fetchone()

        # Summary still reports the would-be change.
        assert summary["dry_run"] is True
        assert summary["changed"] >= 1
        # But the row is untouched.
        assert row["ammonia_z_score"] == pytest.approx(0.5)
        assert row["gas_anomaly_tier"] == "normal"
        assert row["is_anomalous"] == 0
        assert row["gas_anomaly_rescored_at"] is None


class TestRescoreIdempotent:
    def test_second_pass_changes_nothing(self, registered_cat):
        cat_id, _ = registered_cat
        with get_conn() as conn:
            for d in range(2, 14):
                _insert_visit(conn, cat_id, days_ago=d,
                              ammonia=20.0 + (d % 3), methane=10.0 + (d % 2))
            _insert_visit(conn, cat_id, days_ago=1,
                          ammonia=500.0, methane=200.0,
                          tier="normal", is_anomalous=False,
                          health_notes="CONCERNS_PRESENT: no")

            first = rescore_all_visits(conn=conn, dry_run=False)
            second = rescore_all_visits(conn=conn, dry_run=False)

        assert first["changed"] >= 1
        assert second["changed"] == 0
        assert second["unchanged"] == first["seen"]


class TestRescoreSkipsNullSensorRows:
    def test_visit_without_any_gas_data_is_ignored(self, registered_cat):
        cat_id, _ = registered_cat
        with get_conn() as conn:
            _insert_visit(conn, cat_id, days_ago=1,
                          ammonia=None, methane=None,
                          health_notes=None)
            summary = rescore_all_visits(conn=conn, dry_run=False)
        assert summary["seen"] == 0


class TestRescoreInsufficientData:
    def test_first_visit_for_cat_gets_insufficient_tier(self, registered_cat):
        # A single visit with no history fits no model → tier stays
        # insufficient_data even after rescore.
        cat_id, _ = registered_cat
        with get_conn() as conn:
            vid = _insert_visit(conn, cat_id, days_ago=1,
                                ammonia=200.0, methane=100.0,
                                tier="severe", is_anomalous=True,
                                health_notes="CONCERNS_PRESENT: no")
            rescore_all_visits(conn=conn, dry_run=False)
            row = conn.execute(
                "SELECT gas_anomaly_tier, is_anomalous FROM visits "
                "WHERE visit_id = ?", (vid,)
            ).fetchone()
        # No history → no model → insufficient_data.
        assert row["gas_anomaly_tier"] == "insufficient_data"
        # LLM said no, gas tier is insufficient_data → not anomalous.
        assert row["is_anomalous"] == 0
