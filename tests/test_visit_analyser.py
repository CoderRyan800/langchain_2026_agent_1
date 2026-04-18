"""
Tests for visit_analyser.py — Step 4 of the Time-Domain Measurement System
===========================================================================

Covers:
- Chip-ID priority identification
- Similarity DataFrame analysis with sustained-peak gate
- Unknown fallback
- save() round-trip to td_visits table
- Snapshot JSON serialisation
- Image retention sweep
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from litterbox.visit_analyser import (
    TdVisitRecord,
    VisitAnalyser,
    _deserialize_snapshot,
    _serialize_snapshot,
)
from litterbox.image_retention import sweep_old_visit_images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 4, 4, 12, 0, 0, tzinfo=timezone.utc)


def _ts(offset_seconds: float = 0.0) -> datetime:
    """Return a deterministic UTC timestamp offset from a fixed base."""
    return _BASE + timedelta(seconds=offset_seconds)


def _make_config(
    sim_entry_threshold: float = 0.70,
    sustained_peak_samples: int = 3,
) -> dict:
    """Build a minimal td_config-like dict for VisitAnalyser."""
    return {
        "trigger": {
            "similarity_entry_threshold": sim_entry_threshold,
            "similarity_sustained_peak_samples": sustained_peak_samples,
        },
    }


def _entry(offset: float, values: dict) -> dict:
    """Create a single snapshot entry."""
    return {"timestamp": _ts(offset), "values": dict(values)}


# ---------------------------------------------------------------------------
# TestChipIdPriority
# ---------------------------------------------------------------------------


class TestChipIdPriority:
    """Chip-based identification takes priority over similarity."""

    def test_chip_id_present(self):
        """A non-null chip_id in the visit window → is_confirmed=True."""
        config = _make_config()
        analyser = VisitAnalyser(config)

        snapshot = [
            _entry(0, {"weight_g": 5000}),
            _entry(5, {"weight_g": 5500, "chip_id": "Whiskers"}),
            _entry(10, {"weight_g": 5500, "chip_id": "Whiskers"}),
            _entry(15, {"weight_g": 5000}),
        ]

        record = analyser.analyse(snapshot, _ts(0), _ts(15))

        assert record.id_method == "chip"
        assert record.chip_id == "Whiskers"
        assert record.is_confirmed is True

    def test_multiple_chip_ids_most_frequent_wins(self):
        """When multiple chip_ids appear, the most frequent one wins."""
        config = _make_config()
        analyser = VisitAnalyser(config)

        snapshot = [
            _entry(0, {"chip_id": "Luna"}),
            _entry(5, {"chip_id": "Luna"}),
            _entry(10, {"chip_id": "Anna"}),
            _entry(15, {"chip_id": "Luna"}),
        ]

        record = analyser.analyse(snapshot, _ts(0), _ts(15))

        assert record.chip_id == "Luna"
        assert record.is_confirmed is True

    def test_chip_id_looks_up_cat(self, registered_cat):
        """Chip ID matching a registered cat populates cat_id fields."""
        cat_id, cat_name = registered_cat
        config = _make_config()
        analyser = VisitAnalyser(config)

        snapshot = [
            _entry(0, {"chip_id": cat_name}),
            _entry(5, {"chip_id": cat_name}),
        ]

        record = analyser.analyse(snapshot, _ts(0), _ts(5))

        assert record.tentative_cat_id == cat_id
        assert record.confirmed_cat_id == cat_id
        assert record.is_confirmed is True

    def test_chip_id_unregistered_cat(self):
        """Chip ID for an unregistered cat: is_confirmed=True but cat_id=None."""
        config = _make_config()
        analyser = VisitAnalyser(config)

        snapshot = [
            _entry(0, {"chip_id": "UnknownCat"}),
        ]

        record = analyser.analyse(snapshot, _ts(0), _ts(0))

        assert record.chip_id == "UnknownCat"
        assert record.is_confirmed is True
        assert record.tentative_cat_id is None
        assert record.confirmed_cat_id is None

    def test_chip_id_takes_priority_over_similarity(self):
        """Even if similarity is high, chip ID wins."""
        config = _make_config()
        analyser = VisitAnalyser(config)

        snapshot = [
            _entry(0, {"chip_id": "Luna", "similarity_anna": 0.95}),
            _entry(5, {"chip_id": "Luna", "similarity_anna": 0.94}),
            _entry(10, {"chip_id": "Luna", "similarity_anna": 0.93}),
        ]

        record = analyser.analyse(snapshot, _ts(0), _ts(10))

        assert record.id_method == "chip"
        assert record.chip_id == "Luna"


# ---------------------------------------------------------------------------
# TestSimilarityAnalysis
# ---------------------------------------------------------------------------


class TestSimilarityAnalysis:
    """Similarity-based identification via DataFrame analysis."""

    def test_winning_cat_above_threshold_with_sustained_peak(self):
        """Cat with highest mean and P consecutive peaks → tentative ID."""
        config = _make_config(sim_entry_threshold=0.70, sustained_peak_samples=3)
        analyser = VisitAnalyser(config)

        snapshot = [
            _entry(0, {"similarity_anna": 0.85, "similarity_luna": 0.30}),
            _entry(5, {"similarity_anna": 0.88, "similarity_luna": 0.28}),
            _entry(10, {"similarity_anna": 0.82, "similarity_luna": 0.32}),
            _entry(15, {"similarity_anna": 0.90, "similarity_luna": 0.25}),
        ]

        record = analyser.analyse(snapshot, _ts(0), _ts(15))

        assert record.id_method == "similarity"
        assert record.is_confirmed is False
        assert record.top_similarity is not None
        assert record.top_similarity > 0.80

    def test_sustained_peak_not_met(self):
        """Cat exceeds threshold on average but never P consecutive → Unknown."""
        config = _make_config(sim_entry_threshold=0.70, sustained_peak_samples=3)
        analyser = VisitAnalyser(config)

        # Anna alternates above/below threshold — never 3 consecutive.
        snapshot = [
            _entry(0, {"similarity_anna": 0.80}),
            _entry(5, {"similarity_anna": 0.50}),
            _entry(10, {"similarity_anna": 0.80}),
            _entry(15, {"similarity_anna": 0.50}),
            _entry(20, {"similarity_anna": 0.80}),
            _entry(25, {"similarity_anna": 0.50}),
        ]

        record = analyser.analyse(snapshot, _ts(0), _ts(25))

        assert record.id_method == "unknown"

    def test_all_below_threshold(self):
        """All cats below threshold → Unknown."""
        config = _make_config(sim_entry_threshold=0.70)
        analyser = VisitAnalyser(config)

        snapshot = [
            _entry(0, {"similarity_anna": 0.40, "similarity_luna": 0.35}),
            _entry(5, {"similarity_anna": 0.42, "similarity_luna": 0.38}),
            _entry(10, {"similarity_anna": 0.39, "similarity_luna": 0.33}),
        ]

        record = analyser.analyse(snapshot, _ts(0), _ts(10))

        assert record.id_method == "unknown"

    def test_nan_frames_dont_bias_mean(self):
        """Missing frames (absent keys → NaN) should not lower the mean."""
        config = _make_config(sim_entry_threshold=0.70, sustained_peak_samples=2)
        analyser = VisitAnalyser(config)

        # 3 valid frames with anna ~0.85, 2 missing frames.
        # Mean with NaN skipped should be ~0.85, not lowered by zeros.
        snapshot = [
            _entry(0, {"similarity_anna": 0.85}),
            _entry(5, {}),  # missing frame
            _entry(10, {"similarity_anna": 0.86}),
            _entry(15, {}),  # missing frame
            _entry(20, {"similarity_anna": 0.84}),
        ]

        record = analyser.analyse(snapshot, _ts(0), _ts(20))

        assert record.id_method == "similarity"
        assert record.top_similarity is not None
        assert record.top_similarity > 0.80

    def test_similarity_looks_up_cat(self, registered_cat):
        """Winning similarity cat matching a registered cat populates tentative_cat_id."""
        cat_id, cat_name = registered_cat
        config = _make_config(sim_entry_threshold=0.70, sustained_peak_samples=2)
        analyser = VisitAnalyser(config)

        sim_key = f"similarity_{cat_name}"
        snapshot = [
            _entry(0, {sim_key: 0.90}),
            _entry(5, {sim_key: 0.88}),
            _entry(10, {sim_key: 0.92}),
        ]

        record = analyser.analyse(snapshot, _ts(0), _ts(10))

        assert record.tentative_cat_id == cat_id
        assert record.id_method == "similarity"

    def test_visit_window_filtering(self):
        """Only entries within [entry_time, exit_time] are used for ID."""
        config = _make_config(sim_entry_threshold=0.70, sustained_peak_samples=2)
        analyser = VisitAnalyser(config)

        # High similarity outside the visit window, low inside.
        snapshot = [
            _entry(0, {"similarity_anna": 0.95}),   # before visit window
            _entry(5, {"similarity_anna": 0.95}),   # before visit window
            _entry(10, {"similarity_anna": 0.30}),  # inside visit window
            _entry(15, {"similarity_anna": 0.28}),  # inside visit window
            _entry(20, {"similarity_anna": 0.95}),  # after visit window
        ]

        record = analyser.analyse(snapshot, _ts(10), _ts(15))

        assert record.id_method == "unknown"


# ---------------------------------------------------------------------------
# TestUnknownFallback
# ---------------------------------------------------------------------------


class TestUnknownFallback:
    """When no identification method succeeds, id_method = "unknown"."""

    def test_no_chip_no_similarity(self):
        """Weight-only snapshot → Unknown."""
        config = _make_config()
        analyser = VisitAnalyser(config)

        snapshot = [
            _entry(0, {"weight_g": 5000}),
            _entry(5, {"weight_g": 5500}),
            _entry(10, {"weight_g": 5000}),
        ]

        record = analyser.analyse(snapshot, _ts(0), _ts(10))

        assert record.id_method == "unknown"
        assert record.chip_id is None
        assert record.tentative_cat_id is None
        assert record.is_confirmed is False

    def test_empty_visit_window(self):
        """No entries in the visit window → Unknown."""
        config = _make_config()
        analyser = VisitAnalyser(config)

        # All entries are outside the visit window.
        snapshot = [
            _entry(0, {"chip_id": "Luna", "similarity_anna": 0.95}),
        ]

        record = analyser.analyse(snapshot, _ts(100), _ts(200))

        assert record.id_method == "unknown"


# ---------------------------------------------------------------------------
# TestSave
# ---------------------------------------------------------------------------


class TestSave:
    """save() inserts a row into td_visits and returns a valid td_visit_id."""

    def test_save_returns_valid_id(self):
        config = _make_config()
        analyser = VisitAnalyser(config)

        snapshot = [_entry(0, {"weight_g": 5000})]
        record = analyser.analyse(snapshot, _ts(0), _ts(0))

        td_visit_id = analyser.save(record)

        assert td_visit_id is not None
        assert td_visit_id > 0
        assert record.td_visit_id == td_visit_id

    def test_save_round_trip(self):
        """Values stored by save() can be read back from the DB."""
        from litterbox.db import get_conn

        config = _make_config()
        analyser = VisitAnalyser(config)

        snapshot = [
            _entry(0, {"weight_g": 5000, "chip_id": "TestCat"}),
            _entry(5, {"weight_g": 5500, "chip_id": "TestCat"}),
        ]
        record = analyser.analyse(snapshot, _ts(0), _ts(5))
        td_visit_id = analyser.save(record)

        with get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM td_visits WHERE td_visit_id = ?",
                (td_visit_id,),
            ).fetchone()

        assert row is not None
        assert row["id_method"] == "chip"
        assert row["chip_id"] == "TestCat"
        assert row["is_confirmed"] == 1  # SQLite stores bool as int
        assert row["snapshot_json"] is not None

    def test_save_multiple_visits(self):
        """Multiple saves produce distinct td_visit_ids."""
        config = _make_config()
        analyser = VisitAnalyser(config)

        ids = []
        for i in range(3):
            snapshot = [_entry(i * 100, {"weight_g": 5000})]
            record = analyser.analyse(
                snapshot, _ts(i * 100), _ts(i * 100)
            )
            ids.append(analyser.save(record))

        assert len(set(ids)) == 3


# ---------------------------------------------------------------------------
# TestSnapshotSerialization
# ---------------------------------------------------------------------------


class TestSnapshotSerialization:
    """JSON round-trip for buffer snapshots."""

    def test_round_trip(self):
        """Serialize → deserialize preserves structure."""
        snapshot = [
            {"timestamp": _ts(0), "values": {"weight_g": 5000.0, "chip_id": "Luna"}},
            {"timestamp": _ts(5), "values": {"weight_g": 5500.0}},
        ]

        json_str = _serialize_snapshot(snapshot)
        restored = _deserialize_snapshot(json_str)

        assert len(restored) == 2
        assert restored[0]["values"]["weight_g"] == 5000.0
        assert restored[0]["values"]["chip_id"] == "Luna"
        assert restored[1]["values"]["weight_g"] == 5500.0

    def test_timestamps_are_iso_strings(self):
        """Timestamps are serialized as ISO strings."""
        snapshot = [{"timestamp": _ts(0), "values": {}}]

        json_str = _serialize_snapshot(snapshot)
        restored = _deserialize_snapshot(json_str)

        ts_str = restored[0]["timestamp"]
        assert isinstance(ts_str, str)
        # Verify it parses back to a valid datetime.
        dt = datetime.fromisoformat(ts_str)
        assert dt.year == 2026

    def test_empty_snapshot(self):
        """Empty snapshot serializes to an empty JSON array."""
        json_str = _serialize_snapshot([])
        restored = _deserialize_snapshot(json_str)
        assert restored == []


# ---------------------------------------------------------------------------
# TestImageRetention
# ---------------------------------------------------------------------------


class TestImageRetention:
    """sweep_old_visit_images deletes old directories, keeps recent ones."""

    def test_deletes_old_keeps_new(self, tmp_path):
        """Directories older than retention_days are deleted; newer are kept."""
        from datetime import date

        visits_dir = tmp_path / "images" / "visits"
        today = date.today()
        old_date = (today - timedelta(days=10)).isoformat()
        new_date = today.isoformat()

        (visits_dir / old_date / "visit_abc").mkdir(parents=True)
        (visits_dir / new_date / "visit_def").mkdir(parents=True)

        deleted = sweep_old_visit_images(tmp_path / "images", retention_days=7)

        assert deleted == 1
        assert not (visits_dir / old_date).exists()
        assert (visits_dir / new_date).exists()

    def test_empty_base_dir(self, tmp_path):
        """No visits directory → returns 0, no crash."""
        deleted = sweep_old_visit_images(tmp_path / "images", retention_days=7)
        assert deleted == 0

    def test_non_date_directories_ignored(self, tmp_path):
        """Directories that don't parse as dates are left alone."""
        visits_dir = tmp_path / "images" / "visits"
        (visits_dir / "not-a-date").mkdir(parents=True)
        (visits_dir / "README.txt").mkdir(parents=True)

        deleted = sweep_old_visit_images(tmp_path / "images", retention_days=0)

        assert deleted == 0
        assert (visits_dir / "not-a-date").exists()

    def test_boundary_date_not_deleted(self, tmp_path):
        """A directory exactly at the cutoff boundary is NOT deleted."""
        from datetime import date

        visits_dir = tmp_path / "images" / "visits"
        cutoff_date = (date.today() - timedelta(days=7)).isoformat()
        (visits_dir / cutoff_date).mkdir(parents=True)

        deleted = sweep_old_visit_images(tmp_path / "images", retention_days=7)

        # cutoff = today - 7 days; directory date == cutoff → NOT older → keep.
        assert deleted == 0
