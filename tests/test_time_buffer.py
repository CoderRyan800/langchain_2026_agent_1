"""
test_time_buffer.py — pytest suite for Step 1 of the time-domain system
========================================================================

Tests cover:
  * RollingBuffer capacity / eviction behaviour
  * get_channel — correct values and None for absent keys
  * get_timestamps
  * window_span_seconds
  * snapshot immutability
  * is_full / __len__ / __repr__ / clear
  * to_dataframe — full buffer, channel_prefix, NaN handling, time slicing
  * NaN correctness: verify that missing frames stored as NaN, not 0.0,
    give the correct per-column mean (justification test)
  * Similarity DataFrame serialisation round-trip (to_json / read_json)
  * load_td_config — success, FileNotFoundError, missing key, wrong type
  * Config round-trip: check every top-level required field
  * Thread-safety smoke test: concurrent appends do not corrupt the buffer

All tests are pure-Python (no hardware, no LLM, no network calls).
Run with:  pytest tests/test_time_buffer.py -v
"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the modules under test.
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from litterbox.time_buffer import RollingBuffer, load_td_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc

def _ts(offset_seconds: float = 0.0) -> datetime:
    """Return a fixed base UTC datetime offset by the given number of seconds."""
    base = datetime(2026, 4, 1, 12, 0, 0, tzinfo=_UTC)
    return base + timedelta(seconds=offset_seconds)


def _fill(buf: RollingBuffer, n: int, channel: str = "val",
          start_offset: float = 0.0, interval: float = 5.0) -> None:
    """Append n entries to buf with increasing timestamps and values 0..n-1."""
    for i in range(n):
        buf.append(_ts(start_offset + i * interval), {channel: float(i)})


# ===========================================================================
# RollingBuffer — capacity and eviction
# ===========================================================================

class TestCapacity:
    """Buffer must never hold more than max_len entries."""

    def test_capacity_not_exceeded(self):
        """Inserting more entries than max_len never exceeds capacity."""
        buf = RollingBuffer(window_minutes=2, samples_per_minute=6)  # max 12
        _fill(buf, 20)
        assert len(buf) == 12

    def test_oldest_entries_evicted(self):
        """After overflow the oldest entries (lowest values) are gone."""
        buf = RollingBuffer(window_minutes=1, samples_per_minute=10)  # max 10
        _fill(buf, 13)  # insert 13; oldest 3 (values 0,1,2) should be gone
        vals = buf.get_channel("val")
        assert vals[0] == 3.0,  f"Expected 3.0 as oldest, got {vals[0]}"
        assert vals[-1] == 12.0, f"Expected 12.0 as newest, got {vals[-1]}"

    def test_exact_full(self):
        """Inserting exactly max_len entries fills the buffer without eviction."""
        buf = RollingBuffer(window_minutes=1, samples_per_minute=5)  # max 5
        _fill(buf, 5)
        assert len(buf) == 5
        assert buf.is_full()

    def test_empty_initially(self):
        buf = RollingBuffer(window_minutes=10, samples_per_minute=12)
        assert len(buf) == 0
        assert not buf.is_full()

    def test_constructor_rejects_zero_window(self):
        with pytest.raises(ValueError, match="window_minutes"):
            RollingBuffer(window_minutes=0, samples_per_minute=12)

    def test_constructor_rejects_zero_rate(self):
        with pytest.raises(ValueError, match="samples_per_minute"):
            RollingBuffer(window_minutes=10, samples_per_minute=0)


# ===========================================================================
# get_channel
# ===========================================================================

class TestGetChannel:
    """get_channel returns values for present keys and None for absent keys."""

    def test_returns_present_values(self):
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        buf.append(_ts(0), {"weight_g": 4200.0, "ammonia_ppb": 18.5})
        buf.append(_ts(1), {"weight_g": 7100.0, "ammonia_ppb": 21.0})
        assert buf.get_channel("weight_g")   == [4200.0, 7100.0]
        assert buf.get_channel("ammonia_ppb") == [18.5, 21.0]

    def test_returns_none_for_absent_key(self):
        """A channel not present in a sample produces None, not 0."""
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        buf.append(_ts(0), {"weight_g": 4200.0})          # ammonia absent
        buf.append(_ts(1), {"weight_g": 7100.0, "ammonia_ppb": 21.0})
        channel = buf.get_channel("ammonia_ppb")
        assert channel[0] is None,  "Missing key must be None, not 0.0"
        assert channel[1] == 21.0

    def test_unknown_channel_all_none(self):
        """A channel that was never written returns all None."""
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        buf.append(_ts(0), {"weight_g": 4200.0})
        result = buf.get_channel("nonexistent")
        assert result == [None]

    def test_values_ordered_oldest_first(self):
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        for i in range(5):
            buf.append(_ts(i), {"x": float(i)})
        assert buf.get_channel("x") == [0.0, 1.0, 2.0, 3.0, 4.0]


# ===========================================================================
# get_timestamps
# ===========================================================================

class TestGetTimestamps:
    def test_returns_timestamps_in_order(self):
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        times = [_ts(i * 5) for i in range(4)]
        for t in times:
            buf.append(t, {"x": 1.0})
        assert buf.get_timestamps() == times

    def test_empty_buffer_returns_empty_list(self):
        buf = RollingBuffer(window_minutes=1, samples_per_minute=12)
        assert buf.get_timestamps() == []


# ===========================================================================
# window_span_seconds
# ===========================================================================

class TestWindowSpan:
    def test_correct_span(self):
        buf = RollingBuffer(window_minutes=5, samples_per_minute=12)
        buf.append(_ts(0),   {"x": 1})
        buf.append(_ts(30),  {"x": 2})
        buf.append(_ts(120), {"x": 3})
        assert buf.window_span_seconds() == pytest.approx(120.0)

    def test_single_entry_returns_zero(self):
        buf = RollingBuffer(window_minutes=1, samples_per_minute=12)
        buf.append(_ts(0), {"x": 1})
        assert buf.window_span_seconds() == 0.0

    def test_empty_buffer_returns_zero(self):
        buf = RollingBuffer(window_minutes=1, samples_per_minute=12)
        assert buf.window_span_seconds() == 0.0

    def test_span_reflects_eviction(self):
        """After the oldest entries are evicted, span shrinks accordingly."""
        buf = RollingBuffer(window_minutes=1, samples_per_minute=3)  # max 3
        # Insert entries at t=0, 5, 10, 15 — capacity 3 means t=0 is evicted
        for i in range(4):
            buf.append(_ts(i * 5), {"x": float(i)})
        # Buffer now holds t=5, 10, 15 → span = 10 s
        assert buf.window_span_seconds() == pytest.approx(10.0)


# ===========================================================================
# snapshot immutability
# ===========================================================================

class TestSnapshot:
    def test_snapshot_is_a_copy(self):
        """Mutating the returned snapshot list does not affect the buffer."""
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        buf.append(_ts(0), {"x": 1.0})
        snap = buf.snapshot()
        snap.clear()          # wipe the copy
        assert len(buf) == 1, "Buffer should be unchanged after mutating snapshot"

    def test_snapshot_values_are_copies(self):
        """Mutating a value dict inside the snapshot does not affect the buffer."""
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        buf.append(_ts(0), {"x": 1.0})
        snap = buf.snapshot()
        snap[0]["values"]["x"] = 999.0  # mutate the copy
        assert buf.get_channel("x") == [1.0], "Buffer value must not be affected"

    def test_snapshot_ordered_oldest_first(self):
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        for i in range(3):
            buf.append(_ts(i), {"x": float(i)})
        snap = buf.snapshot()
        xs = [s["values"]["x"] for s in snap]
        assert xs == [0.0, 1.0, 2.0]

    def test_snapshot_includes_timestamp(self):
        t = _ts(42)
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        buf.append(t, {"x": 7.0})
        snap = buf.snapshot()
        assert snap[0]["timestamp"] == t


# ===========================================================================
# clear
# ===========================================================================

class TestClear:
    def test_clear_empties_buffer(self):
        buf = RollingBuffer(window_minutes=1, samples_per_minute=12)
        _fill(buf, 10)
        buf.clear()
        assert len(buf) == 0

    def test_can_append_after_clear(self):
        buf = RollingBuffer(window_minutes=1, samples_per_minute=12)
        _fill(buf, 10)
        buf.clear()
        buf.append(_ts(0), {"x": 42.0})
        assert buf.get_channel("x") == [42.0]


# ===========================================================================
# __repr__
# ===========================================================================

class TestRepr:
    def test_repr_contains_key_info(self):
        buf = RollingBuffer(window_minutes=5, samples_per_minute=12)
        r = repr(buf)
        assert "5m" in r
        assert "12/min" in r
        assert "60" in r   # capacity


# ===========================================================================
# to_dataframe
# ===========================================================================

class TestToDataframe:
    def test_all_channels_no_prefix(self):
        """Without a prefix all channels appear as columns."""
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        buf.append(_ts(0), {"weight_g": 4200.0, "ammonia_ppb": 18.5})
        buf.append(_ts(1), {"weight_g": 7100.0, "ammonia_ppb": 21.0})
        df = buf.to_dataframe()
        assert set(df.columns) == {"weight_g", "ammonia_ppb"}
        assert df["weight_g"].tolist() == [4200.0, 7100.0]

    def test_channel_prefix_strips_prefix(self):
        """similarity_ prefix is stripped; columns become cat names."""
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        buf.append(_ts(0), {"similarity_anna": 0.91, "similarity_luna": 0.23,
                             "weight_g": 4200.0})
        df = buf.to_dataframe(channel_prefix="similarity_")
        assert set(df.columns) == {"anna", "luna"}, \
            "Only similarity_ channels should be included; weight_g excluded"

    def test_nan_for_absent_keys(self):
        """Entries that lack a channel key produce NaN, not 0.0."""
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        buf.append(_ts(0), {"similarity_anna": 0.91, "similarity_luna": 0.23})
        buf.append(_ts(1), {})   # no camera frame — all absent
        buf.append(_ts(2), {"similarity_anna": 0.89, "similarity_luna": 0.25})

        df = buf.to_dataframe(channel_prefix="similarity_")
        assert df.shape == (3, 2)
        assert pd.isna(df.iloc[1]["anna"]),  "Missing frame must be NaN"
        assert pd.isna(df.iloc[1]["luna"]),  "Missing frame must be NaN"
        assert not pd.isna(df.iloc[0]["anna"])

    def test_nan_vs_zero_correctness(self):
        """Demonstrate that NaN (not 0.0) gives the correct mean.

        If the missing frame were stored as 0.0 instead of NaN, the mean for
        'anna' would be (0.91 + 0.0 + 0.89) / 3 ≈ 0.60, which would fall
        below the 0.70 entry threshold and incorrectly result in 'Unknown'.
        With NaN and skipna=True the mean is (0.91 + 0.89) / 2 = 0.90,
        correctly identifying Anna.
        """
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        buf.append(_ts(0), {"similarity_anna": 0.91})
        buf.append(_ts(1), {})    # missing frame
        buf.append(_ts(2), {"similarity_anna": 0.89})

        df = buf.to_dataframe(channel_prefix="similarity_")

        nan_mean  = df["anna"].mean(skipna=True)
        zero_mean = df["anna"].fillna(0.0).mean()

        threshold = 0.70
        assert nan_mean  >= threshold, \
            f"NaN mean {nan_mean:.3f} should be above threshold {threshold}"
        assert zero_mean < threshold, \
            f"Zero-fill mean {zero_mean:.3f} should be below threshold (proves NaN matters)"

    def test_timestamp_index(self):
        """DataFrame index must contain the original timestamps."""
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        t0, t1 = _ts(0), _ts(5)
        buf.append(t0, {"similarity_anna": 0.91})
        buf.append(t1, {"similarity_anna": 0.85})
        df = buf.to_dataframe(channel_prefix="similarity_")
        assert list(df.index) == [t0, t1]

    def test_time_range_filter_start(self):
        """Entries before 'start' are excluded."""
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        for i in range(5):
            buf.append(_ts(i * 5), {"x": float(i)})
        df = buf.to_dataframe(start=_ts(10))   # keep t=10, 15, 20
        assert len(df) == 3
        assert df["x"].tolist() == [2.0, 3.0, 4.0]

    def test_time_range_filter_end(self):
        """Entries after 'end' are excluded."""
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        for i in range(5):
            buf.append(_ts(i * 5), {"x": float(i)})
        df = buf.to_dataframe(end=_ts(10))   # keep t=0, 5, 10
        assert len(df) == 3

    def test_empty_buffer_returns_empty_dataframe(self):
        buf = RollingBuffer(window_minutes=1, samples_per_minute=12)
        df = buf.to_dataframe()
        assert df.empty

    def test_serialisation_round_trip(self):
        """to_json(orient='split') → read_json round-trip preserves data and NaN."""
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)
        buf.append(_ts(0), {"similarity_anna": 0.91, "similarity_luna": 0.23})
        buf.append(_ts(5), {})                # NaN row
        buf.append(_ts(10), {"similarity_anna": 0.89, "similarity_luna": 0.25})

        df_orig = buf.to_dataframe(channel_prefix="similarity_")
        json_str = df_orig.to_json(orient="split", date_format="iso")

        # pandas 3.x requires a file-like object; wrap the JSON string.
        import io
        df_restored = pd.read_json(io.StringIO(json_str), orient="split")
        # Restore datetime index from string after round-trip
        df_restored.index = pd.to_datetime(df_restored.index)

        # Check values match (NaN-safe comparison)
        for col in df_orig.columns:
            for i in range(len(df_orig)):
                orig_val = df_orig.iloc[i][col]
                rest_val = df_restored.iloc[i][col]
                if pd.isna(orig_val):
                    assert pd.isna(rest_val), \
                        f"Row {i}, col '{col}': NaN not preserved through JSON"
                else:
                    assert orig_val == pytest.approx(rest_val, abs=1e-6), \
                        f"Row {i}, col '{col}': value mismatch"


# ===========================================================================
# load_td_config
# ===========================================================================

class TestLoadTdConfig:
    def test_loads_default_config(self, tmp_path):
        """A valid config file is loaded and returns all required keys."""
        cfg_path = tmp_path / "td_config.json"
        cfg_path.write_text(json.dumps({
            "window_minutes": 10,
            "samples_per_minute": 12,
            "channels": [{"name": "weight_g", "type": "weight", "enabled": True}],
            "trigger": {"weight_entry_delta_g": 300},
            "image_retention_days": 7,
        }))
        cfg = load_td_config(cfg_path)
        assert cfg["window_minutes"] == 10
        assert cfg["samples_per_minute"] == 12
        assert isinstance(cfg["channels"], list)
        assert isinstance(cfg["trigger"], dict)
        assert cfg["image_retention_days"] == 7

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="td_config.json"):
            load_td_config(tmp_path / "nonexistent.json")

    @pytest.mark.parametrize("missing_key", [
        "window_minutes",
        "samples_per_minute",
        "channels",
        "trigger",
        "image_retention_days",
    ])
    def test_raises_on_missing_required_key(self, tmp_path, missing_key):
        """Each required top-level key, when absent, raises ValueError."""
        base = {
            "window_minutes": 10,
            "samples_per_minute": 12,
            "channels": [],
            "trigger": {},
            "image_retention_days": 7,
        }
        del base[missing_key]
        cfg_path = tmp_path / "td_config.json"
        cfg_path.write_text(json.dumps(base))
        with pytest.raises(ValueError, match=missing_key):
            load_td_config(cfg_path)

    def test_raises_on_wrong_type_for_window_minutes(self, tmp_path):
        """window_minutes must be int; a string value raises ValueError."""
        cfg_path = tmp_path / "td_config.json"
        cfg_path.write_text(json.dumps({
            "window_minutes": "ten",   # wrong type
            "samples_per_minute": 12,
            "channels": [],
            "trigger": {},
            "image_retention_days": 7,
        }))
        with pytest.raises(ValueError, match="window_minutes"):
            load_td_config(cfg_path)

    def test_loads_real_config_file(self):
        """The actual td_config.json shipped with the project must be valid."""
        cfg = load_td_config()   # uses default path (src/litterbox/td_config.json)
        assert cfg["window_minutes"] > 0
        assert cfg["samples_per_minute"] > 0
        assert len(cfg["channels"]) >= 1
        assert "weight_entry_delta_g" in cfg["trigger"]
        assert cfg["image_retention_days"] > 0


# ===========================================================================
# Thread-safety smoke test
# ===========================================================================

class TestThreadSafety:
    def test_concurrent_appends_no_corruption(self):
        """Two threads appending 500 entries each must not corrupt the buffer.

        We do not assert on the exact count (eviction makes that tricky) but
        we do assert that every stored entry has a valid 'values' dict and
        that no exception is raised.
        """
        buf = RollingBuffer(window_minutes=1, samples_per_minute=60)  # cap=60

        errors: list[str] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(500):
                    buf.append(_ts(i * 0.01), {"thread": thread_id, "i": i})
            except Exception as exc:
                errors.append(str(exc))

        t1 = threading.Thread(target=worker, args=(1,))
        t2 = threading.Thread(target=worker, args=(2,))
        t1.start(); t2.start()
        t1.join();  t2.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(buf) == buf.max_len, "Buffer should be full after 1000 appends"
        # Spot-check: every entry is a well-formed dict
        for entry in buf.snapshot():
            assert "timestamp" in entry
            assert "values" in entry
            assert isinstance(entry["values"], dict)
