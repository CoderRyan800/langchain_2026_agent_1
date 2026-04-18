"""
Tests for analyser_pipeline.py — Step 5a of the Time-Domain Measurement System
================================================================================

Covers:
- resample_to_length utility (various lengths, NaN handling, identity)
- BaseAnalyser ABC contract
- AnalyserPipeline run with mock plugins
- Fault isolation: crashing plugin does not block others
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from litterbox.analyser_pipeline import (
    AnalysisResult,
    AnalyserPipeline,
    BaseAnalyser,
    resample_to_length,
)
from litterbox.visit_analyser import TdVisitRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 4, 4, 12, 0, 0, tzinfo=timezone.utc)


def _ts(offset_seconds: float = 0.0) -> datetime:
    return _BASE + timedelta(seconds=offset_seconds)


def _make_record(**overrides) -> TdVisitRecord:
    """Create a minimal TdVisitRecord for tests."""
    defaults = {
        "entry_time": _ts(0),
        "exit_time": _ts(50),
        "snapshot_json": "[]",
        "td_visit_id": 1,
    }
    defaults.update(overrides)
    return TdVisitRecord(**defaults)


def _make_snapshot(
    n_samples: int = 10,
    channel: str = "weight_g",
    base_value: float = 5000.0,
    step: float = 100.0,
    interval: float = 5.0,
) -> list[dict]:
    """Create a synthetic snapshot with linearly increasing channel values."""
    return [
        {
            "timestamp": _ts(i * interval),
            "values": {channel: base_value + i * step},
        }
        for i in range(n_samples)
    ]


# ---------------------------------------------------------------------------
# Mock plugins
# ---------------------------------------------------------------------------


class _EchoPlugin(BaseAnalyser):
    """Captures the waveform it receives and returns a fixed result."""

    def __init__(self, name_str: str = "echo"):
        self._name = name_str
        self.received_waveform = None
        self.received_channel = None
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def analyse(self, waveform, visit_record, channel):
        self.received_waveform = waveform.copy()
        self.received_channel = channel
        self.call_count += 1
        return AnalysisResult(
            plugin_name=self._name,
            anomaly_score=0.1,
            anomaly_level="normal",
            details={"len": len(waveform)},
        )


class _CrashingPlugin(BaseAnalyser):
    """Always raises an exception."""

    @property
    def name(self) -> str:
        return "crasher"

    def analyse(self, waveform, visit_record, channel):
        raise RuntimeError("Intentional test crash")


# ---------------------------------------------------------------------------
# TestResampleToLength
# ---------------------------------------------------------------------------


class TestResampleToLength:
    """resample_to_length utility function."""

    def test_upsample_10_to_64(self):
        """Resampling 10 elements to 64 produces correct shape."""
        raw = np.arange(10, dtype=float)
        result = resample_to_length(raw, 64)
        assert result.shape == (64,)
        # First and last values should be preserved.
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(9.0)

    def test_downsample_100_to_64(self):
        """Resampling 100 elements to 64 produces correct shape."""
        raw = np.linspace(0, 99, 100)
        result = resample_to_length(raw, 64)
        assert result.shape == (64,)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(99.0)

    def test_identity_64_to_64(self):
        """Resampling 64 elements to 64 returns the same values."""
        raw = np.arange(64, dtype=float)
        result = resample_to_length(raw, 64)
        np.testing.assert_array_almost_equal(result, raw)

    def test_single_element(self):
        """Resampling a single-element array fills the output with that value."""
        raw = np.array([42.0])
        result = resample_to_length(raw, 64)
        assert result.shape == (64,)
        np.testing.assert_array_almost_equal(result, np.full(64, 42.0))

    def test_empty_array(self):
        """Empty input → all-NaN output."""
        raw = np.array([], dtype=float)
        result = resample_to_length(raw, 64)
        assert result.shape == (64,)
        assert np.isnan(result).all()

    def test_all_nan(self):
        """All-NaN input → all-NaN output."""
        raw = np.full(10, np.nan)
        result = resample_to_length(raw, 64)
        assert result.shape == (64,)
        assert np.isnan(result).all()

    def test_nan_gap_fill(self):
        """NaN gaps are filled by linear interpolation before resampling."""
        raw = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = resample_to_length(raw, 5)
        # After gap-fill: [1, 2, 3, 4, 5]; resampled to 5 → same.
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_monotonicity_preserved(self):
        """A monotonically increasing input produces a monotonically increasing output."""
        raw = np.linspace(0, 100, 20)
        result = resample_to_length(raw, 64)
        diffs = np.diff(result)
        assert (diffs >= 0).all()


# ---------------------------------------------------------------------------
# TestBaseAnalyser
# ---------------------------------------------------------------------------


class TestBaseAnalyser:
    """BaseAnalyser ABC contract."""

    def test_cannot_instantiate_abstract(self):
        """BaseAnalyser cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAnalyser()

    def test_subclass_without_name_fails(self):
        """A subclass that doesn't implement name fails."""

        class Incomplete(BaseAnalyser):
            def analyse(self, waveform, visit_record, channel):
                return AnalysisResult("x", 0.0, "normal")

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_without_analyse_fails(self):
        """A subclass that doesn't implement analyse fails."""

        class Incomplete(BaseAnalyser):
            @property
            def name(self):
                return "x"

        with pytest.raises(TypeError):
            Incomplete()


# ---------------------------------------------------------------------------
# TestAnalysisResult
# ---------------------------------------------------------------------------


class TestAnalysisResult:
    """AnalysisResult dataclass."""

    def test_creation(self):
        r = AnalysisResult("test", 0.5, "mild", {"key": "value"})
        assert r.plugin_name == "test"
        assert r.anomaly_score == 0.5
        assert r.anomaly_level == "mild"
        assert r.details["key"] == "value"

    def test_default_details(self):
        r = AnalysisResult("test", 0.0, "normal")
        assert r.details == {}


# ---------------------------------------------------------------------------
# TestPipelineRun
# ---------------------------------------------------------------------------


class TestPipelineRun:
    """AnalyserPipeline.run() with mock plugins."""

    def test_single_plugin_receives_correct_waveform(self):
        """Plugin receives a waveform of shape (64,)."""
        plugin = _EchoPlugin()
        pipeline = AnalyserPipeline([plugin])

        snapshot = _make_snapshot(n_samples=20, interval=5.0)
        record = _make_record(entry_time=_ts(0), exit_time=_ts(95))

        results = pipeline.run(record, snapshot, _ts(0), _ts(95))

        assert len(results) == 1
        assert results[0].plugin_name == "echo"
        assert plugin.received_waveform is not None
        assert plugin.received_waveform.shape == (64,)
        assert plugin.call_count == 1

    def test_two_plugins_both_called(self):
        """Both plugins receive the same waveform and produce results."""
        p1 = _EchoPlugin("plugin_a")
        p2 = _EchoPlugin("plugin_b")
        pipeline = AnalyserPipeline([p1, p2])

        snapshot = _make_snapshot(n_samples=10, interval=5.0)
        record = _make_record(entry_time=_ts(0), exit_time=_ts(45))

        results = pipeline.run(record, snapshot, _ts(0), _ts(45))

        assert len(results) == 2
        assert results[0].plugin_name == "plugin_a"
        assert results[1].plugin_name == "plugin_b"
        assert p1.call_count == 1
        assert p2.call_count == 1

    def test_no_plugins_returns_empty(self):
        """No registered plugins → empty result list."""
        pipeline = AnalyserPipeline([])
        snapshot = _make_snapshot()
        record = _make_record()

        results = pipeline.run(record, snapshot, _ts(0), _ts(45))

        assert results == []

    def test_channel_extraction(self):
        """Pipeline extracts the correct channel values from the snapshot."""
        plugin = _EchoPlugin()
        pipeline = AnalyserPipeline([plugin])

        # 5 samples with weight_g = [1000, 2000, 3000, 4000, 5000].
        snapshot = _make_snapshot(
            n_samples=5, channel="weight_g", base_value=1000, step=1000, interval=5.0
        )
        record = _make_record(entry_time=_ts(0), exit_time=_ts(20))

        pipeline.run(record, snapshot, _ts(0), _ts(20))

        wf = plugin.received_waveform
        # Resampled from [1000, 2000, 3000, 4000, 5000] to 64 points.
        # First should be ~1000, last ~5000.
        assert wf[0] == pytest.approx(1000.0)
        assert wf[-1] == pytest.approx(5000.0)

    def test_custom_resample_length(self):
        """Config can override the resampling target length."""
        plugin = _EchoPlugin()
        pipeline = AnalyserPipeline(
            [plugin], config={"eigen": {"resample_length": 32}}
        )

        snapshot = _make_snapshot(n_samples=10, interval=5.0)
        record = _make_record(entry_time=_ts(0), exit_time=_ts(45))

        pipeline.run(record, snapshot, _ts(0), _ts(45))

        assert plugin.received_waveform.shape == (32,)


# ---------------------------------------------------------------------------
# TestFaultIsolation
# ---------------------------------------------------------------------------


class TestFaultIsolation:
    """A crashing plugin does not prevent other plugins from running."""

    def test_crashing_plugin_produces_error_result(self):
        """Crashing plugin → error result with exception message."""
        crasher = _CrashingPlugin()
        pipeline = AnalyserPipeline([crasher])

        snapshot = _make_snapshot(n_samples=10, interval=5.0)
        record = _make_record(entry_time=_ts(0), exit_time=_ts(45))

        results = pipeline.run(record, snapshot, _ts(0), _ts(45))

        assert len(results) == 1
        assert results[0].plugin_name == "crasher"
        assert results[0].anomaly_level == "error"
        assert "Intentional test crash" in results[0].details["error"]

    def test_crash_does_not_block_next_plugin(self):
        """First plugin crashes, second still runs and produces a result."""
        crasher = _CrashingPlugin()
        echo = _EchoPlugin()
        pipeline = AnalyserPipeline([crasher, echo])

        snapshot = _make_snapshot(n_samples=10, interval=5.0)
        record = _make_record(entry_time=_ts(0), exit_time=_ts(45))

        results = pipeline.run(record, snapshot, _ts(0), _ts(45))

        assert len(results) == 2
        assert results[0].anomaly_level == "error"
        assert results[1].plugin_name == "echo"
        assert results[1].anomaly_level == "normal"
        assert echo.call_count == 1
