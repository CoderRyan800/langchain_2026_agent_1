"""
test_sensor_collector.py — Unit tests for Step 2 (SensorCollector)
===================================================================

All tests use mock drivers (no hardware, no CLIP, no camera).  The test
config is constructed inline with a very high sample rate so that multi-tick
tests complete in well under a second.

Test organisation
-----------------
TestBaseDriver      — abstract interface enforcement
TestWeightDriver    — Gaussian noise, zero-noise, non-negative result
TestAmmoniaDriver   — same
TestMethaneDriver   — same
TestChipIdDriver    — present chip, absent chip
TestSimilarityDriver— dict return, None return, copy-on-read
TestSampleOnce      — buffer population, channel expansion, disabled skip,
                      missing driver skip, None similarity handling
TestRunMultipleTicks— 3 ticks → 3 buffer entries
TestStop            — thread terminates within 2 seconds
TestRepr            — repr contains expected tokens
"""

from __future__ import annotations

import math
import time
import threading
from datetime import datetime, timezone
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from litterbox.time_buffer import RollingBuffer
from litterbox.sensor_collector import (
    BaseDriver,
    WeightDriver,
    AmmoniaDriver,
    MethaneDriver,
    ChipIdDriver,
    SimilarityDriver,
    SensorCollector,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_config(
    *,
    channels: list[dict] | None = None,
    samples_per_minute: int = 600,   # one sample every 0.1 s — fast for tests
    window_minutes: int = 1,
) -> dict:
    """Build a minimal td_config-style dict for tests.

    Default channels match the production set (all enabled).  Pass a custom
    list to test disabled channels or partial configurations.
    """
    if channels is None:
        channels = [
            {"name": "weight_g",    "type": "weight",     "enabled": True},
            {"name": "ammonia_ppb", "type": "ammonia",    "enabled": True},
            {"name": "methane_ppb", "type": "methane",    "enabled": True},
            {"name": "chip_id",     "type": "chip_id",    "enabled": True},
            {"name": "similarity",  "type": "similarity", "enabled": True},
        ]
    return {
        "window_minutes":     window_minutes,
        "samples_per_minute": samples_per_minute,
        "channels":           channels,
        "trigger":            {},
        "image_retention_days": 7,
    }


def _make_drivers(
    weight: float = 5000.0,
    ammonia: float = 10.0,
    methane: float = 6.0,
    chip_id: Optional[str] = None,
    cat_scores: Optional[dict] = None,
) -> dict:
    """Return a drivers dict with zero-noise mock scalars."""
    return {
        "weight":     WeightDriver(base_value=weight,   noise_sigma=0),
        "ammonia":    AmmoniaDriver(base_value=ammonia,  noise_sigma=0),
        "methane":    MethaneDriver(base_value=methane,  noise_sigma=0),
        "chip_id":    ChipIdDriver(cat_name=chip_id),
        "similarity": SimilarityDriver(cat_scores=cat_scores),
    }


def _make_buf(config: dict) -> RollingBuffer:
    return RollingBuffer(
        window_minutes=config["window_minutes"],
        samples_per_minute=config["samples_per_minute"],
    )


# ===========================================================================
# TestBaseDriver
# ===========================================================================

class TestBaseDriver:
    """BaseDriver cannot be instantiated directly — it is abstract."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseDriver()  # type: ignore[abstract]

    def test_concrete_subclass_without_read_fails(self):
        """A subclass that omits read() cannot be instantiated."""
        class Incomplete(BaseDriver):
            pass  # no read() method

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_with_read_succeeds(self):
        class Minimal(BaseDriver):
            def read(self):
                return 42.0

        driver = Minimal()
        assert driver.read() == 42.0


# ===========================================================================
# TestWeightDriver
# ===========================================================================

class TestWeightDriver:

    def test_zero_noise_returns_base(self):
        d = WeightDriver(base_value=5400.0, noise_sigma=0)
        assert d.read() == 5400.0

    def test_with_noise_varies(self):
        """With non-zero sigma, 100 readings should not all be identical."""
        d = WeightDriver(base_value=5400.0, noise_sigma=25.0)
        readings = {d.read() for _ in range(100)}
        assert len(readings) > 1, "Noisy driver produced constant output"

    def test_with_noise_near_base(self):
        """Readings should cluster within ±4 sigma of the base (99.99 % CI)."""
        base, sigma = 5400.0, 25.0
        d = WeightDriver(base_value=base, noise_sigma=sigma)
        for _ in range(200):
            r = d.read()
            assert abs(r - base) < 4 * sigma, f"Outlier reading: {r}"

    def test_default_args(self):
        d = WeightDriver()
        assert d.read() == 0.0


# ===========================================================================
# TestAmmoniaDriver
# ===========================================================================

class TestAmmoniaDriver:

    def test_zero_noise_returns_base(self):
        d = AmmoniaDriver(base_value=8.0, noise_sigma=0)
        assert d.read() == 8.0

    def test_clamped_to_zero(self):
        """Even with large negative noise the result must be ≥ 0."""
        d = AmmoniaDriver(base_value=0.1, noise_sigma=50.0)
        for _ in range(100):
            assert d.read() >= 0.0

    def test_default_args(self):
        d = AmmoniaDriver()
        assert d.read() == 0.0


# ===========================================================================
# TestMethaneDriver
# ===========================================================================

class TestMethaneDriver:

    def test_zero_noise_returns_base(self):
        d = MethaneDriver(base_value=5.0, noise_sigma=0)
        assert d.read() == 5.0

    def test_clamped_to_zero(self):
        d = MethaneDriver(base_value=0.1, noise_sigma=50.0)
        for _ in range(100):
            assert d.read() >= 0.0

    def test_default_args(self):
        d = MethaneDriver()
        assert d.read() == 0.0


# ===========================================================================
# TestChipIdDriver
# ===========================================================================

class TestChipIdDriver:

    def test_returns_cat_name(self):
        d = ChipIdDriver(cat_name="luna")
        assert d.read() == "luna"

    def test_returns_none_by_default(self):
        d = ChipIdDriver()
        assert d.read() is None

    def test_explicit_none(self):
        d = ChipIdDriver(cat_name=None)
        assert d.read() is None


# ===========================================================================
# TestSimilarityDriver
# ===========================================================================

class TestSimilarityDriver:

    def test_returns_cat_scores(self):
        scores = {"anna": 0.91, "luna": 0.23}
        d = SimilarityDriver(cat_scores=scores)
        result = d.read()
        assert result == {"anna": 0.91, "luna": 0.23}

    def test_returns_none_for_missing_frame(self):
        d = SimilarityDriver(cat_scores=None)
        assert d.read() is None

    def test_default_is_none(self):
        d = SimilarityDriver()
        assert d.read() is None

    def test_returns_copy(self):
        """Mutating the returned dict must not affect the driver's state."""
        scores = {"anna": 0.91}
        d = SimilarityDriver(cat_scores=scores)
        result = d.read()
        result["anna"] = 0.0       # mutate the returned copy
        assert d.read()["anna"] == 0.91   # original unchanged


# ===========================================================================
# TestSampleOnce — the core integration point
# ===========================================================================

class TestSampleOnce:
    """Tests for SensorCollector._sample_once() in isolation.

    We call _sample_once() directly (not via start/stop) so that tests are
    synchronous, instant, and deterministic.
    """

    def test_scalar_channels_written_to_buffer(self):
        """Weight, ammonia, methane, chip_id values appear in the buffer."""
        cfg  = _make_config()
        buf  = _make_buf(cfg)
        drv  = _make_drivers(
            weight=5400.0, ammonia=12.0, methane=7.0, chip_id="luna"
        )
        # Remove similarity driver so we don't need to worry about it here.
        del drv["similarity"]
        # Disable similarity channel for this test.
        cfg["channels"] = [c for c in cfg["channels"] if c["type"] != "similarity"]

        sc = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc._sample_once()

        assert len(buf) == 1
        snap = buf.snapshot()
        v = snap[0]["values"]
        assert v["weight_g"]    == 5400.0
        assert v["ammonia_ppb"] == 12.0
        assert v["methane_ppb"] == 7.0
        assert v["chip_id"]     == "luna"

    def test_similarity_dict_expanded_into_per_cat_keys(self):
        """SimilarityDriver dict → similarity_<catname> keys in the buffer."""
        cfg = _make_config(channels=[
            {"name": "similarity", "type": "similarity", "enabled": True},
        ])
        buf = _make_buf(cfg)
        drv = {"similarity": SimilarityDriver({"anna": 0.91, "luna": 0.23})}

        sc = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc._sample_once()

        v = buf.snapshot()[0]["values"]
        assert v["similarity_anna"] == 0.91
        assert v["similarity_luna"] == 0.23
        # No un-expanded "similarity" key should exist.
        assert "similarity" not in v

    def test_similarity_none_means_no_keys_written(self):
        """None from SimilarityDriver → no similarity_* keys in buffer entry."""
        cfg = _make_config(channels=[
            {"name": "similarity", "type": "similarity", "enabled": True},
        ])
        buf = _make_buf(cfg)
        drv = {"similarity": SimilarityDriver(cat_scores=None)}

        sc = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc._sample_once()

        v = buf.snapshot()[0]["values"]
        assert not any(k.startswith("similarity_") for k in v), \
            "No similarity keys should appear when the frame is missing"

    def test_disabled_channel_not_sampled(self):
        """A channel with enabled=False must not appear in the buffer entry."""
        cfg = _make_config(channels=[
            {"name": "weight_g",    "type": "weight",  "enabled": True},
            {"name": "ammonia_ppb", "type": "ammonia", "enabled": False},  # disabled
        ])
        buf = _make_buf(cfg)
        drv = {
            "weight":  WeightDriver(base_value=5000.0, noise_sigma=0),
            "ammonia": AmmoniaDriver(base_value=20.0,  noise_sigma=0),
        }

        sc = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc._sample_once()

        v = buf.snapshot()[0]["values"]
        assert "weight_g"    in v,      "Enabled channel must be sampled"
        assert "ammonia_ppb" not in v,  "Disabled channel must not appear"

    def test_missing_driver_skipped_silently(self):
        """If a driver is not in the drivers dict, the channel is skipped."""
        cfg = _make_config(channels=[
            {"name": "weight_g",    "type": "weight",  "enabled": True},
            {"name": "methane_ppb", "type": "methane", "enabled": True},
        ])
        buf = _make_buf(cfg)
        # Only weight driver provided; methane has no driver.
        drv = {"weight": WeightDriver(base_value=5000.0, noise_sigma=0)}

        sc = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc._sample_once()

        v = buf.snapshot()[0]["values"]
        assert "weight_g"    in v
        assert "methane_ppb" not in v

    def test_timestamp_is_utc(self):
        """The buffer entry timestamp should be timezone-aware UTC."""
        cfg = _make_config(channels=[
            {"name": "weight_g", "type": "weight", "enabled": True},
        ])
        buf = _make_buf(cfg)
        drv = {"weight": WeightDriver(base_value=5000.0)}

        sc = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc._sample_once()

        ts = buf.get_timestamps()[0]
        assert ts.tzinfo is not None, "Timestamp must be timezone-aware"
        assert ts.tzinfo == timezone.utc

    def test_multiple_sample_once_appends_multiple_entries(self):
        """Three calls to _sample_once() should add three entries."""
        cfg = _make_config(channels=[
            {"name": "weight_g", "type": "weight", "enabled": True},
        ])
        buf = _make_buf(cfg)
        drv = {"weight": WeightDriver(base_value=5000.0)}

        sc = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc._sample_once()
        sc._sample_once()
        sc._sample_once()

        assert len(buf) == 3

    def test_chip_id_none_stored_as_none(self):
        """chip_id=None is stored as None (not omitted), reflecting a real absence."""
        cfg = _make_config(channels=[
            {"name": "chip_id", "type": "chip_id", "enabled": True},
        ])
        buf = _make_buf(cfg)
        drv = {"chip_id": ChipIdDriver(cat_name=None)}

        sc = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc._sample_once()

        v = buf.snapshot()[0]["values"]
        # chip_id key should exist with value None (not be absent entirely)
        assert "chip_id" in v
        assert v["chip_id"] is None

    def test_five_cats_similarity_expanded(self):
        """Five-cat similarity dict produces five similarity_* keys."""
        cats = {"anna": 0.10, "luna": 0.85, "marina": 0.12,
                "natasha": 0.08, "whiskers": 0.11}
        cfg = _make_config(channels=[
            {"name": "similarity", "type": "similarity", "enabled": True},
        ])
        buf = _make_buf(cfg)
        drv = {"similarity": SimilarityDriver(cat_scores=cats)}

        sc = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc._sample_once()

        v = buf.snapshot()[0]["values"]
        for cat, score in cats.items():
            key = f"similarity_{cat}"
            assert key in v,        f"Expected key {key!r} in buffer entry"
            assert v[key] == score, f"Wrong score for {cat}"


# ===========================================================================
# TestRunMultipleTicks — background thread timing
# ===========================================================================

class TestRunMultipleTicks:
    """Verify that the background thread fires the correct number of times."""

    def test_three_ticks_produce_three_entries(self):
        """At 600 samples/min (0.1 s interval), 3 entries should appear in < 1 s."""
        cfg = _make_config(samples_per_minute=600, channels=[
            {"name": "weight_g", "type": "weight", "enabled": True},
        ])
        buf = _make_buf(cfg)
        drv = {"weight": WeightDriver(base_value=5000.0)}

        sc = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc.start()

        # Give the thread time to fire at least 3 times (0.4 s > 3 × 0.1 s).
        time.sleep(0.4)
        sc.stop()

        assert len(buf) >= 3, \
            f"Expected ≥ 3 buffer entries after 0.4 s, got {len(buf)}"

    def test_entries_have_monotonic_timestamps(self):
        """Timestamps in the buffer must be non-decreasing."""
        cfg = _make_config(samples_per_minute=600, channels=[
            {"name": "weight_g", "type": "weight", "enabled": True},
        ])
        buf = _make_buf(cfg)
        drv = {"weight": WeightDriver(base_value=5000.0)}

        sc = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc.start()
        time.sleep(0.35)
        sc.stop()

        timestamps = buf.get_timestamps()
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], \
                f"Timestamp went backwards at index {i}"


# ===========================================================================
# TestStop
# ===========================================================================

class TestStop:
    """Verify that stop() terminates the background thread promptly."""

    def test_stop_within_two_seconds(self):
        """stop() must return and thread must be dead within 2 s."""
        cfg = _make_config(samples_per_minute=12, channels=[
            {"name": "weight_g", "type": "weight", "enabled": True},
        ])
        buf = _make_buf(cfg)
        drv = {"weight": WeightDriver(base_value=5000.0)}

        sc = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc.start()

        t0 = time.monotonic()
        sc.stop()
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0, f"stop() took {elapsed:.2f} s (limit 2 s)"

    def test_stop_idempotent(self):
        """Calling stop() twice must not raise."""
        cfg = _make_config(channels=[
            {"name": "weight_g", "type": "weight", "enabled": True},
        ])
        buf = _make_buf(cfg)
        drv = {"weight": WeightDriver(base_value=5000.0)}

        sc = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc.start()
        sc.stop()
        sc.stop()   # second call must be harmless

    def test_stop_before_start_is_safe(self):
        """stop() before start() must not raise."""
        cfg  = _make_config()
        buf  = _make_buf(cfg)
        drv  = _make_drivers()
        sc   = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc.stop()   # no thread was started

    def test_double_start_raises(self):
        """Calling start() while already running must raise RuntimeError."""
        cfg = _make_config(channels=[
            {"name": "weight_g", "type": "weight", "enabled": True},
        ])
        buf = _make_buf(cfg)
        drv = {"weight": WeightDriver(base_value=5000.0)}

        sc = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc.start()
        try:
            with pytest.raises(RuntimeError):
                sc.start()
        finally:
            sc.stop()


# ===========================================================================
# TestRepr
# ===========================================================================

class TestRepr:

    def test_repr_contains_interval(self):
        cfg = _make_config(samples_per_minute=12, channels=[
            {"name": "weight_g", "type": "weight", "enabled": True},
        ])
        buf = _make_buf(cfg)
        drv = {"weight": WeightDriver()}
        sc  = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        r   = repr(sc)
        assert "5.0s" in r, f"Expected interval in repr, got: {r}"

    def test_repr_contains_running_false(self):
        cfg = _make_config(channels=[
            {"name": "weight_g", "type": "weight", "enabled": True},
        ])
        buf = _make_buf(cfg)
        drv = {"weight": WeightDriver()}
        sc  = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        assert "running=False" in repr(sc)

    def test_repr_contains_running_true_while_active(self):
        cfg = _make_config(channels=[
            {"name": "weight_g", "type": "weight", "enabled": True},
        ])
        buf = _make_buf(cfg)
        drv = {"weight": WeightDriver()}
        sc  = SensorCollector(config=cfg, drivers=drv, buffer=buf)
        sc.start()
        try:
            assert "running=True" in repr(sc)
        finally:
            sc.stop()
