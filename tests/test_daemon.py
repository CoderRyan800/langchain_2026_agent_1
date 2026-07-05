"""Tests for the continuous-monitor daemon (src/litterbox/daemon.py).

These cover the wiring and the hardware seam without any real hardware, threads,
or network. The daemon's own ``self_test()`` is the end-to-end acceptance test
(a scripted weight ramp must produce exactly one saved visit); the rest exercise
the driver-assembly seam and the visit-handler's fault tolerance.
"""
from unittest.mock import Mock

import pytest

from litterbox import daemon
from litterbox.sensor_collector import BaseDriver
from litterbox.time_buffer import load_td_config


def test_self_test_passes():
    """The deterministic acceptance test detects exactly one visit and returns 0.

    ``self_test()`` isolates its own temp DB, so it is safe to run under pytest
    regardless of the conftest DB fixture.
    """
    assert daemon.self_test() == 0


def test_weight_only_preset_is_valid_and_scale_only():
    """The shipped preset loads through the validating loader and enables only weight."""
    config = load_td_config(daemon.WEIGHT_ONLY_CONFIG)
    enabled = [c["type"] for c in config["channels"] if c.get("enabled")]
    assert enabled == ["weight"]


def test_default_config_is_the_weight_only_preset():
    config = daemon._load_config(None)
    enabled = [c["type"] for c in config["channels"] if c.get("enabled")]
    assert enabled == ["weight"]


def test_build_drivers_simulate_returns_a_weight_driver():
    config = daemon._load_config(None)
    drivers = daemon.build_drivers(config, simulate=True)
    assert set(drivers) == {"weight"}
    assert isinstance(drivers["weight"], BaseDriver)
    reading = drivers["weight"].read()
    assert isinstance(reading, float)


def test_build_scale_driver_stub_raises_until_implemented():
    """The production seam must fail loudly (not silently no-op) until wired."""
    with pytest.raises(NotImplementedError):
        daemon.build_scale_driver(daemon._load_config(None))


def test_build_drivers_requires_weight_channel():
    config = {
        "channels": [{"name": "ammonia_ppb", "type": "ammonia", "enabled": True}],
        "samples_per_minute": 12,
    }
    with pytest.raises(RuntimeError, match="weight"):
        daemon.build_drivers(config, simulate=True)


def test_unknown_enabled_channel_is_skipped_not_fatal(caplog):
    """A channel with no driver yet is warned about and skipped, weight still built."""
    config = {
        "channels": [
            {"name": "weight_g", "type": "weight", "enabled": True},
            {"name": "methane_ppb", "type": "methane", "enabled": True},
        ],
        "samples_per_minute": 12,
    }
    drivers = daemon.build_drivers(config, simulate=True)
    assert set(drivers) == {"weight"}  # methane skipped, no driver yet


def test_visit_handler_swallows_exceptions():
    """A failing analyse/save must not propagate onto the sampler thread."""
    analyser = Mock()
    analyser.analyse.side_effect = RuntimeError("boom")
    handler = daemon.make_visit_handler(analyser)
    # Must not raise despite the analyser blowing up.
    handler([], None, None)
    analyser.analyse.assert_called_once()


def test_scripted_scale_driver_stages_visits():
    """The --simulate mock scale rises above and falls back to baseline."""
    drv = daemon.ScriptedScaleDriver(baseline_g=4500.0, cat_g=700.0, noise_sigma=0.0,
                                     period=4, visit_len=2)
    readings = [drv.read() for _ in range(4)]
    # period=4, visit_len=2 → present, present, absent, absent
    assert readings[0] > 5000 and readings[1] > 5000
    assert readings[2] < 4600 and readings[3] < 4600
