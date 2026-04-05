"""
test_visit_trigger.py — Unit tests for Step 3 (VisitTrigger)
=============================================================

All tests call ``trigger.check(values, timestamp=…)`` synchronously with
hand-crafted values dicts and explicit timestamps.  No background threads,
no real hardware, no CLIP model.

Test organisation
-----------------
TestStateConstants       — KITTY_ABSENT / KITTY_PRESENT string values
TestInitialState         — fresh trigger starts in KITTY_ABSENT
TestWeightTrigger        — entry and exit via weight channel only
TestChipIdTrigger        — entry and exit via chip_id channel only
TestSimilarityTrigger    — entry and exit via similarity channels only
TestMultipleCatsElevated — two cats elevated simultaneously
TestNoSpuriousTrigger    — oscillating weight that never meets entry threshold
TestReset                — reset() mid-visit, state and no spurious callback
TestMixedTriggers        — chip triggers entry, weight triggers exit
TestRepr                 — repr contains expected tokens
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from litterbox.time_buffer import RollingBuffer
from litterbox.visit_trigger import (
    VisitTrigger,
    KITTY_ABSENT,
    KITTY_PRESENT,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _ts(offset_seconds: float = 0.0) -> datetime:
    """Return a deterministic UTC timestamp offset from a fixed epoch."""
    base = datetime(2026, 4, 4, 12, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset_seconds)


def _make_config(
    *,
    weight_entry_delta: float = 300.0,
    weight_exit_delta:  float = 200.0,
    chip_absent_consecutive: int = 3,
    sim_entry_threshold: float = 0.70,
    sim_exit_threshold:  float = 0.50,
) -> dict:
    return {
        "window_minutes":     10,
        "samples_per_minute": 12,
        "channels":           [],   # VisitTrigger doesn't read channels from config
        "trigger": {
            "weight_entry_delta_g":            weight_entry_delta,
            "weight_exit_delta_g":             weight_exit_delta,
            "chip_absent_consecutive":         chip_absent_consecutive,
            "similarity_entry_threshold":      sim_entry_threshold,
            "similarity_exit_threshold":       sim_exit_threshold,
            "similarity_sustained_peak_samples": 3,
        },
        "image_retention_days": 7,
    }


def _make_buf(config: Optional[dict] = None) -> RollingBuffer:
    cfg = config or _make_config()
    return RollingBuffer(
        window_minutes=cfg["window_minutes"],
        samples_per_minute=cfg["samples_per_minute"],
    )


class _Collector:
    """Minimal callback collector — records every on_visit_complete call."""

    def __init__(self):
        self.calls: list[tuple] = []  # (snapshot, entry_time, exit_time)

    def __call__(self, snapshot, entry_time, exit_time):
        self.calls.append((snapshot, entry_time, exit_time))

    @property
    def count(self) -> int:
        return len(self.calls)

    @property
    def last(self) -> tuple:
        return self.calls[-1]


def _trigger(config=None, buf=None, collector=None):
    """Convenience factory: returns (trigger, buf, collector)."""
    cfg = config or _make_config()
    b   = buf or _make_buf(cfg)
    c   = collector or _Collector()
    t   = VisitTrigger(config=cfg, buffer=b, on_visit_complete=c)
    return t, b, c


def _feed_weight(buf: RollingBuffer, weights: list[float], start_offset: float = 0.0):
    """Append weight-only entries to the buffer."""
    for i, w in enumerate(weights):
        buf.append(_ts(start_offset + i * 5), {"weight_g": w})


# ===========================================================================
# TestStateConstants
# ===========================================================================

class TestStateConstants:

    def test_kitty_absent_value(self):
        assert KITTY_ABSENT == "kitty_absent"

    def test_kitty_present_value(self):
        assert KITTY_PRESENT == "kitty_present"

    def test_constants_are_distinct(self):
        assert KITTY_ABSENT != KITTY_PRESENT


# ===========================================================================
# TestInitialState
# ===========================================================================

class TestInitialState:

    def test_starts_absent(self):
        trigger, _, _ = _trigger()
        assert trigger.state == KITTY_ABSENT

    def test_no_callback_before_any_check(self):
        _, _, coll = _trigger()
        assert coll.count == 0


# ===========================================================================
# TestWeightTrigger
# ===========================================================================

class TestWeightTrigger:
    """Entry and exit driven entirely by the weight channel."""

    BASELINE = 5_400.0   # empty-box weight (g)
    ENTRY    = 300.0     # weight_entry_delta_g
    EXIT     = 200.0     # weight_exit_delta_g

    def _setup(self, n_baseline: int = 10):
        """Return a trigger with n_baseline pre-populated weight readings."""
        cfg = _make_config(
            weight_entry_delta=self.ENTRY,
            weight_exit_delta=self.EXIT,
        )
        buf  = _make_buf(cfg)
        coll = _Collector()
        trig = VisitTrigger(config=cfg, buffer=buf, on_visit_complete=coll)

        # Pre-populate the buffer with steady baseline readings so the
        # median baseline is well-established before any cat arrives.
        for i in range(n_baseline):
            buf.append(_ts(i * 5), {"weight_g": self.BASELINE})
        return trig, buf, coll

    def test_weight_ramp_fires_callback_exactly_once(self):
        """Flat → rise → fall produces exactly one callback."""
        trig, buf, coll = self._setup()

        # Feed one entry-level weight reading → should transition to PRESENT
        trig.check({"weight_g": self.BASELINE + self.ENTRY + 50},
                   timestamp=_ts(100))
        assert trig.state == KITTY_PRESENT

        # Feed several readings while cat is in the box
        for k in range(5):
            trig.check({"weight_g": self.BASELINE + 5_000},
                       timestamp=_ts(110 + k * 5))
        assert coll.count == 0  # no exit yet

        # Weight drops back to baseline → exit
        trig.check({"weight_g": self.BASELINE + 50},  # below baseline+exit_delta
                   timestamp=_ts(140))
        assert trig.state == KITTY_ABSENT
        assert coll.count == 1

    def test_entry_and_exit_times_correct(self):
        """entry_time and exit_time passed to callback match check() timestamps."""
        trig, buf, coll = self._setup()

        t_entry = _ts(100)
        t_exit  = _ts(200)

        trig.check({"weight_g": self.BASELINE + self.ENTRY + 1}, timestamp=t_entry)
        trig.check({"weight_g": self.BASELINE + 10},             timestamp=t_exit)

        _, got_entry, got_exit = coll.last
        assert got_entry == t_entry
        assert got_exit  == t_exit

    def test_entry_time_precedes_exit_time(self):
        trig, buf, coll = self._setup()
        trig.check({"weight_g": self.BASELINE + self.ENTRY + 1}, timestamp=_ts(0))
        trig.check({"weight_g": self.BASELINE + 10},             timestamp=_ts(30))
        _, entry_time, exit_time = coll.last
        assert entry_time < exit_time

    def test_snapshot_passed_to_callback_is_list(self):
        """on_visit_complete snapshot should be a list of dicts."""
        trig, buf, coll = self._setup()
        trig.check({"weight_g": self.BASELINE + self.ENTRY + 1}, timestamp=_ts(0))
        trig.check({"weight_g": self.BASELINE + 10},             timestamp=_ts(30))
        snapshot, _, _ = coll.last
        assert isinstance(snapshot, list)

    def test_second_visit_fires_second_callback(self):
        """Two complete visits produce two callback calls."""
        trig, buf, coll = self._setup()

        # Visit 1
        trig.check({"weight_g": self.BASELINE + self.ENTRY + 1}, timestamp=_ts(0))
        trig.check({"weight_g": self.BASELINE + 10},             timestamp=_ts(30))
        assert coll.count == 1

        # Re-establish baseline readings between visits
        for i in range(5):
            buf.append(_ts(40 + i * 5), {"weight_g": self.BASELINE})

        # Visit 2
        trig.check({"weight_g": self.BASELINE + self.ENTRY + 1}, timestamp=_ts(70))
        trig.check({"weight_g": self.BASELINE + 10},             timestamp=_ts(100))
        assert coll.count == 2


# ===========================================================================
# TestNoSpuriousTrigger
# ===========================================================================

class TestNoSpuriousTrigger:
    """Weight oscillates around a level that never clears the entry threshold."""

    BASELINE = 5_400.0
    ENTRY    = 300.0
    EXIT     = 200.0

    def _setup(self):
        cfg = _make_config(
            weight_entry_delta=self.ENTRY,
            weight_exit_delta=self.EXIT,
        )
        buf  = _make_buf(cfg)
        coll = _Collector()
        trig = VisitTrigger(config=cfg, buffer=buf, on_visit_complete=coll)
        # Pre-load a solid baseline
        for i in range(20):
            buf.append(_ts(i * 5), {"weight_g": self.BASELINE})
        return trig, buf, coll

    def test_oscillation_below_entry_threshold_never_fires(self):
        """Weights that never reach baseline + entry_delta must not trigger."""
        trig, _, coll = self._setup()

        # Oscillate between baseline−50 and baseline+250 (never reaching +300)
        for k in range(30):
            w = self.BASELINE + 250 * ((k % 2) * 2 - 1)  # alternates +250/-250
            trig.check({"weight_g": w}, timestamp=_ts(200 + k * 5))

        assert trig.state == KITTY_ABSENT
        assert coll.count == 0

    def test_state_stays_absent_throughout(self):
        """Confirm the state never flips to PRESENT during the oscillation."""
        trig, _, _ = self._setup()
        states = set()
        for k in range(20):
            w = self.BASELINE + 200 * ((k % 2) * 2 - 1)
            trig.check({"weight_g": w}, timestamp=_ts(200 + k * 5))
            states.add(trig.state)
        assert states == {KITTY_ABSENT}


# ===========================================================================
# TestChipIdTrigger
# ===========================================================================

class TestChipIdTrigger:
    """Entry and exit driven by the chip_id channel."""

    def _setup(self, chip_absent_consecutive: int = 3):
        cfg  = _make_config(chip_absent_consecutive=chip_absent_consecutive)
        buf  = _make_buf(cfg)
        coll = _Collector()
        trig = VisitTrigger(config=cfg, buffer=buf, on_visit_complete=coll)
        return trig, buf, coll

    def test_chip_present_triggers_entry(self):
        trig, _, coll = self._setup()
        trig.check({"chip_id": "luna"}, timestamp=_ts(0))
        assert trig.state == KITTY_PRESENT
        assert coll.count == 0

    def test_chip_none_does_not_trigger_entry(self):
        trig, _, coll = self._setup()
        for k in range(5):
            trig.check({"chip_id": None}, timestamp=_ts(k * 5))
        assert trig.state == KITTY_ABSENT
        assert coll.count == 0

    def test_chip_none_consecutive_triggers_exit(self):
        """None × N consecutive readings in PRESENT state → exit fires."""
        n = 3
        trig, _, coll = self._setup(chip_absent_consecutive=n)

        # Enter via chip
        trig.check({"chip_id": "luna"}, timestamp=_ts(0))
        assert trig.state == KITTY_PRESENT

        # N-1 None readings → no exit yet
        for k in range(1, n):
            trig.check({"chip_id": None}, timestamp=_ts(k * 5))
            assert coll.count == 0

        # Nth None reading → exit
        trig.check({"chip_id": None}, timestamp=_ts(n * 5))
        assert trig.state == KITTY_ABSENT
        assert coll.count == 1

    def test_chip_present_resets_absent_counter(self):
        """A non-null chip reading resets the consecutive-absent counter."""
        n = 3
        trig, _, coll = self._setup(chip_absent_consecutive=n)

        # Enter via chip
        trig.check({"chip_id": "luna"}, timestamp=_ts(0))

        # n-1 absent, then chip reappears, then n absent → should still fire
        for k in range(1, n):
            trig.check({"chip_id": None}, timestamp=_ts(k * 5))
        trig.check({"chip_id": "luna"}, timestamp=_ts(n * 5))   # counter reset
        for k in range(1, n):
            trig.check({"chip_id": None}, timestamp=_ts((n + k) * 5))
            assert coll.count == 0

        # Now reach n consecutive None → exit fires
        trig.check({"chip_id": None}, timestamp=_ts((2 * n) * 5))
        assert coll.count == 1

    def test_chip_entry_exit_times_correct(self):
        t_entry = _ts(0)
        t_chip_absent_1 = _ts(5)
        t_chip_absent_2 = _ts(10)
        t_chip_absent_3 = _ts(15)

        trig, _, coll = self._setup(chip_absent_consecutive=3)
        trig.check({"chip_id": "luna"}, timestamp=t_entry)
        trig.check({"chip_id": None},   timestamp=t_chip_absent_1)
        trig.check({"chip_id": None},   timestamp=t_chip_absent_2)
        trig.check({"chip_id": None},   timestamp=t_chip_absent_3)

        _, got_entry, got_exit = coll.last
        assert got_entry == t_entry
        assert got_exit  == t_chip_absent_3


# ===========================================================================
# TestSimilarityTrigger
# ===========================================================================

class TestSimilarityTrigger:
    """Entry and exit driven by similarity channels only."""

    SIM_ENTRY = 0.70
    SIM_EXIT  = 0.50
    N         = 3       # chip_absent_consecutive reused for sim exit

    def _setup(self):
        cfg  = _make_config(
            sim_entry_threshold=self.SIM_ENTRY,
            sim_exit_threshold=self.SIM_EXIT,
            chip_absent_consecutive=self.N,
        )
        buf  = _make_buf(cfg)
        coll = _Collector()
        trig = VisitTrigger(config=cfg, buffer=buf, on_visit_complete=coll)
        return trig, buf, coll

    def test_sim_above_threshold_triggers_entry(self):
        trig, _, coll = self._setup()
        trig.check({"similarity_luna": 0.85}, timestamp=_ts(0))
        assert trig.state == KITTY_PRESENT
        assert coll.count == 0

    def test_sim_below_threshold_does_not_trigger_entry(self):
        trig, _, coll = self._setup()
        for k in range(5):
            trig.check({"similarity_luna": 0.60}, timestamp=_ts(k * 5))
        assert trig.state == KITTY_ABSENT
        assert coll.count == 0

    def test_sim_below_exit_for_n_consecutive_triggers_exit(self):
        """All similarity channels below exit threshold for N consecutive ticks."""
        trig, _, coll = self._setup()

        # Enter via similarity
        trig.check({"similarity_luna": 0.85}, timestamp=_ts(0))
        assert trig.state == KITTY_PRESENT

        # N-1 below-exit readings → no exit yet
        for k in range(1, self.N):
            trig.check({"similarity_luna": 0.30}, timestamp=_ts(k * 5))
            assert coll.count == 0

        # Nth below-exit reading → exit
        trig.check({"similarity_luna": 0.30}, timestamp=_ts(self.N * 5))
        assert trig.state == KITTY_ABSENT
        assert coll.count == 1

    def test_sim_above_exit_resets_counter(self):
        """A tick where any cat is above exit threshold resets the exit counter."""
        trig, _, coll = self._setup()
        trig.check({"similarity_luna": 0.85}, timestamp=_ts(0))

        # N-1 below, then one above (counter reset), then N below → should fire
        for k in range(1, self.N):
            trig.check({"similarity_luna": 0.30}, timestamp=_ts(k * 5))
        trig.check({"similarity_luna": 0.60}, timestamp=_ts(self.N * 5))  # reset
        for k in range(1, self.N):
            trig.check({"similarity_luna": 0.30}, timestamp=_ts((self.N + k) * 5))
            assert coll.count == 0
        trig.check({"similarity_luna": 0.30}, timestamp=_ts(2 * self.N * 5))
        assert coll.count == 1

    def test_multiple_cats_all_must_be_below_for_exit(self):
        """Exit requires ALL cats below the threshold for N consecutive ticks,
        not just one."""
        trig, _, coll = self._setup()

        # Enter via luna
        trig.check({"similarity_luna": 0.85, "similarity_anna": 0.20},
                   timestamp=_ts(0))

        # luna drops, but anna rises above exit threshold → no exit
        for k in range(1, self.N + 2):
            trig.check(
                {"similarity_luna": 0.30, "similarity_anna": 0.55},
                timestamp=_ts(k * 5)
            )
        assert coll.count == 0, "anna above threshold should have reset the counter"

    def test_entry_time_and_exit_time_passed_to_callback(self):
        t_entry = _ts(0)
        t_below = [_ts(5), _ts(10), _ts(15)]

        trig, _, coll = self._setup()
        trig.check({"similarity_luna": 0.85}, timestamp=t_entry)
        for ts in t_below:
            trig.check({"similarity_luna": 0.30}, timestamp=ts)

        _, got_entry, got_exit = coll.last
        assert got_entry == t_entry
        assert got_exit  == t_below[-1]


# ===========================================================================
# TestMultipleCatsElevated
# ===========================================================================

class TestMultipleCatsElevated:
    """Two cats' similarity scores are simultaneously above the entry threshold.
    The visit should still fire exactly once."""

    SIM_ENTRY = 0.70
    SIM_EXIT  = 0.50
    N         = 3

    def _setup(self):
        cfg  = _make_config(
            sim_entry_threshold=self.SIM_ENTRY,
            sim_exit_threshold=self.SIM_EXIT,
            chip_absent_consecutive=self.N,
        )
        buf  = _make_buf(cfg)
        coll = _Collector()
        trig = VisitTrigger(config=cfg, buffer=buf, on_visit_complete=coll)
        return trig, buf, coll

    def test_two_cats_elevated_fires_entry_once(self):
        trig, _, coll = self._setup()
        trig.check({"similarity_luna": 0.85, "similarity_anna": 0.80},
                   timestamp=_ts(0))
        assert trig.state == KITTY_PRESENT
        assert coll.count == 0

    def test_two_cats_elevated_fires_callback_exactly_once_on_exit(self):
        trig, _, coll = self._setup()

        # Both cats elevated → entry
        trig.check({"similarity_luna": 0.85, "similarity_anna": 0.80},
                   timestamp=_ts(0))

        # Both cats drop below exit threshold for N ticks → exit fires once
        for k in range(1, self.N + 1):
            trig.check(
                {"similarity_luna": 0.30, "similarity_anna": 0.25},
                timestamp=_ts(k * 5)
            )

        assert coll.count == 1, f"Expected exactly 1 callback, got {coll.count}"

    def test_exit_requires_all_cats_below(self):
        """If one cat stays above exit threshold, exit must NOT fire."""
        trig, _, coll = self._setup()
        trig.check({"similarity_luna": 0.85, "similarity_anna": 0.80},
                   timestamp=_ts(0))

        # luna drops, anna stays elevated
        for k in range(1, self.N + 2):
            trig.check(
                {"similarity_luna": 0.25, "similarity_anna": 0.55},
                timestamp=_ts(k * 5)
            )
        assert coll.count == 0


# ===========================================================================
# TestReset
# ===========================================================================

class TestReset:

    def test_reset_mid_visit_returns_to_absent(self):
        cfg  = _make_config()
        buf  = _make_buf(cfg)
        coll = _Collector()
        trig = VisitTrigger(config=cfg, buffer=buf, on_visit_complete=coll)

        # Enter via chip
        trig.check({"chip_id": "luna"}, timestamp=_ts(0))
        assert trig.state == KITTY_PRESENT

        # Reset mid-visit
        trig.reset()
        assert trig.state == KITTY_ABSENT

    def test_reset_does_not_fire_callback(self):
        cfg  = _make_config()
        buf  = _make_buf(cfg)
        coll = _Collector()
        trig = VisitTrigger(config=cfg, buffer=buf, on_visit_complete=coll)

        trig.check({"chip_id": "luna"}, timestamp=_ts(0))
        trig.reset()
        assert coll.count == 0

    def test_reset_clears_consecutive_counters(self):
        """After reset, consecutive-absent counts should restart from zero."""
        n    = 3
        cfg  = _make_config(chip_absent_consecutive=n)
        buf  = _make_buf(cfg)
        coll = _Collector()
        trig = VisitTrigger(config=cfg, buffer=buf, on_visit_complete=coll)

        trig.check({"chip_id": "luna"}, timestamp=_ts(0))
        # Build up n-1 absent readings (just short of exit)
        for k in range(1, n):
            trig.check({"chip_id": None}, timestamp=_ts(k * 5))
        assert coll.count == 0

        trig.reset()

        # Re-enter; now need a fresh N absents to exit
        trig.check({"chip_id": "luna"}, timestamp=_ts(100))
        for k in range(1, n):
            trig.check({"chip_id": None}, timestamp=_ts(100 + k * 5))
            assert coll.count == 0   # counter started fresh, no exit yet
        trig.check({"chip_id": None}, timestamp=_ts(100 + n * 5))
        assert coll.count == 1

    def test_reset_from_absent_is_harmless(self):
        cfg  = _make_config()
        _, _, coll = _trigger(cfg)
        # Should not raise
        trig, buf, coll = _trigger(cfg)
        trig.reset()
        assert trig.state == KITTY_ABSENT
        assert coll.count == 0


# ===========================================================================
# TestMixedTriggers
# ===========================================================================

class TestMixedTriggers:
    """Entry triggered by chip, exit triggered by weight."""

    BASELINE = 5_400.0
    ENTRY    = 300.0
    EXIT     = 200.0

    def _setup(self):
        cfg = _make_config(
            weight_entry_delta=self.ENTRY,
            weight_exit_delta=self.EXIT,
            chip_absent_consecutive=3,
        )
        buf  = _make_buf(cfg)
        coll = _Collector()
        # Pre-load baseline weight readings
        for i in range(10):
            buf.append(_ts(i * 5), {"weight_g": self.BASELINE})
        trig = VisitTrigger(config=cfg, buffer=buf, on_visit_complete=coll)
        return trig, buf, coll

    def test_chip_entry_weight_exit(self):
        trig, _, coll = self._setup()

        # Enter via chip (no weight spike yet)
        trig.check({"chip_id": "luna", "weight_g": self.BASELINE + 50},
                   timestamp=_ts(60))
        assert trig.state == KITTY_PRESENT

        # Weight drops → exit fires
        trig.check({"chip_id": None, "weight_g": self.BASELINE + 10},
                   timestamp=_ts(120))
        assert trig.state == KITTY_ABSENT
        assert coll.count == 1

    def test_weight_entry_chip_triggers_exit_consecutively(self):
        """Weight triggers entry; chip_id absence for N ticks triggers exit."""
        trig, _, coll = self._setup()

        # Weight spike above entry threshold
        trig.check({"weight_g": self.BASELINE + self.ENTRY + 50, "chip_id": None},
                   timestamp=_ts(60))
        assert trig.state == KITTY_PRESENT

        # N chip-absent ticks → exit
        for k in range(1, 4):
            trig.check({"weight_g": self.BASELINE + 4_000, "chip_id": None},
                       timestamp=_ts(60 + k * 5))
        assert trig.state == KITTY_ABSENT
        assert coll.count == 1


# ===========================================================================
# TestRepr
# ===========================================================================

class TestRepr:

    def test_repr_contains_state(self):
        trig, _, _ = _trigger()
        assert "kitty_absent" in repr(trig)

    def test_repr_changes_after_entry(self):
        trig, _, _ = _trigger()
        trig.check({"chip_id": "luna"}, timestamp=_ts(0))
        assert "kitty_present" in repr(trig)
