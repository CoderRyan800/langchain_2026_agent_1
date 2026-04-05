"""
visit_trigger.py — Step 3 of the Time-Domain Measurement System
================================================================

Implements a two-state machine that watches the rolling buffer for signs that
a cat has entered or left the litter box and fires a callback when a complete
visit is detected.

States
------
``KITTY_ABSENT``  — the box is empty; we are waiting for an entry trigger.
``KITTY_PRESENT`` — a cat is in the box; we are waiting for an exit trigger.

Transitions
-----------

**ABSENT → PRESENT (first-stage trigger):** any ONE of:

1. ``weight_g`` exceeds ``baseline_weight + weight_entry_delta_g`` where
   ``baseline_weight`` is the median of recent weight readings in the buffer
   at the moment the trigger is evaluated.
2. ``chip_id`` is a non-null string (chip reader sees a microchip).
3. Any ``similarity_<cat>`` value exceeds ``similarity_entry_threshold``.

**PRESENT → ABSENT (second-stage trigger):** any ONE of:

1. ``weight_g`` falls below ``baseline_weight + weight_exit_delta_g``.
   The entry delta (default 300 g) is intentionally larger than the exit
   delta (default 200 g) to create a hysteresis band that prevents rapid
   oscillation around the threshold.
2. ``chip_id`` has been ``None`` for ``chip_absent_consecutive`` consecutive
   samples (default 3).
3. All ``similarity_<cat>`` values have been below ``similarity_exit_threshold``
   for ``chip_absent_consecutive`` consecutive samples.

Callback
--------
When PRESENT → ABSENT fires, ``on_visit_complete`` is called with::

    on_visit_complete(
        snapshot:    list[dict],   # full M-minute buffer copy at moment of exit
        entry_time:  datetime,     # UTC timestamp of the entry transition
        exit_time:   datetime,     # UTC timestamp of the exit transition
    )

Integration with SensorCollector
---------------------------------
``VisitTrigger.check(latest_values)`` should be called immediately after each
``SensorCollector._sample_once()`` fires.  The simplest wiring is to pass the
trigger as the ``on_sample`` callback when constructing ``SensorCollector``::

    trigger   = VisitTrigger(config, buffer, on_visit_complete=handle_visit)
    collector = SensorCollector(config, drivers, buffer,
                                on_sample=trigger.check)
    collector.start()

Testability
-----------
``check()`` accepts an optional ``timestamp`` keyword argument.  When provided
(typically in tests), that datetime is used for entry/exit timestamps instead
of ``datetime.now(timezone.utc)``.  This makes time-dependent assertions
deterministic without mocking the clock.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Callable, Optional

from litterbox.time_buffer import RollingBuffer


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

KITTY_ABSENT:  str = "kitty_absent"
KITTY_PRESENT: str = "kitty_present"

# Sentinel used to distinguish "key absent from values dict" (channel disabled
# or not measured this tick) from "key present with value None" (chip reader is
# active but detected no chip).  Without this distinction, a values dict that
# simply omits chip_id (e.g. a weight-only tick in tests) would be incorrectly
# counted as a chip-absent sample.
_KEY_MISSING = object()


# ===========================================================================
# VisitTrigger
# ===========================================================================

class VisitTrigger:
    """Two-state machine that detects litter-box visit start and end.

    Parameters
    ----------
    config:
        Parsed ``td_config.json`` dict.  All threshold values are read from
        ``config["trigger"]``.
    buffer:
        The shared ``RollingBuffer``.  Used to (a) compute the baseline weight
        and (b) take the full-window snapshot on visit exit.
    on_visit_complete:
        Callable invoked when a visit ends.  Signature::

            def on_visit_complete(
                snapshot:   list[dict],
                entry_time: datetime,
                exit_time:  datetime,
            ) -> None: ...
    """

    def __init__(
        self,
        config: dict,
        buffer: RollingBuffer,
        on_visit_complete: Callable,
    ) -> None:
        trig = config.get("trigger", {})

        # Entry thresholds
        self._weight_entry_delta: float = float(
            trig.get("weight_entry_delta_g", 300)
        )
        # Exit thresholds
        self._weight_exit_delta: float = float(
            trig.get("weight_exit_delta_g", 200)
        )
        self._chip_absent_consecutive: int = int(
            trig.get("chip_absent_consecutive", 3)
        )
        self._sim_entry_threshold: float = float(
            trig.get("similarity_entry_threshold", 0.70)
        )
        self._sim_exit_threshold: float = float(
            trig.get("similarity_exit_threshold", 0.50)
        )

        self._buffer            = buffer
        self._on_visit_complete = on_visit_complete

        # State
        self._state:            str                 = KITTY_ABSENT
        self._entry_time:       Optional[datetime]  = None
        self._baseline_weight:  Optional[float]     = None

        # Consecutive-sample counters used by the exit conditions.
        # Both are reset to 0 whenever we transition to KITTY_PRESENT.
        self._chip_absent_count: int = 0
        self._sim_below_count:   int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current state: ``KITTY_ABSENT`` or ``KITTY_PRESENT``."""
        return self._state

    def check(
        self,
        latest_values: dict,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Evaluate the latest sensor reading against the current state.

        Call this after every ``SensorCollector._sample_once()`` tick.

        Parameters
        ----------
        latest_values:
            The values dict produced by the most recent buffer append.
            Keys are channel names, e.g. ``"weight_g"``, ``"chip_id"``,
            ``"similarity_luna"``.
        timestamp:
            Optional explicit datetime for this tick.  If ``None``,
            ``datetime.now(timezone.utc)`` is used.  Supplying an explicit
            timestamp makes unit tests deterministic without clock mocking.
        """
        now = timestamp if timestamp is not None else datetime.now(timezone.utc)

        if self._state == KITTY_ABSENT:
            self._check_entry(latest_values, now)
        else:
            self._check_exit(latest_values, now)

    def reset(self) -> None:
        """Force the state machine back to KITTY_ABSENT without firing the callback.

        Used by tests and the simulator to reset mid-visit without triggering
        visit analysis.  Clears all internal counters and timestamps.
        """
        self._state            = KITTY_ABSENT
        self._entry_time       = None
        self._baseline_weight  = None
        self._chip_absent_count = 0
        self._sim_below_count   = 0

    # ------------------------------------------------------------------
    # Entry condition evaluation
    # ------------------------------------------------------------------

    def _check_entry(self, values: dict, now: datetime) -> None:
        """Evaluate first-stage (ABSENT → PRESENT) trigger conditions.

        Fires _on_entry() at the first matching condition and returns
        immediately so only one transition fires per tick.
        """
        # --- Condition 1: weight spike ---
        weight = values.get("weight_g")
        if weight is not None:
            # Compute (or refresh) the baseline from the buffer at this moment.
            # We do this on every ABSENT tick so the baseline stays current
            # even during long idle periods.
            bw = self._compute_baseline_weight()
            if bw is not None:
                self._baseline_weight = bw
                if weight > bw + self._weight_entry_delta:
                    self._on_entry(now)
                    return

        # --- Condition 2: chip ID present ---
        # chip_id key is present with a non-None string value.
        chip_id = values.get("chip_id")
        if chip_id is not None:
            self._on_entry(now)
            return

        # --- Condition 3: any similarity channel above entry threshold ---
        for key, val in values.items():
            if key.startswith("similarity_") and val is not None:
                if val > self._sim_entry_threshold:
                    self._on_entry(now)
                    return

    # ------------------------------------------------------------------
    # Exit condition evaluation
    # ------------------------------------------------------------------

    def _check_exit(self, values: dict, now: datetime) -> None:
        """Evaluate second-stage (PRESENT → ABSENT) trigger conditions.

        Fires _on_exit() at the first matching condition and returns.
        Consecutive-sample counters are updated on every PRESENT tick
        even when no exit fires, so they accumulate correctly across ticks.
        """
        # --- Condition 1: weight dropped back to near baseline ---
        weight = values.get("weight_g")
        if (weight is not None
                and self._baseline_weight is not None
                and weight < self._baseline_weight + self._weight_exit_delta):
            self._on_exit(now)
            return

        # --- Condition 2: chip absent for N consecutive samples ---
        # Use the sentinel to distinguish two very different situations:
        #   chip_id key ABSENT  → channel disabled / not measured this tick
        #                         → do NOT count toward the consecutive total
        #   chip_id = None      → reader is active, no chip in range
        #                         → count toward the consecutive total
        #   chip_id = "luna"    → chip detected → reset the counter
        chip_reading = values.get("chip_id", _KEY_MISSING)
        if chip_reading is not _KEY_MISSING:
            if chip_reading is None:
                self._chip_absent_count += 1
            else:
                self._chip_absent_count = 0

            if self._chip_absent_count >= self._chip_absent_consecutive:
                self._on_exit(now)
                return

        # --- Condition 3: all similarity channels below exit threshold ---
        sim_keys = [k for k in values if k.startswith("similarity_")]
        if sim_keys:
            # A None value (missing camera frame) is treated as "not above
            # threshold", i.e. counts as below the exit threshold.
            all_below = all(
                values[k] is None or float(values[k]) < self._sim_exit_threshold
                for k in sim_keys
            )
            if all_below:
                self._sim_below_count += 1
            else:
                self._sim_below_count = 0

            if self._sim_below_count >= self._chip_absent_consecutive:
                self._on_exit(now)
                return

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _on_entry(self, now: datetime) -> None:
        """Transition to KITTY_PRESENT."""
        self._state      = KITTY_PRESENT
        self._entry_time = now
        # Reset exit counters so they start fresh for this visit.
        self._chip_absent_count = 0
        self._sim_below_count   = 0

    def _on_exit(self, now: datetime) -> None:
        """Transition to KITTY_ABSENT and fire the visit-complete callback."""
        exit_time   = now
        entry_time  = self._entry_time
        snapshot    = self._buffer.snapshot()

        # Reset state BEFORE calling the callback so re-entrant calls from
        # within the callback see a clean state.
        self._state            = KITTY_ABSENT
        self._entry_time       = None
        self._baseline_weight  = None
        self._chip_absent_count = 0
        self._sim_below_count   = 0

        self._on_visit_complete(snapshot, entry_time, exit_time)

    # ------------------------------------------------------------------
    # Baseline weight helper
    # ------------------------------------------------------------------

    def _compute_baseline_weight(self) -> Optional[float]:
        """Return the median of all available weight readings in the buffer.

        Uses the full buffer so that even a sparse window (few samples) gives
        a reasonable estimate.  Returns ``None`` if no weight readings exist
        in the buffer yet.
        """
        weights = [
            v for v in self._buffer.get_channel("weight_g")
            if v is not None
        ]
        if not weights:
            return None
        return statistics.median(weights)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VisitTrigger("
            f"state={self._state!r}, "
            f"baseline_weight={self._baseline_weight}, "
            f"chip_absent_count={self._chip_absent_count}, "
            f"sim_below_count={self._sim_below_count})"
        )
