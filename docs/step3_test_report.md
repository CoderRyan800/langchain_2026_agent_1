# Step 3 Test Report — VisitTrigger State Machine

**Date:** 2026-04-04
**Branch:** `feature/time_domain_measurements`
**Python:** 3.12.12
**pytest:** 9.0.2

---

## Summary

| Metric | Value |
|--------|-------|
| Tests collected | 34 |
| Tests passed | **34** |
| Tests failed | 0 (after 1 bug fix — see below) |
| Initial failures | 4 |
| Run time | ~5 s |

All 34 tests pass.  No warnings, no skips.

---

## Files Under Test

| File | Role |
|------|------|
| `src/litterbox/visit_trigger.py` | `VisitTrigger` state machine, `KITTY_ABSENT`/`KITTY_PRESENT` constants |
| `src/litterbox/sensor_collector.py` | `on_sample` callback hook (added in Step 3) |

---

## Test Classes and Coverage

### `TestStateConstants` (3 tests)

| Test | Assertion |
|------|-----------|
| `test_kitty_absent_value` | `KITTY_ABSENT == "kitty_absent"` |
| `test_kitty_present_value` | `KITTY_PRESENT == "kitty_present"` |
| `test_constants_are_distinct` | The two constants are not equal |

---

### `TestInitialState` (2 tests)

| Test | Assertion |
|------|-----------|
| `test_starts_absent` | Freshly constructed trigger starts in `KITTY_ABSENT` |
| `test_no_callback_before_any_check` | No callback fires before `check()` is called |

---

### `TestWeightTrigger` (5 tests)

Tests the weight-only path: flat baseline → sharp rise (entry) → elevated plateau → drop back (exit).

The buffer is pre-populated with 10 identical baseline readings so the median
baseline is well-established before any `check()` calls are made.

| Test | Assertion |
|------|-----------|
| `test_weight_ramp_fires_callback_exactly_once` | Flat → rise → fall produces exactly one callback |
| `test_entry_and_exit_times_correct` | `entry_time` and `exit_time` match the explicit timestamps passed to `check()` |
| `test_entry_time_precedes_exit_time` | `entry_time < exit_time` always |
| `test_snapshot_passed_to_callback_is_list` | The `snapshot` argument is a `list` of dicts |
| `test_second_visit_fires_second_callback` | Two complete weight-ramp visits fire two callbacks |

**Hysteresis:** entry threshold is `baseline + 300 g`, exit threshold is
`baseline + 200 g`.  This 100 g dead band prevents rapid oscillation around
the boundary.  The test feeds values well clear of both thresholds to produce
clean transitions.

---

### `TestNoSpuriousTrigger` (2 tests)

Verifies that weight noise below the entry delta does not produce false entries.

The synthetic signal oscillates between `baseline − 250 g` and `baseline + 250 g`
— never reaching `baseline + 300 g` (the entry threshold).

| Test | Assertion |
|------|-----------|
| `test_oscillation_below_entry_threshold_never_fires` | 30 oscillating samples → 0 callbacks |
| `test_state_stays_absent_throughout` | State is always `KITTY_ABSENT` during oscillation |

---

### `TestChipIdTrigger` (5 tests)

Tests the chip-ID path: chip present → entry; chip absent × N → exit.

| Test | Assertion |
|------|-----------|
| `test_chip_present_triggers_entry` | Non-null `chip_id` value → `KITTY_PRESENT` |
| `test_chip_none_does_not_trigger_entry` | `chip_id=None` × 5 ticks → still `KITTY_ABSENT` |
| `test_chip_none_consecutive_triggers_exit` | N consecutive `chip_id=None` ticks → callback fires |
| `test_chip_present_resets_absent_counter` | A non-null reading mid-sequence resets the counter; need N more absents |
| `test_chip_entry_exit_times_correct` | Entry and exit timestamps match the explicit check() timestamps |

---

### `TestSimilarityTrigger` (6 tests)

Tests the similarity path: score above entry threshold → entry; all cats below exit threshold × N → exit.

| Test | Assertion |
|------|-----------|
| `test_sim_above_threshold_triggers_entry` | One cat above 0.70 → `KITTY_PRESENT` |
| `test_sim_below_threshold_does_not_trigger_entry` | All cats at 0.60 × 5 ticks → still `KITTY_ABSENT` |
| `test_sim_below_exit_for_n_consecutive_triggers_exit` | All cats below 0.50 for N ticks → callback |
| `test_sim_above_exit_resets_counter` | Any cat above 0.50 resets the consecutive counter |
| `test_multiple_cats_all_must_be_below_for_exit` | Exit requires ALL cats below 0.50, not just one |
| `test_entry_time_and_exit_time_passed_to_callback` | Times match the explicit timestamps |

---

### `TestMultipleCatsElevated` (3 tests)

Two cats are both above the entry threshold simultaneously.  The visit should
still produce exactly one entry and one exit callback.

| Test | Assertion |
|------|-----------|
| `test_two_cats_elevated_fires_entry_once` | Entry fires on the first `check()` call, not twice |
| `test_two_cats_elevated_fires_callback_exactly_once_on_exit` | One callback after N ticks of both cats below exit threshold |
| `test_exit_requires_all_cats_below` | If one cat remains above 0.50, exit does not fire |

---

### `TestReset` (4 tests)

| Test | Assertion |
|------|-----------|
| `test_reset_mid_visit_returns_to_absent` | `reset()` during `KITTY_PRESENT` → `KITTY_ABSENT` |
| `test_reset_does_not_fire_callback` | `reset()` never calls `on_visit_complete` |
| `test_reset_clears_consecutive_counters` | After reset, need a fresh N absents to exit on next visit |
| `test_reset_from_absent_is_harmless` | `reset()` while already absent does not raise |

---

### `TestMixedTriggers` (2 tests)

Entry from one channel type, exit from a different channel type.

| Test | Assertion |
|------|-----------|
| `test_chip_entry_weight_exit` | Chip triggers entry; weight drop triggers exit |
| `test_weight_entry_chip_triggers_exit_consecutively` | Weight spike triggers entry; N chip-absent ticks trigger exit |

---

### `TestRepr` (2 tests)

| Test | Assertion |
|------|-----------|
| `test_repr_contains_state` | repr shows `"kitty_absent"` in initial state |
| `test_repr_changes_after_entry` | repr shows `"kitty_present"` after an entry trigger |

---

## Bug Found and Fixed

### Root cause: chip_id absent key counted as chip absent

**Initial failure count:** 4 tests

**Failing tests:**
- `TestWeightTrigger::test_weight_ramp_fires_callback_exactly_once`
- `TestSimilarityTrigger::test_sim_above_exit_resets_counter`
- `TestSimilarityTrigger::test_multiple_cats_all_must_be_below_for_exit`
- `TestMultipleCatsElevated::test_exit_requires_all_cats_below`

**Symptom:** Exit callbacks fired during the middle of a visit even when the
weight was still elevated and similarity scores were still above the exit
threshold.

**Root cause:** In `_check_exit()`, the original code used:

```python
chip_id = values.get("chip_id")
if chip_id is None:
    self._chip_absent_count += 1
```

`dict.get()` returns `None` for both:
1. Key present with value `None` — chip reader active, no chip detected.
2. Key absent from dict — chip channel disabled, or values dict only contains
   weight/similarity keys (as in several tests).

In case 2, the code was incorrectly incrementing the counter.  After 3 ticks
of weight-only or similarity-only values dicts (with no `chip_id` key),
`chip_absent_count` reached the threshold and fired a spurious exit.

**Fix:** Added a module-level sentinel `_KEY_MISSING = object()` to distinguish
the two cases:

```python
chip_reading = values.get("chip_id", _KEY_MISSING)
if chip_reading is not _KEY_MISSING:
    if chip_reading is None:
        self._chip_absent_count += 1
    else:
        self._chip_absent_count = 0
    if self._chip_absent_count >= self._chip_absent_consecutive:
        self._on_exit(now)
        return
# Key absent → channel disabled / not measured this tick → skip
```

**Impact:** The fix correctly models the intended semantics:
- `chip_id` key absent → ignore for consecutive counting (channel not active)
- `chip_id = None` → count as "chip not present" (reader active, no chip)
- `chip_id = "luna"` → reset counter (chip detected)

After the fix, all 34 tests pass.

---

## Design Verification

**Testability via explicit timestamps:** `check(values, timestamp=None)` accepts
an optional explicit `datetime`.  All tests supply explicit timestamps, making
assertions on `entry_time` and `exit_time` fully deterministic without any
clock mocking.

**Pre-emptive exit ordering:** Within `_check_exit()`, the weight condition is
checked first, then chip, then similarity.  Weight is the most reliable sensor
(a digital scale is harder to fool than a camera or RFID reader), so it gets
priority.  If the weight says the cat has left, there is no need to count chip
or similarity samples.

**State reset before callback:** In `_on_exit()`, all state fields are reset
*before* calling `on_visit_complete`.  This means if the callback itself calls
`check()` (re-entrant use), the trigger is already in a clean `KITTY_ABSENT`
state.

**Baseline weight accuracy:** The baseline is the median (not mean) of all
available weight readings in the buffer at the moment of the entry trigger.
Median is robust to outliers — a single anomalous reading does not shift the
baseline significantly.  After a visit ends, the baseline is cleared so it is
recomputed from fresh buffer data at the start of the next visit.
