# Step 2 Test Report — SensorCollector and Hardware Driver Interface

**Date:** 2026-04-04
**Branch:** `feature/time_domain_measurements`
**Python:** 3.12.12
**pytest:** 9.0.2

---

## Summary

| Metric | Value |
|--------|-------|
| Tests collected | 38 |
| Tests passed | **38** |
| Tests failed | 0 |
| Run time | ~37 s (includes two real 0.4 s thread-timing tests) |

All 38 tests pass.  No warnings, no skips.

---

## Files Under Test

| File | Role |
|------|------|
| `src/litterbox/sensor_collector.py` | `BaseDriver` ABC, all 5 driver classes, `SensorCollector` |

---

## Test Classes and Coverage

### `TestBaseDriver` (3 tests)

Verifies the abstract base class contract.

| Test | Assertion |
|------|-----------|
| `test_cannot_instantiate` | `BaseDriver()` raises `TypeError` (abstract class) |
| `test_concrete_subclass_without_read_fails` | A subclass that omits `read()` cannot be instantiated |
| `test_concrete_subclass_with_read_succeeds` | A minimal concrete subclass with `read()` works correctly |

---

### `TestWeightDriver` (4 tests)

| Test | Assertion |
|------|-----------|
| `test_zero_noise_returns_base` | `noise_sigma=0` returns exactly `base_value` |
| `test_with_noise_varies` | 100 noisy readings are not all identical |
| `test_with_noise_near_base` | All 200 readings fall within ±4 σ of `base_value` (99.99 % CI) |
| `test_default_args` | `WeightDriver()` returns 0.0 with no arguments |

---

### `TestAmmoniaDriver` (3 tests)

| Test | Assertion |
|------|-----------|
| `test_zero_noise_returns_base` | Returns exactly `base_value` with zero noise |
| `test_clamped_to_zero` | 100 readings with very large noise still all return ≥ 0.0 |
| `test_default_args` | Returns 0.0 with no arguments |

---

### `TestMethaneDriver` (3 tests)

Same pattern as `AmmoniaDriver` — zero noise, zero clamp, default args.

---

### `TestChipIdDriver` (3 tests)

| Test | Assertion |
|------|-----------|
| `test_returns_cat_name` | Returns the configured string exactly |
| `test_returns_none_by_default` | Default constructor returns `None` |
| `test_explicit_none` | `ChipIdDriver(cat_name=None)` returns `None` |

---

### `TestSimilarityDriver` (4 tests)

| Test | Assertion |
|------|-----------|
| `test_returns_cat_scores` | Returns the configured `{cat: score}` dict |
| `test_returns_none_for_missing_frame` | `cat_scores=None` constructor → `read()` returns `None` |
| `test_default_is_none` | Default constructor returns `None` |
| `test_returns_copy` | Mutating the returned dict does not change the driver's internal state |

The copy-on-read test is important: `SimilarityDriver` stores the reference scores
internally.  If `read()` returned the internal dict directly, a caller could
accidentally mutate the driver's state, causing subsequent reads to return
wrong values.

---

### `TestSampleOnce` (9 tests)

The most important class — tests `SensorCollector._sample_once()` in isolation
(called directly, not via the background thread).

| Test | Assertion |
|------|-----------|
| `test_scalar_channels_written_to_buffer` | weight, ammonia, methane, chip_id values all appear in the buffer entry |
| `test_similarity_dict_expanded_into_per_cat_keys` | `{anna: 0.91, luna: 0.23}` → `similarity_anna`, `similarity_luna` keys in buffer |
| `test_similarity_none_means_no_keys_written` | `None` from `SimilarityDriver` → zero `similarity_*` keys written |
| `test_disabled_channel_not_sampled` | Channel with `enabled: false` in config produces no buffer key |
| `test_missing_driver_skipped_silently` | Channel with no matching driver is skipped without raising |
| `test_timestamp_is_utc` | Appended timestamp is timezone-aware UTC |
| `test_multiple_sample_once_appends_multiple_entries` | Three calls → three buffer entries |
| `test_chip_id_none_stored_as_none` | `chip_id=None` is stored as `None` (key present, value None) |
| `test_five_cats_similarity_expanded` | Five-cat dict → five `similarity_<catname>` keys, correct scores |

#### Key semantic distinction: `chip_id=None` vs missing `similarity_*`

These two cases look similar but have different semantics and different buffer
representations:

- `chip_id = None` — the chip reader is active and found no chip.  The key
  **IS** written with `None`.  Downstream code can use this to count
  consecutive "chip absent" samples.

- `SimilarityDriver` returning `None` — the camera produced no usable frame
  this tick.  **No** `similarity_*` keys are written at all.  The buffer
  entry genuinely lacks those keys, so `to_dataframe()` fills them with `NaN`
  (not 0.0 and not `None`).

---

### `TestRunMultipleTicks` (2 tests)

Uses the actual background thread at 600 samples/minute (0.1 s interval) to
verify timing behaviour.

| Test | Assertion |
|------|-----------|
| `test_three_ticks_produce_three_entries` | At least 3 entries appear in the buffer after 0.4 s |
| `test_entries_have_monotonic_timestamps` | Buffer timestamps are non-decreasing across ticks |

---

### `TestStop` (4 tests)

| Test | Assertion |
|------|-----------|
| `test_stop_within_two_seconds` | `stop()` returns and thread is dead within 2 s |
| `test_stop_idempotent` | Calling `stop()` twice does not raise |
| `test_stop_before_start_is_safe` | `stop()` before `start()` does not raise |
| `test_double_start_raises` | Calling `start()` while already running raises `RuntimeError` |

The `stop()` latency test (≤ 2 s) relies on the `threading.Event.wait(timeout)`
pattern used in `_run()`.  Because the thread waits on an event rather than
sleeping, `stop()` sets the event and the thread wakes immediately — it does
not need to wait for the full tick interval to expire.

---

### `TestRepr` (3 tests)

| Test | Assertion |
|------|-----------|
| `test_repr_contains_interval` | repr contains the tick interval in seconds (e.g. `5.0s`) |
| `test_repr_contains_running_false` | repr shows `running=False` before `start()` |
| `test_repr_contains_running_true_while_active` | repr shows `running=True` after `start()` |

---

## Bugs Found and Fixed

None during Step 2 development.  All 38 tests passed on the first run.

---

## Design Verification

- `BaseDriver` correctly enforces the abstract interface — concrete subclasses
  without `read()` cannot be instantiated.
- `SimilarityDriver.read()` returns a copy of the scores dict, preventing
  inadvertent mutation of driver state by callers.
- `_sample_once()` correctly handles both the normal scalar case and the
  `similarity` special case in one pass over enabled channels.
- The `on_sample` callback (wired to `VisitTrigger.check` in Step 3) is called
  after the buffer append, so the trigger always sees the most up-to-date
  buffer state.
