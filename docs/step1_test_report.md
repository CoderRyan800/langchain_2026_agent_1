# Step 1 Test Report — Time-Domain Measurement System

**Date:** 2026-04-04
**Branch:** `feature/time_domain_measurements`
**Python:** 3.12.12
**pytest:** 9.0.2
**Bokeh:** 3.9.0
**pandas:** 3.0.2

---

## Summary

| Metric | Value |
|--------|-------|
| Tests collected | 42 |
| Tests passed | **42** |
| Tests failed | 0 |
| Run time | ~5 s |

All 42 tests pass.  No warnings, no skips.

---

## Files Under Test

| File | Role |
|------|------|
| `src/litterbox/time_buffer.py` | `RollingBuffer` class + `load_td_config()` |
| `src/litterbox/td_config.json` | Default time-domain configuration |
| `src/litterbox/td_plot.py` | Abstract `PlotBackend` interface + factory |
| `src/litterbox/td_plot_bokeh.py` | Bokeh implementation of `PlotBackend` |

---

## Test Classes and Coverage

### `TestCapacity` (6 tests)

Verifies that the buffer correctly enforces its `maxlen = window_minutes ×
samples_per_minute` capacity.

| Test | Assertion |
|------|-----------|
| `test_capacity_not_exceeded` | len ≤ maxlen after N > maxlen appends |
| `test_oldest_entries_evicted` | oldest entry disappears when buffer overflows |
| `test_exact_full` | len == maxlen when exactly maxlen entries added |
| `test_empty_initially` | freshly created buffer has len == 0 |
| `test_constructor_rejects_zero_window` | `ValueError` for window_minutes ≤ 0 |
| `test_constructor_rejects_zero_rate` | `ValueError` for samples_per_minute ≤ 0 |

---

### `TestGetChannel` (4 tests)

Verifies per-channel value retrieval and missing-key sentinel behaviour.

| Test | Assertion |
|------|-----------|
| `test_returns_present_values` | values present in entries are returned correctly |
| `test_returns_none_for_absent_key` | `None` returned for an entry that omitted the key |
| `test_unknown_channel_all_none` | `get_channel("nonexistent")` returns all-`None` list |
| `test_values_ordered_oldest_first` | first element == oldest entry's value |

---

### `TestGetTimestamps` (2 tests)

| Test | Assertion |
|------|-----------|
| `test_returns_timestamps_in_order` | list is monotonically non-decreasing |
| `test_empty_buffer_returns_empty_list` | returns `[]` on an empty buffer |

---

### `TestWindowSpan` (4 tests)

| Test | Assertion |
|------|-----------|
| `test_correct_span` | span == (last_ts − first_ts).total_seconds() |
| `test_single_entry_returns_zero` | 0.0 for a one-element buffer |
| `test_empty_buffer_returns_zero` | 0.0 for an empty buffer |
| `test_span_reflects_eviction` | span advances when old entries are evicted |

---

### `TestSnapshot` (4 tests)

Verifies copy-on-read semantics — mutating the returned list or its dicts must
not affect the buffer's internal state.

| Test | Assertion |
|------|-----------|
| `test_snapshot_is_a_copy` | appending to snapshot does not change `len(buf)` |
| `test_snapshot_values_are_copies` | mutating snapshot entry's values dict leaves buffer unchanged |
| `test_snapshot_ordered_oldest_first` | first element is the oldest entry |
| `test_snapshot_includes_timestamp` | each snapshot entry has a `"timestamp"` key |

---

### `TestClear` (2 tests)

| Test | Assertion |
|------|-----------|
| `test_clear_empties_buffer` | `len(buf) == 0` after `clear()` |
| `test_can_append_after_clear` | buffer accepts new entries after `clear()` |

---

### `TestRepr` (1 test)

| Test | Assertion |
|------|-----------|
| `test_repr_contains_key_info` | repr contains window size, rate, and capacity |

---

### `TestToDataframe` (9 tests)

The most semantically rich test class.  It validates the bridge between the
raw circular buffer and Step 4's cat-identification logic.

| Test | Assertion |
|------|-----------|
| `test_all_channels_no_prefix` | all channel keys become columns when prefix=None |
| `test_channel_prefix_strips_prefix` | `"similarity_anna"` column becomes `"anna"` |
| `test_nan_for_absent_keys` | absent key in an entry's values dict → `NaN` in DataFrame |
| `test_nan_vs_zero_correctness` | **critical** — see detail below |
| `test_timestamp_index` | DataFrame index matches `buf.get_timestamps()` |
| `test_time_range_filter_start` | entries before `start` are excluded |
| `test_time_range_filter_end` | entries after `end` are excluded |
| `test_empty_buffer_returns_empty_dataframe` | empty buffer → empty DataFrame |
| `test_serialisation_round_trip` | `to_json(orient="split")` → `read_json(StringIO(...))` round-trip preserves values and NaN positions |

#### Critical test: `test_nan_vs_zero_correctness`

This test exists to document *why* absent camera frames must become `NaN` and
not `0.0`.  The scenario is a 3-sample window for two cats where only the
middle sample has no camera frame:

```
t0: similarity_anna=0.91, similarity_luna=0.23
t1: {}                    ← no camera frame this tick
t2: similarity_anna=0.89, similarity_luna=0.25
```

With **NaN (correct)**:
```
anna.mean(skipna=True) = (0.91 + 0.89) / 2 = 0.900  → identified ✓
luna.mean(skipna=True) = (0.23 + 0.25) / 2 = 0.240  → not matched ✓
```

With **zero-fill (wrong)**:
```
anna.mean() = (0.91 + 0.00 + 0.89) / 3 = 0.600  → below threshold → Unknown ✗
luna.mean() = (0.23 + 0.00 + 0.25) / 3 = 0.160  → not matched ✓ (correct for wrong reason)
```

Zero-fill would cause a false "Unknown" for anna even though she was clearly
identified during the two frames where the camera was active.  NaN with
`skipna=True` gives the correct answer.

---

### `TestLoadTdConfig` (7 tests)

| Test | Assertion |
|------|-----------|
| `test_loads_default_config` | returns dict with all 5 required keys |
| `test_raises_file_not_found` | `FileNotFoundError` for nonexistent path |
| `test_raises_on_missing_required_key[window_minutes]` | `ValueError` |
| `test_raises_on_missing_required_key[samples_per_minute]` | `ValueError` |
| `test_raises_on_missing_required_key[channels]` | `ValueError` |
| `test_raises_on_missing_required_key[trigger]` | `ValueError` |
| `test_raises_on_missing_required_key[image_retention_days]` | `ValueError` |
| `test_raises_on_wrong_type_for_window_minutes` | `ValueError` |
| `test_loads_real_config_file` | loads the actual `td_config.json` from the package |

---

### `TestThreadSafety` (1 test)

| Test | Assertion |
|------|-----------|
| `test_concurrent_appends_no_corruption` | two threads each appending 500 items produce `len(buf) == maxlen` with no exceptions |

This test exercises the `threading.Lock` that guards every public method.
Without the lock, concurrent appends to a Python `deque` can produce torn
reads that corrupt the internal state.

---

## Demo Script Output

`tests/step1_demo.py` was run separately to validate end-to-end behaviour with
synthetic data and Bokeh plot generation:

```
[1] Loading td_config.json ...
    window_minutes     : 10
    samples_per_minute : 12
    channels           : ['weight_g', 'ammonia_ppb', 'methane_ppb', 'chip_id', 'similarity']
    image_retention_days: 7

[2] Building synthetic RollingBuffer ...
    RollingBuffer(window=10m, rate=12/min, capacity=120, used=120)
    window span : 595.0 s
    is_full     : True
    weight_g    : min=5356 g  max=10953 g
    ammonia_ppb : min=6.1  max=66.5

[3] Converting similarity channels to DataFrame ...
    shape  : (120, 5)  (rows × cats)
    columns: ['anna', 'marina', 'natasha', 'whiskers', 'luna']
    non-NaN counts per cat:
      anna      : 49 / 120
      luna      : 71 / 120
    mean similarity per cat (NaN-skipped):
      anna      : 0.1832
      luna      : 0.8377   ← visiting cat clearly dominant

[4] Generating Bokeh HTML plots ...
    Saved: output/step1_channels.html
    Saved: output/step1_similarity.html

[6] Time-range filtering — visit window only ...
    visit window rows : 71  (expected ~71)
    luna mean in window: 0.8377
```

Two standalone HTML files were generated.  Each is self-contained (Bokeh JS
inlined) and can be opened offline in any browser.

---

## Bugs Found and Fixed

| # | Location | Bug | Fix |
|---|----------|-----|-----|
| 1 | `td_plot_bokeh.py` | `figure(x_range=None)` raised `ValueError` in Bokeh 3.x | Omit `x_range` kwarg entirely for the first panel; pass it only for subsequent panels |
| 2 | `td_plot_bokeh.py` | `p.circle()` was deprecated in Bokeh 3.4 | Replaced with `p.scatter()` |
| 3 | `test_time_buffer.py` | `pd.read_json(json_str)` raises `FileNotFoundError` in pandas 3.x (treats string as file path) | Wrap with `io.StringIO(json_str)` |

---

## Design Verification

The modular plotter architecture was verified by inspection:

- `td_plot.py` defines `PlotBackend` (ABC) and `get_plot_backend()` factory.
- `td_plot_bokeh.py` is the **only** file that imports Bokeh.
- No other module in the time-domain subsystem imports Bokeh directly.
- Swapping to a different library requires only creating `td_plot_<name>.py` and changing the `get_plot_backend()` call-site argument — zero other files change.
