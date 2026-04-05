"""
time_buffer.py — Step 1 of the Time-Domain Measurement System
==============================================================

Provides two things:

  1. ``RollingBuffer`` — a thread-safe, fixed-capacity circular buffer that
     holds time-stamped measurement dictionaries.  The buffer is completely
     hardware-agnostic: it knows nothing about cats, sensors, or the litter
     box.  It simply stores whatever numeric or string values the caller
     appends, and discards the oldest entry whenever the buffer is full.

  2. ``load_td_config`` — reads and validates ``td_config.json``, the single
     JSON file that governs the entire time-domain subsystem.

Design decisions
----------------
* ``collections.deque(maxlen=N)`` gives O(1) append and automatic eviction of
  the oldest entry.  No manual index arithmetic needed.

* A ``threading.Lock`` guards every public method so that a background
  sampling thread (added in Step 2) and the main thread can both access the
  buffer concurrently without data corruption.

* Each buffer entry is a plain dict::

      {"timestamp": datetime, "values": {"weight_g": 4123.0, "ammonia_ppb": 22.1, ...}}

  The ``values`` sub-dict is **copied** on both write (``append``) and read
  (``snapshot``, ``get_channel``).  This prevents a caller from accidentally
  mutating a stored entry.

* ``to_dataframe`` is the bridge to Step 4's cat-identification logic.  It
  converts the buffer into a pandas DataFrame with a datetime index.  When
  called with ``channel_prefix="similarity_"`` it produces the M×K similarity
  DataFrame described in CLAUDE.md — one column per registered cat, NaN where
  a sample is missing.

Usage example
-------------
::

    from litterbox.time_buffer import RollingBuffer, load_td_config
    from datetime import datetime, timezone

    cfg    = load_td_config()          # reads td_config.json
    buf    = RollingBuffer(
                window_minutes     = cfg["window_minutes"],
                samples_per_minute = cfg["samples_per_minute"])

    buf.append(datetime.now(timezone.utc), {"weight_g": 4200.0, "ammonia_ppb": 18.5})
    buf.append(datetime.now(timezone.utc), {"weight_g": 7350.0, "ammonia_ppb": 21.0})

    print(buf.get_channel("weight_g"))   # [4200.0, 7350.0]
    print(buf.window_span_seconds())     # seconds between first and last entry
    print(len(buf))                      # 2
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Union

# ---------------------------------------------------------------------------
# Default path to the configuration file (same directory as this module).
# Callers can override by passing a path to load_td_config().
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH: Path = Path(__file__).parent / "td_config.json"

# Keys that MUST be present in td_config.json, and their expected Python types.
_REQUIRED_KEYS: dict[str, type] = {
    "window_minutes":    int,
    "samples_per_minute": int,
    "channels":          list,
    "trigger":           dict,
    "image_retention_days": int,
}


# ===========================================================================
# Configuration loader
# ===========================================================================

def load_td_config(path: Optional[Union[str, Path]] = None) -> dict:
    """Load and validate the time-domain configuration file.

    Parameters
    ----------
    path:
        Path to a JSON configuration file.  If ``None`` (default), the file
        ``td_config.json`` located in the same directory as this module is
        used.

    Returns
    -------
    dict
        The parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist at the resolved path.
    ValueError
        If a required top-level key is missing, or its value has the wrong
        Python type.

    Notes
    -----
    The loader performs only *structural* validation (presence and type of
    top-level keys).  It does not validate individual channel entries or
    trigger threshold values — those are validated by the modules that
    consume them (Step 2 onwards).
    """
    resolved = Path(path) if path is not None else _DEFAULT_CONFIG_PATH

    if not resolved.exists():
        raise FileNotFoundError(
            f"Time-domain config not found at '{resolved}'. "
            "Create td_config.json or pass an explicit path to load_td_config()."
        )

    with open(resolved, encoding="utf-8") as fh:
        config: dict = json.load(fh)

    # Validate presence and type of every required key.
    for key, expected_type in _REQUIRED_KEYS.items():
        if key not in config:
            raise ValueError(
                f"td_config.json is missing required key: '{key}'. "
                f"Expected a {expected_type.__name__} value."
            )
        if not isinstance(config[key], expected_type):
            raise ValueError(
                f"td_config.json key '{key}' must be {expected_type.__name__}, "
                f"but got {type(config[key]).__name__}."
            )

    return config


# ===========================================================================
# Rolling buffer
# ===========================================================================

class RollingBuffer:
    """Thread-safe circular buffer of time-stamped measurement dictionaries.

    The buffer holds at most ``window_minutes × samples_per_minute`` entries.
    When it is full and a new entry arrives, the oldest entry is automatically
    discarded (FIFO eviction via ``collections.deque(maxlen=...))``).

    Each entry has the form::

        {
            "timestamp": datetime,            # UTC recommended; any datetime accepted
            "values":    dict[str, Any]       # arbitrary channel name → value
        }

    Parameters
    ----------
    window_minutes:
        How many minutes of history to retain (M in the spec).
    samples_per_minute:
        How many samples are taken per minute (N in the spec).
        Combined with ``window_minutes`` this gives the buffer capacity
        ``max_len = M × N``.

    Thread safety
    -------------
    Every public method acquires the internal ``threading.Lock`` before
    touching the deque.  This allows a background sampling thread (Step 2)
    to call ``append()`` concurrently with the main thread calling
    ``snapshot()`` or ``get_channel()``.

    Example
    -------
    ::

        buf = RollingBuffer(window_minutes=10, samples_per_minute=12)
        # capacity = 120 entries (one every 5 seconds for 10 minutes)

        buf.append(datetime.utcnow(), {"weight_g": 4200.0})
        buf.append(datetime.utcnow(), {"weight_g": 7100.0, "ammonia_ppb": 18.0})

        buf.get_channel("weight_g")    # → [4200.0, 7100.0]
        buf.get_channel("ammonia_ppb") # → [None, 18.0]   (None for missing key)
        len(buf)                       # → 2
    """

    def __init__(self, window_minutes: int, samples_per_minute: int) -> None:
        if window_minutes <= 0:
            raise ValueError(f"window_minutes must be > 0, got {window_minutes}")
        if samples_per_minute <= 0:
            raise ValueError(f"samples_per_minute must be > 0, got {samples_per_minute}")

        self.window_minutes: int = window_minutes
        self.samples_per_minute: int = samples_per_minute
        self.max_len: int = window_minutes * samples_per_minute

        # The deque enforces the capacity automatically: appending to a full
        # deque discards the leftmost (oldest) element before inserting the new one.
        self._buf: deque[dict] = deque(maxlen=self.max_len)

        # Single lock shared by all public methods.
        self._lock: Lock = Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, timestamp: datetime, values: dict[str, Any]) -> None:
        """Add a new measurement to the buffer.

        If the buffer is already at capacity, the oldest entry is evicted
        before the new one is inserted.

        Parameters
        ----------
        timestamp:
            The UTC datetime at which the measurement was taken.
        values:
            A dict mapping channel names to their current readings.
            Values may be ``float``, ``int``, ``str``, or ``None``.
            Absent channels in ``values`` are recorded as missing (not zero).
            The dict is **copied** so that mutations by the caller after this
            call do not affect the stored entry.
        """
        entry = {"timestamp": timestamp, "values": dict(values)}
        with self._lock:
            self._buf.append(entry)

    # ------------------------------------------------------------------
    # Read — bulk
    # ------------------------------------------------------------------

    def snapshot(self) -> list[dict]:
        """Return a copy of the entire buffer, oldest entry first.

        Returns
        -------
        list[dict]
            Each element has the form ``{"timestamp": datetime, "values": dict}``.
            The returned list is an independent copy — mutating it does not
            affect the buffer.
        """
        with self._lock:
            return [
                {"timestamp": e["timestamp"], "values": dict(e["values"])}
                for e in self._buf
            ]

    def get_timestamps(self) -> list[datetime]:
        """Return the timestamp of every buffered entry, oldest first."""
        with self._lock:
            return [e["timestamp"] for e in self._buf]

    # ------------------------------------------------------------------
    # Read — single channel
    # ------------------------------------------------------------------

    def get_channel(self, name: str) -> list[Any]:
        """Return time-ordered values for one named channel.

        Parameters
        ----------
        name:
            The channel key, e.g. ``"weight_g"`` or ``"similarity_anna"``.

        Returns
        -------
        list
            One element per buffered entry, oldest first.
            Entries where ``name`` was not present in the values dict return
            ``None`` (not 0 or any sentinel numeric value).
        """
        with self._lock:
            return [e["values"].get(name) for e in self._buf]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def window_span_seconds(self) -> float:
        """Return the elapsed time between the oldest and newest buffered entry.

        Returns
        -------
        float
            Seconds.  Returns ``0.0`` if the buffer contains fewer than two
            entries.
        """
        with self._lock:
            if len(self._buf) < 2:
                return 0.0
            delta = self._buf[-1]["timestamp"] - self._buf[0]["timestamp"]
            return delta.total_seconds()

    def __len__(self) -> int:
        """Return the current number of entries in the buffer."""
        with self._lock:
            return len(self._buf)

    def is_full(self) -> bool:
        """Return ``True`` if the buffer has reached its maximum capacity."""
        with self._lock:
            return len(self._buf) == self.max_len

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all entries from the buffer.

        Used by tests and the simulator to reset state without creating a
        new buffer object.
        """
        with self._lock:
            self._buf.clear()

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    def to_dataframe(
        self,
        channel_prefix: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> "pandas.DataFrame":  # type: ignore[name-defined]  # noqa: F821
        """Convert buffer contents to a pandas DataFrame.

        This is the bridge between the raw buffer and Step 4's cat-
        identification logic.  The primary use case is building the M×K
        similarity DataFrame described in CLAUDE.md:

        * Call ``to_dataframe(channel_prefix="similarity_")`` on the buffer.
        * The returned DataFrame has one column per registered cat (prefix
          stripped), a datetime index, and ``NaN`` wherever a sample tick
          had no camera frame (i.e. the key was absent from that entry's
          values dict).

        Parameters
        ----------
        channel_prefix:
            If provided, only channels whose names start with this prefix are
            included.  The prefix is stripped from the column names in the
            returned DataFrame.  Pass ``None`` to include all channels.
        start:
            If provided, only entries with ``timestamp >= start`` are included.
        end:
            If provided, only entries with ``timestamp <= end`` are included.

        Returns
        -------
        pandas.DataFrame
            * Index: ``datetime`` timestamps (the ``timestamp`` field of each
              entry), dtype ``datetime64[us]`` or object depending on pandas
              version.
            * Columns: channel names (prefix stripped when ``channel_prefix``
              is given).
            * Values: floats, strings, or ``NaN`` (for absent keys).
            * Returns an empty DataFrame if the buffer is empty or no entries
              fall within the requested time range.

        Notes
        -----
        ``pandas`` is imported lazily inside this method so that the rest of
        ``time_buffer`` can be imported in environments where pandas is not
        installed (e.g. minimal test environments for other modules).

        Missing keys in a sample's values dict are represented as ``NaN``
        (not 0.0).  Callers must use ``skipna=True`` in aggregations to
        exclude these from means and other statistics.

        Example — similarity DataFrame
        --------------------------------
        ::

            buf.append(t0, {"similarity_anna": 0.91, "similarity_luna": 0.23})
            buf.append(t1, {})                          # no camera frame
            buf.append(t2, {"similarity_anna": 0.89, "similarity_luna": 0.25})

            df = buf.to_dataframe(channel_prefix="similarity_")
            # df:
            #             anna   luna
            # t0          0.91   0.23
            # t1           NaN    NaN
            # t2          0.89   0.25

            df.mean(skipna=True)
            # anna    0.90
            # luna    0.24
        """
        import pandas as pd  # lazy import

        snap = self.snapshot()  # thread-safe copy

        # Apply time range filter if requested.
        if start is not None:
            snap = [e for e in snap if e["timestamp"] >= start]
        if end is not None:
            snap = [e for e in snap if e["timestamp"] <= end]

        if not snap:
            return pd.DataFrame()

        records: list[dict] = []
        for entry in snap:
            row: dict[str, Any] = {}
            for key, val in entry["values"].items():
                if channel_prefix is None:
                    row[key] = val
                elif key.startswith(channel_prefix):
                    # Strip the prefix so column names are just the cat names.
                    row[key[len(channel_prefix):]] = val
            records.append(row)

        timestamps = [e["timestamp"] for e in snap]

        # Build DataFrame; missing keys in a row automatically become NaN.
        df = pd.DataFrame(records, index=timestamps)
        df.index.name = "timestamp"
        return df

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RollingBuffer("
            f"window={self.window_minutes}m, "
            f"rate={self.samples_per_minute}/min, "
            f"capacity={self.max_len}, "
            f"used={len(self)})"
        )
