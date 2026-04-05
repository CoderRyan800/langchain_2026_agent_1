"""
sensor_collector.py — Step 2 of the Time-Domain Measurement System
===================================================================

This module bridges the hardware abstraction layer (driver objects) and the
``RollingBuffer`` introduced in Step 1.  It is designed around two concerns:

1. **Driver interface** — every physical sensor (or mock substitute) exposes
   a single ``read()`` method.  The rest of the system never talks to hardware
   directly; it always goes through a driver.  Swapping real hardware for a
   mock during tests requires only passing a different driver object — zero
   other code changes.

2. **SensorCollector** — a background thread that calls every enabled driver
   at a fixed sample rate (from ``td_config.json``), assembles the per-tick
   measurement dict, and appends it to the shared ``RollingBuffer``.

Driver contract
---------------
Each driver must inherit from ``BaseDriver`` and implement ``read()``.

- Scalar drivers (weight, ammonia, methane, chip_id): ``read()`` returns a
  single float, string, or ``None``.

- ``SimilarityDriver``: ``read()`` returns a dict ``{cat_name: score}`` **or**
  ``None`` (when no camera frame is available this tick).  The collector
  expands the dict into per-cat buffer keys named ``similarity_<catname>``.
  A ``None`` return omits **all** cat keys from the buffer entry, which the
  downstream DataFrame machinery converts to ``NaN`` (not 0.0).

CLIP-only policy
----------------
``SimilarityDriver`` in production runs the local CLIP embedder.  No GPT-4o
calls are made during continuous monitoring — that would cost money and
introduce latency on every 5-second tick.  GPT-4o confirmation is reserved
for post-visit analysis in Step 4.

Thread safety
-------------
``SensorCollector`` launches a single daemon thread.  The thread is the only
writer to the buffer during normal operation; ``start()`` and ``stop()`` are
called from the main thread.  A ``threading.Event`` is used for a clean
shutdown: ``stop()`` sets the event and the daemon thread exits at the next
sleep wake-up.

Usage example
-------------
::

    from litterbox.time_buffer import RollingBuffer, load_td_config
    from litterbox.sensor_collector import (
        SensorCollector, WeightDriver, AmmoniaDriver,
        MethaneDriver, ChipIdDriver, SimilarityDriver,
    )

    cfg = load_td_config()
    buf = RollingBuffer(cfg["window_minutes"], cfg["samples_per_minute"])

    drivers = {
        "weight":     WeightDriver(base_value=5400.0, noise_sigma=20.0),
        "ammonia":    AmmoniaDriver(base_value=8.0, noise_sigma=1.0),
        "methane":    MethaneDriver(base_value=5.0, noise_sigma=0.8),
        "chip_id":    ChipIdDriver(cat_name=None),          # no chip reader
        "similarity": SimilarityDriver(cat_scores=None),    # no camera
    }

    collector = SensorCollector(config=cfg, drivers=drivers, buffer=buf)
    collector.start()
    # ... runs in background ...
    collector.stop()
"""

from __future__ import annotations

import random
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional, Union

from litterbox.time_buffer import RollingBuffer


# ===========================================================================
# Abstract driver base
# ===========================================================================

class BaseDriver(ABC):
    """Abstract base class for all hardware (and mock) sensor drivers.

    Every concrete driver — whether it talks to real hardware via I²C/serial
    or returns a pre-configured fake value for tests — must implement this
    single method.

    The return type varies by channel:

    * Scalar channels (weight, gas sensors): ``float`` or ``None``.
    * String channels (chip_id): ``str`` or ``None``.
    * Similarity channel: ``dict[str, float]`` or ``None``.

    ``None`` always means "reading not available this tick" and must be
    propagated as an absent key in the buffer entry, **not** as zero.
    """

    @abstractmethod
    def read(self) -> Union[float, str, dict, None]:
        """Return the current sensor reading, or ``None`` if unavailable."""


# ===========================================================================
# Concrete driver classes
# ===========================================================================

class WeightDriver(BaseDriver):
    """Driver for a weight scale (grams).

    Production: reads from a serial / I²C scale interface.

    This class acts as both a **mock** (used in tests and simulation) and
    a **template** for the production subclass.  Pass ``noise_sigma=0`` for a
    perfectly repeatable value; pass a positive sigma to add realistic
    Gaussian noise.

    Parameters
    ----------
    base_value:
        The nominal weight reading in grams (e.g. 5 400.0 for an empty box).
    noise_sigma:
        Standard deviation of Gaussian noise added to each reading.
        Default 0.0 (no noise).
    """

    def __init__(self, base_value: float = 0.0, noise_sigma: float = 0.0) -> None:
        self._base  = base_value
        self._sigma = noise_sigma

    def read(self) -> Optional[float]:
        """Return base_value + Gaussian(0, noise_sigma) grams."""
        value = self._base + (random.gauss(0.0, self._sigma) if self._sigma else 0.0)
        return round(value, 1)


class AmmoniaDriver(BaseDriver):
    """Driver for an ammonia (NH₃) gas sensor (parts-per-billion).

    Production: reads an ADC connected to an MQ-135 or ENS160 sensor.

    Parameters
    ----------
    base_value:
        Nominal ammonia concentration in ppb.
    noise_sigma:
        Standard deviation of Gaussian noise in ppb.
    """

    def __init__(self, base_value: float = 0.0, noise_sigma: float = 0.0) -> None:
        self._base  = base_value
        self._sigma = noise_sigma

    def read(self) -> Optional[float]:
        """Return base_value + noise, clamped to [0, ∞)."""
        value = self._base + (random.gauss(0.0, self._sigma) if self._sigma else 0.0)
        return round(max(0.0, value), 2)


class MethaneDriver(BaseDriver):
    """Driver for a methane (CH₄) gas sensor (parts-per-billion).

    Production: reads an ADC connected to an MQ-4 or MQ-9 sensor.

    Parameters
    ----------
    base_value:
        Nominal methane concentration in ppb.
    noise_sigma:
        Standard deviation of Gaussian noise in ppb.
    """

    def __init__(self, base_value: float = 0.0, noise_sigma: float = 0.0) -> None:
        self._base  = base_value
        self._sigma = noise_sigma

    def read(self) -> Optional[float]:
        """Return base_value + noise, clamped to [0, ∞)."""
        value = self._base + (random.gauss(0.0, self._sigma) if self._sigma else 0.0)
        return round(max(0.0, value), 2)


class ChipIdDriver(BaseDriver):
    """Driver for an RFID/NFC microchip reader.

    Production: reads the chip reader over serial or USB HID.  Returns the
    cat's registered chip ID string when a chip is in range, or ``None`` when
    no chip is detected.

    Parameters
    ----------
    cat_name:
        The chip ID string to return on every ``read()`` call.
        Pass ``None`` to simulate "no chip in range".
    """

    def __init__(self, cat_name: Optional[str] = None) -> None:
        self._cat_name = cat_name

    def read(self) -> Optional[str]:
        """Return the configured cat name, or None if no chip is present."""
        return self._cat_name


class SimilarityDriver(BaseDriver):
    """Driver for the CLIP-based camera similarity pipeline.

    This driver is unlike the others: its ``read()`` method returns a **dict**
    mapping each registered cat's name to its CLIP cosine similarity score
    against the current camera frame.

    Production behaviour
    --------------------
    On each call, the production implementation would:

    1. Capture a frame from the litter-box camera.
    2. Embed the frame with the local CLIP model (clip-ViT-B-32).
    3. Query the Chroma vector index for all registered cats.
    4. Return ``{cat_name: score, ...}`` for all registered cats.

    If the camera cannot produce a usable frame (lid closed, motion blur,
    etc.) the driver returns ``None`` — **not** a dict of zeros.  The
    collector treats a ``None`` return as "no camera frame this tick" and
    omits all similarity keys from the buffer entry, ensuring they appear as
    ``NaN`` in the downstream DataFrame.

    This class is used as a **mock** during tests and simulation.  Inject
    it with a pre-configured ``cat_scores`` dict (or ``None``) to control
    what the collector sees without any actual camera or CLIP model.

    Parameters
    ----------
    cat_scores:
        Either a dict ``{cat_name: score}`` to return on every ``read()``,
        or ``None`` to simulate a missing/unusable camera frame.
    """

    def __init__(self, cat_scores: Optional[dict[str, float]] = None) -> None:
        self._cat_scores = cat_scores

    def read(self) -> Optional[dict[str, float]]:
        """Return the configured cat scores dict, or None (missing frame)."""
        # Return a copy so callers cannot mutate the driver's internal state.
        return dict(self._cat_scores) if self._cat_scores is not None else None


# ===========================================================================
# Sensor collector
# ===========================================================================

class SensorCollector:
    """Drives enabled sensor channels at a fixed sample rate into a RollingBuffer.

    ``SensorCollector`` is a thin orchestration layer.  It reads the list of
    enabled channels from the config, calls the corresponding driver on each
    tick, assembles the per-tick measurement dict, and hands it to the
    ``RollingBuffer``.

    The only special handling is for the ``similarity`` channel type: the
    driver returns a dict ``{cat_name: score}`` (or ``None``).  The collector
    expands that dict into individual buffer keys named
    ``similarity_<catname>``, consistent with the buffer schema described in
    CLAUDE.md.

    Parameters
    ----------
    config:
        The parsed ``td_config.json`` dict (from ``load_td_config()``).
        ``config["channels"]`` defines which channels are enabled and in what
        order.  ``config["samples_per_minute"]`` sets the tick interval.
    drivers:
        A mapping from **channel type** (the ``"type"`` field in the channel
        config, e.g. ``"weight"``, ``"similarity"``) to a ``BaseDriver``
        instance.  Only drivers for enabled channels need to be present; the
        collector skips channels whose type is absent from this dict.
    buffer:
        The ``RollingBuffer`` to append measurements to.  Usually shared with
        the ``VisitTrigger`` (Step 3) so the trigger can read the most recent
        samples immediately after they are written.
    on_sample:
        Optional callable invoked after every ``_sample_once()`` with the
        assembled values dict as its sole argument.  Wire ``VisitTrigger.check``
        here to integrate the state machine with the collector::

            trigger   = VisitTrigger(config, buffer, on_visit_complete=…)
            collector = SensorCollector(config, drivers, buffer,
                                        on_sample=trigger.check)

        Defaults to ``None`` (no callback).

    Thread safety
    -------------
    ``start()`` / ``stop()`` must be called from the same thread (typically
    the main thread).  The background sampling thread is a daemon thread so it
    will not block process exit if ``stop()`` is never called.
    """

    def __init__(
        self,
        config: dict,
        drivers: dict[str, BaseDriver],
        buffer: RollingBuffer,
        on_sample: Optional[Any] = None,
    ) -> None:
        # Build the ordered list of enabled channels from the config.
        # Each element is a (name, type) tuple, e.g. ("weight_g", "weight").
        self._enabled_channels: list[tuple[str, str]] = [
            (ch["name"], ch["type"])
            for ch in config.get("channels", [])
            if ch.get("enabled", False)
        ]

        self._drivers   = drivers
        self._buffer    = buffer
        self._on_sample = on_sample   # optional post-tick callback

        # Tick interval in seconds — derived from config so tests can override
        # by passing a custom config with a higher samples_per_minute.
        samples_per_minute = config.get("samples_per_minute", 12)
        self._interval: float = 60.0 / samples_per_minute

        # Threading primitives — created here, not in start(), so that
        # _stop_event can be checked even before start() is called.
        self._stop_event: threading.Event  = threading.Event()
        self._thread:     Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Core sampling logic
    # ------------------------------------------------------------------

    def _sample_once(self) -> None:
        """Take one reading from every enabled, drivable channel.

        For each enabled channel:
        - Look up the driver by channel type.  If no driver is registered for
          this type, skip the channel silently (allows partial sensor configs).
        - Call ``driver.read()``.
        - For the ``similarity`` channel type: if the return value is a dict,
          expand it into ``similarity_<catname>`` keys.  If it is ``None``
          (no camera frame), skip entirely — the keys will be absent from the
          buffer entry and appear as ``NaN`` in the DataFrame.
        - For all other channel types: store the scalar value (or ``None``)
          directly under the channel name.

        The assembled values dict is appended to the buffer with the current
        UTC timestamp.
        """
        timestamp = datetime.now(timezone.utc)
        values: dict[str, Any] = {}

        for channel_name, channel_type in self._enabled_channels:
            driver = self._drivers.get(channel_type)
            if driver is None:
                # No driver registered for this channel type — skip quietly.
                continue

            reading = driver.read()

            if channel_type == "similarity":
                # SimilarityDriver returns dict or None.
                # Expand dict into per-cat keys; None → all keys absent.
                if isinstance(reading, dict):
                    for cat_name, score in reading.items():
                        values[f"similarity_{cat_name}"] = score
                # If reading is None, we intentionally add nothing to values.
            else:
                # Scalar / string channels: store under the channel's name.
                # None means "unavailable this tick" — kept as None so the
                # buffer records a genuine absence rather than a sentinel zero.
                values[channel_name] = reading

        self._buffer.append(timestamp, values)

        # Notify the optional post-tick callback (e.g. VisitTrigger.check).
        if self._on_sample is not None:
            self._on_sample(values)

    # ------------------------------------------------------------------
    # Background thread control
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background thread body — calls _sample_once() on each tick.

        Uses ``threading.Event.wait(timeout=interval)`` instead of
        ``time.sleep()`` so that ``stop()`` can wake the thread immediately
        rather than waiting for the current sleep to expire.
        """
        while not self._stop_event.is_set():
            self._sample_once()
            # Wait for the inter-sample interval OR until stop() is called.
            self._stop_event.wait(timeout=self._interval)

    def start(self) -> None:
        """Start the background sampling thread.

        Calling ``start()`` more than once without an intervening ``stop()``
        is a programming error and raises ``RuntimeError``.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(
                "SensorCollector is already running. "
                "Call stop() before calling start() again."
            )

        # Clear the stop event in case the collector was previously stopped
        # and is being restarted.
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run,
            name="SensorCollector",
            daemon=True,   # will not block process exit
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to exit and wait for it to finish.

        Returns as soon as the thread has exited (or immediately if no thread
        was ever started).  ``stop()`` is safe to call multiple times.
        """
        self._stop_event.set()

        if self._thread is not None:
            # 5-second join timeout — the thread should stop within one tick
            # interval because _run() uses Event.wait() rather than sleep().
            self._thread.join(timeout=5.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        channels = ", ".join(f"{n}({t})" for n, t in self._enabled_channels)
        running  = self._thread is not None and self._thread.is_alive()
        return (
            f"SensorCollector("
            f"channels=[{channels}], "
            f"interval={self._interval:.1f}s, "
            f"running={running})"
        )
