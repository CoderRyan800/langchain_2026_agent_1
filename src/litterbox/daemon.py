"""
daemon.py — Continuous time-domain monitor runtime (Mode B)
============================================================

This is the long-running process that turns the time-domain building blocks
(Steps 1–5) into an *unattended monitor*.  It wires the four pieces that
already exist but were never composed into a runtime:

    load_td_config()                    # read td_config.json
        │
        ▼
    RollingBuffer  ◄───────────────┐    # Step 1: rolling window of samples
        ▲                          │
        │ append(ts, values)       │ snapshot()
        │                          │
    SensorCollector ──on_sample──► VisitTrigger ──on_visit_complete──► VisitAnalyser
      (Step 2)                       (Step 3)                            (Step 4)
      drives drivers                 detects entry/exit                  identifies cat,
      every 60/N seconds             from the buffer                     writes td_visits

The daemon owns the process lifecycle: it builds the drivers, starts the
background sampling thread, handles SIGINT/SIGTERM for a clean shutdown, emits
a periodic heartbeat, and runs the daily image-retention sweep.

────────────────────────────────────────────────────────────────────────────
INTEGRATION POINT — this is the ONE function your on-device code must fill in:

    build_scale_driver(config)   (see below)

Everything else is wired for you.  Replace the stub with a BaseDriver whose
``read()`` returns the current total box weight in GRAMS.  See
``docs/PI_SCALE_INTEGRATION.md`` for the full contract and a worked HX711
example.
────────────────────────────────────────────────────────────────────────────

Run modes
---------
    python -m litterbox.daemon --self-test
        Deterministic, hardware-free acceptance test: feeds a scripted weight
        ramp through the real trigger + analyser and asserts exactly one visit
        is detected and saved.  Uses a throwaway temp DB.  Exit code 0 = pass.

    python -m litterbox.daemon --simulate
        Runs the real-time loop with a mock scale that stages a visit every
        couple of minutes.  Proves the live pipeline without any hardware.
        Ctrl-C to stop.

    python -m litterbox.daemon
        Production: builds the real scale driver (build_scale_driver) and runs
        forever.  Fails fast with a helpful message until the driver is
        implemented.

    python -m litterbox.daemon --config /path/to/td_config.weight_only.json
        Same, with an explicit config file (the shipped scale-only preset).
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import signal
import sys
import tempfile
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from litterbox.db import PROJECT_ROOT, init_db
from litterbox.image_retention import sweep_old_visit_images
from litterbox.sensor_collector import BaseDriver, SensorCollector, WeightDriver
from litterbox.time_buffer import RollingBuffer, load_td_config
from litterbox.visit_analyser import VisitAnalyser
from litterbox.visit_trigger import VisitTrigger

log = logging.getLogger("litterbox.daemon")

# The scale-only configuration preset that ships next to this module.  Used by
# default so gas/chip/camera channels are disabled and the trigger runs on the
# weight channel alone.  Override with --config.
WEIGHT_ONLY_CONFIG = Path(__file__).with_name("td_config.weight_only.json")

# How often the main thread wakes to emit a heartbeat and run the daily sweep.
HEARTBEAT_SECONDS = 60.0


# ===========================================================================
# >>> INTEGRATION POINT — implement this for your real scale. <<<
# ===========================================================================

def build_scale_driver(config: dict) -> BaseDriver:
    """Return a driver that reads your physical weight scale.

    THIS IS THE ONE FUNCTION YOU MUST IMPLEMENT to connect real hardware.

    Contract (see docs/PI_SCALE_INTEGRATION.md for the full version):

    * Return a ``BaseDriver`` subclass instance.
    * Its ``read()`` must return the **current total weight on the scale in
      grams** as a ``float`` — that is box + litter + (cat, if present).
    * Return ``None`` (never 0.0, never a stale value) if the scale could not
      be read this tick.  ``None`` means "unknown"; 0.0 means "the box weighs
      nothing", which would corrupt the baseline.
    * ``read()`` is called from the collector's background thread once every
      ``60 / samples_per_minute`` seconds (5 s at the default rate).  It must
      not block for anywhere near that long.  If your scale library blocks,
      cache the latest reading in a separate thread and return the cache here.

    Worked example (HX711 load cell over GPIO) is in the integration guide.
    """
    raise NotImplementedError(
        "No real scale driver is configured yet.\n\n"
        "Edit build_scale_driver() in src/litterbox/daemon.py to return a "
        "BaseDriver that reads your physical scale (grams), then run again.\n"
        "To smoke-test the whole pipeline without any hardware, run:\n"
        "    python -m litterbox.daemon --self-test     # instant, deterministic\n"
        "    python -m litterbox.daemon --simulate      # real-time mock scale\n\n"
        "Full contract + a worked HX711 example: docs/PI_SCALE_INTEGRATION.md"
    )


# ===========================================================================
# Mock scale for --simulate (no hardware needed)
# ===========================================================================

class ScriptedScaleDriver(BaseDriver):
    """Mock scale used by ``--simulate``.

    Sits at an empty-box baseline and stages a cat visit on a fixed cadence so
    the live loop demonstrably detects visits without any hardware attached.
    Deterministic in shape (a visit every ``period`` reads) but with Gaussian
    measurement noise so the readings look realistic.
    """

    def __init__(
        self,
        baseline_g: float = 4500.0,
        cat_g: float = 700.0,
        noise_sigma: float = 12.0,
        period: int = 24,     # reads per cycle (24 * 5 s = one visit every 2 min)
        visit_len: int = 6,   # reads the "cat" is present (6 * 5 s = 30 s)
    ) -> None:
        self._baseline = baseline_g
        self._cat = cat_g
        self._sigma = noise_sigma
        self._period = period
        self._visit_len = visit_len
        self._i = 0

    def read(self) -> Optional[float]:
        phase = self._i % self._period
        self._i += 1
        present = phase < self._visit_len
        value = self._baseline + (self._cat if present else 0.0)
        return round(value + random.gauss(0.0, self._sigma), 1)


# ===========================================================================
# Config + driver assembly
# ===========================================================================

def _load_config(config_path: Optional[str]) -> dict:
    """Load the daemon config: explicit --config, else the weight-only preset,
    else the packaged default td_config.json."""
    if config_path:
        return load_td_config(config_path)
    if WEIGHT_ONLY_CONFIG.exists():
        return load_td_config(WEIGHT_ONLY_CONFIG)
    return load_td_config()


def build_drivers(config: dict, *, simulate: bool) -> dict[str, BaseDriver]:
    """Build one BaseDriver per *enabled* channel type in the config.

    Today only the ``weight`` channel is wired to real hardware (via
    ``build_scale_driver``).  Any other enabled channel is logged as a TODO and
    skipped — the collector tolerates missing drivers, and the trigger/analyser
    tolerate absent channels, so the monitor still runs weight-only.  When a
    later sensor (gas, chip, camera) is added, wire its driver here.
    """
    enabled_types = [
        ch["type"] for ch in config.get("channels", []) if ch.get("enabled", False)
    ]

    drivers: dict[str, BaseDriver] = {}
    for ch_type in enabled_types:
        if ch_type == "weight":
            drivers["weight"] = (
                ScriptedScaleDriver() if simulate else build_scale_driver(config)
            )
        else:
            log.warning(
                "Channel type %r is enabled in the config but has no driver "
                "yet — skipping. Add one in build_drivers() when the sensor is "
                "installed.",
                ch_type,
            )
    if "weight" not in drivers:
        raise RuntimeError(
            "No 'weight' channel is enabled in the config. The scale-only "
            "monitor needs the weight channel. Check your td_config."
        )
    return drivers


# ===========================================================================
# Visit handling
# ===========================================================================

def _visit_summary(record, entry_time: datetime, exit_time: datetime) -> str:
    duration = (exit_time - entry_time).total_seconds()
    who = "unknown cat"
    if record.confirmed_cat_id is not None:
        who = f"cat #{record.confirmed_cat_id} (confirmed via {record.id_method})"
    elif record.tentative_cat_id is not None:
        who = f"cat #{record.tentative_cat_id} (tentative via {record.id_method})"
    return (
        f"visit td:{record.td_visit_id} | {who} | "
        f"{duration:.0f}s | id_method={record.id_method}"
    )


def make_visit_handler(analyser: VisitAnalyser):
    """Build the ``on_visit_complete`` callback.

    IMPORTANT: this callback runs on the SensorCollector's background thread.
    An exception here would kill the sampling thread silently, so the body is
    wrapped in a catch-all — a single bad visit must never take the monitor
    down.
    """

    def _handle_visit(snapshot: list[dict], entry_time: datetime, exit_time: datetime) -> None:
        try:
            record = analyser.analyse(snapshot, entry_time, exit_time)
            analyser.save(record)
            log.info("Visit recorded — %s", _visit_summary(record, entry_time, exit_time))
        except Exception:  # noqa: BLE001 — must not propagate onto the sampler thread
            log.exception("Visit analysis/save failed; monitor continues")

    return _handle_visit


# ===========================================================================
# Main run loop
# ===========================================================================

def run(
    config_path: Optional[str] = None,
    *,
    simulate: bool = False,
    images_dir: Optional[str] = None,
    retention_enabled: bool = True,
) -> int:
    """Start the monitor and block until SIGINT/SIGTERM.  Returns an exit code.

    Parameters
    ----------
    images_dir:
        Root of the visit-image store the daily retention sweep operates on.
        Defaults to ``<repo>/images``.  Set this (or pass ``--images-dir``) to
        point the sweep at wherever your camera actually writes frames — and to
        keep tests/dev off the real directory.
    retention_enabled:
        When ``False`` (``--no-retention``), the daily deletion sweep is
        skipped entirely.  The sweep permanently deletes image directories, so
        it is worth disabling until a camera is actually installed.
    """
    config = _load_config(config_path)
    init_db()

    buffer = RollingBuffer(config["window_minutes"], config["samples_per_minute"])
    analyser = VisitAnalyser(config)
    trigger = VisitTrigger(config, buffer, on_visit_complete=make_visit_handler(analyser))
    drivers = build_drivers(config, simulate=simulate)
    collector = SensorCollector(config, drivers, buffer, on_sample=trigger.check)

    images_base = Path(images_dir) if images_dir else PROJECT_ROOT / "images"
    retention_days = int(config.get("image_retention_days", 7))

    stop = threading.Event()

    def _signal(signum, _frame):
        log.info("Received signal %s — shutting down.", signum)
        stop.set()

    signal.signal(signal.SIGINT, _signal)
    signal.signal(signal.SIGTERM, _signal)

    log.info(
        "Starting monitor%s — %d channel(s), %.1fs tick, %d-minute window.",
        " (SIMULATED scale)" if simulate else "",
        len(drivers),
        60.0 / config["samples_per_minute"],
        config["window_minutes"],
    )
    collector.start()

    last_sweep_day: Optional[date] = None
    try:
        while not stop.is_set():
            stop.wait(timeout=HEARTBEAT_SECONDS)
            if stop.is_set():
                break

            # Heartbeat — proof the sampler is alive (monitoring the monitor).
            timestamps = buffer.get_timestamps()
            if timestamps:
                age = (datetime.now(timezone.utc) - timestamps[-1]).total_seconds()
                log.info(
                    "heartbeat: buffer=%d samples, last sample %.0fs ago, state=%s",
                    len(timestamps), age, trigger.state,
                )
            else:
                log.warning("heartbeat: buffer is EMPTY — is the scale returning data?")

            # Daily image-retention sweep. DESTRUCTIVE: permanently deletes
            # visit-image directories older than the retention window. Disabled
            # with --no-retention (recommended until a camera is installed,
            # since there are no frames to prune and nothing to accidentally
            # delete). Operates only on `images_base`, never the DB.
            today = date.today()
            if retention_enabled and today != last_sweep_day:
                try:
                    removed = sweep_old_visit_images(images_base, retention_days)
                    if removed:
                        log.info(
                            "Retention sweep deleted %d visit-image dir(s) older "
                            "than %d days under %s.",
                            removed, retention_days, images_base,
                        )
                except Exception:  # noqa: BLE001
                    log.exception("Image retention sweep failed; monitor continues")
                last_sweep_day = today
    finally:
        collector.stop()
        log.info("Monitor stopped.")
    return 0


# ===========================================================================
# Self-test — deterministic acceptance test, no hardware, isolated temp DB
# ===========================================================================

def self_test() -> int:
    """Feed a scripted weight ramp through the REAL trigger + analyser and
    assert exactly one visit is detected and persisted.

    This is the acceptance test referenced in docs/PI_SCALE_INTEGRATION.md:
    "a synthetic weight ramp fires exactly one visit."  It exercises the same
    VisitTrigger and VisitAnalyser the live daemon uses, so a green self-test
    means the pipeline is wired correctly end to end.  Writes only to a
    throwaway temp DB.  Returns 0 on pass, 1 on failure.
    """
    from litterbox import db as _db  # patched below for DB isolation

    tmp_dir = tempfile.mkdtemp(prefix="litterbox_selftest_")
    original_db_path = _db.DB_PATH
    _db.DB_PATH = Path(tmp_dir) / "selftest.db"
    try:
        config = _load_config(None)
        buffer = RollingBuffer(config["window_minutes"], config["samples_per_minute"])
        analyser = VisitAnalyser(config)

        saved: list = []

        def _on_visit(snapshot, entry_time, exit_time):
            record = analyser.analyse(snapshot, entry_time, exit_time)
            analyser.save(record)
            saved.append(record)

        trigger = VisitTrigger(config, buffer, on_visit_complete=_on_visit)

        # 15 empty-box samples (baseline), 6 samples with a ~700 g cat present
        # (well above the 300 g entry delta), then back down (below the 200 g
        # exit delta) — one clean rise-then-fall.
        baseline_g = 4500.0
        sequence = [baseline_g] * 15 + [baseline_g + 700.0] * 6 + [baseline_g + 50.0] * 4

        base_ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        for i, weight in enumerate(sequence):
            ts = base_ts + timedelta(seconds=5 * i)
            values = {"weight_g": round(weight, 1)}
            buffer.append(ts, values)
            trigger.check(values, timestamp=ts)

        n = len(saved)
        if n == 1:
            rec = saved[0]
            print("SELF-TEST PASS: exactly one visit detected and saved.")
            print(f"  td_visit_id = {rec.td_visit_id}")
            print(f"  id_method   = {rec.id_method}   (expected 'unknown' for scale-only)")
            print(f"  entry→exit  = {(rec.exit_time - rec.entry_time).total_seconds():.0f}s")
            return 0

        print(f"SELF-TEST FAIL: expected exactly 1 visit, detected {n}.", file=sys.stderr)
        return 1
    finally:
        _db.DB_PATH = original_db_path
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ===========================================================================
# CLI
# ===========================================================================

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="litterbox-monitor",
        description="Continuous time-domain litter-box monitor (Mode B).",
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help="Path to a td_config JSON file (default: the shipped weight-only preset).",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run the live loop with a mock scale (no hardware). Ctrl-C to stop.",
    )
    parser.add_argument(
        "--images-dir", metavar="PATH",
        help="Root of the visit-image store for the retention sweep "
             "(default: <repo>/images). Point this at your camera's output dir.",
    )
    parser.add_argument(
        "--no-retention", action="store_true",
        help="Disable the daily image-retention sweep entirely. Recommended "
             "until a camera is installed (nothing to prune, nothing to delete).",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run the deterministic visit-detection acceptance test and exit.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.self_test:
        return self_test()

    try:
        return run(
            config_path=args.config,
            simulate=args.simulate,
            images_dir=args.images_dir,
            retention_enabled=not args.no_retention,
        )
    except NotImplementedError as exc:
        # The scale driver stub hasn't been implemented yet — print the
        # guidance cleanly rather than dumping a traceback.
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        # Config / driver-assembly problems — actionable, not a crash to debug.
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
