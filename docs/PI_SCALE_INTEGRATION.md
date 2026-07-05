# Raspberry Pi Scale Integration Guide

**Audience:** the Claude Code (or engineer) working *on the Raspberry Pi*, wiring
this monitor to a physical weight scale.

**Goal:** make the continuous monitor read the real scale attached to this Pi,
detect litter-box visits, and record them — starting with **the scale only**.
Other sensors (gas, chip reader, camera) come later; this guide is scoped to the
scale.

Everything downstream of the scale is already built and tested. Your job is to
implement **one function** and tune a few thresholds. This guide tells you
exactly what to do and how to prove it works.

---

## 0. The 60-second mental model

The system samples every sensor on a fixed tick into a rolling in-memory buffer,
watches that buffer for a weight rise-then-fall (a visit), and on each completed
visit identifies the cat and writes a row to the `td_visits` table:

```
your scale ──► WeightScaleDriver.read()  (grams)   ◄── YOU IMPLEMENT THIS
                        │
   SensorCollector samples it every 5 s ──► RollingBuffer
                        │
   VisitTrigger sees weight climb > entry delta, then fall < exit delta
                        │
   VisitAnalyser records the visit  ──►  td_visits table in data/litterbox.db
```

The seam you plug into is the **driver contract**: a class with a `read()`
method. The collector, trigger, analyser, DB, daemon, logging, and shutdown are
done. You do not need to touch them.

**Scale-only scope — read this now:** a scale can detect *that* a cat visited
and *how much* it/its waste weighed, but it **cannot tell which cat**. That needs
the chip reader or camera (added later). So in this phase every visit is recorded
with `id_method = "unknown"`. That is correct and expected. It's perfectly useful
for a **single-cat** household (everything belongs to that one cat). With multiple
cats, visits can't be attributed until you add a chip reader or camera. What you
*do* get from the scale alone: reliable visit detection, per-visit weight, and —
over weeks — the long-term **weight trend** detector, which is the real early
warning for a sick cat losing weight. This whole path is **offline**: no OpenAI /
network calls happen for a scale-only visit.

---

## 1. The happy path (do these in order)

```bash
# 0. You are in the cloned repo on the Pi, in a Python 3.11+ venv/conda env.
pip install -e .                       # installs the `litterbox-monitor` command

# 1. Prove the pipeline works with NO hardware (must print "SELF-TEST PASS"):
python -m litterbox.daemon --self-test

# 2. Watch the live loop with a MOCK scale (Ctrl-C to stop). You'll see it
#    detect a visit every ~2 minutes:
python -m litterbox.daemon --simulate --no-retention

# 3. Now implement the real driver — edit build_scale_driver() in
#    src/litterbox/daemon.py  (see §2 and §3 below).

# 4. Calibrate tare + scale factor (see §4).

# 5. Tune the entry/exit weight deltas for your cat (see §5).

# 6. Run against the real scale:
python -m litterbox.daemon --no-retention
#    Put a known weight on the scale, lift it off, and watch the log say
#    "Visit recorded".  (§6 — acceptance test.)

# 7. Install as a boot service so it runs unattended (§7).
```

If step 1 fails, stop — the environment isn't set up right; don't touch the
driver yet.

---

## 2. The one function you implement

Open [`src/litterbox/daemon.py`](../src/litterbox/daemon.py) and find
`build_scale_driver()`. Replace its `raise NotImplementedError(...)` body so it
returns a driver for your scale.

A driver is any subclass of `BaseDriver` (from
[`src/litterbox/sensor_collector.py`](../src/litterbox/sensor_collector.py)) that
implements `read()`.

### The `read()` contract — obey all five rules

1. **Return grams as a `float`.** The value is the *total* weight currently on
   the scale: box + litter + (cat, if present). Not kilograms, not raw ADC
   counts — grams.
2. **Return `None`, never `0.0`, on a failed/unavailable read.** `None` means
   "unknown this tick" and is handled correctly downstream. `0.0` means "the box
   weighs nothing", which would poison the rolling baseline and break visit
   detection. This is the single most important rule.
3. **Be fast and non-blocking.** `read()` is called on the collector's
   background thread every `60 / samples_per_minute` seconds (5 s by default).
   It must return in well under that. If your scale library blocks or is slow,
   run it in your *own* thread that updates a cached value and have `read()`
   return the cache.
4. **Be thread-safe.** `read()` runs on the sampler thread, not the main thread.
   If it touches GPIO/serial state shared with other code, guard it.
5. **Return the current reading, not an average.** The trigger and analyser do
   their own aggregation. Light smoothing to reject obvious spikes is fine;
   don't hide the entry/exit transitions.

---

## 3. Worked example — HX711 load cell over GPIO

This is a *template*. Use whatever library matches your actual scale hardware;
the only thing that matters is that `read()` honours the contract above.

```python
# At the top of src/litterbox/daemon.py, near the other imports.
# Guard the hardware import so the module still imports on a dev machine
# (and so --self-test / --simulate keep working without the library installed).
try:
    from hx711 import HX711          # e.g. `pip install hx711` — your lib may differ
    import RPi.GPIO as GPIO
except ImportError:                  # not on a Pi / lib not installed
    HX711 = None


class Hx711ScaleDriver(BaseDriver):
    """Reads a load cell via an HX711 amplifier on the Pi's GPIO pins."""

    def __init__(self, dout_pin: int, sck_pin: int, tare_offset: float,
                 scale_factor: float) -> None:
        self._hx = HX711(dout_pin=dout_pin, pd_sck_pin=sck_pin)
        self._tare_offset = tare_offset      # raw counts with an empty box (see §4)
        self._scale_factor = scale_factor    # raw counts per gram        (see §4)

    def read(self):
        try:
            raw = self._hx.get_raw_data(times=3)   # small median, fast
            if not raw:
                return None                        # rule 2: unknown, NOT 0.0
            median_raw = sorted(raw)[len(raw) // 2]
            grams = (median_raw - self._tare_offset) / self._scale_factor
            return round(float(grams), 1)
        except Exception:
            return None                            # rule 2: never crash the tick


def build_scale_driver(config: dict) -> BaseDriver:
    if HX711 is None:
        raise RuntimeError(
            "HX711 library / RPi.GPIO not available. Install them, or run with "
            "--simulate to test without hardware."
        )
    # TODO: put YOUR calibration constants here (from §4) and your wiring pins.
    return Hx711ScaleDriver(
        dout_pin=5, sck_pin=6,
        tare_offset=YOUR_TARE_OFFSET,
        scale_factor=YOUR_SCALE_FACTOR,
    )
```

Also add your hardware libraries to the `pi` extras in
[`setup.py`](../setup.py) (currently an empty placeholder) so the install is
reproducible:

```python
"pi": ["hx711", "RPi.GPIO"],   # match your actual hardware
```

---

## 4. Calibration (tare + scale factor)

`grams = (raw_counts - tare_offset) / scale_factor`

1. **Tare** — with the empty box (+ litter) on the scale, record the raw reading.
   That's `tare_offset`.
2. **Scale factor** — place a *known* weight (a labelled dumbbell, a bag of
   flour, a calibration mass) on the scale. Record the raw reading.
   `scale_factor = (raw_with_weight - tare_offset) / known_grams`.
3. Verify: put a different known weight on and confirm `read()` returns within a
   few grams. The default trigger works comfortably with ±20 g noise; if you're
   noisier than that, increase `samples_per_minute` or smooth lightly.

Keep the litter *in* the box during tare — the baseline is "box as it normally
sits", and the trigger tracks a rolling median so slow litter changes are fine.

---

## 5. Tune the trigger for your cat

Edit
[`src/litterbox/td_config.weight_only.json`](../src/litterbox/td_config.weight_only.json)
→ the `trigger` block:

| Key | Meaning | How to set it |
|---|---|---|
| `weight_entry_delta_g` | Weight rise above the empty-box baseline to declare a cat present. | **Below your lightest cat's body weight.** A 4 kg cat clears the 300 g default easily; a 2 kg kitten also clears 300 g. If you have a very small cat, lower it — but keep it well above your scale's noise so litter shuffling doesn't false-trigger. |
| `weight_exit_delta_g` | Weight must fall back below `baseline + this` to declare the cat gone. | Keep it **smaller** than the entry delta (default 200 < 300). The gap is a hysteresis band that stops flicker at the threshold. |
| `window_minutes` / `samples_per_minute` | Buffer length and sample rate (top of the file). | Defaults (10 min, 12/min = every 5 s) are fine for a scale. Faster sampling = crisper entry/exit timing at more CPU. |

The `chip_*` and `similarity_*` thresholds are inert while those channels are
disabled — ignore them for now.

---

## 6. Acceptance test — how you know it works

**Hardware-free (run this first, and in CI):**

```bash
python -m litterbox.daemon --self-test
# Expected:
#   SELF-TEST PASS: exactly one visit detected and saved.
#   id_method = unknown   (expected 'unknown' for scale-only)
```

This drives the *real* trigger and analyser with a scripted weight ramp and
asserts exactly one visit is detected and persisted. If it fails after you edit
the daemon, you broke the wiring — revert and re-check.

**With the real scale:**

```bash
python -m litterbox.daemon --no-retention --log-level INFO
```

Then physically: put a weight ≥ your entry delta on the box, wait ~15 s, remove
it. Within a tick or two you should see:

```
INFO litterbox.daemon: Visit recorded — visit td:N | unknown cat | 12s | id_method=unknown
```

Confirm it landed in the DB:

```bash
python -c "import sqlite3; c=sqlite3.connect('data/litterbox.db'); \
print(c.execute('SELECT td_visit_id, entry_time, exit_time, id_method FROM td_visits ORDER BY td_visit_id DESC LIMIT 3').fetchall())"
```

You can also open the interactive agent (`litterbox-agent`) and ask it to show
recent/unconfirmed visits.

---

## 7. Run it as a service (starts on boot, restarts on crash)

A ready-to-edit unit is at
[`deploy/litterbox-monitor.service`](../deploy/litterbox-monitor.service). Edit
the `User`, `WorkingDirectory`, `ExecStart` (absolute path to the
`litterbox-monitor` console script in your venv), and `EnvironmentFile`, then:

```bash
sudo cp deploy/litterbox-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now litterbox-monitor
journalctl -u litterbox-monitor -f          # follow the logs
```

The daemon handles `SIGTERM` cleanly, so `systemctl stop` shuts it down
gracefully. `Restart=on-failure` makes it self-heal after a crash or a
power-blip reboot. Cap the journal size (`SystemMaxUse=` in
`/etc/systemd/journald.conf`) so logs can't fill the SD card over months.

---

## 8. Invariants — do NOT break these

The downstream analytics depend on these. If you change them, you'll get subtle
wrong results, not crashes:

- `read()` returns **grams**, or **`None`** for unavailable — never `0.0`.
- `read()` **does not block** the tick and **does not** make network/GPT calls.
  Continuous monitoring is local-only and free by design.
- Don't rename the `weight` channel `type` or the `weight_g` channel `name` in
  the config — the trigger and analyser look for `weight_g` specifically.
- Keep `--no-retention` on until a camera exists. The retention sweep
  permanently deletes image directories; with no camera there's nothing to
  prune and only risk.

---

## 9. What is still NOT built (so you're not surprised)

The scale-only monitor records visits and weights. It does **not** yet:

- **Notify you.** A detected weight anomaly lands in the DB only — there is no
  email/SMS/push yet. If you want alerts on the Pi, that's a separate piece to
  add (an opt-in notifier that fires on the trend detector's `significant`
  tier). Until then, you must query the DB / run the reports to see findings.
- **Identify cats** (needs chip reader or camera — see §0).
- **Pin dependencies.** If `pip install` pulls an ARM wheel that lags, pin the
  offending package. `torch` is only needed once you add the camera (CLIP); a
  pure scale monitor doesn't import it at runtime.

---

## 10. Adding the next sensor later

When the gas sensor / chip reader / camera arrives, the pattern is the same:

1. In `td_config.weight_only.json`, flip that channel's `"enabled"` to `true`.
2. In `build_drivers()` in `daemon.py`, add a branch that constructs its driver
   (an `AmmoniaDriver`/`MethaneDriver`/`ChipIdDriver`/`SimilarityDriver`
   subclass reading real hardware). The collector, trigger, and analyser already
   understand all these channels — a chip reader immediately upgrades visits
   from `unknown` to identified.

Nothing about the scale integration needs to change when you do this.
