"""Generates a reproducible, time-ordered list of simulated visit events."""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sim_config import (
    CATS, VISIT_QUOTAS, TEST_DATA_DIR,
    SIM_START_DATE, SIM_DAYS, TOTAL_VISITS, NUM_ANOMALIES,
)


@dataclass
class SimEvent:
    event_index:    int
    simulated_time: str          # ISO-8601 timestamp
    cat_name:       str
    true_weight_g:  int
    entry_image:    str          # absolute path to cat photo
    is_anomalous:   bool
    visit_id:       Optional[int] = None   # filled in after agent call


def build_schedule(seed: int) -> list[SimEvent]:
    """Return a reproducible, time-ordered list of SimEvent objects.

    The cat visit pool is shuffled to distribute each cat's quota evenly.
    Time slots span morning, midday, afternoon, and evening across SIM_DAYS.
    Anomalous events are chosen randomly from the full pool.
    """
    rng = random.Random(seed)

    # Build and shuffle the visit pool (respects per-cat quotas)
    cat_pool: list[str] = []
    for cat, quota in VISIT_QUOTAS.items():
        cat_pool.extend([cat] * quota)
    rng.shuffle(cat_pool)

    # Generate candidate time slots: 4 blocks × SIM_DAYS days
    start = datetime.strptime(SIM_START_DATE, "%Y-%m-%d")
    time_blocks = [(7, 9), (11, 13), (15, 17), (19, 21)]
    candidates: list[datetime] = []
    for day in range(SIM_DAYS):
        date = start + timedelta(days=day)
        for hour_lo, hour_hi in time_blocks:
            hour   = rng.randint(hour_lo, hour_hi - 1)
            minute = rng.randint(0, 59)
            second = rng.randint(0, 59)
            candidates.append(date.replace(hour=hour, minute=minute, second=second))

    chosen_times = sorted(rng.sample(candidates, TOTAL_VISITS))

    # Choose which events are seeded as anomalous
    anomalous_indices = set(rng.sample(range(TOTAL_VISITS), NUM_ANOMALIES))

    # Build per-cat photo pools (all photos; reference photo at index 0 is included)
    photo_pools: dict[str, list[Path]] = {}
    for cat_name in CATS:
        cat_dir = TEST_DATA_DIR / cat_name
        photos = sorted(p for p in cat_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not photos:
            raise RuntimeError(f"No photos found for {cat_name} in {cat_dir}")
        photo_pools[cat_name] = photos

    # Cycling photo cursor per cat (allows reuse across visits)
    photo_cursors: dict[str, int] = {c: 0 for c in CATS}

    events: list[SimEvent] = []
    for i, (cat_name, sim_time) in enumerate(zip(cat_pool, chosen_times)):
        pool   = photo_pools[cat_name]
        photo  = pool[photo_cursors[cat_name] % len(pool)]
        photo_cursors[cat_name] += 1

        events.append(SimEvent(
            event_index=i + 1,
            simulated_time=sim_time.isoformat(),
            cat_name=cat_name,
            true_weight_g=CATS[cat_name]["true_weight_g"],
            entry_image=str(photo.resolve()),
            is_anomalous=(i in anomalous_indices),
        ))

    return events
