"""Generates simulated sensor readings for a single litter-box visit."""

import random
from dataclasses import dataclass
from typing import Optional

from sim_config import (
    BOX_BASELINE_G,
    WEIGHT_PRE_NOISE_G, WEIGHT_ENTRY_NOISE_G, WEIGHT_EXIT_NOISE_G,
    WASTE_MIN_G, WASTE_MAX_G,
    AMMONIA_NORMAL, AMMONIA_ANOMALY, AMMONIA_NULL_PROB,
    METHANE_NORMAL, METHANE_ANOMALY, METHANE_NULL_PROB,
)


@dataclass
class SensorReadings:
    weight_pre_g:     int
    weight_entry_g:   int
    weight_exit_g:    int
    waste_g_true:     int            # ground-truth deposit; not passed to agent
    ammonia_peak_ppb: Optional[float]
    methane_peak_ppb: Optional[float]


def generate_readings(
    true_weight_g: int,
    is_anomalous: bool,
    rng: random.Random,
) -> SensorReadings:
    """Return one complete set of sensor readings for a visit.

    All weight values are integers (grams).
    Gas values are rounded to one decimal place, or None on sensor dropout.
    """
    weight_pre   = round(BOX_BASELINE_G + rng.gauss(0, WEIGHT_PRE_NOISE_G))
    weight_entry = round(weight_pre + true_weight_g + rng.gauss(0, WEIGHT_ENTRY_NOISE_G))
    waste_true   = round(rng.uniform(WASTE_MIN_G, WASTE_MAX_G))
    weight_exit  = round(weight_pre + waste_true + rng.gauss(0, WEIGHT_EXIT_NOISE_G))

    # Ammonia
    if rng.random() < AMMONIA_NULL_PROB:
        ammonia = None
    elif is_anomalous:
        ammonia = round(rng.uniform(*AMMONIA_ANOMALY), 1)
    else:
        ammonia = round(rng.uniform(*AMMONIA_NORMAL), 1)

    # Methane
    if rng.random() < METHANE_NULL_PROB:
        methane = None
    elif is_anomalous:
        methane = round(rng.uniform(*METHANE_ANOMALY), 1)
    else:
        methane = round(rng.uniform(*METHANE_NORMAL), 1)

    return SensorReadings(
        weight_pre_g=weight_pre,
        weight_entry_g=weight_entry,
        weight_exit_g=weight_exit,
        waste_g_true=waste_true,
        ammonia_peak_ppb=ammonia,
        methane_peak_ppb=methane,
    )
