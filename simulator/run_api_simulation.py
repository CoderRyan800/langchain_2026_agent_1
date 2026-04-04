#!/usr/bin/env python3
"""
API-based litter box simulation — 5 cats, 10 visits each.

Uses the LitterboxAgent Python API exclusively (no direct tool imports,
no CLI invocations).  Clears the simulation database before starting so
every record in the output reflects this run.

Cat roster and body masses:
    Anna     — 3.2 kg  registration: images/cats/anna/001_c25e6c.jpg
    Marina   — 4.0 kg  registration: images/cats/marina/001_aad380.jpg
    Luna     — 5.5 kg  registration: images/cats/luna/001_a150db.jpg
    Natasha  — 5.0 kg  registration: images/cats/natasha/001_4c58dd.jpg
    Whiskers — 6.0 kg  registration: images/cats/whiskers/001_b0c0d4.jpg

Visit images for each cat are cycled from their simulator/cat_pictures/<Cat>/
directories (Whiskers: images/cats/whiskers/).  The first image listed above
is the enrollment reference; visit captures are drawn from the same pool
(index 0 through N-1 modulo pool size).

Sensor model:
    weight_pre   = BOX_BASELINE (2000 g) + Gaussian(0, 20 g)
    weight_entry = weight_pre + body_mass + Gaussian(0, 30 g)
    waste        ~ Uniform(30, 120 g)
    weight_exit  = weight_pre + waste + Gaussian(0, 20 g)
    ammonia      ~ Uniform(5, 60 ppb)  [anomaly: 150-350 ppb]
    methane      ~ Uniform(0, 40 ppb)  [anomaly: 80-180 ppb]

Three visits are seeded as anomalous:
    Anna    visit 8  — elevated methane (digestive concern)
    Luna    visit 3  — elevated ammonia + methane (combined)
    Whiskers visit 5 — elevated ammonia (kidney / urinary concern)
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

# ── project path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from litterbox import LitterboxAgent  # noqa: E402  (after sys.path fix)

# ── simulation directories ────────────────────────────────────────────────────
SIM_DATA_DIR   = PROJECT_ROOT / "simulator" / "api_sim_data"
SIM_IMAGES_DIR = PROJECT_ROOT / "simulator" / "api_sim_images"

# ── cat configuration ─────────────────────────────────────────────────────────
#
# registration_image : absolute path to the reference photo used for enrollment.
#   This is the FIRST (lexicographically sorted) image in each cat's folder.
#
# visit_images       : pool of photos cycled for simulated visit captures.
#   For Anna/Marina/Luna/Natasha these come from simulator/cat_pictures/<Cat>/.
#   For Whiskers (no simulator directory) all three images/cats/whiskers/ photos
#   are used (including the registration photo, which the ID pipeline handles fine).
#
CATS: Dict[str, dict] = {
    "Anna": {
        "body_mass_g":        3200,
        "registration_image": PROJECT_ROOT / "images/cats/anna/001_c25e6c.jpg",
        "visit_images":       sorted(
            (PROJECT_ROOT / "simulator/cat_pictures/Anna").glob("*.jpeg")
        ),
    },
    "Marina": {
        "body_mass_g":        4000,
        "registration_image": PROJECT_ROOT / "images/cats/marina/001_aad380.jpg",
        "visit_images":       sorted(
            (PROJECT_ROOT / "simulator/cat_pictures/Marina").glob("*.jpeg")
        ),
    },
    "Luna": {
        "body_mass_g":        5500,
        "registration_image": PROJECT_ROOT / "images/cats/luna/001_a150db.jpg",
        "visit_images":       sorted(
            (PROJECT_ROOT / "simulator/cat_pictures/Luna").glob("*.jpeg")
        ),
    },
    "Natasha": {
        "body_mass_g":        5000,
        "registration_image": PROJECT_ROOT / "images/cats/natasha/001_4c58dd.jpg",
        "visit_images":       sorted(
            (PROJECT_ROOT / "simulator/cat_pictures/Natasha").glob("*.jpeg")
        ),
    },
    "Whiskers": {
        "body_mass_g":        6000,
        "registration_image": PROJECT_ROOT / "images/cats/whiskers/001_b0c0d4.jpg",
        "visit_images":       sorted(
            (PROJECT_ROOT / "images/cats/whiskers").glob("*.jpg")
        ),
    },
}

VISITS_PER_CAT = 10

# Weight sensor — mild Gaussian noise
BOX_BASELINE_G    = 2000   # tray + litter baseline (g)
WEIGHT_PRE_STD    = 20     # noise on pre-entry reading
WEIGHT_ENTRY_STD  = 30     # noise on cat+box reading
WEIGHT_EXIT_STD   = 20     # noise on post-exit reading
WASTE_MIN_G       = 30
WASTE_MAX_G       = 120

# Gas sensor — normal and anomalous ranges (ppb)
AMMONIA_NORMAL  = (5.0,   60.0)
AMMONIA_ANOMALY = (150.0, 350.0)
METHANE_NORMAL  = (0.0,   40.0)
METHANE_ANOMALY = (80.0,  180.0)

# Seeded anomalies: (cat_name, 0-based visit index within that cat's sequence)
ANOMALOUS_VISITS = {
    ("Anna",     7),   # visit 8  — digestive concern (elevated methane)
    ("Luna",     2),   # visit 3  — urinary + digestive (elevated NH₃ + CH₄)
    ("Whiskers", 4),   # visit 5  — kidney / urinary (elevated ammonia)
}

RANDOM_SEED = 42

# Exit image — placeholder litter box photo used for all exit captures
EXIT_IMAGE = PROJECT_ROOT / "simulator/assets/used_box.jpg"

# Output files
GROUND_TRUTH_PATH = PROJECT_ROOT / "simulator" / "api_sim_ground_truth.json"
REPORT_PATH       = PROJECT_ROOT / "simulator" / "api_simulation_report.md"


# ── sensor model ──────────────────────────────────────────────────────────────

def generate_sensor_readings(
    body_mass_g: int,
    is_anomalous: bool,
    rng: random.Random,
) -> dict:
    """Return Gaussian-noised sensor readings for one visit."""
    weight_pre   = round(BOX_BASELINE_G + rng.gauss(0, WEIGHT_PRE_STD))
    weight_entry = round(weight_pre + body_mass_g + rng.gauss(0, WEIGHT_ENTRY_STD))
    waste_true   = round(rng.uniform(WASTE_MIN_G, WASTE_MAX_G))
    weight_exit  = round(weight_pre + waste_true + rng.gauss(0, WEIGHT_EXIT_STD))

    lo_a, hi_a = AMMONIA_ANOMALY if is_anomalous else AMMONIA_NORMAL
    lo_m, hi_m = METHANE_ANOMALY if is_anomalous else METHANE_NORMAL

    return {
        "weight_pre_g":     weight_pre,
        "weight_entry_g":   weight_entry,
        "weight_exit_g":    weight_exit,
        "waste_g_true":     waste_true,
        "ammonia_peak_ppb": round(rng.uniform(lo_a, hi_a), 1),
        "methane_peak_ppb": round(rng.uniform(lo_m, hi_m), 1),
    }


# ── visit schedule ────────────────────────────────────────────────────────────

@dataclass
class SimVisit:
    cat_name:    str
    visit_num:   int          # 1-based within this cat's sequence
    visit_image: Path
    is_anomalous: bool
    sensor:      dict


def build_schedule(rng: random.Random) -> List[SimVisit]:
    """Build a randomised but reproducible schedule of 50 visits."""
    visits: List[SimVisit] = []
    for cat_name, cfg in CATS.items():
        pool = cfg["visit_images"]
        if not pool:
            raise ValueError(f"No visit images found for {cat_name}")
        for i in range(VISITS_PER_CAT):
            is_anom = (cat_name, i) in ANOMALOUS_VISITS
            visits.append(SimVisit(
                cat_name=cat_name,
                visit_num=i + 1,
                visit_image=pool[i % len(pool)],
                is_anomalous=is_anom,
                sensor=generate_sensor_readings(cfg["body_mass_g"], is_anom, rng),
            ))
    rng.shuffle(visits)
    return visits


# ── report builder ────────────────────────────────────────────────────────────

def _parse_identification(entry_result: str) -> tuple[str, str]:
    """Extract (identity_line, confidence_hint) from record_entry output."""
    for line in entry_result.splitlines():
        ll = line.lower()
        if "identified" in ll or "unknown" in ll or "unconfirmed" in ll:
            return line.strip(), ""
    return entry_result.splitlines()[0].strip() if entry_result else "—", ""


def _parse_health(exit_result: str) -> tuple[bool, str]:
    """Return (is_anomalous, one-line summary) from record_exit output."""
    text_lower = exit_result.lower()
    is_anom = any(kw in text_lower for kw in (
        "concern", "anomal", "warning", "flag", "elevated", "unusual",
        "consult", "veterinarian", "yes",
    ))
    for line in exit_result.splitlines():
        ll = line.lower()
        if any(kw in ll for kw in ("health", "concern", "anomal", "no concern")):
            return is_anom, line.strip()
    return is_anom, exit_result.splitlines()[0].strip() if exit_result else "—"


def build_report(
    visit_log:          List[dict],
    anomalous_result:   str,
    unconfirmed_result: str,
    cat_summaries:      Dict[str, str],
    registration_log:   List[dict],
) -> str:
    lines: List[str] = []

    def h(level: int, title: str) -> None:
        lines.append(f"\n{'#' * level} {title}\n")

    def rule() -> None:
        lines.append("\n---\n")

    # ── title ─────────────────────────────────────────────────────────────────
    lines.append("# Litter Box API Simulation Report\n")
    lines.append(f"**Simulation date:** {__import__('datetime').date.today()}  \n")
    lines.append(f"**Random seed:** {RANDOM_SEED}  \n")
    lines.append(f"**Cats registered:** {len(CATS)}  \n")
    lines.append(f"**Visits per cat:** {VISITS_PER_CAT}  \n")
    lines.append(f"**Total visits:** {len(visit_log)}  \n")
    injected = sum(1 for v in visit_log if v["is_anomalous"])
    lines.append(f"**Injected anomalies:** {injected}  \n")

    # ── cat registration ──────────────────────────────────────────────────────
    rule()
    h(2, "1. Cat Registration")
    lines.append(
        "Each cat was enrolled using the first image from their respective folder.  \n"
        "All database records were cleared before enrollment began.\n"
    )
    lines.append("\n| Cat | Body Mass | Registration Image | Result |\n")
    lines.append( "|-----|----------:|--------------------|---------|\n")
    for entry in registration_log:
        result_line = entry["result"].splitlines()[0][:60] if entry["result"] else "—"
        lines.append(
            f"| {entry['cat']:10s} | {entry['body_mass_g']/1000:.1f} kg "
            f"| `{entry['reg_image']}` "
            f"| {result_line} |\n"
        )

    # ── sensor model note ─────────────────────────────────────────────────────
    rule()
    h(2, "2. Sensor Model")
    lines.append(textwrap.dedent(f"""\
        The litter box is equipped with a weight scale and camera.

        | Parameter | Value |
        |-----------|-------|
        | Box baseline weight | {BOX_BASELINE_G} g |
        | Pre-entry noise (σ) | {WEIGHT_PRE_STD} g |
        | Entry noise (σ)     | {WEIGHT_ENTRY_STD} g |
        | Exit noise (σ)      | {WEIGHT_EXIT_STD} g |
        | Waste range         | {WASTE_MIN_G}–{WASTE_MAX_G} g |
        | Ammonia normal      | {AMMONIA_NORMAL[0]}–{AMMONIA_NORMAL[1]} ppb |
        | Ammonia anomaly     | {AMMONIA_ANOMALY[0]}–{AMMONIA_ANOMALY[1]} ppb |
        | Methane normal      | {METHANE_NORMAL[0]}–{METHANE_NORMAL[1]} ppb |
        | Methane anomaly     | {METHANE_ANOMALY[0]}–{METHANE_ANOMALY[1]} ppb |

        Anomalous visits injected: {injected} (Anna visit 8, Luna visit 3, Whiskers visit 5).
    """))

    # ── visit log ─────────────────────────────────────────────────────────────
    rule()
    h(2, "3. Visit Log (Chronological Replay Order)")
    lines.append(
        "| Seq | Cat | Visit # | Image | Anom? | Weight Pre | Weight Entry | "
        "Waste True | NH₃ ppb | CH₄ ppb | Identification | Health Flag |\n"
    )
    lines.append(
        "|----:|-----|--------:|-------|:-----:|-----------:|-------------:"
        "|-----------:|--------:|--------:|----------------|-------------|\n"
    )
    for v in visit_log:
        s = v["sensor"]
        ident, _ = _parse_identification(v["entry_result"])
        is_h_anom, health_line = _parse_health(v["exit_result"])
        anom_marker = "⚠️" if v["is_anomalous"] else ""
        health_marker = "⚠️ yes" if is_h_anom else "ok"
        # Shorten ident to fit
        ident_short = (ident[:45] + "…") if len(ident) > 48 else ident
        lines.append(
            f"| {v['sequence']:3d} | {v['cat_name']:10s} | {v['visit_num']:7d} "
            f"| `{v['visit_image']:20s}` | {anom_marker:^5s} "
            f"| {s['weight_pre_g']:10d} | {s['weight_entry_g']:12d} "
            f"| {s['waste_g_true']:10d} "
            f"| {s['ammonia_peak_ppb']:7.1f} | {s['methane_peak_ppb']:7.1f} "
            f"| {ident_short} | {health_marker} |\n"
        )

    # ── per-cat summaries ─────────────────────────────────────────────────────
    rule()
    h(2, "4. Per-Cat Visit Summaries")
    lines.append(
        "The following summaries are returned directly by the agent's "
        "`get_visits_by_cat()` API call.\n"
    )
    for cat_name, summary in cat_summaries.items():
        h(3, cat_name)
        lines.append(f"```\n{summary}\n```\n")

    # ── health warnings ───────────────────────────────────────────────────────
    rule()
    h(2, "5. Health Warnings")
    lines.append(
        "> **Disclaimer:** All health findings are preliminary outputs from an "
        "automated vision model and must be reviewed by a licensed veterinarian "
        "before any clinical action is taken.\n\n"
    )
    lines.append("### Anomalous visits flagged by the system\n\n")
    lines.append(f"```\n{anomalous_result}\n```\n")

    # ── unconfirmed identities ────────────────────────────────────────────────
    rule()
    h(2, "6. Unconfirmed Identities")
    lines.append(
        "Visits where the CLIP + GPT-4o pipeline did not produce a "
        "confirmed identity are listed below for human review.\n"
    )
    lines.append(f"```\n{unconfirmed_result}\n```\n")

    # ── sensor statistics ─────────────────────────────────────────────────────
    rule()
    h(2, "7. Sensor Statistics by Cat")
    lines.append("| Cat | Visits | Mean entry weight (g) | Mean waste (g) | "
                 "Mean NH₃ (ppb) | Mean CH₄ (ppb) |\n")
    lines.append("|-----|-------:|----------------------:|---------------:"
                 "|---------------:|---------------:|\n")
    for cat_name in CATS:
        cat_visits = [v for v in visit_log if v["cat_name"] == cat_name]
        n = len(cat_visits)
        avg_entry  = sum(v["sensor"]["weight_entry_g"] for v in cat_visits) / n
        avg_waste  = sum(v["sensor"]["waste_g_true"]   for v in cat_visits) / n
        avg_nh3    = sum(v["sensor"]["ammonia_peak_ppb"] for v in cat_visits) / n
        avg_ch4    = sum(v["sensor"]["methane_peak_ppb"] for v in cat_visits) / n
        expected   = CATS[cat_name]["body_mass_g"] + BOX_BASELINE_G
        lines.append(
            f"| {cat_name:10s} | {n:6d} | {avg_entry:21.0f} (expected ≈ {expected}) "
            f"| {avg_waste:14.1f} | {avg_nh3:15.1f} | {avg_ch4:15.1f} |\n"
        )

    # ── footer ────────────────────────────────────────────────────────────────
    rule()
    lines.append(
        "_Generated by `simulator/run_api_simulation.py` using the "
        "`LitterboxAgent` Python API._\n"
    )

    return "".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  API LITTER BOX SIMULATION — 5 cats × 10 visits each")
    print("=" * 70)

    # Validate visit image pools
    for cat_name, cfg in CATS.items():
        pool = cfg["visit_images"]
        if not pool:
            sys.exit(f"ERROR: No visit images found for {cat_name}. "
                     f"Check simulator/cat_pictures/ or images/cats/.")
        print(f"  {cat_name:10s}: {len(pool)} visit image(s) available")

    # ── 1. Clear simulation database ─────────────────────────────────────────
    print("\n[1/7] Clearing previous simulation data...")
    shutil.rmtree(SIM_DATA_DIR,   ignore_errors=True)
    shutil.rmtree(SIM_IMAGES_DIR, ignore_errors=True)
    SIM_DATA_DIR.mkdir(parents=True)
    SIM_IMAGES_DIR.mkdir(parents=True)
    print(f"      data dir  : {SIM_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"      images dir: {SIM_IMAGES_DIR.relative_to(PROJECT_ROOT)}")

    rng = random.Random(RANDOM_SEED)

    # ── 2. Initialise agent ───────────────────────────────────────────────────
    print("\n[2/7] Initialising LitterboxAgent (Python API)...")
    agent = LitterboxAgent(
        data_dir=str(SIM_DATA_DIR),
        images_dir=str(SIM_IMAGES_DIR),
    )
    print("      Agent ready.")

    # ── 3. Register cats ──────────────────────────────────────────────────────
    print("\n[3/7] Registering cats with reference photos...")
    registration_log: List[dict] = []
    for cat_name, cfg in CATS.items():
        reg_img  = cfg["registration_image"]
        rel_path = reg_img.relative_to(PROJECT_ROOT)
        print(f"\n  {cat_name}")
        print(f"    registration image : {rel_path}")
        result = agent.register_cat(str(reg_img), cat_name)
        print(f"    result             : {result.splitlines()[0]}")
        registration_log.append({
            "cat":         cat_name,
            "body_mass_g": cfg["body_mass_g"],
            "reg_image":   str(rel_path),
            "result":      result,
        })

    print(f"\n  Registered cats:\n  {agent.list_cats()}")

    # ── 4. Build visit schedule ───────────────────────────────────────────────
    print("\n[4/7] Building visit schedule...")
    schedule = build_schedule(rng)
    injected = sum(1 for v in schedule if v.is_anomalous)
    print(f"      Total visits  : {len(schedule)}")
    print(f"      Anomalies seeded: {injected} "
          "(Anna v8 · Luna v3 · Whiskers v5)")

    # ── 5. Replay visits ──────────────────────────────────────────────────────
    print("\n[5/7] Replaying visits (entry → exit for each)...\n")
    visit_log: List[dict] = []
    for seq, visit in enumerate(schedule, 1):
        s = visit.sensor
        anom_tag = " [ANOMALY]" if visit.is_anomalous else ""
        print(f"  [{seq:02d}/50] {visit.cat_name:10s}  visit #{visit.visit_num}"
              f"  img={visit.visit_image.name}{anom_tag}")

        # record_entry — entry image is the cat photo (used by CLIP + GPT-4o ID)
        entry_result = agent.record_entry(
            image_path=str(visit.visit_image),
            weight_pre_g=s["weight_pre_g"],
            weight_entry_g=s["weight_entry_g"],
            ammonia_peak_ppb=s["ammonia_peak_ppb"],
            methane_peak_ppb=s["methane_peak_ppb"],
        )

        # record_exit — exit image is the litter box placeholder (health analysis)
        exit_result = agent.record_exit(
            image_path=str(EXIT_IMAGE),
            weight_exit_g=s["weight_exit_g"],
            ammonia_peak_ppb=s["ammonia_peak_ppb"],
            methane_peak_ppb=s["methane_peak_ppb"],
        )

        # Print first line of each result for live feedback
        print(f"         entry : {entry_result.splitlines()[0]}")
        print(f"         exit  : {exit_result.splitlines()[0]}")

        visit_log.append({
            "sequence":      seq,
            "cat_name":      visit.cat_name,
            "visit_num":     visit.visit_num,
            "is_anomalous":  visit.is_anomalous,
            "visit_image":   visit.visit_image.name,
            "sensor":        s,
            "entry_result":  entry_result,
            "exit_result":   exit_result,
        })

    # ── 6. Save ground truth ──────────────────────────────────────────────────
    print("\n[6/7] Saving ground truth JSON...")
    with open(GROUND_TRUTH_PATH, "w") as fh:
        json.dump(
            {
                "seed":            RANDOM_SEED,
                "cats":            {k: {"body_mass_g": v["body_mass_g"],
                                        "registration_image": str(v["registration_image"]
                                                                  .relative_to(PROJECT_ROOT))}
                                    for k, v in CATS.items()},
                "anomalous_visits": [
                    {"cat": c, "visit_index_0based": i}
                    for c, i in sorted(ANOMALOUS_VISITS)
                ],
                "visits": visit_log,
            },
            fh,
            indent=2,
            default=str,
        )
    print(f"      {GROUND_TRUTH_PATH.relative_to(PROJECT_ROOT)}")

    # ── 7. Generate report ────────────────────────────────────────────────────
    print("\n[7/7] Generating simulation report...")
    anomalous_result   = agent.get_anomalous_visits()
    unconfirmed_result = agent.get_unconfirmed_visits()
    cat_summaries      = {name: agent.get_visits_by_cat(name) for name in CATS}

    report = build_report(
        visit_log=visit_log,
        anomalous_result=anomalous_result,
        unconfirmed_result=unconfirmed_result,
        cat_summaries=cat_summaries,
        registration_log=registration_log,
    )
    with open(REPORT_PATH, "w") as fh:
        fh.write(report)
    print(f"      {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    agent.close()

    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE")
    print(f"  {len(visit_log)} visits · {injected} anomalies injected")
    print(f"  Report → {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
