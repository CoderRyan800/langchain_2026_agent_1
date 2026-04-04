#!/usr/bin/env python3
"""
API-based litter box simulation — 5 cats, 10 visits each.

Uses the LitterboxAgent Python API exclusively (no direct tool imports,
no CLI invocations).  Clears the simulation database before starting so
every record in the output reflects this run.

All images come from images/cats/<cat>/ in the project root.
The first lexicographically-sorted image in each folder is used for
registration; every image in the folder (including the registration
image) is eligible for random visit selection.

Cat roster and body masses:
    Anna     — 3.2 kg  registration: images/cats/anna/001_c25e6c.jpg
    Marina   — 4.0 kg  registration: images/cats/marina/001_aad380.jpg
    Luna     — 5.5 kg  registration: images/cats/luna/001_a150db.jpg
    Natasha  — 5.0 kg  registration: images/cats/natasha/001_4c58dd.jpg
    Whiskers — 6.0 kg  registration: images/cats/whiskers/001_b0c0d4.jpg

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
import re
import random
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ── project path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from litterbox import LitterboxAgent  # noqa: E402

# ── simulation directories ────────────────────────────────────────────────────
SIM_DATA_DIR   = PROJECT_ROOT / "simulator" / "api_sim_data"
SIM_IMAGES_DIR = PROJECT_ROOT / "simulator" / "api_sim_images"

# ── cat configuration ─────────────────────────────────────────────────────────
#
# All images come from images/cats/<cat>/.  The pool includes all jpg/jpeg
# files; the first (sorted) is used for registration, and all are eligible
# for random visit selection.
#
def _cat_images(cat_dir: Path) -> List[Path]:
    """Return all jpg/jpeg images in a folder, sorted lexicographically."""
    return sorted(
        p for p in cat_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg") and p.is_file()
    )

CATS_DIR = PROJECT_ROOT / "images" / "cats"

CATS: Dict[str, dict] = {
    "Anna":    {"body_mass_g": 3200, "dir": CATS_DIR / "anna"},
    "Marina":  {"body_mass_g": 4000, "dir": CATS_DIR / "marina"},
    "Luna":    {"body_mass_g": 5500, "dir": CATS_DIR / "luna"},
    "Natasha": {"body_mass_g": 5000, "dir": CATS_DIR / "natasha"},
    "Whiskers":{"body_mass_g": 6000, "dir": CATS_DIR / "whiskers"},
}

# Resolve image pools at import time and fail fast if anything is missing
for _name, _cfg in CATS.items():
    _pool = _cat_images(_cfg["dir"])
    if not _pool:
        sys.exit(f"ERROR: no images found in {_cfg['dir']}")
    _cfg["all_images"]          = _pool          # full pool for visits
    _cfg["registration_image"]  = _pool[0]       # first sorted = reference

VISITS_PER_CAT = 10

# Weight sensor — mild Gaussian noise
BOX_BASELINE_G   = 2000
WEIGHT_PRE_STD   = 20
WEIGHT_ENTRY_STD = 30
WEIGHT_EXIT_STD  = 20
WASTE_MIN_G      = 30
WASTE_MAX_G      = 120

# Gas sensor ranges (ppb)
AMMONIA_NORMAL  = (5.0,   60.0)
AMMONIA_ANOMALY = (150.0, 350.0)
METHANE_NORMAL  = (0.0,   40.0)
METHANE_ANOMALY = (80.0,  180.0)

# Seeded anomalies: (cat_name, 0-based visit index for that cat)
ANOMALOUS_VISITS = {
    ("Anna",     7),   # visit 8  — digestive (elevated CH₄)
    ("Luna",     2),   # visit 3  — combined  (elevated NH₃ + CH₄)
    ("Whiskers", 4),   # visit 5  — urinary   (elevated NH₃)
}

RANDOM_SEED = 42

# Exit image — placeholder litter box photo used for all exit captures
EXIT_IMAGE = PROJECT_ROOT / "simulator" / "assets" / "used_box.jpg"

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
    lo_a, hi_a   = AMMONIA_ANOMALY if is_anomalous else AMMONIA_NORMAL
    lo_m, hi_m   = METHANE_ANOMALY if is_anomalous else METHANE_NORMAL
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
    cat_name:     str
    visit_num:    int          # 1-based within this cat's sequence
    visit_image:  Path         # randomly chosen from pool
    is_anomalous: bool
    sensor:       dict


def build_schedule(rng: random.Random) -> List[SimVisit]:
    """Build a randomised but reproducible schedule of 50 visits."""
    visits: List[SimVisit] = []
    for cat_name, cfg in CATS.items():
        pool = cfg["all_images"]
        for i in range(VISITS_PER_CAT):
            is_anom = (cat_name, i) in ANOMALOUS_VISITS
            visits.append(SimVisit(
                cat_name=cat_name,
                visit_num=i + 1,
                visit_image=rng.choice(pool),   # random selection from full pool
                is_anomalous=is_anom,
                sensor=generate_sensor_readings(cfg["body_mass_g"], is_anom, rng),
            ))
    rng.shuffle(visits)
    return visits


# ── result parsing ─────────────────────────────────────────────────────────────

def parse_identification(entry_result: str, exit_result: str) -> tuple[str, Optional[float]]:
    """
    Extract (identified_cat, similarity_score) from the combined tool output.

    record_entry returns the CLIP score only on failure ("Cat not identified,
    best score X.XX").  On success the tentative cat name is surfaced in
    record_exit ("Visit #N closed (tentative cat: Name)").
    We combine both to recover the best available information.
    """
    # ── cat name from exit result ─────────────────────────────────────────────
    # "Visit #N closed (tentative cat: Anna)."
    # "Visit #N closed (tentative cat: Unknown)."
    cat = "Unknown"
    m = re.search(r'tentative cat[:\s]+([A-Za-z]+)', exit_result, re.IGNORECASE)
    if m:
        cat = m.group(1)

    # ── CLIP score from entry result ──────────────────────────────────────────
    # Successful ID: "Visit #N opened … Tentative ID: Anna (score: 0.96)"
    score: Optional[float] = None
    m = re.search(r'score[:\s]+([\d.]+)', entry_result, re.IGNORECASE)
    if m:
        score = float(m.group(1))

    # Failure: "Cat not identified (best score 0.89)"
    if score is None:
        m = re.search(r'best\s+score\s+([\d.]+)', entry_result, re.IGNORECASE)
        if m:
            score = float(m.group(1))

    # Bare "score 0.55" fallback anywhere in entry result
    if score is None:
        m = re.search(r'\bscore\b\s+([\d.]+)', entry_result, re.IGNORECASE)
        if m:
            score = float(m.group(1))

    return cat, score


def parse_health_flag(text: str) -> bool:
    """Return True if the exit result contains a health concern."""
    lower = text.lower()
    return any(kw in lower for kw in (
        "concern", "anomal", "warning", "elevated", "unusual",
        "flag", "consult", "veterinarian",
    ))


# ── report builder ────────────────────────────────────────────────────────────

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

    injected = sum(1 for v in visit_log if v["is_anomalous"])

    # ── title ─────────────────────────────────────────────────────────────────
    lines.append("# Litter Box API Simulation Report\n")
    lines.append(f"**Simulation date:** {__import__('datetime').date.today()}  \n")
    lines.append(f"**Random seed:** {RANDOM_SEED}  \n")
    lines.append(f"**Cats registered:** {len(CATS)}  \n")
    lines.append(f"**Visits per cat:** {VISITS_PER_CAT}  \n")
    lines.append(f"**Total visits:** {len(visit_log)}  \n")
    lines.append(f"**Injected anomalies:** {injected}  \n")

    # ── cat registration ──────────────────────────────────────────────────────
    rule()
    h(2, "1. Cat Registration")
    lines.append(
        "Each cat was enrolled using the **first** (lexicographically sorted) "
        "image in their `images/cats/<cat>/` folder.  "
        "The database was cleared before enrollment.\n"
    )
    lines.append("\n| Cat | Body Mass | Registration Image | Pool Size | Result |\n")
    lines.append( "|-----|----------:|--------------------:|----------:|--------|\n")
    for entry in registration_log:
        result_short = entry["result"].splitlines()[0][:55]
        lines.append(
            f"| {entry['cat']:10s} | {entry['body_mass_g']/1000:.1f} kg "
            f"| `{entry['reg_image']}` "
            f"| {entry['pool_size']} images "
            f"| {result_short} |\n"
        )

    # ── sensor model ──────────────────────────────────────────────────────────
    rule()
    h(2, "2. Sensor Model")
    lines.append(textwrap.dedent(f"""\
        The litter box is equipped with a weight scale and camera.

        | Parameter | Value |
        |-----------|-------|
        | Box baseline weight | {BOX_BASELINE_G} g |
        | Pre-entry noise σ   | {WEIGHT_PRE_STD} g |
        | Entry noise σ       | {WEIGHT_ENTRY_STD} g |
        | Exit noise σ        | {WEIGHT_EXIT_STD} g |
        | Waste range         | {WASTE_MIN_G}–{WASTE_MAX_G} g |
        | Ammonia normal      | {AMMONIA_NORMAL[0]}–{AMMONIA_NORMAL[1]} ppb |
        | Ammonia anomaly     | {AMMONIA_ANOMALY[0]}–{AMMONIA_ANOMALY[1]} ppb |
        | Methane normal      | {METHANE_NORMAL[0]}–{METHANE_NORMAL[1]} ppb |
        | Methane anomaly     | {METHANE_ANOMALY[0]}–{METHANE_ANOMALY[1]} ppb |

        Seeded anomalies: **Anna visit 8** (elevated CH₄), **Luna visit 3** (elevated NH₃ + CH₄), **Whiskers visit 5** (elevated NH₃).
    """))

    # ── visit log ─────────────────────────────────────────────────────────────
    rule()
    h(2, "3. Visit Log — Image Used, Cat ID, and Sensor Data")
    lines.append(
        "| Seq | Cat | #  | Image Used | Anom? | "
        "Identified As | Sim Score | "
        "Pre (g) | Entry (g) | Exit (g) | Waste (g) | NH₃ | CH₄ | Health |\n"
    )
    lines.append(
        "|----:|-----|----|------------|:-----:|"
        "--------------|----------:|"
        "-------:|----------:|---------:|----------:|----:|----:|--------|\n"
    )
    for v in visit_log:
        s         = v["sensor"]
        anom_sym  = "⚠️" if v["is_anomalous"] else ""
        id_cat    = v["identified_cat"]
        score_str = f"{v['similarity_score']:.2f}" if v["similarity_score"] is not None else "n/a"
        health    = "⚠️" if v["health_flag"] else "ok"
        img_name  = v["visit_image_name"]
        lines.append(
            f"| {v['sequence']:3d} | {v['cat_name']:10s} | {v['visit_num']:2d} "
            f"| `{img_name}` | {anom_sym:^5s} "
            f"| {id_cat:12s} | {score_str:>9s} "
            f"| {s['weight_pre_g']:7d} | {s['weight_entry_g']:9d} "
            f"| {s['weight_exit_g']:8d} | {s['waste_g_true']:9d} "
            f"| {s['ammonia_peak_ppb']:5.1f} | {s['methane_peak_ppb']:5.1f} "
            f"| {health} |\n"
        )

    # ── identification accuracy ───────────────────────────────────────────────
    rule()
    h(2, "4. Identification Summary")
    lines.append("| Cat | Visits | Identified (tentative) | Unknown | Avg Score |\n")
    lines.append("|-----|-------:|-----------------------:|--------:|----------:|\n")
    for cat_name in CATS:
        cv       = [v for v in visit_log if v["cat_name"] == cat_name]
        known    = [v for v in cv if v["identified_cat"] != "Unknown"]
        unknown  = [v for v in cv if v["identified_cat"] == "Unknown"]
        scores   = [v["similarity_score"] for v in cv if v["similarity_score"] is not None]
        avg_sc   = f"{sum(scores)/len(scores):.2f}" if scores else "—"
        lines.append(
            f"| {cat_name:10s} | {len(cv):6d} | {len(known):22d} | {len(unknown):7d} | {avg_sc:>9s} |\n"
        )

    # ── per-cat summaries ─────────────────────────────────────────────────────
    rule()
    h(2, "5. Per-Cat Visit Summaries (from API)")
    lines.append(
        "The following text is returned directly by `agent.get_visits_by_cat()`.\n"
    )
    for cat_name, summary in cat_summaries.items():
        h(3, cat_name)
        lines.append(f"```\n{summary}\n```\n")

    # ── health warnings ───────────────────────────────────────────────────────
    rule()
    h(2, "6. Health Warnings")
    lines.append(
        "> **Disclaimer:** All health findings are preliminary outputs from an "
        "automated vision model and must be reviewed by a licensed veterinarian "
        "before any clinical action is taken.\n\n"
    )
    lines.append("### Visits flagged as anomalous by the system\n\n")
    lines.append(f"```\n{anomalous_result}\n```\n")

    # detail for seeded anomalies
    h(3, "Seeded anomaly detail")
    lines.append(
        "| Cat | Visit # | NH₃ (ppb) | CH₄ (ppb) | System health flag | Identified as |\n"
    )
    lines.append(
        "|-----|--------:|----------:|----------:|-------------------|---------------|\n"
    )
    for v in visit_log:
        if v["is_anomalous"]:
            health = "⚠️ yes" if v["health_flag"] else "no (missed)"
            lines.append(
                f"| {v['cat_name']:10s} | {v['visit_num']:7d} "
                f"| {v['sensor']['ammonia_peak_ppb']:9.1f} "
                f"| {v['sensor']['methane_peak_ppb']:9.1f} "
                f"| {health} | {v['identified_cat']} |\n"
            )

    # ── unconfirmed identities ────────────────────────────────────────────────
    rule()
    h(2, "7. Unconfirmed Identities")
    lines.append(
        "All visits use tentative (unconfirmed) IDs until a human calls "
        "`confirm_identity()`.  Visits with `Unknown` identity need manual review.\n\n"
    )
    lines.append(f"```\n{unconfirmed_result}\n```\n")

    # ── sensor statistics ─────────────────────────────────────────────────────
    rule()
    h(2, "8. Sensor Statistics by Cat")
    lines.append(
        "| Cat | Body Mass | Mean entry (g) | Expected entry (g) | "
        "Error | Mean waste (g) | Mean NH₃ | Mean CH₄ |\n"
    )
    lines.append(
        "|-----|----------:|---------------:|-------------------:|"
        "------:|---------------:|---------:|---------:|\n"
    )
    for cat_name, cfg in CATS.items():
        cv       = [v for v in visit_log if v["cat_name"] == cat_name]
        n        = len(cv)
        avg_e    = sum(v["sensor"]["weight_entry_g"] for v in cv) / n
        expected = cfg["body_mass_g"] + BOX_BASELINE_G
        err      = avg_e - expected
        avg_w    = sum(v["sensor"]["waste_g_true"]   for v in cv) / n
        avg_nh3  = sum(v["sensor"]["ammonia_peak_ppb"] for v in cv) / n
        avg_ch4  = sum(v["sensor"]["methane_peak_ppb"] for v in cv) / n
        lines.append(
            f"| {cat_name:10s} | {cfg['body_mass_g']/1000:.1f} kg "
            f"| {avg_e:14.0f} | {expected:18d} "
            f"| {err:+.0f} g | {avg_w:14.1f} "
            f"| {avg_nh3:8.1f} | {avg_ch4:8.1f} |\n"
        )

    rule()
    lines.append(
        "_Generated by `simulator/run_api_simulation.py` "
        "using the `LitterboxAgent` Python API._\n"
    )
    return "".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("  API LITTER BOX SIMULATION — 5 cats × 10 visits each")
    print("=" * 72)

    # Print image pool sizes
    for cat_name, cfg in CATS.items():
        pool = cfg["all_images"]
        print(f"  {cat_name:10s}: {len(pool)} image(s) in pool — "
              + ", ".join(p.name for p in pool))

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
        reg_img = cfg["registration_image"]
        pool    = cfg["all_images"]
        rel     = reg_img.relative_to(PROJECT_ROOT)
        print(f"\n  {cat_name}")
        print(f"    registration image : {rel}  ({len(pool)} images in pool)")
        result = agent.register_cat(str(reg_img), cat_name)
        print(f"    result             : {result.splitlines()[0]}")
        registration_log.append({
            "cat":         cat_name,
            "body_mass_g": cfg["body_mass_g"],
            "reg_image":   str(rel),
            "pool_size":   len(pool),
            "result":      result,
        })

    print(f"\n  Registered cats:\n  {agent.list_cats()}")

    # ── 4. Build visit schedule ───────────────────────────────────────────────
    print("\n[4/7] Building visit schedule (random image selection)...")
    schedule = build_schedule(rng)
    injected = sum(1 for v in schedule if v.is_anomalous)
    print(f"      Total visits    : {len(schedule)}")
    print(f"      Seeded anomalies: {injected}  "
          "(Anna v8 · Luna v3 · Whiskers v5)")

    # ── 5. Replay visits ──────────────────────────────────────────────────────
    print("\n[5/7] Replaying visits (entry → exit)...\n")
    visit_log: List[dict] = []

    for seq, visit in enumerate(schedule, 1):
        s        = visit.sensor
        anom_tag = "  [ANOMALY]" if visit.is_anomalous else ""
        img_rel  = visit.visit_image.relative_to(PROJECT_ROOT)
        print(f"  [{seq:02d}/50] {visit.cat_name:10s}  visit #{visit.visit_num}"
              f"  img={visit.visit_image.name}{anom_tag}")

        # record_entry: entry image is the randomly selected cat photo
        entry_result = agent.record_entry(
            image_path=str(visit.visit_image),
            weight_pre_g=s["weight_pre_g"],
            weight_entry_g=s["weight_entry_g"],
            ammonia_peak_ppb=s["ammonia_peak_ppb"],
            methane_peak_ppb=s["methane_peak_ppb"],
        )

        # record_exit: exit image is the litter box placeholder
        exit_result = agent.record_exit(
            image_path=str(EXIT_IMAGE),
            weight_exit_g=s["weight_exit_g"],
            ammonia_peak_ppb=s["ammonia_peak_ppb"],
            methane_peak_ppb=s["methane_peak_ppb"],
        )

        identified_cat, sim_score = parse_identification(entry_result, exit_result)
        health_flag = parse_health_flag(exit_result)

        print(f"         entry  : {entry_result.splitlines()[0]}")
        print(f"         id     : {identified_cat}  score={sim_score}")
        print(f"         exit   : {exit_result.splitlines()[0]}")
        print(f"         health : {'⚠️  FLAGGED' if health_flag else 'ok'}")

        visit_log.append({
            "sequence":         seq,
            "cat_name":         visit.cat_name,
            "visit_num":        visit.visit_num,
            "is_anomalous":     visit.is_anomalous,
            "visit_image_path": str(img_rel),
            "visit_image_name": visit.visit_image.name,
            "identified_cat":   identified_cat,
            "similarity_score": sim_score,
            "health_flag":      health_flag,
            "sensor":           s,
            "entry_result":     entry_result,
            "exit_result":      exit_result,
        })

    # ── 6. Save ground truth ──────────────────────────────────────────────────
    print("\n[6/7] Saving ground truth JSON...")
    with open(GROUND_TRUTH_PATH, "w") as fh:
        json.dump(
            {
                "seed":   RANDOM_SEED,
                "cats":   {
                    k: {
                        "body_mass_g":        v["body_mass_g"],
                        "registration_image": str(v["registration_image"]
                                                  .relative_to(PROJECT_ROOT)),
                        "pool_size":          len(v["all_images"]),
                        "pool":               [str(p.relative_to(PROJECT_ROOT))
                                               for p in v["all_images"]],
                    }
                    for k, v in CATS.items()
                },
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
    print("\n[7/7] Querying results and generating report...")
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

    # ── summary ───────────────────────────────────────────────────────────────
    total   = len(visit_log)
    known   = sum(1 for v in visit_log if v["identified_cat"] != "Unknown")
    unknown = total - known
    flagged = sum(1 for v in visit_log if v["health_flag"])
    print("\n" + "=" * 72)
    print("  SIMULATION COMPLETE")
    print(f"  {total} visits total · {known} identified · {unknown} unknown")
    print(f"  {flagged} visits flagged with health concerns · {injected} anomalies seeded")
    print(f"  Report → {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
