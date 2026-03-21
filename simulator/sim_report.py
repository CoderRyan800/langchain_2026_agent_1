"""Generates the Markdown simulation report.

Joins the ground-truth JSON log with live database records to compute
identity accuracy, weight accuracy, sensor coverage, and anomaly detection
metrics.

Can be run standalone:
    python simulator/sim_report.py
"""

import json
import sys
from pathlib import Path
from typing import Optional

SIM_DIR      = Path(__file__).parent
PROJECT_ROOT = SIM_DIR.parent

sys.path.insert(0, str(SIM_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _fetch_db_row(visit_id: int) -> Optional[dict]:
    from litterbox.db import get_conn  # noqa: PLC0415
    with get_conn() as conn:
        row = conn.execute(
            """SELECT v.*,
                      c.name  AS confirmed_name,
                      tc.name AS tentative_name
               FROM visits v
               LEFT JOIN cats c  ON v.confirmed_cat_id = c.cat_id
               LEFT JOIN cats tc ON v.tentative_cat_id  = tc.cat_id
               WHERE v.visit_id = ?""",
            (visit_id,),
        ).fetchone()
    return dict(row) if row else None


def _identified_as(db_row: dict) -> Optional[str]:
    """Return the cat name the agent assigned (confirmed > tentative > None)."""
    return db_row.get("confirmed_name") or db_row.get("tentative_name")


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate(ground_truth_path: Path, report_path: Path) -> None:
    from litterbox.db import DB_PATH  # noqa: PLC0415

    with open(ground_truth_path) as f:
        events: list[dict] = json.load(f)

    # Join ground truth with DB rows
    rows: list[tuple[dict, Optional[dict]]] = []
    for ev in events:
        vid    = ev.get("visit_id")
        db_row = _fetch_db_row(vid) if vid is not None else None
        rows.append((ev, db_row))

    total = len(rows)
    cat_names = ["Anna", "Marina", "Luna", "Natasha"]
    true_weights = {"Anna": 3200, "Marina": 4000, "Luna": 5000, "Natasha": 5500}

    # ---- Identity accuracy ----
    correct_id    = 0
    wrong_id      = 0
    unidentified  = 0
    for ev, db in rows:
        if db is None:
            unidentified += 1
            continue
        identified = _identified_as(db)
        if identified == ev["cat_name"]:
            correct_id += 1
        elif identified is None:
            unidentified += 1
        else:
            wrong_id += 1

    # ---- Weight accuracy ----
    weight_errors: dict[str, list[float]] = {c: [] for c in cat_names}
    for ev, db in rows:
        if db is None:
            continue
        cw = db.get("cat_weight_g")
        if cw is not None:
            weight_errors[ev["cat_name"]].append(abs(cw - ev["cat_true_weight_g"]))

    # ---- Waste weight ----
    waste_by_cat: dict[str, list[float]] = {c: [] for c in cat_names}
    for ev, db in rows:
        if db is None:
            continue
        ww = db.get("waste_weight_g")
        if ww is not None:
            waste_by_cat[ev["cat_name"]].append(float(ww))

    # ---- Sensor coverage ----
    ammonia_present = sum(1 for ev, _ in rows if ev["ammonia_peak_ppb"] is not None)
    methane_present = sum(1 for ev, _ in rows if ev["methane_peak_ppb"] is not None)

    # ---- Anomaly detection ----
    seeded_detected  = 0
    seeded_missed    = 0
    false_positives  = 0
    for ev, db in rows:
        if db is None:
            continue
        flagged = bool(db.get("is_anomalous"))
        seeded  = bool(ev["is_anomalous_seed"])
        if seeded and flagged:
            seeded_detected += 1
        elif seeded and not flagged:
            seeded_missed += 1
        elif not seeded and flagged:
            false_positives += 1

    # ---- Null handling ----
    null_failures = sum(
        1 for ev, db in rows
        if ev.get("visit_id") is None  # entry call itself failed to return a visit_id
    )

    # ---- Build Markdown ----
    def pct(n: int) -> str:
        return f"{100 * n / total:.0f}%" if total else "—"

    def mean_std(values: list[float]) -> tuple[str, str]:
        if not values:
            return "—", "—"
        m = sum(values) / len(values)
        v = sum((x - m) ** 2 for x in values) / len(values)
        return f"{m:.1f}", f"{v ** 0.5:.1f}"

    lines: list[str] = [
        "# Litter Box Monitor — Simulation Report",
        "",
        "## 1. Run Summary",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Total simulated events | {total} |",
        f"| Visits found in DB | {sum(1 for _, db in rows if db is not None)} |",
        f"| Seeded anomalous events | {sum(1 for ev, _ in rows if ev['is_anomalous_seed'])} |",
        f"| DB path | `{DB_PATH}` |",
        "",
        "---",
        "",
        "## 2. Identity Accuracy",
        "",
        f"| Outcome | Count | % |",
        f"|---------|-------|----|",
        f"| Correctly identified | {correct_id} | {pct(correct_id)} |",
        f"| Wrong identity | {wrong_id} | {pct(wrong_id)} |",
        f"| Unidentified | {unidentified} | {pct(unidentified)} |",
        "",
        "### Per-cat breakdown",
        "",
        "| Cat | Visits | Correct | Wrong | Unidentified |",
        "|-----|--------|---------|-------|--------------|",
    ]

    for cat in cat_names:
        cat_rows = [(ev, db) for ev, db in rows if ev["cat_name"] == cat]
        n = len(cat_rows)
        c = sum(1 for ev, db in cat_rows
                if db and _identified_as(db) == cat)
        w = sum(1 for ev, db in cat_rows
                if db and _identified_as(db) not in (cat, None))
        u = n - c - w
        lines.append(f"| {cat} | {n} | {c} | {w} | {u} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Weight Accuracy",
        "",
        "> Cat weight is derived as `weight_entry_g − weight_pre_g`.",
        "> Error = |measured cat weight − true cat weight|.",
        "",
        "| Cat | True weight (g) | Mean error (g) | Std dev (g) | Samples |",
        "|-----|----------------|----------------|-------------|---------|",
    ]

    for cat in cat_names:
        errs = weight_errors[cat]
        m, s = mean_std(errs)
        lines.append(f"| {cat} | {true_weights[cat]:,} | {m} | {s} | {len(errs)} |")

    lines += [
        "",
        "---",
        "",
        "## 4. Waste Weight",
        "",
        "> Waste weight = `weight_exit_g − weight_pre_g`.",
        "",
        "| Cat | Mean waste deposited (g) | Std dev (g) | Samples |",
        "|-----|--------------------------|-------------|---------|",
    ]

    for cat in cat_names:
        ws = waste_by_cat[cat]
        m, s = mean_std(ws)
        lines.append(f"| {cat} | {m} | {s} | {len(ws)} |")

    lines += [
        "",
        "---",
        "",
        "## 5. Sensor Coverage",
        "",
        f"| Sensor | Present | Null | Coverage |",
        f"|--------|---------|------|----------|",
        f"| Ammonia (NH₃) | {ammonia_present} | {total - ammonia_present} | {pct(ammonia_present)} |",
        f"| Methane (CH₄) | {methane_present} | {total - methane_present} | {pct(methane_present)} |",
        "",
        "---",
        "",
        "## 6. Anomaly Detection",
        "",
        "> 🌱 = seeded by simulator  ⚠️ = flagged by agent health analysis",
        "",
        f"| Outcome | Count |",
        f"|---------|-------|",
        f"| Seeded anomalies detected (true positives) | {seeded_detected} |",
        f"| Seeded anomalies missed (false negatives) | {seeded_missed} |",
        f"| Non-seeded events flagged (false positives) | {false_positives} |",
        "",
        "---",
        "",
        "## 7. Null Sensor Handling",
        "",
        ("✅ All events with null sensor readings completed without errors."
         if null_failures == 0
         else f"❌ {null_failures} event(s) failed to produce a DB visit_id."),
        "",
        "---",
        "",
        "## 8. Raw Event Table",
        "",
        "| # | Sim time | Cat | True wt | Pre | Entry wt | Exit wt | Waste | NH₃ | CH₄ | Seed | DB# | Identified |",
        "|---|----------|-----|--------:|----:|---------:|--------:|------:|----:|----:|------|-----|------------|",
    ]

    for ev, db in rows:
        vid        = ev.get("visit_id") or "—"
        identified = _identified_as(db) if db else "—"
        if identified is None:
            identified = "Unknown"
        anom_seed = "🌱" if ev["is_anomalous_seed"] else ""
        anom_db   = "⚠️"  if db and db.get("is_anomalous") else ""
        anom_col  = f"{anom_seed}{anom_db}" if (anom_seed or anom_db) else "—"
        nh3 = f"{ev['ammonia_peak_ppb']:.0f}" if ev["ammonia_peak_ppb"] is not None else "null"
        ch4 = f"{ev['methane_peak_ppb']:.0f}" if ev["methane_peak_ppb"] is not None else "null"
        t   = ev["simulated_time"][:16]
        lines.append(
            f"| {ev['event_index']} | {t} | {ev['cat_name']} "
            f"| {ev['cat_true_weight_g']:,} "
            f"| {ev['weight_pre_g']:,} | {ev['weight_entry_g']:,} "
            f"| {ev['weight_exit_g']:,} | {ev['waste_g_true']:,} "
            f"| {nh3} | {ch4} | {anom_col} | {vid} | {identified} |"
        )

    report_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sim_config import GROUND_TRUTH_PATH, REPORT_PATH  # noqa: PLC0415
    generate(GROUND_TRUTH_PATH, REPORT_PATH)
    print(f"Report written to {REPORT_PATH}")
