import base64
import mimetypes
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI

from litterbox.db import get_conn, init_db
from litterbox.embeddings import add_to_index, find_candidates, ID_THRESHOLD
from litterbox.health import build_health_prompt, parse_health_response

PROJECT_ROOT = Path(__file__).parent.parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _abs(rel_or_abs: str) -> Path:
    """Resolve a path that may be relative to project root or absolute."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _store_image(src: Path, dest_dir: Path, filename: str) -> str:
    """Copy src into dest_dir/filename; return path relative to PROJECT_ROOT."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    shutil.copy2(str(src), str(dest))
    return str(dest.relative_to(PROJECT_ROOT))


def _b64(path: str) -> Tuple[str, str]:
    """Return (base64_data, mime_type) for an image file."""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def _image_content_block(path: str) -> Dict:
    data, mime = _b64(path)
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def _run_gpt4o_vision(prompt: str, *image_paths: str) -> str:
    """Send a prompt + one or more images to GPT-4o and return the text response."""
    llm = ChatOpenAI(model="gpt-4o")
    content = [{"type": "text", "text": prompt}]
    for p in image_paths:
        content.append(_image_content_block(p))
    response = llm.invoke([HumanMessage(content=content)])
    return response.content


def _log_sensor_events(
    conn,
    visit_id: int,
    phase: str,
    recorded_at: str,
    weight_g: Optional[float] = None,
    ammonia_peak_ppb: Optional[float] = None,
    methane_peak_ppb: Optional[float] = None,
) -> None:
    """Insert rows into visit_sensor_events for any non-None readings."""
    if weight_g is not None:
        conn.execute(
            """INSERT INTO visit_sensor_events
               (visit_id, recorded_at, phase, sensor_type, value_numeric, unit)
               VALUES (?, ?, ?, 'weight', ?, 'g')""",
            (visit_id, recorded_at, phase, weight_g),
        )
    if ammonia_peak_ppb is not None:
        conn.execute(
            """INSERT INTO visit_sensor_events
               (visit_id, recorded_at, phase, sensor_type, value_numeric, unit)
               VALUES (?, ?, ?, 'ammonia', ?, 'ppb')""",
            (visit_id, recorded_at, phase, ammonia_peak_ppb),
        )
    if methane_peak_ppb is not None:
        conn.execute(
            """INSERT INTO visit_sensor_events
               (visit_id, recorded_at, phase, sensor_type, value_numeric, unit)
               VALUES (?, ?, ?, 'methane', ?, 'ppb')""",
            (visit_id, recorded_at, phase, methane_peak_ppb),
        )


def _identify_cat(
    image_path: str,
) -> Tuple[Optional[int], Optional[str], float, str]:
    """
    Run the two-stage cat identification pipeline.

    Stage 1 — CLIP nearest-neighbor search against stored reference images.
    Stage 2 — GPT-4o visual confirmation for candidates above ID_THRESHOLD.

    Returns (cat_id, cat_name, similarity_score, reasoning).
    All values are None/0.0/"..." when identification fails.
    """
    candidates = find_candidates(image_path, n_results=3)
    if not candidates:
        return None, None, 0.0, "No reference images in database yet."

    for cat_name, cat_id, score, ref_path in candidates:
        if score < ID_THRESHOLD:
            break  # candidates are sorted descending; no point continuing
        prompt = (
            f"Image 1 is a new litter-box camera photo. "
            f"Image 2 is a known reference photo of a cat named {cat_name}. "
            f"Does Image 1 appear to show the same cat as Image 2? "
            f"Answer YES or NO, then give a brief reason."
        )
        answer = _run_gpt4o_vision(prompt, image_path, ref_path)
        if answer.strip().upper().startswith("YES"):
            return cat_id, cat_name, score, answer

    best_name, _, best_score, _ = candidates[0]
    return (
        None,
        None,
        best_score,
        f"Could not confirm identity. Best candidate '{best_name}' scored "
        f"{best_score:.2f} (threshold {ID_THRESHOLD}).",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def register_cat_image(image_path: str, cat_name: str) -> str:
    """Register a reference photo for a named cat.

    image_path: absolute path or path relative to the project root.
    cat_name: the cat's name — MUST always be provided; ask the user if missing.
    """
    init_db()
    src = _abs(image_path)
    if not src.exists():
        return f"Error: file not found: {image_path}"

    with get_conn() as conn:
        row = conn.execute(
            "SELECT cat_id FROM cats WHERE name = ?", (cat_name,)
        ).fetchone()
        if row:
            cat_id = row["cat_id"]
        else:
            cur = conn.execute("INSERT INTO cats (name) VALUES (?)", (cat_name,))
            cat_id = cur.lastrowid

        count = conn.execute(
            "SELECT COUNT(*) FROM cat_images WHERE cat_id = ?", (cat_id,)
        ).fetchone()[0]

        dest_dir = IMAGES_DIR / "cats" / cat_name.lower().replace(" ", "_")
        filename = f"{count + 1:03d}_{uuid.uuid4().hex[:6]}.jpg"
        stored_path = _store_image(src, dest_dir, filename)

        chroma_id = str(uuid.uuid4())
        add_to_index(chroma_id, str(PROJECT_ROOT / stored_path), cat_name, cat_id)

        conn.execute(
            "INSERT INTO cat_images (cat_id, file_path, chroma_id) VALUES (?, ?, ?)",
            (cat_id, stored_path, chroma_id),
        )

    return (
        f"Registered reference image #{count + 1} for '{cat_name}'. "
        f"Stored at: {stored_path}"
    )


@tool
def record_entry(
    image_path: str,
    weight_pre_g: Optional[float] = None,
    weight_entry_g: Optional[float] = None,
    ammonia_peak_ppb: Optional[float] = None,
    methane_peak_ppb: Optional[float] = None,
) -> str:
    """Record a cat entering the litter box (called by the sensor system).

    image_path:       path to the entry photo captured by the camera.
    weight_pre_g:     box + litter baseline weight before the cat entered, in grams
                      (omit if no scale).
    weight_entry_g:   box + litter + cat weight at the moment of entry, in grams
                      (omit if no scale).
    ammonia_peak_ppb: peak ammonia (NH3) sensor reading in ppb (omit if no gas sensor).
    methane_peak_ppb: peak methane (CH4) sensor reading in ppb (omit if no gas sensor).

    Creates a new visit record and runs the cat identification pipeline.
    """
    init_db()
    src = _abs(image_path)
    if not src.exists():
        return f"Error: file not found: {image_path}"

    now = datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    dest_dir = IMAGES_DIR / "visits" / date_str
    filename = f"{uuid.uuid4().hex[:8]}_entry.jpg"
    stored_path = _store_image(src, dest_dir, filename)

    cat_id, cat_name, score, reasoning = _identify_cat(str(src))

    cat_weight_g = None
    if weight_pre_g is not None and weight_entry_g is not None:
        cat_weight_g = weight_entry_g - weight_pre_g

    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO visits
               (entry_time, entry_image_path, tentative_cat_id, similarity_score,
                weight_pre_g, weight_entry_g, cat_weight_g,
                ammonia_peak_ppb, methane_peak_ppb)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now.isoformat(), stored_path, cat_id, score,
             weight_pre_g, weight_entry_g, cat_weight_g,
             ammonia_peak_ppb, methane_peak_ppb),
        )
        visit_id = cur.lastrowid

        if weight_pre_g is not None:
            _log_sensor_events(conn, visit_id, "pre_entry", now.isoformat(),
                               weight_g=weight_pre_g)
        if weight_entry_g is not None or ammonia_peak_ppb is not None or methane_peak_ppb is not None:
            _log_sensor_events(conn, visit_id, "entry", now.isoformat(),
                               weight_g=weight_entry_g,
                               ammonia_peak_ppb=ammonia_peak_ppb,
                               methane_peak_ppb=methane_peak_ppb)

    if cat_name:
        id_msg = f"Tentative ID: {cat_name} (similarity {score:.2f})."
    else:
        id_msg = f"Cat not identified (best score {score:.2f}). Flagged for human review."

    sensor_parts = []
    if cat_weight_g is not None:
        sensor_parts.append(f"cat weight {cat_weight_g:.0f} g")
    if ammonia_peak_ppb is not None:
        sensor_parts.append(f"NH\u2083 {ammonia_peak_ppb:.0f} ppb")
    if methane_peak_ppb is not None:
        sensor_parts.append(f"CH\u2084 {methane_peak_ppb:.0f} ppb")
    sensor_msg = f"Sensors: {', '.join(sensor_parts)}.\n" if sensor_parts else ""

    return (
        f"Visit #{visit_id} opened at {now.strftime('%Y-%m-%d %H:%M:%S')} UTC.\n"
        f"Entry image: {stored_path}\n"
        f"{id_msg}\n"
        f"{sensor_msg}"
        f"Reasoning: {reasoning}"
    )


@tool
def record_exit(
    image_path: str,
    weight_exit_g: Optional[float] = None,
    ammonia_peak_ppb: Optional[float] = None,
    methane_peak_ppb: Optional[float] = None,
) -> str:
    """Record a cat exiting the litter box and run a health analysis (called by the sensor system).

    image_path:       path to the exit photo captured by the camera.
    weight_exit_g:    box + litter + waste weight after the cat has left, in grams
                      (omit if no scale).
    ammonia_peak_ppb: peak ammonia (NH3) reading during/after the visit in ppb
                      (omit if no gas sensor).
    methane_peak_ppb: peak methane (CH4) reading during/after the visit in ppb
                      (omit if no gas sensor).

    Automatically associates with the most recent open visit.
    If no open visit exists, an orphan exit record is created with a warning.
    """
    init_db()
    src = _abs(image_path)
    if not src.exists():
        return f"Error: file not found: {image_path}"

    now = datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    dest_dir = IMAGES_DIR / "visits" / date_str
    filename = f"{uuid.uuid4().hex[:8]}_exit.jpg"
    stored_path = _store_image(src, dest_dir, filename)

    with get_conn() as conn:
        open_visit = conn.execute(
            "SELECT * FROM visits WHERE exit_time IS NULL ORDER BY entry_time DESC LIMIT 1"
        ).fetchone()

        if open_visit is None:
            cur = conn.execute(
                """INSERT INTO visits
                   (exit_time, exit_image_path, is_orphan_exit,
                    weight_exit_g, ammonia_peak_ppb, methane_peak_ppb)
                   VALUES (?, ?, TRUE, ?, ?, ?)""",
                (now.isoformat(), stored_path,
                 weight_exit_g, ammonia_peak_ppb, methane_peak_ppb),
            )
            orphan_id = cur.lastrowid
            return (
                f"⚠️  WARNING: No open visit found. "
                f"Orphan exit record #{orphan_id} created — human review required.\n"
                f"Exit image stored at: {stored_path}"
            )

        visit_id = open_visit["visit_id"]
        entry_image_abs = str(PROJECT_ROOT / open_visit["entry_image_path"])

        # Compute waste weight using the pre-entry baseline stored at entry time
        weight_pre_g = open_visit["weight_pre_g"]
        waste_weight_g = None
        if weight_exit_g is not None and weight_pre_g is not None:
            waste_weight_g = weight_exit_g - weight_pre_g

        # For peak gas readings, take the higher of entry and exit readings
        existing_ammonia = open_visit["ammonia_peak_ppb"]
        final_ammonia = (
            max(ammonia_peak_ppb, existing_ammonia)
            if ammonia_peak_ppb is not None and existing_ammonia is not None
            else (ammonia_peak_ppb if ammonia_peak_ppb is not None else existing_ammonia)
        )
        existing_methane = open_visit["methane_peak_ppb"]
        final_methane = (
            max(methane_peak_ppb, existing_methane)
            if methane_peak_ppb is not None and existing_methane is not None
            else (methane_peak_ppb if methane_peak_ppb is not None else existing_methane)
        )

        conn.execute(
            """UPDATE visits
               SET exit_time = ?, exit_image_path = ?,
                   weight_exit_g = ?, waste_weight_g = ?,
                   ammonia_peak_ppb = ?, methane_peak_ppb = ?
               WHERE visit_id = ?""",
            (now.isoformat(), stored_path,
             weight_exit_g, waste_weight_g,
             final_ammonia, final_methane,
             visit_id),
        )

        if weight_exit_g is not None or ammonia_peak_ppb is not None or methane_peak_ppb is not None:
            _log_sensor_events(conn, visit_id, "exit", now.isoformat(),
                               weight_g=weight_exit_g,
                               ammonia_peak_ppb=ammonia_peak_ppb,
                               methane_peak_ppb=methane_peak_ppb)

        # Fetch the complete visit row for the health prompt
        visit_row = conn.execute(
            "SELECT * FROM visits WHERE visit_id = ?", (visit_id,)
        ).fetchone()

    # Health analysis — compare entry and exit images, enriched with sensor data
    health_prompt = build_health_prompt(
        weight_pre_g=visit_row["weight_pre_g"],
        weight_entry_g=visit_row["weight_entry_g"],
        weight_exit_g=visit_row["weight_exit_g"],
        cat_weight_g=visit_row["cat_weight_g"],
        waste_weight_g=waste_weight_g,
        ammonia_peak_ppb=visit_row["ammonia_peak_ppb"],
        methane_peak_ppb=visit_row["methane_peak_ppb"],
    )
    health_text = _run_gpt4o_vision(health_prompt, entry_image_abs, str(src))
    is_anomalous, health_notes = parse_health_response(health_text)

    with get_conn() as conn:
        conn.execute(
            "UPDATE visits SET health_notes = ?, is_anomalous = ? WHERE visit_id = ?",
            (health_notes, is_anomalous, visit_id),
        )
        tentative = conn.execute(
            """SELECT c.name FROM visits v
               LEFT JOIN cats c ON v.tentative_cat_id = c.cat_id
               WHERE v.visit_id = ?""",
            (visit_id,),
        ).fetchone()

    cat_label = (tentative["name"] if tentative and tentative["name"] else "Unknown")
    flag = "⚠️  ANOMALY FLAGGED — veterinary review recommended" if is_anomalous else "No anomalies detected"

    sensor_parts = []
    if visit_row["cat_weight_g"] is not None:
        sensor_parts.append(f"cat weight {visit_row['cat_weight_g']:.0f} g")
    if waste_weight_g is not None:
        sensor_parts.append(f"waste {waste_weight_g:.0f} g")
    if visit_row["ammonia_peak_ppb"] is not None:
        sensor_parts.append(f"NH\u2083 {visit_row['ammonia_peak_ppb']:.0f} ppb")
    if visit_row["methane_peak_ppb"] is not None:
        sensor_parts.append(f"CH\u2084 {visit_row['methane_peak_ppb']:.0f} ppb")
    sensor_msg = f"Sensors: {', '.join(sensor_parts)}.\n" if sensor_parts else ""

    return (
        f"Visit #{visit_id} closed (tentative cat: {cat_label}).\n"
        f"Exit image: {stored_path}\n"
        f"{sensor_msg}"
        f"Health: {flag}\n\n"
        f"{health_notes}"
    )


@tool
def confirm_identity(visit_id: int, cat_name: str) -> str:
    """Confirm the true identity of a cat for a given visit.

    visit_id: the visit number to confirm.
    cat_name: the cat's actual name as recognised by the owner.
    The cat must already be registered. Permanently sets confirmed_cat_id.
    """
    init_db()
    with get_conn() as conn:
        cat = conn.execute(
            "SELECT cat_id FROM cats WHERE name = ?", (cat_name,)
        ).fetchone()
        if not cat:
            return (
                f"Error: no cat named '{cat_name}' in the database. "
                f"Register them first with register_cat_image."
            )
        visit = conn.execute(
            "SELECT visit_id FROM visits WHERE visit_id = ?", (visit_id,)
        ).fetchone()
        if not visit:
            return f"Error: visit #{visit_id} not found."

        conn.execute(
            "UPDATE visits SET confirmed_cat_id = ?, is_confirmed = TRUE WHERE visit_id = ?",
            (cat["cat_id"], visit_id),
        )
    return f"Visit #{visit_id} confirmed: cat is '{cat_name}'."


@tool
def get_visits_by_date(date_str: str) -> str:
    """List all litter box visits for a given date.

    date_str: date in YYYY-MM-DD format.
    """
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT v.visit_id, v.entry_time, v.exit_time,
                      tc.name AS tentative_name, cc.name AS confirmed_name,
                      v.is_confirmed, v.is_anomalous, v.similarity_score, v.is_orphan_exit
               FROM visits v
               LEFT JOIN cats tc ON v.tentative_cat_id = tc.cat_id
               LEFT JOIN cats cc ON v.confirmed_cat_id = cc.cat_id
               WHERE DATE(v.entry_time) = ? OR DATE(v.exit_time) = ?
               ORDER BY v.entry_time""",
            (date_str, date_str),
        ).fetchall()

    if not rows:
        return f"No visits found for {date_str}."

    lines = [f"Visits on {date_str} ({len(rows)} total):"]
    for r in rows:
        cat = (
            r["confirmed_name"]
            if r["is_confirmed"]
            else (f"~{r['tentative_name']}" if r["tentative_name"] else "Unknown")
        )
        confirmed_marker = "✓" if r["is_confirmed"] else "?"
        anomaly = " ⚠️" if r["is_anomalous"] else ""
        orphan = " [ORPHAN EXIT]" if r["is_orphan_exit"] else ""
        score = f" sim={r['similarity_score']:.2f}" if r["similarity_score"] else ""
        lines.append(
            f"  #{r['visit_id']} {cat} {confirmed_marker}{anomaly}{orphan}{score} | "
            f"in={r['entry_time'] or '—'}  out={r['exit_time'] or 'open'}"
        )
    return "\n".join(lines)


@tool
def get_visits_by_cat(cat_name: str) -> str:
    """List all visits for a cat, by confirmed or tentative name.

    cat_name: the cat's registered name.
    """
    init_db()
    with get_conn() as conn:
        cat = conn.execute(
            "SELECT cat_id FROM cats WHERE name = ?", (cat_name,)
        ).fetchone()
        if not cat:
            return f"No cat named '{cat_name}' found in the database."

        rows = conn.execute(
            """SELECT v.visit_id, v.entry_time, v.exit_time,
                      v.is_confirmed, v.is_anomalous, v.similarity_score
               FROM visits v
               WHERE v.confirmed_cat_id = ?
                  OR (v.tentative_cat_id = ? AND v.is_confirmed = FALSE)
               ORDER BY v.entry_time DESC""",
            (cat["cat_id"], cat["cat_id"]),
        ).fetchall()

    if not rows:
        return f"No visits found for '{cat_name}'."

    lines = [f"Visits for '{cat_name}' ({len(rows)} total):"]
    for r in rows:
        status = "confirmed" if r["is_confirmed"] else "tentative"
        anomaly = " ⚠️" if r["is_anomalous"] else ""
        score = f" sim={r['similarity_score']:.2f}" if r["similarity_score"] else ""
        lines.append(
            f"  #{r['visit_id']} [{status}]{anomaly}{score} | "
            f"{r['entry_time']} → {r['exit_time'] or 'open'}"
        )
    return "\n".join(lines)


@tool
def get_anomalous_visits() -> str:
    """List all visits flagged as potentially anomalous by the health analysis."""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT v.visit_id, v.entry_time,
                      tc.name AS tentative_name, cc.name AS confirmed_name,
                      v.is_confirmed, v.health_notes
               FROM visits v
               LEFT JOIN cats tc ON v.tentative_cat_id = tc.cat_id
               LEFT JOIN cats cc ON v.confirmed_cat_id = cc.cat_id
               WHERE v.is_anomalous = TRUE
               ORDER BY v.entry_time DESC"""
        ).fetchall()

    if not rows:
        return "No anomalous visits on record."

    lines = [f"{len(rows)} anomalous visit(s):"]
    for r in rows:
        cat = (
            r["confirmed_name"]
            if r["is_confirmed"]
            else (f"~{r['tentative_name']}" if r["tentative_name"] else "Unknown")
        )
        status = "confirmed" if r["is_confirmed"] else "tentative ID"
        lines.append(f"\n  Visit #{r['visit_id']} — {cat} ({status}) at {r['entry_time']}")
        if r["health_notes"]:
            snippet = r["health_notes"][:250].replace("\n", " ")
            lines.append(f"    {snippet}…")
    return "\n".join(lines)


@tool
def get_unconfirmed_visits() -> str:
    """List all visits that still have a tentative (unconfirmed) cat ID."""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT v.visit_id, v.entry_time, tc.name AS tentative_name,
                      v.similarity_score, v.is_anomalous
               FROM visits v
               LEFT JOIN cats tc ON v.tentative_cat_id = tc.cat_id
               WHERE v.is_confirmed = FALSE
               ORDER BY v.entry_time DESC"""
        ).fetchall()

    if not rows:
        return "No unconfirmed visits — all visits have been confirmed."

    lines = [f"{len(rows)} unconfirmed visit(s):"]
    for r in rows:
        cat = f"~{r['tentative_name']}" if r["tentative_name"] else "Unknown"
        score = f" (sim={r['similarity_score']:.2f})" if r["similarity_score"] else ""
        anomaly = " ⚠️" if r["is_anomalous"] else ""
        lines.append(f"  #{r['visit_id']}: {cat}{score}{anomaly} at {r['entry_time']}")
    return "\n".join(lines)


@tool
def get_visit_images(visit_id: int) -> str:
    """Return the stored image paths for a given visit.

    visit_id: the visit number.
    """
    init_db()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT entry_image_path, exit_image_path FROM visits WHERE visit_id = ?",
            (visit_id,),
        ).fetchone()

    if not row:
        return f"Visit #{visit_id} not found."

    return (
        f"Images for visit #{visit_id}:\n"
        f"  Entry: {row['entry_image_path'] or '— not recorded'}\n"
        f"  Exit:  {row['exit_image_path'] or '— visit still open or not recorded'}"
    )


@tool
def retroactive_recognition(cat_name: str, since_date: str) -> str:
    """Review unknown visits since a given date and retroactively identify those
    that belong to a newly registered cat.

    cat_name:   the cat's registered name (must already be in the database).
    since_date: date in YYYY-MM-DD format — the owner's acquisition date for
                this cat. Only visits on or after this date are reviewed.

    Runs the full two-stage CLIP + GPT-4o pipeline on every unknown visit in
    that window. Visits confirmed as this cat are marked is_confirmed=TRUE and
    assigned both tentative_cat_id and confirmed_cat_id.
    Returns a plain-text summary of all reviewed visits and their outcomes.
    """
    init_db()
    from datetime import date as _date

    # Validate date format
    try:
        _date.fromisoformat(since_date)
    except ValueError:
        return f"Error: '{since_date}' is not a valid date. Use YYYY-MM-DD format."

    with get_conn() as conn:
        cat = conn.execute(
            "SELECT cat_id FROM cats WHERE name = ?", (cat_name,)
        ).fetchone()
        if not cat:
            return (
                f"Error: no cat named '{cat_name}' found. "
                f"Register them first with register_cat_image."
            )
        cat_id = cat["cat_id"]

        unknown_visits = conn.execute(
            """SELECT visit_id, entry_image_path, entry_time
               FROM visits
               WHERE tentative_cat_id IS NULL
                 AND is_confirmed = FALSE
                 AND is_orphan_exit = FALSE
                 AND entry_image_path IS NOT NULL
                 AND DATE(entry_time) >= ?
               ORDER BY entry_time""",
            (since_date,),
        ).fetchall()

    if not unknown_visits:
        return (
            f"No unknown visits found on or after {since_date}. "
            f"Nothing to retroactively review for '{cat_name}'."
        )

    matched, skipped, unmatched = [], [], []

    for row in unknown_visits:
        visit_id = row["visit_id"]
        entry_image_abs = str(_abs(row["entry_image_path"]))

        if not Path(entry_image_abs).exists():
            skipped.append((visit_id, row["entry_time"], "image file missing"))
            continue

        identified_cat_id, identified_name, score, reasoning = _identify_cat(entry_image_abs)

        if identified_cat_id == cat_id:
            with get_conn() as conn:
                conn.execute(
                    """UPDATE visits
                       SET tentative_cat_id = ?,
                           confirmed_cat_id  = ?,
                           is_confirmed      = TRUE,
                           similarity_score  = ?
                       WHERE visit_id = ?""",
                    (cat_id, cat_id, score, visit_id),
                )
            matched.append((visit_id, row["entry_time"], score))
        else:
            unmatched.append((visit_id, row["entry_time"], identified_name, score))

    lines = [
        f"Retroactive recognition for '{cat_name}' since {since_date}:",
        f"  Visits reviewed : {len(unknown_visits)}",
        f"  Confirmed match : {len(matched)}",
        f"  No match        : {len(unmatched)}",
        f"  Skipped (error) : {len(skipped)}",
    ]

    if matched:
        lines.append("\nMatched visits (now confirmed):")
        for vid, ts, sim in matched:
            lines.append(f"  #{vid} at {ts}  sim={sim:.2f}")

    if unmatched:
        lines.append("\nVisits that did not match:")
        for vid, ts, other_name, sim in unmatched:
            best = f"best candidate '{other_name}'" if other_name else "no candidate"
            lines.append(f"  #{vid} at {ts}  {best} sim={sim:.2f}")

    if skipped:
        lines.append("\nSkipped visits:")
        for vid, ts, reason in skipped:
            lines.append(f"  #{vid} at {ts}  reason: {reason}")

    return "\n".join(lines)


@tool
def list_cats() -> str:
    """List all registered cats and their reference image counts."""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT c.name, COUNT(ci.image_id) AS image_count, c.created_at
               FROM cats c
               LEFT JOIN cat_images ci ON c.cat_id = ci.cat_id
               GROUP BY c.cat_id
               ORDER BY c.name"""
        ).fetchall()

    if not rows:
        return "No cats registered yet."

    lines = ["Registered cats:"]
    for r in rows:
        lines.append(
            f"  {r['name']}: {r['image_count']} reference image(s) "
            f"(registered {r['created_at'][:10]})"
        )
    return "\n".join(lines)


@tool
def eigen_report(cat_name: str) -> str:
    """Generate an eigenanalysis report for a cat's time-domain visit waveforms.

    Produces an HTML file with:
    - Overlay plot of all stored weight waveforms (zero-mean)
    - Data table showing each visit's date, mean weight (DC term),
      explained variance, residual, number of eigenvectors (N),
      anomaly level, and expansion coefficients for the N principal components

    The report is saved to output/eigen_<cat_name>.html.
    """
    from litterbox.eigen_query import generate_report, get_visit_summary

    init_db()

    summaries = get_visit_summary(cat_name)
    if not summaries:
        return f"No eigenanalysis data found for '{cat_name}'. Visits must be processed through the time-domain pipeline first."

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = cat_name.lower().replace(" ", "_")
    output_path = output_dir / f"eigen_{safe_name}.html"

    generate_report(cat_name, output_path=output_path)

    # Also return a text summary for the agent's response.
    scored = [s for s in summaries if s["eigen_ev"] is not None]
    lines = [
        f"Eigenanalysis report for {cat_name}: {len(summaries)} visit(s), "
        f"{len(scored)} scored.",
        f"Report saved to: {output_path}",
        "",
    ]
    for s in summaries:
        ev_str = f"{s['eigen_ev']:.4f}" if s['eigen_ev'] is not None else "unscored"
        dc_str = f"{s['dc_term']:.1f}" if s['dc_term'] is not None else "—"
        n_str = str(s['n_components']) if s['n_components'] is not None else "—"
        lines.append(
            f"  Visit {s['visit_number']} ({s['entry_time'][:16]}): "
            f"DC={dc_str}g  EV={ev_str}  N={n_str}  [{s['anomaly_level']}]"
        )

    return "\n".join(lines)


ALL_TOOLS = [
    register_cat_image,
    record_entry,
    record_exit,
    confirm_identity,
    retroactive_recognition,
    get_visits_by_date,
    get_visits_by_cat,
    get_anomalous_visits,
    get_unconfirmed_visits,
    get_visit_images,
    list_cats,
    eigen_report,
]
