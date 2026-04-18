"""
eigen_query.py — Query and reporting for eigenanalysis results
===============================================================

Provides functions to retrieve per-cat eigenanalysis data and generate
an HTML report with:

- Waveform overlay plots (inline SVG — no external JS dependencies)
- Summary data table: visit date/time, DC term (mean weight), explained
  variance, number of signal eigenvectors (N), and expansion coefficients
  for the N principal components

All functions read from the ``eigen_waveforms``, ``eigen_models``,
``td_visits``, and ``cats`` tables.

Usage from Python::

    from litterbox.eigen_query import get_visit_summary, generate_report

    df = get_visit_summary("Luna")
    html = generate_report("Luna", output_path="output/luna_eigen.html")

Usage as a LangChain tool::

    The ``eigen_report`` tool in ``tools.py`` wraps ``generate_report()``
    so the agent can produce the report on request.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from litterbox.db import get_conn, init_db


# ===========================================================================
# Query functions
# ===========================================================================

def get_visit_summary(
    cat_name: str,
    channel: str = "weight_g",
) -> list[dict]:
    """Return a per-visit summary for a cat's eigenanalysis results.

    Each dict contains:
        visit_number, entry_time, exit_time, dc_term, eigen_ev,
        eigen_residual, n_components, model_type, anomaly_level,
        signal_coefficients (list of first N coefficients)

    Returns an empty list if the cat has no scored visits.
    """
    init_db()

    with get_conn() as conn:
        rows = conn.execute(
            """SELECT
                 ew.waveform_id,
                 tv.entry_time,
                 tv.exit_time,
                 ew.dc_term,
                 ew.eigen_ev,
                 ew.eigen_residual,
                 ew.coefficients_json,
                 ew.nan_fraction,
                 em.n_components,
                 em.regularized,
                 em.cat_id AS model_cat_id
               FROM eigen_waveforms ew
               JOIN td_visits tv ON ew.td_visit_id = tv.td_visit_id
               JOIN cats c ON ew.cat_id = c.cat_id
               LEFT JOIN eigen_models em ON ew.model_id = em.model_id
               WHERE c.name = ? AND ew.channel = ?
               ORDER BY tv.entry_time""",
            (cat_name, channel),
        ).fetchall()

    summaries = []
    for i, row in enumerate(rows):
        n_comp = row["n_components"] if row["n_components"] else None

        # Extract signal-subspace coefficients (first N).
        signal_coeffs = None
        if row["coefficients_json"] and n_comp:
            all_coeffs = json.loads(row["coefficients_json"])
            signal_coeffs = all_coeffs[:n_comp]

        model_type = None
        if row["model_cat_id"] is not None:
            model_type = "per_cat"
        elif row["eigen_ev"] is not None:
            model_type = "pooled"

        anomaly_level = _classify_ev(row["eigen_ev"]) if row["eigen_ev"] is not None else "unscored"

        summaries.append({
            "visit_number": i + 1,
            "entry_time": row["entry_time"],
            "exit_time": row["exit_time"],
            "dc_term": row["dc_term"],
            "eigen_ev": row["eigen_ev"],
            "eigen_residual": row["eigen_residual"],
            "n_components": n_comp,
            "model_type": model_type,
            "anomaly_level": anomaly_level,
            "signal_coefficients": signal_coeffs,
        })

    return summaries


def get_waveforms(
    cat_name: str,
    channel: str = "weight_g",
) -> tuple[list[np.ndarray], list[str]]:
    """Return all stored waveforms for a cat as numpy arrays.

    Returns
    -------
    (waveforms, timestamps)
        waveforms: list of 1-D numpy arrays (zero-mean, length L)
        timestamps: list of entry_time strings (ISO format)
    """
    init_db()

    with get_conn() as conn:
        rows = conn.execute(
            """SELECT ew.vector_json, tv.entry_time
               FROM eigen_waveforms ew
               JOIN td_visits tv ON ew.td_visit_id = tv.td_visit_id
               JOIN cats c ON ew.cat_id = c.cat_id
               WHERE c.name = ? AND ew.channel = ?
               ORDER BY tv.entry_time""",
            (cat_name, channel),
        ).fetchall()

    waveforms = [np.array(json.loads(r["vector_json"])) for r in rows]
    timestamps = [r["entry_time"] for r in rows]
    return waveforms, timestamps


def get_model(
    cat_name: str,
    channel: str = "weight_g",
) -> Optional[dict]:
    """Return the most recent eigenmodel for a cat (or pooled if per-cat doesn't exist).

    Returns a dict with eigenvalues, eigenvectors (as numpy arrays),
    n_components, n_waveforms, regularized.  Returns None if no model exists.
    """
    init_db()

    with get_conn() as conn:
        # Try per-cat first.
        row = conn.execute(
            """SELECT em.*
               FROM eigen_models em
               JOIN cats c ON em.cat_id = c.cat_id
               WHERE c.name = ? AND em.channel = ?
               ORDER BY em.computed_at DESC LIMIT 1""",
            (cat_name, channel),
        ).fetchone()

        if row is None:
            # Fall back to pooled model.
            row = conn.execute(
                """SELECT * FROM eigen_models
                   WHERE cat_id IS NULL AND channel = ?
                   ORDER BY computed_at DESC LIMIT 1""",
                (channel,),
            ).fetchone()

    if row is None:
        return None

    return {
        "model_id": row["model_id"],
        "cat_id": row["cat_id"],
        "eigenvalues": np.array(json.loads(row["eigenvalues_json"])),
        "eigenvectors": np.array(json.loads(row["eigenvectors_json"])),
        "n_components": row["n_components"],
        "n_waveforms": row["n_waveforms"],
        "regularized": bool(row["regularized"]),
        "computed_at": row["computed_at"],
    }


# ===========================================================================
# HTML report generator
# ===========================================================================

def generate_report(
    cat_name: str,
    channel: str = "weight_g",
    output_path: Optional[str | Path] = None,
) -> str:
    """Generate an HTML report for a cat's eigenanalysis history.

    The report contains:
    - An inline SVG plot of all waveforms overlaid
    - A data table with visit date, DC term, EV, N, anomaly level,
      and signal-subspace expansion coefficients

    Parameters
    ----------
    cat_name:
        Name of the cat to report on.
    channel:
        Sensor channel (default ``"weight_g"``).
    output_path:
        If provided, write the HTML to this file and return the path.
        If None, return the HTML string directly.

    Returns
    -------
    str
        The HTML content, or the output file path if ``output_path`` was given.
    """
    summaries = get_visit_summary(cat_name, channel)
    waveforms, timestamps = get_waveforms(cat_name, channel)

    html = _build_html(cat_name, channel, summaries, waveforms, timestamps)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return str(path)

    return html


# ===========================================================================
# Internal — HTML building
# ===========================================================================

def _classify_ev(ev: Optional[float]) -> str:
    """Classify an EV value into an anomaly level string."""
    if ev is None:
        return "unscored"
    if ev >= 0.90:
        return "normal"
    elif ev >= 0.70:
        return "mild"
    elif ev >= 0.40:
        return "significant"
    else:
        return "major"


_ANOMALY_COLORS = {
    "normal": "#2e7d32",
    "mild": "#f9a825",
    "significant": "#e65100",
    "major": "#b71c1c",
    "unscored": "#757575",
}

# Waveform line colors — cycle through for multiple visits.
_LINE_COLORS = [
    "#1976d2", "#d32f2f", "#388e3c", "#7b1fa2",
    "#f57c00", "#0097a7", "#c2185b", "#455a64",
    "#6d4c41", "#00838f", "#ad1457", "#4527a0",
]


def _build_html(
    cat_name: str,
    channel: str,
    summaries: list[dict],
    waveforms: list[np.ndarray],
    timestamps: list[str],
) -> str:
    """Assemble the full HTML document."""
    svg = _build_waveform_svg(waveforms, timestamps, cat_name, channel)
    table = _build_data_table(summaries)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eigenanalysis Report — {cat_name}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1200px; margin: 40px auto; padding: 0 20px;
    color: #222; background: #fafafa;
  }}
  h1 {{ color: #1565c0; margin-bottom: 4px; }}
  h2 {{ color: #424242; margin-top: 32px; }}
  .subtitle {{ color: #757575; font-size: 14px; margin-bottom: 24px; }}
  .no-data {{ color: #757575; font-style: italic; padding: 20px; }}
  svg {{ display: block; margin: 16px 0; background: #fff;
         border: 1px solid #e0e0e0; border-radius: 4px; }}
  table {{
    border-collapse: collapse; width: 100%; font-size: 13px;
    background: #fff; border: 1px solid #e0e0e0; border-radius: 4px;
  }}
  th {{
    background: #e3f2fd; padding: 8px 10px; text-align: left;
    border-bottom: 2px solid #90caf9; white-space: nowrap;
  }}
  td {{
    padding: 6px 10px; border-bottom: 1px solid #eeeeee;
    vertical-align: top;
  }}
  tr:hover td {{ background: #f5f5f5; }}
  .ev-normal {{ color: #2e7d32; font-weight: 600; }}
  .ev-mild {{ color: #f9a825; font-weight: 600; }}
  .ev-significant {{ color: #e65100; font-weight: 600; }}
  .ev-major {{ color: #b71c1c; font-weight: 600; }}
  .ev-unscored {{ color: #757575; }}
  .coeffs {{ font-family: "SF Mono", Monaco, Consolas, monospace;
             font-size: 11px; color: #555; max-width: 400px;
             word-break: break-all; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0 16px; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 12px; }}
  .legend-swatch {{ width: 20px; height: 3px; border-radius: 1px; }}
</style>
</head>
<body>
<h1>Eigenanalysis Report — {cat_name}</h1>
<p class="subtitle">Channel: {channel} &middot; {len(summaries)} visit(s)</p>

<h2>Waveforms (zero-mean)</h2>
{svg}

<h2>Visit Summary</h2>
{table}

</body>
</html>"""


def _build_waveform_svg(
    waveforms: list[np.ndarray],
    timestamps: list[str],
    cat_name: str,
    channel: str,
) -> str:
    """Build an inline SVG plot of all waveforms overlaid."""
    if not waveforms:
        return '<p class="no-data">No waveforms stored yet.</p>'

    width = 800
    height = 320
    pad_left = 60
    pad_right = 20
    pad_top = 20
    pad_bottom = 40
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    # Compute global y range across all waveforms.
    all_vals = np.concatenate(waveforms)
    y_min = float(np.min(all_vals))
    y_max = float(np.max(all_vals))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    y_margin = (y_max - y_min) * 0.08
    y_min -= y_margin
    y_max += y_margin

    L = len(waveforms[0])

    def x_px(i: int) -> float:
        return pad_left + (i / (L - 1)) * plot_w if L > 1 else pad_left + plot_w / 2

    def y_px(v: float) -> float:
        return pad_top + (1 - (v - y_min) / (y_max - y_min)) * plot_h

    lines = []
    for idx, wf in enumerate(waveforms):
        color = _LINE_COLORS[idx % len(_LINE_COLORS)]
        points = " ".join(f"{x_px(i):.1f},{y_px(v):.1f}" for i, v in enumerate(wf))
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="1.5" opacity="0.7"/>'
        )

    # Y-axis ticks.
    y_ticks = np.linspace(y_min, y_max, 6)
    tick_lines = []
    for yt in y_ticks:
        yp = y_px(yt)
        tick_lines.append(
            f'<line x1="{pad_left}" y1="{yp:.1f}" x2="{pad_left + plot_w}" '
            f'y2="{yp:.1f}" stroke="#e0e0e0" stroke-width="0.5"/>'
        )
        tick_lines.append(
            f'<text x="{pad_left - 6}" y="{yp + 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="#757575">{yt:.1f}</text>'
        )

    # X-axis label.
    x_label_marks = []
    for frac, label in [(0, "0"), (0.25, "16"), (0.5, "32"), (0.75, "48"), (1.0, "63")]:
        xp = pad_left + frac * plot_w
        x_label_marks.append(
            f'<text x="{xp:.1f}" y="{pad_top + plot_h + 20}" text-anchor="middle" '
            f'font-size="10" fill="#757575">{label}</text>'
        )

    # Legend.
    legend_items = []
    for idx, ts in enumerate(timestamps):
        color = _LINE_COLORS[idx % len(_LINE_COLORS)]
        short_ts = ts[:16] if len(ts) > 16 else ts
        legend_items.append(
            f'<span class="legend-item">'
            f'<span class="legend-swatch" style="background:{color}"></span>'
            f'Visit {idx + 1} ({short_ts})</span>'
        )

    legend_html = f'<div class="legend">{"".join(legend_items)}</div>'

    svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect x="{pad_left}" y="{pad_top}" width="{plot_w}" height="{plot_h}"
        fill="#fafafa" stroke="#bdbdbd" stroke-width="0.5"/>
  {"".join(tick_lines)}
  {"".join(x_label_marks)}
  <text x="{pad_left + plot_w / 2}" y="{height - 4}" text-anchor="middle"
        font-size="11" fill="#757575">Sample index</text>
  <text x="14" y="{pad_top + plot_h / 2}" text-anchor="middle"
        font-size="11" fill="#757575" transform="rotate(-90,14,{pad_top + plot_h / 2})">Amplitude (zero-mean)</text>
  {"".join(lines)}
</svg>
{legend_html}"""

    return svg


def _build_data_table(summaries: list[dict]) -> str:
    """Build the HTML data table."""
    if not summaries:
        return '<p class="no-data">No visit data available.</p>'

    rows_html = []
    for s in summaries:
        ev_val = f"{s['eigen_ev']:.4f}" if s['eigen_ev'] is not None else "—"
        ev_class = f"ev-{s['anomaly_level']}"
        dc_val = f"{s['dc_term']:.1f}" if s['dc_term'] is not None else "—"
        n_val = str(s['n_components']) if s['n_components'] is not None else "—"
        residual_val = f"{s['eigen_residual']:.4f}" if s['eigen_residual'] is not None else "—"

        # Format signal coefficients.
        if s['signal_coefficients']:
            coeffs_str = ", ".join(f"{c:.3f}" for c in s['signal_coefficients'])
        else:
            coeffs_str = "—"

        entry_short = s['entry_time'][:19] if s['entry_time'] else "—"

        rows_html.append(f"""<tr>
  <td>{s['visit_number']}</td>
  <td>{entry_short}</td>
  <td>{dc_val}</td>
  <td class="{ev_class}">{ev_val}</td>
  <td>{residual_val}</td>
  <td>{n_val}</td>
  <td class="{ev_class}">{s['anomaly_level']}</td>
  <td class="coeffs">{coeffs_str}</td>
</tr>""")

    return f"""<table>
<thead><tr>
  <th>#</th>
  <th>Entry Time</th>
  <th>DC (mean g)</th>
  <th>Explained Var</th>
  <th>Residual</th>
  <th>N</th>
  <th>Anomaly</th>
  <th>Signal Coefficients (c<sub>1</sub>…c<sub>N</sub>)</th>
</tr></thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>"""
