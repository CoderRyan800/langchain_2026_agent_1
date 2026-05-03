"""
history_plot.py — Per-cat visit-history plots
==============================================

Generates a self-contained HTML report with three stacked time-series
sub-plots for one cat:

  1. Cat weight (g)
  2. NH₃ peak (ppb)
  3. CH₄ peak (ppb)

Anomalous visits are rendered as red ``×`` markers; normal visits as small
blue dots. Hover tooltips show the visit ID, timestamp, raw reading,
z-score, tier, and which model produced the score (per_cat / pooled /
insufficient_data). Reference lines on the gas channels show the alarm
thresholds back-projected from log-space (median + 2σ and median + 3σ
of the cat's robust log-Gaussian fit) — when there's enough history.

This module is distinct from ``td_plot_bokeh.py``: that one plots rolling
buffer waveforms; this one plots accumulated visits over days/weeks.
Different scope, different abstraction.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.plotting import figure, output_file, save

from litterbox.db import get_conn, init_db
from litterbox.gas_anomaly import _fit_log_gaussian


_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "output"
_PLOT_WIDTH = 920
_PANEL_HEIGHT = 250

# Subset of common visit columns we read for plotting.
_QUERY = """
    SELECT v.visit_id, v.entry_time, v.is_anomalous,
           v.cat_weight_g,
           v.ammonia_peak_ppb, v.methane_peak_ppb,
           v.ammonia_z_score,  v.methane_z_score,
           v.gas_anomaly_tier, v.gas_anomaly_model_used
      FROM visits v
     WHERE (v.confirmed_cat_id = ?
            OR (v.tentative_cat_id = ? AND v.is_confirmed = FALSE))
       AND v.entry_time >= ?
     ORDER BY v.entry_time
"""


def _parse_ts(ts: str) -> datetime:
    """Parse the entry_time string into a (naive) datetime for Bokeh."""
    # Strip any timezone suffix — Bokeh's datetime axis is naive-friendly,
    # and our DB stores ISO 8601 with optional ``+00:00``.
    if ts.endswith("Z"):
        ts = ts[:-1]
    if "+" in ts[10:]:
        ts = ts.rsplit("+", 1)[0]
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        # Fallback for older rows that may have non-ISO format.
        return datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")


def _split_anomalous(rows: list, value_col: str) -> tuple[dict, dict]:
    """Partition rows into (normal, anomalous) ColumnDataSource dicts.

    Both dicts share the same keys so we can pass them to glyph methods
    with identical column names. Rows whose ``value_col`` is ``None`` are
    dropped — there's nothing to plot for them on this channel.
    """
    keys = ("x", "y", "visit_id", "ts", "z", "tier", "model")
    normal = {k: [] for k in keys}
    anomalous = {k: [] for k in keys}

    z_col = "ammonia_z_score" if value_col == "ammonia_peak_ppb" else \
            "methane_z_score" if value_col == "methane_peak_ppb" else None

    for r in rows:
        v = r[value_col]
        if v is None:
            continue
        bucket = anomalous if r["is_anomalous"] else normal
        bucket["x"].append(_parse_ts(r["entry_time"]))
        bucket["y"].append(float(v))
        bucket["visit_id"].append(int(r["visit_id"]))
        bucket["ts"].append(r["entry_time"])
        bucket["z"].append(
            f"{r[z_col]:+.2f}" if z_col and r[z_col] is not None else "—"
        )
        bucket["tier"].append(r["gas_anomaly_tier"] or "—")
        bucket["model"].append(r["gas_anomaly_model_used"] or "—")

    return normal, anomalous


def _gas_threshold_levels(
    values: list[float],
) -> Optional[tuple[float, float, float]]:
    """Return (median_ppb, mild_ppb, significant_ppb) or ``None``.

    Computes the same robust log-Gaussian fit the gas anomaly detector uses,
    then back-projects the location and the ``median ± kσ`` bands from log
    space via ``expm1``. Returns ``None`` when the fit is degenerate (too
    few points, or all values constant).

    The median is the cat's "normal" anchor: a solid line on the plot
    showing where the bulk of this cat's readings sit. The mild /
    significant lines mark the alarm tiers.
    """
    clean = [v for v in values if v is not None]
    model = _fit_log_gaussian(clean)
    if model is None:
        return None
    median, sigma = model
    median_ppb = math.expm1(median)
    mild = math.expm1(median + 2.0 * sigma)
    significant = math.expm1(median + 3.0 * sigma)
    return median_ppb, mild, significant


def _build_panel(
    rows: list,
    value_col: str,
    title: str,
    y_label: str,
    show_reference_lines: bool,
    shared_x_range,
    log_y_axis: bool = False,
):
    """Render one stacked panel for the given channel.

    Gas channels pass ``log_y_axis=True`` and ``show_reference_lines=True``.
    The detector works in ``log1p`` space, so a log y-axis matches the
    geometry the model actually sees: median and the alarm tiers become
    roughly evenly spaced bands rather than getting squashed against the
    bottom by an order-of-magnitude alarm threshold. A solid green line
    at the median anchors the cat's "normal" position; orange / red
    dashed lines mark the mild / significant alarm tiers.
    """
    normal, anomalous = _split_anomalous(rows, value_col)

    fig_kwargs: dict = dict(
        title=title,
        x_axis_type="datetime",
        width=_PLOT_WIDTH,
        height=_PANEL_HEIGHT,
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    if shared_x_range is not None:
        fig_kwargs["x_range"] = shared_x_range
    if log_y_axis:
        fig_kwargs["y_axis_type"] = "log"
    p = figure(**fig_kwargs)
    p.yaxis.axis_label = y_label

    # Reference lines (only on gas channels with enough history).
    if show_reference_lines:
        all_values = [r[value_col] for r in rows if r[value_col] is not None]
        levels = _gas_threshold_levels(all_values)
        if levels is not None:
            median_ppb, mild_ppb, significant_ppb = levels
            # Median: solid green, anchors the cat's "normal" position.
            p.add_layout(Span(
                location=median_ppb, dimension="width",
                line_color="#2ca02c", line_alpha=0.55, line_width=1,
            ))
            # Mild alarm threshold (z=2): orange dashed.
            p.add_layout(Span(
                location=mild_ppb, dimension="width",
                line_color="#ff8c00", line_dash="dashed", line_alpha=0.6,
                line_width=1,
            ))
            # Significant alarm threshold (z=3): red dashed.
            p.add_layout(Span(
                location=significant_ppb, dimension="width",
                line_color="#d62728", line_dash="dashed", line_alpha=0.7,
                line_width=1,
            ))

    # Tooltip identical for both glyphs.
    tooltip = [
        ("visit", "#@visit_id"),
        ("when",  "@ts"),
        (y_label, "@y{0.00}"),
        ("z",     "@z"),
        ("tier",  "@tier"),
        ("model", "@model"),
    ]

    if normal["x"]:
        src_normal = ColumnDataSource(normal)
        p.scatter(
            "x", "y", source=src_normal,
            size=6, color="#1f77b4", alpha=0.75,
            legend_label="normal",
        )

    if anomalous["x"]:
        src_anom = ColumnDataSource(anomalous)
        p.scatter(
            "x", "y", source=src_anom,
            size=12, color="#d62728", alpha=0.95,
            marker="x", line_width=3,
            legend_label="anomalous",
        )

    p.add_tools(HoverTool(tooltips=tooltip, mode="mouse"))

    # Only style the legend when at least one glyph layer added it.
    if normal["x"] or anomalous["x"]:
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = "9pt"

    return p


def plot_cat_history(
    cat_name: str,
    days: int = 90,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate a Bokeh HTML history plot for one cat.

    Parameters
    ----------
    cat_name:
        Registered cat name.
    days:
        How many days of history to plot, counting back from now.
        Defaults to 90.
    output_path:
        Where to write the HTML file. Defaults to
        ``output/cat_history_<name>.html`` (relative to the project root).
        ``output/`` is gitignored.

    Returns
    -------
    Path
        The absolute path to the generated HTML file.

    Raises
    ------
    ValueError
        If ``cat_name`` does not match a registered cat, or if no visits
        fall in the requested window. The agent's wrapper tool catches
        these and converts them into a friendly error string.
    """
    init_db()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff.replace(tzinfo=None).isoformat()

    with get_conn() as conn:
        cat = conn.execute(
            "SELECT cat_id FROM cats WHERE name = ?", (cat_name,)
        ).fetchone()
        if not cat:
            raise ValueError(f"Cat {cat_name!r} is not registered.")
        rows = conn.execute(
            _QUERY, (cat["cat_id"], cat["cat_id"], cutoff_iso)
        ).fetchall()

    if not rows:
        raise ValueError(
            f"No visits found for {cat_name!r} in the last {days} days."
        )

    # Build panels — first one establishes the shared x range.
    weight_panel = _build_panel(
        rows, "cat_weight_g",
        title=f"{cat_name} — last {days} days",
        y_label="cat weight (g)",
        show_reference_lines=False,
        shared_x_range=None,
    )
    nh3_panel = _build_panel(
        rows, "ammonia_peak_ppb",
        title="NH₃ peak (ppb, log scale) — green: median, orange: mild (z=2), red: significant (z=3)",
        y_label="NH₃ ppb (log)",
        show_reference_lines=True,
        shared_x_range=weight_panel.x_range,
        log_y_axis=True,
    )
    ch4_panel = _build_panel(
        rows, "methane_peak_ppb",
        title="CH₄ peak (ppb, log scale) — green: median, orange: mild (z=2), red: significant (z=3)",
        y_label="CH₄ ppb (log)",
        show_reference_lines=True,
        shared_x_range=weight_panel.x_range,
        log_y_axis=True,
    )

    if output_path is None:
        _DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = _DEFAULT_OUTPUT_DIR / f"cat_history_{cat_name}.html"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_file(str(output_path), title=f"{cat_name} history")
    save(column(weight_panel, nh3_panel, ch4_panel))
    return output_path
