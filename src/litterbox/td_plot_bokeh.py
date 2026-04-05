"""
td_plot_bokeh.py — Bokeh implementation of the PlotBackend interface
=====================================================================

This module is the **only** file that imports Bokeh.  Everything else in the
time-domain subsystem imports from ``td_plot`` (the abstract interface) and
calls ``get_plot_backend("bokeh")`` to obtain an instance of this class.

To replace Bokeh with a different library, create ``td_plot_<name>.py``,
implement the same ``Backend`` class, and change the ``get_plot_backend()``
call-site argument.  Zero other files need to change.

Output format
-------------
Both plot methods produce a self-contained HTML file when ``output_path`` is
provided.  The HTML includes all Bokeh JavaScript inline so it can be opened
in any browser without an internet connection.  When ``output_path`` is
``None`` the plot is opened in the default browser via ``bokeh.plotting.show``.

Visual layout
-------------
* ``plot_channels`` — one figure per channel, stacked vertically and linked
  on the x-axis so panning/zooming one panel moves all of them.
* ``plot_similarity_dataframe`` — a single figure with one coloured line per
  registered cat, a legend, and a horizontal reference line at the entry
  threshold (0.70 by default).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Legend,
    LegendItem,
    Span,
)
from bokeh.palettes import Category10  # up to 10 distinct colours
from bokeh.plotting import figure, output_file, save, show

from litterbox.td_plot import PlotBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Bokeh needs numeric x-values (milliseconds since Unix epoch) when the axis
# type is "datetime".  This function converts Python datetime objects.
def _to_ms(dt: datetime) -> float:
    """Convert a Python datetime to milliseconds since the Unix epoch."""
    return dt.timestamp() * 1_000.0


def _none_to_nan(values: list) -> list[float]:
    """Replace None with float('nan') so Bokeh renders gaps rather than zeros."""
    import math
    return [float("nan") if v is None else float(v) for v in values]


def _colour_cycle(n: int) -> list[str]:
    """Return n distinct hex colours, cycling through the Category10 palette."""
    palette = Category10[max(n, 3)]  # Category10 requires at least 3 colours
    return [palette[i % len(palette)] for i in range(n)]


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class Backend(PlotBackend):
    """Bokeh implementation of ``PlotBackend``.

    Instantiated by ``get_plot_backend("bokeh")``.  Should not be imported
    directly — always go through the factory so the backend is swappable.
    """

    # Default figure dimensions.
    _PLOT_WIDTH: int  = 900
    _PLOT_HEIGHT: int = 220   # per-channel panel height
    _SIM_HEIGHT: int  = 400   # similarity plot height

    # Threshold reference line drawn on the similarity plot.
    _DEFAULT_SIM_THRESHOLD: float = 0.70

    def plot_channels(
        self,
        timestamps: list,
        channels: dict[str, list],
        title: str = "Rolling Buffer",
        output_path: Optional[Path] = None,
    ) -> None:
        """Render one vertically-stacked panel per channel, x-axes linked.

        Parameters
        ----------
        timestamps:
            List of ``datetime`` objects.  Bokeh datetime axis requires these
            to be timezone-naive or consistently timezone-aware.
        channels:
            Dict mapping channel name → list of numeric values (same length
            as ``timestamps``).  ``None`` values are converted to NaN so
            Bokeh draws a gap instead of connecting through zero.
        title:
            Overall title shown above the top panel.
        output_path:
            If provided, save an HTML file to this path.  Otherwise open a
            browser tab.
        """
        if not channels:
            raise ValueError("plot_channels: 'channels' dict is empty.")
        if not timestamps:
            raise ValueError("plot_channels: 'timestamps' list is empty.")

        x_ms = [_to_ms(t) for t in timestamps]

        figures = []
        shared_x_range = None  # first figure establishes the range

        channel_names = list(channels.keys())
        for i, name in enumerate(channel_names):
            values = _none_to_nan(channels[name])

            # Only the top panel shows the overall title; the others show
            # just the channel name so the stacked layout is compact.
            panel_title = title if i == 0 else name

            # Bokeh rejects x_range=None — omit the kwarg entirely for the
            # first panel and pass the Range1d object for subsequent panels.
            fig_kwargs: dict = dict(
                title=panel_title,
                x_axis_type="datetime",
                width=self._PLOT_WIDTH,
                height=self._PLOT_HEIGHT,
                tools="pan,box_zoom,wheel_zoom,reset,save",
            )
            if shared_x_range is not None:
                fig_kwargs["x_range"] = shared_x_range

            p = figure(**fig_kwargs)

            source = ColumnDataSource({"x": x_ms, "y": values,
                                       "ts": [str(t) for t in timestamps]})

            p.line("x", "y", source=source, line_width=2,
                   color="#1f77b4", line_alpha=0.85)
            p.scatter("x", "y", source=source, size=4,
                      color="#1f77b4", alpha=0.6)

            p.add_tools(HoverTool(
                tooltips=[("time", "@ts"), (name, "@y{0.00}")],
                mode="vline",
            ))

            p.yaxis.axis_label = name
            p.xaxis.axis_label = "Time" if i == len(channel_names) - 1 else ""

            if shared_x_range is None:
                shared_x_range = p.x_range   # link subsequent panels

            figures.append(p)

        layout = column(*figures)

        if output_path is not None:
            output_file(str(output_path), title=title)
            save(layout)
        else:
            show(layout)

    def plot_similarity_dataframe(
        self,
        df: pd.DataFrame,
        title: str = "Cat Similarity Scores",
        output_path: Optional[Path] = None,
        threshold: float = _DEFAULT_SIM_THRESHOLD,
    ) -> None:
        """Render overlapping lines for every cat's similarity score over time.

        Each column in ``df`` is drawn as a separate coloured line.  A
        horizontal dashed reference line marks the entry-detection threshold
        so the owner can see how close each cat came to triggering recognition.

        Parameters
        ----------
        df:
            A timestamp-indexed pandas DataFrame, one column per registered
            cat, ``NaN`` where no camera frame was available.  Produced by
            ``RollingBuffer.to_dataframe(channel_prefix="similarity_")``.
        title:
            Plot title.
        output_path:
            Save path, or ``None`` to open a browser tab.
        threshold:
            Y-value for the horizontal reference line.  Defaults to 0.70
            (the value in td_config.json).  Pass ``None`` to omit the line.
        """
        if df.empty:
            raise ValueError("plot_similarity_dataframe: DataFrame is empty.")

        cat_names = list(df.columns)
        colours   = _colour_cycle(len(cat_names))

        p = figure(
            title=title,
            x_axis_type="datetime",
            width=self._PLOT_WIDTH,
            height=self._SIM_HEIGHT,
            tools="pan,box_zoom,wheel_zoom,reset,save",
            y_range=(0.0, 1.05),
        )

        legend_items: list[LegendItem] = []

        for cat, colour in zip(cat_names, colours):
            col_values = df[cat].tolist()
            x_ms       = [_to_ms(t) for t in df.index]
            y_vals     = [float("nan") if pd.isna(v) else float(v)
                          for v in col_values]
            ts_strs    = [str(t) for t in df.index]

            source = ColumnDataSource({"x": x_ms, "y": y_vals, "ts": ts_strs})

            line   = p.line("x", "y", source=source, line_width=2,
                            color=colour, line_alpha=0.85)
            circle = p.scatter("x", "y", source=source, size=5,
                              color=colour, alpha=0.6)

            p.add_tools(HoverTool(
                renderers=[line],
                tooltips=[("time", "@ts"), (cat, "@y{0.000}")],
                mode="vline",
            ))

            legend_items.append(LegendItem(label=cat, renderers=[line, circle]))

        # Horizontal reference line for the entry-detection threshold.
        if threshold is not None:
            threshold_line = Span(
                location=threshold,
                dimension="width",
                line_color="red",
                line_dash="dashed",
                line_width=1.5,
                line_alpha=0.7,
            )
            p.add_layout(threshold_line)
            # Annotate the threshold line in the legend.
            legend_items.append(
                LegendItem(
                    label=f"entry threshold ({threshold:.2f})",
                    renderers=[p.line([], [], line_color="red",
                                     line_dash="dashed", line_width=1.5)],
                )
            )

        legend = Legend(items=legend_items, location="top_right")
        p.add_layout(legend)
        p.legend.click_policy = "hide"   # clicking a legend entry toggles its line

        p.yaxis.axis_label = "CLIP similarity score"
        p.xaxis.axis_label = "Time"
        p.y_range.start    = 0.0
        p.y_range.end      = 1.05

        if output_path is not None:
            output_file(str(output_path), title=title)
            save(p)
        else:
            show(p)
