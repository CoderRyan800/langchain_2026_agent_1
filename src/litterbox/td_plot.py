"""
td_plot.py — Abstract plotting interface for the time-domain subsystem
=======================================================================

This module defines the **only** import boundary between the time-domain
system and any plotting library.  All other modules (``time_buffer``,
``sensor_collector``, ``visit_trigger``, ``visit_analyser``) are completely
unaware of Bokeh, Plotly, Matplotlib, or any other visualisation library.

How to swap the backend
-----------------------
1. Create a new file ``td_plot_<name>.py`` (e.g. ``td_plot_plotly.py``).
2. Inside it, define a class named ``Backend`` that inherits from
   ``PlotBackend`` and implements all abstract methods.
3. Call ``get_plot_backend("plotly")`` — no other code changes anywhere.

That is the only contract: one class named ``Backend``, one module named
``td_plot_<name>.py``.

Current backends
----------------
* ``"bokeh"``  — ``td_plot_bokeh.py``  (default, ships with this repo)

Usage
-----
::

    from litterbox.td_plot import get_plot_backend

    backend = get_plot_backend("bokeh")          # or "plotly", "matplotlib", …

    backend.plot_channels(
        timestamps = buf.get_timestamps(),
        channels   = {"weight_g": buf.get_channel("weight_g"),
                      "ammonia_ppb": buf.get_channel("ammonia_ppb")},
        title      = "Live sensor feed",
        output_path= Path("output/live.html"),
    )

    df = buf.to_dataframe(channel_prefix="similarity_")
    backend.plot_similarity_dataframe(
        df          = df,
        title       = "Cat similarity scores",
        output_path = Path("output/similarity.html"),
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import Optional

import pandas as pd


# ===========================================================================
# Abstract interface
# ===========================================================================

class PlotBackend(ABC):
    """Abstract base class that every plot backend must implement.

    A backend is responsible for:

    * ``plot_channels`` — general-purpose multi-line plot of scalar channels
      against time.  Suitable for weight, gas sensors, chip-ID frequency, etc.

    * ``plot_similarity_dataframe`` — specialised K-line plot for the M×K
      similarity DataFrame produced by ``RollingBuffer.to_dataframe()``.

    Both methods save to a file when ``output_path`` is provided, or open an
    interactive window / notebook display when it is ``None``.
    """

    @abstractmethod
    def plot_channels(
        self,
        timestamps: list,
        channels: dict[str, list],
        title: str = "Rolling Buffer",
        output_path: Optional[Path] = None,
    ) -> None:
        """Plot one or more scalar channels against time.

        Parameters
        ----------
        timestamps:
            Ordered list of ``datetime`` values that form the x-axis.
            Must be the same length as every value list in ``channels``.
        channels:
            Mapping of channel name → list of values (floats or ``None``).
            ``None`` values should be rendered as gaps in the line, not zeros.
            All lists must have the same length as ``timestamps``.
        title:
            Human-readable plot title.
        output_path:
            If given, save the plot to this file (format inferred from the
            extension, e.g. ``.html`` for Bokeh, ``.png`` for Matplotlib).
            If ``None``, display interactively.
        """

    @abstractmethod
    def plot_similarity_dataframe(
        self,
        df: pd.DataFrame,
        title: str = "Cat Similarity Scores",
        output_path: Optional[Path] = None,
    ) -> None:
        """Plot the K-cat similarity DataFrame as overlapping lines.

        Parameters
        ----------
        df:
            A timestamp-indexed pandas DataFrame with one column per
            registered cat and ``NaN`` where no camera frame was available.
            Produced by ``RollingBuffer.to_dataframe(channel_prefix="similarity_")``.
        title:
            Human-readable plot title.
        output_path:
            If given, save to this file.  If ``None``, display interactively.
        """


# ===========================================================================
# Backend factory
# ===========================================================================

# Registry of known backend names.  Adding a new backend requires only
# creating td_plot_<name>.py and adding an entry here (optional — the
# factory works via importlib even without a registry entry, but the registry
# makes the help message useful).
_KNOWN_BACKENDS: dict[str, str] = {
    "bokeh": "td_plot_bokeh",
}


def get_plot_backend(name: str = "bokeh") -> PlotBackend:
    """Return a ``PlotBackend`` instance for the requested library.

    Parameters
    ----------
    name:
        Backend identifier, e.g. ``"bokeh"``, ``"plotly"``, ``"matplotlib"``.
        The string is used to import ``litterbox.td_plot_<name>``.

    Returns
    -------
    PlotBackend
        An instance of the backend's ``Backend`` class.

    Raises
    ------
    ValueError
        If the module ``td_plot_<name>.py`` cannot be imported (library not
        installed, or file does not exist).

    Notes
    -----
    The factory imports lazily so that uninstalled libraries only cause an
    error when the backend is actually requested, not at import time.
    """
    module_name = f"litterbox.td_plot_{name}"
    try:
        mod = import_module(module_name)
    except ImportError as exc:
        known = ", ".join(f'"{k}"' for k in _KNOWN_BACKENDS)
        raise ValueError(
            f"Cannot load plot backend '{name}' (tried to import '{module_name}'). "
            f"Install the required library and ensure the module file exists. "
            f"Known backends: {known}. "
            f"Original error: {exc}"
        ) from exc

    if not hasattr(mod, "Backend"):
        raise ValueError(
            f"Module '{module_name}' does not define a class named 'Backend'. "
            "Every plot backend module must expose a class named 'Backend' "
            "that inherits from PlotBackend."
        )

    backend = mod.Backend()
    if not isinstance(backend, PlotBackend):
        raise TypeError(
            f"'{module_name}.Backend' must inherit from PlotBackend."
        )
    return backend
