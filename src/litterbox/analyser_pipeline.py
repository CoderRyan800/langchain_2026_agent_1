"""
analyser_pipeline.py — Step 5a of the Time-Domain Measurement System
=====================================================================

Thin plugin orchestration layer for time-domain waveform analysers.

After ``VisitAnalyser.save()`` persists the visit record, the
``AnalyserPipeline`` extracts the weight waveform from the buffer snapshot,
resamples it to a fixed length, and passes it to every registered plugin.
Each plugin performs its own analysis (eigendecomposition, FFT, ML
classification, etc.) and returns an ``AnalysisResult``.

Architecture
------------
::

    on_visit_complete fires
            │
            ▼
    VisitAnalyser.analyse() + save()
            │  returns TdVisitRecord
            ▼
    AnalyserPipeline.run(record, snapshot, ...)
            │
    ┌───────┼───────────┐
    ▼       ▼           ▼
    Plugin1 Plugin2   Plugin3
    (Eigen) (FFT?)    (ML?)

Each plugin:

- Implements ``BaseAnalyser`` (``name`` property + ``analyse()`` method).
- Receives a **resampled** waveform of length L (default 64).  The pipeline
  handles resampling so plugins don't duplicate that logic.
- The waveform is **not** mean-subtracted — each plugin owns its own
  preprocessing (a wavelet plugin may not want DC removal; an ML plugin
  may want a different normalisation).
- Returns an ``AnalysisResult`` with a score and plugin-specific details.
- Is **fault-isolated**: if one plugin raises, the others still run and
  an error result is recorded for the failing plugin.

Resampling utility
------------------
``resample_to_length()`` is a public module-level function so that plugins
and tests can call it directly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from litterbox.visit_analyser import TdVisitRecord

logger = logging.getLogger(__name__)


# ===========================================================================
# Data-transfer object
# ===========================================================================

@dataclass
class AnalysisResult:
    """Result returned by a single analyser plugin for one visit.

    Attributes
    ----------
    plugin_name:
        Short identifier for the plugin (e.g. ``"eigen"``).
    anomaly_score:
        Scalar in ``[0, 1]``.  ``0.0`` = normal, ``1.0`` = maximum anomaly.
    anomaly_level:
        Human-readable classification: ``"normal"``, ``"mild"``,
        ``"significant"``, ``"major"``, ``"insufficient_data"``, or
        ``"error"`` (when the plugin raised an exception).
    details:
        Plugin-specific payload (e.g. explained variance, eigenvalues,
        expansion coefficients, residual norm, etc.).
    """

    plugin_name:   str
    anomaly_score: float
    anomaly_level: str
    details:       dict = field(default_factory=dict)


# ===========================================================================
# Abstract plugin base
# ===========================================================================

class BaseAnalyser(ABC):
    """Abstract base class for time-domain waveform analyser plugins.

    Every concrete plugin must implement:

    - ``name`` — a short string identifier (used in logs and DB records).
    - ``analyse(waveform, visit_record, channel)`` — takes the resampled
      waveform and returns an ``AnalysisResult``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this analyser (e.g. ``"eigen"``)."""

    @abstractmethod
    def analyse(
        self,
        waveform: np.ndarray,
        visit_record: TdVisitRecord,
        channel: str,
    ) -> AnalysisResult:
        """Analyse a resampled waveform and return a result.

        Parameters
        ----------
        waveform:
            1-D numpy array of length L (default 64), resampled from the
            raw visit-window channel data.  **Not** mean-subtracted — the
            plugin owns its own preprocessing.  NaN values indicate samples
            where the sensor had no reading.
        visit_record:
            The ``TdVisitRecord`` for this visit.  Provides ``td_visit_id``,
            ``cat_id``, entry/exit times, etc.
        channel:
            The sensor channel name (e.g. ``"weight_g"``).

        Returns
        -------
        AnalysisResult
        """


# ===========================================================================
# Resampling utility
# ===========================================================================

def resample_to_length(
    raw: np.ndarray,
    target_length: int = 64,
) -> np.ndarray:
    """Resample a 1-D array to *target_length* using linear interpolation.

    NaN gaps in the input are filled by linear interpolation before
    resampling so the output is as clean as possible.  If the entire input
    is NaN (or empty), the output is all-NaN.

    Parameters
    ----------
    raw:
        1-D numpy array of arbitrary length.
    target_length:
        Desired output length (default 64).

    Returns
    -------
    np.ndarray
        1-D array of shape ``(target_length,)``.
    """
    if len(raw) == 0:
        return np.full(target_length, np.nan)

    # --- Gap-fill NaN values via linear interpolation ---
    filled = raw.copy().astype(float)
    nans = np.isnan(filled)

    if nans.all():
        return np.full(target_length, np.nan)

    if nans.any():
        # np.interp requires the x-coordinates of known points.
        x_all = np.arange(len(filled))
        x_known = x_all[~nans]
        y_known = filled[~nans]
        filled[nans] = np.interp(x_all[nans], x_known, y_known)

    # --- Resample to target_length ---
    if len(filled) == target_length:
        return filled

    x_old = np.linspace(0, 1, len(filled))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, filled)


# ===========================================================================
# Pipeline
# ===========================================================================

class AnalyserPipeline:
    """Orchestrate registered analyser plugins on each completed visit.

    Parameters
    ----------
    analysers:
        List of ``BaseAnalyser`` plugin instances to run on each visit.
    config:
        Parsed ``td_config.json`` dict.  Used to read the resampling
        target length from ``config.get("eigen", {}).get("resample_length", 64)``.
    """

    def __init__(
        self,
        analysers: list[BaseAnalyser],
        config: dict | None = None,
    ) -> None:
        self._analysers = list(analysers)
        config = config or {}
        self._resample_length: int = int(
            config.get("eigen", {}).get("resample_length", 64)
        )

    def run(
        self,
        record: TdVisitRecord,
        snapshot: list[dict],
        entry_time: datetime,
        exit_time: datetime,
        channel: str = "weight_g",
    ) -> list[AnalysisResult]:
        """Extract a channel waveform, resample it, and pass to all plugins.

        Parameters
        ----------
        record:
            The persisted ``TdVisitRecord`` (must have ``td_visit_id`` set).
        snapshot:
            Full M-minute buffer snapshot as a list of dicts.
        entry_time:
            Visit entry timestamp.
        exit_time:
            Visit exit timestamp.
        channel:
            The sensor channel to extract (default ``"weight_g"``).

        Returns
        -------
        list[AnalysisResult]
            One result per registered plugin.  Plugins that raised an
            exception produce an ``AnalysisResult`` with
            ``anomaly_level="error"``.
        """
        if not self._analysers:
            return []

        # --- Extract channel data from visit window ---
        raw_values: list[float] = []
        for entry in snapshot:
            if entry_time <= entry["timestamp"] <= exit_time:
                val = entry["values"].get(channel)
                raw_values.append(float(val) if val is not None else np.nan)

        raw = np.array(raw_values, dtype=float)

        # --- Resample to fixed length ---
        waveform = resample_to_length(raw, self._resample_length)

        # --- Run each plugin ---
        results: list[AnalysisResult] = []
        for analyser in self._analysers:
            try:
                result = analyser.analyse(waveform, record, channel)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Plugin %r raised %s: %s",
                    analyser.name, type(exc).__name__, exc,
                )
                results.append(AnalysisResult(
                    plugin_name=analyser.name,
                    anomaly_score=0.0,
                    anomaly_level="error",
                    details={"error": str(exc)},
                ))

        return results
