"""
Tests for eigen_query.py — Query and HTML reporting for eigenanalysis results
==============================================================================

Covers:
- get_visit_summary with scored and unscored visits
- get_waveforms round-trip
- get_model retrieval (per-cat and pooled fallback)
- generate_report HTML structure
- eigen_report LangChain tool
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from litterbox.db import get_conn, init_db
from litterbox.eigen_query import (
    generate_report,
    get_model,
    get_visit_summary,
    get_waveforms,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 4, 10, 8, 0, 0, tzinfo=timezone.utc)


def _ts(offset_seconds: float = 0.0) -> datetime:
    return _BASE + timedelta(seconds=offset_seconds)


def _insert_cat(name: str) -> int:
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO cats (name) VALUES (?)", (name,))
        return cur.lastrowid


def _insert_scored_visit(
    cat_id: int,
    visit_offset: float,
    dc_term: float = 5000.0,
    ev: float = 0.95,
    residual: float = 1.2,
    n_components: int = 3,
    L: int = 64,
    seed: int = 42,
) -> int:
    """Insert a td_visit + eigen_waveform + eigen_model for a scored visit.

    Returns the td_visit_id.
    """
    rng = np.random.RandomState(seed)
    vector = rng.randn(L)
    vector -= vector.mean()
    coefficients = rng.randn(L)

    entry_time = _ts(visit_offset)
    exit_time = _ts(visit_offset + 120)

    with get_conn() as conn:
        # td_visit
        cur = conn.execute(
            """INSERT INTO td_visits (entry_time, exit_time, snapshot_json)
               VALUES (?, ?, ?)""",
            (entry_time.isoformat(), exit_time.isoformat(), "[]"),
        )
        td_visit_id = cur.lastrowid

        # eigen_model
        eigenvalues = np.sort(rng.rand(L))[::-1].tolist()
        eigenvectors = np.eye(L).tolist()
        cur = conn.execute(
            """INSERT INTO eigen_models
               (cat_id, channel, eigenvalues_json, eigenvectors_json,
                n_components, n_waveforms, explained_variance_threshold, regularized)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cat_id, "weight_g",
                json.dumps(eigenvalues), json.dumps(eigenvectors),
                n_components, 20, 0.95, False,
            ),
        )
        model_id = cur.lastrowid

        # eigen_waveform
        conn.execute(
            """INSERT INTO eigen_waveforms
               (td_visit_id, cat_id, channel, vector_json, dc_term,
                coefficients_json, eigen_ev, eigen_residual, model_id,
                raw_length, nan_fraction)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                td_visit_id, cat_id, "weight_g",
                json.dumps(vector.tolist()), dc_term,
                json.dumps(coefficients.tolist()), ev, residual, model_id,
                L, 0.0,
            ),
        )

    return td_visit_id


def _insert_unscored_visit(cat_id: int, visit_offset: float, L: int = 64) -> int:
    """Insert a td_visit + eigen_waveform with no scoring (no model, no coefficients)."""
    rng = np.random.RandomState(99)
    vector = rng.randn(L)
    vector -= vector.mean()

    entry_time = _ts(visit_offset)
    exit_time = _ts(visit_offset + 60)

    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO td_visits (entry_time, exit_time, snapshot_json)
               VALUES (?, ?, ?)""",
            (entry_time.isoformat(), exit_time.isoformat(), "[]"),
        )
        td_visit_id = cur.lastrowid

        conn.execute(
            """INSERT INTO eigen_waveforms
               (td_visit_id, cat_id, channel, vector_json, dc_term,
                raw_length, nan_fraction)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                td_visit_id, cat_id, "weight_g",
                json.dumps(vector.tolist()), 4800.0,
                L, 0.05,
            ),
        )

    return td_visit_id


# ---------------------------------------------------------------------------
# TestGetVisitSummary
# ---------------------------------------------------------------------------


class TestGetVisitSummary:
    """get_visit_summary() returns correct per-visit data."""

    def test_scored_visits(self):
        cat_id = _insert_cat("Luna")
        _insert_scored_visit(cat_id, 0, dc_term=5100.0, ev=0.96, n_components=3, seed=1)
        _insert_scored_visit(cat_id, 300, dc_term=5200.0, ev=0.82, n_components=4, seed=2)
        _insert_scored_visit(cat_id, 600, dc_term=5050.0, ev=0.35, n_components=3, seed=3)

        summaries = get_visit_summary("Luna")

        assert len(summaries) == 3
        assert summaries[0]["dc_term"] == pytest.approx(5100.0)
        assert summaries[0]["eigen_ev"] == pytest.approx(0.96)
        assert summaries[0]["n_components"] == 3
        assert summaries[0]["anomaly_level"] == "normal"
        assert summaries[1]["anomaly_level"] == "mild"
        assert summaries[2]["anomaly_level"] == "major"

    def test_signal_coefficients_length_matches_n(self):
        cat_id = _insert_cat("Anna")
        _insert_scored_visit(cat_id, 0, n_components=5, seed=10)

        summaries = get_visit_summary("Anna")

        assert len(summaries) == 1
        assert summaries[0]["signal_coefficients"] is not None
        assert len(summaries[0]["signal_coefficients"]) == 5

    def test_unscored_visit(self):
        cat_id = _insert_cat("Marina")
        _insert_unscored_visit(cat_id, 0)

        summaries = get_visit_summary("Marina")

        assert len(summaries) == 1
        assert summaries[0]["eigen_ev"] is None
        assert summaries[0]["anomaly_level"] == "unscored"
        assert summaries[0]["signal_coefficients"] is None

    def test_no_visits(self):
        _insert_cat("Ghost")
        summaries = get_visit_summary("Ghost")
        assert summaries == []

    def test_nonexistent_cat(self):
        summaries = get_visit_summary("DoesNotExist")
        assert summaries == []


# ---------------------------------------------------------------------------
# TestGetWaveforms
# ---------------------------------------------------------------------------


class TestGetWaveforms:
    """get_waveforms() returns correct arrays and timestamps."""

    def test_round_trip(self):
        cat_id = _insert_cat("WaveCat")
        _insert_scored_visit(cat_id, 0, seed=42)
        _insert_scored_visit(cat_id, 300, seed=43)

        waveforms, timestamps = get_waveforms("WaveCat")

        assert len(waveforms) == 2
        assert len(timestamps) == 2
        assert waveforms[0].shape == (64,)
        assert waveforms[1].shape == (64,)
        # Waveforms should be zero-mean.
        assert abs(waveforms[0].sum()) < 1e-10
        assert abs(waveforms[1].sum()) < 1e-10

    def test_empty(self):
        _insert_cat("EmptyCat")
        waveforms, timestamps = get_waveforms("EmptyCat")
        assert waveforms == []
        assert timestamps == []


# ---------------------------------------------------------------------------
# TestGetModel
# ---------------------------------------------------------------------------


class TestGetModel:
    """get_model() retrieves the most recent eigenmodel."""

    def test_per_cat_model(self):
        cat_id = _insert_cat("ModelCat")
        _insert_scored_visit(cat_id, 0, n_components=4, seed=50)

        model = get_model("ModelCat")

        assert model is not None
        assert model["n_components"] == 4
        assert model["eigenvalues"].shape == (64,)
        assert model["eigenvectors"].shape == (64, 64)

    def test_no_model(self):
        _insert_cat("NoModelCat")
        model = get_model("NoModelCat")
        assert model is None


# ---------------------------------------------------------------------------
# TestGenerateReport
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """generate_report() produces valid HTML."""

    def test_html_contains_key_elements(self):
        cat_id = _insert_cat("ReportCat")
        _insert_scored_visit(cat_id, 0, dc_term=5100.0, ev=0.96, n_components=3, seed=1)
        _insert_scored_visit(cat_id, 300, dc_term=5200.0, ev=0.45, n_components=3, seed=2)

        html = generate_report("ReportCat")

        assert "ReportCat" in html
        assert "<svg" in html
        assert "<table" in html
        assert "Explained Var" in html
        assert "Signal Coefficients" in html
        assert "5100.0" in html
        assert "0.9600" in html
        assert "normal" in html
        assert "significant" in html

    def test_html_to_file(self, tmp_path):
        cat_id = _insert_cat("FileCat")
        _insert_scored_visit(cat_id, 0, seed=60)

        output_path = tmp_path / "report.html"
        result = generate_report("FileCat", output_path=output_path)

        assert result == str(output_path)
        assert output_path.exists()
        content = output_path.read_text()
        assert "FileCat" in content
        assert "<svg" in content

    def test_no_data_report(self):
        _insert_cat("EmptyReport")
        html = generate_report("EmptyReport")
        assert "No waveforms" in html
        assert "No visit data" in html

    def test_unscored_visit_in_report(self):
        cat_id = _insert_cat("UnscoredReport")
        _insert_unscored_visit(cat_id, 0)

        html = generate_report("UnscoredReport")

        assert "<svg" in html  # waveform still plotted
        assert "unscored" in html


# ---------------------------------------------------------------------------
# TestEigenReportTool
# ---------------------------------------------------------------------------


class TestEigenReportTool:
    """The eigen_report LangChain tool wrapper."""

    def test_tool_returns_summary(self, tmp_path, monkeypatch):
        import litterbox.tools as tools_mod

        monkeypatch.setattr(tools_mod, "PROJECT_ROOT", tmp_path)

        cat_id = _insert_cat("ToolCat")
        _insert_scored_visit(cat_id, 0, dc_term=5300.0, ev=0.92, n_components=3, seed=70)

        from litterbox.tools import eigen_report
        result = eigen_report.invoke({"cat_name": "ToolCat"})

        assert "ToolCat" in result
        assert "1 visit(s)" in result
        assert "1 scored" in result
        assert "5300.0" in result
        assert "normal" in result

        # HTML file should exist.
        html_path = tmp_path / "output" / "eigen_toolcat.html"
        assert html_path.exists()

    def test_tool_no_data(self):
        from litterbox.tools import eigen_report
        result = eigen_report.invoke({"cat_name": "NobodyCat"})
        assert "No eigenanalysis data" in result
