"""
Tests for the per-cat history plot module.

Verifies that the plotting function produces a valid HTML file under the
right conditions and surfaces clear errors otherwise. Does not visually
verify the rendered plot — that's a manual check — but confirms the data
selection, anomaly partitioning, and reference-line computation behave
correctly.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from litterbox.db import get_conn
from litterbox.history_plot import (
    plot_cat_history,
    _gas_threshold_levels,
    _split_anomalous,
    _parse_ts,
    _build_panel,
)


# ---------------------------------------------------------------------------
# Helpers — seed the test DB
# ---------------------------------------------------------------------------

def _insert_visit(
    conn,
    cat_id: int,
    *,
    days_ago: int,
    cat_weight_g=None,
    ammonia_peak_ppb=None,
    methane_peak_ppb=None,
    is_anomalous=False,
    ammonia_z=None,
    methane_z=None,
    tier=None,
    model=None,
):
    ts = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago)).isoformat()
    cur = conn.execute(
        """INSERT INTO visits
           (entry_time, tentative_cat_id, is_anomalous,
            cat_weight_g, ammonia_peak_ppb, methane_peak_ppb,
            ammonia_z_score, methane_z_score,
            gas_anomaly_tier, gas_anomaly_model_used)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ts, cat_id, is_anomalous,
         cat_weight_g, ammonia_peak_ppb, methane_peak_ppb,
         ammonia_z, methane_z, tier, model),
    )
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestParseTs:
    def test_iso_with_utc_suffix(self):
        dt = _parse_ts("2026-04-04T10:30:00+00:00")
        assert dt.year == 2026 and dt.month == 4

    def test_iso_with_z_suffix(self):
        dt = _parse_ts("2026-04-04T10:30:00Z")
        assert dt.year == 2026

    def test_naive_iso(self):
        dt = _parse_ts("2026-04-04T10:30:00")
        assert dt.year == 2026

    def test_microseconds_supported(self):
        dt = _parse_ts("2026-04-04T10:30:00.123456+00:00")
        assert dt.year == 2026


class TestGasThresholdLevels:
    def test_returns_three_levels_when_enough_data(self):
        # Tight cluster around log1p ≈ 3 (≈ 20 ppb), 8 readings.
        values = [18, 19, 20, 21, 22, 19, 20, 21]
        result = _gas_threshold_levels(values)
        assert result is not None
        median_ppb, mild, significant = result
        # Median sits within the data range, both alarms above it.
        assert min(values) <= median_ppb <= max(values)
        assert mild > median_ppb
        assert significant > mild

    def test_median_back_projected_to_ppb(self):
        # Six values spanning log1p ∈ [0.5, 1.5] in 0.2 steps → median in
        # log space = 1.0 → expm1(1.0) = e - 1 ≈ 1.718. Non-zero MAD so the
        # fit is not flagged degenerate.
        log_targets = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        values = [math.exp(t) - 1 for t in log_targets]
        result = _gas_threshold_levels(values)
        assert result is not None
        median_ppb, _, _ = result
        assert median_ppb == pytest.approx(math.e - 1, abs=0.05)

    def test_none_when_too_few(self):
        assert _gas_threshold_levels([20.0]) is None

    def test_none_when_constant(self):
        assert _gas_threshold_levels([20.0, 20.0, 20.0, 20.0]) is None

    def test_drops_nones(self):
        values = [None, 18, 19, None, 20, 21, 22]
        result = _gas_threshold_levels(values)
        assert result is not None


class TestSplitAnomalous:
    def test_partitions_by_is_anomalous(self):
        rows = [
            {"visit_id": 1, "entry_time": "2026-04-01T08:00:00",
             "is_anomalous": False, "cat_weight_g": 5000.0,
             "ammonia_peak_ppb": 20, "methane_peak_ppb": 10,
             "ammonia_z_score": 0.1, "methane_z_score": 0.2,
             "gas_anomaly_tier": "normal", "gas_anomaly_model_used": "per_cat"},
            {"visit_id": 2, "entry_time": "2026-04-02T08:00:00",
             "is_anomalous": True, "cat_weight_g": 5050.0,
             "ammonia_peak_ppb": 200, "methane_peak_ppb": 150,
             "ammonia_z_score": 5.5, "methane_z_score": 4.8,
             "gas_anomaly_tier": "severe", "gas_anomaly_model_used": "per_cat"},
        ]
        normal, anomalous = _split_anomalous(rows, "ammonia_peak_ppb")
        assert normal["visit_id"] == [1]
        assert anomalous["visit_id"] == [2]
        assert "+5.50" in anomalous["z"][0]

    def test_drops_rows_with_null_value(self):
        rows = [
            {"visit_id": 1, "entry_time": "2026-04-01T08:00:00",
             "is_anomalous": False, "cat_weight_g": 5000.0,
             "ammonia_peak_ppb": None, "methane_peak_ppb": 10,
             "ammonia_z_score": None, "methane_z_score": 0.2,
             "gas_anomaly_tier": None, "gas_anomaly_model_used": None},
        ]
        normal, anomalous = _split_anomalous(rows, "ammonia_peak_ppb")
        assert normal["visit_id"] == []
        assert anomalous["visit_id"] == []

    def test_z_score_em_dash_when_missing(self):
        rows = [
            {"visit_id": 1, "entry_time": "2026-04-01T08:00:00",
             "is_anomalous": False, "cat_weight_g": 5000.0,
             "ammonia_peak_ppb": 20, "methane_peak_ppb": 10,
             "ammonia_z_score": None, "methane_z_score": None,
             "gas_anomaly_tier": None, "gas_anomaly_model_used": None},
        ]
        normal, _ = _split_anomalous(rows, "ammonia_peak_ppb")
        assert normal["z"] == ["—"]


# ---------------------------------------------------------------------------
# _build_panel — log axis and reference-line layout
# ---------------------------------------------------------------------------

def _make_rows(n: int, value_col: str, value_fn):
    """Build n synthetic visit rows for _build_panel."""
    return [
        {
            "visit_id": i,
            "entry_time": f"2026-04-{(i % 28) + 1:02d}T08:00:00",
            "is_anomalous": False,
            "cat_weight_g": 4000.0,
            "ammonia_peak_ppb": value_fn(i) if value_col == "ammonia_peak_ppb" else 20.0,
            "methane_peak_ppb": value_fn(i) if value_col == "methane_peak_ppb" else 10.0,
            "ammonia_z_score": 0.1,
            "methane_z_score": 0.1,
            "gas_anomaly_tier": "normal",
            "gas_anomaly_model_used": "per_cat",
        }
        for i in range(n)
    ]


class TestBuildPanelLogAxis:
    def test_log_axis_kwarg_sets_log_y_scale(self):
        from bokeh.models import LogScale
        rows = _make_rows(8, "ammonia_peak_ppb", lambda i: 20.0 + (i % 5))
        panel = _build_panel(
            rows, "ammonia_peak_ppb",
            title="t", y_label="ppb",
            show_reference_lines=True,
            shared_x_range=None,
            log_y_axis=True,
        )
        assert isinstance(panel.y_scale, LogScale)

    def test_default_axis_is_linear(self):
        from bokeh.models import LinearScale
        rows = _make_rows(8, "cat_weight_g", lambda i: 4000.0 + i)
        panel = _build_panel(
            rows, "cat_weight_g",
            title="t", y_label="g",
            show_reference_lines=False,
            shared_x_range=None,
        )
        assert isinstance(panel.y_scale, LinearScale)

    def test_three_reference_spans_added_when_fit_succeeds(self):
        # Tight cluster, enough data → fit succeeds → median + mild + significant lines.
        from bokeh.models import Span
        rows = _make_rows(10, "ammonia_peak_ppb", lambda i: 20.0 + (i % 5))
        panel = _build_panel(
            rows, "ammonia_peak_ppb",
            title="t", y_label="ppb",
            show_reference_lines=True,
            shared_x_range=None,
            log_y_axis=True,
        )
        spans = [r for r in panel.center if isinstance(r, Span)]
        assert len(spans) == 3

    def test_median_span_uses_solid_green_style(self):
        from bokeh.models import Span
        rows = _make_rows(10, "ammonia_peak_ppb", lambda i: 20.0 + (i % 5))
        panel = _build_panel(
            rows, "ammonia_peak_ppb",
            title="t", y_label="ppb",
            show_reference_lines=True,
            shared_x_range=None,
            log_y_axis=True,
        )
        spans = sorted(
            [r for r in panel.center if isinstance(r, Span)],
            key=lambda s: s.location,
        )
        # Lowest location is the median: solid green, no dash pattern.
        median_span = spans[0]
        assert median_span.line_color == "#2ca02c"
        assert median_span.line_dash in (None, "solid", [])

    def test_alarm_spans_are_dashed_orange_and_red(self):
        from bokeh.models import Span
        rows = _make_rows(10, "ammonia_peak_ppb", lambda i: 20.0 + (i % 5))
        panel = _build_panel(
            rows, "ammonia_peak_ppb",
            title="t", y_label="ppb",
            show_reference_lines=True,
            shared_x_range=None,
            log_y_axis=True,
        )
        spans = sorted(
            [r for r in panel.center if isinstance(r, Span)],
            key=lambda s: s.location,
        )
        _, mild, significant = spans
        assert mild.line_color == "#ff8c00"
        assert significant.line_color == "#d62728"

    def test_no_reference_spans_when_data_too_sparse(self):
        from bokeh.models import Span
        rows = _make_rows(1, "ammonia_peak_ppb", lambda i: 20.0)
        panel = _build_panel(
            rows, "ammonia_peak_ppb",
            title="t", y_label="ppb",
            show_reference_lines=True,
            shared_x_range=None,
            log_y_axis=True,
        )
        spans = [r for r in panel.center if isinstance(r, Span)]
        assert spans == []


# ---------------------------------------------------------------------------
# plot_cat_history — end-to-end
# ---------------------------------------------------------------------------

class TestPlotCatHistory:
    def test_unknown_cat_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not registered"):
            plot_cat_history("Nobody", days=90, output_path=tmp_path / "x.html")

    def test_no_visits_in_window_raises(self, registered_cat, tmp_path):
        cat_id, name = registered_cat
        with get_conn() as conn:
            _insert_visit(conn, cat_id, days_ago=200, cat_weight_g=4000)
        with pytest.raises(ValueError, match="No visits found"):
            plot_cat_history(name, days=90, output_path=tmp_path / "x.html")

    def test_creates_html_file(self, registered_cat, tmp_path):
        cat_id, name = registered_cat
        with get_conn() as conn:
            for d in range(20):
                _insert_visit(
                    conn, cat_id, days_ago=d,
                    cat_weight_g=4000 + d * 5,
                    ammonia_peak_ppb=20 + (d % 5),
                    methane_peak_ppb=8 + (d % 3),
                    is_anomalous=(d == 5),
                    ammonia_z=2.5 if d == 5 else 0.1,
                    tier="mild" if d == 5 else "normal",
                    model="per_cat",
                )
        out = tmp_path / "cat_history.html"
        result = plot_cat_history(name, days=90, output_path=out)
        assert result == out
        assert out.exists()
        # Bokeh-emitted HTML should contain the cat name in the title.
        content = out.read_text()
        assert name in content

    def test_default_output_path_under_output_dir(self, registered_cat, monkeypatch, tmp_path):
        cat_id, name = registered_cat
        # Redirect the module-level default output dir to tmp_path.
        import litterbox.history_plot as hp
        monkeypatch.setattr(hp, "_DEFAULT_OUTPUT_DIR", tmp_path / "output")
        with get_conn() as conn:
            _insert_visit(conn, cat_id, days_ago=1, cat_weight_g=4200)
        path = plot_cat_history(name, days=30)
        assert path.parent == tmp_path / "output"
        assert path.name == f"cat_history_{name}.html"
        assert path.exists()

    def test_anomalous_and_normal_both_rendered(self, registered_cat, tmp_path):
        # Mixed visits, all NH3 readings, one anomalous → output HTML must
        # mention the legend labels for both glyph layers.
        cat_id, name = registered_cat
        with get_conn() as conn:
            _insert_visit(conn, cat_id, days_ago=1,  cat_weight_g=4000,
                          ammonia_peak_ppb=22, is_anomalous=False)
            _insert_visit(conn, cat_id, days_ago=2,  cat_weight_g=4010,
                          ammonia_peak_ppb=200, is_anomalous=True,
                          ammonia_z=5.5, tier="severe", model="per_cat")
        out = tmp_path / "h.html"
        plot_cat_history(name, days=30, output_path=out)
        content = out.read_text()
        assert "normal" in content
        assert "anomalous" in content

    def test_respects_days_window(self, registered_cat, tmp_path):
        # 3 in-window visits, 2 out-of-window. Both windows must succeed
        # but the in-window plot must not contain the old visits' weights.
        cat_id, name = registered_cat
        with get_conn() as conn:
            _insert_visit(conn, cat_id, days_ago=10,  cat_weight_g=4000)
            _insert_visit(conn, cat_id, days_ago=20,  cat_weight_g=4010)
            _insert_visit(conn, cat_id, days_ago=80,  cat_weight_g=4020)
            _insert_visit(conn, cat_id, days_ago=120, cat_weight_g=4900)
            _insert_visit(conn, cat_id, days_ago=200, cat_weight_g=4901)
        out_30 = tmp_path / "30.html"
        plot_cat_history(name, days=30, output_path=out_30)
        # Out-of-window weights must not appear.
        content_30 = out_30.read_text()
        assert "4900" not in content_30
        assert "4901" not in content_30


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------

class TestPlotCatHistoryTool:
    def test_tool_returns_path_string(self, registered_cat, tmp_path, monkeypatch):
        from litterbox.tools import plot_cat_history as plot_tool
        import litterbox.history_plot as hp
        monkeypatch.setattr(hp, "_DEFAULT_OUTPUT_DIR", tmp_path / "output")
        cat_id, name = registered_cat
        with get_conn() as conn:
            _insert_visit(conn, cat_id, days_ago=1, cat_weight_g=4000)
        result = plot_tool.invoke({"cat_name": name})
        assert "saved to" in result.lower()
        assert name in result

    def test_tool_handles_unknown_cat_gracefully(self):
        from litterbox.tools import plot_cat_history as plot_tool
        result = plot_tool.invoke({"cat_name": "Nobody"})
        assert result.lower().startswith("error")
        assert "not registered" in result

    def test_tool_passes_days_param(self, registered_cat, tmp_path, monkeypatch):
        from litterbox.tools import plot_cat_history as plot_tool
        import litterbox.history_plot as hp
        monkeypatch.setattr(hp, "_DEFAULT_OUTPUT_DIR", tmp_path / "output")
        cat_id, name = registered_cat
        with get_conn() as conn:
            _insert_visit(conn, cat_id, days_ago=2, cat_weight_g=4000)
        result = plot_tool.invoke({"cat_name": name, "days": 7})
        assert "last 7 days" in result
