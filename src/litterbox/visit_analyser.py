"""
visit_analyser.py — Step 4 of the Time-Domain Measurement System
=================================================================

When the ``VisitTrigger`` (Step 3) fires its ``on_visit_complete`` callback,
this module takes the buffer snapshot and determines **which cat** was in the
box, then persists the visit to the ``td_visits`` table.

Cat identification follows a strict priority order:

1. **Chip ID** — if any sample in the visit window has a non-null ``chip_id``
   channel, the most frequent non-null value is used.  The visit is marked
   ``is_confirmed = True`` because chip-based ID is authoritative.

2. **Similarity DataFrame** — when no chip ID is available and the
   ``similarity`` channel is enabled, a per-cat DataFrame of CLIP scores is
   built from the visit window.  The cat with the highest mean score wins,
   provided it exceeds the configured threshold **and** the sustained-peak
   gate passes (P consecutive samples above threshold).  The visit is marked
   as a tentative (unconfirmed) ID, consistent with the existing CLIP
   pipeline behaviour.

3. **Unknown** — if neither method produces a result, the visit is recorded
   with ``id_method = "unknown"`` and flagged for human review.

The ``TdVisitRecord`` dataclass mirrors the ``td_visits`` table exactly and
is the data-transfer object between ``analyse()`` and ``save()``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from litterbox.db import get_conn, init_db


# ===========================================================================
# Data-transfer object
# ===========================================================================

@dataclass
class TdVisitRecord:
    """One row in the ``td_visits`` table.

    All fields mirror the DB columns.  ``td_visit_id`` is ``None`` until
    ``VisitAnalyser.save()`` populates it from ``lastrowid``.
    """

    entry_time:       datetime
    exit_time:        datetime
    snapshot_json:    str
    chip_id:          Optional[str]   = None
    tentative_cat_id: Optional[int]   = None
    confirmed_cat_id: Optional[int]   = None
    is_confirmed:     bool            = False
    id_method:        str             = "unknown"
    top_similarity:   Optional[float] = None
    images_dir:       Optional[str]   = None
    health_notes:     Optional[str]   = None
    is_anomalous:     bool            = False
    td_visit_id:      Optional[int]   = None


# ===========================================================================
# JSON helpers
# ===========================================================================

def _serialize_snapshot(snapshot: list[dict]) -> str:
    """Serialize a buffer snapshot to JSON, converting datetimes to ISO strings."""

    def _default(obj: Any) -> str:
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    return json.dumps(snapshot, default=_default)


def _deserialize_snapshot(raw: str) -> list[dict]:
    """Deserialize a snapshot JSON string back to a list of dicts."""
    return json.loads(raw)


# ===========================================================================
# VisitAnalyser
# ===========================================================================

class VisitAnalyser:
    """Analyse a buffer snapshot to identify the visiting cat and persist the visit.

    Parameters
    ----------
    config:
        Parsed ``td_config.json`` dict.  Thresholds are read from
        ``config["trigger"]``.
    """

    def __init__(self, config: dict) -> None:
        trig = config.get("trigger", {})
        self._sim_entry_threshold: float = float(
            trig.get("similarity_entry_threshold", 0.70)
        )
        self._sustained_peak_samples: int = int(
            trig.get("similarity_sustained_peak_samples", 3)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        snapshot: list[dict],
        entry_time: datetime,
        exit_time: datetime,
    ) -> TdVisitRecord:
        """Determine cat identity from the buffer snapshot.

        Parameters
        ----------
        snapshot:
            Full M-minute buffer at the moment of visit exit.  Each element
            is ``{"timestamp": datetime, "values": dict}``.
        entry_time:
            UTC timestamp of the KITTY_ABSENT → KITTY_PRESENT transition.
        exit_time:
            UTC timestamp of the KITTY_PRESENT → KITTY_ABSENT transition.

        Returns
        -------
        TdVisitRecord
            Populated record ready for ``save()``.  ``td_visit_id`` is
            ``None`` until ``save()`` is called.
        """
        snapshot_json = _serialize_snapshot(snapshot)

        # Filter to the visit window.
        visit_window = [
            e for e in snapshot
            if entry_time <= e["timestamp"] <= exit_time
        ]

        # --- Priority 1: Chip ID ---
        result = self._check_chip_id(visit_window)
        if result is not None:
            return TdVisitRecord(
                entry_time=entry_time,
                exit_time=exit_time,
                snapshot_json=snapshot_json,
                **result,
            )

        # --- Priority 2: Similarity DataFrame ---
        result = self._check_similarity(visit_window)
        if result is not None:
            return TdVisitRecord(
                entry_time=entry_time,
                exit_time=exit_time,
                snapshot_json=snapshot_json,
                **result,
            )

        # --- Priority 3: Unknown ---
        return TdVisitRecord(
            entry_time=entry_time,
            exit_time=exit_time,
            snapshot_json=snapshot_json,
            id_method="unknown",
        )

    def save(self, record: TdVisitRecord) -> int:
        """Persist a ``TdVisitRecord`` to the ``td_visits`` table.

        Parameters
        ----------
        record:
            The visit record to save.  ``td_visit_id`` will be set on the
            record object as a side effect.

        Returns
        -------
        int
            The new ``td_visit_id``.
        """
        init_db()  # idempotent — ensures td_visits exists

        with get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO td_visits
                   (entry_time, exit_time, chip_id,
                    tentative_cat_id, confirmed_cat_id, is_confirmed,
                    id_method, top_similarity, snapshot_json,
                    images_dir, health_notes, is_anomalous)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.entry_time.isoformat(),
                    record.exit_time.isoformat(),
                    record.chip_id,
                    record.tentative_cat_id,
                    record.confirmed_cat_id,
                    record.is_confirmed,
                    record.id_method,
                    record.top_similarity,
                    record.snapshot_json,
                    record.images_dir,
                    record.health_notes,
                    record.is_anomalous,
                ),
            )
            record.td_visit_id = cur.lastrowid

        return record.td_visit_id

    # ------------------------------------------------------------------
    # Chip ID logic
    # ------------------------------------------------------------------

    def _check_chip_id(self, visit_window: list[dict]) -> Optional[dict]:
        """Check for chip-based identification.

        Returns a dict of field overrides for TdVisitRecord, or None if
        no chip ID was found in the visit window.
        """
        chip_values = [
            e["values"]["chip_id"]
            for e in visit_window
            if e["values"].get("chip_id") is not None
        ]
        if not chip_values:
            return None

        # Most frequent non-null chip ID.
        winning_chip = Counter(chip_values).most_common(1)[0][0]

        # Look up cat by name (chip_id stores the cat name string).
        cat_id = self._lookup_cat_by_name(winning_chip)

        return {
            "chip_id": winning_chip,
            "tentative_cat_id": cat_id,
            "confirmed_cat_id": cat_id,
            "is_confirmed": True,
            "id_method": "chip",
        }

    # ------------------------------------------------------------------
    # Similarity DataFrame logic
    # ------------------------------------------------------------------

    def _check_similarity(self, visit_window: list[dict]) -> Optional[dict]:
        """Check for similarity-based identification.

        Builds a per-cat DataFrame from ``similarity_*`` keys in the visit
        window, computes column means, and checks the sustained-peak gate.

        Returns a dict of field overrides for TdVisitRecord, or None if
        similarity data is absent or no cat passes the threshold.
        """
        import pandas as pd  # lazy import

        df = self._build_similarity_df(visit_window)
        if df.empty or df.shape[1] == 0:
            return None

        col_means = df.mean(skipna=True)
        if col_means.empty:
            return None

        winning_cat = col_means.idxmax()
        winning_mean = col_means[winning_cat]

        # Check threshold.
        if winning_mean < self._sim_entry_threshold:
            return None

        # Sustained-peak gate: winning cat must exceed threshold for at
        # least P consecutive samples in the visit window.
        if not self._sustained_peak_check(df[winning_cat]):
            return None

        cat_id = self._lookup_cat_by_name(winning_cat)

        return {
            "tentative_cat_id": cat_id,
            "is_confirmed": False,
            "id_method": "similarity",
            "top_similarity": float(winning_mean),
        }

    def _build_similarity_df(self, visit_window: list[dict]) -> "pd.DataFrame":
        """Build a similarity DataFrame from visit-window entries.

        Columns are cat names (``similarity_`` prefix stripped).
        Index is timestamps.  Missing values are ``NaN``.
        """
        import pandas as pd

        if not visit_window:
            return pd.DataFrame()

        records: list[dict] = []
        timestamps: list[datetime] = []

        for entry in visit_window:
            row: dict[str, float] = {}
            for key, val in entry["values"].items():
                if key.startswith("similarity_"):
                    cat_name = key[len("similarity_"):]
                    row[cat_name] = val
            records.append(row)
            timestamps.append(entry["timestamp"])

        if not any(records):
            return pd.DataFrame()

        df = pd.DataFrame(records, index=timestamps)
        df.index.name = "timestamp"
        return df

    def _sustained_peak_check(self, series: "pd.Series") -> bool:
        """Check if *series* exceeds the entry threshold for P consecutive non-NaN samples.

        NaN values (missing camera frames) are skipped entirely — they
        neither count toward the consecutive run nor reset it.  This
        prevents missing frames from breaking an otherwise sustained peak.
        """
        import math

        consecutive = 0
        for val in series:
            # Skip NaN / None — they don't reset the streak.
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            if val >= self._sim_entry_threshold:
                consecutive += 1
                if consecutive >= self._sustained_peak_samples:
                    return True
            else:
                consecutive = 0
        return False

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lookup_cat_by_name(name: str) -> Optional[int]:
        """Look up a cat_id by name.  Returns None if not found."""
        with get_conn() as conn:
            row = conn.execute(
                "SELECT cat_id FROM cats WHERE name = ?", (name,)
            ).fetchone()
        return row["cat_id"] if row else None
