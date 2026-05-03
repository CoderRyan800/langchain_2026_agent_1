"""
rescore.py — Re-score gas anomaly columns on existing visits
=============================================================

The ``is_anomalous`` flag and the ``gas_anomaly_*`` columns are written
exactly once, when ``record_exit`` runs. As the detector evolves —
threshold-based → mean+std → median+MAD — older visits keep the verdict
that was true at the time they were scored. That's a problem for plots
and reports that span dates: identical readings can show different
verdicts purely because they were processed by different detector
versions.

This module re-scores every visit's gas anomaly columns against the
**current** detector. The fit pool for each visit is "everything else
in the visits table with non-null gas readings", same exclusion logic
the live detector uses at ``record_exit`` time. The LLM-side verdict
is preserved where ``health_notes`` followed the structured format
(parsed via ``parse_health_response``); refusals and missing notes
default to LLM-not-anomalous.

Final verdict: ``is_anomalous = LLM_yes_from_health_notes OR (gas_tier
in ALARM_TIERS)`` — exactly what the live ``record_exit`` produces today.

Idempotent. Safe to run multiple times. Sets a
``gas_anomaly_rescored_at`` timestamp on every row that was actually
changed, so it's auditable when the rescore happened.

Usage::

    python -m litterbox.rescore --dry-run    # preview
    python -m litterbox.rescore              # apply
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from litterbox.db import get_conn, init_db
from litterbox.gas_anomaly import score_gas_visit, ALARM_TIERS
from litterbox.health import parse_health_response


def _llm_verdict_from_notes(notes: Optional[str]) -> bool:
    """Recover the LLM's CONCERNS_PRESENT verdict from stored health_notes.

    Returns ``False`` for missing notes, refusal text, the new placeholder
    ("Health analysis unavailable —..."), and any other unstructured reply.
    Returns ``True`` only when the response followed the format and said
    ``CONCERNS_PRESENT: yes``.
    """
    if not notes:
        return False
    is_anomalous, _ = parse_health_response(notes)
    return is_anomalous


def rescore_all_visits(
    conn: Optional[sqlite3.Connection] = None,
    *,
    dry_run: bool = False,
) -> dict:
    """Re-score every visit's gas anomaly using the current detector.

    Parameters
    ----------
    conn:
        Optional open connection. If ``None``, a fresh one is opened and
        closed by this function. Tests pass a tmp-path connection.
    dry_run:
        When ``True``, compute and report changes without writing any
        UPDATE.

    Returns
    -------
    dict
        Summary with counts (``seen``, ``changed``, ``unchanged``,
        ``flag_changed_to_true``, ``flag_changed_to_false``) and a
        ``changes`` list of per-visit before/after dicts (capped to
        avoid memory blowups in pathological cases — the count fields
        are always exact).
    """
    init_db()
    own_conn = conn is None
    if own_conn:
        conn = get_conn()

    try:
        rows = conn.execute(
            """SELECT visit_id, tentative_cat_id, confirmed_cat_id,
                      ammonia_peak_ppb, methane_peak_ppb,
                      ammonia_z_score, methane_z_score,
                      gas_anomaly_tier, is_anomalous, health_notes
                 FROM visits
                WHERE ammonia_peak_ppb IS NOT NULL
                   OR methane_peak_ppb IS NOT NULL
                ORDER BY entry_time"""
        ).fetchall()

        rescored_at = datetime.now(timezone.utc).isoformat()
        seen = 0
        changed = 0
        unchanged = 0
        flag_to_true = 0
        flag_to_false = 0
        changes = []

        for r in rows:
            seen += 1
            cat_id = r["confirmed_cat_id"] or r["tentative_cat_id"]

            new_score = score_gas_visit(
                conn,
                cat_id=cat_id,
                ammonia_peak_ppb=r["ammonia_peak_ppb"],
                methane_peak_ppb=r["methane_peak_ppb"],
                exclude_visit_id=r["visit_id"],
            )

            llm_anomalous = _llm_verdict_from_notes(r["health_notes"])
            new_is_anomalous = (
                llm_anomalous or new_score["overall_tier"] in ALARM_TIERS
            )

            old = (
                r["ammonia_z_score"], r["methane_z_score"],
                r["gas_anomaly_tier"], bool(r["is_anomalous"]),
            )
            new = (
                new_score["ammonia_z"], new_score["methane_z"],
                new_score["overall_tier"], new_is_anomalous,
            )

            if old == new:
                unchanged += 1
                continue

            changed += 1
            if old[3] != new[3]:
                if new[3]:
                    flag_to_true += 1
                else:
                    flag_to_false += 1

            changes.append({
                "visit_id":     r["visit_id"],
                "old_nh3_z":    old[0],
                "old_ch4_z":    old[1],
                "old_tier":     old[2],
                "old_flag":     old[3],
                "new_nh3_z":    new[0],
                "new_ch4_z":    new[1],
                "new_tier":     new[2],
                "new_flag":     new[3],
            })

            if not dry_run:
                conn.execute(
                    """UPDATE visits SET
                          ammonia_z_score         = ?,
                          methane_z_score         = ?,
                          gas_anomaly_tier        = ?,
                          gas_anomaly_n_samples   = ?,
                          gas_anomaly_model_used  = ?,
                          is_anomalous            = ?,
                          gas_anomaly_rescored_at = ?
                        WHERE visit_id = ?""",
                    (
                        new_score["ammonia_z"],
                        new_score["methane_z"],
                        new_score["overall_tier"],
                        new_score["n_samples"],
                        new_score["model_used"],
                        new_is_anomalous,
                        rescored_at,
                        r["visit_id"],
                    ),
                )

        if not dry_run:
            conn.commit()

        return {
            "seen":                  seen,
            "changed":               changed,
            "unchanged":             unchanged,
            "flag_changed_to_true":  flag_to_true,
            "flag_changed_to_false": flag_to_false,
            "rescored_at":           rescored_at,
            "dry_run":               dry_run,
            "changes":               changes,
        }
    finally:
        if own_conn:
            conn.close()


def _format_z(z) -> str:
    return f"{z:+.2f}" if isinstance(z, (int, float)) else "—"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score gas anomaly columns on existing visits "
                    "against the current detector."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print changes without writing any UPDATE.",
    )
    parser.add_argument(
        "--show", type=int, default=20,
        help="Print up to this many sample changes (default 20).",
    )
    args = parser.parse_args()

    summary = rescore_all_visits(dry_run=args.dry_run)
    label = "DRY RUN" if args.dry_run else "APPLIED"

    print(f"[{label}] visits seen:    {summary['seen']}")
    print(f"          changed:        {summary['changed']}")
    print(f"          unchanged:      {summary['unchanged']}")
    print(f"          flag → True:    {summary['flag_changed_to_true']}")
    print(f"          flag → False:   {summary['flag_changed_to_false']}")
    if summary["changes"]:
        print()
        print(f"Sample changes (showing up to {args.show}):")
        print(f"  {'visit':>6}  {'NH₃ z (old → new)':>22}  "
              f"{'CH₄ z (old → new)':>22}  {'tier (old → new)':>30}  flag")
        for c in summary["changes"][:args.show]:
            tier_change = f"{c['old_tier']} → {c['new_tier']}"
            flag_change = f"{c['old_flag']} → {c['new_flag']}"
            print(
                f"  #{c['visit_id']:<5} "
                f" {_format_z(c['old_nh3_z']):>9} → {_format_z(c['new_nh3_z']):<9}"
                f" {_format_z(c['old_ch4_z']):>9} → {_format_z(c['new_ch4_z']):<9}"
                f" {tier_change:>30}  {flag_change}"
            )


if __name__ == "__main__":
    main()
