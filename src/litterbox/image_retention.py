"""
image_retention.py — Visit image cleanup utility
=================================================

Deletes visit image directories older than a configurable retention window.
Camera frames captured during visits are stored under::

    <images_base>/visits/YYYY-MM-DD/<visit_uuid>/frame_NNNN.jpg

This module walks the date-level directories and removes any whose date is
older than ``retention_days`` from today.

The function takes ``images_base`` as an explicit parameter so tests can
point it at a temporary directory without monkeypatching module-level state.
"""

from __future__ import annotations

import shutil
from datetime import date, timedelta
from pathlib import Path


def sweep_old_visit_images(images_base: Path, retention_days: int) -> int:
    """Delete visit image directories older than *retention_days*.

    Parameters
    ----------
    images_base:
        Root of the image store (e.g. ``PROJECT_ROOT / "images"``).
        The function looks for date-named subdirectories under
        ``images_base / "visits"``.
    retention_days:
        Directories whose parsed date is more than this many days
        before today are removed.

    Returns
    -------
    int
        Number of date directories deleted.
    """
    visits_dir = images_base / "visits"
    if not visits_dir.is_dir():
        return 0

    cutoff = date.today() - timedelta(days=retention_days)
    deleted = 0

    for child in sorted(visits_dir.iterdir()):
        if not child.is_dir():
            continue
        try:
            dir_date = date.fromisoformat(child.name)
        except ValueError:
            # Not a YYYY-MM-DD directory — skip silently.
            continue
        if dir_date < cutoff:
            shutil.rmtree(child)
            deleted += 1

    return deleted
