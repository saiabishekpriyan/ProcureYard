"""Idempotent seed loader — reads CSVs and inserts into DB.

Normalises malformed rider IDs without modifying source files:
  R7   → R007
  r19  → R019
  (general rule: strip, uppercase, re-pad to 3 digits)

Uses INSERT ... ON CONFLICT DO NOTHING so repeated calls are safe.
"""
from __future__ import annotations

import csv
import logging
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from config import settings
from models import Rider, Trip, PayoutLine

log = logging.getLogger(__name__)


def _normalize_rider_id(raw: str) -> str:
    """Normalise to canonical RXXX (3-digit, uppercase).

    Examples:
        R001 → R001   R7 → R007   r19 → R019   R035 → R035
    """
    raw = raw.strip()
    m = re.fullmatch(r"[Rr](\d+)", raw)
    if not m:
        return raw
    return f"R{int(m.group(1)):03d}"


def seed_riders(db: Session, csv_path: str) -> int:
    path = Path(csv_path)
    if not path.exists():
        log.warning("riders.csv not found at %s — skipping", csv_path)
        return 0

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "rider_id": _normalize_rider_id(row["rider_id"]),
                "name": row["name"].strip(),
                "city": row["city"].strip(),
                "joined_on": datetime.strptime(row["joined_on"].strip(), "%Y-%m-%d").date(),
            })

    if not rows:
        return 0

    stmt = pg_insert(Rider).values(rows).on_conflict_do_nothing(index_elements=["rider_id"])
    result = db.execute(stmt)
    db.commit()
    inserted = result.rowcount
    log.info("Seeded riders: %d inserted (of %d)", inserted, len(rows))
    return inserted


def seed_trips(db: Session, csv_path: str) -> int:
    path = Path(csv_path)
    if not path.exists():
        log.warning("trips.csv not found at %s — skipping", csv_path)
        return 0

    # Build set of known rider IDs for fast FK check
    known_riders: set[str] = {r[0] for r in db.execute(text("SELECT rider_id FROM riders")).fetchall()}

    rows = []
    skipped_rider = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rider_id = _normalize_rider_id(row["rider_id"])
            if rider_id not in known_riders:
                skipped_rider += 1
                continue

            started_at_str = row["started_at"].strip()
            if started_at_str.endswith("Z"):
                started_at_str = started_at_str[:-1] + "+00:00"
            started_at = datetime.fromisoformat(started_at_str)

            rows.append({
                "trip_id": row["trip_id"].strip(),
                "rider_id": rider_id,
                "started_at": started_at,
                "status": row["status"].strip(),
                "distance_km": Decimal(row["distance_km"].strip()),
                "surge_multiplier": Decimal(row["surge_multiplier"].strip()),
            })

    if skipped_rider:
        log.warning("Skipped %d trips with unknown rider_id", skipped_rider)

    if not rows:
        return 0

    # Insert in chunks to avoid huge batches
    CHUNK = 500
    inserted = 0
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i : i + CHUNK]
        stmt = pg_insert(Trip).values(chunk).on_conflict_do_nothing(index_elements=["trip_id"])
        result = db.execute(stmt)
        db.commit()
        inserted += result.rowcount

    log.info("Seeded trips: %d inserted (of %d), %d skipped (unknown rider)", inserted, len(rows), skipped_rider)
    return inserted


def seed_payout_lines(db: Session, csv_path: str) -> int:
    path = Path(csv_path)
    if not path.exists():
        log.warning("payout_lines.csv not found at %s — skipping", csv_path)
        return 0

    known_riders: set[str] = {r[0] for r in db.execute(text("SELECT rider_id FROM riders")).fetchall()}

    rows = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rider_id = _normalize_rider_id(row["rider_id"])
            if rider_id not in known_riders:
                skipped += 1
                continue

            rows.append({
                "line_id": row["line_id"].strip(),
                "payout_date": datetime.strptime(row["payout_date"].strip(), "%Y-%m-%d").date(),
                "rider_id": rider_id,
                "line_type": row["line_type"].strip(),
                "trip_id": row.get("trip_id", "").strip() or None,
                "amount": int(row["amount"].strip()),
            })

    if not rows:
        return 0

    CHUNK = 500
    inserted = 0
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i : i + CHUNK]
        stmt = pg_insert(PayoutLine).values(chunk).on_conflict_do_nothing(index_elements=["line_id"])
        result = db.execute(stmt)
        db.commit()
        inserted += result.rowcount

    log.info("Seeded payout_lines: %d inserted (of %d), %d skipped", inserted, len(rows), skipped)
    return inserted


def run_all(db: Session) -> None:
    """Seed all tables in dependency order."""
    seed_riders(db, settings.riders_csv)
    seed_trips(db, settings.trips_csv)
    seed_payout_lines(db, settings.payout_lines_csv)
