"""Pure payout calculation functions — no I/O, no side effects.

Policy (docs/policy.md):
  - ₹25 base + ₹6 per km beyond the first 2 km.
  - Surge multiplier applies to the WHOLE trip fare (base + distance portion).
  - Round each trip fare to the nearest rupee; 0.5 rounds UP (ROUND_HALF_UP).
  - ₹150 daily incentive if rider completed >= 12 trips on the same IST calendar day.
  - ₹10 penalty for every trip the RIDER cancels.
  - Customer cancellations: no fare, no penalty.
  - "Day" = IST calendar day.
  - Dispute window = last 7 days.
"""
from __future__ import annotations

from datetime import datetime, date, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence

IST = timezone(timedelta(hours=5, minutes=30))
BASE_FARE = Decimal("25")
PER_KM = Decimal("6")
FREE_KM = Decimal("2")
INCENTIVE_THRESHOLD = 12
INCENTIVE_AMOUNT = 150   # rupees
CANCEL_PENALTY = 10      # rupees per rider-cancelled trip


def trip_fare(distance_km: Decimal | float | str, surge: Decimal | float | str) -> int:
    """Return the integer rupee fare for one completed trip.

    Args:
        distance_km: trip distance in kilometres.
        surge:       surge multiplier (1.0 means no surge).

    Returns:
        Fare in whole rupees, rounded ROUND_HALF_UP.

    >>> trip_fare(1.0, 1.0)   # <= 2 km, no surge → base only
    25
    >>> trip_fare(2.0, 1.0)   # exactly 2 km, still base only
    25
    >>> trip_fare(3.0, 1.0)   # 1 km beyond → 25 + 6 = 31
    31
    >>> trip_fare(5.0, 1.0)   # 3 km beyond → 25 + 18 = 43
    43
    >>> trip_fare(5.0, 1.2)   # surge on full fare → 43 * 1.2 = 51.6 → 52
    52
    >>> trip_fare(3.0, 1.5)   # 31 * 1.5 = 46.5 → 47 (ROUND_HALF_UP)
    47
    """
    d = Decimal(str(distance_km))
    s = Decimal(str(surge))
    extra_km = max(d - FREE_KM, Decimal("0"))
    fare = (BASE_FARE + PER_KM * extra_km) * s
    return int(fare.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def daily_incentive_amount(completed_count: int) -> int:
    """Return ₹150 if the rider completed >= 12 trips that day, else 0.

    >>> daily_incentive_amount(11)
    0
    >>> daily_incentive_amount(12)
    150
    >>> daily_incentive_amount(20)
    150
    """
    return INCENTIVE_AMOUNT if completed_count >= INCENTIVE_THRESHOLD else 0


def cancellation_penalty_amount(rider_cancel_count: int) -> int:
    """Return the total penalty (positive number) for rider-cancelled trips.

    Caller stores/applies this as a negative amount in the ledger.

    >>> cancellation_penalty_amount(0)
    0
    >>> cancellation_penalty_amount(2)
    20
    """
    return CANCEL_PENALTY * rider_cancel_count


def utc_to_ist_date(utc_dt: datetime) -> date:
    """Convert a UTC-aware datetime to its IST calendar date.

    >>> from datetime import datetime, timezone
    >>> utc_to_ist_date(datetime(2026, 9, 19, 20, 0, tzinfo=timezone.utc))
    datetime.date(2026, 9, 20)
    """
    return (utc_dt.astimezone(IST)).date()


def compute_expected_payout(
    trips: Sequence[dict],
    for_date: date,
) -> dict:
    """Compute the expected payout for a rider on a specific IST calendar day.

    Each element in `trips` must have:
      - started_at: datetime (UTC-aware)
      - status: str  ('completed' | 'cancelled_by_rider' | 'cancelled_by_customer')
      - distance_km: Decimal or float
      - surge_multiplier: Decimal or float

    Returns a dict:
      {
        "fare_total": int,          # sum of all trip fares (rupees)
        "incentive": int,           # 0 or 150
        "penalty": int,             # total penalty (positive; apply as negative)
        "net_payout": int,          # fare_total + incentive - penalty
        "completed_count": int,
        "rider_cancel_count": int,
        "trip_breakdown": [{"trip_id": ..., "fare": ...}, ...],
      }
    """
    day_trips = [
        t for t in trips
        if utc_to_ist_date(t["started_at"]) == for_date
    ]

    fare_total = 0
    completed_count = 0
    rider_cancel_count = 0
    breakdown = []

    for t in day_trips:
        tid = t.get("trip_id", "?")
        status = t["status"]
        if status == "completed":
            fare = trip_fare(t["distance_km"], t["surge_multiplier"])
            fare_total += fare
            completed_count += 1
            breakdown.append({"trip_id": tid, "fare": fare})
        elif status == "cancelled_by_rider":
            rider_cancel_count += 1
            breakdown.append({"trip_id": tid, "fare": 0, "penalty": CANCEL_PENALTY})
        # cancelled_by_customer: skip entirely

    incentive = daily_incentive_amount(completed_count)
    penalty = cancellation_penalty_amount(rider_cancel_count)
    net = fare_total + incentive - penalty

    return {
        "fare_total": fare_total,
        "incentive": incentive,
        "penalty": penalty,
        "net_payout": net,
        "completed_count": completed_count,
        "rider_cancel_count": rider_cancel_count,
        "trip_breakdown": breakdown,
    }


def dispute_window_start(reference_date: date | None = None) -> date:
    """Return the earliest IST date that falls within the 7-day dispute window.

    >>> from datetime import date
    >>> dispute_window_start(date(2026, 9, 22))
    datetime.date(2026, 9, 15)
    """
    ref = reference_date or datetime.now(IST).date()
    return ref - timedelta(days=7)
