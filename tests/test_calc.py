"""Unit tests for calc.py — pure payout calculation functions.

Run: pytest tests/test_calc.py -v
No DB, no HTTP, no mocks needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import pytest
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal

from calc import (
    trip_fare,
    daily_incentive_amount,
    cancellation_penalty_amount,
    utc_to_ist_date,
    compute_expected_payout,
    dispute_window_start,
)

IST = timezone(timedelta(hours=5, minutes=30))


# ── trip_fare ──────────────────────────────────────────────────────────────────

class TestTripFare:
    def test_base_only_under_2km(self):
        assert trip_fare(1.0, 1.0) == 25

    def test_base_only_exactly_2km(self):
        assert trip_fare(2.0, 1.0) == 25

    def test_one_km_beyond(self):
        # 25 + 6*1 = 31
        assert trip_fare(3.0, 1.0) == 31

    def test_three_km_beyond(self):
        # 25 + 6*3 = 43
        assert trip_fare(5.0, 1.0) == 43

    def test_surge_1_2_rounds_up(self):
        # 43 * 1.2 = 51.6 → 52 (ROUND_HALF_UP)
        assert trip_fare(5.0, 1.2) == 52

    def test_surge_1_5_half_rounds_up(self):
        # 31 * 1.5 = 46.5 → 47 (ROUND_HALF_UP)
        assert trip_fare(3.0, 1.5) == 47

    def test_surge_applies_to_base_too(self):
        # No extra km: 25 * 1.5 = 37.5 → 38
        assert trip_fare(1.0, 1.5) == 38

    def test_no_surge(self):
        # surge_multiplier = 1.0 means no surge
        assert trip_fare(4.0, 1.0) == 25 + 12  # 37

    def test_decimal_distance(self):
        # 2.5 km: 0.5 extra → 25 + 3 = 28; * 1.0 = 28
        assert trip_fare(2.5, 1.0) == 28

    def test_half_rupee_rounds_up(self):
        # Designed to hit exactly x.5
        # distance 2 km: fare = 25. surge 1.02 → 25*1.02 = 25.5 → 26
        assert trip_fare(2.0, 1.02) == 26

    def test_string_inputs(self):
        # Accepts string inputs (from CSV)
        assert trip_fare("3.0", "1.0") == 31


# ── daily_incentive_amount ─────────────────────────────────────────────────────

class TestDailyIncentive:
    def test_zero_trips(self):
        assert daily_incentive_amount(0) == 0

    def test_eleven_trips(self):
        assert daily_incentive_amount(11) == 0

    def test_twelve_trips(self):
        assert daily_incentive_amount(12) == 150

    def test_twenty_trips(self):
        assert daily_incentive_amount(20) == 150


# ── cancellation_penalty_amount ────────────────────────────────────────────────

class TestCancellationPenalty:
    def test_no_cancellations(self):
        assert cancellation_penalty_amount(0) == 0

    def test_one_cancel(self):
        assert cancellation_penalty_amount(1) == 10

    def test_two_cancels(self):
        assert cancellation_penalty_amount(2) == 20

    def test_five_cancels(self):
        assert cancellation_penalty_amount(5) == 50


# ── utc_to_ist_date ────────────────────────────────────────────────────────────

class TestUtcToIstDate:
    def test_before_midnight_ist(self):
        # 19:00 UTC = 00:30 IST next day
        dt = datetime(2026, 9, 19, 19, 0, tzinfo=timezone.utc)
        assert utc_to_ist_date(dt) == date(2026, 9, 20)

    def test_early_morning_utc(self):
        # 02:30 UTC = 08:00 IST same day
        dt = datetime(2026, 9, 12, 2, 30, tzinfo=timezone.utc)
        assert utc_to_ist_date(dt) == date(2026, 9, 12)

    def test_just_before_ist_midnight(self):
        # 18:29 UTC = 23:59 IST same day
        dt = datetime(2026, 9, 12, 18, 29, tzinfo=timezone.utc)
        assert utc_to_ist_date(dt) == date(2026, 9, 12)

    def test_exactly_ist_midnight(self):
        # 18:30 UTC = 00:00 IST next day
        dt = datetime(2026, 9, 12, 18, 30, tzinfo=timezone.utc)
        assert utc_to_ist_date(dt) == date(2026, 9, 13)


# ── compute_expected_payout ────────────────────────────────────────────────────

def _make_trip(trip_id, status, distance_km, surge, utc_hour=8, utc_date=date(2026, 9, 20)):
    return {
        "trip_id": trip_id,
        "started_at": datetime(utc_date.year, utc_date.month, utc_date.day,
                               utc_hour, 0, tzinfo=timezone.utc),
        "status": status,
        "distance_km": distance_km,
        "surge_multiplier": surge,
    }


class TestComputeExpectedPayout:
    FOR_DATE = date(2026, 9, 20)

    def test_single_completed_no_surge(self):
        trips = [_make_trip("T1", "completed", 3.0, 1.0)]
        result = compute_expected_payout(trips, self.FOR_DATE)
        assert result["fare_total"] == 31
        assert result["incentive"] == 0
        assert result["penalty"] == 0
        assert result["net_payout"] == 31
        assert result["completed_count"] == 1

    def test_customer_cancel_excluded(self):
        trips = [_make_trip("T1", "cancelled_by_customer", 5.0, 1.0)]
        result = compute_expected_payout(trips, self.FOR_DATE)
        assert result["fare_total"] == 0
        assert result["penalty"] == 0
        assert result["net_payout"] == 0

    def test_rider_cancel_penalty(self):
        trips = [_make_trip("T1", "cancelled_by_rider", 5.0, 1.0)]
        result = compute_expected_payout(trips, self.FOR_DATE)
        assert result["fare_total"] == 0
        assert result["penalty"] == 10
        assert result["net_payout"] == -10
        assert result["rider_cancel_count"] == 1

    def test_incentive_at_12_completed(self):
        trips = [_make_trip(f"T{i}", "completed", 1.0, 1.0) for i in range(12)]
        result = compute_expected_payout(trips, self.FOR_DATE)
        assert result["completed_count"] == 12
        assert result["incentive"] == 150

    def test_no_incentive_at_11(self):
        trips = [_make_trip(f"T{i}", "completed", 1.0, 1.0) for i in range(11)]
        result = compute_expected_payout(trips, self.FOR_DATE)
        assert result["incentive"] == 0

    def test_mixed_statuses(self):
        trips = [
            _make_trip("T1", "completed", 5.0, 1.0),          # fare=43
            _make_trip("T2", "cancelled_by_rider", 0, 1.0),   # penalty=10
            _make_trip("T3", "cancelled_by_customer", 5.0, 1.0),  # nothing
        ]
        result = compute_expected_payout(trips, self.FOR_DATE)
        assert result["fare_total"] == 43
        assert result["penalty"] == 10
        assert result["net_payout"] == 33
        assert result["completed_count"] == 1

    def test_only_trips_on_date_counted(self):
        """Trips on a different IST date must be ignored."""
        other_date = date(2026, 9, 19)
        trips = [
            _make_trip("T1", "completed", 3.0, 1.0, utc_date=other_date),
            _make_trip("T2", "completed", 3.0, 1.0, utc_date=self.FOR_DATE),
        ]
        result = compute_expected_payout(trips, self.FOR_DATE)
        assert result["completed_count"] == 1

    def test_surge_applied_to_full_fare(self):
        # 5km: 25 + 18 = 43. surge 1.2 → 43*1.2=51.6→52
        trips = [_make_trip("T1", "completed", 5.0, 1.2)]
        result = compute_expected_payout(trips, self.FOR_DATE)
        assert result["fare_total"] == 52


# ── dispute_window_start ───────────────────────────────────────────────────────

class TestDisputeWindowStart:
    def test_7_days_back(self):
        ref = date(2026, 9, 22)
        assert dispute_window_start(ref) == date(2026, 9, 15)
