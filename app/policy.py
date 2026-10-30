"""Deterministic policy guards — NO LLM involvement.

All money-movement decisions go through these functions.
The LLM is allowed to call tools that eventually invoke these guards,
but it can never bypass them.
"""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

AUTO_PAY_LIMIT = 200       # ₹ per dispute
DAILY_AUTO_PAY_ONCE = True # only one auto-pay per rider per IST calendar day


class PolicyViolation(Exception):
    """Raised when a proposed payment violates a policy rule."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def check_auto_pay_eligible(
    amount: int,
    rider_id: str,
    ist_date: date,
    db: "Session",
) -> None:
    """Raise PolicyViolation if the auto-pay cannot proceed.

    Rules checked (in order):
    1. Amount must be > 0.
    2. Amount must not exceed ₹200.
    3. No prior successful auto-pay for this rider on this IST calendar day.
    4. Amount must not exceed the calculated shortfall (caller must supply shortfall
       as `amount`; this function trusts it was computed correctly).

    Args:
        amount:   Proposed payout amount in rupees (whole number).
        rider_id: Rider being paid.
        ist_date: IST calendar date of the dispute day.
        db:       Live SQLAlchemy session.

    Raises:
        PolicyViolation with a human-readable reason.
    """
    from models import PaymentAttempt
    from sqlalchemy import func

    if amount <= 0:
        raise PolicyViolation("zero_amount", "Payout amount must be positive.")

    if amount > AUTO_PAY_LIMIT:
        raise PolicyViolation(
            "exceeds_auto_pay_limit",
            f"Amount ₹{amount} exceeds the ₹{AUTO_PAY_LIMIT} auto-pay limit. "
            "This requires ops approval.",
        )

    # Check if any successful payment already went out today for this rider
    idem_prefix = f"payout-{rider_id}-{ist_date.isoformat()}"
    existing = (
        db.query(PaymentAttempt)
        .filter(
            PaymentAttempt.rider_id == rider_id,
            PaymentAttempt.status == "success",
            PaymentAttempt.idem_key.like(f"{idem_prefix}%"),
        )
        .first()
    )
    if existing:
        raise PolicyViolation(
            "already_paid_today",
            f"Rider {rider_id} already received an auto-pay of ₹{existing.amount} "
            f"on {ist_date.isoformat()} (payout {existing.payout_id}). "
            "A second payment today requires ops approval.",
        )


def make_idem_key(rider_id: str, ist_date: date) -> str:
    """Return the canonical idempotency key for a rider's daily payout.

    Deterministic → safe to retry across process restarts.
    """
    return f"payout-{rider_id}-{ist_date.isoformat()}"


def shortfall(expected: int, already_paid: int) -> int:
    """Return the amount still owed; never negative.

    >>> shortfall(150, 100)
    50
    >>> shortfall(100, 150)
    0
    """
    return max(expected - already_paid, 0)
