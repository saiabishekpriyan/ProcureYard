"""ORM models — 7 tables.

Table map:
  riders          — seed from riders.csv
  trips           — seed from trips.csv
  payout_lines    — seed from payout_lines.csv
  messages        — incoming webhook messages (UNIQUE on message_id for dedup)
  audit_trace     — ordered agent steps for /trace/{rider_id}
  ops_queue       — items waiting for human action for /ops/pending
  payment_attempts— PaySwift call log (UNIQUE on idem_key for exactly-once)
"""
import uuid
from datetime import datetime, date
from decimal import Decimal

from sqlalchemy import (
    String, Integer, Numeric, Date, DateTime, Text, JSON,
    ForeignKey, UniqueConstraint, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


# ── Seed tables ────────────────────────────────────────────────────────────────

class Rider(Base):
    __tablename__ = "riders"

    rider_id:  Mapped[str] = mapped_column(String(10), primary_key=True)
    name:      Mapped[str] = mapped_column(String(120))
    city:      Mapped[str] = mapped_column(String(80))
    joined_on: Mapped[date] = mapped_column(Date)


class Trip(Base):
    __tablename__ = "trips"

    trip_id:          Mapped[str]     = mapped_column(String(20), primary_key=True)
    rider_id:         Mapped[str]     = mapped_column(String(10), ForeignKey("riders.rider_id", ondelete="CASCADE"))
    started_at:       Mapped[datetime]= mapped_column(DateTime(timezone=True))   # stored as UTC
    status:           Mapped[str]     = mapped_column(String(40))                # completed | cancelled_by_rider | cancelled_by_customer
    distance_km:      Mapped[Decimal] = mapped_column(Numeric(8, 2))
    surge_multiplier: Mapped[Decimal] = mapped_column(Numeric(4, 2))


class PayoutLine(Base):
    __tablename__ = "payout_lines"

    line_id:     Mapped[str] = mapped_column(String(20), primary_key=True)
    payout_date: Mapped[date]= mapped_column(Date)                               # IST calendar date
    rider_id:    Mapped[str] = mapped_column(String(10), ForeignKey("riders.rider_id", ondelete="CASCADE"))
    line_type:   Mapped[str] = mapped_column(String(40))                         # trip | daily_incentive | cancellation_penalty
    trip_id:     Mapped[str | None] = mapped_column(String(20), nullable=True)
    amount:      Mapped[int] = mapped_column(Integer)                            # rupees; negative for penalties


# ── Application tables ─────────────────────────────────────────────────────────

class Message(Base):
    """Incoming webhook message. UNIQUE(message_id) absorbs vendor re-deliveries."""
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("message_id", name="uq_messages_message_id"),)

    id:          Mapped[str]      = mapped_column(String(36), primary_key=True, default=_uuid)
    message_id:  Mapped[str]      = mapped_column(String(120))    # wamid from vendor
    rider_id:    Mapped[str]      = mapped_column(String(10))
    text:        Mapped[str]      = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    reply:       Mapped[str|None] = mapped_column(Text, nullable=True)          # agent reply stored here


class AuditTrace(Base):
    """One row per agent step, ordered by `at`. Powers GET /trace/{rider_id}."""
    __tablename__ = "audit_trace"

    id:       Mapped[str]      = mapped_column(String(36), primary_key=True, default=_uuid)
    rider_id: Mapped[str]      = mapped_column(String(10), index=True)
    at:       Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    type:     Mapped[str]      = mapped_column(String(20))   # message_in | tool_call | decision | reply | error
    name:     Mapped[str]      = mapped_column(String(120))
    input:    Mapped[dict|None]= mapped_column(JSON, nullable=True)
    output:   Mapped[dict|str|None] = mapped_column(JSON, nullable=True)


class OpsQueueItem(Base):
    """Items waiting for human approval or escalation. Powers GET /ops/pending."""
    __tablename__ = "ops_queue"

    id:          Mapped[str]          = mapped_column(String(36), primary_key=True, default=_uuid)
    rider_id:    Mapped[str]          = mapped_column(String(10), index=True)
    type:        Mapped[str]          = mapped_column(String(20))         # approval | escalation
    amount:      Mapped[int|None]     = mapped_column(Integer, nullable=True)
    reason:      Mapped[str]          = mapped_column(Text)
    created_at:  Mapped[datetime]     = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[datetime|None]= mapped_column(DateTime(timezone=True), nullable=True)
    resolution:  Mapped[str|None]     = mapped_column(String(20), nullable=True)  # approved | rejected | null


class PaymentAttempt(Base):
    """Tracks every PaySwift POST. UNIQUE(idem_key) prevents double-charge rows."""
    __tablename__ = "payment_attempts"
    __table_args__ = (UniqueConstraint("idem_key", name="uq_payment_attempts_idem_key"),)

    id:         Mapped[str]       = mapped_column(String(36), primary_key=True, default=_uuid)
    rider_id:   Mapped[str]       = mapped_column(String(10), index=True)
    idem_key:   Mapped[str]       = mapped_column(String(200))
    payout_id:  Mapped[str|None]  = mapped_column(String(50), nullable=True)   # PaySwift payout_id on success
    amount:     Mapped[int]       = mapped_column(Integer)
    reference:  Mapped[str|None]  = mapped_column(String(200), nullable=True)
    status:     Mapped[str]       = mapped_column(String(20), default="pending") # pending | success | failed
    error:      Mapped[str|None]  = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
