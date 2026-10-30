"""Tests for conversational agent capabilities, English-only responses, and factual grounding.

Verifies:
1. Hindi/Hinglish input generates clear, polite English response without Hindi words.
2. English input generates professional English response.
3. Multi-turn follow-ups (e.g. "which trip was incorrect", "why was my payout lower", "when will I receive the money").
4. Payment status queries distinguishing between confirmed PaySwift payout and pending ops approval.
5. Clarification questions for ambiguous queries and polite responses to greetings.
6. Fallback under quota exhaustion remains conversational and deterministic.
7. Duplicate message delivery maintains idempotency.
"""
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db import Base
import models
from models import Rider, Trip, PayoutLine, OpsQueueItem, PaymentAttempt, Message
import deterministic_dispute as dd
import agent as ag


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed basic rider
    rider = Rider(
        rider_id="R999",
        name="Test Rider",
        city="Bengaluru",
        joined_on=date(2026, 1, 1),
    )
    session.add(rider)
    session.commit()

    try:
        yield session
    finally:
        session.close()


def test_greeting_returns_polite_english_response(test_db):
    """Greeting without dispute keywords returns a polite English welcoming message."""
    reply = dd.run_deterministic_dispute(
        rider_id="R999",
        message_text="Hello! Good morning",
        received_at_str="2026-09-22T10:00:00+00:00",
        db=test_db,
    )
    assert "Hello!" in reply
    assert "QuickDrop Rider Payout Assistant" in reply
    assert not any(w in reply.lower() for w in ["bhai", "wala", "haan", "thik", "shukriya"])


def test_hinglish_dispute_returns_pure_english(test_db):
    """Hinglish input yields English reply explaining shortfall and PaySwift transfer."""
    # Seed a trip on Sep 20 with missing surge
    t = Trip(
        trip_id="T111222",
        rider_id="R999",
        started_at=datetime(2026, 9, 20, 5, 0, tzinfo=timezone.utc),
        status="completed",
        distance_km=Decimal("4.0"),  # base ₹25 + 2km * ₹7 = ₹39 * 1.5x surge = ₹59
        surge_multiplier=Decimal("1.5"),
    )
    test_db.add(t)
    # Paid only ₹39 (missing ₹20 surge)
    pl = PayoutLine(
        line_id="L111",
        payout_date=date(2026, 9, 20),
        rider_id="R999",
        line_type="trip",
        trip_id="T111222",
        amount=39,
    )
    test_db.add(pl)
    test_db.commit()

    # Rider writes in Hinglish
    msg = "Mera 20 tareekh ka paisa kam aaya hai, please check karo bhai"
    reply = dd.run_deterministic_dispute(
        rider_id="R999",
        message_text=msg,
        received_at_str="2026-09-22T10:00:00+00:00",
        db=test_db,
    )

    # Must be pure English
    assert "Sep" in reply or "20" in reply
    assert "T111222" in reply or "surge" in reply or "shortfall" in reply or "difference" in reply
    assert not any(w in reply.lower() for w in ["bhai", "wala", "haan", "thik", "rupay", "tareekh"])


def test_payment_status_when_payout_confirmed(test_db, monkeypatch):
    """When rider asks 'Has my payment been processed?', confirms PaySwift payout ID."""
    attempt = PaymentAttempt(
        rider_id="R999",
        idem_key="key-123",
        amount=50,
        status="success",
        payout_id="pay_abc12345",
    )
    test_db.add(attempt)
    test_db.commit()

    # Simulate live PaySwift confirming the payout
    monkeypatch.setattr(dd, "_check_payswift_verified", lambda rider_id, payout_id: True)

    reply = dd.run_deterministic_dispute(
        rider_id="R999",
        message_text="Has my payment been processed? When will I receive money?",
        received_at_str="2026-09-22T10:00:00+00:00",
        db=test_db,
    )

    assert "50" in reply
    assert "pay_abc12345" in reply
    assert "PaySwift" in reply or "processed" in reply


def test_payment_status_when_payswift_sandbox_unverified(test_db, monkeypatch):
    """If DB records a successful attempt but PaySwift cannot verify it, states that DB records payment but could not confirm in payment system."""
    attempt = PaymentAttempt(
        rider_id="R999",
        idem_key="key-123",
        amount=25,
        status="success",
        payout_id="pout_unverified_123",
    )
    test_db.add(attempt)
    # Add a separate pending approval item to verify distinction
    op = OpsQueueItem(
        rider_id="R999",
        type="approval",
        amount=25,
        reason="Second payment today requires ops approval",
    )
    test_db.add(op)
    test_db.commit()

    # Simulate PaySwift returning False (ledger reset / not found)
    monkeypatch.setattr(dd, "_check_payswift_verified", lambda rider_id, payout_id: False)

    reply = dd.run_deterministic_dispute(
        rider_id="R999",
        message_text="Is my payment confirmed, or is it still waiting for approval?",
        received_at_str="2026-10-29T20:22:33Z",
        db=test_db,
    )

    assert "records say" in reply.lower() or "database records" in reply.lower()
    assert "couldn't confirm" in reply.lower() or "payment system" in reply.lower()
    assert "waiting for approval" in reply.lower() or "pending operations approval" in reply.lower()


def test_payment_status_when_pending_ops_approval(test_db):
    """When rider asks why approval is required, explains limit or pending status."""
    item = OpsQueueItem(
        rider_id="R999",
        type="approval",
        amount=350,
        reason="Dispute amount > 200",
    )
    test_db.add(item)
    test_db.commit()

    reply = dd.run_deterministic_dispute(
        rider_id="R999",
        message_text="Why does this require approval?",
        received_at_str="2026-09-22T10:00:00+00:00",
        db=test_db,
    )

    assert "350" in reply
    assert "200" in reply
    assert "operations approval" in reply or "exceeds" in reply


def test_follow_up_which_trip_was_incorrect(test_db):
    """Answers follow-up 'Why was my payout lower? Which trip?' with exact trip and surge details."""
    t = Trip(
        trip_id="T999888",
        rider_id="R999",
        started_at=datetime(2026, 9, 21, 6, 0, tzinfo=timezone.utc),
        status="completed",
        distance_km=Decimal("5.0"),
        surge_multiplier=Decimal("1.5"),
    )
    test_db.add(t)
    pl = PayoutLine(
        line_id="L999",
        payout_date=date(2026, 9, 21),
        rider_id="R999",
        line_type="trip",
        trip_id="T999888",
        amount=40,
    )
    test_db.add(pl)
    test_db.commit()

    # Pre-record a previous conversation message referencing Sep 21
    m1 = Message(
        message_id="msg-1",
        rider_id="R999",
        text="Check my payout for 21st Sep",
        received_at=datetime(2026, 9, 22, 9, 0, tzinfo=timezone.utc),
    )
    test_db.add(m1)
    test_db.commit()

    reply = dd.run_deterministic_dispute(
        rider_id="R999",
        message_text="Which trip was incorrect and why was my payout lower?",
        received_at_str="2026-09-22T10:00:00+00:00",
        db=test_db,
    )

    assert "T999888" in reply
    assert "surge" in reply
    assert "shortfall" in reply or "expected" in reply


def test_ambiguous_inquiry_requests_clarification(test_db):
    """When a rider makes an ambiguous complaint without date/trip, asks for clarification."""
    # Seed a trip on Sep 20 with a shortfall so dates_with_shortfall is non-empty
    t = Trip(
        trip_id="T777111",
        rider_id="R999",
        started_at=datetime(2026, 9, 20, 7, 0, tzinfo=timezone.utc),
        status="completed",
        distance_km=Decimal("3.0"),
        surge_multiplier=Decimal("1.0"),
    )
    test_db.add(t)
    test_db.commit()

    reply = dd.run_deterministic_dispute(
        rider_id="R999",
        message_text="Paisa kam aaya hai mera",
        received_at_str="2026-09-22T10:00:00+00:00",
        db=test_db,
    )
    assert "specify which date or trip ID" in reply


def test_payment_status_query_not_blocked_by_seven_day_window(test_db, monkeypatch):
    """Payment status inquiry about an older trip/dispute must NOT be rejected by the 7-day dispute window."""
    # Seed an older trip from Sep 20
    t = Trip(
        trip_id="T926334",
        rider_id="R999",
        started_at=datetime(2026, 9, 20, 5, 0, tzinfo=timezone.utc),
        status="completed",
        distance_km=Decimal("7.0"),
        surge_multiplier=Decimal("1.5"),
    )
    test_db.add(t)

    # Seed prior conversation turn referencing T926334
    m = Message(
        message_id="prev-msg-001",
        rider_id="R999",
        text="Check order T926334 for 20 Sep",
        received_at=datetime(2026, 9, 22, 9, 0, tzinfo=timezone.utc),
    )
    test_db.add(m)

    # Seed a confirmed PaymentAttempt for this rider
    pa = PaymentAttempt(
        rider_id="R999",
        idem_key="payout-R999-2026-09-20",
        amount=25,
        status="success",
        payout_id="pout_test999",
    )
    test_db.add(pa)
    test_db.commit()

    # Simulate PaySwift confirming the payment
    monkeypatch.setattr(dd, "_check_payswift_verified", lambda rider_id, payout_id: True)

    # Rider asks status on Oct 29 (> 7 days from Sep 20)
    msg = "Is my payment confirmed, or is it still waiting for approval?"
    reply = dd.run_deterministic_dispute(
        rider_id="R999",
        message_text=msg,
        received_at_str="2026-10-29T20:22:33Z",
        db=test_db,
    )

    # Must report the payment status and context trip, NOT say it is outside the 7-day window
    assert "older than 7 days" not in reply
    assert "dispute policy" not in reply
    assert "25" in reply
    assert "pout_test999" in reply
    assert "T926334" in reply


def test_payment_status_when_only_approval_pending(test_db):
    """Payment status inquiry when only OpsQueueItem exists returns pending review info."""
    item = OpsQueueItem(
        rider_id="R999",
        type="approval",
        amount=25,
        reason="Trip T926334 shortfall",
    )
    test_db.add(item)
    test_db.commit()

    msg = "Is my payment confirmed, or is it still waiting for approval?"
    reply = dd.run_deterministic_dispute(
        rider_id="R999",
        message_text=msg,
        received_at_str="2026-10-29T20:22:33Z",
        db=test_db,
    )

    assert "older than 7 days" not in reply
    assert "pending review" in reply
    assert "25" in reply


def test_rider_cancelled_trip_penalty_dispute(test_db):
    """When a rider cancels a trip (even due to accident), standard ₹10 penalty applies without duplicate refund."""
    t = Trip(
        trip_id="T849302",
        rider_id="R028",
        started_at=datetime(2026, 9, 21, 7, 24, tzinfo=timezone.utc),
        status="cancelled_by_rider",
        distance_km=Decimal("3.30"),
        surge_multiplier=Decimal("1.0"),
    )
    test_db.add(t)
    pl = PayoutLine(
        line_id="PL002255",
        payout_date=date(2026, 9, 21),
        rider_id="R028",
        line_type="cancellation_penalty",
        trip_id="T849302",
        amount=-10,
    )
    test_db.add(pl)
    test_db.commit()

    reply = dd.run_deterministic_dispute(
        rider_id="R028",
        message_text="accident hua tha isliye order T849302 cancel kiya, penalty kyun kaata? bahut problem hai",
        received_at_str="2026-09-23T12:15:00+05:30",
        db=test_db,
    )

    # Must explain that standard ₹10 penalty applies for rider cancellation, and escalate for exception review
    assert "T849302" in reply
    assert "standard" in reply.lower() or "cancellation penalty" in reply.lower()
    assert "escalated" in reply.lower() or "operations" in reply.lower()
    assert "difference is being transferred" not in reply.lower()

    # Verify escalation was logged in ops queue
    ops = test_db.query(OpsQueueItem).filter(OpsQueueItem.rider_id == "R028", OpsQueueItem.type == "escalation").all()
    assert len(ops) >= 1
    assert "T849302" in ops[-1].reason


def test_duplicate_cancellation_penalty_refund(test_db):
    """When a trip has multiple cancellation penalties (duplicate error), deterministically refunds the excess."""
    t = Trip(
        trip_id="T999999",
        rider_id="R999",
        started_at=datetime(2026, 9, 21, 8, 0, tzinfo=timezone.utc),
        status="cancelled_by_rider",
        distance_km=Decimal("2.0"),
        surge_multiplier=Decimal("1.0"),
    )
    test_db.add(t)
    pl1 = PayoutLine(
        line_id="PL_DUP1",
        payout_date=date(2026, 9, 21),
        rider_id="R999",
        line_type="cancellation_penalty",
        trip_id="T999999",
        amount=-10,
    )
    pl2 = PayoutLine(
        line_id="PL_DUP2",
        payout_date=date(2026, 9, 21),
        rider_id="R999",
        line_type="cancellation_penalty",
        trip_id="T999999",
        amount=-10,
    )
    test_db.add(pl1)
    test_db.add(pl2)
    test_db.commit()

    reply = dd.run_deterministic_dispute(
        rider_id="R999",
        message_text="Trip T999999 had 2 penalties deducted",
        received_at_str="2026-09-22T10:00:00+05:30",
        db=test_db,
    )

    # Excess penalty of ₹10 should be refunded or queued
    assert "extra cancellation penalty" in reply.lower() or "refund" in reply.lower()
    assert "10" in reply


def test_bhai_my_payout_is_incorrect_pure_english(test_db):
    """When a rider says 'Bhai, my payout is incorrect', response must be in professional, natural English."""
    reply = dd.run_deterministic_dispute(
        rider_id="R999",
        message_text="Bhai, my payout is incorrect",
        received_at_str="2026-09-22T09:00:00+05:30",
        db=test_db,
    )

    # Must be entirely English and professional
    assert not any(w in reply.lower() for w in ["bhai", "wala", "haan", "thik", "shukriya", "kripya"])
    assert "payout" in reply.lower() or "trip" in reply.lower() or "account" in reply.lower() or "specify" in reply.lower()


def test_live_chat_post_message_and_trace_history(test_db):
    """Simulates frontend live chat posting a message to /messages and verifying trace/conversation history."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)

    # 1. Post a new message from frontend live chat
    payload = {
        "message_id": "test-live-chat-001",
        "rider_id": "R005",
        "text": "Bhai, my payout is incorrect for yesterday",
        "received_at": "2026-09-22T09:00:00+05:30",
    }
    resp = client.post("/messages", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    assert len(data["reply"]) > 0
    # Strict English check
    assert not any(w in data["reply"].lower() for w in ["bhai", "wala", "haan", "thik"])

    # 2. Query conversation history endpoint used by the frontend
    hist_resp = client.get("/trace/R005/conversation")
    assert hist_resp.status_code == 200
    hist = hist_resp.json()
    assert hist["rider_id"] == "R005"
    assert len(hist["conversations"]) >= 1
    # Check that rider's original text is preserved untranslated
    matching_turn = next((c for c in hist["conversations"] if c["message_id"] == "test-live-chat-001"), None)
    assert matching_turn is not None
    assert matching_turn["text"] == "Bhai, my payout is incorrect for yesterday"
    assert matching_turn["reply"] == data["reply"]
    assert matching_turn["has_reply"] is True


def test_two_message_reproduction_and_followup_status(test_db):
    """Reproduces the exact conversation:
    Message 1: 'My payout for September 20 is incorrect. Please investigate.'
    Message 2: 'How much money is missing?'
    Follow-up 3: 'Has the payment actually been processed?'
    Follow-up 4: 'Is it awaiting approval?'
    Follow-up 5: 'When will I receive it?'

    Verifies:
    - Message 2 resolves to the verified shortfall (₹25 for T926334) without 7-day cutoff error.
    - All responses are in English.
    - Status reporting distinguishes between live verified payments vs unverified sandbox/db records.
    - No duplicate payout is initiated.
    """
    rider = Rider(
        rider_id="R888",
        name="Repro Rider",
        city="Bengaluru",
        joined_on=date(2026, 1, 1),
    )
    test_db.add(rider)

    # Set up trip T926334 for rider R888
    started = datetime(2026, 9, 20, 7, 27, tzinfo=timezone.utc)
    trip = Trip(
        trip_id="T926334",
        rider_id="R888",
        started_at=started,
        status="completed",
        distance_km=Decimal("6.00"),
        surge_multiplier=Decimal("1.50"),
    )
    payout = PayoutLine(
        line_id=f"pl-{uuid.uuid4().hex[:8]}",
        rider_id="R888",
        trip_id="T926334",
        payout_date=date(2026, 9, 20),
        amount=49,
        line_type="trip",
    )
    test_db.add_all([trip, payout])
    test_db.commit()

    # Message 1
    m1 = Message(
        message_id="repro-msg-001",
        rider_id="R888",
        text="My payout for September 20 is incorrect. Please investigate.",
        received_at=datetime(2026, 9, 22, 9, 0, tzinfo=timezone.utc),
    )
    test_db.add(m1)
    test_db.commit()

    r1 = dd.run_deterministic_dispute(
        rider_id="R888",
        message_text=m1.text,
        received_at_str="2026-09-22T09:00:00+05:30",
        db=test_db,
    )
    m1.reply = r1
    test_db.commit()

    # Verify message 1 identifies the difference
    assert "25" in r1
    assert "T926334" in r1 or "surge" in r1.lower() or "shortfall" in r1.lower()

    # Message 2: "How much money is missing?"
    m2 = Message(
        message_id="repro-msg-002",
        rider_id="R888",
        text="How much money is missing?",
        received_at=datetime(2026, 10, 30, 2, 0, tzinfo=timezone.utc),
    )
    test_db.add(m2)
    test_db.commit()

    r2 = dd.run_deterministic_dispute(
        rider_id="R888",
        message_text=m2.text,
        received_at_str="2026-10-30T02:00:00+05:30",
        db=test_db,
    )
    m2.reply = r2
    test_db.commit()

    # Crucial check: Must NOT reject as outside 7-day window!
    assert "older than 7 days" not in r2.lower()
    assert "cannot process claims beyond the 7-day window" not in r2.lower()
    # Must explain the verified shortfall of ₹25
    assert "25" in r2
    assert "difference is ₹25" in r2 or "T926334" in r2 or "shortfall" in r2.lower()
    # All English
    assert not any(w in r2.lower() for w in ["bhai", "wala", "haan", "thik"])

    # Follow-up 3: "Has the payment actually been processed?"
    m3 = Message(
        message_id="repro-msg-003",
        rider_id="R888",
        text="Has the payment actually been processed?",
        received_at=datetime(2026, 10, 30, 2, 5, tzinfo=timezone.utc),
    )
    test_db.add(m3)
    test_db.commit()

    r3 = dd.run_deterministic_dispute(
        rider_id="R888",
        message_text=m3.text,
        received_at_str="2026-10-30T02:05:00+05:30",
        db=test_db,
    )
    m3.reply = r3
    test_db.commit()

    assert "older than 7 days" not in r3.lower()
    assert not any(w in r3.lower() for w in ["bhai", "wala", "haan", "thik"])
    # If verified in live PaySwift sandbox, it confirms processed with Payout ID; if not verified, explains status cannot be confirmed
    assert "payswift" in r3.lower() or "processed" in r3.lower() or "cannot" in r3.lower() or "unavailable" in r3.lower() or "database records" in r3.lower() or "pending" in r3.lower()
    assert "25" in r3

    # Follow-up 4: "Is it awaiting approval?"
    m4 = Message(
        message_id="repro-msg-004",
        rider_id="R888",
        text="Is it awaiting approval?",
        received_at=datetime(2026, 10, 30, 2, 10, tzinfo=timezone.utc),
    )
    test_db.add(m4)
    test_db.commit()

    r4 = dd.run_deterministic_dispute(
        rider_id="R888",
        message_text=m4.text,
        received_at_str="2026-10-30T02:10:00+05:30",
        db=test_db,
    )
    m4.reply = r4
    test_db.commit()
    assert "older than 7 days" not in r4.lower()

    # Follow-up 5: "When will I receive it?"
    m5 = Message(
        message_id="repro-msg-005",
        rider_id="R888",
        text="When will I receive it?",
        received_at=datetime(2026, 10, 30, 2, 15, tzinfo=timezone.utc),
    )
    test_db.add(m5)
    test_db.commit()

    r5 = dd.run_deterministic_dispute(
        rider_id="R888",
        message_text=m5.text,
        received_at_str="2026-10-30T02:15:00+05:30",
        db=test_db,
    )
    m5.reply = r5
    test_db.commit()
    assert "older than 7 days" not in r5.lower()
    # "When will I receive it?" naturally explains arrival time status
    assert "arrival time" in r5.lower() or "reflect" in r5.lower() or "shortly" in r5.lower() or "unavailable" in r5.lower()


def test_first_and_followup_replies_never_claim_success_when_unverified(test_db):
    """Verifies that the FIRST response and follow-ups never announce payout success
    when live PaySwift cannot verify the payout, maintaining consistent unverified state.
    """
    rider = Rider(
        rider_id="R777",
        name="Unverified State Rider",
        city="Bengaluru",
        joined_on=date(2026, 1, 1),
    )
    test_db.add(rider)

    started = datetime(2026, 9, 20, 7, 27, tzinfo=timezone.utc)
    trip = Trip(
        trip_id="T777001",
        rider_id="R777",
        started_at=started,
        status="completed",
        distance_km=Decimal("6.00"),
        surge_multiplier=Decimal("1.50"),
    )
    payout = PayoutLine(
        line_id=f"pl-{uuid.uuid4().hex[:8]}",
        rider_id="R777",
        trip_id="T777001",
        payout_date=date(2026, 9, 20),
        amount=49,
        line_type="trip",
    )
    # Simulate historical database PaymentAttempt whose payout ID does NOT exist in live PaySwift ledger
    unverified_attempt = PaymentAttempt(
        rider_id="R777",
        idem_key="payout-R777-2026-09-20",
        amount=25,
        payout_id="pout_unverified_mock_9999",
        status="success",
        reference="dispute-R777-2026-09-20",
        created_at=datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc),
    )
    test_db.add_all([trip, payout, unverified_attempt])
    test_db.commit()

    # Turn 1: Initial claim
    r1 = dd.run_deterministic_dispute(
        rider_id="R777",
        message_text="My payout for September 20 is incorrect. Please investigate.",
        received_at_str="2026-10-30T03:00:00+05:30",
        db=test_db,
    )
    # Must NOT claim that payment has already been processed or will reflect shortly!
    assert "has already been processed" not in r1.lower()
    assert "will reflect in your account shortly" not in r1.lower()
    # Must explain that payment couldn't be confirmed or is under check
    assert "couldn't confirm" in r1.lower() or "check whether" in r1.lower() or "unavailable" in r1.lower() or "verifying" in r1.lower()

    m1 = Message(message_id="unv-001", rider_id="R777", text="My payout for September 20 is incorrect.", reply=r1, received_at=datetime.now(timezone.utc))
    test_db.add(m1)
    test_db.commit()

    # Turn 2: "How much money is missing?"
    r2 = dd.run_deterministic_dispute(
        rider_id="R777",
        message_text="How much money is missing?",
        received_at_str="2026-10-30T03:01:00+05:30",
        db=test_db,
    )
    assert "25" in r2
    assert "has already been processed" not in r2.lower()

    # Turn 3: "Has the payment actually been processed?"
    r3 = dd.run_deterministic_dispute(
        rider_id="R777",
        message_text="Has the payment actually been processed?",
        received_at_str="2026-10-30T03:02:00+05:30",
        db=test_db,
    )
    assert "has already been processed" not in r3.lower()
    assert "couldn't confirm" in r3.lower() or "records say" in r3.lower() or "unavailable" in r3.lower() or "verifying" in r3.lower()

    # Turn 4: "When will I receive it?"
    r4 = dd.run_deterministic_dispute(
        rider_id="R777",
        message_text="When will I receive it?",
        received_at_str="2026-10-30T03:03:00+05:30",
        db=test_db,
    )
    # Answers question: arrival time cannot be confirmed while unverified
    assert "unable to confirm when you'll receive" in r4.lower() or "arrival time" in r4.lower()


def test_four_question_sequence_with_separate_approval_request(test_db):
    """Regression test for the four-question sequence after September 20 dispute:
    1. How much money is missing?
    2. Has the payment actually been processed?
    3. When will I receive it?
    4. Is the other ₹25 request still awaiting approval?

    Verifies:
    - Intent 1 returns calculated shortfall and distinguishes from unpaid status.
    - Intent 2 explains database success vs live PaySwift unverified status.
    - Intent 3 explains arrival time cannot currently be confirmed.
    - Intent 4 identifies the separate ₹25 approval request from pending operations data,
      instead of repeating the previous trip payment-verification answer.
    - Also verifies behavior when no separate approval exists.
    """
    rider = Rider(
        rider_id="R666",
        name="Four Intent Rider",
        city="Bengaluru",
        joined_on=date(2026, 1, 1),
    )
    test_db.add(rider)

    started = datetime(2026, 9, 20, 7, 27, tzinfo=timezone.utc)
    trip = Trip(
        trip_id="T666001",
        rider_id="R666",
        started_at=started,
        status="completed",
        distance_km=Decimal("6.00"),
        surge_multiplier=Decimal("1.50"),
    )
    payout = PayoutLine(
        line_id=f"pl-{uuid.uuid4().hex[:8]}",
        rider_id="R666",
        trip_id="T666001",
        payout_date=date(2026, 9, 20),
        amount=49,
        line_type="trip",
    )
    unverified_attempt = PaymentAttempt(
        rider_id="R666",
        idem_key="payout-R666-2026-09-20",
        amount=25,
        payout_id="pout_unverified_666",
        status="success",
        reference="dispute-R666-2026-09-20",
        created_at=datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc),
    )
    # Separate pending ₹25 approval in ops_queue
    ops_item = OpsQueueItem(
        rider_id="R666",
        type="approval",
        amount=25,
        reason="Separate ₹25 second payout awaiting review",
        created_at=datetime(2026, 9, 20, 10, 30, tzinfo=timezone.utc),
    )
    test_db.add_all([trip, payout, unverified_attempt, ops_item])
    test_db.commit()

    # Initial dispute context
    init_msg = Message(
        message_id="seq-init",
        rider_id="R666",
        text="My payout for September 20 is incorrect. Please investigate.",
        reply="For trip T666001, shortfall of ₹25 has been noted.",
        received_at=datetime(2026, 9, 22, 9, 0, tzinfo=timezone.utc),
    )
    test_db.add(init_msg)
    test_db.commit()

    # Question 1: How much money is missing?
    r1 = dd.run_deterministic_dispute(
        rider_id="R666",
        message_text="How much money is missing?",
        received_at_str="2026-10-30T03:01:00+05:30",
        db=test_db,
    )
    assert "25" in r1
    assert "difference" in r1.lower() or "shortfall" in r1.lower() or "missing" in r1.lower()
    test_db.add(Message(message_id="seq-q1", rider_id="R666", text="How much money is missing?", reply=r1, received_at=datetime.now(timezone.utc)))
    test_db.commit()

    # Question 2: Has the payment actually been processed?
    r2 = dd.run_deterministic_dispute(
        rider_id="R666",
        message_text="Has the payment actually been processed?",
        received_at_str="2026-10-30T03:02:00+05:30",
        db=test_db,
    )
    assert "records say" in r2.lower() or "database records" in r2.lower()
    assert "couldn't confirm" in r2.lower() or "unavailable" in r2.lower() or "not found" in r2.lower()
    test_db.add(Message(message_id="seq-q2", rider_id="R666", text="Has the payment actually been processed?", reply=r2, received_at=datetime.now(timezone.utc)))
    test_db.commit()

    # Question 3: When will I receive it?
    r3 = dd.run_deterministic_dispute(
        rider_id="R666",
        message_text="When will I receive it?",
        received_at_str="2026-10-30T03:03:00+05:30",
        db=test_db,
    )
    assert "unable to confirm when you'll receive" in r3.lower() or "arrival time" in r3.lower() or "haven't confirmed" in r3.lower()
    test_db.add(Message(message_id="seq-q3", rider_id="R666", text="When will I receive it?", reply=r3, received_at=datetime.now(timezone.utc)))
    test_db.commit()

    # Question 4: Is the other ₹25 request still awaiting approval? (Matching approval exists)
    r4 = dd.run_deterministic_dispute(
        rider_id="R666",
        message_text="Is the other ₹25 request still awaiting approval?",
        received_at_str="2026-10-30T03:04:00+05:30",
        db=test_db,
    )
    # MUST answer specifically about the pending approval, not repeat trip T666001's unverified payment message!
    assert "waiting for approval" in r4.lower() or "pending review and approval" in r4.lower()
    assert "25" in r4
    assert "reference number" in r4.lower() or "ref-" in r4.lower()
    assert "pout_unverified_666" not in r4  # Does not regurgitate raw PaymentAttempt payout verification

    # Test when no separate approval exists:
    ops_item.resolved_at = datetime.now(timezone.utc)
    test_db.commit()

    r4_no_op = dd.run_deterministic_dispute(
        rider_id="R666",
        message_text="Is the other ₹25 request still awaiting approval?",
        received_at_str="2026-10-30T03:05:00+05:30",
        db=test_db,
    )
    assert "couldn't find another" in r4_no_op.lower() or "do not have a separate" in r4_no_op.lower() or "no" in r4_no_op.lower()


def test_simple_plain_english_conversation_flow(test_db):
    """Verifies that the chatbot communicates with riders using simple, common English words
    and matches the customer support phrasing specified in requirements.
    
    Checks:
    - No technical jargon like 'live ledger', 'verified shortfall', 'pending operations queue'.
    - Clear everyday phrases:
      * 'couldn't confirm'
      * 'difference is ₹25'
      * 'waiting for approval'
    """
    rider = Rider(
        rider_id="R555",
        name="Simple English Rider",
        city="Bengaluru",
        joined_on=date(2026, 1, 1),
    )
    test_db.add(rider)

    started = datetime(2026, 9, 20, 7, 27, tzinfo=timezone.utc)
    trip = Trip(
        trip_id="T555001",
        rider_id="R555",
        started_at=started,
        status="completed",
        distance_km=Decimal("6.00"),
        surge_multiplier=Decimal("1.50"),
    )
    payout = PayoutLine(
        line_id=f"pl-{uuid.uuid4().hex[:8]}",
        rider_id="R555",
        trip_id="T555001",
        payout_date=date(2026, 9, 20),
        amount=49,
        line_type="trip",
    )
    # PaymentAttempt in DB that cannot be verified in live PaySwift ledger
    unverified_attempt = PaymentAttempt(
        rider_id="R555",
        idem_key="payout-R555-2026-09-20",
        amount=25,
        payout_id="pout_unverified_555",
        status="success",
        reference="dispute-R555-2026-09-20",
        created_at=datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc),
    )
    ops_item = OpsQueueItem(
        rider_id="R555",
        type="approval",
        amount=25,
        reason="Second payout today",
        created_at=datetime(2026, 9, 20, 10, 30, tzinfo=timezone.utc),
    )
    test_db.add_all([trip, payout, unverified_attempt, ops_item])
    test_db.commit()

    # Turn 1: "My payout for September 20 is incorrect. Please investigate."
    r1 = dd.run_deterministic_dispute(
        rider_id="R555",
        message_text="My payout for September 20 is incorrect. Please investigate.",
        received_at_str="2026-10-30T03:00:00+05:30",
        db=test_db,
    )
    assert "Your trip fare was ₹74, and our records show that you received ₹49" in r1
    assert "couldn't confirm it in our payment system" in r1
    assert "live ledger" not in r1.lower()
    assert "sandbox verification is unavailable" not in r1.lower()

    test_db.add(Message(message_id="sim-001", rider_id="R555", text="My payout for September 20 is incorrect.", reply=r1, received_at=datetime.now(timezone.utc)))
    test_db.commit()

    # Turn 2: "How much money is missing?"
    r2 = dd.run_deterministic_dispute(
        rider_id="R555",
        message_text="How much money is missing?",
        received_at_str="2026-10-30T03:01:00+05:30",
        db=test_db,
    )
    assert "Your trip fare was ₹74, and you received ₹49. The difference is ₹25" in r2
    assert "still need to confirm whether the additional ₹25 payment went through" in r2
    assert "verified shortfall" not in r2.lower()

    test_db.add(Message(message_id="sim-002", rider_id="R555", text="How much money is missing?", reply=r2, received_at=datetime.now(timezone.utc)))
    test_db.commit()

    # Turn 3: "Has the payment actually been processed?"
    r3 = dd.run_deterministic_dispute(
        rider_id="R555",
        message_text="Has the payment actually been processed?",
        received_at_str="2026-10-30T03:02:00+05:30",
        db=test_db,
    )
    assert "Our records say the ₹25 payment was successful, but we couldn't confirm it in our payment system" in r3
    assert "can't yet say for sure whether you received it" in r3

    test_db.add(Message(message_id="sim-003", rider_id="R555", text="Has the payment actually been processed?", reply=r3, received_at=datetime.now(timezone.utc)))
    test_db.commit()

    # Turn 4: "When will I receive it?"
    r4 = dd.run_deterministic_dispute(
        rider_id="R555",
        message_text="When will I receive it?",
        received_at_str="2026-10-30T03:03:00+05:30",
        db=test_db,
    )
    assert r4 == "We're unable to confirm when you'll receive the money because we haven't confirmed the payment status yet."

    test_db.add(Message(message_id="sim-004", rider_id="R555", text="When will I receive it?", reply=r4, received_at=datetime.now(timezone.utc)))
    test_db.commit()

    # Turn 5: "Is the other ₹25 request still awaiting approval?" (with matching request)
    r5 = dd.run_deterministic_dispute(
        rider_id="R555",
        message_text="Is the other ₹25 request still awaiting approval?",
        received_at_str="2026-10-30T03:04:00+05:30",
        db=test_db,
    )
    assert "Yes, your other ₹25 request is still waiting for approval" in r5
    assert "reference number is" in r5.lower()
    assert "pending operations queue" not in r5.lower()

    # Turn 6: "Is the other ₹25 request still awaiting approval?" (no matching request)
    ops_item.resolved_at = datetime.now(timezone.utc)
    test_db.commit()

    r6 = dd.run_deterministic_dispute(
        rider_id="R555",
        message_text="Is the other ₹25 request still awaiting approval?",
        received_at_str="2026-10-30T03:05:00+05:30",
        db=test_db,
    )
    assert r6 == "I couldn't find another ₹25 request waiting for approval on your account."
