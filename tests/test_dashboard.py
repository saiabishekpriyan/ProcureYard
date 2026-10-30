"""Tests for the Operations Dashboard UI, conversational trace, and ops action endpoints."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from main import app
from db import SessionLocal
from models import OpsQueueItem, AuditTrace, Message


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestDashboardAndConversationalTrace:
    def test_dashboard_html_served_on_root_and_dashboard(self, client):
        r1 = client.get("/")
        assert r1.status_code == 200
        assert "text/html" in r1.headers["content-type"]
        assert "ProcureYard Dispute Desk" in r1.text
        assert "Live Rider Chat" in r1.text
        assert "Conversational" in r1.text or "Trace Explorer" in r1.text

        r2 = client.get("/dashboard")
        assert r2.status_code == 200
        assert r2.text == r1.text

    def test_approve_and_reject_ops_item(self, client):
        db = SessionLocal()
        test_item_1 = OpsQueueItem(
            rider_id="R998",
            type="approval",
            amount=250,
            reason="Test dashboard approve action",
        )
        test_item_2 = OpsQueueItem(
            rider_id="R998",
            type="escalation",
            reason="Test dashboard reject action",
        )
        db.add(test_item_1)
        db.add(test_item_2)
        db.commit()
        id1 = test_item_1.id
        id2 = test_item_2.id
        db.close()

        try:
            # 1. Approve item 1
            res1 = client.post(f"/ops/{id1}/approve", json={"notes": "Approved via test"})
            assert res1.status_code == 200
            data1 = res1.json()
            assert data1["success"] is True
            assert data1["resolution"] == "approved"

            # 2. Reject item 2
            res2 = client.post(f"/ops/{id2}/reject", json={"notes": "Rejected via test"})
            assert res2.status_code == 200
            data2 = res2.json()
            assert data2["success"] is True
            assert data2["resolution"] == "rejected"

            # 3. Neither should appear in /ops/pending anymore
            pending = client.get("/ops/pending").json()
            pending_ids = [p["id"] for p in pending]
            assert id1 not in pending_ids
            assert id2 not in pending_ids

            # 4. Trace should have audit records
            trace = client.get("/trace/R998").json()
            trace_types = [t["type"] for t in trace]
            assert "ops_action" in trace_types

        finally:
            db = SessionLocal()
            db.query(OpsQueueItem).filter(OpsQueueItem.rider_id == "R998").delete()
            db.query(AuditTrace).filter(AuditTrace.rider_id == "R998").delete()
            db.commit()
            db.close()

    def test_conversational_trace_with_replies_and_chronology(self, client):
        """Verify GET /trace/{rider_id}/conversation returns messages and replies chronologically."""
        db = SessionLocal()
        now1 = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone.utc)
        now2 = datetime(2026, 9, 22, 9, 2, 0, tzinfo=timezone.utc)

        msg1 = Message(
            message_id="msg-conv-test-01",
            rider_id="R997",
            text="First message from rider",
            received_at=now1,
            reply="First agent reply to rider",
        )
        msg2 = Message(
            message_id="msg-conv-test-02",
            rider_id="R997",
            text="Second message from rider",
            received_at=now2,
            reply=None,  # historical message with missing reply
        )
        db.add(msg1)
        db.add(msg2)
        db.commit()
        db.close()

        try:
            resp = client.get("/trace/R997/conversation")
            assert resp.status_code == 200
            data = resp.json()

            assert data["rider_id"] == "R997"
            convs = data["conversations"]
            assert len(convs) == 2

            # Chronological order verified
            assert convs[0]["message_id"] == "msg-conv-test-01"
            assert convs[0]["text"] == "First message from rider"
            assert convs[0]["reply"] == "First agent reply to rider"
            assert convs[0]["has_reply"] is True

            # Missing historical reply verified
            assert convs[1]["message_id"] == "msg-conv-test-02"
            assert convs[1]["text"] == "Second message from rider"
            assert convs[1]["reply"] is None
            assert convs[1]["has_reply"] is False

        finally:
            db = SessionLocal()
            db.query(Message).filter(Message.rider_id == "R997").delete()
            db.commit()
            db.close()

    def test_duplicate_delivery_message_recording(self, client):
        """Duplicate message delivery should return identical reply and be deduplicated."""
        payload = {
            "message_id": "msg-conv-dedup-01",
            "rider_id": "R996",
            "text": "bhai mera payout missing hai 20 tarikh ka",
            "received_at": "2026-09-22T09:00:00+05:30",
        }
        try:
            r1 = client.post("/messages", json=payload)
            assert r1.status_code == 200
            reply1 = r1.json()["reply"]
            assert reply1 and len(reply1) > 0

            # Duplicate delivery
            r2 = client.post("/messages", json=payload)
            assert r2.status_code == 200
            reply2 = r2.json()["reply"]
            assert reply1 == reply2

            # Conversation trace contains exactly 1 turn for this message ID
            conv_resp = client.get("/trace/R996/conversation")
            assert conv_resp.status_code == 200
            turns = conv_resp.json()["conversations"]
            assert len(turns) == 1
            assert turns[0]["message_id"] == "msg-conv-dedup-01"
            assert turns[0]["reply"] == reply1

            # Reply trace recorded in AuditTrace
            trace_resp = client.get("/trace/R996")
            assert trace_resp.status_code == 200
            trace_records = trace_resp.json()
            reply_traces = [t for t in trace_records if t["type"] == "reply"]
            assert len(reply_traces) >= 1
            assert reply_traces[-1]["output"] == reply1

        finally:
            db = SessionLocal()
            db.query(Message).filter(Message.rider_id == "R996").delete()
            db.query(AuditTrace).filter(AuditTrace.rider_id == "R996").delete()
            db.commit()
            db.close()
