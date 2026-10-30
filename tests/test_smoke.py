"""Integration smoke test — requires the app running on localhost:8000.

Run after `docker compose up -d`:
  pytest tests/test_smoke.py -v

Or against a custom URL:
  SMOKE_URL=http://localhost:8000 pytest tests/test_smoke.py -v
"""
import os
import pytest
import httpx

BASE_URL = os.getenv("SMOKE_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=15) as c:
        yield c


class TestSmoke:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_ops_pending_empty(self, client):
        r = client.get("/ops/pending")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_trace_unknown_rider_empty(self, client):
        r = client.get("/trace/SMOKE-UNKNOWN-RIDER")
        assert r.status_code == 200
        assert r.json() == []

    def test_messages_requires_body(self, client):
        r = client.post("/messages", json={})
        assert r.status_code == 422  # validation error

    def test_messages_invalid_message_id(self, client):
        r = client.post("/messages", json={
            "message_id": "",
            "rider_id": "R001",
            "text": "test",
            "received_at": "2026-09-22T09:00:00+05:30",
        })
        assert r.status_code == 422
