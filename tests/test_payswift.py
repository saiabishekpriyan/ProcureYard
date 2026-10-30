"""Unit tests for payswift.py — use httpx.MockTransport so tests are fully sync.

httpx.MockTransport intercepts at the transport layer: the real AsyncClient
calls our sync handler and returns immediately without network I/O or asyncio
event loop interference.  We drive the coroutines with asyncio.run() in plain
sync test functions, bypassing pytest-asyncio entirely.

Run with:
    python -m pytest tests/test_payswift.py -p no:asyncio -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import asyncio
import json
import pytest
import httpx


# ── Transport helpers ──────────────────────────────────────────────────────────

def _make_response(status: int, body: dict) -> httpx.Response:
    """Build an httpx.Response with the given status and JSON body."""
    content = json.dumps(body).encode()
    return httpx.Response(
        status_code=status,
        headers={"content-type": "application/json"},
        content=content,
    )


def _fixed_transport(status: int, body: dict) -> httpx.MockTransport:
    """Transport that always returns the same response."""
    resp = _make_response(status, body)
    def handler(request: httpx.Request) -> httpx.Response:
        return resp
    return httpx.MockTransport(handler)


def _sequence_transport(responses: list[tuple[int, dict]]) -> httpx.MockTransport:
    """Transport that returns successive (status, body) pairs."""
    pairs = list(responses)
    idx = [0]
    def handler(request: httpx.Request) -> httpx.Response:
        i = min(idx[0], len(pairs) - 1)
        idx[0] += 1
        status, body = pairs[i]
        return _make_response(status, body)
    return httpx.MockTransport(handler)


def _run_with_transport(transport: httpx.MockTransport, coro_fn, *args, **kwargs):
    """
    Temporarily replace payswift._client with an AsyncClient backed by
    the given transport, run coro_fn(*args, **kwargs) to completion, then
    restore the original client.
    """
    import payswift

    saved = payswift._client
    client = httpx.AsyncClient(
        base_url="http://mock-payswift",
        transport=transport,
        timeout=5,
    )
    payswift._client = client
    try:
        return asyncio.run(coro_fn(*args, **kwargs))
    finally:
        # Close the mock client synchronously
        asyncio.run(client.aclose())
        payswift._client = saved


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestPostPayout:
    def test_success_200(self):
        import payswift
        transport = _fixed_transport(200, {
            "payout_id": "pout_abc123", "rider_id": "R001",
            "amount": 25, "status": "processed",
            "created_at": "2026-10-29T14:00:00Z",
        })
        result = _run_with_transport(
            transport, payswift.post_payout, "R001", 25, "test-ref", "idem-key-1"
        )
        assert result["payout_id"] == "pout_abc123"
        assert result["status"] == "processed"

    def test_success_201(self):
        import payswift
        transport = _fixed_transport(201, {
            "payout_id": "pout_new", "status": "processed",
            "amount": 50, "created_at": "2026-10-29T14:00:00Z",
        })
        result = _run_with_transport(
            transport, payswift.post_payout, "R001", 50, "ref", "idem-key-2"
        )
        assert result["payout_id"] == "pout_new"

    def test_400_raises_validation_error(self):
        import payswift
        transport = _fixed_transport(400, {
            "error": "invalid_amount",
            "message": "amount must be an integer between 1 and 10000",
        })
        with pytest.raises(payswift.PaySwiftValidationError) as exc_info:
            _run_with_transport(
                transport, payswift.post_payout, "R001", 0, "ref", "idem-key-3"
            )
        assert exc_info.value.error == "invalid_amount"

    def test_409_raises_idempotency_conflict(self):
        import payswift
        transport = _fixed_transport(409, {
            "error": "idempotency_key_reused",
            "message": "Idempotency-Key was used with a different request body",
        })
        with pytest.raises(payswift.PaySwiftIdempotencyConflict):
            _run_with_transport(
                transport, payswift.post_payout, "R001", 25, "ref", "idem-key-4"
            )

    def test_503_retries_and_eventually_succeeds(self):
        import payswift
        transport = _sequence_transport([
            (503, {"error": "service_unavailable"}),
            (503, {"error": "service_unavailable"}),
            (200, {"payout_id": "pout_retry_ok", "status": "processed",
                   "amount": 25, "created_at": "2026-10-29T14:00:00Z"}),
        ])
        # Patch asyncio.sleep inside payswift to avoid real delays
        original_sleep = payswift.asyncio.sleep
        async def instant_sleep(seconds):
            pass
        payswift.asyncio.sleep = instant_sleep
        try:
            result = _run_with_transport(
                transport, payswift.post_payout, "R001", 25, "ref", "idem-key-5"
            )
        finally:
            payswift.asyncio.sleep = original_sleep
        assert result["payout_id"] == "pout_retry_ok"

    def test_all_503_raises_max_retries(self):
        import payswift
        transport = _fixed_transport(503, {"error": "service_unavailable"})
        original_sleep = payswift.asyncio.sleep
        async def instant_sleep(seconds):
            pass
        payswift.asyncio.sleep = instant_sleep
        try:
            with pytest.raises(payswift.PaySwiftMaxRetriesExceeded):
                _run_with_transport(
                    transport, payswift.post_payout, "R001", 25, "ref", "idem-key-6"
                )
        finally:
            payswift.asyncio.sleep = original_sleep


class TestGetPayouts:
    def test_returns_data_list(self):
        import payswift
        transport = _fixed_transport(200, {
            "data": [{"payout_id": "p1", "amount": 25}], "count": 1,
        })
        result = _run_with_transport(
            transport, payswift.get_payouts, rider_id="R001"
        )
        assert len(result) == 1
        assert result[0]["payout_id"] == "p1"

    def test_empty_ledger(self):
        import payswift
        transport = _fixed_transport(200, {"data": [], "count": 0})
        result = _run_with_transport(transport, payswift.get_payouts)
        assert result == []
