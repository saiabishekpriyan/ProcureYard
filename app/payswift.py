"""Thin PaySwift HTTP client — verified against live sandbox (session 3).

Verified contract:
  POST /v1/payouts → 200 on success (first call AND idempotent repeat)
  GET  /v1/payouts?rider_id=... → 200, body: {data:[...], count:N}
  Item fields: payout_id, rider_id, amount, reference, status, created_at (UTC Z)

Error responses:
  400 invalid_amount | invalid_rider_id | invalid_json  → do NOT retry
  409 idempotency_key_reused                           → do NOT retry, raise
  429 (unseen but possible)                            → retry same key
  503 service_unavailable | 504 gateway_timeout        → retry same key
  timeout                                              → UNCERTAIN, reconcile first

Retry policy: bounded exponential backoff (0.5s, 1s, 2s), SAME idem_key every time.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from config import settings

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.payswift_base_url,
            timeout=settings.payswift_timeout,
        )
    return _client


async def close_client() -> None:
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()


# ── Custom exceptions ──────────────────────────────────────────────────────────

class PaySwiftError(Exception):
    """Base class for all PaySwift client errors."""


class PaySwiftValidationError(PaySwiftError):
    """HTTP 400 — bad request; do not retry."""
    def __init__(self, error: str, message: str | None = None):
        self.error = error
        super().__init__(message or error)


class PaySwiftIdempotencyConflict(PaySwiftError):
    """HTTP 409 — same key used with a different body; do not retry."""


class PaySwiftUncertain(PaySwiftError):
    """Timeout — outcome unknown; caller must reconcile against GET /v1/payouts."""


class PaySwiftMaxRetriesExceeded(PaySwiftError):
    """All retry attempts exhausted on 5xx / 429."""


# ── Internals ─────────────────────────────────────────────────────────────────

_RETRY_STATUSES = {429, 500, 503, 504}
_NO_RETRY_STATUSES = {400, 409}


async def _post_with_retry(
    rider_id: str,
    amount: int,
    reference: str,
    idem_key: str,
) -> dict:
    """POST /v1/payouts with bounded retries on transient errors."""
    client = get_client()
    payload = {"rider_id": rider_id, "amount": amount, "reference": reference}
    headers = {
        "Content-Type": "application/json",
        "Idempotency-Key": idem_key,
    }
    delays = [0.5, 1.0, 2.0]  # exponential backoff

    last_exc: Exception | None = None
    for attempt, delay in enumerate(delays[: settings.payswift_max_retries], start=1):
        try:
            resp = await client.post("/v1/payouts", json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            log.warning("PaySwift POST timeout on attempt %d (key=%s)", attempt, idem_key)
            raise PaySwiftUncertain(
                f"Timeout after {settings.payswift_timeout}s on attempt {attempt}. "
                "Reconcile against GET /v1/payouts before retrying."
            ) from exc
        except httpx.RequestError as exc:
            log.warning("PaySwift POST network error on attempt %d: %s", attempt, exc)
            last_exc = exc
            await asyncio.sleep(delay)
            continue

        body: dict = {}
        try:
            body = resp.json()
        except Exception:
            pass

        if resp.status_code in (200, 201):
            log.info(
                "PaySwift payout created: payout_id=%s rider=%s amount=%s attempt=%d",
                body.get("payout_id"), rider_id, amount, attempt,
            )
            return body

        if resp.status_code == 409:
            raise PaySwiftIdempotencyConflict(
                f"Idempotency key conflict (key={idem_key}): {body.get('message', '')}"
            )

        if resp.status_code == 400:
            raise PaySwiftValidationError(
                error=body.get("error", "unknown"),
                message=body.get("message"),
            )

        if resp.status_code in _RETRY_STATUSES:
            log.warning(
                "PaySwift %d on attempt %d (key=%s); retrying in %.1fs",
                resp.status_code, attempt, idem_key, delay,
            )
            last_exc = PaySwiftMaxRetriesExceeded(f"HTTP {resp.status_code}: {body}")
            await asyncio.sleep(delay)
            continue

        # Unexpected status — do not retry
        raise PaySwiftError(f"Unexpected HTTP {resp.status_code}: {body}")

    raise PaySwiftMaxRetriesExceeded(
        f"PaySwift POST failed after {settings.payswift_max_retries} attempts. "
        f"Last error: {last_exc}"
    )


# ── Public API ─────────────────────────────────────────────────────────────────

async def post_payout(
    rider_id: str,
    amount: int,
    reference: str,
    idem_key: str,
) -> dict:
    """Create a payout via PaySwift.

    Returns the response body dict on success.
    Raises PaySwiftValidationError, PaySwiftIdempotencyConflict,
           PaySwiftUncertain, or PaySwiftMaxRetriesExceeded on failure.
    """
    return await _post_with_retry(rider_id, amount, reference, idem_key)


async def get_payouts(rider_id: str | None = None) -> list[dict]:
    """Fetch payouts from PaySwift ledger (for reconciliation).

    Args:
        rider_id: If provided, filter to a single rider.

    Returns:
        List of payout item dicts: [{payout_id, rider_id, amount, reference, status, created_at}, ...]
    """
    client = get_client()
    params: dict[str, Any] = {}
    if rider_id:
        params["rider_id"] = rider_id

    try:
        resp = await client.get("/v1/payouts", params=params)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except httpx.TimeoutException as exc:
        raise PaySwiftUncertain("Timeout fetching PaySwift ledger.") from exc
    except httpx.RequestError as exc:
        raise PaySwiftError(f"Network error fetching PaySwift ledger: {exc}") from exc


async def reconcile_payout(rider_id: str, idem_key: str) -> dict | None:
    """Check whether a payout with the given reference already landed in PaySwift.

    Used after a timeout to determine whether to retry or treat as success.
    Returns the matching payout dict, or None if not found.
    """
    items = await get_payouts(rider_id=rider_id)
    # idem_key is used as reference in our POST calls
    for item in items:
        if item.get("reference") == idem_key:
            return item
    return None
