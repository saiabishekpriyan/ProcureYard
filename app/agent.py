"""LLM agent — Gemini-powered, tool-calling loop.

What the MODEL decides:
  - Whether the rider's complaint is about a missing fare, surge, incentive, or penalty.
  - Which specific trip(s) the rider is referring to.
  - The tone and wording of the reply.
  - Whether to ask the rider for clarification.

What the CODE decides (never delegated to the model):
  - Every rupee calculation (calc.py).
  - Whether a payout is policy-eligible (policy.py).
  - Whether to call PaySwift (payswift.py).
  - Deduplication of messages (DB UNIQUE on message_id).
  - All dispute window checks.

Tool list (available to the model):
  lookup_rider         → Rider profile + recent payout dates
  lookup_trips         → Trips for rider on a given IST date
  lookup_payout_lines  → Historical payout lines for a rider
  calculate_payout     → Expected payout for a date (pure calc.py)
  trigger_payout       → Initiate PaySwift payment (policy guards run first)
  queue_approval       → Add item to ops_queue (type=approval)
  queue_escalation     → Add item to ops_queue (type=escalation)
"""
from __future__ import annotations

import inspect
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import google.ai.generativelanguage as glm
from sqlalchemy.orm import Session

import calc
import policy
import payswift as ps
from config import settings
from models import AuditTrace, OpsQueueItem, PaymentAttempt, Rider, Trip, PayoutLine

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


# ── Audit helper ──────────────────────────────────────────────────────────────

def _trace(db: Session, rider_id: str, type_: str, name: str,
           input_: Any = None, output: Any = None) -> None:
    db.add(AuditTrace(
        rider_id=rider_id,
        type=type_,
        name=name,
        input=input_,
        output=output,
        at=datetime.now(timezone.utc),
    ))
    db.commit()


# ── Tool implementations ───────────────────────────────────────────────────────

def tool_lookup_rider(rider_id: str, db: Session) -> dict:
    rider = db.get(Rider, rider_id)
    if not rider:
        return {"error": f"Rider {rider_id} not found"}
    lines = (
        db.query(PayoutLine)
        .filter(PayoutLine.rider_id == rider_id)
        .order_by(PayoutLine.payout_date.desc())
        .limit(30)
        .all()
    )
    dates = sorted({str(l.payout_date) for l in lines}, reverse=True)[:10]
    return {
        "rider_id": rider_id,
        "name": rider.name,
        "city": rider.city,
        "recent_payout_dates": dates,
    }


def tool_lookup_trips(rider_id: str, ist_date_str: str, db: Session) -> dict:
    """Return trips for a rider on a given IST calendar date."""
    from datetime import date as date_type
    try:
        for_date = datetime.strptime(ist_date_str, "%Y-%m-%d").date()
    except ValueError:
        return {"error": f"Invalid date format: {ist_date_str}. Use YYYY-MM-DD."}

    # Query trips whose IST date matches
    # IST = UTC + 5:30, so UTC range for a given IST date is [date-5:30 → date+18:30)
    from datetime import datetime as dt
    day_start_utc = dt.combine(for_date, dt.min.time(), tzinfo=timezone.utc) - timedelta(hours=5, minutes=30)
    day_end_utc = day_start_utc + timedelta(days=1)

    trips = (
        db.query(Trip)
        .filter(
            Trip.rider_id == rider_id,
            Trip.started_at >= day_start_utc,
            Trip.started_at < day_end_utc,
        )
        .all()
    )

    return {
        "rider_id": rider_id,
        "ist_date": ist_date_str,
        "trips": [
            {
                "trip_id": t.trip_id,
                "status": t.status,
                "distance_km": float(t.distance_km),
                "surge_multiplier": float(t.surge_multiplier),
                "started_at_utc": t.started_at.isoformat(),
            }
            for t in trips
        ],
    }


def tool_lookup_payout_lines(rider_id: str, ist_date_str: str, db: Session) -> dict:
    """Return the historical payout lines for a rider on a given IST date."""
    try:
        for_date = datetime.strptime(ist_date_str, "%Y-%m-%d").date()
    except ValueError:
        return {"error": f"Invalid date format: {ist_date_str}"}

    lines = (
        db.query(PayoutLine)
        .filter(PayoutLine.rider_id == rider_id, PayoutLine.payout_date == for_date)
        .all()
    )
    return {
        "rider_id": rider_id,
        "ist_date": ist_date_str,
        "lines": [
            {"line_id": l.line_id, "line_type": l.line_type,
             "trip_id": l.trip_id, "amount": l.amount}
            for l in lines
        ],
        "total_paid": sum(l.amount for l in lines),
    }


def tool_calculate_payout(rider_id: str, ist_date_str: str, db: Session) -> dict:
    """Compute the expected payout for a rider on a given IST date (pure calc)."""
    trips_result = tool_lookup_trips(rider_id, ist_date_str, db)
    if "error" in trips_result:
        return trips_result

    try:
        for_date = datetime.strptime(ist_date_str, "%Y-%m-%d").date()
    except ValueError:
        return {"error": f"Invalid date: {ist_date_str}"}

    trip_dicts = []
    for t in trips_result["trips"]:
        started_at = datetime.fromisoformat(t["started_at_utc"])
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        trip_dicts.append({
            "trip_id": t["trip_id"],
            "started_at": started_at,
            "status": t["status"],
            "distance_km": t["distance_km"],
            "surge_multiplier": t["surge_multiplier"],
        })

    result = calc.compute_expected_payout(trip_dicts, for_date)
    result["rider_id"] = rider_id
    result["ist_date"] = ist_date_str
    return result


def tool_trigger_payout(
    rider_id: str,
    amount: int,
    ist_date_str: str,
    reason: str,
    db: Session,
) -> dict:
    """Policy-guarded synchronous payout trigger (runs PaySwift POST).

    All policy checks happen HERE, not in the LLM.
    Returns {"success": True, "payout_id": ...} or {"success": False, "reason": ...}.
    """
    import asyncio

    try:
        for_date = datetime.strptime(ist_date_str, "%Y-%m-%d").date()
    except ValueError:
        return {"success": False, "reason": f"Invalid date: {ist_date_str}"}

    # 1. Policy guard
    try:
        policy.check_auto_pay_eligible(amount, rider_id, for_date, db)
    except policy.PolicyViolation as exc:
        return {"success": False, "reason": exc.message, "code": exc.code}

    # 2. Build idempotency key and reference
    idem_key = policy.make_idem_key(rider_id, for_date)
    reference = f"dispute-{rider_id}-{ist_date_str}"

    # 3. Upsert payment attempt row
    attempt = db.query(PaymentAttempt).filter_by(idem_key=idem_key).first()
    if not attempt:
        attempt = PaymentAttempt(
            rider_id=rider_id,
            idem_key=idem_key,
            amount=amount,
            reference=reference,
            status="pending",
        )
        db.add(attempt)
        db.commit()
    elif attempt.status == "success":
        return {"success": True, "payout_id": attempt.payout_id, "cached": True}

    # 4. Call PaySwift (sync wrapper around async client)
    def _run(coro):
        """Run async coroutine from sync context safely."""
        try:
            loop = asyncio.get_running_loop()
            # Already in an event loop — use ThreadPoolExecutor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(coro)

    try:
        result = _run(ps.post_payout(rider_id, amount, reference, idem_key))
        payout_id = result.get("payout_id")
        attempt.status = "success"
        attempt.payout_id = payout_id
        db.commit()
        return {"success": True, "payout_id": payout_id}

    except ps.PaySwiftValidationError as exc:
        attempt.status = "failed"
        attempt.error = str(exc)
        db.commit()
        return {"success": False, "reason": f"PaySwift validation error: {exc}"}

    except ps.PaySwiftIdempotencyConflict as exc:
        attempt.status = "failed"
        attempt.error = str(exc)
        db.commit()
        return {"success": False, "reason": "Idempotency key conflict — ops review needed."}

    except ps.PaySwiftUncertain as exc:
        # Timeout — reconcile
        try:
            found = _run(ps.reconcile_payout(rider_id, idem_key))
            if found:
                attempt.status = "success"
                attempt.payout_id = found.get("payout_id")
                db.commit()
                return {"success": True, "payout_id": found.get("payout_id"), "reconciled": True}
        except Exception:
            pass
        attempt.status = "pending"
        attempt.error = f"Uncertain: {exc}"
        db.commit()
        return {"success": False, "reason": f"PaySwift timeout — reconciliation pending. {exc}"}

    except ps.PaySwiftMaxRetriesExceeded as exc:
        attempt.status = "failed"
        attempt.error = str(exc)
        db.commit()
        return {"success": False, "reason": f"PaySwift temporarily unavailable: {exc}"}


def tool_queue_approval(
    rider_id: str, amount: int, reason: str, db: Session
) -> dict:
    item = OpsQueueItem(rider_id=rider_id, type="approval", amount=amount, reason=reason)
    db.add(item)
    db.commit()
    return {"queued": True, "id": item.id, "type": "approval"}


def tool_queue_escalation(rider_id: str, reason: str, db: Session) -> dict:
    item = OpsQueueItem(rider_id=rider_id, type="escalation", amount=None, reason=reason)
    db.add(item)
    db.commit()
    return {"queued": True, "id": item.id, "type": "escalation"}


# ── Tool dispatch ──────────────────────────────────────────────────────────────

TOOLS = {
    "lookup_rider": tool_lookup_rider,
    "lookup_trips": tool_lookup_trips,
    "lookup_payout_lines": tool_lookup_payout_lines,
    "calculate_payout": tool_calculate_payout,
    "trigger_payout": tool_trigger_payout,
    "queue_approval": tool_queue_approval,
    "queue_escalation": tool_queue_escalation,
}

# ── Gemini tool declarations (glm-based, compatible with google-generativeai 0.8.x) ──
#
# glm.Type is an enum — string values like "object" are NOT accepted.
# Use glm.Type.OBJECT, glm.Type.STRING, glm.Type.INTEGER, etc.

_TOOLS_GLM = glm.Tool(function_declarations=[
    glm.FunctionDeclaration(
        name="lookup_rider",
        description="Look up a rider's profile and recent payout dates.",
        parameters=glm.Schema(
            type=glm.Type.OBJECT,
            properties={
                "rider_id": glm.Schema(type=glm.Type.STRING),
            },
            required=["rider_id"],
        ),
    ),
    glm.FunctionDeclaration(
        name="lookup_trips",
        description="Return all trips for a rider on a specific IST calendar date.",
        parameters=glm.Schema(
            type=glm.Type.OBJECT,
            properties={
                "rider_id": glm.Schema(type=glm.Type.STRING),
                "ist_date_str": glm.Schema(
                    type=glm.Type.STRING,
                    description="YYYY-MM-DD in IST",
                ),
            },
            required=["rider_id", "ist_date_str"],
        ),
    ),
    glm.FunctionDeclaration(
        name="lookup_payout_lines",
        description="Return the historical payout lines for a rider on an IST date.",
        parameters=glm.Schema(
            type=glm.Type.OBJECT,
            properties={
                "rider_id": glm.Schema(type=glm.Type.STRING),
                "ist_date_str": glm.Schema(type=glm.Type.STRING),
            },
            required=["rider_id", "ist_date_str"],
        ),
    ),
    glm.FunctionDeclaration(
        name="calculate_payout",
        description="Calculate the expected payout for a rider on a given IST date using policy rules.",
        parameters=glm.Schema(
            type=glm.Type.OBJECT,
            properties={
                "rider_id": glm.Schema(type=glm.Type.STRING),
                "ist_date_str": glm.Schema(type=glm.Type.STRING),
            },
            required=["rider_id", "ist_date_str"],
        ),
    ),
    glm.FunctionDeclaration(
        name="trigger_payout",
        description=(
            "Initiate a PaySwift payout for a rider. "
            "Policy guards run automatically — do NOT call this if the amount exceeds 200 or if ops approval is needed. "
            "Use queue_approval instead for larger amounts."
        ),
        parameters=glm.Schema(
            type=glm.Type.OBJECT,
            properties={
                "rider_id": glm.Schema(type=glm.Type.STRING),
                "amount": glm.Schema(
                    type=glm.Type.INTEGER,
                    description="Amount in rupees (1-200)",
                ),
                "ist_date_str": glm.Schema(type=glm.Type.STRING),
                "reason": glm.Schema(type=glm.Type.STRING),
            },
            required=["rider_id", "amount", "ist_date_str", "reason"],
        ),
    ),
    glm.FunctionDeclaration(
        name="queue_approval",
        description="Queue an item for ops team approval (amounts > 200 or second payment today).",
        parameters=glm.Schema(
            type=glm.Type.OBJECT,
            properties={
                "rider_id": glm.Schema(type=glm.Type.STRING),
                "amount": glm.Schema(type=glm.Type.INTEGER),
                "reason": glm.Schema(type=glm.Type.STRING),
            },
            required=["rider_id", "amount", "reason"],
        ),
    ),
    glm.FunctionDeclaration(
        name="queue_escalation",
        description="Escalate to ops team for non-financial issues (fraud suspicion, unclear dispute, dispute outside 7-day window).",
        parameters=glm.Schema(
            type=glm.Type.OBJECT,
            properties={
                "rider_id": glm.Schema(type=glm.Type.STRING),
                "reason": glm.Schema(type=glm.Type.STRING),
            },
            required=["rider_id", "reason"],
        ),
    ),
])


# ── Main agent entry point ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Rider Payout Support Assistant for QuickDrop.

Your job is to resolve rider payout complaints fairly, quickly, and politely in conversational multi-turn dialogue.
Always sound like a helpful customer support person, not a technical system. Use simple, everyday English words.

LANGUAGE & TONE REQUIREMENT:
- All rider-facing responses MUST be written in natural, clear, and simple English ONLY.
- Even if the rider writes in Hindi, Hinglish, or any other language, ALWAYS respond in clear English.
- Never use Hindi or Hinglish words such as "bhai", "wala", "haan", or "thik hai" in any reply.
- Use short, clear sentences and familiar words. A rider with no technical knowledge should easily understand every response.
- Replace technical words with plain language:
  * Avoid "payment verification is unavailable" → use "We couldn't confirm your payment."
  * Avoid "transaction is not found in the live ledger" → use "We couldn't find this payment in our payment records."
  * Avoid "verified shortfall" → use "The amount still missing" or "The difference is...".
  * Avoid "pending operations queue" → use "Requests waiting for approval."
  * Avoid "payment status remains unverified" → use "We haven't been able to confirm whether the payment went through."
  * Instead of "The expected fare was ₹74 and ₹49 was paid" → use "Your trip fare was ₹74, but you received ₹49. The difference is ₹25."

CONVERSATIONAL & POLICY RULES:
1. ALWAYS call calculate_payout and lookup_payout_lines before deciding anything about money.
2. If the rider is owed money and the shortfall is ≤ ₹200, use trigger_payout. The system will enforce the limit.
3. If the shortfall is > ₹200, use queue_approval with the full amount — never split payments or auto-pay over ₹200.
4. If a dispute is a brand new claim and outside the 7-day window, use queue_escalation. Never apply the 7-day limit to follow-up questions or status inquiries about an existing dispute or previously discussed trip.
5. Ground all statements in actual database records and PaySwift status:
   - Distinguish between amounts calculated as owed, payments requested, payments successfully confirmed in PaySwift, and requests waiting for approval.
   - Never claim a payment was processed based only on a database record; reconcile with live PaySwift status.
   - If payment is confirmed in PaySwift, let the rider know politely (e.g. "A payment of ₹25 has been confirmed and sent to your account.").
   - If we cannot confirm the payment in our payment system, explain that clearly: "Our records say the ₹25 payment was successful, but we couldn't confirm it in our payment system. So, we can't yet say for sure whether you received it."
   - If asked when they will receive it and payment is not verified, state: "We're unable to confirm when you'll receive the money because we haven't confirmed the payment status yet."
   - If approval is pending, explain simply that requests over ₹200 or second payments today are waiting for approval.
6. Answer follow-up questions directly using simple language:
   - "How much money is missing?": Explain the trip fare, the amount received, and the difference (e.g., "Your trip fare was ₹74, and you received ₹49. The difference is ₹25. However, we still need to confirm whether the additional ₹25 payment went through.").
   - "Has the payment actually been processed?": "Our records say the ₹25 payment was successful, but we couldn't confirm it in our payment system. So, we can't yet say for sure whether you received it."
   - "When will I receive it?": "We're unable to confirm when you'll receive the money because we haven't confirmed the payment status yet."
   - "Is the other ₹25 request still awaiting approval?":
     * If matching request exists: "Yes, your other ₹25 request is still waiting for approval. Its reference number is [actual request ID]."
     * If no matching request exists: "I couldn't find another ₹25 request waiting for approval on your account."
7. Resolve references like "that trip", "the other one", or "the one on the 20th" using conversation history and verified records.
8. Ask relevant clarifying questions when the rider's request is ambiguous.
9. Reply in a warm, polite, and helpful tone in English. Never reveal internal function names or database error details.
10. The dispute window is 7 days for new claims. Politely decline new claims outside this window in simple English.
"""


def _dispatch_tool(tool_name: str, tool_args: dict, db: Session) -> dict:
    """Execute a named tool, injecting db if needed. Returns a JSON-serialisable dict."""
    tool_fn = TOOLS.get(tool_name)
    if tool_fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        sig = inspect.signature(tool_fn)
        if "db" in sig.parameters:
            return tool_fn(**tool_args, db=db)
        return tool_fn(**tool_args)
    except Exception as exc:
        log.exception("Tool %s failed: %s", tool_name, exc)
        return {"error": str(exc)}



def run_agent(
    rider_id: str,
    message_text: str,
    received_at: str,
    db: Session,
    gemini_api_key: str,
) -> str:
    """Run the LLM agent for one rider message. Returns the reply string."""
    import google.generativeai as genai
    from google.generativeai.types import RequestOptions
    from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded

    if not gemini_api_key:
        log.warning("GEMINI_API_KEY not set — using deterministic dispute workflow")
        _trace(db, rider_id, "message_in", "rider_message",
               input_={"text": message_text, "received_at": received_at,
                       "model": "deterministic_dispute"})
        try:
            from deterministic_dispute import run_deterministic_dispute
            return run_deterministic_dispute(rider_id, message_text, received_at, db)
        except Exception:
            log.exception("Agent: deterministic dispute failed when GEMINI_API_KEY not set")
            return (
                "Thank you for reaching out. We've logged your concern and our "
                "team will review your payout and follow up shortly."
            )

    model_name = settings.gemini_model
    api_timeout = settings.gemini_timeout  # seconds per API call

    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
            tools=[_TOOLS_GLM],
        )
    except Exception:
        log.exception("Agent: failed to construct GenerativeModel (model=%s)", model_name)
        _trace(db, rider_id, "error", "model_construction_failed",
               output={"model": model_name})
        return (
            "Thank you for reaching out. We've logged your concern and our "
            "team will review your payout and follow up shortly."
        )

    _trace(db, rider_id, "message_in", "rider_message",
           input_={"text": message_text, "received_at": received_at,
                   "model": model_name})

    user_content = (
        f"Rider {rider_id} says (received {received_at}):\n{message_text}"
    )

    # retry=None disables the SDK's built-in auto-retry so that quota errors
    # (429 ResourceExhausted) surface immediately instead of being retried
    # internally for minutes before our wall-clock timeout fires.
    req_opts = RequestOptions(timeout=api_timeout, retry=None)

    try:
        chat = model.start_chat(enable_automatic_function_calling=False)

        # ── Agentic loop ──────────────────────────────────────────────────────────
        # Pattern:
        #   1. Send the user message (first turn) or nothing (subsequent turns).
        #   2. If the model calls tools, dispatch them and feed results back in one
        #      send_message call using a list of function_response Parts.
        #   3. If the model returns text, that's the final reply.
        # ─────────────────────────────────────────────────────────────────────────
        max_turns = 10
        next_message: Any = user_content

        for turn in range(max_turns):
            log.info(
                "Agent: sending turn %d to %s for rider %s (timeout=%ds)",
                turn, model_name, rider_id, api_timeout,
            )
            try:
                response = chat.send_message(next_message, request_options=req_opts)

            except ResourceExhausted as quota_exc:
                # HTTP 429 — quota exhausted.  Do NOT retry; fall back immediately to deterministic dispute workflow.
                log.warning(
                    "Agent: quota exhausted on turn %d for rider %s (model=%s). "
                    "Falling back to deterministic dispute workflow. Detail: %s",
                    turn, rider_id, model_name, quota_exc,
                )
                _trace(db, rider_id, "error", "quota_exhausted",
                       output={"turn": turn, "model": model_name, "http_status": 429})
                try:
                    from deterministic_dispute import run_deterministic_dispute
                    return run_deterministic_dispute(rider_id, message_text, received_at, db)
                except Exception:
                    log.exception("Agent: deterministic dispute failed for rider %s after quota exhaustion", rider_id)
                    return (
                        "We're currently experiencing high demand on our AI service. "
                        "Your concern has been logged and our team will review your "
                        "payout manually and follow up shortly."
                    )

            except DeadlineExceeded as timeout_exc:
                log.warning(
                    "Agent: API call timed out on turn %d for rider %s "
                    "(model=%s, timeout=%ds): %s",
                    turn, rider_id, model_name, api_timeout, timeout_exc,
                )
                _trace(db, rider_id, "error", "api_timeout",
                       output={"turn": turn, "model": model_name,
                               "timeout_s": api_timeout})
                return (
                    "Thank you for reaching out. We've reviewed your concern and our "
                    "team will follow up shortly."
                )

            except Exception as call_exc:
                # Other transport / server errors (500, 503, network failure, etc.)
                log.exception(
                    "Agent: API call failed on turn %d for rider %s (model=%s): %s",
                    turn, rider_id, model_name, call_exc,
                )
                _trace(db, rider_id, "error", "api_call_failed",
                       output={"turn": turn, "model": model_name,
                               "error": type(call_exc).__name__})
                return (
                    "Thank you for reaching out. We've reviewed your concern and our "
                    "team will follow up shortly."
                )

            log.info(
                "Agent: turn %d completed for rider %s (%d parts)",
                turn, rider_id, len(response.parts),
            )

            # ── Diagnose empty / blocked candidates ───────────────────────────────
            if not response.candidates:
                log.warning(
                    "Agent: no candidates in response for rider %s turn %d "
                    "(prompt_feedback=%s)",
                    rider_id, turn, getattr(response, "prompt_feedback", None),
                )
                _trace(db, rider_id, "error", "no_candidates",
                       output={"turn": turn,
                               "prompt_feedback": str(
                                   getattr(response, "prompt_feedback", None))})
                return (
                    "Thank you for reaching out. We've reviewed your concern and our "
                    "team will follow up shortly."
                )

            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            # finish_reason values: STOP=1, MAX_TOKENS=2, SAFETY=3, RECITATION=4,
            #   OTHER=5, MALFORMED_FUNCTION_CALL=10  (integer enum in SDK 0.8.x)

            # Collect all function calls in this response
            function_calls = [
                part.function_call
                for part in response.parts
                if hasattr(part, "function_call") and part.function_call.name
            ]

            if not function_calls:
                # No tool calls → model produced a text reply (or blocked)
                try:
                    raw_text = response.text  # raises ValueError if blocked
                except ValueError as ve:
                    log.warning(
                        "Agent: response.text raised for rider %s "
                        "(finish_reason=%s): %s",
                        rider_id, finish_reason, ve,
                    )
                    _trace(db, rider_id, "error", "response_text_error",
                           output={"turn": turn,
                                   "finish_reason": str(finish_reason),
                                   "error": str(ve)})
                    return (
                        "Thank you for reaching out. We've reviewed your concern "
                        "and our team will follow up shortly."
                    )

                reply = raw_text.strip() if raw_text else ""
                if not reply:
                    log.warning(
                        "Agent: empty text reply for rider %s "
                        "(finish_reason=%s, parts=%d). Returning fallback.",
                        rider_id, finish_reason, len(response.parts),
                    )
                    _trace(db, rider_id, "error", "empty_text_reply",
                           output={"turn": turn,
                                   "finish_reason": str(finish_reason),
                                   "parts": len(response.parts)})
                    return (
                        "Thank you for reaching out. We've reviewed your concern "
                        "and our team will follow up shortly."
                    )

                _trace(db, rider_id, "reply", "agent_reply", output=reply)
                log.info("Agent: final reply produced for rider %s (%d chars)",
                         rider_id, len(reply))
                return reply

            # Dispatch all tools and collect responses
            response_parts = []
            for fn in function_calls:
                tool_name = fn.name
                tool_args = dict(fn.args)

                _trace(db, rider_id, "tool_call", tool_name, input_=tool_args)
                tool_result = _dispatch_tool(tool_name, tool_args, db)
                _trace(db, rider_id, "tool_result", tool_name, output=tool_result)

                # Build a function_response Part using glm types
                part = glm.Part(
                    function_response=glm.FunctionResponse(
                        name=tool_name,
                        response={"result": tool_result},
                    )
                )
                response_parts.append(part)

            # Feed all tool results back in a single message
            next_message = response_parts

        # Fallback if max turns hit
        fallback = (
            "I'm looking into your payout concern and will follow up shortly. "
            "If this is urgent, please contact our support team."
        )
        _trace(db, rider_id, "error", "max_turns_exceeded",
               output={"turns": max_turns, "fallback": fallback})
        return fallback

    except Exception:
        log.exception("Agent: error during chat loop for rider %s (model=%s)",
                      rider_id, model_name)
        _trace(db, rider_id, "error", "chat_loop_exception",
               output={"model": model_name})
        try:
            from deterministic_dispute import run_deterministic_dispute
            return run_deterministic_dispute(rider_id, message_text, received_at, db)
        except Exception:
            return (
                "Thank you for reaching out. We've logged your concern and our "
                "team will review your payout and follow up shortly."
            )
