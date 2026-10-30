"""POST /messages — main webhook endpoint."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import settings
from db import get_db
from models import Message
import agent as ag

log = logging.getLogger(__name__)
router = APIRouter()

_FALLBACK = (
    "I'm having trouble processing your request right now. "
    "Our team will follow up with you shortly."
)

# Hard wall-clock timeout (seconds) for the entire agent call, including all
# tool turns.  Slightly longer than gemini_timeout so the SDK's per-call
# deadline fires first and produces a clean error trace.
_AGENT_WALL_TIMEOUT_EXTRA = 30  # added on top of gemini_timeout


class MessageIn(BaseModel):
    message_id: str
    rider_id: str
    text: str
    received_at: str  # ISO with tz, e.g. 2026-09-22T09:05:00+05:30

    @field_validator("message_id", "rider_id", "text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()


class MessageOut(BaseModel):
    reply: str


@router.post("/messages", response_model=MessageOut)
def post_message(body: MessageIn, db: Session = Depends(get_db)) -> MessageOut:
    """Receive a rider message, run the agent, return the reply.

    Idempotent: if message_id already exists AND has a non-empty reply,
    returns the stored reply. If the stored reply is NULL or empty (i.e. a
    previous run was interrupted), we attempt exactly ONE retry and then
    permanently store the result (even if it's the fallback), so the same
    message never triggers an unbounded agent retry loop.
    """
    # Parse received_at
    try:
        received_at = datetime.fromisoformat(body.received_at)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid received_at: {body.received_at}")

    # Check for existing message (duplicate delivery)
    existing = db.query(Message).filter_by(message_id=body.message_id).first()
    if existing:
        if existing.reply:
            # Already fully processed — return cached reply without hitting the AI
            log.info("Duplicate message_id=%s — returning cached reply", body.message_id)
            return MessageOut(reply=existing.reply)
        else:
            # Previous run was interrupted (reply=None/empty) — retry ONCE,
            # then permanently store whatever we get (including the fallback).
            log.warning(
                "message_id=%s found with empty reply — retrying agent once for rider %s",
                body.message_id, existing.rider_id,
            )
            reply = _run_agent_safe(existing.rider_id, existing.text,
                                    body.received_at, db)
            existing.reply = reply  # store permanently so we never retry again
            ag._trace(db, existing.rider_id, "reply", "agent_reply", input_={"message_id": body.message_id}, output=reply)
            db.commit()
            log.info("message_id=%s reply committed after retry", body.message_id)
            return MessageOut(reply=reply)

    # Persist message row (reply=None initially — filled after agent runs)
    msg = Message(
        message_id=body.message_id,
        rider_id=body.rider_id,
        text=body.text,
        received_at=received_at,
    )
    try:
        db.add(msg)
        db.flush()  # detect UNIQUE violation early
    except IntegrityError:
        db.rollback()
        existing = db.query(Message).filter_by(message_id=body.message_id).first()
        if existing and existing.reply:
            return MessageOut(reply=existing.reply)
        # Row exists but reply is empty — one retry then store permanently
        reply = _run_agent_safe(body.rider_id, body.text, body.received_at, db)
        if existing:
            existing.reply = reply
            ag._trace(db, body.rider_id, "reply", "agent_reply", input_={"message_id": body.message_id}, output=reply)
            db.commit()
        return MessageOut(reply=reply)

    # Run agent with a hard wall-clock timeout as belt-and-suspenders on top of
    # the per-call SDK timeout (RequestOptions(timeout=...)) inside run_agent().
    reply = _run_agent_safe(body.rider_id, body.text, body.received_at, db)
    msg.reply = reply
    ag._trace(db, body.rider_id, "reply", "agent_reply", input_={"message_id": body.message_id}, output=reply)
    db.commit()
    log.info("POST /messages message_id=%s committed reply (%d chars)",
             body.message_id, len(reply))
    return MessageOut(reply=reply)


def _run_agent_safe(rider_id: str, text: str, received_at: str, db: Session) -> str:
    """Call run_agent with a hard wall-clock timeout. Always returns non-empty str."""
    wall_timeout = settings.gemini_timeout + _AGENT_WALL_TIMEOUT_EXTRA
    log.info("Agent call starting for rider %s (wall_timeout=%ds)", rider_id, wall_timeout)

    try:
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(
            ag.run_agent,
            rider_id=rider_id,
            message_text=text,
            received_at=received_at,
            db=db,
            gemini_api_key=settings.gemini_api_key,
        )
        try:
            reply = future.result(timeout=wall_timeout)
        except FuturesTimeoutError:
            log.error(
                "Agent wall-clock timeout (%ds) exceeded for rider %s — returning fallback",
                wall_timeout, rider_id,
            )
            future.cancel()
            reply = _FALLBACK
        finally:
            # shutdown(wait=False) so we don't block waiting for a hung thread.
            pool.shutdown(wait=False)
    except Exception:
        log.exception("Agent error for rider %s", rider_id)
        reply = _FALLBACK

    if not reply or not reply.strip():
        log.error(
            "Agent returned empty reply for rider %s — substituting fallback", rider_id
        )
        reply = _FALLBACK

    log.info("Agent call completed for rider %s (%d chars)", rider_id, len(reply))
    return reply


