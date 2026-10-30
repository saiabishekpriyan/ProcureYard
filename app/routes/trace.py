"""GET /trace/{rider_id} — ordered agent step log and conversations."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Any

from db import get_db
from models import AuditTrace, Message

router = APIRouter()


class TraceStep(BaseModel):
    at: str
    type: str
    name: str
    input: Any
    output: Any


class ConversationTurn(BaseModel):
    message_id: str
    rider_id: str
    text: str
    reply: str | None
    received_at: str
    has_reply: bool


class RiderHistory(BaseModel):
    rider_id: str
    conversations: list[ConversationTurn]
    traces: list[TraceStep]


@router.get("/trace/{rider_id}", response_model=list[TraceStep])
def get_trace(rider_id: str, db: Session = Depends(get_db)) -> list[TraceStep]:
    """Return all agent steps for a rider, ordered chronologically."""
    rows = (
        db.query(AuditTrace)
        .filter(AuditTrace.rider_id == rider_id)
        .order_by(AuditTrace.at)
        .all()
    )
    return [
        TraceStep(
            at=row.at.isoformat(),
            type=row.type,
            name=row.name,
            input=row.input,
            output=row.output,
        )
        for row in rows
    ]


@router.get("/trace/{rider_id}/conversation", response_model=RiderHistory)
def get_rider_conversation(rider_id: str, db: Session = Depends(get_db)) -> RiderHistory:
    """Return chronological conversation turns and technical trace events for a rider."""
    # 1. Fetch messages
    msg_rows = (
        db.query(Message)
        .filter(Message.rider_id == rider_id)
        .order_by(Message.received_at.asc(), Message.created_at.asc())
        .all()
    )

    conversations = [
        ConversationTurn(
            message_id=m.message_id,
            rider_id=m.rider_id,
            text=m.text,
            reply=m.reply,
            received_at=m.received_at.isoformat() if m.received_at else m.created_at.isoformat(),
            has_reply=bool(m.reply and m.reply.strip()),
        )
        for m in msg_rows
    ]

    # 2. Fetch technical trace steps
    trace_rows = (
        db.query(AuditTrace)
        .filter(AuditTrace.rider_id == rider_id)
        .order_by(AuditTrace.at.asc())
        .all()
    )

    traces = [
        TraceStep(
            at=row.at.isoformat(),
            type=row.type,
            name=row.name,
            input=row.input,
            output=row.output,
        )
        for row in trace_rows
    ]

    return RiderHistory(
        rider_id=rider_id,
        conversations=conversations,
        traces=traces,
    )
