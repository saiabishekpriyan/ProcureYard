"""GET /ops/pending — items awaiting human action and action endpoints."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db
from models import OpsQueueItem, AuditTrace

log = logging.getLogger(__name__)
router = APIRouter()


class OpsItem(BaseModel):
    id: str
    rider_id: str
    type: str
    amount: int | None
    reason: str
    created_at: str


class ResolveRequest(BaseModel):
    notes: str | None = None


class ActionResponse(BaseModel):
    success: bool
    message: str
    id: str
    resolution: str
    resolved_at: str


@router.get("/ops/pending", response_model=list[OpsItem])
def get_ops_pending(db: Session = Depends(get_db)) -> list[OpsItem]:
    """Return all unresolved ops queue items (approvals and escalations)."""
    rows = (
        db.query(OpsQueueItem)
        .filter(OpsQueueItem.resolved_at.is_(None))
        .order_by(OpsQueueItem.created_at)
        .all()
    )
    return [
        OpsItem(
            id=row.id,
            rider_id=row.rider_id,
            type=row.type,
            amount=row.amount,
            reason=row.reason,
            created_at=row.created_at.isoformat(),
        )
        for row in rows
    ]


@router.post("/ops/{item_id}/approve", response_model=ActionResponse)
def approve_ops_item(
    item_id: str,
    body: ResolveRequest | None = None,
    db: Session = Depends(get_db),
) -> ActionResponse:
    """Approve a pending ops queue item."""
    item = db.query(OpsQueueItem).filter(OpsQueueItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Ops queue item not found")
    if item.resolved_at:
        raise HTTPException(status_code=400, detail="Item already resolved")

    now = datetime.now(timezone.utc)
    item.resolved_at = now
    item.resolution = "approved"

    notes = body.notes if body else None
    db.add(
        AuditTrace(
            rider_id=item.rider_id,
            type="ops_action",
            name="item_approved",
            input={"item_id": item.id, "type": item.type, "amount": item.amount, "notes": notes},
            output={"resolution": "approved"},
            at=now,
        )
    )
    db.commit()
    log.info("Ops item %s approved for rider %s", item_id, item.rider_id)
    return ActionResponse(
        success=True,
        message=f"Item {item_id} successfully approved",
        id=item.id,
        resolution="approved",
        resolved_at=now.isoformat(),
    )


@router.post("/ops/{item_id}/reject", response_model=ActionResponse)
def reject_ops_item(
    item_id: str,
    body: ResolveRequest | None = None,
    db: Session = Depends(get_db),
) -> ActionResponse:
    """Reject a pending ops queue item."""
    item = db.query(OpsQueueItem).filter(OpsQueueItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Ops queue item not found")
    if item.resolved_at:
        raise HTTPException(status_code=400, detail="Item already resolved")

    now = datetime.now(timezone.utc)
    item.resolved_at = now
    item.resolution = "rejected"

    notes = body.notes if body else None
    db.add(
        AuditTrace(
            rider_id=item.rider_id,
            type="ops_action",
            name="item_rejected",
            input={"item_id": item.id, "type": item.type, "amount": item.amount, "notes": notes},
            output={"resolution": "rejected"},
            at=now,
        )
    )
    db.commit()
    log.info("Ops item %s rejected for rider %s", item_id, item.rider_id)
    return ActionResponse(
        success=True,
        message=f"Item {item_id} successfully rejected",
        id=item.id,
        resolution="rejected",
        resolved_at=now.isoformat(),
    )
