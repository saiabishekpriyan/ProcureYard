"""FastAPI application entry point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import Base, SessionLocal, engine
import seed
from routes.messages import router as messages_router
from routes.trace import router as trace_router
from routes.ops import router as ops_router
import payswift
from pathlib import Path
from fastapi.responses import HTMLResponse, FileResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create tables, seed data. Shutdown: close PaySwift client."""
    log.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)

    log.info("Seeding reference data...")
    db = SessionLocal()
    try:
        seed.run_all(db)
    finally:
        db.close()

    log.info("Application startup complete.")
    yield

    log.info("Shutting down PaySwift client...")
    await payswift.close_client()


app = FastAPI(
    title="QuickDrop Rider Payout Dispute Desk",
    description="AI-powered dispute resolution for QuickDrop riders.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(messages_router)
app.include_router(trace_router)
app.include_router(ops_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard() -> HTMLResponse:
    """Serve the single-page Operations Dashboard UI."""
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>QuickDrop Operations Dashboard</h1><p>dashboard.html not found.</p>", status_code=404)
