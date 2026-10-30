# QuickDrop Rider Payout Dispute Desk

AI-powered payout dispute resolution for QuickDrop delivery riders via WhatsApp-style messaging.

## Quick Start

```bash
# Set your Gemini API key
export GEMINI_API_KEY=your_key_here   # Linux/macOS
$env:GEMINI_API_KEY = "your_key_here" # Windows PowerShell

# Start the full stack (PaySwift sandbox + PostgreSQL + app)
docker compose up

# App is ready when: GET http://localhost:8000/health → {"status":"ok"}
```

## Run Evals

```bash
python evals/run.py http://localhost:8000
```

Reads `data/conversations.json` (24 scenarios), posts each rider turn to `/messages`, and checks:
- Expected approval items in `/ops/pending`
- Expected escalation items in `/ops/pending`
- Expected phrases in the final reply

## Environment Variables

See [`.env.example`](.env.example) for the full list. Only `GEMINI_API_KEY` needs to be supplied; all others have working defaults inside Docker Compose.

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /` or `GET /dashboard` | Web Operations Dashboard (Queue approvals & Rider dispute trace explorer) |
| `GET /health` | Liveness check → `{"status":"ok"}` |
| `POST /messages` | Receive a rider message, run agent, return `{"reply":"..."}` |
| `GET /trace/{rider_id}` | Ordered agent step log for a rider |
| `GET /ops/pending` | Items awaiting human approval or escalation |
| `POST /ops/{id}/approve` | Approve a pending ops queue item |
| `POST /ops/{id}/reject` | Reject a pending ops queue item |

## Architecture

```
POST /messages
    │
    ├─ Dedup on message_id (UNIQUE DB constraint)
    │
    └─ run_agent()
          │
          ├─ Gemini 1.5 Flash (tool-calling loop, ≤10 turns)
          │     tools: lookup_rider, lookup_trips, lookup_payout_lines,
          │            calculate_payout, trigger_payout,
          │            queue_approval, queue_escalation
          │
          ├─ calc.py   ← all rupee arithmetic (no LLM involvement)
          ├─ policy.py ← policy guards: ₹200 cap, once-per-day, dispute window
          └─ payswift.py ← PaySwift HTTP client (retry 429/5xx, reconcile on timeout)
```

### What the model decides
- Whether the complaint is about a missing fare, surge, incentive, or duplicate penalty
- Which date/trip the rider is referring to
- The tone and wording of the reply to the rider

### What the code decides (never the model)
- Every rupee calculation (`calc.py`)
- Whether a payout is policy-eligible (`policy.py`)
- Whether to call PaySwift (`payswift.py`)
- Deduplication of incoming messages (DB UNIQUE on `message_id`)

## Business Rules

- **Fare**: ₹25 base + ₹6/km beyond the first 2 km; surge multiplier on whole fare; ROUND_HALF_UP
- **Daily incentive**: ₹150 if the rider completes ≥ 12 trips on the same IST calendar day
- **Penalty**: ₹10 per rider-cancelled trip (customer cancellations: no fare, no penalty)
- **Auto-pay limit**: ₹200 per dispute; once per rider per IST calendar day
- **Dispute window**: 7 IST calendar days

## Assumptions

1. `payout_date` in `payout_lines.csv` is the IST calendar date matching `started_at` UTC timestamps.
2. Malformed rider IDs in CSVs (`R7`, `r19`) are silently normalised to `R007`, `R019` without modifying source files.
3. `POST /v1/payouts` returns HTTP 200 (not 201) for both first and idempotent repeats — verified against the running sandbox.
4. Trip `T694881` with `distance_km=2600` is treated as valid input data; the fare calculation produces a large number which the policy's ₹200 cap prevents from being auto-paid.
5. `conversations.json` contains 24 scenarios (BRIEF.md says 25 — actual count is 24).
6. `GEMINI_API_KEY` is required for the agent; if absent, the app still starts and returns a safe fallback reply.

## What was skipped

- Ops dashboard UI (HTML page for approvals) — time constraint; `/ops/pending` JSON endpoint is present
- GitHub Actions CI for Docker integration tests — unit tests only in CI
- Multi-turn conversation state (conversation history is stateless per message; each message is handled independently)

## Eval Results

*Run `python evals/run.py http://localhost:8000` to see current results.*
