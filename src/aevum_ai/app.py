"""HTTP surface for Aevum AI.

Four endpoints, one per panel in the design. Each takes the caller's own
credentials, reads the decision through the Aevum API as that person, computes
the measured facts, and asks the agent for the words around them.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .aevum_client import AevumClient, Caller, UpstreamError
from .agents import analyst
from .analysis import analyse
from .config import settings
from .schemas import ApiError, AskRequest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aevum_ai")

app = FastAPI(
    title="Aevum AI",
    version="0.1.0",
    description="Decision analysis for Aevum. Reads as the caller; computes every number.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings().cors_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _error(status: int, code: str, message: str) -> JSONResponse:
    reason = {
        400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found",
        429: "Too Many Requests", 502: "Bad Gateway", 503: "Service Unavailable",
    }.get(status, "Error")
    return JSONResponse(
        status_code=status,
        content=ApiError(status=status, error=reason, code=code, message=message).model_dump(),
    )


@app.exception_handler(UpstreamError)
async def _upstream(_: Request, exc: UpstreamError) -> JSONResponse:
    # The backend's refusal is the right answer; passing it through unchanged
    # means the frontend's existing code-based handling just works.
    return _error(exc.status, exc.code, exc.message)


@app.exception_handler(HTTPException)
async def _http(_: Request, exc: HTTPException) -> JSONResponse:
    # Reuse the backend's own codes so the frontend branches on one vocabulary.
    code = {
        401: "UNAUTHENTICATED",
        403: "ACCESS_DENIED",
        404: "NOT_FOUND",
        422: "VALIDATION_FAILED",
    }.get(exc.status_code, "BAD_REQUEST")
    return _error(exc.status_code, code, str(exc.detail))


def _caller(authorization: str | None, organization_id: str | None) -> Caller:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="A bearer token is required.")
    return Caller(authorization=authorization, organization_id=organization_id)


async def _load(decision_id: str, caller: Caller) -> tuple[Any, dict[str, Any], str]:
    state = await AevumClient(caller).decision_state(decision_id)
    metrics = analyse(
        decision=state["decision"],
        tallies=state["tallies"],
        participants=state["participants"],
        ballots=state["ballots"],
    )
    recent = _recent(state["timeline"], state["participants"])
    return metrics, state["decision"], recent


def _recent(timeline: list[dict[str, Any]], participants: list[dict[str, Any]]) -> str:
    """The last few events, with actor ids resolved to names."""
    names = {p["userId"]: (p.get("displayName") or p.get("email") or "someone") for p in participants}
    verbs = {
        "DECISION_CREATED": "opened the decision",
        "OPTION_ADDED": "added an option",
        "PARTICIPANT_ADDED": "brought someone in",
        "VOTE_CAST": "voted",
        "VOTING_OPENED": "opened the ballot",
        "VOTING_LOCKED": "locked the ballot",
        "DECISION_FINALIZED": "finalized the decision",
    }
    lines = []
    for event in timeline[:8]:
        who = names.get(event.get("actorUserId"), "someone")
        what = verbs.get(event.get("eventType"), event.get("eventType", "").lower().replace("_", " "))
        lines.append(f"- {who} {what} at {event.get('occurredAt')}")
    return "\n".join(lines)


@app.get("/health")
async def health() -> dict[str, Any]:
    config = settings()
    configured = analyst.available()
    return {
        "status": "UP",
        "modelConfigured": configured,
        # Stated plainly so an operator can see, without reading code, whether
        # the panels are model-written or measured-and-templated.
        "mode": "model" if configured else "deterministic",
        # Which provider and model, because "the panels read oddly" is nearly
        # always answered by one of these two values being unexpected.
        "provider": config.model_provider if configured else None,
        "model": (
            config.nim_model if config.model_provider == "nvidia" else config.model_id
        ) if configured else None,
    }


@app.post("/ai/decisions/{decision_id}/summary")
async def summary(
    decision_id: str,
    authorization: str | None = Header(default=None),
    x_organization_id: str | None = Header(default=None),
):
    caller = _caller(authorization, x_organization_id)
    metrics, decision, recent = await _load(decision_id, caller)
    return await analyst.summarise(metrics, decision, recent)


@app.post("/ai/decisions/{decision_id}/insight")
async def insight(
    decision_id: str,
    authorization: str | None = Header(default=None),
    x_organization_id: str | None = Header(default=None),
):
    caller = _caller(authorization, x_organization_id)
    metrics, _decision, recent = await _load(decision_id, caller)
    return await analyst.insight(metrics, recent)


@app.post("/ai/decisions/{decision_id}/report")
async def report(
    decision_id: str,
    authorization: str | None = Header(default=None),
    x_organization_id: str | None = Header(default=None),
):
    caller = _caller(authorization, x_organization_id)
    metrics, _decision, recent = await _load(decision_id, caller)
    return await analyst.report(metrics, recent)


@app.post("/ai/decisions/{decision_id}/ask")
async def ask(
    decision_id: str,
    body: AskRequest,
    authorization: str | None = Header(default=None),
    x_organization_id: str | None = Header(default=None),
):
    caller = _caller(authorization, x_organization_id)
    metrics, decision, recent = await _load(decision_id, caller)
    return await analyst.ask(metrics, decision, recent, body.question)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings().port)


if __name__ == "__main__":
    main()
