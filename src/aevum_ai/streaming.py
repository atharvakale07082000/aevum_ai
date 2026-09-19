"""The Ask answer as a server-sent event stream.

``POST /ai/decisions/{id}/ask/stream`` answers the same question as ``/ask``,
but as it is written:

    event: metrics   the measured figures, whole — computed before any model call
    event: token     {"text": "..."} — zero or more, in order
    event: done      the final AskAnswer, identical in shape to ``/ask``
    event: error     an ApiError envelope, if generation fails part-way

Everything that can refuse the request — no bearer token, the backend denying
or not finding the decision, a malformed question — happens *before* the
stream opens, so those stay ordinary JSON errors with their real status. Only a
failure after the first byte has to be reported in-band.

A client that goes away stops the generation: the frame generator checks for
the disconnect between events, and Starlette cancels it outright when the
connection drops. Either way the analyst's stream is closed, which cancels the
model invocation rather than leaving it running for nobody.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse

from .agents import analyst
from .schemas import ApiError, AskRequest

log = logging.getLogger("aevum_ai.streaming")

router = APIRouter()


def _frame(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


async def _frames(request: Request, metrics: Any, events: AsyncIterator[tuple[str, Any]]) -> AsyncIterator[bytes]:
    # The figures do not depend on the model, so they go out before it is asked.
    yield _frame("metrics", analyst._metrics_out(metrics).model_dump())
    try:
        async for kind, payload in events:
            if await request.is_disconnected():
                log.info("ask stream: client disconnected, stopping generation")
                return
            if kind == "token":
                yield _frame("token", {"text": payload})
            elif kind == "done":
                yield _frame("done", payload.model_dump())
    except asyncio.CancelledError:
        log.info("ask stream: cancelled, stopping generation")
        raise
    except Exception as exc:  # noqa: BLE001 — the status line has gone; report in-band
        log.warning("ask stream failed after it started: %s", exc)
        yield _frame("error", ApiError(
            status=502, error="Bad Gateway", code="AI_STREAM_FAILED",
            message="The answer could not be finished.",
        ).model_dump())
    finally:
        # Closing the analyst's generator is what cancels the model call.
        await events.aclose()  # type: ignore[attr-defined]


@router.post("/ai/decisions/{decision_id}/ask/stream")
async def ask_stream(
    decision_id: str,
    body: AskRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_organization_id: str | None = Header(default=None),
):
    # Resolved at call time: app.py imports this module, and reading the helpers
    # through the module keeps one definition of how a decision is loaded.
    from . import app as app_module

    caller = app_module._caller(authorization, x_organization_id)
    metrics, decision, recent = await app_module._load(decision_id, caller)
    events = analyst.ask_stream(metrics, decision, recent, body.question)
    return StreamingResponse(
        _frames(request, metrics, events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            # Stops nginx-style proxies holding tokens back until the end.
            "X-Accel-Buffering": "no",
        },
    )
