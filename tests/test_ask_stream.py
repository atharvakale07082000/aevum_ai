"""The streamed Ask answer.

Three promises from the sprint task, each pinned here:

1. The measured figures arrive first, whole, before any token — they are
   computed before the model is called, so there is no reason to make a reader
   wait for them.
2. Tokens then arrive as they are produced, and a `done` event carries the same
   structured answer the non-streaming endpoint returns.
3. A client that goes away stops the generation. An orphaned model call keeps
   spending quota (OpenRouter's free tier is 20 rpm) on an answer nobody will read.

Errors found before streaming starts — no bearer, the backend refusing the
decision — stay ordinary JSON errors with their real status. A 200 stream that
says "403" inside it would bypass every client's status handling.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import httpx
import pytest

from aevum_ai import app as app_module
from aevum_ai.aevum_client import UpstreamError
from aevum_ai.agents import analyst
from aevum_ai.analysis import analyse
from aevum_ai.config import settings

DECISION_ID = "11111111-1111-1111-1111-111111111111"
PATH = f"/ai/decisions/{DECISION_ID}/ask/stream"
AUTH = {"Authorization": "Bearer test-token", "X-Organization-Id": "org-1"}

DECISION = {
    "id": DECISION_ID,
    "title": "Which region next?",
    "status": "VOTING",
    "votingType": "SINGLE",
    "options": [
        {"id": "a", "label": "Lisbon", "description": "Cheaper."},
        {"id": "b", "label": "Toronto", "description": "Closer."},
    ],
}


def _metrics():
    return analyse(
        decision=DECISION,
        tallies=[
            {"optionId": "a", "voteCount": 2, "totalWeight": 2},
            {"optionId": "b", "voteCount": 1, "totalWeight": 1},
        ],
        participants=[
            {"userId": "u1", "role": "LEAD"},
            {"userId": "u2", "role": "MEMBER"},
            {"userId": "u3", "role": "MEMBER"},
        ],
    )


@pytest.fixture(autouse=True)
def deterministic(monkeypatch):
    monkeypatch.delenv("AEVUM_AI_MODEL_PROVIDER", raising=False)
    # Tests do not need the human-paced delay between deterministic chunks.
    monkeypatch.setenv("AEVUM_AI_ASK_STREAM_CHUNK_DELAY_MS", "0")
    settings.cache_clear()
    yield
    settings.cache_clear()


@pytest.fixture
def loaded(monkeypatch):
    """Stand in for the Aevum API: `_load` returns a real, measured decision."""
    calls: list[str] = []

    async def fake_load(decision_id, caller):
        calls.append(decision_id)
        return _metrics(), DECISION, "- A voted at now"

    monkeypatch.setattr(app_module, "_load", fake_load)
    return calls


def _parse_sse(raw: str) -> list[tuple[str, Any]]:
    events = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        name, data = "message", []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].strip())
        events.append((name, json.loads("\n".join(data)) if data else None))
    return events


async def _post(path: str, headers: dict[str, str], body: dict[str, Any]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, headers=headers, json=body)


# ------------------------------------------------------------- event order

async def test_metrics_come_first_then_tokens_then_done(loaded):
    res = await _post(PATH, AUTH, {"question": "Where are we stuck?"})

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(res.text)
    names = [n for n, _ in events]

    assert names[0] == "metrics", names
    assert names[-1] == "done", names
    assert names.count("token") >= 3, "the deterministic answer must stream in several chunks"
    assert set(names[1:-1]) == {"token"}

    metrics = events[0][1]
    assert metrics["ballotsCast"] == 3
    assert metrics["eligibleVoters"] == 3
    assert metrics["leadingLabel"] == "Lisbon"

    done = events[-1][1]
    assert done["source"] == "deterministic"
    assert done["decisionId"] == DECISION_ID
    assert done["groundedOn"]
    # The tokens are the answer, in order, nothing more and nothing less.
    assert "".join(e[1]["text"] for e in events if e[0] == "token") == done["answer"]


async def test_the_done_event_matches_the_non_streaming_answer(loaded):
    # Compared against the analyst rather than the /ask route: /ask goes through
    # the panel cache, which is not what this test is about.
    streamed = _parse_sse((await _post(PATH, AUTH, {"question": "Where are we stuck?"})).text)[-1][1]
    whole = (await analyst.ask(_metrics(), DECISION, "", "Where are we stuck?")).model_dump()
    for key in ("decisionId", "source", "answer", "groundedOn"):
        assert streamed[key] == whole[key], key


# ------------------------------------------------ errors before the stream

async def test_no_bearer_is_a_json_401_not_a_stream(loaded):
    res = await _post(PATH, {}, {"question": "Where are we stuck?"})
    assert res.status_code == 401
    assert res.headers["content-type"].startswith("application/json")
    assert res.json()["code"] == "UNAUTHENTICATED"
    assert loaded == [], "the decision must not be read without a caller"


@pytest.mark.parametrize("status,code", [(403, "ACCESS_DENIED"), (404, "NOT_FOUND")])
async def test_an_upstream_refusal_keeps_its_status(monkeypatch, status, code):
    async def refusing_load(decision_id, caller):
        raise UpstreamError(status, code, "No.")

    monkeypatch.setattr(app_module, "_load", refusing_load)
    res = await _post(PATH, AUTH, {"question": "Where are we stuck?"})
    assert res.status_code == status
    assert res.headers["content-type"].startswith("application/json")
    assert res.json()["code"] == code


async def test_an_empty_question_is_rejected_before_streaming(loaded):
    res = await _post(PATH, AUTH, {"question": ""})
    assert res.status_code == 422
    assert loaded == []


async def test_a_failure_mid_stream_is_an_error_event(loaded, monkeypatch):
    async def breaks(*_args, **_kwargs):
        yield ("token", "Half an ")
        raise RuntimeError("boom")

    monkeypatch.setattr(analyst, "ask_stream", breaks)
    events = _parse_sse((await _post(PATH, AUTH, {"question": "q"})).text)
    assert [n for n, _ in events] == ["metrics", "token", "error"]
    error = events[-1][1]
    assert {"timestamp", "status", "error", "code", "message"} <= set(error)
    assert error["status"] == 502


# ------------------------------------------------------------ cancellation

async def test_a_disconnect_stops_the_generation(loaded, monkeypatch):
    """Drive the ASGI app by hand so the client can hang up mid-answer."""
    produced: list[int] = []
    closed = asyncio.Event()

    async def endless(*_args, **_kwargs):
        try:
            i = 0
            while True:
                produced.append(i)
                yield ("token", f"t{i} ")
                i += 1
                await asyncio.sleep(0.01)
        finally:
            closed.set()

    monkeypatch.setattr(analyst, "ask_stream", endless)

    body = json.dumps({"question": "q"}).encode()
    sent: list[dict[str, Any]] = []
    first_token = asyncio.Event()
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await first_token.wait()
        await asyncio.sleep(0.05)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and b"event: token" in message.get("body", b""):
            first_token.set()

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": PATH, "raw_path": PATH.encode(),
        "query_string": b"", "root_path": "", "server": ("test", 80), "client": ("test", 1),
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"authorization", b"Bearer test-token"),
        ],
    }

    await asyncio.wait_for(app_module.app(scope, receive, send), timeout=5)
    await asyncio.wait_for(closed.wait(), timeout=1)

    count = len(produced)
    await asyncio.sleep(0.1)
    assert len(produced) == count, "tokens kept being produced after the client left"
    assert count < 50, f"generation ran on for {count} tokens after a 50ms disconnect"
    bodies = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert bodies.startswith(b"event: metrics")
    assert b"event: done" not in bodies


class _FakeAgent:
    """A model that streams slowly and honours the cancel signal like Strands does."""

    def __init__(self) -> None:
        self.cancel_signal: threading.Event | None = None
        self.closed = False
        self.deltas = 0

    async def stream_async(self, prompt, *, cancel_signal=None, **_kwargs):
        self.cancel_signal = cancel_signal
        try:
            for i in range(1000):
                if cancel_signal is not None and cancel_signal.is_set():
                    return
                self.deltas += 1
                yield {"data": f"word{i} "}
                await asyncio.sleep(0.005)
            yield {"result": None}
        finally:
            self.closed = True


async def test_model_mode_forwards_deltas_and_ends_with_a_model_answer(monkeypatch):
    class ShortAgent(_FakeAgent):
        async def stream_async(self, prompt, *, cancel_signal=None, **_kwargs):
            self.prompt = prompt
            for word in ["Lisbon ", "is ", "narrowly ", "ahead."]:
                yield {"data": word}
            yield {"reasoningText": "should never reach the reader"}
            yield {"result": None}

    agent = ShortAgent()
    monkeypatch.setattr(analyst, "_agent", lambda: agent)
    events = [e async for e in analyst.ask_stream(_metrics(), DECISION, "", "Who is ahead?")]

    assert [e[1] for e in events[:-1]] == ["Lisbon ", "is ", "narrowly ", "ahead."]
    kind, answer = events[-1]
    assert kind == "done"
    assert answer.source == "model"
    assert answer.answer == "Lisbon is narrowly ahead."
    # The same prompt, and therefore the same no-invented-numbers rule, as /ask.
    assert "DECISION STATE" in agent.prompt and "Who is ahead?" in agent.prompt


async def test_closing_the_stream_cancels_the_model_call(monkeypatch):
    agent = _FakeAgent()
    monkeypatch.setattr(analyst, "_agent", lambda: agent)

    stream = analyst.ask_stream(_metrics(), DECISION, "", "Where are we stuck?")
    for _ in range(3):
        kind, _text = await stream.__anext__()
        assert kind == "token"
    await stream.aclose()

    assert agent.cancel_signal is not None and agent.cancel_signal.is_set()
    assert agent.closed, "the underlying model stream was left open"
    deltas = agent.deltas
    await asyncio.sleep(0.05)
    assert agent.deltas == deltas


async def test_a_model_failure_ends_in_the_same_fallback_as_ask(monkeypatch):
    class Failing(_FakeAgent):
        async def stream_async(self, prompt, **_kwargs):
            raise RuntimeError("provider down")
            yield  # pragma: no cover

    monkeypatch.setattr(analyst, "_agent", lambda: Failing())
    events = [e async for e in analyst.ask_stream(_metrics(), DECISION, "", "q")]
    assert events[-1][0] == "done"
    assert events[-1][1].source == "deterministic"
    assert "could not be answered" in events[-1][1].answer
