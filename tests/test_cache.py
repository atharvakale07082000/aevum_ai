"""The analysis cache: one full read and one agent run per decision version.

"One upstream read" is taken to mean: a repeat call still makes ONE cheap,
authorised freshness probe as the caller (so a caller who has lost access gets
the refusal, never somebody's cached analysis), but it does not repeat the
multi-request state load and does not re-run the agent.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from aevum_ai import aevum_client, app as app_module, cache as cache_module
from aevum_ai.aevum_client import AevumClient, Caller, UpstreamError
from aevum_ai.schemas import AskAnswer, ConsensusReport, DecisionSummary, LiveInsight

DECISION_ID = "dec-1"
ALICE = "Bearer alice"
BOB = "Bearer bob"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Upstream:
    """A fake Aevum API, as seen through AevumClient. Counts every read."""

    def __init__(self) -> None:
        self.version = "v1"
        self.probes = 0
        self.loads = 0
        self.denied: dict[str, UpstreamError] = {}
        self.load_error: UpstreamError | None = None
        self.load_gate: asyncio.Event | None = None
        self.votes = 0

    def client_class(upstream):  # noqa: N805 — a factory bound to this fake
        class FakeClient:
            def __init__(self, caller: Caller, base_url: str | None = None) -> None:
                self.caller = caller

            def _check(self) -> None:
                refusal = upstream.denied.get(self.caller.authorization)
                if refusal:
                    raise refusal

            async def decision_version(self, decision_id: str) -> str:
                upstream.probes += 1
                self._check()
                return upstream.version

            async def decision_state(self, decision_id: str) -> dict[str, Any]:
                upstream.loads += 1
                self._check()
                if upstream.load_gate is not None:
                    await upstream.load_gate.wait()
                if upstream.load_error is not None:
                    raise upstream.load_error
                return {
                    "decision": {
                        "id": decision_id, "title": "Q3", "status": "VOTING", "votingType": "SINGLE",
                        "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                    },
                    "tallies": [{"optionId": "a", "voteCount": upstream.votes, "totalWeight": upstream.votes}],
                    "participants": [{"userId": "u1", "role": "MEMBER"}],
                    "timeline": [],
                    "ballots": None,
                }

        return FakeClient

    def vote(self) -> None:
        self.votes += 1
        self.version = f"v{self.votes + 1}"


class Analyst:
    """Stands in for the agent. Counts runs per job."""

    def __init__(self) -> None:
        self.runs: dict[str, int] = {"summary": 0, "insight": 0, "report": 0, "ask": 0}
        self.fail_next = False
        self.degrade_next = False
        self.model_configured = True

    def _source(self) -> str:
        if self.degrade_next:
            self.degrade_next = False
            return "deterministic"
        return "model"

    def _maybe_fail(self) -> None:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("agent exploded")

    async def summarise(self, metrics, decision, recent):
        self.runs["summary"] += 1
        self._maybe_fail()
        return DecisionSummary(
            decisionId=metrics.decision_id, source=self._source(), headline=f"{metrics.ballots_cast} votes",
            summary="s", consensusRisk={"level": metrics.risk_level, "title": "t", "body": "b"},
            considerations=[], metrics={
                "standings": [], "leadingOptionId": None, "leadingLabel": None, "margin": 0,
                "eligibleVoters": 0, "ballotsCast": metrics.ballots_cast, "participation": 0,
                "rolesDiverge": False, "seniorFavouriteLabel": None, "seniorAgreement": 0.0,
                "seniorVoters": 0, "riskLevel": metrics.risk_level, "riskReasons": [],
                "consensusLikelihood": 0.5,
            },
        )

    async def insight(self, metrics, recent):
        self.runs["insight"] += 1
        self._maybe_fail()
        return LiveInsight(decisionId=metrics.decision_id, source=self._source(), body=f"{metrics.ballots_cast}")

    async def report(self, metrics, recent):
        self.runs["report"] += 1
        self._maybe_fail()
        return ConsensusReport(
            decisionId=metrics.decision_id, source=self._source(), likelihood=0.5, likelihoodBasis="x",
            optionId=None, optionLabel=None, body="r",
        )

    async def ask(self, metrics, decision, recent, question):
        self.runs["ask"] += 1
        self._maybe_fail()
        return AskAnswer(
            decisionId=metrics.decision_id, source=self._source(), answer=question, groundedOn=[],
        )

    def available(self) -> bool:
        return self.model_configured


@pytest.fixture
def world(monkeypatch):
    upstream = Upstream()
    analyst = Analyst()
    clock = Clock()
    fake = upstream.client_class()
    monkeypatch.setattr(app_module, "AevumClient", fake)
    monkeypatch.setattr(cache_module, "AevumClient", fake)
    for name in ("summarise", "insight", "report", "ask", "available"):
        monkeypatch.setattr(app_module.analyst, name, getattr(analyst, name))
    cache_module.reset(ttl_seconds=300, max_entries=64, clock=clock)
    yield upstream, analyst, clock
    cache_module.reset()


async def call(panel: str, auth: str = ALICE, question: str | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app_module.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        body = {"question": question} if question is not None else None
        return await http.post(
            f"/ai/decisions/{DECISION_ID}/{panel}",
            headers={"Authorization": auth, "X-Organization-Id": "org-1"},
            json=body,
        )


async def test_two_identical_calls_make_one_full_read_and_one_agent_run(world):
    upstream, analyst, _ = world
    first = await call("summary")
    second = await call("summary")
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert upstream.loads == 1
    assert analyst.runs["summary"] == 1
    # Each call still asks the backend, as the caller, whether it may read this
    # decision and which version it is on. That probe is the only repeat read.
    assert upstream.probes == 2


async def test_a_vote_invalidates_immediately_not_on_a_timer(world):
    upstream, analyst, clock = world
    before = await call("insight")
    upstream.vote()
    clock.now += 1  # well inside the TTL
    after = await call("insight")
    assert before.json()["body"] == "0" and after.json()["body"] == "1"
    assert upstream.loads == 2
    assert analyst.runs["insight"] == 2


async def test_different_panels_share_the_state_load_but_run_the_agent_separately(world):
    upstream, analyst, _ = world
    for panel in ("summary", "insight", "report"):
        assert (await call(panel)).status_code == 200
    assert upstream.loads == 1
    assert analyst.runs == {"summary": 1, "insight": 1, "report": 1, "ask": 0}


async def test_ask_is_keyed_on_the_normalised_question(world):
    upstream, analyst, _ = world
    await call("ask", question="Who is ahead?")
    await call("ask", question="  who IS   ahead? ")
    await call("ask", question="Why is B behind?")
    assert analyst.runs["ask"] == 2
    assert upstream.loads == 1


async def test_concurrent_panels_coalesce_into_one_state_load(world):
    upstream, analyst, _ = world
    upstream.load_gate = asyncio.Event()
    tasks = [asyncio.create_task(call(p)) for p in ("summary", "insight", "report", "summary")]
    await asyncio.sleep(0.05)
    upstream.load_gate.set()
    responses = await asyncio.gather(*tasks)
    assert all(r.status_code == 200 for r in responses)
    assert upstream.loads == 1
    # The two identical summary requests also share one agent run.
    assert analyst.runs == {"summary": 1, "insight": 1, "report": 1, "ask": 0}


@pytest.mark.parametrize("status,code", [(403, "ACCESS_DENIED"), (404, "NOT_FOUND")])
async def test_a_caller_who_cannot_read_gets_the_refusal_even_when_warm(world, status, code):
    upstream, analyst, _ = world
    assert (await call("summary", auth=ALICE)).status_code == 200
    upstream.denied[BOB] = UpstreamError(status, code, "no")
    refused = await call("summary", auth=BOB)
    assert refused.status_code == status
    assert refused.json()["code"] == code
    assert "headline" not in refused.json()
    assert analyst.runs["summary"] == 1


async def test_callers_do_not_share_entries(world):
    # Ballot visibility (and so the role-divergence analysis) depends on who is
    # asking, so one caller's analysis is never handed to another.
    upstream, analyst, _ = world
    await call("summary", auth=ALICE)
    await call("summary", auth=BOB)
    assert upstream.loads == 2
    assert analyst.runs["summary"] == 2


async def test_a_revoked_caller_is_refused_after_being_served(world):
    upstream, analyst, _ = world
    assert (await call("report")).status_code == 200
    upstream.denied[ALICE] = UpstreamError(403, "NOT_A_PARTICIPANT", "gone")
    assert (await call("report")).status_code == 403


async def test_a_failed_state_load_is_not_cached(world):
    upstream, analyst, _ = world
    upstream.load_error = UpstreamError(502, "UPSTREAM_UNAVAILABLE", "down")
    assert (await call("summary")).status_code == 502
    upstream.load_error = None
    assert (await call("summary")).status_code == 200
    assert upstream.loads == 2


async def test_a_failed_agent_run_is_not_cached(world):
    upstream, analyst, _ = world
    analyst.fail_next = True
    assert (await call("insight")).status_code == 500
    assert (await call("insight")).status_code == 200
    assert analyst.runs["insight"] == 2
    # The state itself was fine, so it was kept.
    assert upstream.loads == 1


async def test_a_degraded_answer_is_not_cached_when_a_model_is_configured(world):
    upstream, analyst, _ = world
    analyst.degrade_next = True
    assert (await call("insight")).json()["source"] == "deterministic"
    assert (await call("insight")).json()["source"] == "model"
    assert analyst.runs["insight"] == 2


async def test_deterministic_output_is_cached_when_no_model_is_configured(world):
    upstream, analyst, _ = world
    analyst.model_configured = False
    analyst.degrade_next = True
    await call("insight")
    await call("insight")
    assert analyst.runs["insight"] == 1


async def test_ttl_expiry_recomputes(world):
    upstream, analyst, clock = world
    await call("summary")
    clock.now += 299
    await call("summary")
    assert analyst.runs["summary"] == 1
    clock.now += 2
    await call("summary")
    assert analyst.runs["summary"] == 2
    assert upstream.loads == 2


async def test_lru_bounds_memory():
    clock = Clock()
    c = cache_module.AnalysisCache(ttl_seconds=300, max_entries=3, clock=clock)
    runs = 0

    async def compute():
        nonlocal runs
        runs += 1
        return runs

    for key in ("a", "b", "c"):
        await c.get_or_compute(key, compute)
    await c.get_or_compute("a", compute)  # touch a: b is now least recent
    await c.get_or_compute("d", compute)  # evicts b
    assert len(c) == 3
    assert runs == 4
    await c.get_or_compute("a", compute)
    assert runs == 4
    await c.get_or_compute("b", compute)
    assert runs == 5


async def test_a_cancelled_first_waiter_does_not_cancel_the_shared_load():
    c = cache_module.AnalysisCache(ttl_seconds=300, max_entries=8, clock=Clock())
    gate = asyncio.Event()
    runs = 0

    async def compute():
        nonlocal runs
        runs += 1
        await gate.wait()
        return "done"

    first = asyncio.create_task(c.get_or_compute("k", compute))
    await asyncio.sleep(0)
    second = asyncio.create_task(c.get_or_compute("k", compute))
    await asyncio.sleep(0)
    first.cancel()
    gate.set()
    assert await second == "done"
    assert runs == 1


async def test_ttl_zero_disables_caching(world):
    upstream, analyst, _ = world
    cache_module.reset(ttl_seconds=0, max_entries=64, clock=Clock())
    await call("summary")
    await call("summary")
    assert analyst.runs["summary"] == 2
    assert upstream.loads == 2


# ------------------------------------------------------------- the probe itself

def _transport(handler):
    real = httpx.AsyncClient

    class Patched(real):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    return Patched


def _decision(updated_at: str = "2026-09-19T10:00:00.000Z") -> dict[str, Any]:
    return {
        "id": DECISION_ID, "status": "VOTING", "updatedAt": updated_at, "finalizedOptionId": None,
        "options": [{"id": "a"}, {"id": "b"}],
    }


async def test_probe_version_changes_with_a_timeline_event_even_if_updated_at_does_not(monkeypatch):
    # Adding a participant or changing a role does not touch updatedAt in the
    # contract double, and two votes can land in the same millisecond. Every
    # one of those writes a timeline event, so the version follows the events.
    events = [{"id": "e1", "occurredAt": "2026-09-19T10:00:00.000Z"}]
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path + ("?" + request.url.query.decode() if request.url.query else ""))
        assert request.headers["Authorization"] == ALICE
        if request.url.path.endswith("/timeline"):
            return httpx.Response(200, json={"data": list(reversed(events))})
        return httpx.Response(200, json=_decision())

    monkeypatch.setattr(aevum_client.httpx, "AsyncClient", _transport(handler))
    client = AevumClient(Caller(ALICE, "org-1"), base_url="http://api")
    v1 = await client.decision_version(DECISION_ID)
    assert await client.decision_version(DECISION_ID) == v1
    events.append({"id": "e2", "occurredAt": "2026-09-19T10:00:00.000Z"})
    assert await client.decision_version(DECISION_ID) != v1
    assert f"/api/decisions/{DECISION_ID}" in seen
    assert any(s.startswith(f"/api/decisions/{DECISION_ID}/timeline") for s in seen)


async def test_probe_version_changes_with_updated_at(monkeypatch):
    stamp = {"v": "2026-09-19T10:00:00.000Z"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/timeline"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json=_decision(stamp["v"]))

    monkeypatch.setattr(aevum_client.httpx, "AsyncClient", _transport(handler))
    client = AevumClient(Caller(ALICE, None), base_url="http://api")
    v1 = await client.decision_version(DECISION_ID)
    stamp["v"] = "2026-09-19T10:00:00.001Z"
    assert await client.decision_version(DECISION_ID) != v1


async def test_probe_passes_the_decision_refusal_through(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/timeline"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404, json={"code": "NOT_FOUND", "message": "Decision not found"})

    monkeypatch.setattr(aevum_client.httpx, "AsyncClient", _transport(handler))
    with pytest.raises(UpstreamError) as refused:
        await AevumClient(Caller(BOB, None), base_url="http://api").decision_version(DECISION_ID)
    assert refused.value.status == 404
    assert refused.value.code == "NOT_FOUND"
