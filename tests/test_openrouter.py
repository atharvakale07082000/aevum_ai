"""The OpenRouter layer: throttling, wiring, and the fallback that must hold.

None of this talks to OpenRouter. A test that needs a real key is a test that does
not run in CI, and the things worth asserting here — that the rate limiter
actually delays, that both entry points go through it, that a missing key
degrades instead of failing — are all observable without one.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from aevum_ai import openrouter
from aevum_ai.agents import analyst
from aevum_ai.config import Settings, settings


@pytest.fixture(autouse=True)
def _clear_caches():
    """Settings and the model are both cached; every test needs a clean slate."""
    settings.cache_clear()
    openrouter.get_openrouter_model.cache_clear()
    openrouter._throttle.__init__()  # type: ignore[misc]
    yield
    settings.cache_clear()
    openrouter.get_openrouter_model.cache_clear()


# --------------------------------------------------------------- rate limiter


async def test_bucket_allows_up_to_the_limit_without_waiting():
    bucket = openrouter._RpmBucket(limit=5)
    started = time.monotonic()
    for _ in range(5):
        await bucket.acquire()
    assert time.monotonic() - started < 0.1


async def test_bucket_makes_the_call_over_the_limit_wait():
    """The point of the bucket: the 3rd call in a 2-per-minute budget blocks.

    Verified by patching the sliding window to two seconds rather than sixty, so
    the assertion is about behaviour rather than about the test's patience.
    """
    bucket = openrouter._RpmBucket(limit=2)
    await bucket.acquire()
    await bucket.acquire()

    # Age the window so the wait is short but real.
    bucket._stamps[0] -= 59.9
    bucket._stamps[1] -= 59.9

    started = time.monotonic()
    await bucket.acquire()
    waited = time.monotonic() - started
    assert waited > 0.05, f"third call did not wait (waited {waited:.3f}s)"


async def test_bucket_is_a_sliding_window_not_a_fixed_one():
    """A fixed window would allow the whole budget twice across a boundary."""
    bucket = openrouter._RpmBucket(limit=3)
    for _ in range(3):
        await bucket.acquire()
    # Two of the three calls fall out of the window; the third does not.
    bucket._stamps[0] -= 61
    bucket._stamps[1] -= 61

    started = time.monotonic()
    await bucket.acquire()
    await bucket.acquire()
    assert time.monotonic() - started < 0.1
    assert len(bucket._stamps) == 3


async def test_concurrency_cap_is_enforced():
    """Four panels open at once; only `openrouter_max_concurrency` run at a time."""
    import os

    os.environ["AEVUM_AI_OPENROUTER_MAX_CONCURRENCY"] = "2"
    os.environ["AEVUM_AI_OPENROUTER_RPM_LIMIT"] = "1000"
    settings.cache_clear()
    openrouter._throttle.__init__()  # type: ignore[misc]

    inflight = 0
    peak = 0

    async def call():
        nonlocal inflight, peak
        async with openrouter._throttle:
            inflight += 1
            peak = max(peak, inflight)
            await asyncio.sleep(0.05)
            inflight -= 1

    await asyncio.gather(*(call() for _ in range(6)))
    assert peak <= 2, f"{peak} calls ran at once against a cap of 2"

    del os.environ["AEVUM_AI_OPENROUTER_MAX_CONCURRENCY"]
    del os.environ["AEVUM_AI_OPENROUTER_RPM_LIMIT"]


# ------------------------------------------------------------------- wiring


def test_both_entry_points_are_overridden():
    """Wrapping only `stream` would leave every real call unthrottled.

    Every job in this service asks for structured output, so `structured_output`
    is the path that actually carries traffic. This asserts the override exists
    on the subclass rather than being inherited from OpenAIModel.
    """
    assert "stream" in openrouter.OpenRouterModel.__dict__
    assert "structured_output" in openrouter.OpenRouterModel.__dict__


def test_openrouter_model_is_built_from_settings(monkeypatch):
    monkeypatch.setenv("AEVUM_AI_MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("AEVUM_AI_OPENROUTER_API_KEY", "sk-or-not-a-real-key")
    monkeypatch.setenv("AEVUM_AI_OPENROUTER_MODEL", "test/test-model")
    settings.cache_clear()
    openrouter.get_openrouter_model.cache_clear()

    model = openrouter.get_openrouter_model()
    assert isinstance(model, openrouter.OpenRouterModel)
    config = model.get_config()
    assert config["model_id"] == "test/test-model"
    # Without this the model spends the budget on a thinking block that never
    # reaches the reader and can crowd out the JSON on a structured call.
    assert config["params"]["extra_body"]["reasoning"]["enabled"] is False


def test_default_model_is_the_validated_one():
    """A model that cannot drive structured output degrades silently.

    Pinned deliberately: this is the model validated against a real Strands
    structured-output call. Changing it is a decision, and a failing test is the
    right way to be asked to make it on purpose.
    """
    assert Settings().openrouter_model == "nvidia/nemotron-3-ultra-550b-a55b:free"


def test_token_budget_covers_deliberation():
    """1200 was too small for a reasoning model and failed whole requests."""
    assert Settings().max_tokens >= 3000


# ------------------------------------------------------- degraded, not broken


def test_openrouter_without_a_key_falls_back_rather_than_raising(monkeypatch):
    """A missing variable is a misconfiguration, not an outage.

    The figures do not need a model, and they are the part that matters. So the
    service answers with template prose rather than 503-ing every panel.
    """
    monkeypatch.setenv("AEVUM_AI_MODEL_PROVIDER", "openrouter")
    monkeypatch.delenv("AEVUM_AI_OPENROUTER_API_KEY", raising=False)
    settings.cache_clear()

    assert analyst._model() is None
    assert analyst.available() is False


def test_provider_none_is_still_deterministic(monkeypatch):
    monkeypatch.delenv("AEVUM_AI_MODEL_PROVIDER", raising=False)
    settings.cache_clear()
    assert analyst.available() is False


async def test_every_job_returns_measured_figures_with_no_model(monkeypatch):
    """The four jobs are coroutines now; the deterministic path must still hold.

    This is the guarantee the whole design rests on: with no provider at all,
    every panel still answers, with the same numbers, marked as deterministic.
    """
    monkeypatch.delenv("AEVUM_AI_MODEL_PROVIDER", raising=False)
    settings.cache_clear()

    from aevum_ai.analysis import analyse

    decision = {
        "id": "11111111-1111-1111-1111-111111111111",
        "title": "Which region next?",
        "status": "VOTING",
        "votingType": "SINGLE",
        "options": [
            {"id": "a", "label": "Lisbon", "description": "Cheaper."},
            {"id": "b", "label": "Toronto", "description": "Closer."},
        ],
    }
    tallies = [
        {"optionId": "a", "voteCount": 2, "totalWeight": 2},
        {"optionId": "b", "voteCount": 1, "totalWeight": 1},
    ]
    participants = [
        {"userId": "u1", "displayName": "A", "role": "LEAD"},
        {"userId": "u2", "displayName": "B", "role": "MEMBER"},
        {"userId": "u3", "displayName": "C", "role": "MEMBER"},
    ]
    metrics = analyse(decision=decision, tallies=tallies, participants=participants)

    summary = await analyst.summarise(metrics, decision, "")
    assert summary.source == "deterministic"
    assert summary.metrics.ballotsCast == 3

    live = await analyst.insight(metrics, "")
    assert live.source == "deterministic"

    rep = await analyst.report(metrics, "")
    assert rep.source == "deterministic"

    answer = await analyst.ask(metrics, decision, "", "Where are we stuck?")
    assert answer.source == "deterministic"


# ------------------------------------------------------------ configuration


def test_every_variable_in_env_example_binds_to_a_real_setting():
    """A typo here fails silently, which is the worst way for it to fail.

    `pydantic-settings` ignores unknown environment variables. So
    `AEVUM_AI_OPENROUTR_API_KEY` in a .env does not raise — it leaves
    `openrouter_api_key` unset, the provider falls back, and every panel quietly
    returns template prose while the operator looks at a filled-in .env and
    wonders why the key is not working.
    """
    import pathlib

    example = pathlib.Path(__file__).parent.parent / ".env.example"
    names = [
        line.split("=", 1)[0]
        for line in example.read_text().splitlines()
        if line.strip().startswith("AEVUM_AI_") and "=" in line
    ]
    assert names, "no variables found in .env.example"

    fields = set(Settings.model_fields)
    dead = [n for n in names if n.removeprefix("AEVUM_AI_").lower() not in fields]
    assert not dead, f"variables that bind to nothing: {dead}"


def test_a_filled_in_env_turns_the_provider_on(monkeypatch):
    """The .env shape shipped in .env.example actually reaches the model."""
    monkeypatch.setenv("AEVUM_AI_MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("AEVUM_AI_OPENROUTER_API_KEY", "sk-or-fake")
    monkeypatch.setenv("AEVUM_AI_OPENROUTER_RPM_LIMIT", "20")
    monkeypatch.setenv("AEVUM_AI_MAX_TOKENS", "3000")
    settings.cache_clear()

    config = settings()
    assert config.model_provider == "openrouter"
    assert config.openrouter_rpm_limit == 20
    assert config.max_tokens == 3000
    assert analyst.available() is True
