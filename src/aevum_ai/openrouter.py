"""OpenRouter model factory — the single place a Strands model is built.

``OpenRouterModel`` subclasses the Strands ``OpenAIModel`` pointed at
OpenRouter's OpenAI-compatible endpoint and adds the two throttles a free-tier
key needs:

- a **concurrency cap**, because a decision room opens four panels at once and
  each one is a separate generation;
- an **RPM token bucket** sized to OpenRouter's free tier, because exceeding it
  returns 429s that surface as four simultaneously broken panels.

**Both entry points are throttled, not just ``stream``.** Every job in this
service goes through ``structured_output`` — the analyst is asked for a typed
narrative, never free text — so wrapping only ``stream`` would have left every
real call unthrottled while looking correct.

**The primitives are created inside the running loop, not at import.** Strands'
synchronous ``Agent.__call__`` runs each invocation in a **fresh event loop on a
new thread**, and an ``asyncio.Lock`` awaited from two different loops raises
"got Future attached to a different loop". This service therefore calls
``invoke_async`` from its async endpoints — one loop, no thread bridge — and the
lazy construction below is the belt to that braces.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from functools import lru_cache
from typing import Any, AsyncGenerator

from strands.models.openai import OpenAIModel

from .config import settings

log = logging.getLogger("aevum_ai.openrouter")


class _RpmBucket:
    """Sliding-window limiter: at most ``limit`` acquisitions per 60 seconds.

    A sliding window rather than a fixed one because the failure it prevents is
    bursty: four panels fire on room open, and a fixed window happily allows the
    whole budget at :59 and the whole budget again at :01.
    """

    def __init__(self, limit: int) -> None:
        self.limit = max(1, limit)
        self._stamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._stamps and now - self._stamps[0] > 60.0:
                    self._stamps.popleft()
                if len(self._stamps) < self.limit:
                    self._stamps.append(now)
                    return
                wait = 60.0 - (now - self._stamps[0]) + 0.05
            # Worth a warning, not a debug line: from the reader's side this is
            # a panel that sits on a skeleton for a minute with no explanation,
            # and the log is the only place that says why.
            log.warning("OpenRouter rate limit reached; holding this call for %.1fs", wait)
            await asyncio.sleep(wait)


class _Throttle:
    """The RPM bucket and the concurrency cap, built on first use in this loop."""

    def __init__(self) -> None:
        self._bucket: _RpmBucket | None = None
        self._slots: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _ensure(self) -> tuple[_RpmBucket, asyncio.Semaphore]:
        loop = asyncio.get_running_loop()
        # Rebuild if we have somehow landed on a different loop. Throttling
        # per-loop is imperfect, but it is strictly better than raising
        # "Future attached to a different loop" in the middle of a panel.
        if self._loop is not loop or self._bucket is None or self._slots is None:
            config = settings()
            self._bucket = _RpmBucket(config.openrouter_rpm_limit)
            self._slots = asyncio.Semaphore(max(1, config.openrouter_max_concurrency))
            self._loop = loop
        return self._bucket, self._slots

    async def __aenter__(self) -> None:
        bucket, slots = self._ensure()
        await bucket.acquire()
        await slots.acquire()

    async def __aexit__(self, *_: Any) -> None:
        if self._slots is not None:
            self._slots.release()


_throttle = _Throttle()


class OpenRouterModel(OpenAIModel):
    """``OpenAIModel`` against OpenRouter, rate-limited on both entry points."""

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator:
        async with _throttle:
            async for event in super().stream(*args, **kwargs):
                yield event

    async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator:
        # This is the one that matters here: every analysis job asks for a typed
        # narrative, so this is the path essentially all real traffic takes.
        async with _throttle:
            async for event in super().structured_output(*args, **kwargs):
                yield event


@lru_cache(maxsize=2)
def get_openrouter_model() -> OpenRouterModel:
    """The cached OpenRouter model. Cached because building one opens an HTTP client."""
    config = settings()
    return OpenRouterModel(
        client_args={
            "api_key": config.openrouter_api_key,
            "base_url": config.openrouter_base_url,
        },
        model_id=config.openrouter_model,
        params={
            "temperature": config.temperature,
            # The budget has to cover deliberation as well as the answer: a
            # 1200 cap on a reasoning model hit MaxTokensReached mid-turn and
            # failed the whole request rather than returning a shorter answer.
            "max_tokens": config.max_tokens,
            # OpenRouter's unified reasoning switch. Left on, the model spends
            # budget on a thinking block that never reaches the reader, and on
            # a structured-output call it can crowd out the JSON entirely.
            "extra_body": {"reasoning": {"enabled": False}},
        },
    )
