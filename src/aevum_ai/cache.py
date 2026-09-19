"""A short-lived, in-process cache of analyses, keyed on the decision version.

Opening a decision room fires four panels at once, and each used to re-read the
whole decision (five REST calls) and re-run the agent. Nothing about the answer
changes until the decision does, so the result is reused for the life of the
decision *version*.

What "one upstream read" means here
-----------------------------------
A repeat call is not served blind. Every call first makes one cheap, authorised
freshness probe **as the caller** (`AevumClient.decision_version`: the decision
itself plus the head of its timeline). That probe is what enforces access — a
caller who cannot read the decision gets the backend's 403/404 even when the
cache is warm — and it yields the version the key is built on. What a warm call
skips is the full multi-request state load and the agent run.

The key
-------
    (decision id, caller, version)              -> loaded state, shared by panels
    (decision id, caller, version, job[, q])    -> one panel's result

- *version* changes the moment a vote, option, participant, role change or
  lifecycle transition lands, so invalidation is immediate rather than timed.
- *caller* (a hash of the bearer token and organisation, never the token
  itself) is part of the key because the analysis depends on who asks: ballots
  come from a participant-only endpoint, so a participant's analysis carries a
  role-divergence claim that a workspace member without that grant must not
  receive. Four panels from one person still share everything.
- *q* for `ask` is the question, whitespace-collapsed and case-folded.

In-flight work is cached too, so concurrent identical requests coalesce into a
single state load and a single agent run. Failures are never stored, and a
result the analyst produced by falling back after a model error is not stored
either, so a transient outage is not pinned for the TTL.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass
from typing import Any, TypeVar

from .aevum_client import AevumClient, Caller
from .agents import analyst
from .config import settings

T = TypeVar("T")

Loader = Callable[[str, Caller], Awaitable[Any]]


@dataclass
class _Entry:
    task: asyncio.Task
    expires_at: float | None = None  # None while in flight


@dataclass
class Stats:
    state_loads: int = 0
    agent_runs: int = 0
    hits: int = 0
    coalesced: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "stateLoads": self.state_loads,
            "agentRuns": self.agent_runs,
            "hits": self.hits,
            "coalesced": self.coalesced,
        }


class AnalysisCache:
    """TTL + LRU, with single-flight for in-progress computations."""

    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max(1, max_entries)
        self._clock = clock
        self._entries: OrderedDict[Hashable, _Entry] = OrderedDict()
        self.stats = Stats()

    def __len__(self) -> int:
        return len(self._entries)

    async def get_or_compute(
        self,
        key: Hashable,
        compute: Callable[[], Awaitable[T]],
        cacheable: Callable[[T], bool] = lambda _: True,
    ) -> T:
        if self.ttl <= 0:
            return await compute()

        entry = self._entries.get(key)
        if entry is not None:
            if entry.expires_at is None:
                self.stats.coalesced += 1
                self._entries.move_to_end(key)
                return await asyncio.shield(entry.task)
            if entry.expires_at > self._clock():
                self.stats.hits += 1
                self._entries.move_to_end(key)
                return entry.task.result()
            del self._entries[key]

        # The work runs in its own task and waiters are shielded from it, so a
        # client that disconnects does not cancel the load for the others
        # sharing it.
        task = asyncio.ensure_future(compute())
        entry = _Entry(task=task)
        self._entries[key] = entry
        self._evict()
        task.add_done_callback(lambda t: self._settle(key, entry, t, cacheable))
        return await asyncio.shield(task)

    def _settle(self, key: Hashable, entry: _Entry, task: asyncio.Task, cacheable: Callable[[Any], bool]) -> None:
        if self._entries.get(key) is not entry:
            return  # evicted or replaced while in flight
        ok = not task.cancelled() and task.exception() is None
        if ok:
            try:
                ok = bool(cacheable(task.result()))
            except Exception:  # noqa: BLE001 — a bad predicate must not fail the call
                ok = False
        if ok:
            entry.expires_at = self._clock() + self.ttl
        else:
            del self._entries[key]

    def _evict(self) -> None:
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


_cache: AnalysisCache | None = None


def get_cache() -> AnalysisCache:
    global _cache
    if _cache is None:
        config = settings()
        _cache = AnalysisCache(config.cache_ttl_seconds, config.cache_max_entries)
    return _cache


def reset(
    ttl_seconds: float | None = None,
    max_entries: int | None = None,
    clock: Callable[[], float] | None = None,
) -> None:
    """Drop everything. With arguments, rebuild with those settings (tests)."""
    global _cache
    if ttl_seconds is None and max_entries is None and clock is None:
        _cache = None
        return
    config = settings()
    _cache = AnalysisCache(
        config.cache_ttl_seconds if ttl_seconds is None else ttl_seconds,
        config.cache_max_entries if max_entries is None else max_entries,
        clock or time.monotonic,
    )


def _caller_key(caller: Caller) -> str:
    raw = f"{caller.authorization}\0{caller.organization_id or ''}".encode()
    return hashlib.sha256(raw).hexdigest()


def normalise_question(question: str) -> str:
    return " ".join(question.split()).casefold()


def _keep(result: Any) -> bool:
    """A fallback written after a model failure is an answer, not *the* answer."""
    return not (getattr(result, "source", None) == "deterministic" and analyst.available())


async def cached_state(decision_id: str, caller: Caller, loader: Loader, version: str | None = None) -> Any:
    """The loaded state for this decision version, shared by every panel.

    Probes first (as the caller) unless the caller already did.
    """
    cache = get_cache()
    if version is None:
        version = await AevumClient(caller).decision_version(decision_id)

    async def load() -> Any:
        cache.stats.state_loads += 1
        return await loader(decision_id, caller)

    return await cache.get_or_compute(("state", decision_id, _caller_key(caller), version), load)


async def panel(
    decision_id: str,
    caller: Caller,
    job: str,
    loader: Loader,
    run: Callable[[Any], Awaitable[T]],
    question: str | None = None,
) -> T:
    """One panel's result, computed at most once per decision version.

    `loader` is the app's `_load`; `run` receives what it returns and calls the
    agent. The probe always runs, so access is re-checked on every call.
    """
    cache = get_cache()
    if cache.ttl <= 0:
        # Disabled: behave exactly as before, with no extra probe.
        return await run(await loader(decision_id, caller))
    version = await AevumClient(caller).decision_version(decision_id)
    key: tuple[Hashable, ...] = ("panel", decision_id, _caller_key(caller), version, job)
    if question is not None:
        key += (normalise_question(question),)

    async def compute() -> T:
        state = await cached_state(decision_id, caller, loader, version=version)
        cache.stats.agent_runs += 1
        return await run(state)

    return await cache.get_or_compute(key, compute, cacheable=_keep)
