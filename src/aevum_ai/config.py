"""Configuration, read once from the environment."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AEVUM_AI_", extra="ignore")

    #: Where the Aevum REST API lives. This service is a client of it, never a
    #: peer: it holds no database and no credentials of its own.
    api_base_url: str = "http://localhost:8080"

    #: "nvidia", "anthropic" or "bedrock". Left unset, the service still
    #: answers — with measured numbers and template prose, marked
    #: source="deterministic". Every provider produces identical figures; only
    #: the prose around them changes.
    model_provider: Literal["nvidia", "anthropic", "bedrock", "none"] = "none"

    # ── NVIDIA NIM (OpenAI-compatible) ──────────────────────────────────────
    #: The same endpoint and model the ai-tutor backend runs its Strands agents
    #: on, so one key covers both services.
    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    #: Do not swap this casually.
    #:
    #: Every job in this service asks the agent for STRUCTURED OUTPUT — a typed
    #: narrative, never free text. The ai-tutor backend found that models which
    #: chat perfectly well can still fail Strands' structured output and degrade
    #: silently rather than raising, and that a chain-of-thought model will
    #: spend the entire token budget deliberating and return nothing at all.
    #: A replacement has to be checked against a real panel call, not a prompt.
    nim_model: str = "nvidia/nemotron-3-super-120b-a12b"

    #: NVIDIA's free tier is 40 requests/minute. Enforced by a sliding-window
    #: bucket in nim.py, because the alternative is four panels returning 429
    #: at once, which reads as the product being broken.
    nim_rpm_limit: int = 40

    #: Opening a decision room fires four generations within a second of each
    #: other. Without a cap they queue inside NVIDIA rather than here, where at
    #: least the wait is visible in a log line.
    nim_max_concurrency: int = 4

    # ── Anthropic / Bedrock ─────────────────────────────────────────────────
    anthropic_api_key: str | None = None
    model_id: str = "claude-sonnet-4-6"
    bedrock_region: str = "us-west-2"

    #: Shared across providers. 3000 rather than the original 1200 because a
    #: reasoning model spends tokens before the visible answer, and running out
    #: mid-turn fails the whole request instead of shortening it.
    max_tokens: int = 3000
    temperature: float = 0.2

    port: int = 8090
    #: The Next.js origin. Only needed for direct browser calls; in the normal
    #: setup the frontend reaches this service through its own server proxy.
    cors_origin: str = "http://localhost:3000"

    request_timeout_seconds: float = 20.0


@lru_cache
def settings() -> Settings:
    return Settings()
