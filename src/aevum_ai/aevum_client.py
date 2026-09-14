"""Read-only client for the Aevum REST API.

**This service borrows the caller's identity; it never has one of its own.**

Every request carries the bearer token and `X-Organization-Id` that arrived on
the inbound request, so the AI service can only ever read what the person
asking could already read. That is deliberate and load-bearing: a service
account with its own credentials would become a second path into tenant data,
sitting outside `TenantFilter`, and the first bug in it would be a cross-tenant
leak. Borrowing the caller's token means there is no new boundary to get wrong.

It follows that a 403 from the backend is the correct answer to pass straight
back to the client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .config import settings


class UpstreamError(Exception):
    """The Aevum API refused or could not be reached."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Caller:
    """The identity on the inbound request, passed through unchanged."""

    authorization: str
    organization_id: str | None

    def headers(self) -> dict[str, str]:
        headers = {"Authorization": self.authorization, "Accept": "application/json"}
        if self.organization_id:
            headers["X-Organization-Id"] = self.organization_id
        return headers


class AevumClient:
    def __init__(self, caller: Caller, base_url: str | None = None) -> None:
        self._caller = caller
        self._base_url = (base_url or settings().api_base_url).rstrip("/")

    async def _get(self, client: httpx.AsyncClient, path: str) -> Any:
        try:
            response = await client.get(f"{self._base_url}{path}", headers=self._caller.headers())
        except httpx.RequestError as exc:
            raise UpstreamError(502, "UPSTREAM_UNAVAILABLE", "The Aevum API is unreachable.") from exc

        if response.status_code >= 400:
            # Pass the backend's own code through rather than flattening it:
            # the frontend already knows what NOT_A_PARTICIPANT means, and a
            # generic 502 here would turn a clear refusal into a mystery.
            body = _safe_json(response)
            raise UpstreamError(
                response.status_code,
                body.get("code", "UPSTREAM_ERROR"),
                body.get("message", "The Aevum API refused this request."),
            )
        return response.json()

    async def decision_state(self, decision_id: str) -> dict[str, Any]:
        """Everything needed to analyse one decision.

        Ballots come from `/votes/all`, which is participant-only: a tally is a
        group fact, but who voted for what is a personal one. If the caller can
        read the decision without being a participant, that call is refused —
        and the refusal is handled rather than fatal. The analysis then omits
        the role-divergence claim instead of estimating it, because an analysis
        that quietly guesses who voted for what is worse than one that says
        less.
        """
        timeout = httpx.Timeout(settings().request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            decision = await self._get(client, f"/api/decisions/{decision_id}")
            results = await self._get(client, f"/api/decisions/{decision_id}/results")
            participants = await self._get(client, f"/api/decisions/{decision_id}/participants")
            timeline = await self._get(client, f"/api/decisions/{decision_id}/timeline?limit=20")
            try:
                ballots = await self._get(client, f"/api/decisions/{decision_id}/votes/all")
            except UpstreamError as exc:
                # 403: not a participant. 404: an older backend without the
                # endpoint. Neither should cost the caller their whole panel.
                if exc.status not in (403, 404):
                    raise
                ballots = None

        return {
            "decision": decision,
            "tallies": results.get("data", []),
            "participants": participants.get("data", []),
            "timeline": timeline.get("data", []),
            "ballots": ballots.get("data", []) if ballots else None,
        }


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        value = response.json()
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}
