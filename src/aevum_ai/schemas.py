"""Wire contracts.

These are the shapes the Next.js frontend consumes. Errors deliberately reuse
the Java backend's `ApiError` envelope — the client already parses exactly one
error contract and branches on `code`, and inventing a second one here would
mean the AI panels need their own error handling for no benefit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Source = Literal["model", "deterministic"]
RiskLevel = Literal["LOW", "MODERATE", "HIGH"]
#: Kept small and closed so the UI can map each to an icon, as the design does.
ConsiderationTone = Literal["cost", "risk", "performance", "people", "time"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApiError(BaseModel):
    """Byte-for-byte the backend's error envelope."""

    timestamp: str = Field(default_factory=_now)
    status: int
    error: str
    code: str
    message: str


class OptionStandingOut(BaseModel):
    optionId: str
    label: str
    voteCount: int
    totalWeight: int
    share: float


class MetricsOut(BaseModel):
    """Measured facts. Nothing here came from a language model."""

    standings: list[OptionStandingOut]
    leadingOptionId: str | None
    leadingLabel: str | None
    margin: float
    eligibleVoters: int
    ballotsCast: int
    participation: float
    rolesDiverge: bool
    seniorFavouriteLabel: str | None
    seniorAgreement: float
    seniorVoters: int
    riskLevel: RiskLevel
    riskReasons: list[str]
    consensusLikelihood: float


class ConsensusRisk(BaseModel):
    level: RiskLevel
    title: str
    body: str


class Consideration(BaseModel):
    title: str
    detail: str
    tone: ConsiderationTone


class DecisionSummary(BaseModel):
    """Powers the room's "AI Summary" tab and the voting rail."""

    decisionId: str
    generatedAt: str = Field(default_factory=_now)
    source: Source
    headline: str
    summary: str
    consensusRisk: ConsensusRisk
    considerations: list[Consideration]
    metrics: MetricsOut


class LiveInsight(BaseModel):
    """The dark "LIVE INSIGHT" card in the decision room."""

    decisionId: str
    generatedAt: str = Field(default_factory=_now)
    source: Source
    body: str


class ConsensusReport(BaseModel):
    """The timeline's "INTELLIGENCE INSIGHT" card."""

    decisionId: str
    generatedAt: str = Field(default_factory=_now)
    source: Source
    #: 0-1. A stated heuristic over margin and participation, not a prediction.
    likelihood: float
    likelihoodBasis: str
    optionId: str | None
    optionLabel: str | None
    body: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class AskAnswer(BaseModel):
    """The "Ask AI about this decision" box."""

    decisionId: str
    generatedAt: str = Field(default_factory=_now)
    source: Source
    answer: str
    #: What the answer was grounded on, so a reader can check it.
    groundedOn: list[str]


# ---------------------------------------------------------------- model I/O

class _Narrative(BaseModel):
    """What the agent is asked to produce for a summary.

    Numbers are absent on purpose: every figure the UI shows comes from
    `analysis.py`. The model supplies language, not arithmetic.
    """

    headline: str = Field(description="One short line naming where the decision stands. No numbers.")
    summary: str = Field(description="Two or three sentences of plain prose for a busy reader.")
    riskTitle: str = Field(description="Three or four words naming the main risk, e.g. 'Consensus risk'.")
    riskBody: str = Field(description="One or two sentences explaining the risk in the room's own terms.")
    considerations: list[Consideration] = Field(
        description="Two to four trade-offs the options actually turn on, drawn from their text.",
    )


class _InsightNarrative(BaseModel):
    body: str = Field(description="One or two sentences on what just changed and what it means. No invented numbers.")


class _ReportNarrative(BaseModel):
    body: str = Field(description="Two sentences summarising whether the room is converging, and on what.")


class _Answer(BaseModel):
    answer: str = Field(description="A direct answer grounded only in the supplied decision state.")
