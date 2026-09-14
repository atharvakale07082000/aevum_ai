"""Deterministic analysis of a decision's state.

Everything in this module is arithmetic over data the Aevum API returned. No
model is involved, and that separation is the point.

The designs call for lines like "80% of senior architects favor Hybrid Cloud"
and "84% probability of consensus". A number like that is exactly the kind of
claim a language model will produce fluently and wrongly, and a decision record
that quietly contains an invented figure is worse than one with no analysis at
all — people act on it, and the audit trail makes it look official.

So the split is strict:

  * this module computes every number, from real votes and real roles;
  * the agent is handed those numbers and writes the prose around them;
  * the system prompt forbids inventing any figure that is not passed in.

If the model is unavailable, the numbers are still right and the service falls
back to template prose. Degraded, not wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

# Decision roles, least to most privileged — mirrors DecisionRole.java.
DECISION_ROLES = ["VIEWER", "MEMBER", "MODERATOR", "LEAD"]

RiskLevel = Literal["LOW", "MODERATE", "HIGH"]


def _role_at_least(role: str | None, required: str) -> bool:
    if role is None or role not in DECISION_ROLES:
        return False
    return DECISION_ROLES.index(role) >= DECISION_ROLES.index(required)


@dataclass
class OptionStanding:
    option_id: str
    label: str
    vote_count: int
    total_weight: int
    share: float  # percent of total weight, 0-100

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionMetrics:
    """Facts about where a decision stands. Every field is measured."""

    decision_id: str
    title: str
    status: str
    voting_type: str

    standings: list[OptionStanding] = field(default_factory=list)
    leading: OptionStanding | None = None
    runner_up: OptionStanding | None = None

    #: Percentage points between the leading option and the next.
    margin: float = 0.0
    #: Participants entitled to a ballot (MEMBER and above).
    eligible_voters: int = 0
    #: Distinct participants who have cast one.
    ballots_cast: int = 0
    #: ballots_cast / eligible_voters, 0-1.
    participation: float = 0.0

    #: Set when senior participants favour a different option from the rest.
    senior_favourite: OptionStanding | None = None
    member_favourite: OptionStanding | None = None
    roles_diverge: bool = False
    #: Share of senior (MODERATOR/LEAD) ballots behind their favourite, 0-100.
    senior_agreement: float = 0.0
    senior_voters: int = 0

    risk_level: RiskLevel = "MODERATE"
    #: Plain sentences naming what drove the risk level. Shown to the user and
    #: given to the agent, so the reasoning is never a black box.
    risk_reasons: list[str] = field(default_factory=list)

    #: Heuristic, NOT a model output. See `consensus_likelihood`.
    consensus_likelihood: float = 0.5

    def summary_for_prompt(self) -> str:
        """The facts, flattened for the agent. Deliberately terse and literal."""
        lines = [
            f"Decision: {self.title}",
            f"Status: {self.status}",
            f"Voting method: {self.voting_type}",
            f"Eligible voters: {self.eligible_voters}",
            f"Ballots cast: {self.ballots_cast}",
            f"Participation: {round(self.participation * 100)}%",
            "Standings (share of total weight):",
        ]
        for s in self.standings:
            lines.append(
                f"  - {s.label}: {round(s.share)}% "
                f"({s.vote_count} vote{'s' if s.vote_count != 1 else ''}, weight {s.total_weight})"
            )
        if self.leading:
            lines.append(f"Leading: {self.leading.label} by {round(self.margin)} points")
        if self.roles_diverge and self.senior_favourite and self.member_favourite:
            lines.append(
                f"Role divergence: senior participants (moderators and leads) favour "
                f"{self.senior_favourite.label} while other members favour "
                f"{self.member_favourite.label}"
            )
        if self.senior_voters:
            lines.append(
                f"Senior agreement: {round(self.senior_agreement)}% of {self.senior_voters} "
                f"senior ballot(s) are behind their favourite"
            )
        lines.append(f"Risk level: {self.risk_level}")
        for reason in self.risk_reasons:
            lines.append(f"  reason: {reason}")
        return "\n".join(lines)


def _standing(option: dict[str, Any], tally: dict[str, Any] | None, total_weight: int) -> OptionStanding:
    weight = int(tally["totalWeight"]) if tally else 0
    return OptionStanding(
        option_id=option["id"],
        label=option["label"],
        vote_count=int(tally["voteCount"]) if tally else 0,
        total_weight=weight,
        share=(weight / total_weight * 100) if total_weight else 0.0,
    )


def analyse(
    decision: dict[str, Any],
    tallies: list[dict[str, Any]],
    participants: list[dict[str, Any]],
    ballots: list[dict[str, Any]] | None = None,
) -> DecisionMetrics:
    """Reduce a decision's live state to measured facts.

    `ballots` is optional: per-voter ballots let us say something about *who*
    is behind an option, which is what the design's "lead votes are heavily
    favouring X" line actually claims. Without them the role-divergence fields
    stay empty rather than being guessed at.
    """
    options = decision.get("options") or []
    by_option = {t["optionId"]: t for t in tallies}
    total_weight = sum(int(t["totalWeight"]) for t in tallies)

    # Zero-fill: the API omits options nobody has voted for, and an option
    # missing from the analysis reads as an option that does not exist.
    standings = [_standing(o, by_option.get(o["id"]), total_weight) for o in options]
    standings.sort(key=lambda s: (-s.total_weight, s.label))

    leading = standings[0] if standings and standings[0].total_weight > 0 else None
    runner_up = standings[1] if len(standings) > 1 and leading else None
    margin = (leading.share - runner_up.share) if (leading and runner_up) else (100.0 if leading else 0.0)

    eligible = [p for p in participants if _role_at_least(p.get("role"), "MEMBER")]
    voters = {b["userId"] for b in (ballots or [])}
    ballots_cast = len(voters) if ballots is not None else sum(s.vote_count for s in standings)
    participation = (ballots_cast / len(eligible)) if eligible else 0.0

    metrics = DecisionMetrics(
        decision_id=decision["id"],
        title=decision["title"],
        status=decision["status"],
        voting_type=decision["votingType"],
        standings=standings,
        leading=leading,
        runner_up=runner_up,
        margin=round(margin, 1),
        eligible_voters=len(eligible),
        ballots_cast=ballots_cast,
        participation=round(min(participation, 1.0), 3),
    )

    _apply_role_split(metrics, participants, ballots or [], standings)
    _apply_risk(metrics)
    metrics.consensus_likelihood = consensus_likelihood(metrics)
    return metrics


def _apply_role_split(
    metrics: DecisionMetrics,
    participants: list[dict[str, Any]],
    ballots: list[dict[str, Any]],
    standings: list[OptionStanding],
) -> None:
    """Does seniority pull the room in a different direction?

    This is the one claim in the designs that is genuinely interesting and
    genuinely checkable, so it is computed rather than narrated.
    """
    if not ballots:
        return

    role_by_user = {p["userId"]: p.get("role") for p in participants}
    by_id = {s.option_id: s for s in standings}

    senior: dict[str, int] = {}
    other: dict[str, int] = {}
    for ballot in ballots:
        bucket = senior if _role_at_least(role_by_user.get(ballot["userId"]), "MODERATOR") else other
        bucket[ballot["optionId"]] = bucket.get(ballot["optionId"], 0) + int(ballot.get("weight", 1))

    metrics.senior_voters = sum(
        1 for b in ballots if _role_at_least(role_by_user.get(b["userId"]), "MODERATOR")
    )

    if senior:
        top_id = max(senior, key=lambda k: senior[k])
        metrics.senior_favourite = by_id.get(top_id)
        total = sum(senior.values())
        metrics.senior_agreement = round(senior[top_id] / total * 100, 1) if total else 0.0
    if other:
        top_id = max(other, key=lambda k: other[k])
        metrics.member_favourite = by_id.get(top_id)

    metrics.roles_diverge = bool(
        metrics.senior_favourite
        and metrics.member_favourite
        and metrics.senior_favourite.option_id != metrics.member_favourite.option_id
    )


def _apply_risk(metrics: DecisionMetrics) -> None:
    """Risk that the room is not actually converging.

    Thresholds are stated rather than tuned, so anyone reading a result can
    check the judgement rather than take it on faith.
    """
    reasons: list[str] = []
    score = 0

    if metrics.ballots_cast == 0:
        metrics.risk_level = "HIGH"
        metrics.risk_reasons = ["Nobody has voted yet."]
        return

    if metrics.margin < 10:
        score += 2
        reasons.append(
            f"The leading option is only {round(metrics.margin)} points ahead — close enough that "
            "one more vote could change it."
        )
    elif metrics.margin < 25:
        score += 1
        reasons.append(f"The lead is {round(metrics.margin)} points, which is not yet decisive.")

    if metrics.participation < 0.5:
        score += 2
        reasons.append(
            f"Only {round(metrics.participation * 100)}% of eligible participants have voted, so the "
            "current standing may not reflect the room."
        )
    elif metrics.participation < 0.8:
        score += 1
        reasons.append(f"Participation is {round(metrics.participation * 100)}% — some voices are missing.")

    if metrics.roles_diverge:
        score += 2
        reasons.append(
            "Senior participants and other members currently favour different options, which usually "
            "means the trade-off has not been talked through."
        )

    if len([s for s in metrics.standings if s.total_weight > 0]) >= 3 and metrics.margin < 20:
        score += 1
        reasons.append("Support is spread across three or more options.")

    metrics.risk_level = "HIGH" if score >= 4 else "MODERATE" if score >= 2 else "LOW"
    if not reasons:
        reasons.append(
            f"{round(metrics.leading.share) if metrics.leading else 0}% of the weight is behind one option "
            f"with {round(metrics.participation * 100)}% participation."
        )
    metrics.risk_reasons = reasons


def consensus_likelihood(metrics: DecisionMetrics) -> float:
    """A stated heuristic, not a prediction.

    The design shows "84% probability of consensus". There is no model behind
    such a number and pretending otherwise would be dishonest, so this is an
    explicit formula that anyone can check:

        0.50  baseline
      + margin / 200          a 100-point lead adds 0.50
      + (participation - 0.5) / 5   full turnout adds 0.10, none removes 0.10
      - 0.15 when roles diverge

    clamped to [0.05, 0.95]. It is surfaced to the UI alongside the word
    "estimate" and the inputs that produced it.
    """
    value = 0.5 + (metrics.margin / 200) + ((metrics.participation - 0.5) / 5)
    if metrics.roles_diverge:
        value -= 0.15
    return round(max(0.05, min(0.95, value)), 2)
