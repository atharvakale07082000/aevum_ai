"""The decision analyst, built on Strands.

One agent, four jobs — summary, live insight, consensus report, and answering a
question. They share a system prompt because they share a single rule, and
stating it once is how it stays true across all four.

**The rule: the agent writes language, never arithmetic.** Every number the UI
displays is computed in `analysis.py` from real votes and real roles, passed in
as text, and the agent is told in the strongest terms not to produce any figure
of its own. A fabricated "84% of architects favour X" inside a decision record
is worse than no analysis: it is durable, it looks official, and people act on
it.

When no model is configured the same functions return template prose over the
same measured numbers, marked `source="deterministic"`. The panels degrade;
they never lie.
"""

from __future__ import annotations

import logging
from typing import Any

from strands import Agent

from ..analysis import DecisionMetrics
from ..config import settings
from ..schemas import (
    AskAnswer,
    ConsensusReport,
    ConsensusRisk,
    Consideration,
    DecisionSummary,
    LiveInsight,
    MetricsOut,
    OptionStandingOut,
    _Answer,
    _InsightNarrative,
    _Narrative,
    _ReportNarrative,
)

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the analyst for Aevum, a platform where teams make decisions on the \
record. You help a group see where their decision actually stands.

Absolute rules, in order of importance:

1. NEVER state a number, percentage, count or probability that was not given to \
   you in the DECISION STATE block. Not as an estimate, not as an illustration, \
   not rounded from something else. If you want to express magnitude, use words \
   — "narrowly ahead", "most of the room", "barely anyone". The interface \
   renders the real figures beside your words; inventing one puts a false \
   number into a permanent record that people will act on.
2. Never invent participants, options, quotes, events or reasons. Everything \
   you say must be traceable to the DECISION STATE block.
3. If the state does not support a claim, say what is missing instead. \
   "Too few people have voted to read anything into this" is a useful answer.

Voice: plain, direct, and concrete. Write for a busy person who has to act. \
Short sentences. No hedging stacks, no consultant register, no "it is important \
to note". Do not congratulate the team or editorialise about the process. Name \
the trade-off, name what would resolve it.

You are describing a live situation, not delivering a verdict. The group \
decides; you make the state of play legible.
"""


def _model():
    """The configured model, or None when the service runs analysis-only."""
    config = settings()
    if config.model_provider == "nvidia":
        if not config.nvidia_api_key:
            # Misconfiguration, not an outage. Saying so and continuing beats
            # 503-ing every panel because one variable is missing — the figures
            # do not need a model and are the part that actually matters.
            log.warning("model_provider=nvidia but no NVIDIA_API_KEY; falling back to deterministic output")
            return None
        from ..nim import get_nim_model

        return get_nim_model()
    if config.model_provider == "anthropic":
        if not config.anthropic_api_key:
            log.warning("model_provider=anthropic but no API key; falling back to deterministic output")
            return None
        from strands.models.anthropic import AnthropicModel

        return AnthropicModel(
            client_args={"api_key": config.anthropic_api_key},
            model_id=config.model_id,
            max_tokens=config.max_tokens,
            params={"temperature": config.temperature},
        )
    if config.model_provider == "bedrock":
        from strands.models import BedrockModel

        return BedrockModel(
            model_id=config.model_id,
            region_name=config.bedrock_region,
            temperature=config.temperature,
        )
    return None


def _agent() -> Agent | None:
    model = _model()
    if model is None:
        return None
    # No tools: the state is assembled by the caller and handed over whole.
    # Giving the agent its own fetch tools would let it wander outside the
    # caller's grant, which is exactly what aevum_client refuses to allow.
    return Agent(model=model, system_prompt=SYSTEM_PROMPT)


def available() -> bool:
    return _model() is not None


def _state_block(metrics: DecisionMetrics, options_detail: str, recent: str) -> str:
    return (
        "DECISION STATE\n"
        "==============\n"
        f"{metrics.summary_for_prompt()}\n\n"
        "OPTIONS AS WRITTEN\n"
        "==================\n"
        f"{options_detail}\n\n"
        "RECENT ACTIVITY\n"
        "===============\n"
        f"{recent or '(nothing recorded yet)'}\n"
    )


def _metrics_out(metrics: DecisionMetrics) -> MetricsOut:
    return MetricsOut(
        standings=[
            OptionStandingOut(
                optionId=s.option_id, label=s.label, voteCount=s.vote_count,
                totalWeight=s.total_weight, share=round(s.share, 1),
            )
            for s in metrics.standings
        ],
        leadingOptionId=metrics.leading.option_id if metrics.leading else None,
        leadingLabel=metrics.leading.label if metrics.leading else None,
        margin=metrics.margin,
        eligibleVoters=metrics.eligible_voters,
        ballotsCast=metrics.ballots_cast,
        participation=metrics.participation,
        rolesDiverge=metrics.roles_diverge,
        seniorFavouriteLabel=metrics.senior_favourite.label if metrics.senior_favourite else None,
        seniorAgreement=metrics.senior_agreement,
        seniorVoters=metrics.senior_voters,
        riskLevel=metrics.risk_level,
        riskReasons=metrics.risk_reasons,
        consensusLikelihood=metrics.consensus_likelihood,
    )


# ------------------------------------------------------------------ summary

async def summarise(metrics: DecisionMetrics, decision: dict[str, Any], recent: str) -> DecisionSummary:
    options_detail = "\n".join(
        f"- {o['label']}: {o.get('description') or '(no description given)'}"
        for o in (decision.get("options") or [])
    ) or "(no options proposed yet)"

    agent = _agent()
    if agent is None:
        return _template_summary(metrics, decision)

    prompt = (
        f"{_state_block(metrics, options_detail, recent)}\n"
        "Write the summary panel for this decision. The considerations must be trade-offs the "
        "options themselves turn on, taken from their descriptions — not generic project advice."
    )
    try:
        result = await agent.invoke_async(prompt, structured_output_model=_Narrative)
        narrative: _Narrative = result.structured_output
    except Exception as exc:  # noqa: BLE001 — a model failure must not fail the panel
        log.warning("summary generation failed, using deterministic output: %s", exc)
        return _template_summary(metrics, decision)

    return DecisionSummary(
        decisionId=metrics.decision_id,
        source="model",
        headline=narrative.headline,
        summary=narrative.summary,
        consensusRisk=ConsensusRisk(
            level=metrics.risk_level, title=narrative.riskTitle, body=narrative.riskBody
        ),
        considerations=narrative.considerations[:4],
        metrics=_metrics_out(metrics),
    )


def _template_summary(metrics: DecisionMetrics, decision: dict[str, Any]) -> DecisionSummary:
    """Measured numbers, template prose. Honest about being unaided."""
    if metrics.ballots_cast == 0:
        headline = "No votes yet"
        summary = (
            f"{len(decision.get('options') or [])} option(s) are on the table and "
            f"{metrics.eligible_voters} participant(s) can vote. Nothing has been cast."
        )
    elif metrics.leading:
        headline = f"{metrics.leading.label} is ahead"
        summary = (
            f"{metrics.leading.label} holds {round(metrics.leading.share)}% of the weight "
            f"from {metrics.ballots_cast} ballot(s), "
            f"{round(metrics.margin)} points clear of the next option. "
            f"{round(metrics.participation * 100)}% of eligible participants have voted."
        )
    else:
        headline = "No clear position"
        summary = "Votes have been cast but no option is ahead."

    considerations = [
        Consideration(
            title=o["label"][:60],
            detail=(o.get("description") or "No description was given for this option."),
            tone="risk",
        )
        for o in (decision.get("options") or [])[:3]
    ]

    return DecisionSummary(
        decisionId=metrics.decision_id,
        source="deterministic",
        headline=headline,
        summary=summary,
        consensusRisk=ConsensusRisk(
            level=metrics.risk_level,
            title="Consensus risk",
            body=" ".join(metrics.risk_reasons),
        ),
        considerations=considerations,
        metrics=_metrics_out(metrics),
    )


# ------------------------------------------------------------- live insight

async def insight(metrics: DecisionMetrics, recent: str) -> LiveInsight:
    agent = _agent()
    if agent is None:
        body = (
            " ".join(metrics.risk_reasons)
            if metrics.risk_reasons
            else "Not enough has happened yet to say anything useful."
        )
        return LiveInsight(decisionId=metrics.decision_id, source="deterministic", body=body)

    prompt = (
        f"{_state_block(metrics, '', recent)}\n"
        "In one or two sentences, tell the room what the current state of play means for them "
        "right now. This sits in a small card that updates as people vote."
    )
    try:
        result = await agent.invoke_async(prompt, structured_output_model=_InsightNarrative)
        return LiveInsight(
            decisionId=metrics.decision_id, source="model", body=result.structured_output.body
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("insight generation failed: %s", exc)
        return LiveInsight(
            decisionId=metrics.decision_id,
            source="deterministic",
            body=" ".join(metrics.risk_reasons) or "Not enough has happened yet to say anything useful.",
        )


# ---------------------------------------------------------- consensus report

LIKELIHOOD_BASIS = (
    "A stated heuristic over the lead margin and turnout, not a prediction: "
    "0.50 baseline, plus margin/200, plus (participation − 0.5)/5, minus 0.15 when "
    "senior and member votes diverge."
)


async def report(metrics: DecisionMetrics, recent: str) -> ConsensusReport:
    agent = _agent()
    option_id = metrics.leading.option_id if metrics.leading else None
    option_label = metrics.leading.label if metrics.leading else None

    if agent is None:
        body = (
            f"{option_label} currently leads." if option_label else "No option leads yet."
        ) + " " + " ".join(metrics.risk_reasons)
        return ConsensusReport(
            decisionId=metrics.decision_id, source="deterministic",
            likelihood=metrics.consensus_likelihood, likelihoodBasis=LIKELIHOOD_BASIS,
            optionId=option_id, optionLabel=option_label, body=body.strip(),
        )

    prompt = (
        f"{_state_block(metrics, '', recent)}\n"
        "Write two sentences for a consensus report on this decision's history page: whether the "
        "room is converging, and on what. Do not state the likelihood figure — the interface "
        "renders it separately."
    )
    try:
        result = await agent.invoke_async(prompt, structured_output_model=_ReportNarrative)
        body = result.structured_output.body
        source = "model"
    except Exception as exc:  # noqa: BLE001
        log.warning("report generation failed: %s", exc)
        body = " ".join(metrics.risk_reasons)
        source = "deterministic"

    return ConsensusReport(
        decisionId=metrics.decision_id, source=source,
        likelihood=metrics.consensus_likelihood, likelihoodBasis=LIKELIHOOD_BASIS,
        optionId=option_id, optionLabel=option_label, body=body,
    )


# ---------------------------------------------------------------------- ask

async def ask(metrics: DecisionMetrics, decision: dict[str, Any], recent: str, question: str) -> AskAnswer:
    options_detail = "\n".join(
        f"- {o['label']}: {o.get('description') or '(no description given)'}"
        for o in (decision.get("options") or [])
    ) or "(no options proposed yet)"

    grounded = [
        f"{len(decision.get('options') or [])} option(s) on this decision",
        f"{metrics.ballots_cast} ballot(s) from {metrics.eligible_voters} eligible participant(s)",
        f"status {metrics.status}",
    ]

    agent = _agent()
    if agent is None:
        return AskAnswer(
            decisionId=metrics.decision_id,
            source="deterministic",
            answer=(
                "Answering questions needs a language model, and none is configured for this "
                "deployment. The panel above still shows the measured state of the decision."
            ),
            groundedOn=grounded,
        )

    prompt = (
        f"{_state_block(metrics, options_detail, recent)}\n"
        f"A participant asks: {question!r}\n\n"
        "Answer using only the state above. If it does not contain the answer, say so plainly "
        "and name what would be needed."
    )
    try:
        result = await agent.invoke_async(prompt, structured_output_model=_Answer)
        return AskAnswer(
            decisionId=metrics.decision_id, source="model",
            answer=result.structured_output.answer, groundedOn=grounded,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("ask failed: %s", exc)
        return AskAnswer(
            decisionId=metrics.decision_id, source="deterministic",
            answer="That question could not be answered right now. Try again in a moment.",
            groundedOn=grounded,
        )
