"""The numbers must be right, because the agent is forbidden from producing them.

If `analysis.py` is wrong, every panel is confidently wrong and nothing else in
the system will catch it — the model is explicitly told to trust these figures.
"""

from __future__ import annotations

import pytest

from aevum_ai.analysis import analyse, consensus_likelihood


def decision(status: str = "VOTING", voting: str = "SINGLE", options: list[tuple[str, str]] | None = None):
    options = options or [("opt-a", "Managed Kubernetes"), ("opt-b", "Serverless-First")]
    return {
        "id": "dec-1",
        "title": "Q3 infrastructure",
        "status": status,
        "votingType": voting,
        "options": [{"id": i, "label": label, "description": f"{label} description"} for i, label in options],
    }


def participants(*roles: str):
    return [{"userId": f"u{i}", "role": role} for i, role in enumerate(roles)]


class TestStandings:
    def test_options_with_no_votes_are_still_ranked(self):
        """The API omits them; leaving them out makes an option look nonexistent."""
        metrics = analyse(decision(), [{"optionId": "opt-a", "voteCount": 2, "totalWeight": 2}],
                          participants("MEMBER", "MEMBER"))
        assert [s.option_id for s in metrics.standings] == ["opt-a", "opt-b"]
        assert metrics.standings[1].total_weight == 0
        assert metrics.standings[1].share == 0

    def test_shares_are_percentages_of_total_weight(self):
        metrics = analyse(
            decision(),
            [{"optionId": "opt-a", "voteCount": 3, "totalWeight": 3},
             {"optionId": "opt-b", "voteCount": 1, "totalWeight": 1}],
            participants("MEMBER", "MEMBER", "MEMBER", "MEMBER"),
        )
        assert round(metrics.standings[0].share) == 75
        assert round(metrics.standings[1].share) == 25
        assert metrics.leading.option_id == "opt-a"
        assert round(metrics.margin) == 50

    def test_no_votes_means_no_leader(self):
        metrics = analyse(decision(), [], participants("MEMBER"))
        assert metrics.leading is None
        assert metrics.ballots_cast == 0
        assert metrics.risk_level == "HIGH"


class TestEligibility:
    def test_viewers_are_not_eligible_voters(self):
        """A viewer inflating the denominator would understate participation."""
        metrics = analyse(decision(), [{"optionId": "opt-a", "voteCount": 2, "totalWeight": 2}],
                          participants("LEAD", "MEMBER", "VIEWER", "VIEWER"))
        assert metrics.eligible_voters == 2
        assert metrics.participation == 1.0

    def test_participation_cannot_exceed_one(self):
        metrics = analyse(decision(), [{"optionId": "opt-a", "voteCount": 9, "totalWeight": 9}],
                          participants("MEMBER"))
        assert metrics.participation == 1.0


class TestRoleDivergence:
    def test_divergence_is_detected_from_ballots(self):
        metrics = analyse(
            decision(),
            [{"optionId": "opt-a", "voteCount": 1, "totalWeight": 1},
             {"optionId": "opt-b", "voteCount": 1, "totalWeight": 1}],
            [{"userId": "lead", "role": "LEAD"}, {"userId": "mem", "role": "MEMBER"}],
            ballots=[{"userId": "lead", "optionId": "opt-a", "weight": 1},
                     {"userId": "mem", "optionId": "opt-b", "weight": 1}],
        )
        assert metrics.roles_diverge is True
        assert metrics.senior_favourite.option_id == "opt-a"
        assert metrics.member_favourite.option_id == "opt-b"
        assert metrics.senior_voters == 1

    def test_agreement_is_not_claimed_without_ballots(self):
        """Without per-voter data the claim is unsupported, so it is not made."""
        metrics = analyse(decision(), [{"optionId": "opt-a", "voteCount": 2, "totalWeight": 2}],
                          participants("LEAD", "MEMBER"))
        assert metrics.roles_diverge is False
        assert metrics.senior_favourite is None
        assert metrics.senior_voters == 0


class TestRisk:
    def test_a_close_race_with_low_turnout_is_high_risk(self):
        metrics = analyse(
            decision(),
            [{"optionId": "opt-a", "voteCount": 1, "totalWeight": 1},
             {"optionId": "opt-b", "voteCount": 1, "totalWeight": 1}],
            participants(*(["MEMBER"] * 10)),
        )
        assert metrics.risk_level == "HIGH"
        assert any("points ahead" in r for r in metrics.risk_reasons)
        assert any("eligible participants have voted" in r for r in metrics.risk_reasons)

    def test_a_clear_win_with_full_turnout_is_low_risk(self):
        metrics = analyse(decision(), [{"optionId": "opt-a", "voteCount": 3, "totalWeight": 3}],
                          participants("MEMBER", "MEMBER", "MEMBER"))
        assert metrics.risk_level == "LOW"

    def test_every_risk_level_comes_with_a_stated_reason(self):
        """A level with no reason is an oracle, and nobody can argue with it."""
        metrics = analyse(decision(), [{"optionId": "opt-a", "voteCount": 3, "totalWeight": 3}],
                          participants("MEMBER", "MEMBER", "MEMBER"))
        assert metrics.risk_reasons


class TestLikelihood:
    def test_it_stays_inside_the_stated_bounds(self):
        low = analyse(decision(), [], participants("MEMBER"))
        assert 0.05 <= consensus_likelihood(low) <= 0.95

        runaway = analyse(decision(), [{"optionId": "opt-a", "voteCount": 5, "totalWeight": 5}],
                          participants(*(["MEMBER"] * 5)))
        assert 0.05 <= runaway.consensus_likelihood <= 0.95

    def test_divergence_lowers_it(self):
        base = analyse(
            decision(),
            [{"optionId": "opt-a", "voteCount": 2, "totalWeight": 2},
             {"optionId": "opt-b", "voteCount": 1, "totalWeight": 1}],
            [{"userId": "a", "role": "LEAD"}, {"userId": "b", "role": "MEMBER"}, {"userId": "c", "role": "MEMBER"}],
            ballots=[{"userId": "a", "optionId": "opt-a", "weight": 1},
                     {"userId": "b", "optionId": "opt-a", "weight": 1},
                     {"userId": "c", "optionId": "opt-b", "weight": 1}],
        )
        diverging = analyse(
            decision(),
            [{"optionId": "opt-a", "voteCount": 2, "totalWeight": 2},
             {"optionId": "opt-b", "voteCount": 1, "totalWeight": 1}],
            [{"userId": "a", "role": "LEAD"}, {"userId": "b", "role": "MEMBER"}, {"userId": "c", "role": "MEMBER"}],
            ballots=[{"userId": "a", "optionId": "opt-b", "weight": 1},
                     {"userId": "b", "optionId": "opt-a", "weight": 1},
                     {"userId": "c", "optionId": "opt-a", "weight": 1}],
        )
        assert diverging.consensus_likelihood < base.consensus_likelihood


class TestPromptBlock:
    def test_the_prompt_states_only_measured_facts(self):
        metrics = analyse(
            decision(),
            [{"optionId": "opt-a", "voteCount": 3, "totalWeight": 3},
             {"optionId": "opt-b", "voteCount": 1, "totalWeight": 1}],
            participants("LEAD", "MEMBER", "MEMBER", "MEMBER"),
        )
        block = metrics.summary_for_prompt()
        assert "Managed Kubernetes: 75%" in block
        assert "Eligible voters: 4" in block
        assert "Risk level:" in block
        # The agent is told never to invent a figure; it must therefore be given
        # every figure the panel will show.
        assert "Participation:" in block


@pytest.mark.parametrize("status", ["DRAFT", "OPEN", "VOTING", "REVIEW", "FINALIZED", "ARCHIVED"])
def test_all_six_states_analyse_without_error(status):
    metrics = analyse(decision(status=status), [], participants("MEMBER"))
    assert metrics.status == status


def test_removed_participants_are_not_counted_as_eligible():
    from aevum_ai.analysis import analyse
    decision = {"id": "d", "title": "t", "status": "VOTING", "votingType": "SINGLE", "options": []}
    participants = [
        {"userId": "a", "role": "LEAD", "status": "ACTIVE"},
        {"userId": "b", "role": "MEMBER", "status": "REMOVED"},
        {"userId": "c", "role": "MEMBER"},
    ]
    m = analyse(decision=decision, tallies=[], participants=participants, ballots=[])
    assert m.eligible_voters == 2
