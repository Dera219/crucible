"""Pre-registration.

Most of what follows asserts that something is *impossible*. A protocol that can be talked out
of its own rules is decoration, and both of the holes tested here were found in Apex after the
fact: a journal edited on disk kept its original fingerprint, and the mechanism field could be
satisfied with a slogan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.preregistration import (
    EdgeSource,
    Evidence,
    Hypothesis,
    KillCriteria,
    PreregistrationError,
    ResearchJournal,
    Verdict,
    assess,
)

MECHANISM = (
    "Index funds tracking the benchmark must buy an added constituent on the effective date "
    "regardless of price, because tracking error is their mandate and price is not. That forced "
    "demand is supplied by someone, and the compensation for supplying it is the spread between "
    "the announcement drift and the reversion afterwards."
)


def hypothesis(**overrides: object) -> Hypothesis:
    defaults: dict[str, object] = {
        "name": "index-inclusion-drift",
        "claim": "Beats an equal-weight baseline in a majority of OOS folds, net of costs.",
        "edge_source": EdgeSource.STRUCTURAL_CONSTRAINT,
        "mechanism": MECHANISM,
        "universe": "US common shares, top 1000 by trailing dollar volume",
        "horizon_bars": 5,
        "warmup_bars": 60,
        "parameters": ("lookback", "threshold"),
        "kill": KillCriteria(),
        "trial_budget": 5,
    }
    defaults.update(overrides)
    return Hypothesis(**defaults)  # type: ignore[arg-type]


def passing_evidence(**overrides: object) -> Evidence:
    defaults: dict[str, object] = {
        "ic_mean": 0.03,
        "ic_t_statistic": 4.0,
        "monotonicity": 0.9,
        "oos_fold_win_rate": 0.7,
        "folds_scored": 8,
        "folds_testable": True,
        "annual_turnover": 6.0,
        "survived_deflation": True,
        "trials_used": 3,
        "test_window_bars": 252,
    }
    defaults.update(overrides)
    return Evidence(**defaults)  # type: ignore[arg-type]


class TestMechanismIsRequired:
    def test_a_slogan_does_not_construct(self) -> None:
        with pytest.raises(ValueError, match="who is on the other side"):
            hypothesis(mechanism="Momentum works.")

    def test_whitespace_padding_does_not_satisfy_it(self) -> None:
        with pytest.raises(ValueError, match="characters"):
            hypothesis(mechanism="Momentum works." + " " * 500)

    def test_a_real_mechanism_constructs(self) -> None:
        assert hypothesis().edge_source is EdgeSource.STRUCTURAL_CONSTRAINT


class TestConstructionValidation:
    def test_an_undescribed_universe_is_refused(self) -> None:
        with pytest.raises(ValueError, match="universe must be described"):
            hypothesis(universe="   ")

    def test_more_parameters_than_the_cap_is_caught_at_construction(self) -> None:
        """So the cap cannot be discovered after fitting."""
        with pytest.raises(ValueError, match="parameters declared but kill criteria allow"):
            hypothesis(parameters=("a", "b", "c", "d"), kill=KillCriteria(max_parameters=2))

    def test_a_zero_trial_budget_is_refused(self) -> None:
        with pytest.raises(ValueError, match="trial_budget must be >= 1"):
            hypothesis(trial_budget=0)

    def test_impossible_kill_thresholds_are_refused(self) -> None:
        with pytest.raises(ValueError, match="min_oos_fold_win_rate"):
            KillCriteria(min_oos_fold_win_rate=1.5)
        with pytest.raises(ValueError, match="min_monotonicity"):
            KillCriteria(min_monotonicity=2.0)
        with pytest.raises(ValueError, match="max_annual_turnover"):
            KillCriteria(max_annual_turnover=0.0)


class TestFingerprint:
    def test_identical_hypotheses_share_one(self) -> None:
        assert hypothesis().fingerprint == hypothesis().fingerprint

    def test_loosening_a_criterion_changes_it(self) -> None:
        """The core property: the altered claim is visibly a different claim."""
        strict = hypothesis(kill=KillCriteria(min_ic_t_statistic=3.0))
        loose = hypothesis(kill=KillCriteria(min_ic_t_statistic=1.0))

        assert strict.fingerprint != loose.fingerprint

    def test_extending_the_trial_budget_changes_it(self) -> None:
        assert hypothesis(trial_budget=5).fingerprint != hypothesis(trial_budget=50).fingerprint

    def test_parameter_order_does_not_change_it(self) -> None:
        assert (
            hypothesis(parameters=("a", "b")).fingerprint
            == hypothesis(parameters=("b", "a")).fingerprint
        )

    def test_notes_are_excluded_so_annotation_is_not_tampering(self) -> None:
        assert hypothesis().fingerprint == hypothesis(notes="revisit after WRDS").fingerprint


class TestStructuralInvalidity:
    def test_warmup_exceeding_the_window_is_invalid_not_killed(self) -> None:
        """Silence is not underperformance. Recording it as a rejection means it is never
        revisited even though it was never actually tried."""
        verdict, violations = assess(
            hypothesis(warmup_bars=300), passing_evidence(test_window_bars=252)
        )

        assert verdict is Verdict.INVALID
        assert violations[0].criterion == "warmup_exceeds_window"
        assert "silence, not performance" in violations[0].detail

    def test_a_strategy_that_never_traded_is_invalid(self) -> None:
        verdict, violations = assess(hypothesis(), passing_evidence(folds_testable=False))

        assert verdict is Verdict.INVALID
        assert violations[0].criterion == "never_traded"

    def test_no_folds_is_invalid(self) -> None:
        verdict, violations = assess(hypothesis(), Evidence())

        assert verdict is Verdict.INVALID
        assert violations[0].criterion == "no_folds"

    def test_exceeding_the_trial_budget_is_invalid(self) -> None:
        """The deflation bar was set for the declared budget. Running past it means the result
        has not been corrected for the search that found it."""
        verdict, violations = assess(hypothesis(trial_budget=5), passing_evidence(trials_used=40))

        assert verdict is Verdict.INVALID
        assert violations[0].criterion == "trial_budget_exceeded"
        assert "not been corrected for the search" in violations[0].detail


class TestKillCriteria:
    def test_clean_evidence_survives(self) -> None:
        verdict, violations = assess(hypothesis(), passing_evidence())

        assert verdict is Verdict.SURVIVED
        assert violations == ()

    @pytest.mark.parametrize(
        ("field", "value", "criterion"),
        [
            ("ic_mean", 0.001, "min_ic"),
            ("ic_t_statistic", 1.0, "min_ic_t_statistic"),
            ("monotonicity", 0.1, "min_monotonicity"),
            ("oos_fold_win_rate", 0.2, "min_oos_fold_win_rate"),
            ("annual_turnover", 50.0, "max_annual_turnover"),
            ("survived_deflation", False, "require_deflation_survival"),
        ],
    )
    def test_each_criterion_can_kill(self, field: str, value: object, criterion: str) -> None:
        verdict, violations = assess(hypothesis(), passing_evidence(**{field: value}))

        assert verdict is Verdict.KILLED
        assert any(v.criterion == criterion for v in violations)

    def test_an_unmeasured_criterion_cannot_pass(self) -> None:
        """Absence of evidence is not evidence of passing."""
        verdict, violations = assess(hypothesis(), passing_evidence(ic_mean=None))

        assert verdict is Verdict.KILLED
        assert "was not measured" in violations[0].detail

    def test_a_negative_ic_of_large_magnitude_still_fails_the_min_ic_bar(self) -> None:
        verdict, violations = assess(
            hypothesis(), passing_evidence(ic_mean=-0.08, ic_t_statistic=-5.0)
        )

        assert verdict is Verdict.KILLED
        assert any(v.criterion == "min_ic" for v in violations)

    def test_capacity_is_only_checked_when_required(self) -> None:
        lenient = assess(hypothesis(), passing_evidence(capacity=None))[0]
        strict = assess(
            hypothesis(kill=KillCriteria(min_capacity=1e6)), passing_evidence(capacity=1e3)
        )

        assert lenient is Verdict.SURVIVED
        assert strict[0] is Verdict.KILLED
        assert any(v.criterion == "min_capacity" for v in strict[1])


class TestJournalIntegrity:
    def test_an_edited_registration_is_refused_on_load(self, tmp_path: Path) -> None:
        """The hash only catches edits made through the API unless it is re-verified. A file
        edited directly keeps its original fingerprint while its contents say something else."""
        path = tmp_path / "journal.jsonl"
        ResearchJournal(path).register(hypothesis(kill=KillCriteria(min_ic_t_statistic=3.0)))

        lines = path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        record["payload"]["kill"]["min_ic_t_statistic"] = 0.0  # loosen, keep the hash
        lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(PreregistrationError, match="has been altered"):
            ResearchJournal(path)

    def test_the_refusal_reports_both_hashes(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.jsonl"
        original = hypothesis()
        ResearchJournal(path).register(original)

        lines = path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        record["payload"]["horizon_bars"] = 999
        lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(PreregistrationError) as exc:
            ResearchJournal(path)

        assert original.fingerprint in str(exc.value)
        assert "hash to" in str(exc.value)

    def test_annotating_notes_on_disk_does_not_trip_the_check(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.jsonl"
        ResearchJournal(path).register(hypothesis())

        lines = path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        record["payload"]["notes"] = "revisit after WRDS lands"
        lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        assert len(ResearchJournal(path).registrations()) == 1

    def test_a_corrupt_line_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.jsonl"
        path.write_text("not json\n", encoding="utf-8")

        with pytest.raises(PreregistrationError, match="must not be hand-edited"):
            ResearchJournal(path)


class TestJournalProtocol:
    def test_evidence_for_an_unregistered_claim_is_refused(self, tmp_path: Path) -> None:
        """Loosening a criterion and re-running produces a new fingerprint, which must be
        registered as the new claim it is."""
        journal = ResearchJournal(tmp_path / "journal.jsonl")
        journal.register(hypothesis(kill=KillCriteria(min_ic_t_statistic=3.0)))
        loosened = hypothesis(kill=KillCriteria(min_ic_t_statistic=1.0))

        with pytest.raises(PreregistrationError, match="is not registered"):
            journal.record_trial(loosened.fingerprint, sharpe=0.9)

    def test_registering_twice_is_idempotent(self, tmp_path: Path) -> None:
        journal = ResearchJournal(tmp_path / "journal.jsonl")
        first = journal.register(hypothesis())
        second = journal.register(hypothesis())

        assert first == second
        assert len(journal.registrations()) == 1

    def test_the_journal_survives_a_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.jsonl"
        journal = ResearchJournal(path)
        fingerprint = journal.register(hypothesis())
        journal.record_trial(fingerprint, sharpe=0.1)

        reopened = ResearchJournal(path)

        assert reopened.is_registered(fingerprint)
        assert reopened.trials_for(fingerprint) == 1

    def test_lifetime_trials_span_every_hypothesis(self, tmp_path: Path) -> None:
        """The honest denominator: deflating against one hypothesis's own trials understates the
        search when that hypothesis is itself the ninth thing you tried."""
        journal = ResearchJournal(tmp_path / "journal.jsonl")
        first = journal.register(hypothesis(name="first"))
        second = journal.register(hypothesis(name="second"))
        for _ in range(3):
            journal.record_trial(first, sharpe=0.1)
        for _ in range(4):
            journal.record_trial(second, sharpe=0.2)

        assert journal.trials_for(second) == 4
        assert journal.lifetime_trials() == 7

    def test_the_summary_counts_attempts_and_flags_overruns(self, tmp_path: Path) -> None:
        journal = ResearchJournal(tmp_path / "journal.jsonl")
        fingerprint = journal.register(hypothesis(trial_budget=2))
        for _ in range(5):
            journal.record_trial(fingerprint, sharpe=0.1)
        journal.record_verdict(fingerprint, Verdict.KILLED, ())

        summary = journal.summarize()

        assert "trials 5/2" in summary
        assert "OVER BUDGET" in summary
        assert "killed" in summary

    def test_multiple_hypotheses_prompt_the_honest_denominator(self, tmp_path: Path) -> None:
        journal = ResearchJournal(tmp_path / "journal.jsonl")
        for i in range(3):
            fingerprint = journal.register(hypothesis(name=f"attempt-{i}"))
            journal.record_trial(fingerprint, sharpe=0.1)

        assert "everything you tried and abandoned" in journal.summarize()

    def test_an_empty_journal_summarises_cleanly(self, tmp_path: Path) -> None:
        assert "empty" in ResearchJournal(tmp_path / "journal.jsonl").summarize()
