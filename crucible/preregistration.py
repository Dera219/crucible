"""Decide what would falsify the idea, before running it.

Everything else in this library makes it cheap to test a hypothesis. That speed is the problem
this module exists for. When a backtest takes an hour, a researcher runs six and remembers all of
them; when it takes two seconds, they run four hundred, remember the good one, and report it as
though it were the first. The arithmetic in `crucible.significance` is unforgiving about that —
against a fifty-trial search, an edge with an annualised Sharpe near 1.27 needs roughly twenty
years of daily data to distinguish itself from luck — so a trial count that drifts upward
silently is not a bookkeeping problem, it is the difference between a provable result and an
unprovable one.

## What is fixed in advance

A `Hypothesis` records the mechanism, the universe, the horizon, the warmup, the parameter count,
the **trial budget**, and the criteria that would kill it. It is hashed. Change any of it — loosen
a threshold, extend the budget, add a parameter — and the fingerprint changes, so the altered
claim is visibly a different claim rather than the original one that happened to survive.

The journal recomputes each registration's fingerprint from its stored payload on load and
refuses on mismatch. Without that check the hash only catches edits made through the API, and a
file edited directly keeps its original fingerprint while its contents say something else — which
is exactly the corruption the hash was supposed to make impossible. Found in Apex the hard way.

## The mechanism field is a forcing function

It requires prose, at length, naming who is on the other side of the trade and why they are
willing to lose money to you. "Momentum works" does not construct. This is deliberately the most
annoying field in the class: in a hypothesis space of thousands of computable signals, the only
cheap filter that works is whether you can explain why the effect should exist at all.

## What this cannot do

It cannot stop you registering a second hypothesis after the first one fails. Nothing can — and a
tool claiming to would just be lying. What it does is make the *count* visible: `summarize()`
reports every hypothesis ever registered against a journal, so a strategy that is the ninth
attempt is recorded as the ninth attempt, and deflation can be run against the honest total
rather than against today's.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "EdgeSource",
    "Evidence",
    "Hypothesis",
    "KillCriteria",
    "PreregistrationError",
    "ResearchJournal",
    "Verdict",
    "Violation",
    "assess",
]

#: Below this the mechanism is a slogan rather than an argument.
_MIN_MECHANISM_CHARACTERS = 120


class PreregistrationError(RuntimeError):
    """The protocol was violated, or the journal cannot be trusted."""


class EdgeSource(StrEnum):
    """Where the money is supposed to come from.

    Four categories, because in practice there are four. A hypothesis that fits none of them is
    almost always a pattern found by searching rather than an effect reasoned about, and the
    distinction is the single cheapest filter available before any data is touched.
    """

    #: You are paid to hold something genuinely unpleasant. Persistent, but you bear the risk.
    RISK_PREMIUM = "risk_premium"
    #: You are paid for taking the other side when somebody needs to transact now and you do not.
    LIQUIDITY_PROVISION = "liquidity_provision"
    #: Someone must trade for non-economic reasons — mandates, index rules, tax dates, redemptions.
    STRUCTURAL_CONSTRAINT = "structural_constraint"
    #: You extract signal from data faster or more thoroughly than the people on the other side.
    INFORMATION_PROCESSING = "information_processing"


class Verdict(StrEnum):
    SURVIVED = "survived"
    KILLED = "killed"
    #: No usable evidence was produced. Distinct from KILLED on purpose: a hypothesis that could
    #: not be tested has not been rejected, and grading it either way invents a result.
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class Violation:
    criterion: str
    detail: str


@dataclass(frozen=True, slots=True)
class KillCriteria:
    """The numbers at which the idea is abandoned. Written before the run, not after.

    Defaults are deliberately demanding. Every one is a number worth choosing on purpose, but
    inheriting a strict bar is survivable and inheriting a lax one is not.
    """

    #: Minimum mean information coefficient. Realistic for a useful equity signal is 0.02-0.05.
    min_ic: float = 0.01
    #: Minimum |t| on the IC. Below 2 the edge is indistinguishable from zero.
    min_ic_t_statistic: float = 2.0
    #: Minimum quantile monotonicity. A signal where only the extreme bucket moves is a screen
    #: for something unusual, not a graded prediction, and it does not survive a regime change.
    min_monotonicity: float = 0.5
    #: Minimum fraction of out-of-sample folds genuinely won — traded, and beat the baseline.
    min_oos_fold_win_rate: float = 0.5
    #: Maximum annualised turnover. An edge that only exists at 300% monthly turnover is not one.
    max_annual_turnover: float = 12.0
    #: Must the result survive deflation against the honest lifetime trial count?
    require_deflation_survival: bool = True
    #: Maximum free parameters. Each one is another axis to overfit along.
    max_parameters: int = 3
    #: Minimum capacity in account currency. An edge with no capacity is a hobby.
    min_capacity: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_oos_fold_win_rate <= 1.0:
            raise ValueError(
                f"min_oos_fold_win_rate must lie in [0, 1], got {self.min_oos_fold_win_rate}"
            )
        if not -1.0 <= self.min_monotonicity <= 1.0:
            raise ValueError(f"min_monotonicity must lie in [-1, 1], got {self.min_monotonicity}")
        if self.min_ic_t_statistic < 0:
            raise ValueError(
                f"min_ic_t_statistic must be non-negative, got {self.min_ic_t_statistic}"
            )
        if self.max_annual_turnover <= 0:
            raise ValueError(
                f"max_annual_turnover must be positive, got {self.max_annual_turnover}"
            )
        if self.max_parameters < 0:
            raise ValueError(f"max_parameters must be non-negative, got {self.max_parameters}")

    def to_json(self) -> dict[str, Any]:
        return {
            "min_ic": self.min_ic,
            "min_ic_t_statistic": self.min_ic_t_statistic,
            "min_monotonicity": self.min_monotonicity,
            "min_oos_fold_win_rate": self.min_oos_fold_win_rate,
            "max_annual_turnover": self.max_annual_turnover,
            "require_deflation_survival": self.require_deflation_survival,
            "max_parameters": self.max_parameters,
            "min_capacity": self.min_capacity,
        }


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A claim, fixed in advance and hashed so that changing it is visible."""

    name: str
    claim: str
    edge_source: EdgeSource
    mechanism: str
    universe: str
    horizon_bars: int
    warmup_bars: int
    parameters: tuple[str, ...]
    kill: KillCriteria
    #: How many backtests you are permitting yourself before deflation eats the result. Choose it
    #: knowing the arithmetic: more trials raise the bar the winner must clear, so a large budget
    #: is not generosity, it is a debt.
    trial_budget: int
    notes: str = ""

    def __post_init__(self) -> None:
        cleaned = " ".join(self.mechanism.split())
        if len(cleaned) < _MIN_MECHANISM_CHARACTERS:
            raise ValueError(
                f"mechanism is {len(cleaned)} characters; at least "
                f"{_MIN_MECHANISM_CHARACTERS} are required.\n"
                f"\n"
                f"State who is on the other side of this trade and why they are willing to lose "
                f"money to you. In a space of thousands of computable signals, whether you can "
                f"explain why an effect should exist is the only cheap filter that works — and "
                f"writing it down before the run is what stops it being reverse-engineered from "
                f"a result afterwards."
            )
        if not self.universe.strip():
            raise ValueError("universe must be described; 'whatever the data had' is not one")
        if self.horizon_bars < 1:
            raise ValueError(f"horizon_bars must be >= 1, got {self.horizon_bars}")
        if self.warmup_bars < 0:
            raise ValueError(f"warmup_bars must be non-negative, got {self.warmup_bars}")
        if self.trial_budget < 1:
            raise ValueError(f"trial_budget must be >= 1, got {self.trial_budget}")
        if len(self.parameters) > self.kill.max_parameters:
            raise ValueError(
                f"{len(self.parameters)} parameters declared but kill criteria allow "
                f"{self.kill.max_parameters}. Caught at construction so the cap cannot be "
                f"discovered after fitting."
            )

    def canonical(self) -> dict[str, Any]:
        """Everything that defines the claim. `notes` is excluded so annotation is not tampering."""
        return {
            "name": self.name,
            "claim": self.claim,
            "edge_source": self.edge_source.value,
            "mechanism": " ".join(self.mechanism.split()),
            "universe": self.universe,
            "horizon_bars": self.horizon_bars,
            "warmup_bars": self.warmup_bars,
            "parameters": sorted(self.parameters),
            "kill": self.kill.to_json(),
            "trial_budget": self.trial_budget,
        }

    @property
    def fingerprint(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Evidence:
    """What the run produced. Every field optional, because a run can fail to produce it."""

    ic_mean: float | None = None
    ic_t_statistic: float | None = None
    monotonicity: float | None = None
    oos_fold_win_rate: float | None = None
    folds_scored: int = 0
    folds_testable: bool = False
    annual_turnover: float | None = None
    survived_deflation: bool | None = None
    trials_used: int = 0
    capacity: float | None = None
    test_window_bars: int = 0


def assess(hypothesis: Hypothesis, evidence: Evidence) -> tuple[Verdict, tuple[Violation, ...]]:
    """Grade evidence against criteria fixed before the run.

    Structural problems are checked first and return INVALID rather than KILLED. The distinction
    is load-bearing: a hypothesis whose warmup exceeded its test window produced silence, not
    underperformance, and recording it as a rejection means it will never be revisited even
    though it was never actually tried.
    """
    structural: list[Violation] = []

    if hypothesis.warmup_bars >= evidence.test_window_bars > 0:
        structural.append(
            Violation(
                "warmup_exceeds_window",
                f"warmup of {hypothesis.warmup_bars} bars does not fit in a "
                f"{evidence.test_window_bars}-bar test window, so the signal could never "
                f"converge before being scored. What was measured is silence, not performance.",
            )
        )
    if evidence.folds_scored == 0:
        structural.append(Violation("no_folds", "no fold produced a scorable result"))
    elif not evidence.folds_testable:
        structural.append(
            Violation(
                "never_traded",
                "no scored fold ever placed a trade. A flat line is not a result, and grading "
                "it as a loss credits the strategy with something it never did.",
            )
        )
    if evidence.trials_used > hypothesis.trial_budget:
        structural.append(
            Violation(
                "trial_budget_exceeded",
                f"{evidence.trials_used} trials used against a declared budget of "
                f"{hypothesis.trial_budget}. The deflation bar was set for the budget; running "
                f"past it means this result has not been corrected for the search that found it.",
            )
        )

    if structural:
        return Verdict.INVALID, tuple(structural)

    kill = hypothesis.kill
    violations: list[Violation] = []

    def check(
        name: str, value: float | None, threshold: float, comparison: str, detail: str
    ) -> None:
        if value is None:
            violations.append(Violation(name, f"{name} was not measured, so it cannot pass"))
            return
        failed = value < threshold if comparison == "min" else value > threshold
        if failed:
            violations.append(Violation(name, detail))

    check(
        "min_ic",
        evidence.ic_mean,
        kill.min_ic,
        "min",
        f"IC {evidence.ic_mean} below the {kill.min_ic} bar",
    )
    check(
        "min_ic_t_statistic",
        abs(evidence.ic_t_statistic) if evidence.ic_t_statistic is not None else None,
        kill.min_ic_t_statistic,
        "min",
        f"IC t-statistic {evidence.ic_t_statistic} below the {kill.min_ic_t_statistic} bar — "
        f"the edge is indistinguishable from zero",
    )
    check(
        "min_monotonicity",
        evidence.monotonicity,
        kill.min_monotonicity,
        "min",
        f"quantile monotonicity {evidence.monotonicity} below {kill.min_monotonicity} — the "
        f"buckets do not grade, so this is a screen rather than a prediction",
    )
    check(
        "min_oos_fold_win_rate",
        evidence.oos_fold_win_rate,
        kill.min_oos_fold_win_rate,
        "min",
        f"won {evidence.oos_fold_win_rate} of out-of-sample folds against a "
        f"{kill.min_oos_fold_win_rate} bar",
    )
    check(
        "max_annual_turnover",
        evidence.annual_turnover,
        kill.max_annual_turnover,
        "max",
        f"annual turnover {evidence.annual_turnover} above {kill.max_annual_turnover}",
    )
    if kill.min_capacity > 0:
        check(
            "min_capacity",
            evidence.capacity,
            kill.min_capacity,
            "min",
            f"capacity {evidence.capacity} below the {kill.min_capacity} minimum — an edge "
            f"with no capacity is a hobby",
        )
    if kill.require_deflation_survival and not evidence.survived_deflation:
        violations.append(
            Violation(
                "require_deflation_survival",
                f"did not survive deflation against {evidence.trials_used} trial(s) — "
                f"indistinguishable from the best of the attempts discarded to find it",
            )
        )

    return (Verdict.KILLED if violations else Verdict.SURVIVED), tuple(violations)


@dataclass(frozen=True, slots=True)
class JournalEntry:
    kind: str
    timestamp: str
    fingerprint: str
    payload: dict[str, Any]


class ResearchJournal:
    """Append-only record of every hypothesis registered and every trial run against it."""

    __slots__ = ("_entries", "path")

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._entries: list[JournalEntry] = []
        self._load()

    @staticmethod
    def _fingerprint_of(payload: dict[str, Any]) -> str:
        canonical = {k: v for k, v in payload.items() if k != "notes"}
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _load(self) -> None:
        if not self.path.exists():
            return
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                entry = JournalEntry(
                    kind=str(raw["kind"]),
                    timestamp=str(raw["timestamp"]),
                    fingerprint=str(raw["fingerprint"]),
                    payload=dict(raw["payload"]),
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise PreregistrationError(
                    f"{self.path}:{number} is not a valid journal record: {exc}. The journal is "
                    f"append-only and must not be hand-edited."
                ) from exc

            # A stored hash that no longer describes its payload is the one corruption this file
            # exists to prevent. Without this check the fingerprint only catches edits made
            # through the API, and a file edited directly keeps its original hash while its
            # contents say something else.
            if entry.kind == "registration":
                actual = self._fingerprint_of(entry.payload)
                if actual != entry.fingerprint:
                    raise PreregistrationError(
                        f"{self.path}:{number} has been altered: filed under fingerprint "
                        f"{entry.fingerprint} but its contents hash to {actual}. The record no "
                        f"longer describes the hypothesis it claims to, and the journal cannot "
                        f"know which version you meant."
                    )
            self._entries.append(entry)

    def _append(self, kind: str, fingerprint: str, payload: dict[str, Any]) -> None:
        entry = JournalEntry(
            kind=kind,
            timestamp=datetime.now(UTC).isoformat(),
            fingerprint=fingerprint,
            payload=payload,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "kind": entry.kind,
                        "timestamp": entry.timestamp,
                        "fingerprint": entry.fingerprint,
                        "payload": entry.payload,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        self._entries.append(entry)

    def register(self, hypothesis: Hypothesis) -> str:
        """Record a hypothesis before running it. Returns its fingerprint."""
        fingerprint = hypothesis.fingerprint
        if self.is_registered(fingerprint):
            return fingerprint
        payload = hypothesis.canonical() | {"notes": hypothesis.notes}
        self._append("registration", fingerprint, payload)
        return fingerprint

    def is_registered(self, fingerprint: str) -> bool:
        return any(e.kind == "registration" and e.fingerprint == fingerprint for e in self._entries)

    def record_trial(self, fingerprint: str, *, sharpe: float, label: str = "") -> None:
        """Record one backtest run in the search.

        Refuses an unregistered fingerprint. That is the whole point: evidence for a claim that
        was never written down cannot be filed, so loosening a criterion and re-running produces
        a new fingerprint that must be registered as the new claim it is.
        """
        if not self.is_registered(fingerprint):
            raise PreregistrationError(
                f"{fingerprint} is not registered. A hypothesis must be recorded before evidence "
                f"is gathered for it — otherwise the criteria can be written once the result is "
                f"known, which is the failure this journal exists to prevent."
            )
        self._append("trial", fingerprint, {"sharpe": sharpe, "label": label})

    def record_verdict(
        self, fingerprint: str, verdict: Verdict, violations: tuple[Violation, ...]
    ) -> None:
        self._append(
            "verdict",
            fingerprint,
            {
                "verdict": verdict.value,
                "violations": [{"criterion": v.criterion, "detail": v.detail} for v in violations],
            },
        )

    def trials_for(self, fingerprint: str) -> int:
        return sum(1 for e in self._entries if e.kind == "trial" and e.fingerprint == fingerprint)

    def lifetime_trials(self) -> int:
        """Every trial in this journal, across every hypothesis.

        The honest denominator. Deflating against one hypothesis's own trials understates the
        search when the hypothesis is itself the ninth thing you tried.
        """
        return sum(1 for e in self._entries if e.kind == "trial")

    def registrations(self) -> list[dict[str, Any]]:
        return [
            e.payload | {"fingerprint": e.fingerprint}
            for e in self._entries
            if e.kind == "registration"
        ]

    def summarize(self) -> str:
        """Every hypothesis ever registered, with its trial count and verdict.

        Read before believing a result. A strategy that is the ninth attempt is recorded as the
        ninth attempt, which is the number deflation should be run against.
        """
        registrations = [e for e in self._entries if e.kind == "registration"]
        if not registrations:
            return f"{self.path}: empty"

        lines = [
            f"{self.path}: {len(registrations)} hypothesis(es), "
            f"{self.lifetime_trials()} lifetime trial(s)"
        ]
        for entry in registrations:
            verdicts = [
                e.payload["verdict"]
                for e in self._entries
                if e.kind == "verdict" and e.fingerprint == entry.fingerprint
            ]
            outcome = verdicts[-1] if verdicts else "no verdict recorded"
            budget = entry.payload.get("trial_budget", "?")
            used = self.trials_for(entry.fingerprint)
            over = "  OVER BUDGET" if isinstance(budget, int) and used > budget else ""
            lines.append(
                f"  {entry.fingerprint}  {entry.payload['name']:<28} "
                f"trials {used}/{budget}  {outcome}{over}"
            )
        if len(registrations) > 1:
            lines.append(
                f"  Deflation should run against {self.lifetime_trials()} trials, not one "
                f"hypothesis's own — the search includes everything you tried and abandoned."
            )
        return "\n".join(lines)
