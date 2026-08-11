"""Is this result better than the best of the attempts you discarded to find it?

A Sharpe ratio computed from a backtest answers the wrong question. It says how well this
strategy did on this sample. The question that decides whether to trade it is whether that
performance is distinguishable from the best result a search of the same size would have produced
from noise alone — and those differ by a lot once the search is more than a handful of variants.

Two corrections, in order.

**Probabilistic Sharpe (Bailey & López de Prado).** A Sharpe estimated from a short sample is
noisy, and returns are skewed and fat-tailed, which makes the naive standard error optimistic.
PSR gives the probability the *true* Sharpe exceeds a benchmark, accounting for sample length,
skew and kurtosis. A Sharpe of 1.5 over sixty observations of fat-tailed returns is much weaker
evidence than the number suggests.

**Deflated Sharpe.** The benchmark is then raised to the expected maximum Sharpe a search of `N`
independent trials would produce by luck. This is the correction almost nobody applies and it is
usually the one that decides the answer: sweeping fifty parameter pairs and reporting the best is
not fifty times more likely to find an edge, it is fifty draws from a distribution whose maximum
drifts upward with every draw.

## The trial count must be honest, and it must include the losers

`TrialLedger` exists because researchers do not count. Every backtest run during a search is a
trial — the discarded ones, the embarrassing ones, the ones abandoned halfway. Feeding only the
survivors reproduces the exact bias the deflation exists to correct, and does so while looking
rigorous.

Learned the hard way in Apex: recording the *same* trial repeatedly also defeats it. The
benchmark scales with the dispersion of the trial Sharpes, so identical values collapse it to
zero. Fifty trials all scoring 0.9 bought no protection whatsoever while the report looked as
careful as any other, so identical trials are now refused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = [
    "SignificanceError",
    "SignificanceReport",
    "TrialLedger",
    "deflated_sharpe",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
]

Array = npt.NDArray[np.float64]

#: Euler-Mascheroni constant, from the expected-maximum-of-N-Gaussians approximation.
_EULER = 0.5772156649015329


class SignificanceError(ValueError):
    """The statistics cannot be computed, or the inputs would produce a misleading number."""


def _clean(returns: Array | list[float]) -> Array:
    values = np.asarray(returns, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        raise SignificanceError(
            f"need at least 2 finite returns, got {finite.size}. A Sharpe from fewer "
            f"observations is not a weak estimate, it is undefined."
        )
    return finite


def sharpe_ratio(returns: Array | list[float]) -> float:
    """Per-period Sharpe. Not annualised — annualising hides the sample size that matters."""
    values = _clean(returns)
    std = float(values.std(ddof=1))
    if std == 0:
        raise SignificanceError(
            "returns have zero variance — Sharpe statistics are undefined. This usually means "
            "the strategy never traded, which is a no-op rather than a risk-free return."
        )
    return float(values.mean() / std)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Inverse normal CDF by bisection. Adequate here and avoids a scipy dependency."""
    if not 0.0 < p < 1.0:
        raise SignificanceError(f"probability must lie in (0, 1), got {p}")
    low, high = -10.0, 10.0
    for _ in range(200):
        middle = (low + high) / 2
        if _normal_cdf(middle) < p:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def probabilistic_sharpe_ratio(
    returns: Array | list[float], *, benchmark_sharpe: float = 0.0
) -> float:
    """Probability the true per-period Sharpe exceeds `benchmark_sharpe`.

    Accounts for sample length, skew and excess kurtosis. Negative skew and fat tails — which
    describe essentially every real return series — *reduce* the probability relative to the
    naive estimate, because they make the sample Sharpe a less reliable estimate of the truth.
    """
    values = _clean(returns)
    n = values.size
    observed = sharpe_ratio(values)

    centred = values - values.mean()
    std = float(values.std(ddof=1))
    skew = float(np.mean(centred**3) / std**3)
    kurtosis = float(np.mean(centred**4) / std**4)

    denominator = math.sqrt(
        max(1e-12, 1.0 - skew * observed + (kurtosis - 1.0) / 4.0 * observed**2)
    )
    statistic = (observed - benchmark_sharpe) * math.sqrt(n - 1) / denominator
    return _normal_cdf(statistic)


def expected_max_sharpe(*, n_trials: int, trials_sharpe_variance: float) -> float:
    """Expected best Sharpe from `n_trials` independent draws of pure noise.

    The bar a search winner has to clear. Grows with both the number of trials and how much they
    differ from one another: a wide search over genuinely different ideas produces a higher lucky
    maximum than a narrow one, which is why the trial *count* alone is not enough to deflate by.
    """
    if n_trials < 1:
        raise SignificanceError(f"n_trials must be >= 1, got {n_trials}")
    if trials_sharpe_variance < 0:
        raise SignificanceError(
            f"trials_sharpe_variance must be non-negative, got {trials_sharpe_variance}"
        )
    if n_trials == 1 or trials_sharpe_variance == 0:
        return 0.0

    spread = math.sqrt(trials_sharpe_variance)
    n = float(n_trials)
    return spread * (
        (1.0 - _EULER) * _normal_ppf(1.0 - 1.0 / n) + _EULER * _normal_ppf(1.0 - 1.0 / (n * math.e))
    )


@dataclass(frozen=True, slots=True)
class SignificanceReport:
    """Everything needed to decide whether a search winner is real."""

    sharpe: float
    n_returns: int
    n_trials: int
    expected_max_sharpe: float
    psr_vs_zero: float
    deflated_psr: float

    @property
    def survives_deflation(self) -> bool:
        """The conventional 95% bar.

        Failing it means the result is indistinguishable from the best of your own discarded
        attempts. That is not the same as the strategy being bad — it means this evidence cannot
        tell the difference, which is a reason to gather more, not to trade.
        """
        return self.deflated_psr >= 0.95

    def summary(self) -> str:
        return (
            f"SR/period {self.sharpe:+.4f} over {self.n_returns} returns | "
            f"P(true SR > 0) = {self.psr_vs_zero:.1%} | "
            f"best of {self.n_trials} lucky trials scores {self.expected_max_sharpe:+.4f} | "
            f"DSR: P(beats luck) = {self.deflated_psr:.1%}"
        )

    def verdict(self) -> str:
        if self.survives_deflation:
            return (
                f"survives deflation at {self.deflated_psr:.1%} — worth walking forward, which "
                f"is a different question from whether it is worth trading"
            )
        return (
            f"does NOT survive deflation ({self.deflated_psr:.1%}) — indistinguishable from the "
            f"best of the {self.n_trials - 1} attempts discarded to find it"
        )


def deflated_sharpe(
    returns: Array | list[float], *, trial_sharpes: list[float]
) -> SignificanceReport:
    """Full significance for a SELECTED strategy, given every trial that competed.

    `trial_sharpes` must contain the per-period Sharpe of every strategy the search evaluated,
    including the discarded ones. Passing only the survivors reproduces the exact bias this
    function exists to correct.
    """
    if not trial_sharpes:
        raise SignificanceError(
            "trial_sharpes is empty — pass every Sharpe the search evaluated, including the "
            "ones you threw away. They are what set the bar."
        )
    if len(trial_sharpes) >= 3 and len(set(trial_sharpes)) == 1:
        raise SignificanceError(
            f"all {len(trial_sharpes)} trials have an identical Sharpe of {trial_sharpes[0]:.6f}. "
            f"The deflation benchmark scales with how much the trials differ, so zero dispersion "
            f"collapses it to zero and the correction stops correcting. Genuinely different "
            f"backtests do not produce bit-identical Sharpes — the same trial has been recorded "
            f"repeatedly, and a fictional trial count is worse than none because it looks "
            f"rigorous."
        )

    n = len(trial_sharpes)
    mean_trial = sum(trial_sharpes) / n
    variance = sum((s - mean_trial) ** 2 for s in trial_sharpes) / n
    benchmark = expected_max_sharpe(n_trials=n, trials_sharpe_variance=variance)

    return SignificanceReport(
        sharpe=sharpe_ratio(returns),
        n_returns=int(_clean(returns).size),
        n_trials=n,
        expected_max_sharpe=benchmark,
        psr_vs_zero=probabilistic_sharpe_ratio(returns),
        deflated_psr=probabilistic_sharpe_ratio(returns, benchmark_sharpe=benchmark),
    )


@dataclass(slots=True)
class TrialLedger:
    """Counts every strategy the search touched, because researchers do not.

    Append-only by design. Forgetting trials is the bias, so there is no method to remove one.
    """

    trial_sharpes: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def record(self, returns: Array | list[float], *, label: str = "") -> float:
        """Register one trial. Returns its Sharpe for convenience.

        A trial whose returns are degenerate — no variance, too few observations — still counts
        toward the trial total. It was an attempt, and the search got to see its result before
        discarding it.
        """
        self.labels.append(label)
        try:
            sharpe = sharpe_ratio(returns)
        except SignificanceError:
            sharpe = 0.0
        self.trial_sharpes.append(sharpe)
        return sharpe

    @property
    def n_trials(self) -> int:
        return len(self.trial_sharpes)

    def best(self) -> tuple[int, float]:
        if not self.trial_sharpes:
            raise SignificanceError("no trials recorded")
        index = int(np.argmax(self.trial_sharpes))
        return index, self.trial_sharpes[index]

    def report(self, winner_returns: Array | list[float]) -> SignificanceReport:
        return deflated_sharpe(winner_returns, trial_sharpes=self.trial_sharpes)
