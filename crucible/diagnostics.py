"""Does this signal predict anything? Ask before drawing an equity curve.

An equity curve is the *last* thing to look at, and almost everyone looks first. It compounds
predictive power, position sizing, cost assumptions, compounding luck, and the particular path
this sample happened to take, into a single line that goes up or down. When it goes up you cannot
tell which of those did it. When it goes down you cannot tell which to fix.

The diagnostics here separate them. They answer, in order:

1. **Is there predictive content at all?** — `information_coefficient`
2. **Over what horizon does it survive?** — `ic_decay`, which sets your rebalance frequency
3. **Is it a signal or just a filter?** — `quantile_spread` and its monotonicity
4. **How fast does it turn over?** — `autocorrelation`
5. **Given the breadth, what Sharpe is even possible?** — `fundamental_law`

A signal that fails these fails cheaply, in milliseconds, before you have built anything around
it. A signal that passes has earned a backtest.

## Why rank correlation rather than plain correlation

Returns have fat tails. A single 40% day in one name dominates a Pearson correlation and can
manufacture a relationship out of one observation — and the observation is usually a data error,
a missed split, or a takeover. Rank correlation asks only whether the ordering was right, which
is the thing a cross-sectional portfolio actually harvests: you hold the top names against the
bottom ones, and by how much the top name won is somebody else's problem.

## What counts as a good number

Realistic IC for a genuinely useful equity signal is **0.02 to 0.05**. That sounds like nothing
and is not: with enough breadth, being right 51.5% of the time across hundreds of simultaneous
bets is a business. An IC of 0.3 on daily equity data is not a discovery, it is a bug — check for
lookahead before celebrating, because `crucible.causality` catches the mechanical kind but not a
target that leaked into your data upstream.

## These functions look forward, deliberately

Every one of them uses returns from after the signal date. That is what evaluation *is*. They
must never be called from inside a signal definition, and they are the reason `forward_returns`
is named for what it does — so its appearance on the right-hand side of a signal is obvious.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from crucible.ops import cs_rank
from crucible.panel import Panel, PanelError, align

__all__ = [
    "ICReport",
    "QuantileReport",
    "autocorrelation",
    "fundamental_law",
    "ic_decay",
    "information_coefficient",
    "quantile_spread",
]

Array = npt.NDArray[np.float64]

#: Below this many investable names a cross-sectional statistic is not measuring anything.
_MIN_BREADTH = 5


def _forward_return(prices: Panel, *, horizon: int, skip: int) -> Array:
    """Return from `t + skip` to `t + skip + horizon`, as a matrix aligned on t.

    `skip` is `execution_lag - 1`, so it is 0 for the standard convention: the signal at *t*
    earns the return from *t* to *t+1*, which is strictly after the signal and therefore causal.
    Getting this off by one does not produce a wrong-looking number — it produces a number near
    zero, and a diagnostic that reports "no edge" for an edge that is really there.
    """
    values = prices.values
    n_times = values.shape[0]
    out = np.full_like(values, np.nan)
    start, end = skip, skip + horizon
    if end < n_times:
        with np.errstate(divide="ignore", invalid="ignore"):
            base = values[start : n_times - horizon]
            future = values[end:]
            out[: n_times - end] = np.where(base == 0, np.nan, (future - base) / base)
    return out


def _rank_rows(values: Array, min_breadth: int) -> Array:
    """Percentile-rank each row, leaving thin rows as NaN."""
    panel = Panel(
        values=values,
        index=tuple(range(values.shape[0])),  # type: ignore[arg-type]
        assets=tuple(str(i) for i in range(values.shape[1])),
    )
    return cs_rank(panel, min_breadth=min_breadth).values


def _row_correlation(left: Array, right: Array, min_breadth: int) -> Array:
    """Pearson correlation per row over cells valid in both. NaN where too thin."""
    n_times = left.shape[0]
    out = np.full(n_times, np.nan)
    for row in range(n_times):
        valid = np.isfinite(left[row]) & np.isfinite(right[row])
        if int(valid.sum()) < min_breadth:
            continue
        x, y = left[row, valid], right[row, valid]
        x_centred, y_centred = x - x.mean(), y - y.mean()
        denominator = np.sqrt((x_centred**2).sum() * (y_centred**2).sum())
        if denominator == 0:
            continue
        out[row] = float((x_centred * y_centred).sum() / denominator)
    return out


@dataclass(frozen=True, slots=True)
class ICReport:
    """Per-period rank correlation between a signal and the return it predicted."""

    per_period: Array
    horizon: int
    execution_lag: int
    breadth: Array

    @property
    def observations(self) -> int:
        return int(np.isfinite(self.per_period).sum())

    @property
    def mean(self) -> float:
        finite = self.per_period[np.isfinite(self.per_period)]
        return float(finite.mean()) if finite.size else 0.0

    @property
    def std(self) -> float:
        finite = self.per_period[np.isfinite(self.per_period)]
        return float(finite.std(ddof=1)) if finite.size > 1 else 0.0

    @property
    def icir(self) -> float:
        """Mean IC over its own standard deviation — the stability of the edge, not its size.

        A signal with IC 0.02 every period is worth far more than one averaging 0.05 by being
        wildly right and wildly wrong, because the second is indistinguishable from luck until
        you have a very long sample.
        """
        return self.mean / self.std if self.std > 0 else 0.0

    @property
    def t_statistic(self) -> float:
        """How confident we are the true IC is non-zero.

        Overstated whenever ICs are autocorrelated, which they are for any signal with a horizon
        longer than the rebalance interval — overlapping windows reuse the same returns. Treat
        this as an upper bound on your confidence, never a p-value to quote.
        """
        if self.observations < 2:
            return 0.0
        if self.std == 0:
            # A constant IC. Zero variance with a zero mean is genuinely nothing; zero variance
            # with a non-zero mean is a deterministic relationship, which is infinitely
            # significant in-sample and almost always a leaked target. Returning 0.0 here — the
            # first version did — reported the most obvious lookahead bug in existence as
            # "no detectable edge", which is the worst false negative this module could produce.
            return 0.0 if self.mean == 0 else float(np.copysign(np.inf, self.mean))
        return self.icir * float(np.sqrt(self.observations))

    @property
    def hit_rate(self) -> float:
        """Fraction of periods where the signal pointed the right way at all."""
        finite = self.per_period[np.isfinite(self.per_period)]
        return float((finite > 0).mean()) if finite.size else 0.0

    def verdict(self) -> str:
        """A plain reading, including the direction nobody expects to need."""
        if self.observations < 20:
            return f"too few periods ({self.observations}) to say anything"
        # Implausibility is checked BEFORE significance. A leaked target produces a constant IC
        # near 1.0, whose zero variance made the t-test say "no edge" — so an edge too good to
        # be true must be caught on its magnitude, not on its significance.
        if abs(self.mean) > 0.15:
            return (
                f"IC of {self.mean:.3f} is implausibly high for daily equity data. Suspect "
                f"lookahead in the DATA rather than the code — a restated fundamental, a "
                f"survivorship-filtered universe, an adjusted price that used a later split, "
                f"or the prediction target itself having leaked into a feature."
            )
        if abs(self.t_statistic) < 2:
            return "no detectable edge — IC is indistinguishable from zero"
        if self.mean < 0:
            return (
                "significantly NEGATIVE IC — the signal predicts the opposite of what it claims. "
                "Check the sign before inverting it: a sign error and a real contrarian effect "
                "look identical here."
            )
        return f"plausible edge: IC {self.mean:.4f}, t {self.t_statistic:.2f}"

    def summary(self) -> str:
        return (
            f"IC(h={self.horizon}, lag={self.execution_lag}) mean {self.mean:+.4f} | "
            f"std {self.std:.4f} | ICIR {self.icir:+.3f} | t {self.t_statistic:+.2f} | "
            f"hit {self.hit_rate:.1%} | {self.observations} periods | "
            f"median breadth {np.nanmedian(self.breadth):.0f}\n  {self.verdict()}"
        )


def information_coefficient(
    signal: Panel,
    prices: Panel,
    *,
    horizon: int = 1,
    execution_lag: int = 1,
    min_breadth: int = _MIN_BREADTH,
) -> ICReport:
    """Rank correlation between the signal and the return it could have earned.

    Args:
        signal: `(T, N)` signal values. Only the ordering within each row matters.
        prices: `(T, N)` adjusted prices.
        horizon: Periods over which the prediction is scored.
        execution_lag: Same meaning as in `crucible.engine.backtest` — periods between the
            signal's timestamp and the start of the return it earns. 1 credits the signal at *t*
            with the return from *t* to *t+1*. Shares the parameter with the engine so an IC and
            a backtest cannot silently disagree about what was being measured.
        min_breadth: Rows with fewer valid pairs than this are not scored.

    Returns:
        An `ICReport`. Read `verdict()` before anything else.
    """
    if horizon < 1:
        raise PanelError(f"horizon must be >= 1, got {horizon}")
    if execution_lag < 1:
        raise PanelError(
            f"execution_lag must be >= 1, got {execution_lag}. Zero would credit the signal with "
            f"the return of the period it was computed from, which scores the past."
        )
    skip = execution_lag - 1

    signal_panel, price_panel = align(signal, prices)
    forward = _forward_return(price_panel, horizon=horizon, skip=skip)

    signal_ranks = _rank_rows(signal_panel.values, min_breadth)
    forward_ranks = _rank_rows(forward, min_breadth)
    per_period = _row_correlation(signal_ranks, forward_ranks, min_breadth)

    both_valid = np.isfinite(signal_panel.values) & np.isfinite(forward)
    return ICReport(
        per_period=per_period,
        horizon=horizon,
        execution_lag=execution_lag,
        breadth=both_valid.sum(axis=1).astype(np.float64),
    )


def ic_decay(
    signal: Panel,
    prices: Panel,
    *,
    horizons: tuple[int, ...] = (1, 2, 5, 10, 20, 60),
    execution_lag: int = 1,
) -> dict[int, ICReport]:
    """How long the prediction survives — and therefore how often you need to trade.

    The most cost-relevant diagnostic in this module. If IC is 0.030 at one day and 0.028 at
    five, you are paying five times the turnover for 7% more signal, and weekly rebalancing is
    strictly better. If IC collapses to zero by day two, the edge is real but lives entirely
    inside the bid-ask spread and belongs to somebody with a colocated server.

    Reading it: divide by the horizon to compare like with like. A signal whose IC rises with
    horizon is usually not a faster signal decaying, it is a slow signal being measured too
    quickly.
    """
    return {
        horizon: information_coefficient(
            signal, prices, horizon=horizon, execution_lag=execution_lag
        )
        for horizon in horizons
    }


@dataclass(frozen=True, slots=True)
class QuantileReport:
    """Mean forward return by signal bucket. The shape matters more than the spread."""

    mean_returns: Array
    counts: Array
    n_quantiles: int
    horizon: int

    @property
    def spread(self) -> float:
        """Top bucket minus bottom. The headline, and the least informative number here."""
        if len(self.mean_returns) < 2:
            return 0.0
        return float(self.mean_returns[-1] - self.mean_returns[0])

    @property
    def monotonicity(self) -> float:
        """Rank correlation between bucket index and mean return, in [-1, 1].

        The number that separates a signal from a filter. A real cross-sectional effect grades
        smoothly across buckets: the second decile beats the third, which beats the fourth.
        A signal where only the extreme bucket moves and the middle is flat is usually picking
        out a handful of outliers, and outliers are exactly what does not repeat out of sample.
        """
        finite = np.isfinite(self.mean_returns)
        if finite.sum() < 3:
            return 0.0
        buckets = np.arange(len(self.mean_returns), dtype=np.float64)[finite]
        returns = self.mean_returns[finite]
        bucket_ranks = buckets.argsort().argsort().astype(np.float64)
        return_ranks = returns.argsort().argsort().astype(np.float64)
        centred_b = bucket_ranks - bucket_ranks.mean()
        centred_r = return_ranks - return_ranks.mean()
        denominator = np.sqrt((centred_b**2).sum() * (centred_r**2).sum())
        return float((centred_b * centred_r).sum() / denominator) if denominator else 0.0

    @property
    def driven_by_one_tail(self) -> bool:
        """Is the spread coming from the extreme buckets rather than being graded across them?

        Compares the inner spread (second-from-each-end) against what a perfectly linear signal
        would produce, which is `(n - 3) / (n - 1)` of the outer spread — 0.5 for five buckets,
        0.78 for ten. Flagging below half the *outer* spread was the first attempt and it fires
        on an exactly linear signal, because that is precisely where a linear signal sits. A
        warning that triggers on the ideal case teaches you to ignore warnings, which is worse
        than not having it.

        The bar here is 60% of the linear expectation: comfortably clear for a graded signal,
        and unmissable for one where the middle is flat and the tails do all the work.
        """
        finite = self.mean_returns[np.isfinite(self.mean_returns)]
        if len(finite) < 4 or self.spread == 0:
            return False
        inner = abs(float(finite[-2] - finite[1]))
        linear_expectation = (len(finite) - 3) / (len(finite) - 1)
        return inner / abs(self.spread) < 0.6 * linear_expectation

    def summary(self) -> str:
        lines = [
            f"{self.n_quantiles} buckets, horizon {self.horizon}: "
            f"spread {self.spread * 10_000:+.1f}bps | monotonicity {self.monotonicity:+.2f}"
        ]
        for i, (value, count) in enumerate(zip(self.mean_returns, self.counts, strict=True)):
            bar = "" if not np.isfinite(value) else "█" * int(abs(value) * 20_000)
            lines.append(f"  Q{i + 1:<2} {value * 10_000:+8.2f}bps  n={count:>6.0f}  {bar}")
        if self.driven_by_one_tail:
            lines.append(
                "  WARNING: the spread is driven by one extreme bucket. This is a screen for "
                "something unusual, not a graded prediction, and it rarely survives a regime "
                "where that thing stops happening."
            )
        if abs(self.monotonicity) < 0.5:
            lines.append(
                "  WARNING: buckets are not monotone. A real cross-sectional effect grades "
                "smoothly; this one does not."
            )
        return "\n".join(lines)


def quantile_spread(
    signal: Panel,
    prices: Panel,
    *,
    n_quantiles: int = 5,
    horizon: int = 1,
    execution_lag: int = 1,
    min_breadth: int = _MIN_BREADTH,
) -> QuantileReport:
    """Bucket assets by signal each period and average their forward returns.

    The standard first look at a cross-sectional signal, and the fastest way to see whether it
    is graded or lumpy. Buckets are formed *within each row*, so this is causal in the same way
    `cs_rank` is — the bucketing never sees another timestamp.
    """
    if n_quantiles < 2:
        raise PanelError(f"n_quantiles must be >= 2, got {n_quantiles}")
    if execution_lag < 1:
        raise PanelError(f"execution_lag must be >= 1, got {execution_lag}")

    signal_panel, price_panel = align(signal, prices)
    forward = _forward_return(price_panel, horizon=horizon, skip=execution_lag - 1)
    ranks = _rank_rows(signal_panel.values, min_breadth)

    sums = np.zeros(n_quantiles)
    counts = np.zeros(n_quantiles)

    for row in range(ranks.shape[0]):
        valid = np.isfinite(ranks[row]) & np.isfinite(forward[row])
        if int(valid.sum()) < min_breadth:
            continue
        # Ranks are already in [0, 1]; clip keeps rank 1.0 inside the last bucket.
        buckets = np.clip((ranks[row, valid] * n_quantiles).astype(int), 0, n_quantiles - 1)
        np.add.at(sums, buckets, forward[row, valid])
        np.add.at(counts, buckets, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)

    return QuantileReport(
        mean_returns=means, counts=counts, n_quantiles=n_quantiles, horizon=horizon
    )


def autocorrelation(signal: Panel, *, lag: int = 1, min_breadth: int = _MIN_BREADTH) -> float:
    """Mean cross-sectional rank correlation between a signal and its own lagged self.

    A proxy for the turnover the signal will force on you before any portfolio construction has
    happened. Near 1.0 means the ordering barely changes and rebalancing is cheap; near 0 means
    you are re-sorting the universe every period and paying for it.

    Read alongside `ic_decay`: a signal that decays slowly *and* has low autocorrelation is
    telling you the ordering churns for reasons unrelated to the prediction, and smoothing it
    with `ts_decay` will cut costs without costing edge.
    """
    if lag < 1:
        raise PanelError(f"lag must be >= 1, got {lag}")
    ranks = _rank_rows(signal.values, min_breadth)
    lagged = np.full_like(ranks, np.nan)
    if lag < ranks.shape[0]:
        lagged[lag:] = ranks[:-lag]
    correlations = _row_correlation(ranks, lagged, min_breadth)
    finite = correlations[np.isfinite(correlations)]
    return float(finite.mean()) if finite.size else 0.0


def fundamental_law(
    ic: float, breadth: float, *, periods_per_year: int = 252, transfer_coefficient: float = 1.0
) -> float:
    """Grinold's approximation: the information ratio a given skill and breadth can support.

        IR ≈ TC · IC · √(breadth × periods per year)

    The reality check worth running before building anything. It is why a person trading five
    names cannot reach a Sharpe of 2 no matter how good their idea is, and why a fund with a
    much *weaker* signal across three thousand names can: skill sets the edge per bet, breadth
    decides how many times you get to collect it.

    `transfer_coefficient` is the fraction of the signal that survives contact with reality —
    position limits, costs, liquidity, the names you cannot actually short. It is well below 1
    in practice, and assuming otherwise is how a paper IR of 2 becomes a live IR of 0.6.

    Returns an annualised information ratio. Treat it as a ceiling, not a forecast.
    """
    if breadth <= 0:
        raise ValueError(f"breadth must be positive, got {breadth}")
    if not 0 < transfer_coefficient <= 1:
        raise ValueError(f"transfer_coefficient must lie in (0, 1], got {transfer_coefficient}")
    return float(transfer_coefficient * ic * np.sqrt(breadth * periods_per_year))


def diagnose(
    signal: Panel,
    prices: Panel,
    *,
    execution_lag: int = 1,
    n_quantiles: int = 5,
    periods_per_year: int = 252,
) -> str:
    """Everything above, in the order that kills a bad signal soonest.

    Run this before a backtest, every time. Most signals die here in under a second, and the
    ones that do not have earned the more expensive question of whether the edge survives costs.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)

        headline = information_coefficient(signal, prices, horizon=1, execution_lag=execution_lag)
        decay = ic_decay(signal, prices, execution_lag=execution_lag)
        buckets = quantile_spread(
            signal, prices, n_quantiles=n_quantiles, horizon=1, execution_lag=execution_lag
        )
        persistence = autocorrelation(signal)
        breadth = float(np.nanmedian(headline.breadth))

    lines = [
        f"── {signal.name} ──",
        headline.summary(),
        "",
        "IC by horizon (divide by horizon to compare like with like):",
    ]
    for horizon, report in decay.items():
        per_period = report.mean / horizon
        lines.append(
            f"  h={horizon:<3} IC {report.mean:+.4f}  per-period {per_period:+.5f}  "
            f"t {report.t_statistic:+.2f}"
        )

    lines += [
        "",
        buckets.summary(),
        "",
        f"signal autocorrelation (lag 1): {persistence:.3f}",
    ]
    if persistence < 0.5 and abs(headline.mean) > 0:
        lines.append("  the ordering churns fast — try ts_decay before concluding costs kill this")

    if breadth > 0:
        ceiling = fundamental_law(abs(headline.mean), breadth, periods_per_year=periods_per_year)
        realistic = fundamental_law(
            abs(headline.mean), breadth, periods_per_year=periods_per_year, transfer_coefficient=0.5
        )
        lines += [
            "",
            f"Fundamental law: with IC {abs(headline.mean):.4f} and breadth {breadth:.0f},",
            f"  ceiling IR {ceiling:.2f} at perfect transfer, {realistic:.2f} at a realistic 0.5.",
            "  A backtest Sharpe materially above the ceiling is a bug, not a discovery.",
        ]

    return "\n".join(lines)
