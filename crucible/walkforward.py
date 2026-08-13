"""Out-of-sample evaluation, with the warmup bug that broke Apex designed out.

Apex's walk-forward splits a series into train and test windows and backtests on the test window
alone. That is correct in principle and it has a defect that only appears when the strategy is
slow: a 150-bar indicator inside a 100-bar test fold never warms up, never emits a signal, never
trades, and returns exactly 0.00% — which then *beats* the baseline in every losing fold by not
participating. Apex's own honesty baseline hit it. The sweep winner on SPY survived deflation at
96.6%, was walked forward as the protocol demands, and scored 0.00% in all nine folds.

The fault is not the strategy's and not the fold size's. It is that a test window and a *scoring*
window were treated as the same thing.

## Warmup comes from the train window

Each fold here carries three boundaries rather than two:

    [........ train ........][ warmup ][........ test ........]
                             └─ signal computed from here ─────┘
                                       └─ but only scored here ┘

The signal is computed over `warmup + test`, so indicators arrive at the first scored bar already
converged. Only the test portion is scored. The warmup bars are drawn from data strictly before
the test window — which is exactly what a live system has on its first day — so nothing here
leaks.

`Fold.usable` is False when the requested warmup is not available, and `walk_forward` reports
those folds as **skipped** rather than scoring them at zero. A fold that could not run is not a
fold the strategy lost.

## Non-overlapping test windows

`step` defaults to `test_size`, so test windows tile rather than overlap. Overlapping windows
reuse the same bars across folds and make the out-of-sample sample size look larger than it is —
which inflates every t-statistic computed from the folds, exactly when you most want it honest.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from crucible.costs import CostModel
from crucible.engine import BacktestResult, backtest
from crucible.panel import Panel, PanelError

__all__ = [
    "Fold",
    "FoldOutcome",
    "WalkForwardResult",
    "expanding_folds",
    "rolling_folds",
    "walk_forward",
]

Array = npt.NDArray[np.float64]
SignalFunction = Callable[[Panel], Panel]


@dataclass(frozen=True, slots=True)
class Fold:
    """One train/test split, with the warmup span the test window needs to be scorable."""

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    warmup: int

    def __post_init__(self) -> None:
        if not self.train_start < self.train_end <= self.test_start < self.test_end:
            raise PanelError(
                f"fold {self.index}: boundaries must satisfy "
                f"train_start < train_end <= test_start < test_end, got "
                f"({self.train_start}, {self.train_end}, {self.test_start}, {self.test_end}). "
                f"Test data overlapping or preceding train data is validation on the past."
            )
        if self.warmup < 0:
            raise PanelError(f"fold {self.index}: warmup must be non-negative, got {self.warmup}")

    @property
    def usable(self) -> bool:
        """Is the requested warmup actually available before the test window?

        False means this fold cannot be scored. Reporting it as a zero-return loss — which is
        what happens when a slow signal meets a short window — credits the strategy with a
        result it never produced.
        """
        return self.test_start - self.warmup >= self.train_start

    @property
    def signal_start(self) -> int:
        return max(self.train_start, self.test_start - self.warmup)

    @property
    def test_length(self) -> int:
        return self.test_end - self.test_start

    def train(self, panel: Panel) -> Panel:
        return panel.slice_time(self.train_start, self.train_end)

    def signal_window(self, panel: Panel) -> Panel:
        """Warmup plus test. What the signal function sees."""
        return panel.slice_time(self.signal_start, self.test_end)

    def test(self, panel: Panel) -> Panel:
        """The scored window only."""
        return panel.slice_time(self.test_start, self.test_end)

    def scored_rows(self) -> slice:
        """Where the test window sits inside `signal_window`."""
        offset = self.test_start - self.signal_start
        return slice(offset, offset + self.test_length)


def rolling_folds(
    panel: Panel,
    *,
    train_size: int,
    test_size: int,
    warmup: int = 0,
    step: int | None = None,
) -> list[Fold]:
    """Fixed-length train windows sliding forward.

    Args:
        panel: Any panel on the target index; only its length is used.
        train_size: Rows in each training window.
        test_size: Rows scored per fold.
        warmup: Rows before the test window over which the signal may converge. Set this to at
            least the longest lookback the signal uses, or the first bars of every fold are
            measuring warmup rather than the strategy.
        step: Rows between successive test windows. Defaults to `test_size`, which tiles them —
            overlapping windows reuse bars and inflate every t-statistic computed from the folds.
    """
    if train_size < 1 or test_size < 1:
        raise PanelError("train_size and test_size must be >= 1")
    if warmup < 0:
        raise PanelError(f"warmup must be non-negative, got {warmup}")
    if warmup >= train_size:
        raise PanelError(
            f"warmup {warmup} must be smaller than train_size {train_size}, otherwise every "
            f"fold's warmup would reach past the start of its own training window and no fold "
            f"would be usable."
        )
    stride = test_size if step is None else step
    if stride < 1:
        raise PanelError(f"step must be >= 1, got {stride}")
    if train_size + test_size > len(panel):
        raise PanelError(
            f"need at least train_size + test_size = {train_size + test_size} rows, "
            f"got {len(panel)}"
        )

    folds: list[Fold] = []
    start = 0
    while start + train_size + test_size <= len(panel):
        train_end = start + train_size
        folds.append(
            Fold(
                index=len(folds),
                train_start=start,
                train_end=train_end,
                test_start=train_end,
                test_end=train_end + test_size,
                warmup=warmup,
            )
        )
        start += stride
    return folds


def expanding_folds(
    panel: Panel,
    *,
    initial_train: int,
    test_size: int,
    warmup: int = 0,
) -> list[Fold]:
    """Training windows that grow, always starting at row zero.

    Closer to how a live system actually accumulates history, and it gives later folds more data
    to fit on. The cost is that early and late folds are not comparable — a result that improves
    over time may be a strategy learning, or may simply be a longer window smoothing the estimate.
    """
    if initial_train < 1 or test_size < 1:
        raise PanelError("initial_train and test_size must be >= 1")
    if warmup >= initial_train:
        raise PanelError(f"warmup {warmup} must be smaller than initial_train {initial_train}")

    folds: list[Fold] = []
    train_end = initial_train
    while train_end + test_size <= len(panel):
        folds.append(
            Fold(
                index=len(folds),
                train_start=0,
                train_end=train_end,
                test_start=train_end,
                test_end=train_end + test_size,
                warmup=warmup,
            )
        )
        train_end += test_size
    return folds


@dataclass(frozen=True, slots=True)
class FoldOutcome:
    """What one fold produced, including the reason it produced nothing."""

    index: int
    strategy_return: float
    baseline_return: float
    traded: bool
    turnover: float
    skipped: bool = False
    reason: str = ""

    @property
    def excess(self) -> float:
        return self.strategy_return - self.baseline_return

    @property
    def genuine_win(self) -> bool:
        """Beat the baseline *while trading*.

        A fold the strategy sat out is not a fold it won. Zero return beats every negative
        baseline, and counting that as a win is how a strategy that never traded ends up
        reported as defensive.
        """
        return self.traded and not self.skipped and self.excess > 0


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Every fold, and the honest summary of them."""

    outcomes: tuple[FoldOutcome, ...]
    #: Net returns of every scored fold, concatenated in fold order — the out-of-sample series
    #: any Sharpe should be computed on. Row 0 of each fold is structurally gross-zero: the
    #: engine earns its first return one period after the entry rebalance, so that row carries
    #: only the fold's entry costs (and any day-one delisting drag). It is kept rather than
    #: dropped, deliberately — discarding it would erase the cost of putting the book on, once
    #: per fold, which flatters every statistic computed from this series.
    returns: Array

    @property
    def scored(self) -> tuple[FoldOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.skipped)

    @property
    def skipped(self) -> tuple[FoldOutcome, ...]:
        return tuple(o for o in self.outcomes if o.skipped)

    @property
    def no_ops(self) -> tuple[FoldOutcome, ...]:
        return tuple(o for o in self.scored if not o.traded)

    @property
    def genuine_wins(self) -> int:
        return sum(1 for o in self.outcomes if o.genuine_win)

    @property
    def win_rate(self) -> float:
        scored = self.scored
        return self.genuine_wins / len(scored) if scored else 0.0

    @property
    def mean_excess(self) -> float:
        scored = self.scored
        return float(np.mean([o.excess for o in scored])) if scored else 0.0

    @property
    def testable(self) -> bool:
        """Did this evaluation actually measure anything?

        False when every fold was skipped or sat out. The distinction matters: a strategy that
        could not be tested has not been rejected, and reporting it as a loss invents a result.
        """
        return bool(self.scored) and any(o.traded for o in self.scored)

    def summary(self) -> str:
        if not self.outcomes:
            return "no folds"
        if not self.testable:
            return (
                f"UNTESTABLE: {len(self.skipped)} fold(s) skipped for insufficient warmup, "
                f"{len(self.no_ops)} scored fold(s) never traded. Nothing was measured — this "
                f"is silence, not underperformance."
            )

        lines = [
            f"{len(self.scored)}/{len(self.outcomes)} folds scored | "
            f"genuine wins {self.genuine_wins}/{len(self.scored)} "
            f"({self.win_rate:.0%}) | mean excess {self.mean_excess:+.2%}"
        ]
        for outcome in self.outcomes:
            if outcome.skipped:
                lines.append(f"  fold {outcome.index:<3} SKIPPED — {outcome.reason}")
                continue
            flag = "" if outcome.traded else "   NO-OP, never traded"
            lines.append(
                f"  fold {outcome.index:<3} strategy {outcome.strategy_return:>+8.2%}  "
                f"baseline {outcome.baseline_return:>+8.2%}  "
                f"excess {outcome.excess:>+8.2%}{flag}"
            )
        if self.no_ops:
            lines.append(
                f"  {len(self.no_ops)} fold(s) never traded. Their 0.00% beats every negative "
                f"baseline without participating, and is not counted as a win."
            )
        if self.skipped:
            lines.append(
                f"  {len(self.skipped)} fold(s) skipped for insufficient warmup — raise warmup, "
                f"lengthen the sample, or use a faster signal."
            )
        return "\n".join(lines)


def walk_forward(
    folds: Sequence[Fold],
    signal_function: SignalFunction,
    prices: Panel,
    *,
    costs: CostModel | None = None,
    baseline: SignalFunction | None = None,
    execution_lag: int = 1,
    **backtest_kwargs: object,
) -> WalkForwardResult:
    """Score a signal out of sample, fold by fold.

    Args:
        folds: From `rolling_folds` or `expanding_folds`.
        signal_function: Maps a price panel to target weights. Called on `warmup + test` and
            scored on `test` only, so indicators are converged at the first scored bar.
        prices: Full price panel.
        costs: Cost model. Passing `None` measures how often the strategy trades, not whether
            it works.
        baseline: Comparison weights, defaulting to equal-weight-everything. A strategy that
            cannot beat holding the universe has not earned the complexity.
        execution_lag: Passed through to the engine.
        **backtest_kwargs: Forwarded to `backtest`. In particular `returns=` accepts the
            vendor's full-sample return series (`Dataset.returns`); `align()` inside the engine
            intersects it down to each fold's test window, so no per-fold slicing is needed.

    Returns:
        A `WalkForwardResult`. Check `.testable` before reading any number in it.
    """
    if not folds:
        raise PanelError("walk_forward needs at least one fold")

    def equal_weight(panel: Panel) -> Panel:
        counts = panel.investable.sum(axis=1, keepdims=True).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(panel.investable & (counts > 0), 1.0 / counts, 0.0)
        return panel.with_values(weights, name="equal_weight")

    reference = baseline or equal_weight
    outcomes: list[FoldOutcome] = []
    all_returns: list[Array] = []

    for fold in folds:
        if not fold.usable:
            outcomes.append(
                FoldOutcome(
                    index=fold.index,
                    strategy_return=0.0,
                    baseline_return=0.0,
                    traded=False,
                    turnover=0.0,
                    skipped=True,
                    reason=(
                        f"warmup {fold.warmup} reaches before the training window "
                        f"(test starts at row {fold.test_start}, train at {fold.train_start})"
                    ),
                )
            )
            continue

        window = fold.signal_window(prices)
        if window.n_times <= execution_lag:
            outcomes.append(
                FoldOutcome(
                    index=fold.index,
                    strategy_return=0.0,
                    baseline_return=0.0,
                    traded=False,
                    turnover=0.0,
                    skipped=True,
                    reason=f"window of {window.n_times} rows is too short for lag {execution_lag}",
                )
            )
            continue

        # The check above guards the SIGNAL window (warmup + test), but the engine is handed
        # only the test slice — so a fold whose test window is shorter than the lag passed the
        # first guard and then raised PanelError inside `_score`, killing the whole evaluation
        # over its final, structurally short fold. A fold that cannot run is skipped with its
        # reason recorded, exactly like an unusable-warmup fold; it is not a loss and not fatal.
        if fold.test_length <= execution_lag:
            outcomes.append(
                FoldOutcome(
                    index=fold.index,
                    strategy_return=0.0,
                    baseline_return=0.0,
                    traded=False,
                    turnover=0.0,
                    skipped=True,
                    reason=(
                        f"test window of {fold.test_length} row(s) is too short for "
                        f"execution_lag {execution_lag}; the engine needs at least "
                        f"{execution_lag + 1}"
                    ),
                )
            )
            continue

        scored = fold.scored_rows()
        strategy_weights = signal_function(window)
        baseline_weights = reference(window)

        test_prices = fold.test(prices)
        strategy = _score(
            strategy_weights, scored, test_prices, costs, execution_lag, backtest_kwargs
        )
        control = _score(
            baseline_weights, scored, test_prices, costs, execution_lag, backtest_kwargs
        )

        outcomes.append(
            FoldOutcome(
                index=fold.index,
                strategy_return=strategy.total_return,
                baseline_return=control.total_return,
                traded=strategy.traded,
                turnover=float(np.nansum(strategy.turnover)),
            )
        )
        all_returns.append(strategy.net_returns)

    combined = np.concatenate(all_returns) if all_returns else np.zeros(0)
    return WalkForwardResult(outcomes=tuple(outcomes), returns=combined)


def _score(
    weights: Panel,
    scored: slice,
    test_prices: Panel,
    costs: CostModel | None,
    execution_lag: int,
    extra: dict[str, object],
) -> BacktestResult:
    """Backtest the test-window slice of a signal computed over warmup + test."""
    sliced = weights.slice_time(scored.start, scored.stop)
    aligned = test_prices.with_values(sliced.values, name=weights.name)
    return backtest(
        aligned,
        test_prices,
        costs=costs,
        execution_lag=execution_lag,
        **extra,  # type: ignore[arg-type]
    )
