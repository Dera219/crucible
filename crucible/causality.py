"""Prove a signal cannot see the future, mechanically, in milliseconds.

Apex enforces no-lookahead structurally: a signal emitted on bar *t* fills at bar *t+1*'s open,
and a strategy cannot opt out. That works, and it only covers the *execution* boundary. It says
nothing about whether the signal itself was computed from information that existed at *t*, and in
a vectorised research framework that is where lookahead actually lives.

## The bug this exists to catch

Not `shift(-1)`. That is too obvious to survive review.

The one that gets shipped is **full-sample normalisation**:

    signal = (raw - raw.mean()) / raw.std()      # over the whole history

This looks like every textbook. It is also a signal that knows the mean and variance of a
distribution that had not finished happening, and it inflates results in a way that is invisible
in-sample, invisible out-of-sample under a naive split, and only shows up as an unexplained gap
between backtest and live. Its cousins are just as respectable-looking: fitting a scaler on all
data before splitting, ranking against the full period, winsorising to global quantiles,
dropping assets that were later delisted.

## The test

A causal function has one property: **its output at time t depends only on inputs at or before
t.** So perturb everything strictly after *t*, recompute, and compare the head.

    original    = f(panel)
    perturbed   = f(panel with all rows after t replaced by noise)
    assert original[:t+1] is bit-identical to perturbed[:t+1]

If any value at or before *t* moved, the function read something from after *t*. There is no
ambiguity and no judgement involved — it is not a heuristic, it is the definition of causality
turned into an assertion.

The perturbation is deliberately violent (shuffled rows, scaled magnitudes, injected NaNs) so
that a function which merely *touches* the future almost certainly *changes* because of it. And
the cut points sweep the whole sample rather than testing one, because a function can be causal
everywhere except during its warmup.

## What this does not catch

Lookahead that is in the *data* rather than the code: a vendor restating a fundamental and
back-filling it to the original report date, a price series adjusted with knowledge of a later
split, an index membership list that reflects today's constituents. No amount of code analysis
finds those — they are handled by point-in-time data and by `crucible.panel`'s insistence that
NaN means "not investable then". This module proves the transformation is honest; it cannot prove
the inputs were.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from crucible.panel import Panel

__all__ = ["CausalityViolation", "assert_causal", "check_causality", "CausalityReport"]

SignalFunction = Callable[..., Panel]


class CausalityViolation(AssertionError):
    """A function's output changed at or before a timestamp when only later data was altered."""


@dataclass(frozen=True, slots=True)
class CausalityReport:
    """Where a function looked forward, if it did."""

    passed: bool
    cut_points_tested: tuple[int, ...]
    first_failure: int | None = None
    failing_cells: int = 0
    worst_delta: float = 0.0
    detail: str = ""

    def describe(self) -> str:
        if self.passed:
            return (
                f"causal: output was unchanged at every cut point "
                f"{list(self.cut_points_tested)} under perturbation of all later data"
            )
        return (
            f"NOT CAUSAL: at cut point t={self.first_failure}, {self.failing_cells} cell(s) at or "
            f"before t changed when only data after t was perturbed "
            f"(largest change {self.worst_delta:.6g}).\n{self.detail}"
        )


def _perturb(panel: Panel, cut: int, rng: np.random.Generator) -> Panel:
    """Replace every row strictly after `cut` with aggressively different data.

    Three perturbations at once, because each catches a different kind of peeking: shuffling rows
    catches anything order-dependent, rescaling catches means and variances, and injecting NaNs
    catches counts and coverage-dependent branches. A function that reads the future is unlikely
    to survive all three unchanged.
    """
    values = panel.values.copy()
    tail = values[cut + 1 :]
    if tail.size == 0:
        return panel

    shuffled = tail[rng.permutation(tail.shape[0])]
    scaled = shuffled * rng.uniform(2.0, 10.0, size=shuffled.shape)
    holes = rng.random(scaled.shape) < 0.15
    scaled[holes] = np.nan

    values[cut + 1 :] = scaled
    return panel.with_values(values)


def _compare_heads(original: Panel, perturbed: Panel, cut: int) -> tuple[int, float]:
    """Count cells at or before `cut` that differ. NaN is equal to NaN here."""
    left = original.values[: cut + 1]
    right = perturbed.values[: cut + 1]

    both_nan = np.isnan(left) & np.isnan(right)
    differs = ~(np.isclose(left, right, rtol=0, atol=0, equal_nan=True) | both_nan)
    count = int(differs.sum())
    if count == 0:
        return 0, 0.0

    with np.errstate(invalid="ignore"):
        deltas = np.abs(left - right)
    finite = deltas[np.isfinite(deltas)]
    worst = float(finite.max()) if finite.size else float("inf")
    return count, worst


def check_causality(
    function: SignalFunction,
    panel: Panel,
    *,
    cut_points: Sequence[int] | None = None,
    seed: int = 0,
    trials: int = 3,
) -> CausalityReport:
    """Test whether `function` reads data from after each cut point.

    Args:
        function: Takes a `Panel`, returns a `Panel` on the same index. The signal under test.
        panel: Input data. Should be long enough to include the function's warmup.
        cut_points: Row indices to test. Defaults to a spread across the sample, deliberately
            including an early point — a function can be causal everywhere except during warmup,
            and warmup is where partial-window logic goes wrong.
        seed: Base seed. Perturbations are pseudo-random but reproducible.
        trials: Independent perturbations per cut point. One unlucky perturbation can coincide
            with a function's output by chance; three essentially cannot.

    Returns:
        A `CausalityReport`. `passed` is True only if no output at or before any cut point moved
        under any perturbation.
    """
    if panel.n_times < 4:
        raise ValueError(
            f"causality needs at least 4 timestamps to have anything to perturb, got "
            f"{panel.n_times}"
        )

    if cut_points is None:
        n = panel.n_times
        candidates = {n // 8, n // 4, n // 2, (3 * n) // 4, n - 2}
        cut_points = tuple(sorted(c for c in candidates if 0 <= c < n - 1))

    baseline = function(panel)
    if baseline.n_times != panel.n_times:
        raise ValueError(
            f"function returned {baseline.n_times} rows for {panel.n_times} input rows. "
            f"Causality is defined per-timestamp, so the output must stay on the input's index."
        )

    for cut in cut_points:
        for trial in range(trials):
            rng = np.random.default_rng(seed + cut * 1000 + trial)
            perturbed_input = _perturb(panel, cut, rng)
            perturbed_output = function(perturbed_input)
            count, worst = _compare_heads(baseline, perturbed_output, cut)
            if count:
                rows, columns = np.where(
                    ~(
                        np.isclose(
                            baseline.values[: cut + 1],
                            perturbed_output.values[: cut + 1],
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                        )
                        | (
                            np.isnan(baseline.values[: cut + 1])
                            & np.isnan(perturbed_output.values[: cut + 1])
                        )
                    )
                )
                first = int(rows.min())
                asset = panel.assets[int(columns[rows.argmin()])]
                return CausalityReport(
                    passed=False,
                    cut_points_tested=tuple(cut_points),
                    first_failure=cut,
                    failing_cells=count,
                    worst_delta=worst,
                    detail=(
                        f"Earliest affected cell: row {first} ({panel.index[first]}), asset "
                        f"{asset}. That value is computed from data that had not happened yet.\n"
                        f"The usual cause is a statistic taken over the whole sample — a mean, a "
                        f"standard deviation, a rank, a quantile, a fitted scaler — rather than "
                        f"over a trailing window. Use ts_* or cs_* operators, which are causal "
                        f"by construction."
                    ),
                )

    return CausalityReport(passed=True, cut_points_tested=tuple(cut_points))


def assert_causal(
    function: SignalFunction,
    panel: Panel,
    *,
    cut_points: Sequence[int] | None = None,
    seed: int = 0,
    trials: int = 3,
) -> None:
    """Raise `CausalityViolation` unless `function` is causal on `panel`.

    Call this on every signal before it is backtested. It costs milliseconds and it removes the
    single most expensive class of research error — the one that produces a beautiful equity
    curve nobody could have traded.
    """
    report = check_causality(function, panel, cut_points=cut_points, seed=seed, trials=trials)
    if not report.passed:
        raise CausalityViolation(report.describe())
