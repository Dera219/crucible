"""The signal algebra: cross-sectional and time-series operators over panels.

Two families, and the distinction is the whole vocabulary.

**Cross-sectional** operators (`cs_*`) work along a single row — one timestamp, all assets. They
answer "how does this asset compare to its peers *right now*". These are what make relative
statements expressible, and relative statements are where the documented anomalies live:
momentum is a cross-sectional rank of trailing returns, value is a cross-sectional rank of a
fundamental ratio, low-vol is a cross-sectional rank of realised volatility. A row-wise operator
cannot see another timestamp, so it is causal by construction.

**Time-series** operators (`ts_*`) work down a column — one asset, a trailing window. They answer
"how does this asset compare to its own past". These are causal only if the window is trailing,
which is why every one of them is built on `Panel.rolling` and `Panel.shift`, and why
`Panel.shift` refuses negative arguments.

## Why this is a library rather than a convention

The classic quant research bug is not a negative shift — that is too obvious to survive review.
It is **normalising over the full sample**: z-scoring a signal using the mean and standard
deviation of the entire history, including the part that had not happened yet. The result is a
signal that knows the future distribution, and it is invisible because the code looks like exactly
what a textbook says to write.

Every operator here is either row-wise or trailing-window, so full-sample normalisation is not
something you can accidentally reach for. And `crucible.causality` mechanically proves it: perturb
everything after time *t*, recompute, and assert nothing at or before *t* moved. An operator that
peeks fails that test in milliseconds, no matter how respectable it looks.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import numpy.typing as npt

from crucible.panel import Panel, PanelError


@contextmanager
def _quiet_empty_slices() -> Iterator[None]:
    """Suppress numpy's all-NaN-slice warnings, which here are expected rather than surprising.

    A row with no investable assets is a normal state — a market holiday, the start of a
    universe, a symbol not yet listed — and every operator below already converts those rows to
    NaN deliberately. Letting numpy warn about each one buries real warnings under thousands of
    lines of noise, which is how real warnings stop being read.

    Deliberately narrow: only the two messages that mean "you asked for a statistic of nothing".
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore", message="Degrees of freedom <= 0", category=RuntimeWarning
        )
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        yield


__all__ = [
    "cs_demean",
    "cs_neutralize",
    "cs_rank",
    "cs_scale",
    "cs_winsorize",
    "cs_zscore",
    "ts_decay",
    "ts_rank",
    "ts_zscore",
]

Array = npt.NDArray[np.float64]

#: Rows with fewer investable assets than this cannot support a cross-sectional statement and
#: are emitted as all-NaN. Three names do not have a top decile; computing one produces a number
#: that looks like every other number in the column and means nothing.
_MIN_BREADTH = 5


def _row_valid(values: Array) -> npt.NDArray[np.bool_]:
    return ~np.isnan(values)


def _thin_rows(values: Array, minimum: int) -> npt.NDArray[np.bool_]:
    return _row_valid(values).sum(axis=1) < minimum


def cs_rank(panel: Panel, *, min_breadth: int = _MIN_BREADTH, pct: bool = True) -> Panel:
    """Rank each asset against its peers at the same timestamp.

    Returns values in [0, 1] by default, where 1 is the largest. Ties take their average rank,
    which matters more than it sounds: with integer-valued inputs, breaking ties by position
    would make the result depend on column order, and column order is an accident of how the
    data was loaded.

    NaN inputs stay NaN — an asset that was not investable does not get a rank, and must not
    then be selected into a portfolio.
    """
    values = panel.values
    out = np.full_like(values, np.nan)

    for row in range(values.shape[0]):
        valid = _row_valid(values[row])
        count = int(valid.sum())
        if count < min_breadth:
            continue
        observed = values[row, valid]
        order = observed.argsort(kind="stable")
        ranks = np.empty(count, dtype=np.float64)
        ranks[order] = np.arange(1, count + 1, dtype=np.float64)

        # Average ranks within tied groups, so the result does not depend on column order.
        unique, inverse, counts = np.unique(observed, return_inverse=True, return_counts=True)
        if len(unique) < count:
            sums = np.zeros(len(unique), dtype=np.float64)
            np.add.at(sums, inverse, ranks)
            ranks = (sums / counts)[inverse]

        out[row, valid] = (ranks - 1) / (count - 1) if pct and count > 1 else ranks

    return panel.with_values(out, name=f"cs_rank({panel.name})")


def cs_demean(panel: Panel, *, min_breadth: int = _MIN_BREADTH) -> Panel:
    """Subtract the cross-sectional mean, so the row sums to zero.

    This is what makes a signal *market neutral* in the crudest sense: the average position is
    zero, so the strategy expresses a view about relative performance rather than about the
    market going up. Without it, almost any long-only signal on US equities from 2020 to 2025
    "works", because it is a leveraged bet on a bull market wearing a lab coat.
    """
    values = panel.values
    with np.errstate(invalid="ignore"), _quiet_empty_slices():
        means = np.nanmean(np.where(_row_valid(values), values, np.nan), axis=1, keepdims=True)
    out = values - means
    out[_thin_rows(values, min_breadth)] = np.nan
    return panel.with_values(out, name=f"cs_demean({panel.name})")


def cs_zscore(panel: Panel, *, min_breadth: int = _MIN_BREADTH) -> Panel:
    """Standardise within each row: (x - row mean) / row standard deviation.

    Row-wise on purpose. Standardising over the full sample — the reflex, and what most tutorials
    show — gives the signal the future's mean and variance, which is lookahead that survives
    review because the code looks textbook-correct. `crucible.causality` catches it in
    milliseconds; this function makes it hard to write in the first place.
    """
    values = panel.values
    masked = np.where(_row_valid(values), values, np.nan)
    with np.errstate(invalid="ignore"), _quiet_empty_slices():
        means = np.nanmean(masked, axis=1, keepdims=True)
        stds = np.nanstd(masked, axis=1, keepdims=True)
    # A zero-variance row carries no information; NaN rather than a divide-by-zero infinity.
    stds = np.where(stds == 0, np.nan, stds)
    out = (values - means) / stds
    out[_thin_rows(values, min_breadth)] = np.nan
    return panel.with_values(out, name=f"cs_zscore({panel.name})")


def cs_winsorize(panel: Panel, *, limit: float = 0.01, min_breadth: int = _MIN_BREADTH) -> Panel:
    """Clip each row to its own [limit, 1 - limit] quantiles.

    One bad print, one stock that fell 99% on a data error, one un-adjusted split — any of these
    dominates a mean or a regression and quietly becomes the strategy. Winsorising per row keeps
    the ordering intact while refusing to let a single cell decide the portfolio.
    """
    if not 0 <= limit < 0.5:
        raise PanelError(f"limit must lie in [0, 0.5), got {limit}")
    if limit == 0:
        return panel

    values = panel.values
    out = values.copy()
    for row in range(values.shape[0]):
        valid = _row_valid(values[row])
        if int(valid.sum()) < min_breadth:
            out[row] = np.nan
            continue
        observed = values[row, valid]
        low, high = np.quantile(observed, [limit, 1 - limit])
        out[row, valid] = np.clip(observed, low, high)
    return panel.with_values(out, name=f"cs_winsorize({panel.name},{limit})")


def cs_scale(panel: Panel, *, target: float = 1.0, min_breadth: int = _MIN_BREADTH) -> Panel:
    """Scale each row so the absolute values sum to `target`.

    This is the step that turns a signal into something portfolio-shaped: gross exposure becomes
    a constant you chose, rather than an accident of how large the raw signal happened to be that
    day. Without it, a signal whose magnitude drifts over time silently changes its own leverage.
    """
    if target <= 0:
        raise PanelError(f"target must be positive, got {target}")
    values = panel.values
    masked = np.where(_row_valid(values), values, np.nan)
    with np.errstate(invalid="ignore"), _quiet_empty_slices():
        gross = np.nansum(np.abs(masked), axis=1, keepdims=True)
    gross = np.where(gross == 0, np.nan, gross)
    out = values / gross * target
    out[_thin_rows(values, min_breadth)] = np.nan
    return panel.with_values(out, name=f"cs_scale({panel.name})")


def cs_neutralize(panel: Panel, factor: Panel, *, min_breadth: int = _MIN_BREADTH) -> Panel:
    """Remove the part of a signal explained by another panel, row by row.

    Ordinary least squares within each timestamp; what comes back is the residual. Use it to ask
    whether a signal survives once the obvious explanation is taken out — neutralise momentum
    against beta, or a fundamental ratio against market cap, and see what is left.

    A signal that vanishes under neutralisation was the control variable all along, which is a
    finding rather than a disappointment: it is cheaper to learn that here than after funding it.
    """
    if factor.index != panel.index or factor.assets != panel.assets:
        raise PanelError(
            f"cs_neutralize: {panel.name!r} and {factor.name!r} are not aligned. Use align()."
        )

    values, exposures = panel.values, factor.values
    out = np.full_like(values, np.nan)

    for row in range(values.shape[0]):
        valid = _row_valid(values[row]) & _row_valid(exposures[row])
        if int(valid.sum()) < min_breadth:
            continue
        y = values[row, valid]
        x = exposures[row, valid]
        design = np.column_stack([np.ones_like(x), x])
        # lstsq rather than a normal-equation inverse: a factor with near-zero cross-sectional
        # spread makes X'X singular, and lstsq degrades to the minimum-norm solution instead of
        # returning garbage with a warning nobody reads.
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        out[row, valid] = y - design @ coefficients

    return panel.with_values(out, name=f"cs_neutralize({panel.name},{factor.name})")


def ts_zscore(panel: Panel, window: int, *, min_periods: int | None = None) -> Panel:
    """Standardise each asset against its own trailing window.

    Answers "is this unusual *for this asset*", which is a different question from `cs_zscore`'s
    "is this unusual compared to its peers". Both are useful and they are frequently confused;
    the naming here makes the choice explicit at every call site.
    """
    mean = panel.rolling(window, "mean", min_periods=min_periods)
    std = panel.rolling(window, "std", min_periods=min_periods)
    safe = np.where(std.values == 0, np.nan, std.values)
    return panel.with_values(
        (panel.values - mean.values) / safe, name=f"ts_zscore({panel.name},{window})"
    )


def ts_decay(panel: Panel, window: int) -> Panel:
    """Linearly-weighted trailing average — today weighted `window`, oldest weighted 1.

    The standard remedy for turnover. A raw signal that flips sign on noise trades constantly and
    pays the spread every time; decaying it keeps the shape and smooths the flicker. It is also
    the cheapest honest way to make a signal survive costs, and worth trying before concluding a
    result is real: an edge that only exists at 300% monthly turnover usually is not one.
    """
    if window < 1:
        raise PanelError(f"window must be >= 1, got {window}")
    values = panel.values
    weights = np.arange(1, window + 1, dtype=np.float64)
    out = np.full_like(values, np.nan)

    for row in range(values.shape[0]):
        low = max(0, row - window + 1)
        block = values[low : row + 1]
        active = weights[-block.shape[0] :][:, None]
        valid = _row_valid(block)
        weighted = np.where(valid, block * active, 0.0).sum(axis=0)
        total = np.where(valid, active, 0.0).sum(axis=0)
        # Require a full window, matching Panel.rolling: a partial decay is a shorter decay.
        complete = valid.sum(axis=0) == window
        with np.errstate(invalid="ignore", divide="ignore"):
            out[row] = np.where(complete & (total > 0), weighted / total, np.nan)

    return panel.with_values(out, name=f"ts_decay({panel.name},{window})")


def ts_rank(panel: Panel, window: int) -> Panel:
    """Where does today sit within this asset's own trailing window, in [0, 1]?

    Robust to level and to regime: a price that has tripled still ranks in the middle of its
    recent range if it has been flat for a month. Useful when the raw magnitude of a signal
    drifts but its ordering does not.
    """
    if window < 1:
        raise PanelError(f"window must be >= 1, got {window}")
    values = panel.values
    out = np.full_like(values, np.nan)

    for row in range(values.shape[0]):
        low = max(0, row - window + 1)
        block = values[low : row + 1]
        if block.shape[0] < window:
            continue
        current = block[-1]
        valid = _row_valid(block)
        complete = valid.sum(axis=0) == window
        below = (np.where(valid, block, np.inf) < current).sum(axis=0)
        out[row] = np.where(complete, below / (window - 1) if window > 1 else 0.0, np.nan)

    return panel.with_values(out, name=f"ts_rank({panel.name},{window})")
