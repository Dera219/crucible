"""The panel: aligned (time x asset) matrices, and the reason everything else is possible.

Apex's core type is a `PriceSeries` — one symbol, a list of bars, a strategy that sees one bar at
a time. That shape is honest and it is also a ceiling, because it can only express **absolute**
statements: buy when this thing does that. Almost every documented equity anomaly is
**relative** — momentum, value, quality, low-volatility, short-term reversal are all statements
about one asset *compared to others at the same instant*. You cannot say "long the top decile,
short the bottom" to a backtester that sees one symbol at a time. Not slowly, not awkwardly:
you cannot say it at all.

So the core type here is a matrix. Rows are timestamps, columns are assets, and every field —
close, volume, market cap, a computed signal — is a matrix on the same index. Once data is
shaped this way, a cross-sectional operation is one line of numpy, testing a thousand parameter
combinations is a loop over seconds rather than hours, and the questions worth asking become
askable.

## NaN is a first-class value and it means "not investable"

The single most important design decision in this file.

A retail backtest usually starts from a list of tickers that exist *today*, which quietly asks
"how would this have done, given that I knew in advance which companies would survive and get
included?" That is survivorship bias, it inflates nearly every result, and it is invisible
because nothing errors — the data just silently lacks the companies that failed.

Here, an asset that was not listed, was halted, has been delisted, or was not a universe member
at time *t* carries `NaN` at `[t, asset]`. It occupies a column, it holds a place, and every
operation propagates its absence honestly. `Panel.investable` is the boolean mask of "could this
have been traded, then", and portfolio construction is required to respect it.

That makes survivorship bias a thing you must *actively opt into* by loading a bad universe,
rather than a thing that happens to you by default.

## Immutability

Every panel is frozen and every operation returns a new one. The underlying arrays are marked
read-only, because numpy views alias and a single in-place `+=` on a slice can silently rewrite
history behind three other objects that thought they held their own copy. Research code is
edited fast and read rarely; it should not be possible to corrupt a dataset by accident.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import numpy.typing as npt

__all__ = ["Panel", "PanelError", "align"]

Array = npt.NDArray[np.float64]


class PanelError(ValueError):
    """The panel is malformed, or an operation would produce a malformed one."""


@contextmanager
def _quiet_empty_slices() -> Iterator[None]:
    """Suppress numpy's all-NaN-slice warnings, which here are expected rather than surprising.

    A window containing no investable observations is normal — a symbol before it listed, a
    halted name, a universe that has not started. Every caller already turns those into NaN on
    purpose. Warning about each one buries real warnings under noise.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore", message="Degrees of freedom <= 0", category=RuntimeWarning
        )
        warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)
        yield


def _freeze(values: Array) -> Array:
    """Return a read-only view, copying only when the input is not already float64.

    Marking arrays read-only is not politeness. Numpy slices are views, so `panel.values[0] += 1`
    on a shared array rewrites history for every object holding a reference — a class of bug
    that produces plausible numbers and no error at all.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.base is not None or array.flags.writeable:
        array = array.copy()
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class Panel:
    """One field of data, aligned across time and assets.

    Attributes:
        values: `(n_times, n_assets)` float64, read-only. `NaN` means not investable.
        index: Ascending, strictly increasing, timezone-naive UTC dates or datetimes.
        assets: Column labels, unique, in the same order as `values` columns.
        name: What this field is. Used in error messages and reports.
    """

    values: Array
    index: tuple[datetime | date, ...]
    assets: tuple[str, ...]
    name: str = "panel"

    def __post_init__(self) -> None:
        values = _freeze(self.values)
        object.__setattr__(self, "values", values)

        if values.ndim != 2:
            raise PanelError(f"{self.name}: values must be 2-D (time x asset), got {values.ndim}-D")
        if values.shape != (len(self.index), len(self.assets)):
            raise PanelError(
                f"{self.name}: values {values.shape} does not match "
                f"{len(self.index)} timestamps x {len(self.assets)} assets"
            )
        if len(set(self.assets)) != len(self.assets):
            duplicates = sorted({a for a in self.assets if self.assets.count(a) > 1})
            raise PanelError(f"{self.name}: duplicate asset columns {duplicates}")
        if any(
            later <= earlier for earlier, later in zip(self.index, self.index[1:], strict=False)
        ):
            raise PanelError(
                f"{self.name}: index must be strictly increasing. An out-of-order or duplicated "
                f"timestamp silently breaks every shift, and therefore every no-lookahead "
                f"guarantee that depends on one."
            )

    # ── shape ────────────────────────────────────────────────────────────────

    @property
    def shape(self) -> tuple[int, int]:
        rows, columns = self.values.shape
        return rows, columns

    @property
    def n_times(self) -> int:
        return len(self.index)

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    def __len__(self) -> int:
        return self.n_times

    def __repr__(self) -> str:
        span = f"{self.index[0]} → {self.index[-1]}" if self.index else "empty"
        return (
            f"Panel({self.name!r}, {self.n_times}x{self.n_assets}, {span}, "
            f"{self.coverage:.1%} investable)"
        )

    # ── investability ────────────────────────────────────────────────────────

    @property
    def investable(self) -> npt.NDArray[np.bool_]:
        """Boolean mask of cells that were tradable at that timestamp.

        Portfolio construction must respect this. Holding a name on a date where it is False is
        holding something that did not exist, could not be bought, or was not in the universe —
        the mechanism by which backtests earn returns nobody could have collected.
        """
        return ~np.isnan(self.values)

    @property
    def coverage(self) -> float:
        """Fraction of cells that are investable. A headline number worth watching."""
        if self.values.size == 0:
            return 0.0
        return float(self.investable.mean())

    def count_per_row(self) -> npt.NDArray[np.int64]:
        """Investable assets at each timestamp.

        A cross-sectional strategy needs breadth. If this collapses to three names in 2020, the
        deciles computed there are noise however good the full-sample average looks.
        """
        return self.investable.sum(axis=1).astype(np.int64)

    # ── construction ─────────────────────────────────────────────────────────

    def with_values(self, values: Array, *, name: str | None = None) -> Panel:
        """A new panel on the same index and assets. The workhorse of the signal algebra."""
        return Panel(
            values=values,
            index=self.index,
            assets=self.assets,
            name=self.name if name is None else name,
        )

    def rename(self, name: str) -> Panel:
        return Panel(values=self.values, index=self.index, assets=self.assets, name=name)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Sequence[float]],
        *,
        index: Iterable[datetime | date],
        assets: Iterable[str],
        name: str = "panel",
    ) -> Panel:
        return cls(
            values=np.asarray(rows, dtype=np.float64),
            index=tuple(index),
            assets=tuple(assets),
            name=name,
        )

    @classmethod
    def empty(cls, *, assets: Iterable[str], name: str = "panel") -> Panel:
        columns = tuple(assets)
        return cls(
            values=np.empty((0, len(columns)), dtype=np.float64),
            index=(),
            assets=columns,
            name=name,
        )

    # ── selection ────────────────────────────────────────────────────────────

    def column(self, asset: str) -> Array:
        try:
            return self.values[:, self.assets.index(asset)]
        except ValueError as exc:
            raise PanelError(f"{self.name}: no column {asset!r}") from exc

    def select(self, assets: Iterable[str]) -> Panel:
        """Restrict to a subset of assets, preserving the requested order."""
        wanted = tuple(assets)
        missing = [a for a in wanted if a not in self.assets]
        if missing:
            raise PanelError(f"{self.name}: unknown assets {missing}")
        positions = [self.assets.index(a) for a in wanted]
        return Panel(
            values=self.values[:, positions], index=self.index, assets=wanted, name=self.name
        )

    def slice_time(self, start: int | None = None, stop: int | None = None) -> Panel:
        """Positional time slice. Used by walk-forward, which must never reorder rows."""
        return Panel(
            values=self.values[start:stop],
            index=self.index[start:stop],
            assets=self.assets,
            name=self.name,
        )

    def between(self, start: date | datetime, end: date | datetime) -> Panel:
        """Inclusive time window by timestamp."""
        keep = [i for i, stamp in enumerate(self.index) if start <= stamp <= end]
        if not keep:
            return Panel.empty(assets=self.assets, name=self.name)
        return self.slice_time(keep[0], keep[-1] + 1)

    # ── the operation everything depends on ──────────────────────────────────

    def shift(self, periods: int = 1) -> Panel:
        """Move data FORWARD in time by `periods` rows, filling the head with NaN.

        `shift(1)` means "the value as of the previous timestamp", which is what a signal is
        allowed to know when it acts. This is the single most important function in the library
        and the one whose sign is easiest to get backwards — a negative shift pulls the future
        into the present and produces a backtest that is not wrong so much as fictional.

        Negative values are therefore refused outright. There is no legitimate research use for
        looking forward that is not better expressed as `forward_returns`, which is explicitly
        named for what it does and is only ever used as a *target*, never as an input.
        """
        if periods < 0:
            raise PanelError(
                f"{self.name}: shift({periods}) would move future data into the present. If you "
                f"want a prediction target, use forward_returns() — which is named for what it "
                f"does, so it cannot be mistaken for an input."
            )
        if periods == 0:
            return self
        out = np.full_like(self.values, np.nan)
        if periods < self.n_times:
            out[periods:] = self.values[:-periods]
        return self.with_values(out)

    def pct_change(self, periods: int = 1) -> Panel:
        """Simple return over `periods` rows, using only past data. Zero prices yield NaN."""
        if periods < 1:
            raise PanelError(f"{self.name}: pct_change needs periods >= 1, got {periods}")
        previous = self.shift(periods).values
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(previous == 0, np.nan, (self.values - previous) / previous)
        return self.with_values(out, name=f"{self.name}.pct_change({periods})")

    def log_return(self, periods: int = 1) -> Panel:
        previous = self.shift(periods).values
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((previous > 0) & (self.values > 0), self.values / previous, np.nan)
            out = np.log(ratio)
        return self.with_values(out, name=f"{self.name}.log_return({periods})")

    def forward_returns(self, periods: int = 1) -> Panel:
        """Return realised over the NEXT `periods` rows. A TARGET, never an input.

        Deliberately not expressible as `shift(-n)`. Naming it for what it is makes it obvious
        in a diff when it appears somewhere it should not: a forward return on the right-hand
        side of a signal definition is always a bug, and it is easy to miss when it is spelled
        as a negative shift buried in a chain of operations.
        """
        if periods < 1:
            raise PanelError(f"{self.name}: forward_returns needs periods >= 1, got {periods}")
        out = np.full_like(self.values, np.nan)
        if periods < self.n_times:
            future = self.values[periods:]
            current = self.values[:-periods]
            with np.errstate(divide="ignore", invalid="ignore"):
                out[:-periods] = np.where(current == 0, np.nan, (future - current) / current)
        return self.with_values(out, name=f"{self.name}.forward_returns({periods})")

    # ── rolling windows ──────────────────────────────────────────────────────

    def rolling(
        self, window: int, statistic: str = "mean", *, min_periods: int | None = None
    ) -> Panel:
        """Trailing-window statistic, inclusive of the current row.

        `min_periods` defaults to the full window. Emitting a value from a partial window is how
        a 200-day average quietly becomes a 12-day average for the first ten months of a
        backtest — the early period then looks unusually tradable and drags the whole result.
        """
        if window < 1:
            raise PanelError(f"{self.name}: window must be >= 1, got {window}")
        required = window if min_periods is None else min_periods
        if not 1 <= required <= window:
            raise PanelError(f"{self.name}: min_periods must lie in [1, {window}], got {required}")

        functions: dict[str, Callable[..., Array]] = {
            "mean": np.nanmean,
            "std": np.nanstd,
            "sum": np.nansum,
            "min": np.nanmin,
            "max": np.nanmax,
            "median": np.nanmedian,
        }
        if statistic not in functions:
            raise PanelError(
                f"{self.name}: unknown statistic {statistic!r}; expected one of {sorted(functions)}"
            )
        function = functions[statistic]

        out = np.full_like(self.values, np.nan)
        for row in range(self.n_times):
            low = max(0, row - window + 1)
            block = self.values[low : row + 1]
            valid = (~np.isnan(block)).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"), _quiet_empty_slices():
                computed = function(block, axis=0)
            out[row] = np.where(valid >= required, computed, np.nan)
        return self.with_values(out, name=f"{self.name}.rolling({window},{statistic})")

    # ── arithmetic ───────────────────────────────────────────────────────────

    def _combine(
        self, other: Panel | float, operation: Callable[[Array, Array | np.float64], Array]
    ) -> Panel:
        """Elementwise arithmetic against another panel or a scalar.

        The operation is passed as the ufunc itself rather than as a name looked up on `np`.
        A string lookup is untyped, so a typo would survive review and fail at runtime on the
        one branch nobody exercised.
        """
        right: Array | np.float64
        if isinstance(other, Panel):
            if other.index != self.index or other.assets != self.assets:
                raise PanelError(
                    f"cannot combine {self.name!r} and {other.name!r}: different index or assets. "
                    f"Use align() first — silently broadcasting mismatched panels is how a "
                    f"signal ends up paired with the wrong asset's returns."
                )
            right = other.values
        else:
            right = np.float64(other)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = operation(self.values, right)
        return self.with_values(result)

    def __add__(self, other: Panel | float) -> Panel:
        return self._combine(other, np.add)

    def __sub__(self, other: Panel | float) -> Panel:
        return self._combine(other, np.subtract)

    def __mul__(self, other: Panel | float) -> Panel:
        return self._combine(other, np.multiply)

    def __truediv__(self, other: Panel | float) -> Panel:
        return self._combine(other, np.divide)

    def __neg__(self) -> Panel:
        return self.with_values(-self.values)

    def abs(self) -> Panel:
        return self.with_values(np.abs(self.values))

    def clip(self, low: float | None = None, high: float | None = None) -> Panel:
        return self.with_values(np.clip(self.values, low, high))

    def where(self, condition: npt.NDArray[np.bool_], *, otherwise: float = np.nan) -> Panel:
        """Keep values where `condition` holds; elsewhere `otherwise` (NaN by default)."""
        if condition.shape != self.shape:
            raise PanelError(
                f"{self.name}: mask {condition.shape} does not match panel {self.shape}"
            )
        return self.with_values(np.where(condition, self.values, otherwise))

    def fill_na(self, value: float) -> Panel:
        """Replace NaN with a constant.

        Use sparingly and never on prices. Filling a missing price invents a trade that could not
        have happened; filling a missing *signal* with zero merely says "no opinion", which is
        usually what you mean.
        """
        return self.with_values(np.nan_to_num(self.values, nan=value, posinf=value, neginf=value))


def align(*panels: Panel, name: str | None = None) -> tuple[Panel, ...]:
    """Restrict panels to their common timestamps and assets, in a deterministic order.

    Alignment is explicit rather than automatic. Two panels differing by one timestamp is
    normally a data bug — a missing bar, a vendor gap — and silently taking the intersection
    would hide it. Calling this is the caller saying "I know they differ and I mean to drop the
    difference".
    """
    if not panels:
        raise PanelError("align() needs at least one panel")
    if len(panels) == 1:
        return panels

    common_times = sorted(set.intersection(*(set(p.index) for p in panels)))
    common_assets = sorted(set.intersection(*(set(p.assets) for p in panels)))
    if not common_times or not common_assets:
        raise PanelError(
            f"align(): panels share no common {'timestamps' if not common_times else 'assets'}. "
            f"Indices: {[(p.name, p.n_times, p.n_assets) for p in panels]}"
        )

    out: list[Panel] = []
    for panel in panels:
        rows = [panel.index.index(stamp) for stamp in common_times]
        columns = [panel.assets.index(asset) for asset in common_assets]
        out.append(
            Panel(
                values=panel.values[np.ix_(rows, columns)],
                index=tuple(common_times),
                assets=tuple(common_assets),
                name=name or panel.name,
            )
        )
    return tuple(out)
