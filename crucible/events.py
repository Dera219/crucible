"""Event studies: judging a claim about discrete events rather than a daily ranking.

Everything else in this package grades a **cross-sectional** signal — how well does a number
rank three thousand names today, does the response grade smoothly across quintiles, what does it
cost to keep re-ranking. That machinery is the right instrument for momentum, value, or
reversal, and it is the wrong instrument for "index funds must sell this spin-off next Tuesday
regardless of price."

An event claim has a different shape. There are forty of them a year, not three thousand a day.
Each is an independent bet. Nothing is held most of the time. Asking for an information
coefficient across a cross-section where 3,698 of 3,700 weights are zero produces a number that
is degenerate rather than merely small, and quintile monotonicity across two positions means
nothing at all. Relaxing the cross-sectional thresholds until an event strategy passes is worse
than useless: it removes the discipline without replacing it.

So this module supplies the instrument that actually judges the claim — abnormal returns aligned
in event time, cumulated per event, and a t-statistic computed **across events** rather than
across days.

## Why abnormal rather than raw

A spin-off distributed in March 2020 fell because everything fell. Subtracting a benchmark is
what separates "this event moved the stock" from "the market moved that week", and without it an
event sample clustered in a drawdown looks like a discovery.

The default is market-adjusted: subtract the equal-weighted mean return of every investable
asset that day. It needs no estimation window, cannot be destabilised by a short or noisy
pre-event history, and does not silently assume a beta that was estimated on data the event
itself contaminated.

## The failure this module is built to expose

Events cluster. Russell reconstitution happens on one day; spin-offs bunch after earnings
season; bankruptcies arrive together because recessions cause them together. A t-statistic
across events assumes the events are independent, and clustered events are emphatically not —
forty events sharing one date are closer to one observation than to forty.

`EventStudy.clustering` measures this and `evaluate` refuses on it, because it is the specific
way an event study manufactures significance from nothing, and it is invisible in the headline
number.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from crucible.panel import Panel

Array = npt.NDArray[np.float64]

#: Below this many events, a cross-sectional t-statistic is not measuring anything. Chosen to be
#: uncomfortable: thirty is already thin for a claim about a decade.
MIN_USABLE_EVENTS = 30


class EventError(ValueError):
    """The event set or the window is malformed."""


@dataclass(frozen=True, slots=True)
class Event:
    """One occurrence: which asset, and the date the market learns.

    `date` is the date the information becomes actionable, not the date the filing was signed or
    the corporate action was announced internally. Getting this wrong shifts the whole study by
    the length of the mistake and usually manufactures a result, because the drift being measured
    is exactly the drift around the true date.
    """

    asset: str
    date: date | datetime

    def __post_init__(self) -> None:
        if not self.asset:
            raise EventError("event asset must be named")


@dataclass(frozen=True, slots=True)
class EventStudy:
    """Abnormal returns aligned in event time, plus the statistics that judge them.

    Attributes:
        abnormal: `(n_events, window)` abnormal returns. NaN where data was unavailable.
        offsets: Event-time offsets, e.g. -5..+20, aligned with `abnormal`'s columns.
        assets: The asset for each row.
        dates: The event date for each row, as an index position into the source panel.
        dropped: Events that could not be placed — unknown asset, date outside the panel, or a
            window that would run off the end. Counted rather than silently discarded, because a
            study that quietly kept only the events with complete data has selected on survival.
    """

    abnormal: Array
    offsets: npt.NDArray[np.int64]
    assets: tuple[str, ...]
    dates: tuple[int, ...]
    dropped: int

    @property
    def n_events(self) -> int:
        return int(self.abnormal.shape[0])

    @property
    def car(self) -> Array:
        """Cumulative abnormal return per event, summed over the whole window.

        Summed rather than compounded: over a window of days these agree to a rounding error,
        and the sum is what the t-statistic below is defined on.
        """
        with np.errstate(invalid="ignore"):
            return np.nansum(self.abnormal, axis=1)

    def car_over(self, start: int, end: int) -> Array:
        """Cumulative abnormal return between two event-time offsets, inclusive.

        The window a strategy could actually trade is rarely the window worth plotting — the
        pre-event days are there to show whether the drift began before the date you think the
        information arrived, which is the cheapest available test that the date is right.
        """
        if start > end:
            raise EventError(f"start offset {start} is after end offset {end}")
        mask = (self.offsets >= start) & (self.offsets <= end)
        if not mask.any():
            raise EventError(f"no offsets in [{start}, {end}]; window is {self.window}")
        with np.errstate(invalid="ignore"):
            return np.nansum(self.abnormal[:, mask], axis=1)

    @property
    def window(self) -> tuple[int, int]:
        return (int(self.offsets[0]), int(self.offsets[-1]))

    @property
    def mean_car(self) -> float:
        finite = self.car[np.isfinite(self.car)]
        return float(finite.mean()) if finite.size else float("nan")

    @property
    def t_statistic(self) -> float:
        """Cross-sectional t across events.

        This is the number an event study lives or dies by, and it is only as honest as the
        independence of the events — see `clustering`.
        """
        finite = self.car[np.isfinite(self.car)]
        if finite.size < 2:
            return float("nan")
        deviation = finite.std(ddof=1)
        if deviation == 0.0:
            return float("nan")
        return float(finite.mean() / (deviation / math.sqrt(finite.size)))

    @property
    def win_rate(self) -> float:
        """Fraction of events with a positive CAR.

        Reported next to the mean because they disagree in the case that matters: a mean carried
        by two enormous winners out of forty is a lottery ticket wearing the costume of an edge.
        """
        finite = self.car[np.isfinite(self.car)]
        return float((finite > 0).mean()) if finite.size else float("nan")

    @property
    def clustering(self) -> float:
        """Fraction of events that share their date with at least one other event, in [0, 1].

        The t-statistic assumes independent observations. Events on the same day share whatever
        the market did that day, so a study of forty events that all fall on Russell
        reconstitution day has closer to one observation than forty — and the t-statistic has no
        way of knowing.
        """
        if not self.dates:
            return 0.0
        _, counts = np.unique(np.asarray(self.dates), return_counts=True)
        shared = int(counts[counts > 1].sum())
        return shared / len(self.dates)

    @property
    def average_path(self) -> Array:
        """Mean abnormal return at each event-time offset. The picture, not the number.

        Worth looking at before trusting any headline: drift that starts well before offset zero
        means the event date is wrong or the information leaked, and drift that reverses
        immediately after means the strategy is collecting a bid-ask bounce.
        """
        with np.errstate(invalid="ignore"):
            return np.nanmean(self.abnormal, axis=0)


def _index_positions(panel: Panel) -> dict[date | datetime, int]:
    return {stamp: position for position, stamp in enumerate(panel.index)}


def market_adjusted(returns: Panel) -> Panel:
    """Subtract the equal-weighted mean return of the investable universe each day.

    Equal-weighted rather than cap-weighted on purpose. A cap-weighted benchmark is dominated by
    the largest names, and the events this module exists for happen to small ones — subtracting a
    mega-cap index from a micro-cap event return removes the wrong thing and calls the residual
    abnormal.
    """
    with np.errstate(invalid="ignore"):
        market = np.nanmean(returns.values, axis=1, keepdims=True)
    return returns.with_values(returns.values - market, name=f"{returns.name}_abnormal")


def event_study(
    returns: Panel,
    events: Sequence[Event],
    *,
    pre: int = 5,
    post: int = 20,
    benchmark: Panel | None = None,
) -> EventStudy:
    """Align abnormal returns around each event.

    Args:
        returns: Per-period simple returns, assets in columns.
        events: The occurrences. Duplicates are kept — if the same asset has two events, both
            are real observations.
        pre: Periods before the event to include. Present so the pre-event path can be read;
            drift beginning before offset zero is the signature of a wrong date.
        post: Periods after the event, and the part a strategy could trade.
        benchmark: Abnormal returns to use instead of the market-adjusted default. Supply this
            when you have a defensible expected-return model; the default assumes only that the
            average investable asset is the right thing to subtract.

    Returns:
        An `EventStudy`. Events whose window would run off either end of the panel are dropped
        and counted, never truncated — a truncated window is a different measurement wearing the
        same name.
    """
    if pre < 0 or post < 0:
        raise EventError(f"pre and post must be non-negative, got pre={pre}, post={post}")
    if pre + post == 0:
        raise EventError("a window of zero periods measures nothing")

    abnormal_panel = benchmark if benchmark is not None else market_adjusted(returns)
    if abnormal_panel.shape != returns.shape:
        raise EventError(
            f"benchmark shape {abnormal_panel.shape} does not match returns {returns.shape}"
        )

    positions = _index_positions(returns)
    columns = {asset: column for column, asset in enumerate(returns.assets)}
    offsets = np.arange(-pre, post + 1, dtype=np.int64)

    rows: list[Array] = []
    kept_assets: list[str] = []
    kept_dates: list[int] = []
    dropped = 0

    for event in events:
        column = columns.get(event.asset)
        position = positions.get(event.date)
        if column is None or position is None:
            dropped += 1
            continue
        start = position - pre
        stop = position + post + 1
        if start < 0 or stop > returns.n_times:
            dropped += 1
            continue
        rows.append(abnormal_panel.values[start:stop, column])
        kept_assets.append(event.asset)
        kept_dates.append(position)

    if rows:
        stacked = np.vstack(rows).astype(np.float64)
    else:
        stacked = np.empty((0, offsets.size), dtype=np.float64)

    return EventStudy(
        abnormal=stacked,
        offsets=offsets,
        assets=tuple(kept_assets),
        dates=tuple(kept_dates),
        dropped=dropped,
    )


@dataclass(frozen=True, slots=True)
class EventKillCriteria:
    """The numbers at which an event claim is abandoned, fixed before the run.

    Deliberately not reusing `KillCriteria`. Its fields describe a cross-sectional signal —
    information coefficient, quantile monotonicity, annual turnover — and none of them is
    computable for a claim about forty discrete events. Borrowing the type and setting the
    inapplicable fields to zero would look like discipline while removing it.
    """

    #: Fewer events than this and the t-statistic is decoration.
    min_events: int = MIN_USABLE_EVENTS
    #: Minimum mean cumulative abnormal return over the traded window.
    min_mean_car: float = 0.0
    #: Minimum |t| across events. Same bar as the cross-sectional criteria, same reason.
    min_t_statistic: float = 2.0
    #: Minimum fraction of events that were individually positive. Guards against a mean carried
    #: by two outliers, which is the event-study version of a signal driven by one tail.
    min_win_rate: float = 0.5
    #: Maximum fraction of events allowed to share a date with another event. Above this the
    #: t-statistic is counting one market day many times.
    max_clustering: float = 0.5
    #: Maximum fraction of supplied events that may be dropped for missing data. A study that
    #: silently kept only the events with complete history has selected on survival.
    max_dropped_fraction: float = 0.2
    #: Must the result survive deflation against the honest lifetime trial count?
    require_deflation_survival: bool = True
    #: Maximum free parameters. Present so an `EventKillCriteria` can stand where a
    #: `KillCriteria` does in `Hypothesis`, and for the same reason: each one is an axis to
    #: overfit along.
    max_parameters: int = 3

    def __post_init__(self) -> None:
        if self.min_events < 2:
            raise ValueError(f"min_events must be at least 2, got {self.min_events}")
        if not 0.0 <= self.min_win_rate <= 1.0:
            raise ValueError(f"min_win_rate must lie in [0, 1], got {self.min_win_rate}")
        if not 0.0 <= self.max_clustering <= 1.0:
            raise ValueError(f"max_clustering must lie in [0, 1], got {self.max_clustering}")
        if not 0.0 <= self.max_dropped_fraction <= 1.0:
            raise ValueError(
                f"max_dropped_fraction must lie in [0, 1], got {self.max_dropped_fraction}"
            )
        if self.min_t_statistic < 0:
            raise ValueError(f"min_t_statistic must be non-negative, got {self.min_t_statistic}")

    def to_json(self) -> dict[str, Any]:
        """Serialised for the preregistration hash.

        Required by `Hypothesis`, which hashes its criteria so that moving a bar after seeing a
        result changes the hash and is therefore visible. Criteria that cannot be serialised
        cannot be committed to, which would defeat the point of registering them.
        """
        return {
            "min_events": self.min_events,
            "min_mean_car": self.min_mean_car,
            "min_t_statistic": self.min_t_statistic,
            "min_win_rate": self.min_win_rate,
            "max_clustering": self.max_clustering,
            "max_dropped_fraction": self.max_dropped_fraction,
            "require_deflation_survival": self.require_deflation_survival,
            "max_parameters": self.max_parameters,
        }


@dataclass(frozen=True, slots=True)
class EventVerdict:
    """Whether the claim survived, and every reason it did not."""

    survived: bool
    reasons: tuple[str, ...]
    mean_car: float
    t_statistic: float
    win_rate: float
    n_events: int
    clustering: float

    def __str__(self) -> str:
        headline = "SURVIVED" if self.survived else "KILLED"
        body = "\n".join(f"  - {reason}" for reason in self.reasons)
        summary = (
            f"{headline}: {self.n_events} events, "
            f"CAR {self.mean_car:+.4f}, t {self.t_statistic:+.2f}"
        )
        return summary + (f"\n{body}" if body else "")


def evaluate(
    study: EventStudy,
    criteria: EventKillCriteria,
    *,
    traded_window: tuple[int, int] | None = None,
) -> EventVerdict:
    """Judge an event study against criteria fixed in advance.

    Args:
        study: The aligned abnormal returns.
        criteria: The bars, written before the run.
        traded_window: Offsets a strategy would actually hold, inclusive. Defaults to offset 0
            through the end of the window — the pre-event days exist to be looked at, not to be
            claimed as profit, and including them in the headline is how a study takes credit for
            drift it could not have traded.
    """
    low, high = traded_window if traded_window is not None else (0, int(study.offsets[-1]))
    reasons: list[str] = []

    if study.n_events == 0:
        return EventVerdict(
            survived=False,
            reasons=("no events were placed in the panel at all",),
            mean_car=float("nan"),
            t_statistic=float("nan"),
            win_rate=float("nan"),
            n_events=0,
            clustering=0.0,
        )

    car = study.car_over(low, high)
    finite = car[np.isfinite(car)]
    mean_car = float(finite.mean()) if finite.size else float("nan")
    win_rate = float((finite > 0).mean()) if finite.size else float("nan")
    if finite.size >= 2 and finite.std(ddof=1) > 0:
        t_statistic = float(finite.mean() / (finite.std(ddof=1) / math.sqrt(finite.size)))
    else:
        t_statistic = float("nan")

    supplied = study.n_events + study.dropped
    dropped_fraction = study.dropped / supplied if supplied else 0.0

    if study.n_events < criteria.min_events:
        reasons.append(
            f"{study.n_events} events is below the minimum {criteria.min_events}; "
            f"a t-statistic on this many is decoration"
        )
    if not math.isfinite(mean_car) or mean_car < criteria.min_mean_car:
        reasons.append(f"mean CAR {mean_car:+.4f} is below {criteria.min_mean_car:+.4f}")
    if not math.isfinite(t_statistic) or abs(t_statistic) < criteria.min_t_statistic:
        reasons.append(f"|t| {abs(t_statistic):.2f} is below {criteria.min_t_statistic:.2f}")
    if not math.isfinite(win_rate) or win_rate < criteria.min_win_rate:
        reasons.append(
            f"win rate {win_rate:.2f} is below {criteria.min_win_rate:.2f}; "
            f"the mean is being carried by a few events"
        )
    if study.clustering > criteria.max_clustering:
        reasons.append(
            f"{study.clustering:.0%} of events share a date with another (limit "
            f"{criteria.max_clustering:.0%}); the t-statistic is counting one market day "
            f"many times"
        )
    if dropped_fraction > criteria.max_dropped_fraction:
        reasons.append(
            f"{dropped_fraction:.0%} of supplied events were dropped for missing data "
            f"(limit {criteria.max_dropped_fraction:.0%}); the survivors are a selected sample"
        )

    return EventVerdict(
        survived=not reasons,
        reasons=tuple(reasons),
        mean_car=mean_car,
        t_statistic=t_statistic,
        win_rate=win_rate,
        n_events=study.n_events,
        clustering=study.clustering,
    )
