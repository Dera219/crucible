"""Which assets were investable, on each date, according to rules fixed in advance.

The most consequential file in the library, and the least interesting to read. Everything
downstream — signals, portfolios, the whole equity curve — is a statement about a universe, and
if the universe is wrong then nothing computed from it means anything, however careful the
arithmetic afterwards.

## The bias this exists to prevent

Take today's list of large-cap US equities, pull ten years of history for those tickers, and
backtest. It is the most natural thing in the world and it is broken in two directions at once:

- **Survivorship.** Companies that went bankrupt, got acquired, or fell out of the index are
  simply absent. The sample is a winners' list, and any strategy that buys equities looks good
  on it.
- **Look-ahead membership.** A company that joined the index in 2024 appears in your 2015 data.
  You are holding it for nine years on the strength of a decision nobody had made yet.

Neither raises an error. Both routinely add several percent a year to a backtest, which is more
than most real edges are worth — so a strategy can be entirely a bias artefact and still look
like a discovery.

The defence is that membership is computed **as of each date**, from a listing calendar and from
screens evaluated on trailing data only. An asset that had not listed, has delisted, or fails
today's screen carries `NaN`, and `crucible.engine` refuses to hold what it cannot see.

## Hysteresis, and why the buffer is not a detail

A universe defined as "the top 500 by dollar volume" churns. Names hovering at rank 500 cross in
and out week after week, and each crossing is a full round trip — bought at one reconstitution,
sold at the next — for reasons that have nothing to do with the strategy. On a 500-name universe
this manufactures tens of percent of annual turnover, and the costs land on your result while the
turnover appears in no signal.

So entry and exit use different thresholds: a name must reach rank 400 to *join* and fall past
rank 600 to *leave*. Inside the band, incumbency wins. This is what real index providers do, for
the same reason.

## What this module cannot fix

If the data has no delisted securities, no amount of careful masking recovers them. Point-in-time
correctness is a property of the *source*, and this module can only avoid adding new bias on top.
`Universe.audit()` reports the symptoms of a survivorship-filtered feed — a suspiciously constant
member count, no exits ever — so at minimum the problem is visible rather than assumed away.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import numpy.typing as npt

from crucible.panel import Panel, PanelError

__all__ = ["Universe", "UniverseAudit", "liquidity_screen", "listing_mask"]

Mask = npt.NDArray[np.bool_]


def listing_mask(
    index: Sequence[datetime | date],
    assets: Sequence[str],
    listings: Mapping[str, tuple[date | None, date | None]],
) -> Mask:
    """Build an investability mask from a listing calendar.

    Args:
        index: Timestamps, ascending.
        assets: Column labels.
        listings: `{symbol: (first_tradable, last_tradable)}`. `None` on either side means
            "unbounded in that direction". A symbol absent from the mapping is treated as
            **never investable** rather than always — silence about an asset is not permission
            to trade it, and defaulting the other way is how an unknown ticker quietly becomes
            a nine-year position.

    Returns:
        `(len(index), len(assets))` boolean mask.
    """
    mask = np.zeros((len(index), len(assets)), dtype=bool)
    stamps = [stamp.date() if isinstance(stamp, datetime) else stamp for stamp in index]

    for column, symbol in enumerate(assets):
        window = listings.get(symbol)
        if window is None:
            continue
        first, last = window
        if first is not None and last is not None and last < first:
            raise PanelError(
                f"{symbol}: delisted {last} before listing {first}. A calendar that runs "
                f"backwards will silently produce an empty universe rather than an error."
            )
        for row, stamp in enumerate(stamps):
            if first is not None and stamp < first:
                continue
            if last is not None and stamp > last:
                continue
            mask[row, column] = True
    return mask


def liquidity_screen(
    dollar_volume: Panel,
    prices: Panel,
    *,
    top_n: int | None = 500,
    min_dollar_volume: float = 1_000_000.0,
    min_price: float = 5.0,
    window: int = 20,
    entry_rank: int | None = None,
    exit_rank: int | None = None,
) -> Mask:
    """Rank-and-threshold assets on trailing liquidity, with hysteresis.

    Every input is a trailing average, never a point observation: a single day's volume spikes on
    news and a universe built from it churns on noise. Every statistic is computed over a window
    ending at the current row, so the screen is causal in the same sense as `ts_*` operators.

    Args:
        dollar_volume: `(T, N)` traded notional per period.
        prices: `(T, N)` prices, for the penny-stock floor.
        top_n: Keep roughly this many names. `None` disables the rank screen and keeps every
            asset clearing the absolute floors.
        min_dollar_volume: Absolute floor on trailing average notional.
        min_price: Absolute floor on price. Sub-$5 names have wide relative spreads, are often
            un-shortable, and are where cross-sectional signals go to produce fictional returns.
        window: Trailing periods for the averages.
        entry_rank: Rank a non-member must reach to join. Defaults to `0.8 * top_n`.
        exit_rank: Rank a member must fall past to leave. Defaults to `1.2 * top_n`.

    Returns:
        `(T, N)` boolean mask.
    """
    if dollar_volume.index != prices.index or dollar_volume.assets != prices.assets:
        raise PanelError("liquidity_screen: dollar_volume and prices must be aligned")
    if top_n is not None and top_n < 1:
        raise PanelError(f"top_n must be >= 1 or None, got {top_n}")

    average_volume = dollar_volume.rolling(window, "mean", min_periods=max(2, window // 2))
    average_price = prices.rolling(window, "mean", min_periods=max(2, window // 2))

    eligible = (
        (average_volume.values >= min_dollar_volume)
        & (average_price.values >= min_price)
        & np.isfinite(dollar_volume.values)
        & np.isfinite(prices.values)
    )

    if top_n is None:
        return eligible

    enter_at = entry_rank if entry_rank is not None else max(1, int(top_n * 0.8))
    leave_at = exit_rank if exit_rank is not None else int(top_n * 1.2)
    if enter_at > leave_at:
        raise PanelError(
            f"entry_rank {enter_at} must not exceed exit_rank {leave_at}: a name would have to "
            f"be more liquid to stay than to join, which inverts the hysteresis and makes the "
            f"churn worse rather than better."
        )

    n_times, n_assets = eligible.shape
    mask = np.zeros_like(eligible)
    member = np.zeros(n_assets, dtype=bool)

    for row in range(n_times):
        volumes = np.where(eligible[row], average_volume.values[row], np.nan)
        valid = np.isfinite(volumes)
        if not valid.any():
            member = np.zeros(n_assets, dtype=bool)
            continue

        # Rank 1 is the most liquid. Ranks are dense over eligible names only.
        order = np.argsort(-volumes[valid], kind="stable")
        ranks = np.full(n_assets, np.inf)
        ranks[np.flatnonzero(valid)[order]] = np.arange(1, int(valid.sum()) + 1)

        joining = (~member) & (ranks <= enter_at)
        leaving = member & (ranks > leave_at)
        member = (member | joining) & ~leaving & eligible[row]
        mask[row] = member

    return mask


@dataclass(frozen=True, slots=True)
class UniverseAudit:
    """Symptoms of a universe that cannot be what it claims to be."""

    mean_members: float
    min_members: int
    max_members: int
    entries: int
    exits: int
    member_count: npt.NDArray[np.int64]

    @property
    def looks_survivorship_filtered(self) -> bool:
        """Did anything ever actually leave?

        Over any multi-year window a real US equity universe loses names constantly to
        bankruptcy, acquisition, and delisting. A universe with no exits is not a stable
        universe; it is a list of survivors, and every backtest run on it is measuring the
        benefit of having known the future.
        """
        return self.exits == 0 and self.mean_members > 0

    @property
    def churn_per_period(self) -> float:
        periods = max(1, len(self.member_count) - 1)
        return (self.entries + self.exits) / periods

    def summary(self) -> str:
        lines = [
            f"members: mean {self.mean_members:.0f}, range [{self.min_members}, "
            f"{self.max_members}] | {self.entries} entries, {self.exits} exits "
            f"({self.churn_per_period:.2f}/period)"
        ]
        if self.looks_survivorship_filtered:
            lines.append(
                "  WARNING: no asset ever left this universe. Over a multi-year window a real "
                "equity universe loses names to bankruptcy, acquisition and delisting "
                "constantly. This is a survivors' list, and results computed on it measure the "
                "benefit of foreknowledge. No masking here can recover what the source omits."
            )
        if self.churn_per_period > 2.0:
            lines.append(
                f"  WARNING: {self.churn_per_period:.1f} membership changes per period. Each is "
                f"a forced round trip unrelated to any signal. Widen the hysteresis band or "
                f"reconstitute less often."
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Universe:
    """A point-in-time investability mask, and the panels restricted to it."""

    mask: Mask
    index: tuple[datetime | date, ...]
    assets: tuple[str, ...]
    name: str = "universe"

    def __post_init__(self) -> None:
        if self.mask.shape != (len(self.index), len(self.assets)):
            raise PanelError(
                f"{self.name}: mask {self.mask.shape} does not match "
                f"{len(self.index)} timestamps x {len(self.assets)} assets"
            )

    @classmethod
    def from_masks(
        cls,
        *masks: Mask,
        index: Sequence[datetime | date],
        assets: Sequence[str],
        name: str = "universe",
    ) -> Universe:
        """Intersect several screens. An asset must pass every one to be investable."""
        if not masks:
            raise PanelError("from_masks needs at least one mask")
        combined = masks[0].copy()
        for extra in masks[1:]:
            if extra.shape != combined.shape:
                raise PanelError(f"mask shapes differ: {combined.shape} vs {extra.shape}")
            combined &= extra
        return cls(mask=combined, index=tuple(index), assets=tuple(assets), name=name)

    def apply(self, panel: Panel) -> Panel:
        """Blank out everything outside the universe, so non-members cannot reach a portfolio.

        Applied to the *signal*, not only to the weights. A non-member that still carries a
        signal value distorts every cross-sectional statistic computed from that row — it takes
        up a rank, shifts the mean, moves the quantile boundaries — even if it is never held.
        """
        if panel.index != self.index or panel.assets != self.assets:
            raise PanelError(
                f"{self.name}: panel {panel.name!r} is not aligned to this universe. Use align()."
            )
        return panel.with_values(np.where(self.mask, panel.values, np.nan))

    def members_on(self, when: datetime | date) -> tuple[str, ...]:
        try:
            row = self.index.index(when)
        except ValueError as exc:
            raise PanelError(f"{self.name}: {when} is not in the index") from exc
        return tuple(a for a, member in zip(self.assets, self.mask[row], strict=True) if member)

    def audit(self) -> UniverseAudit:
        """Check the universe for the shapes a broken one takes. Run it before trusting a result."""
        counts = self.mask.sum(axis=1).astype(np.int64)
        if len(self.mask) < 2:
            return UniverseAudit(
                mean_members=float(counts.mean()) if counts.size else 0.0,
                min_members=int(counts.min()) if counts.size else 0,
                max_members=int(counts.max()) if counts.size else 0,
                entries=0,
                exits=0,
                member_count=counts,
            )
        changes = self.mask[1:].astype(np.int8) - self.mask[:-1].astype(np.int8)
        return UniverseAudit(
            mean_members=float(counts.mean()),
            min_members=int(counts.min()),
            max_members=int(counts.max()),
            entries=int((changes == 1).sum()),
            exits=int((changes == -1).sum()),
            member_count=counts,
        )
