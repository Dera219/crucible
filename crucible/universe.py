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

So entry and exit use different thresholds: on a top-500 screen, a name must reach rank 300 to
*join* and fall past rank 800 to *leave*. Inside the band, incumbency wins. This is what real
index providers do, for the same reason. The band's width was set by measuring churn on the real
CRSP extract — the numbers are in `liquidity_screen`'s docstring — after the original, narrower
default was found to manufacture 37.7% of the book per year in forced round trips.

## Screens compose, and one of them has to read a different panel

Membership is an intersection of masks — a listing calendar, a liquidity rank, a price floor —
and `Universe.from_masks` takes as many as there are. `price_floor_screen` is the odd one out:
every other screen reads the same adjusted price panel the engine does, and that one must read
the vendor's raw, unadjusted panel instead, because a "$5 line" drawn on a total-return index
drifts away from five dollars with every split and dividend. Measured on the CRSP extract, below
$5 on the tape US common stock returned **-0.64% annualised with a median 21-day return of
-4.00%** over 2015-2025, survivorship-free; at or above it, +9.85%. Removing that segment is
hygiene and not an edge, and the function's docstring is at pains to say so.

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

__all__ = [
    "Universe",
    "UniverseAudit",
    "liquidity_screen",
    "listing_mask",
    "price_floor_screen",
]

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
        min_price: Absolute floor on the trailing average of whatever price panel is handed in.
            Sub-$5 names have wide relative spreads, are often un-shortable, and are where
            cross-sectional signals go to produce fictional returns. This floor is the coarse
            one and is not the load-bearing one: the panel a backtest passes here is normally
            `Dataset.prices`, which is total-return adjusted, so the line drifts with every
            split and dividend and stops meaning the number written in the argument. Compose
            `price_floor_screen`, which reads `Dataset.raw_prices` and refuses to run without
            it, for a floor that means dollars. The measurements are in its docstring.
        window: Trailing periods for the averages.
        entry_rank: Rank a non-member must reach to join. Defaults to `0.6 * top_n`.
        exit_rank: Rank a member must fall past to leave. Defaults to `1.6 * top_n`.

    The default band is sized empirically, not by taste. On the real CRSP extract — top-1000
    over ~5,000 investable names, daily reconstitution, 2015-2025 — churn against band width:

        enter/leave     mean members   churn/period   forced round trips
        800 / 1200          997            3.34          37.7% of book/yr
        700 / 1400         1038            2.05          20.2%
        600 / 1600         1026            1.50          13.4%   <- default
        500 / 2000          974            1.14           9.5%

    The original 0.8x/1.2x default manufactured 37.7% of the book per year in round trips no
    signal asked for. 0.6x/1.6x cuts that by two-thirds while mean membership stays on target;
    going wider still buys little and lets members ride down to rank 2000, at which point a
    "top-1000" universe is holding names nowhere near the top 1000.

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

    enter_at = entry_rank if entry_rank is not None else max(1, int(top_n * 0.6))
    leave_at = exit_rank if exit_rank is not None else int(top_n * 1.6)
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


def price_floor_screen(
    raw_prices: Panel | None,
    *,
    min_price: float = 5.0,
    entry_price: float | None = None,
) -> Mask:
    """Drop names trading below a floor on their RAW, unadjusted price.

    ## Why the panel this reads is not the panel everything else reads

    `Dataset.prices` is total-return *adjusted*: it is compounded forward from each security's
    first observed price, so its level is an index, not a price. A "$5 line" drawn on it is a
    line on that index — it drifts with every dividend, split and reverse split, and after a few
    years it has stopped meaning "five dollars" without anything having gone visibly wrong.

    Measured on the verified 2015-2025 CRSP extract, over 7,698,163 liquid name-days, the
    adjusted level runs at a median of 1.00x the raw price but a p99 of 3.16x and a maximum of
    70x — the splits. A $5 line on the adjusted panel therefore disagrees with the tape on
    195,447 name-days across 1,556 securities (2.54%), in both directions:

        16,874 name-days   below $5 on the tape, at or above the adjusted line — admitted
                           by an adjusted floor. These are the names the floor exists to remove.
       178,573 name-days   at or above $5 on the tape, below the adjusted line — excluded
                           by an adjusted floor. Mostly reverse splits, which is the corporate
                           action failing companies use to hold a listing: the tape jumps 1:10
                           and the return index does not move at all.

    `Dataset.raw_prices` is the vendor's unadjusted series and exists for exactly this; its own
    comment says never to compute returns from it. It is `None` for loaders that cannot supply
    one, and this function raises rather than quietly using `prices` instead. A screen that
    silently measures the wrong thing is worse than no screen: it is reported as applied, it
    shows up in the audit, and it removes nothing it was asked to remove.

    ## The evidence for the floor

    Forward 21-session returns by raw price bucket, on the same extract (10,586,496 common-stock
    name-days, 6,635 securities, 2,766 sessions), restricted to sessions turning over more than
    $500k and computed survivorship-free — a security that disappears is liquidated at CRSP's own
    delisting return and held flat afterwards, so the 6,068 recorded delistings (mean -4.7%, 331
    worse than -50%) are inside these numbers rather than missing from them:

        raw price        obs     annualised   median 21d   delisted within 21d
        <$1          117,933       -8.75%       -9.47%           3.05%
        $1-2         196,171       +0.55%       -4.80%           0.51%
        $2-5         529,145       +0.80%       -2.83%           0.46%
        $5-10        785,871       +6.50%       -0.31%           0.51%
        $10-30     2,295,885      +10.39%       +0.35%           0.56%
        $30-100    2,682,042       +9.92%       +0.69%           0.39%
        >$100      1,041,367      +11.04%       +0.84%           0.24%

    Monotone in price, and the shape of the cheap buckets is the whole argument. The median
    sub-$1 name loses 9.5% over the next month while the bucket's mean loses 0.8%: most of them
    bleed and a few multiply, which is the lottery-preference signature, and a cross-sectional
    book holding the bucket collects the median far more often than it collects the mean. They
    also delist inside the next month at roughly six times the rate of everything else.

    **$5 is the default because it is where the sign changes**, not because it is a round number:
    +0.80% annualised below it against +6.50% just above, and a median that goes from -2.83% to
    approximately flat. It is also the line most institutional mandates and marginability rules
    already draw, so a universe built to it is comparable with published work. In aggregate the
    floor removes 843,249 of 7,648,414 liquid name-days — 11.0% of breadth, returning **-0.64%
    annualised with a median 21-day return of -4.00%** — and keeps 88.97%, returning +9.85%.

    A $10 floor would look better still, and is refused on purpose. It would cut a further 10.3%
    of breadth out of a bucket earning +6.50% a year; removing names *because they are cheap* at
    that point is a factor tilt, and a tilt belongs in a signal and a registered hypothesis where
    it is deflated for having been chosen, not smuggled into the universe definition where
    nothing scores it.

    The effect has not decayed. Splitting at 2020-10-06, annualised by bucket:

        bucket    2015-2020    2020-2025
        <$1         +1.45%      -13.78%
        $1-2        +2.20%       -0.34%
        $2-5        +9.02%       -4.70%
        $5-10      +10.39%       +3.33%
        $10-30     +10.68%      +10.04%

    Every cheap bucket is *worse* in the second half while the rest of the market is flat, so
    this is a live characteristic rather than a fact about 2015 that has since been arbitraged.

    One caveat, stated because it was checked rather than assumed: the extract these were measured
    on carries no `DlyPrcFlg`, so the quote-only and non-trading sessions that
    `CIZ_NON_TRADED_PRICE_FLAGS` withdraws could not be marked. The $500k restriction removes them
    anyway — such a session is one where nothing traded, and of the 7,710,527 sessions clearing
    $500k, **zero** have null or zero share volume, against 0.85% of the extract overall.

    ## Why entry sits above exit

    A bare floor is a threshold, and this module already knows what thresholds do to a universe:
    names oscillating across the line round-trip the book for reasons no signal asked for. On the
    same extract, membership flips per year against the entry threshold, floor held at $5:

        floor / entry    mean members    flips/yr    forced round trips
        $5.00 / $5.00        2,479         3,033        61.2% of book/yr
        $5.00 / $5.50        2,465         1,529        31.0%
        $5.00 / $6.00        2,453         1,123        22.9%   <- default (1.2x)
        $5.00 / $7.50        2,417           695        14.4%
        $5.00 / $10.00       2,354           510        10.8%

    A bare $5 line manufactures 61.2% of the book per year in round trips — worse than the 37.7%
    that made `liquidity_screen`'s rank band too narrow to keep. Requiring $6.00 to join cuts
    that by nearly two-thirds and costs 1.0% of mean membership, and the slice it defers is the
    weakest one above the floor: the $5-6 zone returns +2.57% annualised with a -1.58% median,
    against +9.85% for everything at or above $5. Wider buffers keep helping and start deleting
    names that are genuinely fine, which is the tilt this function just refused to make.

    Exit is at the floor itself and is *not* buffered, deliberately: the asymmetry is the point.
    A name falling through $5 is entering the segment measured above, and every extra session it
    is held there is a session inside a -4.00% median. Entry is where patience is cheap; exit is
    where it is expensive.

    ## What this is not

    It is not an edge, and no part of it should be reported as one. It removes a segment that
    destroyed value over eleven years; removing something bad is hygiene, and hygiene raises the
    floor of a result without adding anything to its top. Any strategy that looks profitable
    *because* this screen was applied is a strategy whose profit was a short position in the
    penny-stock complex, which is expensive to borrow, frequently un-shortable, and not what the
    backtest claims to be doing. The honest description of this function is that it stops a
    cross-sectional signal from spending its breadth on names whose prices are not really prices.

    Point observations are used, not trailing averages — the reverse of `liquidity_screen`'s
    rule, and for a stated reason. Volume spikes on news and needs smoothing; a price does not
    spike, it moves, and a name that has crashed from $20 to $2 is a $2 stock today. A 20-day
    trailing mean would hold it in the universe for roughly ten more sessions, which is exactly
    the window the table above prices at -9.47%. Churn is handled by the entry buffer instead,
    which does not delay any exit.

    Args:
        raw_prices: `(T, N)` unadjusted prices — `Dataset.raw_prices`, never `Dataset.prices`.
            Typed as optional so `Dataset.raw_prices` can be passed straight through: `None`
            raises here rather than tempting the caller into the adjusted panel.
        min_price: The floor, and the exit threshold. A name is dropped on any session its raw
            price is strictly below this. Inclusive at the boundary: exactly $5.00 clears a
            $5.00 floor, matching `liquidity_screen`'s `>=`.
        entry_price: What a non-member must reach to join. Defaults to `1.2 * min_price`. Pass
            `min_price` itself for a bare floor with no hysteresis, accepting the churn above.

    Returns:
        `(T, N)` boolean mask. Sessions with no raw price at all are `False` — a missing price is
        not evidence of clearing the floor — but they do not change membership either way, so a
        halt does not eject a name that was fine before it and is fine after it.
    """
    if raw_prices is None:
        raise PanelError(
            "price_floor_screen: no raw price panel. Dataset.raw_prices is None for sources that "
            "supply no unadjusted series (dataset_from_panels, load_crsp_csv); load_crsp_ciz "
            "supplies one. Falling back to Dataset.prices is not offered: that panel is a "
            "total-return index compounded from an arbitrary base, so the floor would drift with "
            "every split and dividend and would disagree with the tape on 2.54% of liquid "
            "name-days. A screen that silently measures the wrong thing is worse than no screen, "
            "because it is reported as applied and removes nothing."
        )
    if not np.isfinite(min_price) or min_price < 0:
        raise PanelError(
            f"min_price must be finite and >= 0, got {min_price}. A negative floor admits "
            f"exactly the rows a price panel uses to mean 'no trade happened'."
        )

    enter_at = entry_price if entry_price is not None else min_price * 1.2
    if not np.isfinite(enter_at) or enter_at < min_price:
        raise PanelError(
            f"entry_price {enter_at} must be finite and >= min_price {min_price}: a name would "
            f"have to be cheaper to join than to stay, which inverts the hysteresis and makes "
            f"boundary churn worse rather than better."
        )

    price = raw_prices.values
    priced = np.isfinite(price)
    # NaN compares False both ways, so a missing price neither admits nor ejects. It is masked
    # out below on its own session and leaves membership untouched.
    joining = price >= enter_at
    leaving = price < min_price

    mask = np.zeros(price.shape, dtype=bool)
    member = np.zeros(raw_prices.n_assets, dtype=bool)
    for row in range(raw_prices.n_times):
        member = (member | joining[row]) & ~leaving[row]
        mask[row] = member & priced[row]
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
