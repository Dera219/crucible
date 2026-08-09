"""What it costs to trade, modelled the way it actually behaves.

Apex charges a constant spread in basis points plus a linear impact term. That is a reasonable
first approximation and it is wrong in the specific way that matters most: **it is linear in
size**, so a backtest can trade ten times more and pay only ten times more. Real markets do not
work that way, and a strategy that is profitable at $10k and ruinous at $1m looks identical under
a linear model.

## The square-root law

The empirical regularity — Almgren, Kyle, and roughly every practitioner study since — is that
temporary market impact scales with the **square root** of participation:

    impact ≈ η · σ · √(Q / ADV)

where `Q` is the quantity traded, `ADV` the average daily volume, and `σ` the asset's volatility.
The intuition is that liquidity replenishes as you trade, so the marginal cost of the next share
falls, but the total cost still grows superlinearly in the fraction of the day's volume you
consume. Doubling your size does not double your cost — it costs about 1.41x per share, so 2.83x
in total.

That curvature is the whole point. It is what makes capacity a real constraint rather than an
afterthought, and it is why `capacity_notional()` below can answer "at what size does this edge
stop existing" — a question a linear model cannot even be asked.

## The three components

**Spread.** Half the quoted spread, paid on every share, in both directions. Unavoidable if you
cross; and resting orders that never fill are not free either, they just lose money invisibly by
not participating.

**Impact.** The square-root term above. Scales with volatility because a volatile asset's book is
thinner relative to its price moves.

**Commission.** Zero at most US retail brokers now, but kept explicit rather than assumed: a model
with no line for it is a model that silently breaks the day you trade an asset class that has one.

Every default here is pessimistic. A strategy that only survives optimistic fills does not
survive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["CostModel", "CostReport"]

Array = npt.NDArray[np.float64]

#: Fraction of a day's volume above which the square-root law stops being a useful description.
#: Beyond roughly a tenth of ADV you are no longer a price taker sampling a book, you are the
#: reason the price moved, and any model calibrated on ordinary trades is extrapolating.
_PARTICIPATION_WARNING = 0.10


@dataclass(frozen=True, slots=True)
class CostReport:
    """Costs for one rebalance, decomposed. Reported per-period as a fraction of equity."""

    spread: Array
    impact: Array
    commission: Array
    #: Peak fraction of a single asset's ADV consumed at each timestamp. Above ~10% the impact
    #: estimate is extrapolation rather than measurement, and the backtest is quietly fiction.
    peak_participation: Array

    @property
    def total(self) -> Array:
        return self.spread + self.impact + self.commission

    @property
    def strained_periods(self) -> int:
        """Timestamps where the model was pushed past where it is credible."""
        return int((self.peak_participation > _PARTICIPATION_WARNING).sum())

    def summary(self) -> str:
        total = self.total
        parts = [
            f"mean cost {np.nanmean(total) * 10_000:.2f}bps/period",
            f"spread {np.nanmean(self.spread) * 10_000:.2f}",
            f"impact {np.nanmean(self.impact) * 10_000:.2f}",
        ]
        line = " | ".join(parts)
        if self.strained_periods:
            line += (
                f"\n  WARNING: {self.strained_periods} period(s) traded above "
                f"{_PARTICIPATION_WARNING:.0%} of ADV, where the square-root law is "
                f"extrapolation. Treat those returns as unmeasured, not as achieved."
            )
        return line


@dataclass(frozen=True, slots=True)
class CostModel:
    """Spread, square-root impact, and commission.

    Defaults are roughly liquid US equities through a retail broker, and deliberately on the
    expensive side. Measure yours and override; do not assume these are yours.
    """

    #: Full quoted spread in basis points. Half is paid per side.
    spread_bps: float = 3.0
    #: Impact coefficient η. Around 0.5-1.0 in most published calibrations; 1.0 is pessimistic.
    impact_coefficient: float = 1.0
    #: Per-share commission in currency units. Zero at most US retail brokers.
    commission_per_share: float = 0.0
    #: Floor on the volatility used in the impact term, so a suspiciously calm asset cannot
    #: produce a suspiciously free trade. Annualised, like every input to this model.
    min_volatility: float = 0.10
    #: Rebalance periods per year, used to convert annualised volatility into the per-period
    #: volatility the square-root law is stated in.
    #:
    #: This conversion is not cosmetic. The impact formula is calibrated on the volatility of the
    #: interval over which the trade is executed — a day, for daily bars. Feeding it an annualised
    #: number inflates impact by √252 ≈ 16x, which does not merely make results conservative: it
    #: makes every strategy look as though it has no capacity, so the one diagnostic that should
    #: separate a real edge from a small one stops discriminating. Measured before the fix, a
    #: 20bps edge on a $100m-ADV name reported a capacity of $6,400.
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if self.spread_bps < 0:
            raise ValueError(f"spread_bps must be non-negative, got {self.spread_bps}")
        if self.impact_coefficient < 0:
            raise ValueError(
                f"impact_coefficient must be non-negative, got {self.impact_coefficient}"
            )
        if self.commission_per_share < 0:
            raise ValueError(
                f"commission_per_share must be non-negative, got {self.commission_per_share}"
            )
        if self.min_volatility <= 0:
            raise ValueError(f"min_volatility must be positive, got {self.min_volatility}")
        if self.periods_per_year < 1:
            raise ValueError(f"periods_per_year must be >= 1, got {self.periods_per_year}")

    @property
    def _vol_scale(self) -> float:
        """Annualised-to-per-period volatility divisor. See `periods_per_year`."""
        return float(np.sqrt(self.periods_per_year))

    def charge_row(
        self,
        *,
        traded_weight: Array,
        equity: float,
        prices: Array,
        dollar_volume: Array,
        volatility: Array,
    ) -> tuple[float, float, float, float]:
        """Cost of one rebalance at one timestamp, as fractions of equity.

        Returns `(spread, impact, commission, peak_participation)`.

        Per-row rather than batched because the engine needs it sequentially: impact depends on
        absolute notional, notional depends on equity, and equity depends on every cost charged
        before it. Computing the whole matrix in one pass would price every trade against a
        starting equity the strategy no longer had.
        """
        if equity <= 0:
            return 0.0, 0.0, 0.0, 0.0

        traded_notional = np.abs(traded_weight) * equity
        with np.errstate(divide="ignore", invalid="ignore"):
            participation = np.where(dollar_volume > 0, traded_notional / dollar_volume, np.nan)
            shares = np.where(prices > 0, traded_notional / prices, 0.0)

        spread = float(np.nansum(traded_notional * (self.spread_bps / 2 / 10_000)))

        # Participation clipped at 1.0: consuming more than a day's entire volume is outside
        # anything the square-root law was fitted on, so the number beyond it is meaningless
        # rather than merely conservative.
        effective_vol = np.maximum(volatility, self.min_volatility) / self._vol_scale
        clipped = np.clip(np.nan_to_num(participation, nan=0.0), 0.0, 1.0)
        impact = float(
            np.nansum(self.impact_coefficient * effective_vol * np.sqrt(clipped) * traded_notional)
        )
        commission = float(np.nansum(shares * self.commission_per_share))

        finite = participation[np.isfinite(participation)]
        peak = float(finite.max()) if finite.size else 0.0
        return spread / equity, impact / equity, commission / equity, peak

    def charge(
        self,
        *,
        traded_weight: Array,
        equity: Array,
        prices: Array,
        dollar_volume: Array,
        volatility: Array,
    ) -> CostReport:
        """Cost of one rebalance, as a fraction of equity, per timestamp.

        Args:
            traded_weight: `(T, N)` absolute change in portfolio weight per asset. This is
                turnover *after* accounting for drift — see `crucible.engine`, where computing
                it against un-drifted weights is the mistake that inflates costs by 20-40%.
            equity: `(T,)` portfolio value at each rebalance.
            prices: `(T, N)` price per share.
            dollar_volume: `(T, N)` average daily traded notional. The denominator of
                participation, and the number that decides whether a strategy has capacity.
            volatility: `(T, N)` annualised volatility per asset.

        Returns:
            A `CostReport` whose components are fractions of equity per period.
        """
        traded_notional = np.abs(traded_weight) * equity[:, None]

        with np.errstate(divide="ignore", invalid="ignore"):
            participation = np.where(dollar_volume > 0, traded_notional / dollar_volume, np.nan)
            shares = np.where(prices > 0, traded_notional / prices, 0.0)

        half_spread = self.spread_bps / 2 / 10_000
        spread_cost = traded_notional * half_spread

        # The square-root law. Participation is clipped at 1.0 because consuming more than a
        # day's entire volume is outside anything this formula was fitted on — the number it
        # would produce is not conservative, it is meaningless.
        effective_vol = np.maximum(volatility, self.min_volatility) / self._vol_scale
        clipped = np.clip(np.nan_to_num(participation, nan=0.0), 0.0, 1.0)
        impact_cost = self.impact_coefficient * effective_vol * np.sqrt(clipped) * traded_notional

        commission_cost = shares * self.commission_per_share

        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(equity > 0, 1.0 / equity, 0.0)[:, None]

        return CostReport(
            spread=np.nansum(spread_cost * scale, axis=1),
            impact=np.nansum(impact_cost * scale, axis=1),
            commission=np.nansum(commission_cost * scale, axis=1),
            peak_participation=np.nan_to_num(np.nanmax(participation, axis=1), nan=0.0),
        )

    def capacity_notional(
        self,
        *,
        expected_edge_bps: float,
        dollar_volume: float,
        volatility: float,
        turnover_per_period: float = 1.0,
    ) -> float:
        """Largest book size at which the edge still covers its own costs.

        The question a linear cost model cannot be asked. Solves for the notional at which
        round-trip cost equals `expected_edge_bps`:

            spread + η·σ·√(Q / ADV) = edge

        Returns the notional in the same units as `dollar_volume`. A small number here is the
        most useful finding a promising backtest can produce — an edge with no capacity is a
        hobby, and better to learn it now than after building an execution stack for it.
        """
        if dollar_volume <= 0:
            raise ValueError(f"dollar_volume must be positive, got {dollar_volume}")
        if turnover_per_period <= 0:
            raise ValueError(f"turnover_per_period must be positive, got {turnover_per_period}")

        edge = expected_edge_bps / 10_000
        spread = self.spread_bps / 2 / 10_000 * turnover_per_period
        budget = edge - spread
        if budget <= 0:
            return 0.0

        effective_vol = max(volatility, self.min_volatility) / self._vol_scale
        if self.impact_coefficient == 0 or effective_vol == 0:
            return float("inf")

        # budget = η·σ·√(Q/ADV)·turnover  ⇒  Q = ADV·(budget / (η·σ·turnover))²
        root = budget / (self.impact_coefficient * effective_vol * turnover_per_period)
        return float(dollar_volume * root**2)
