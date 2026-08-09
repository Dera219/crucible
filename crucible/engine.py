"""The backtest: target weights in, an honest equity curve out.

Vectorised where it can be, sequential where it must be. Path-dependent quantities — drift,
compounding, turnover against actual holdings — cannot be computed as a matrix expression without
getting them subtly wrong, so the loop over time is explicit and the vectorisation lives inside
each step.

Three details decide whether any number this produces means anything.

## 1. Turnover is measured against DRIFTED weights, not against last period's target

The mistake that quietly inflates costs by 20-40% in most homemade backtests.

You held 5% of the book in an asset. It outperformed, so by the next rebalance it is 5.4% of the
book without your having bought a share. If your new target is 5.4%, your turnover is **zero** —
the market did the rebalancing for you. Comparing target-to-target says you traded 0.4%, charges
you for it, and does so on every asset on every rebalance.

The error runs the other way too: a target equal to the previous target after an adverse move
requires *buying*, and target-to-target reports nothing. So it is not a conservative
approximation; it is noise added to the cost estimate in both directions.

## 2. Execution happens after the signal, never at the same instant

`execution_lag` rows separate a signal from the position it produces. The default of 1 means a
signal computed from the close of *t* is held from *t* through *t+1* — you traded at that close.
That is the academic convention and it is mildly optimistic: you learned the close and traded at
it in the same instant.

`execution_lag=2` is the paranoid version — trade on the following open. Run both. A strategy
whose edge disappears between lag 1 and lag 2 is a strategy whose edge lives entirely in the
close it could not have traded at, which is worth knowing before funding it.

## 3. Delisting is an assumption, not a fact

When a held asset becomes non-investable, this engine assumes you exited at the last known price
— `delisting_return=0.0`. That is **optimistic**, and deliberately visible rather than buried: real
delistings are frequently bankruptcies, and a strategy that systematically buys distressed names
would show none of the loss.

`BacktestResult.delisting_events` counts how often it happened. If that number is large, the
default is doing real work in your result and you should set `delisting_return` to something
punitive and re-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import numpy.typing as npt

from crucible.costs import CostModel, CostReport
from crucible.panel import Panel, PanelError, align

__all__ = ["BacktestResult", "backtest"]

Array = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """What happened, with the assumptions that produced it attached."""

    net_returns: Array
    gross_returns: Array
    equity: Array
    held_weights: Array
    turnover: Array
    costs: CostReport
    index: tuple[datetime | date, ...]
    assets: tuple[str, ...]
    #: Times a held position became non-investable and was assumed liquidated at
    #: `delisting_return`. Large values mean the assumption is carrying your result.
    delisting_events: int
    execution_lag: int

    @property
    def total_return(self) -> float:
        if len(self.equity) < 2 or self.equity[0] == 0:
            return 0.0
        return float(self.equity[-1] / self.equity[0] - 1)

    @property
    def traded(self) -> bool:
        """Did this strategy ever actually place a trade?

        The check Trading_Algorithm never made. A backtest that never traded produces a flat
        line, no errors, and a plausible-looking zero — the failure mode that most resembles
        success.
        """
        return bool(np.nansum(self.turnover) > 0)

    @property
    def annual_turnover(self, periods_per_year: int = 252) -> float:
        if len(self.turnover) == 0:
            return 0.0
        return float(np.nanmean(self.turnover) * periods_per_year)

    def summary(self, periods_per_year: int = 252) -> str:
        if not self.traded:
            return "NO-OP: the strategy never traded. This is a flat line, not a result."

        returns = self.net_returns[np.isfinite(self.net_returns)]
        if len(returns) < 2:
            return "insufficient periods to summarise"

        mean, std = float(np.mean(returns)), float(np.std(returns, ddof=1))
        sharpe = mean / std * np.sqrt(periods_per_year) if std > 0 else 0.0
        peak = np.maximum.accumulate(self.equity)
        drawdown = float(np.min(np.where(peak > 0, self.equity / peak - 1, 0.0)))
        gross_total = float(np.nansum(self.gross_returns))
        cost_total = float(np.nansum(self.costs.total))

        turnover = float(np.nanmean(self.turnover)) * periods_per_year
        lines = [
            f"return {self.total_return:+.2%} | sharpe {sharpe:.2f} | "
            f"maxDD {drawdown:.2%} | turnover {turnover:.1f}x/yr",
            f"  gross {gross_total:+.2%} − costs {cost_total:.2%} "
            f"({cost_total / abs(gross_total):.0%} of gross)"
            if gross_total
            else f"  costs {cost_total:.2%}",
            f"  {self.costs.summary()}",
        ]
        if self.delisting_events:
            lines.append(
                f"  {self.delisting_events} delisting event(s) assumed to exit flat — "
                f"optimistic if any were distressed"
            )
        return "\n".join(lines)


def backtest(
    target_weights: Panel,
    prices: Panel,
    *,
    costs: CostModel | None = None,
    dollar_volume: Panel | None = None,
    volatility: Panel | None = None,
    execution_lag: int = 1,
    delisting_return: float = 0.0,
    initial_equity: float = 1.0,
) -> BacktestResult:
    """Run target weights against realised prices.

    Args:
        target_weights: `(T, N)` desired portfolio weights as fractions of equity. NaN is
            treated as zero — no opinion means no position. Weights on assets that are not
            investable at that timestamp are forced to zero, because a weight you could not
            have held is not a weight.
        prices: `(T, N)` prices, already adjusted for splits and dividends.
        costs: Cost model. `None` runs frictionless, which is a diagnostic and never a result —
            every strategy is profitable if trading is free, so a zero-cost backtest mostly
            measures how often the strategy trades.
        dollar_volume: `(T, N)` for participation. Defaults to effectively infinite liquidity,
            which disables the impact term and should be treated as a known overstatement.
        volatility: `(T, N)` annualised. Defaults to a trailing 20-period estimate from `prices`.
        execution_lag: Rows between a signal and the position it produces. 1 is the convention;
            2 is the paranoid version. Run both.
        delisting_return: Return applied when a held asset becomes non-investable. The 0.0
            default assumes you exited at the last known price, which is optimistic.
        initial_equity: Starting capital.

    Returns:
        A `BacktestResult`. Check `.traded` before believing anything else in it.
    """
    if execution_lag < 1:
        raise PanelError(
            f"execution_lag must be at least 1, got {execution_lag}. Zero means the signal was "
            f"computed and executed at the same instant, which is lookahead at the execution "
            f"boundary and the one thing every backtest gets to be wrong about for free."
        )

    weights_panel, price_panel = align(target_weights, prices)
    if volatility is None:
        volatility_panel = price_panel.pct_change(1).rolling(20, "std") * np.sqrt(252)
    else:
        (volatility_panel,) = align(volatility.select(price_panel.assets))
    if dollar_volume is None:
        # Effectively unlimited liquidity: participation collapses to zero, impact vanishes.
        volume_values = np.full(price_panel.shape, np.inf)
    else:
        (volume_panel,) = align(dollar_volume.select(price_panel.assets))
        volume_values = volume_panel.values

    n_times, n_assets = price_panel.shape
    if n_times <= execution_lag:
        raise PanelError(f"need more than execution_lag={execution_lag} timestamps, got {n_times}")

    asset_returns = price_panel.pct_change(1).values
    investable = price_panel.investable
    targets = np.nan_to_num(weights_panel.shift(execution_lag).values, nan=0.0)
    targets = np.where(investable, targets, 0.0)

    vol_values = np.nan_to_num(volatility_panel.values, nan=0.0)
    price_values = np.nan_to_num(price_panel.values, nan=0.0)

    cost_model = costs or CostModel(spread_bps=0.0, impact_coefficient=0.0)

    equity = np.zeros(n_times)
    gross = np.zeros(n_times)
    net = np.zeros(n_times)
    turnover = np.zeros(n_times)
    held_history = np.zeros((n_times, n_assets))
    spread_cost = np.zeros(n_times)
    impact_cost = np.zeros(n_times)
    commission_cost = np.zeros(n_times)
    participation = np.zeros(n_times)

    held = np.zeros(n_assets)
    capital = float(initial_equity)
    delistings = 0

    # One sequential pass. Costs depend on equity, equity depends on prior costs and on the
    # delisting drag, so nothing here can be computed as a matrix expression without pricing
    # trades against capital the strategy no longer had.
    for t in range(n_times):
        # A held position in something no longer investable is liquidated at the assumed
        # delisting return. This is a REAL return contribution, not a bookkeeping note — an
        # earlier version applied it to a local and then rebuilt the equity curve without it,
        # which made `delisting_return=-1.0` indistinguishable from exiting flat.
        drag = 0.0
        gone = (held != 0) & ~investable[t]
        if gone.any():
            delistings += int(gone.sum())
            drag = float(np.sum(held[gone]) * delisting_return)
            held = np.where(gone, 0.0, held)

        # Rebalance from DRIFTED holdings to target. This difference is the real turnover.
        traded = np.abs(targets[t] - held)
        turnover[t] = float(traded.sum())

        spread, impact, commission, peak = cost_model.charge_row(
            traded_weight=traded,
            equity=capital,
            prices=price_values[t],
            dollar_volume=volume_values[t],
            volatility=vol_values[t],
        )
        spread_cost[t] = spread
        impact_cost[t] = impact
        commission_cost[t] = commission
        participation[t] = peak

        held = targets[t].copy()
        held_history[t] = held

        period_gross = gross[t]
        net[t] = period_gross + drag - (spread + impact + commission)
        capital *= 1.0 + net[t]
        if capital <= 0:
            capital = 0.0
            equity[t:] = 0.0
            break
        equity[t] = capital

        if t + 1 >= n_times:
            break

        period_return = asset_returns[t + 1]
        usable = np.isfinite(period_return)
        gross_next = float(np.sum(held[usable] * period_return[usable]))
        gross[t + 1] = gross_next

        growth = 1.0 + gross_next
        if growth <= 0:
            # Total loss. Compounding a negative equity produces nonsense, not a small number.
            gross[t + 1] = -1.0
            continue

        # Drift: positions grow with their own returns, then re-express as fractions of the new
        # equity. Skipping this is the 20-40% cost error described in the module docstring.
        held = np.where(usable, held * (1.0 + period_return) / growth, held / growth)

    report = CostReport(
        spread=spread_cost,
        impact=impact_cost,
        commission=commission_cost,
        peak_participation=participation,
    )

    return BacktestResult(
        net_returns=net,
        gross_returns=gross,
        equity=equity,
        held_weights=held_history,
        turnover=turnover,
        costs=report,
        index=price_panel.index,
        assets=price_panel.assets,
        delisting_events=delistings,
        execution_lag=execution_lag,
    )
