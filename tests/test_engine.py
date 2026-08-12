"""The backtest engine and the cost model.

Three properties carry every number this produces: turnover is measured against drifted weights,
the delisting assumption actually affects the result, and the equity curve reconciles with the
returns series. All three were wrong in the first draft and all three failed silently.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from crucible.costs import CostModel
from crucible.engine import backtest
from crucible.ops import cs_demean, cs_rank, cs_scale
from crucible.panel import Panel, PanelError


def mk(values: np.ndarray, name: str = "close") -> Panel:
    n_times, n_assets = values.shape
    return Panel(
        values=values,
        index=tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(n_times)),
        assets=tuple(f"A{i}" for i in range(n_assets)),
        name=name,
    )


def random_prices(n_times: int = 200, n_assets: int = 20, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    return mk(np.cumprod(1 + rng.normal(0.0003, 0.014, (n_times, n_assets)), axis=0) * 100)


class TestDrift:
    """The mistake that inflates costs by 20-40% in most homemade backtests."""

    def test_a_single_asset_at_full_weight_never_trades_again(self) -> None:
        """Drift keeps a 100% position at 100%. Any turnover here is invented."""
        rng = np.random.default_rng(0)
        prices = mk(np.cumprod(1 + rng.normal(0.001, 0.02, (50, 1)), axis=0) * 100)

        result = backtest(prices.with_values(np.ones(prices.shape)), prices, costs=CostModel())

        assert result.turnover[2:].sum() == pytest.approx(0.0, abs=1e-12)

    def test_a_target_that_tracks_drift_costs_nothing(self) -> None:
        """The market rebalanced for you. Target-to-target differencing would charge for it."""
        values = np.ones((40, 2))
        values[:, 0] = np.cumprod(np.full(40, 1.02))
        values[:, 1] = np.cumprod(np.full(40, 0.98))
        prices = mk(values)

        targets = np.zeros((40, 2))
        held = np.array([0.5, 0.5])
        for t in range(40):
            targets[t] = held
            if t + 1 < 40:
                returns = (values[t + 1] - values[t]) / values[t]
                growth = 1 + (held * returns).sum()
                held = held * (1 + returns) / growth

        # At execution_lag=1 the target for period t is used as-is: no compensating shift.
        result = backtest(prices.with_values(targets), prices, costs=CostModel())
        naive = float(np.abs(np.diff(targets, axis=0, prepend=targets[:1])).sum())

        assert result.turnover[1:].sum() == pytest.approx(0.0, abs=1e-9)
        assert naive > 0.5  # what the wrong method would have charged for

    def test_an_equal_weight_target_does_require_rebalancing(self) -> None:
        """The converse, so the test above is not passing for the wrong reason: holding a
        constant equal-weight target genuinely means trading as prices diverge."""
        values = np.ones((40, 2))
        values[:, 0] = np.cumprod(np.full(40, 1.02))
        values[:, 1] = np.cumprod(np.full(40, 0.98))
        prices = mk(values)

        result = backtest(prices.with_values(np.full((40, 2), 0.5)), prices, costs=CostModel())

        assert result.turnover[2:].sum() > 0.1


class TestAccounting:
    def test_the_equity_curve_reconciles_with_the_returns(self) -> None:
        prices = random_prices()
        weights = cs_scale(cs_demean(cs_rank(prices.pct_change(20))))

        result = backtest(weights, prices, costs=CostModel())

        assert result.equity[-1] == pytest.approx(float(np.prod(1 + result.net_returns)), rel=1e-9)

    def test_net_equals_gross_minus_costs(self) -> None:
        prices = random_prices()
        weights = cs_scale(cs_demean(cs_rank(prices.pct_change(20))))

        result = backtest(weights, prices, costs=CostModel(spread_bps=5.0))

        np.testing.assert_allclose(
            result.net_returns, result.gross_returns - result.costs.total, atol=1e-12
        )

    def test_costs_reduce_the_result(self) -> None:
        prices = random_prices()
        weights = cs_scale(cs_demean(cs_rank(prices.pct_change(20))))

        free = backtest(weights, prices)
        paid = backtest(weights, prices, costs=CostModel(spread_bps=10.0))

        assert paid.total_return < free.total_return

    def test_a_weight_on_a_non_investable_asset_is_forced_to_zero(self) -> None:
        """A weight you could not have held is not a weight."""
        values = np.full((20, 2), 100.0)
        values[:, 1] = np.nan
        prices = mk(values)

        result = backtest(prices.with_values(np.full((20, 2), 0.5)), prices, costs=CostModel())

        assert np.all(result.held_weights[:, 1] == 0.0)


class TestDelisting:
    def test_the_assumption_actually_affects_the_result(self) -> None:
        """It did not, in the first draft: the drag was applied to a local and then discarded
        when the equity curve was rebuilt from the returns series."""
        values = np.full((20, 3), 100.0)
        values[10:, 2] = np.nan
        prices = mk(values)
        weights = prices.with_values(np.full((20, 3), 1 / 3))

        flat = backtest(weights, prices, costs=CostModel())
        wiped = backtest(weights, prices, costs=CostModel(), delisting_return=-1.0)

        assert flat.total_return - wiped.total_return == pytest.approx(1 / 3, abs=0.01)

    def test_delisting_events_are_counted(self) -> None:
        values = np.full((20, 3), 100.0)
        values[10:, 2] = np.nan
        prices = mk(values)

        result = backtest(prices.with_values(np.full((20, 3), 1 / 3)), prices, costs=CostModel())

        assert result.delisting_events == 1

    def test_a_haircut_lands_between_the_extremes(self) -> None:
        values = np.full((20, 3), 100.0)
        values[10:, 2] = np.nan
        prices = mk(values)
        weights = prices.with_values(np.full((20, 3), 1 / 3))

        flat = backtest(weights, prices, costs=CostModel())
        half = backtest(weights, prices, costs=CostModel(), delisting_return=-0.5)
        wiped = backtest(weights, prices, costs=CostModel(), delisting_return=-1.0)

        assert wiped.total_return < half.total_return < flat.total_return


class TestExecutionLag:
    def test_zero_lag_is_refused(self) -> None:
        """Signal and execution at the same instant is lookahead at the execution boundary."""
        prices = random_prices(n_times=30)

        with pytest.raises(PanelError, match="execution_lag must be at least 1"):
            backtest(prices.with_values(np.zeros(prices.shape)), prices, execution_lag=0)

    def test_a_larger_lag_changes_the_result(self) -> None:
        """If it does not, the signal is not actually driving anything."""
        prices = random_prices()
        weights = cs_scale(cs_demean(cs_rank(prices.pct_change(20))))

        one = backtest(weights, prices, costs=CostModel())
        two = backtest(weights, prices, costs=CostModel(), execution_lag=2)

        assert one.total_return != two.total_return

    def test_too_few_timestamps_for_the_lag_is_refused(self) -> None:
        prices = random_prices(n_times=3)

        with pytest.raises(PanelError, match="need more than execution_lag"):
            backtest(prices.with_values(np.zeros(prices.shape)), prices, execution_lag=5)


class TestNoOpDetection:
    def test_a_strategy_that_never_traded_reports_it_loudly(self) -> None:
        """A flat line, no errors, a plausible zero — the failure mode that most resembles
        success, and the one Trading_Algorithm shipped."""
        prices = random_prices()

        result = backtest(prices.with_values(np.zeros(prices.shape)), prices, costs=CostModel())

        assert not result.traded
        assert "NO-OP" in result.summary()

    def test_a_trading_strategy_reports_traded(self) -> None:
        prices = random_prices()
        weights = cs_scale(cs_demean(cs_rank(prices.pct_change(20))))

        assert backtest(weights, prices, costs=CostModel()).traded


class TestSquareRootImpact:
    def test_impact_grows_superlinearly_in_size(self) -> None:
        """The property a linear model cannot express, and the reason capacity is real."""
        model = CostModel(spread_bps=0.0, impact_coefficient=1.0)
        common = {
            "traded_weight": np.array([1.0]),
            "prices": np.array([100.0]),
            "dollar_volume": np.array([1e8]),
            "volatility": np.array([0.25]),
        }

        _, small, _, _ = model.charge_row(equity=1e6, **common)
        _, large, _, _ = model.charge_row(equity=1e8, **common)

        # 100x the size at the same participation multiplier => 10x the cost in bps.
        assert large / small == pytest.approx(10.0, rel=0.01)

    def test_impact_uses_per_period_not_annualised_volatility(self) -> None:
        """Feeding annualised volatility straight in inflates impact by sqrt(252) and makes
        every strategy look uncapacitated. Measured before the fix: a 20bps edge on a $100m-ADV
        name reported $6,400 of capacity."""
        model = CostModel(spread_bps=0.0, impact_coefficient=1.0)

        _, impact, _, participation = model.charge_row(
            traded_weight=np.array([1.0]),
            equity=1e6,
            prices=np.array([100.0]),
            dollar_volume=np.array([1e8]),
            volatility=np.array([0.25]),
        )

        assert participation == pytest.approx(0.01)
        # 0.25/sqrt(252) * sqrt(0.01) = 15.7bps, not 250bps.
        assert impact * 10_000 == pytest.approx(15.75, rel=0.01)

    def test_participation_beyond_a_day_of_volume_is_clipped(self) -> None:
        model = CostModel(spread_bps=0.0, impact_coefficient=1.0)

        _, impact, _, participation = model.charge_row(
            traded_weight=np.array([1.0]),
            equity=1e10,
            prices=np.array([100.0]),
            dollar_volume=np.array([1e8]),
            volatility=np.array([0.25]),
        )

        assert participation > 1.0
        assert np.isfinite(impact)

    def test_strained_periods_are_flagged(self) -> None:
        model = CostModel(spread_bps=0.0, impact_coefficient=1.0)
        prices = random_prices(n_times=30, n_assets=5)
        volume = prices.with_values(np.full(prices.shape, 1_000.0))
        weights = cs_scale(cs_demean(cs_rank(prices.pct_change(5))))

        result = backtest(weights, prices, costs=model, dollar_volume=volume, initial_equity=1e9)

        assert result.costs.strained_periods > 0
        assert "extrapolation" in result.costs.summary()


class TestCapacity:
    def test_capacity_scales_with_the_square_of_the_edge(self) -> None:
        """Q = ADV·(budget / (η·σ))², so doubling the edge quadruples capacity."""
        model = CostModel(spread_bps=0.0, impact_coefficient=1.0)

        small = model.capacity_notional(expected_edge_bps=10, dollar_volume=1e8, volatility=0.25)
        large = model.capacity_notional(expected_edge_bps=20, dollar_volume=1e8, volatility=0.25)

        assert large / small == pytest.approx(4.0, rel=0.01)

    def test_an_edge_smaller_than_the_spread_has_no_capacity(self) -> None:
        model = CostModel(spread_bps=10.0)

        assert (
            model.capacity_notional(expected_edge_bps=2, dollar_volume=1e8, volatility=0.25) == 0.0
        )

    def test_capacity_is_a_usable_number_for_a_liquid_name(self) -> None:
        """A sanity floor. The pre-fix code reported $6,400 here, which would have made every
        result look capacity-constrained and stopped the diagnostic discriminating."""
        model = CostModel(spread_bps=0.0, impact_coefficient=1.0)

        capacity = model.capacity_notional(expected_edge_bps=20, dollar_volume=1e8, volatility=0.25)

        assert 1e6 < capacity < 1e8

    def test_a_non_positive_volume_is_refused(self) -> None:
        with pytest.raises(ValueError, match="dollar_volume must be positive"):
            CostModel().capacity_notional(expected_edge_bps=20, dollar_volume=0, volatility=0.25)


class TestCostModelValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("spread_bps", -1.0),
            ("impact_coefficient", -1.0),
            ("commission_per_share", -1.0),
            ("min_volatility", 0.0),
            ("periods_per_year", 0),
        ],
    )
    def test_invalid_parameters_are_refused(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match=field):
            CostModel(**{field: value})  # type: ignore[arg-type]


class TestTheDecompositionAddsUp:
    """Found by fuzzing `net == gross - costs` over 3000 randomised backtests: it failed on 885
    of them. The delisting drag is a third term and was reported in neither column, so a run
    dominated by one bankruptcy showed "gross 0.00% - costs 0.01%" against an actual -40%."""

    @staticmethod
    def with_a_delisting() -> Panel:
        values = np.full((20, 2), 100.0)
        values[10:, 1] = np.nan
        return mk(values)

    def test_net_equals_gross_plus_drag_minus_costs(self) -> None:
        prices = self.with_a_delisting()

        result = backtest(
            prices.with_values(np.full((20, 2), 0.5)),
            prices,
            costs=CostModel(),
            delisting_return={"A1": -0.80},
        )

        np.testing.assert_allclose(
            result.net_returns,
            result.gross_returns + result.delisting_drag - result.costs.total,
            atol=1e-12,
        )

    def test_the_drag_is_exposed_rather_than_hidden(self) -> None:
        prices = self.with_a_delisting()

        result = backtest(
            prices.with_values(np.full((20, 2), 0.5)),
            prices,
            costs=CostModel(),
            delisting_return={"A1": -0.80},
        )

        assert result.delisting_drag.sum() == pytest.approx(-0.40, abs=1e-9)
        assert "delistings -40.00%" in result.summary()

    def test_a_run_without_delistings_has_no_drag(self) -> None:
        prices = random_prices()
        weights = cs_scale(cs_demean(cs_rank(prices.pct_change(20))))

        result = backtest(weights, prices, costs=CostModel())

        assert not result.delisting_drag.any()
        assert "delistings" not in result.summary()

    @pytest.mark.parametrize("seed", range(12))
    def test_the_identity_holds_under_randomised_inputs(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n_times, n_assets = 60, 6
        values = np.cumprod(1 + rng.normal(0, 0.02, (n_times, n_assets)), axis=0) * 100
        values[int(rng.integers(30, 55)) :, int(rng.integers(0, n_assets))] = np.nan
        prices = mk(values)

        result = backtest(
            prices.with_values(rng.normal(0, 0.3, (n_times, n_assets))),
            prices,
            costs=CostModel(spread_bps=float(rng.uniform(0, 20))),
            delisting_return=float(rng.uniform(-1, 0)),
        )

        np.testing.assert_allclose(
            result.net_returns,
            result.gross_returns + result.delisting_drag - result.costs.total,
            atol=1e-12,
        )
        assert result.equity[-1] == pytest.approx(float(np.prod(1 + result.net_returns)), rel=1e-7)
