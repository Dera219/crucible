"""The diagnostic layer.

The central test is `TestAgainstAPlantedSignal`: data is constructed with a known predictive
relationship of a known strength, and the IC must recover it. A diagnostic that cannot find an
edge you put there yourself is worse than no diagnostic, because it will report "no edge" for
every real one you ever find and you will believe it.

The first version failed exactly that test — an off-by-one made a true IC of 0.13 measure at
0.008 — which is why this file exists in the shape it does.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from crucible.diagnostics import (
    QuantileReport,
    autocorrelation,
    diagnose,
    fundamental_law,
    ic_decay,
    information_coefficient,
    quantile_spread,
)
from crucible.panel import Panel, PanelError

NOISE = 0.014
STRENGTH = 0.0018


def planted(
    n_times: int = 800, n_assets: int = 80, strength: float = STRENGTH, seed: int = 7
) -> tuple[Panel, Panel]:
    """Prices whose next-period return is driven by a hidden score of known strength.

    Returns `(hidden_signal, prices)`. The true Pearson IC is
    `strength / sqrt(strength² + noise²)`, so the measurement has something exact to be checked
    against rather than merely looking plausible.
    """
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=(n_times, n_assets))
    noise = rng.normal(0, NOISE, (n_times, n_assets))
    returns = np.zeros((n_times, n_assets))
    returns[1:] = strength * hidden[:-1] + noise[1:]

    index = tuple(date(2022, 1, 1) + timedelta(days=i) for i in range(n_times))
    assets = tuple(f"A{i:02d}" for i in range(n_assets))
    prices = Panel(np.cumprod(1 + returns, axis=0) * 100, index, assets, "close")
    return prices.with_values(hidden, name="planted"), prices


def theoretical_ic(strength: float = STRENGTH) -> float:
    return strength / np.sqrt(strength**2 + NOISE**2)


class TestAgainstAPlantedSignal:
    def test_it_recovers_an_edge_of_known_strength(self) -> None:
        signal, prices = planted()

        report = information_coefficient(signal, prices)

        assert report.mean == pytest.approx(theoretical_ic(), abs=0.01)
        assert report.t_statistic > 10

    def test_a_stronger_planted_edge_measures_stronger(self) -> None:
        weak, weak_prices = planted(strength=0.0009)
        strong, strong_prices = planted(strength=0.0036)

        assert (
            information_coefficient(weak, weak_prices).mean
            < information_coefficient(strong, strong_prices).mean
        )

    def test_pure_noise_measures_as_no_edge(self) -> None:
        _, prices = planted()
        rng = np.random.default_rng(99)
        noise = prices.with_values(rng.normal(size=prices.shape), name="noise")

        report = information_coefficient(noise, prices)

        assert abs(report.t_statistic) < 2
        assert "no detectable edge" in report.verdict()

    def test_an_inverted_signal_is_reported_as_negative(self) -> None:
        """A sign error and a real contrarian effect look identical here, and the verdict says
        so rather than silently suggesting you flip it."""
        signal, prices = planted()

        report = information_coefficient(
            prices.with_values(-signal.values, name="inverted"), prices
        )

        assert report.mean < 0
        assert "NEGATIVE" in report.verdict()
        assert "sign error" in report.verdict()

    def test_the_default_lag_matches_the_engine(self) -> None:
        """The off-by-one that motivated this file. execution_lag=1 must credit the signal at t
        with the return from t to t+1, which is the same period the engine's default earns."""
        signal, prices = planted()

        at_one = information_coefficient(signal, prices, execution_lag=1)
        at_two = information_coefficient(signal, prices, execution_lag=2)

        assert at_one.mean == pytest.approx(theoretical_ic(), abs=0.01)
        assert abs(at_two.t_statistic) < 3  # a one-day edge is gone by the next period


class TestICStatistics:
    def test_icir_measures_stability_not_size(self) -> None:
        signal, prices = planted()

        report = information_coefficient(signal, prices)

        assert report.icir == pytest.approx(report.mean / report.std, rel=1e-9)

    def test_hit_rate_exceeds_a_half_for_a_real_edge(self) -> None:
        signal, prices = planted()

        assert information_coefficient(signal, prices).hit_rate > 0.6

    def test_too_few_periods_refuses_to_conclude(self) -> None:
        signal, prices = planted(n_times=18)

        assert "too few periods" in information_coefficient(signal, prices).verdict()

    def test_an_implausibly_high_ic_is_called_out(self) -> None:
        """crucible.causality catches mechanical lookahead; this catches the kind that arrives
        already baked into the data."""
        _, prices = planted()
        leaked = prices.with_values(prices.forward_returns(1).values, name="leaked_target")

        report = information_coefficient(leaked, prices)

        assert "implausibly high" in report.verdict()
        assert "in the DATA" in report.verdict()

    def test_a_zero_horizon_is_refused(self) -> None:
        signal, prices = planted(n_times=50)

        with pytest.raises(PanelError, match="horizon must be >= 1"):
            information_coefficient(signal, prices, horizon=0)

    def test_a_zero_execution_lag_is_refused(self) -> None:
        signal, prices = planted(n_times=50)

        with pytest.raises(PanelError, match="execution_lag must be >= 1"):
            information_coefficient(signal, prices, execution_lag=0)


class TestICDecay:
    def test_a_one_period_edge_decays_per_period(self) -> None:
        """The diagnostic that sets rebalance frequency: per-period IC should collapse as the
        horizon extends past the edge's natural life."""
        signal, prices = planted()

        decay = ic_decay(signal, prices, horizons=(1, 5, 20))
        per_period = {h: r.mean / h for h, r in decay.items()}

        assert per_period[1] > per_period[5] > per_period[20]

    def test_every_requested_horizon_is_returned(self) -> None:
        signal, prices = planted(n_times=200)

        assert set(ic_decay(signal, prices, horizons=(1, 3, 7))) == {1, 3, 7}


class TestQuantiles:
    def test_a_real_signal_grades_monotonically(self) -> None:
        signal, prices = planted()

        report = quantile_spread(signal, prices, n_quantiles=5)

        assert report.monotonicity == pytest.approx(1.0)
        assert report.spread > 0

    def test_buckets_are_evenly_populated(self) -> None:
        signal, prices = planted()

        counts = quantile_spread(signal, prices, n_quantiles=5).counts

        assert counts.std() / counts.mean() < 0.01

    def test_noise_produces_no_meaningful_spread(self) -> None:
        _, prices = planted()
        rng = np.random.default_rng(3)
        noise = prices.with_values(rng.normal(size=prices.shape), name="noise")

        assert abs(quantile_spread(noise, prices).spread) < 5e-4

    def test_fewer_than_two_buckets_is_refused(self) -> None:
        signal, prices = planted(n_times=50)

        with pytest.raises(PanelError, match="n_quantiles must be >= 2"):
            quantile_spread(signal, prices, n_quantiles=1)


class TestTailDrivenDetection:
    """The first threshold fired on a perfectly linear signal, which is exactly where a linear
    signal sits. A warning that triggers on the ideal case teaches you to ignore warnings."""

    @staticmethod
    def report(values: list[float]) -> QuantileReport:
        array = np.array(values, dtype=float)
        return QuantileReport(
            mean_returns=array,
            counts=np.full(len(array), 100.0),
            n_quantiles=len(array),
            horizon=1,
        )

    @pytest.mark.parametrize(
        "buckets",
        [
            [-2, -1, 0, 1, 2],
            list(range(-5, 5)),
            [-2.6, -0.9, 0.02, 1.2, 2.5],
            [-3, -1.2, -0.3, 0.6, 2.5],
        ],
    )
    def test_a_graded_signal_is_not_flagged(self, buckets: list[float]) -> None:
        assert not self.report(buckets).driven_by_one_tail

    @pytest.mark.parametrize(
        "buckets",
        [
            [0, 0, 0, 0, 5],
            [-5, 0, 0, 0, 5],
            [0, 0.1, 0.1, 0.1, 8],
        ],
    )
    def test_a_tail_driven_signal_is_flagged(self, buckets: list[float]) -> None:
        assert self.report(buckets).driven_by_one_tail

    def test_the_warning_appears_in_the_summary(self) -> None:
        assert "one extreme bucket" in self.report([0, 0, 0, 0, 5]).summary()

    def test_a_non_monotone_signal_is_warned_about(self) -> None:
        assert "not monotone" in self.report([1, -1, 2, -2, 0]).summary()


class TestAutocorrelation:
    def test_a_persistent_signal_scores_high(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.normal(size=(200, 40))
        smooth = np.cumsum(base, axis=0) / np.arange(1, 201)[:, None]
        panel = Panel(
            smooth,
            tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(200)),
            tuple(f"A{i}" for i in range(40)),
            "smooth",
        )

        assert autocorrelation(panel) > 0.8

    def test_a_churning_signal_scores_near_zero(self) -> None:
        rng = np.random.default_rng(2)
        panel = Panel(
            rng.normal(size=(200, 40)),
            tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(200)),
            tuple(f"A{i}" for i in range(40)),
            "churn",
        )

        assert abs(autocorrelation(panel)) < 0.15

    def test_a_zero_lag_is_refused(self) -> None:
        signal, _ = planted(n_times=50)

        with pytest.raises(PanelError, match="lag must be >= 1"):
            autocorrelation(signal, lag=0)


class TestFundamentalLaw:
    def test_information_ratio_scales_with_the_root_of_breadth(self) -> None:
        """Why a person trading five names cannot reach a Sharpe of 2 however good the idea."""
        narrow = fundamental_law(0.05, breadth=5)
        wide = fundamental_law(0.05, breadth=500)

        assert wide / narrow == pytest.approx(10.0, rel=1e-9)

    def test_it_scales_linearly_with_skill(self) -> None:
        assert fundamental_law(0.06, 100) / fundamental_law(0.02, 100) == pytest.approx(3.0)

    def test_the_transfer_coefficient_reduces_the_ceiling(self) -> None:
        full = fundamental_law(0.03, 200)
        realistic = fundamental_law(0.03, 200, transfer_coefficient=0.5)

        assert realistic == pytest.approx(full * 0.5)

    def test_it_bounds_an_actual_backtest(self) -> None:
        """The check worth running on any promising result: a Sharpe materially above the
        ceiling is a bug, not a discovery."""
        from crucible.costs import CostModel
        from crucible.engine import backtest
        from crucible.ops import cs_demean, cs_rank, cs_scale

        signal, prices = planted()
        report = information_coefficient(signal, prices)
        result = backtest(
            cs_scale(cs_demean(cs_rank(signal))), prices, costs=CostModel(spread_bps=3.0)
        )
        returns = result.net_returns[np.isfinite(result.net_returns)]
        sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)

        ceiling = fundamental_law(report.mean, float(np.nanmedian(report.breadth)))

        assert 0 < sharpe < ceiling

    @pytest.mark.parametrize(("breadth", "transfer"), [(0, 1.0), (-5, 1.0), (10, 0.0), (10, 1.5)])
    def test_invalid_inputs_are_refused(self, breadth: float, transfer: float) -> None:
        with pytest.raises(ValueError):
            fundamental_law(0.03, breadth, transfer_coefficient=transfer)


class TestDiagnoseReport:
    def test_it_runs_end_to_end_and_names_the_signal(self) -> None:
        signal, prices = planted(n_times=300, n_assets=40)

        text = diagnose(signal, prices)

        assert "planted" in text
        assert "IC by horizon" in text
        assert "Fundamental law" in text
        assert "buckets" in text

    def test_it_reports_no_edge_for_noise(self) -> None:
        _, prices = planted(n_times=300, n_assets=40)
        rng = np.random.default_rng(5)
        noise = prices.with_values(rng.normal(size=prices.shape), name="noise")

        assert "no detectable edge" in diagnose(noise, prices)
