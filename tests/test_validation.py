"""Walk-forward and significance.

Two claims carry this file. Walk-forward must distinguish "the strategy lost" from "the strategy
never ran" — Apex conflated them and reported a signal that could not warm up inside its test
folds as having lost every one. And deflation must discriminate in both directions: a test that
always says no is as useless as one that always says yes.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from crucible.costs import CostModel
from crucible.ops import cs_demean, cs_rank, cs_scale
from crucible.panel import Panel, PanelError
from crucible.significance import (
    SignificanceError,
    TrialLedger,
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from crucible.walkforward import expanding_folds, rolling_folds, walk_forward


def prices(n_times: int = 1400, n_assets: int = 40, seed: int = 5) -> Panel:
    rng = np.random.default_rng(seed)
    return Panel(
        np.cumprod(1 + rng.normal(0.0003, 0.014, (n_times, n_assets)), axis=0) * 100,
        tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(n_times)),
        tuple(f"A{i:02d}" for i in range(n_assets)),
        "close",
    )


def slow_signal(panel: Panel) -> Panel:
    """A 150-period lookback. Apex's exact failing case."""
    return cs_scale(cs_demean(cs_rank(panel.pct_change(150))))


def fast_signal(panel: Panel) -> Panel:
    return cs_scale(cs_demean(cs_rank(panel.pct_change(5))))


class TestWarmupFixesApexsFailure:
    """A 150-bar signal inside 100-bar test folds never warms up, never trades, and returns
    0.00% — which beats every losing baseline by not participating. Apex scored that as a loss."""

    def test_without_warmup_the_result_is_untestable_not_a_loss(self) -> None:
        panel = prices()
        folds = rolling_folds(panel, train_size=400, test_size=100, warmup=0)

        result = walk_forward(folds, slow_signal, panel, costs=CostModel())

        assert not result.testable
        assert "UNTESTABLE" in result.summary()
        assert "silence, not underperformance" in result.summary()
        assert len(result.no_ops) == len(result.scored)

    def test_with_warmup_every_fold_actually_runs(self) -> None:
        panel = prices()
        folds = rolling_folds(panel, train_size=400, test_size=100, warmup=160)

        result = walk_forward(folds, slow_signal, panel, costs=CostModel())

        assert result.testable
        assert not result.no_ops
        assert len(result.scored) == len(folds)

    def test_a_no_op_fold_is_never_counted_as_a_win(self) -> None:
        panel = prices()
        folds = rolling_folds(panel, train_size=400, test_size=100, warmup=0)

        result = walk_forward(folds, slow_signal, panel, costs=CostModel())

        assert result.genuine_wins == 0

    def test_warmup_comes_from_before_the_test_window(self) -> None:
        """Which is exactly what a live system has on its first day, so nothing leaks."""
        folds = rolling_folds(prices(), train_size=400, test_size=100, warmup=160)

        for fold in folds:
            assert fold.signal_start < fold.test_start
            assert fold.signal_start >= fold.train_start


class TestFoldConstruction:
    def test_test_windows_do_not_overlap_by_default(self) -> None:
        """Overlapping windows reuse bars and inflate every t-statistic computed from them."""
        folds = rolling_folds(prices(), train_size=400, test_size=100)

        for earlier, later in zip(folds, folds[1:], strict=False):
            assert earlier.test_end <= later.test_start

    def test_test_always_follows_train(self) -> None:
        for fold in rolling_folds(prices(), train_size=400, test_size=100):
            assert fold.train_end <= fold.test_start

    def test_expanding_folds_grow_the_training_window(self) -> None:
        folds = expanding_folds(prices(), initial_train=400, test_size=100)

        assert all(f.train_start == 0 for f in folds)
        assert folds[1].train_end > folds[0].train_end

    def test_warmup_larger_than_train_is_refused(self) -> None:
        with pytest.raises(PanelError, match="no fold would be usable"):
            rolling_folds(prices(), train_size=100, test_size=50, warmup=100)

    def test_an_unusable_fold_is_skipped_not_scored(self) -> None:
        fold = rolling_folds(prices(), train_size=400, test_size=100, warmup=399)[0]

        assert fold.usable
        assert fold.signal_start >= fold.train_start

    def test_too_short_a_panel_is_refused(self) -> None:
        with pytest.raises(PanelError, match="need at least"):
            rolling_folds(prices(n_times=100), train_size=400, test_size=100)

    def test_backwards_boundaries_are_refused(self) -> None:
        from crucible.walkforward import Fold

        with pytest.raises(PanelError, match="validation on the past"):
            Fold(index=0, train_start=0, train_end=100, test_start=50, test_end=150, warmup=0)


class TestWalkForwardScoring:
    def test_a_baseline_is_run_on_every_fold(self) -> None:
        panel = prices()
        folds = rolling_folds(panel, train_size=400, test_size=100, warmup=20)

        result = walk_forward(folds, fast_signal, panel, costs=CostModel())

        assert all(o.baseline_return != 0.0 for o in result.scored)

    def test_win_rate_counts_only_genuine_wins(self) -> None:
        panel = prices()
        folds = rolling_folds(panel, train_size=400, test_size=100, warmup=20)

        result = walk_forward(folds, fast_signal, panel, costs=CostModel())

        assert result.genuine_wins == sum(1 for o in result.scored if o.traded and o.excess > 0)

    def test_returns_are_concatenated_across_folds(self) -> None:
        panel = prices()
        folds = rolling_folds(panel, train_size=400, test_size=100, warmup=20)

        result = walk_forward(folds, fast_signal, panel, costs=CostModel())

        assert len(result.returns) == sum(f.test_length for f in folds)

    def test_no_folds_is_refused(self) -> None:
        with pytest.raises(PanelError, match="at least one fold"):
            walk_forward([], fast_signal, prices())


class TestSharpeStatistics:
    def test_sharpe_matches_the_definition(self) -> None:
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, 500)

        assert sharpe_ratio(returns) == pytest.approx(returns.mean() / returns.std(ddof=1))

    def test_zero_variance_is_refused_as_a_no_op(self) -> None:
        with pytest.raises(SignificanceError, match="never traded"):
            sharpe_ratio(np.zeros(100))

    def test_a_single_observation_is_refused(self) -> None:
        with pytest.raises(SignificanceError, match="at least 2 finite returns"):
            sharpe_ratio([0.01])

    def test_psr_rises_with_sample_length(self) -> None:
        rng = np.random.default_rng(1)
        short = rng.normal(0.0008, 0.012, 60)
        long = np.concatenate([short, rng.normal(0.0008, 0.012, 940)])

        assert probabilistic_sharpe_ratio(long) > probabilistic_sharpe_ratio(short)

    def test_fat_tails_reduce_confidence(self) -> None:
        """Negative skew and fat tails make the sample Sharpe a less reliable estimate, which is
        the entire reason PSR exists."""
        rng = np.random.default_rng(2)
        n = 250
        gaussian = rng.normal(0.0015, 0.012, n)
        crashy = np.concatenate([rng.normal(0.0022, 0.008, n - 12), rng.normal(-0.05, 0.02, 12)])

        assert probabilistic_sharpe_ratio(crashy) < probabilistic_sharpe_ratio(gaussian)


class TestDeflation:
    def test_the_bar_rises_with_trial_count(self) -> None:
        low = expected_max_sharpe(n_trials=5, trials_sharpe_variance=0.01)
        high = expected_max_sharpe(n_trials=500, trials_sharpe_variance=0.01)

        assert high > low > 0

    def test_a_single_trial_needs_no_deflation(self) -> None:
        assert expected_max_sharpe(n_trials=1, trials_sharpe_variance=0.01) == 0.0

    def test_it_kills_a_noise_sweep_winner(self) -> None:
        """The whole point. PSR alone passes several of these at 99%+."""
        rng = np.random.default_rng(4)
        ledger = TrialLedger()
        best, best_sharpe = None, -np.inf
        for _ in range(50):
            returns = rng.normal(0.0004, 0.012, 500)
            sharpe = ledger.record(returns)
            if sharpe > best_sharpe:
                best, best_sharpe = returns, sharpe

        report = ledger.report(best)

        assert report.psr_vs_zero > 0.95  # naive test would have passed it
        assert not report.survives_deflation

    def test_it_passes_a_strong_edge_on_a_long_sample(self) -> None:
        """A test that always says no is as useless as one that always says yes."""
        rng = np.random.default_rng(9)
        ledger = TrialLedger()
        for _ in range(50):
            ledger.record(rng.normal(0.0, 0.012, 2000))

        report = ledger.report(rng.normal(0.15 * 0.012, 0.012, 2000))

        assert report.survives_deflation

    def test_a_weak_edge_on_a_short_sample_is_killed(self) -> None:
        rng = np.random.default_rng(9)
        ledger = TrialLedger()
        for _ in range(50):
            ledger.record(rng.normal(0.0, 0.012, 500))

        report = ledger.report(rng.normal(0.05 * 0.012, 0.012, 500))

        assert not report.survives_deflation

    def test_identical_trials_are_refused(self) -> None:
        """Zero dispersion collapses the benchmark to zero, so the correction stops correcting
        while the report looks as rigorous as any other."""
        rng = np.random.default_rng(0)

        with pytest.raises(SignificanceError, match="identical Sharpe"):
            deflated_sharpe(rng.normal(0, 0.01, 200), trial_sharpes=[0.9] * 50)

    def test_two_identical_trials_are_tolerated(self) -> None:
        rng = np.random.default_rng(0)

        assert deflated_sharpe(rng.normal(0, 0.01, 200), trial_sharpes=[0.5, 0.5]).n_trials == 2

    def test_an_empty_ledger_is_refused(self) -> None:
        rng = np.random.default_rng(0)

        with pytest.raises(SignificanceError, match="every Sharpe the search evaluated"):
            deflated_sharpe(rng.normal(0, 0.01, 200), trial_sharpes=[])

    def test_the_verdict_names_the_discarded_attempts(self) -> None:
        rng = np.random.default_rng(4)
        ledger = TrialLedger()
        for _ in range(20):
            ledger.record(rng.normal(0.0004, 0.012, 300))

        verdict = ledger.report(rng.normal(0.0004, 0.012, 300)).verdict()

        assert "19 attempts discarded" in verdict or "survives deflation" in verdict


class TestTrialLedger:
    def test_it_counts_every_trial_including_degenerate_ones(self) -> None:
        """A trial the search saw and discarded still counted against you."""
        ledger = TrialLedger()
        ledger.record(np.random.default_rng(0).normal(0, 0.01, 100))
        ledger.record(np.zeros(100))  # a no-op strategy

        assert ledger.n_trials == 2

    def test_the_best_trial_is_identifiable(self) -> None:
        ledger = TrialLedger()
        rng = np.random.default_rng(0)
        ledger.record(rng.normal(0.0, 0.01, 200), label="flat")
        ledger.record(rng.normal(0.003, 0.01, 200), label="good")

        index, _ = ledger.best()

        assert ledger.labels[index] == "good"

    def test_there_is_no_way_to_forget_a_trial(self) -> None:
        """Forgetting trials is the bias, so removal is not offered."""
        ledger = TrialLedger()

        assert not hasattr(ledger, "remove")
        assert not hasattr(ledger, "discard")


class TestEvidenceRequirements:
    """How much data a realistic edge needs to prove itself. Directly relevant to whether a
    5.4-year vendor window can support any conclusion at all."""

    def test_a_realistic_edge_needs_years_not_months(self) -> None:
        rng = np.random.default_rng(11)

        def survives(true_sharpe: float, n: int) -> bool:
            ledger = TrialLedger()
            for _ in range(50):
                ledger.record(rng.normal(0.0, 0.012, n))
            return ledger.report(rng.normal(true_sharpe * 0.012, 0.012, n)).survives_deflation

        # An annualised Sharpe near 1.3 still needs many years against a 50-trial search.
        assert not survives(0.08, 500)
        assert survives(0.08, 8000)
