"""The causality checker.

The tests that matter are the ones where a *plausible-looking* function fails. Anyone can catch
`shift(-1)`. What ships is full-sample normalisation, which looks like the textbook and knows the
future's distribution, and this file exists to prove the checker catches it every time.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import numpy as np
import pytest

from crucible.causality import CausalityViolation, assert_causal, check_causality
from crucible.ops import (
    cs_demean,
    cs_neutralize,
    cs_rank,
    cs_scale,
    cs_winsorize,
    cs_zscore,
    ts_decay,
    ts_rank,
    ts_zscore,
)
from crucible.panel import Panel


def prices(n_times: int = 60, n_assets: int = 12, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    values = np.cumprod(1 + rng.normal(0.0004, 0.012, (n_times, n_assets)), axis=0) * 100
    return Panel(
        values=values,
        index=tuple(date(2025, 1, 1) + timedelta(days=i) for i in range(n_times)),
        assets=tuple(f"A{i:02d}" for i in range(n_assets)),
        name="close",
    )


class TestCausalOperatorsPass:
    """Every operator in the library, and realistic compositions of them."""

    @pytest.mark.parametrize(
        ("label", "function"),
        [
            ("cs_rank", cs_rank),
            ("cs_zscore", cs_zscore),
            ("cs_demean", cs_demean),
            ("cs_scale", cs_scale),
            ("cs_winsorize", cs_winsorize),
            ("ts_zscore(20)", lambda p: ts_zscore(p, 20)),
            ("ts_decay(5)", lambda p: ts_decay(p, 5)),
            ("ts_rank(20)", lambda p: ts_rank(p, 20)),
            ("rolling mean", lambda p: p.rolling(20, "mean")),
            ("rolling std", lambda p: p.rolling(20, "std")),
            ("shift(1)", lambda p: p.shift(1)),
            ("pct_change(5)", lambda p: p.pct_change(5)),
            ("log_return(5)", lambda p: p.log_return(5)),
        ],
    )
    def test_each_operator_is_causal(self, label: str, function: Callable[[Panel], Panel]) -> None:
        assert check_causality(function, prices()).passed, label

    @pytest.mark.parametrize(
        ("label", "function"),
        [
            ("momentum", lambda p: cs_rank(p.pct_change(20).shift(1))),
            ("reversal", lambda p: -cs_zscore(p.pct_change(1))),
            ("low-vol", lambda p: -cs_rank(p.pct_change(1).rolling(20, "std"))),
            ("decayed momentum", lambda p: ts_decay(cs_demean(p.pct_change(20)), 5)),
            ("neutralised", lambda p: cs_neutralize(cs_rank(p.pct_change(10)), cs_rank(p))),
            (
                "full stack",
                lambda p: cs_scale(
                    cs_demean(cs_winsorize(cs_zscore(ts_decay(p.pct_change(20), 5))))
                ),
            ),
        ],
    )
    def test_realistic_signal_chains_are_causal(
        self, label: str, function: Callable[[Panel], Panel]
    ) -> None:
        assert check_causality(function, prices()).passed, label


class TestTheBugsThatActuallyShip:
    """Each of these looks like something a competent person would write."""

    def test_full_sample_zscore_is_caught(self) -> None:
        """The classic. Normalising with the mean and std of the whole history — including the
        part that had not happened — is textbook-shaped and knows the future's distribution."""

        def peeking(panel: Panel) -> Panel:
            values = panel.values
            return panel.with_values((values - np.nanmean(values)) / np.nanstd(values))

        report = check_causality(peeking, prices())

        assert not report.passed
        assert report.failing_cells > 0
        assert "whole sample" in report.describe()

    def test_global_minmax_scaling_is_caught(self) -> None:
        def peeking(panel: Panel) -> Panel:
            values = panel.values
            low, high = np.nanmin(values), np.nanmax(values)
            return panel.with_values((values - low) / (high - low))

        assert not check_causality(peeking, prices()).passed

    def test_a_full_sample_quantile_clip_is_caught(self) -> None:
        """Winsorising to global quantiles rather than per-row ones."""

        def peeking(panel: Panel) -> Panel:
            low, high = np.nanquantile(panel.values, [0.01, 0.99])
            return panel.with_values(np.clip(panel.values, low, high))

        assert not check_causality(peeking, prices()).passed

    def test_an_explicit_one_bar_peek_is_caught(self) -> None:
        def peeking(panel: Panel) -> Panel:
            out = np.full_like(panel.values, np.nan)
            out[:-1] = panel.values[1:]
            return panel.with_values(out)

        assert not check_causality(peeking, prices()).passed

    def test_a_centred_rolling_window_is_caught(self) -> None:
        """A centred window is half future by construction, and reads as innocuous."""

        def peeking(panel: Panel) -> Panel:
            values = panel.values
            out = np.full_like(values, np.nan)
            for row in range(2, values.shape[0] - 2):
                out[row] = values[row - 2 : row + 3].mean(axis=0)
            return panel.with_values(out)

        assert not check_causality(peeking, prices()).passed

    def test_a_signal_that_uses_forward_returns_is_caught(self) -> None:
        """The target leaking into the feature — how a model reaches 0.99 R-squared."""

        def peeking(panel: Panel) -> Panel:
            return cs_rank(panel.forward_returns(1))

        assert not check_causality(peeking, prices()).passed

    def test_a_late_only_peek_is_still_caught(self) -> None:
        """A function honest for most of the sample and peeking only near the end. A single cut
        point could miss this; the default sweep does not."""

        def peeking(panel: Panel) -> Panel:
            values = panel.values
            out = values.copy()
            out[-10:-1] = values[-9:]
            return panel.with_values(out)

        assert not check_causality(peeking, prices()).passed


class TestReporting:
    def test_a_violation_names_the_earliest_affected_cell(self) -> None:
        def peeking(panel: Panel) -> Panel:
            return panel.with_values(panel.values - np.nanmean(panel.values))

        report = check_causality(peeking, prices())

        assert report.first_failure is not None
        assert "Earliest affected cell" in report.detail
        assert "trailing window" in report.detail

    def test_a_pass_says_what_was_tested(self) -> None:
        report = check_causality(cs_rank, prices())

        assert "unchanged at every cut point" in report.describe()
        assert len(report.cut_points_tested) >= 3

    def test_assert_causal_raises_on_a_peeking_function(self) -> None:
        def peeking(panel: Panel) -> Panel:
            return panel.with_values(panel.values / np.nanmax(panel.values))

        with pytest.raises(CausalityViolation, match="NOT CAUSAL"):
            assert_causal(peeking, prices())

    def test_assert_causal_is_silent_on_a_clean_function(self) -> None:
        assert_causal(lambda p: cs_rank(p.pct_change(10)), prices())


class TestCheckerRobustness:
    def test_a_function_that_changes_row_count_is_refused(self) -> None:
        """Causality is defined per-timestamp, so a dropped warmup makes it undefined."""

        def truncating(panel: Panel) -> Panel:
            return panel.slice_time(10, None)

        with pytest.raises(ValueError, match="must stay on the input's index"):
            check_causality(truncating, prices())

    def test_too_short_a_panel_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 4 timestamps"):
            check_causality(cs_rank, prices(n_times=3))

    def test_results_are_reproducible(self) -> None:
        panel = prices()

        first = check_causality(cs_zscore, panel, seed=7)
        second = check_causality(cs_zscore, panel, seed=7)

        assert first == second

    def test_an_all_nan_signal_is_causal_not_a_false_positive(self) -> None:
        """A signal still in warmup emits nothing. Nothing is trivially causal, and reporting a
        violation here would train people to ignore the checker."""
        assert check_causality(lambda p: ts_zscore(p, 500), prices()).passed

    def test_a_constant_signal_is_causal(self) -> None:
        assert check_causality(lambda p: p.with_values(np.ones_like(p.values)), prices()).passed

    def test_perturbation_reaches_every_tested_cut_point(self) -> None:
        """A guard on the checker itself: if the perturbation were too gentle, a peeking function
        might survive by luck. This one peeks at exactly one row and must still be caught at
        every cut point that precedes it."""

        def peek_at_row(target: int) -> Callable[[Panel], Panel]:
            def peeking(panel: Panel) -> Panel:
                out = panel.values.copy()
                out[:target] += np.nansum(panel.values[target])
                return panel.with_values(out)

            return peeking

        for cut in (8, 15, 30, 45):
            report = check_causality(peek_at_row(cut + 1), prices(), cut_points=[cut])
            assert not report.passed, f"peek just after cut {cut} was not caught"
