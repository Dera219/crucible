"""The signal algebra.

Causality is proven separately, in test_causality.py, against every operator here. This file
tests what each one actually computes — and in particular the properties a portfolio depends on:
that ranks do not depend on column order, that a demeaned row sums to zero, that thin rows
produce nothing rather than a number that looks like every other number.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

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
from crucible.panel import Panel, PanelError


def panel(rows: list[list[float]], *, name: str = "p") -> Panel:
    width = len(rows[0])
    return Panel.from_rows(
        rows,
        index=[date(2025, 1, 1) + timedelta(days=i) for i in range(len(rows))],
        assets=[f"A{i}" for i in range(width)],
        name=name,
    )


class TestCrossSectionalRank:
    def test_ranks_span_zero_to_one(self) -> None:
        ranked = cs_rank(panel([[10.0, 20.0, 30.0, 40.0, 50.0]]))

        assert ranked.values[0].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_ties_take_their_average_rank(self) -> None:
        """Otherwise the result depends on column order, which is an accident of loading."""
        ranked = cs_rank(panel([[5.0, 5.0, 5.0, 5.0, 5.0]]))

        assert ranked.values[0].tolist() == pytest.approx([0.5] * 5)

    def test_rank_is_invariant_to_column_order(self) -> None:
        base = panel([[10.0, 20.0, 20.0, 40.0, 50.0]])
        reordered = base.select(["A4", "A2", "A0", "A3", "A1"])

        forward = cs_rank(base)
        backward = cs_rank(reordered)

        for asset in base.assets:
            assert forward.column(asset)[0] == pytest.approx(backward.column(asset)[0])

    def test_nan_inputs_stay_nan(self) -> None:
        """An asset that was not investable does not get a rank, so it cannot be selected."""
        ranked = cs_rank(panel([[1.0, np.nan, 3.0, 4.0, 5.0, 6.0]]))

        assert np.isnan(ranked.values[0, 1])
        assert not np.isnan(ranked.values[0, 0])

    def test_a_thin_row_produces_nothing(self) -> None:
        """Three names do not have a top decile."""
        ranked = cs_rank(panel([[1.0, 2.0, 3.0, np.nan, np.nan]]))

        assert np.isnan(ranked.values[0]).all()

    def test_min_breadth_is_configurable(self) -> None:
        ranked = cs_rank(panel([[1.0, 2.0, 3.0]]), min_breadth=3)

        assert not np.isnan(ranked.values[0]).any()


class TestCrossSectionalNormalisation:
    def test_demeaned_rows_sum_to_zero(self) -> None:
        """The crudest form of market neutrality: the average position is zero, so the strategy
        expresses relative performance rather than a leveraged bet on the market."""
        demeaned = cs_demean(panel([[1.0, 2.0, 3.0, 4.0, 5.0]]))

        assert demeaned.values[0].sum() == pytest.approx(0.0)

    def test_zscore_produces_unit_dispersion(self) -> None:
        scored = cs_zscore(panel([[1.0, 2.0, 3.0, 4.0, 5.0]]))

        assert scored.values[0].mean() == pytest.approx(0.0)
        assert scored.values[0].std() == pytest.approx(1.0)

    def test_a_zero_variance_row_yields_nan_not_infinity(self) -> None:
        scored = cs_zscore(panel([[3.0, 3.0, 3.0, 3.0, 3.0]]))

        assert np.isnan(scored.values[0]).all()

    def test_scale_normalises_gross_exposure(self) -> None:
        """Without it, a signal whose magnitude drifts silently changes its own leverage."""
        scaled = cs_scale(panel([[1.0, -2.0, 3.0, -4.0, 5.0]]))

        assert np.abs(scaled.values[0]).sum() == pytest.approx(1.0)

    def test_scale_preserves_sign_and_ordering(self) -> None:
        scaled = cs_scale(panel([[1.0, -2.0, 3.0, -4.0, 5.0]]))

        assert np.sign(scaled.values[0]).tolist() == [1.0, -1.0, 1.0, -1.0, 1.0]

    def test_a_non_positive_scale_target_is_refused(self) -> None:
        with pytest.raises(PanelError, match="target must be positive"):
            cs_scale(panel([[1.0, 2.0, 3.0, 4.0, 5.0]]), target=0.0)


class TestWinsorize:
    def test_an_outlier_is_clipped_without_reordering(self) -> None:
        """One bad print must not become the strategy."""
        values = [[1.0, 2.0, 3.0, 4.0, 1000.0]]

        clipped = cs_winsorize(panel(values), limit=0.2)

        assert clipped.values[0, 4] < 1000.0
        assert clipped.values[0, 4] >= clipped.values[0, 3]

    def test_a_zero_limit_is_a_no_op(self) -> None:
        p = panel([[1.0, 2.0, 3.0, 4.0, 5.0]])

        assert cs_winsorize(p, limit=0.0) is p

    def test_a_limit_at_or_above_a_half_is_refused(self) -> None:
        with pytest.raises(PanelError, match=r"\[0, 0.5\)"):
            cs_winsorize(panel([[1.0, 2.0, 3.0, 4.0, 5.0]]), limit=0.5)


class TestNeutralize:
    def test_a_signal_identical_to_the_factor_is_fully_removed(self) -> None:
        """A signal that vanishes under neutralisation was the control variable all along."""
        factor = panel([[1.0, 2.0, 3.0, 4.0, 5.0]], name="size")

        residual = cs_neutralize(factor, factor)

        assert residual.values[0] == pytest.approx([0.0] * 5, abs=1e-10)

    def test_an_orthogonal_signal_survives(self) -> None:
        signal = panel([[1.0, -1.0, 1.0, -1.0, 0.0]], name="signal")
        factor = panel([[1.0, 2.0, 3.0, 4.0, 5.0]], name="size")

        residual = cs_neutralize(signal, factor)

        assert np.abs(residual.values[0]).sum() > 1.0

    def test_the_residual_is_uncorrelated_with_the_factor(self) -> None:
        rng = np.random.default_rng(0)
        signal = panel([list(rng.normal(size=20))], name="signal")
        factor = panel([list(rng.normal(size=20))], name="size")

        residual = cs_neutralize(signal, factor)

        assert np.corrcoef(residual.values[0], factor.values[0])[0, 1] == pytest.approx(0, abs=1e-9)

    def test_misaligned_panels_are_refused(self) -> None:
        signal = panel([[1.0, 2.0, 3.0, 4.0, 5.0]], name="signal")
        factor = Panel.from_rows(
            [[1.0, 2.0, 3.0, 4.0, 5.0]],
            index=[date(2030, 1, 1)],
            assets=[f"A{i}" for i in range(5)],
            name="size",
        )

        with pytest.raises(PanelError, match="not aligned"):
            cs_neutralize(signal, factor)


class TestTimeSeriesOperators:
    def test_ts_zscore_measures_against_an_assets_own_past(self) -> None:
        rows = [[float(i)] for i in range(1, 11)]

        scored = ts_zscore(panel(rows), 5)

        assert np.isnan(scored.values[3, 0])
        assert scored.values[4, 0] == pytest.approx(1.414, abs=0.01)

    def test_ts_decay_weights_recent_data_more(self) -> None:
        rising = ts_decay(panel([[1.0], [2.0], [3.0]]), 3)

        assert rising.values[2, 0] == pytest.approx((1 * 1 + 2 * 2 + 3 * 3) / 6)

    def test_ts_decay_requires_a_full_window(self) -> None:
        decayed = ts_decay(panel([[1.0], [2.0], [3.0]]), 3)

        assert np.isnan(decayed.values[0, 0])
        assert np.isnan(decayed.values[1, 0])

    def test_ts_rank_places_today_in_its_own_history(self) -> None:
        ranked = ts_rank(panel([[1.0], [2.0], [3.0], [4.0], [5.0]]), 5)

        assert ranked.values[4, 0] == pytest.approx(1.0)

    def test_ts_rank_is_robust_to_level(self) -> None:
        """A price that has tripled still ranks mid-range if it has been flat for a month."""
        flat_high = ts_rank(panel([[300.0], [300.0], [301.0], [299.0], [300.0]]), 5)

        assert 0.0 < flat_high.values[4, 0] < 1.0

    def test_a_non_positive_window_is_refused(self) -> None:
        with pytest.raises(PanelError, match="window must be >= 1"):
            ts_decay(panel([[1.0]]), 0)


class TestComposition:
    def test_operators_chain_and_keep_the_lineage_in_the_name(self) -> None:
        """The name is the audit trail — a report that says what it actually computed."""
        result = cs_scale(cs_demean(cs_rank(panel([[1.0, 2.0, 3.0, 4.0, 5.0]], name="mom"))))

        assert result.name == "cs_scale(cs_demean(cs_rank(mom)))"

    def test_a_realistic_momentum_signal_is_market_neutral_and_unit_gross(self) -> None:
        rng = np.random.default_rng(1)
        prices = Panel(
            values=np.cumprod(1 + rng.normal(0.0004, 0.012, (40, 10)), axis=0) * 100,
            index=tuple(date(2025, 1, 1) + timedelta(days=i) for i in range(40)),
            assets=tuple(f"A{i}" for i in range(10)),
            name="close",
        )

        weights = cs_scale(cs_demean(cs_rank(prices.pct_change(20))))

        final = weights.values[-1]
        assert final.sum() == pytest.approx(0.0, abs=1e-12)
        assert np.abs(final).sum() == pytest.approx(1.0)
