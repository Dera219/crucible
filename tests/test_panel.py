"""The panel.

Two properties carry the design: NaN means "not investable" and propagates honestly, and shift
only ever moves data forward. Everything else in the library assumes both.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from crucible.panel import Panel, PanelError, align


def panel(rows: list[list[float]], *, name: str = "p", n_assets: int | None = None) -> Panel:
    width = n_assets or (len(rows[0]) if rows else 0)
    return Panel.from_rows(
        rows,
        index=[date(2025, 1, 1) + timedelta(days=i) for i in range(len(rows))],
        assets=[f"A{i}" for i in range(width)],
        name=name,
    )


class TestValidation:
    def test_a_mismatched_shape_is_refused(self) -> None:
        with pytest.raises(PanelError, match="does not match"):
            Panel(values=np.zeros((3, 2)), index=(date(2025, 1, 1),), assets=("A", "B"))

    def test_duplicate_assets_are_refused(self) -> None:
        with pytest.raises(PanelError, match="duplicate asset"):
            Panel(values=np.zeros((1, 2)), index=(date(2025, 1, 1),), assets=("A", "A"))

    def test_an_unsorted_index_is_refused(self) -> None:
        """An out-of-order timestamp silently breaks every shift, and therefore every
        no-lookahead guarantee built on one."""
        with pytest.raises(PanelError, match="strictly increasing"):
            Panel(
                values=np.zeros((2, 1)),
                index=(date(2025, 1, 2), date(2025, 1, 1)),
                assets=("A",),
            )

    def test_a_duplicated_timestamp_is_refused(self) -> None:
        with pytest.raises(PanelError, match="strictly increasing"):
            Panel(
                values=np.zeros((2, 1)),
                index=(date(2025, 1, 1), date(2025, 1, 1)),
                assets=("A",),
            )

    def test_a_one_dimensional_array_is_refused(self) -> None:
        with pytest.raises(PanelError, match="must be 2-D"):
            Panel(values=np.zeros(3), index=(date(2025, 1, 1),), assets=("A",))


class TestImmutability:
    def test_values_are_read_only(self) -> None:
        """A numpy slice is a view; one in-place edit can rewrite history behind three other
        objects that thought they held their own copy."""
        p = panel([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError):
            p.values[0, 0] = 99.0

    def test_the_source_array_cannot_mutate_the_panel(self) -> None:
        source = np.array([[1.0, 2.0]])
        p = Panel(values=source, index=(date(2025, 1, 1),), assets=("A", "B"))

        source[0, 0] = 99.0

        assert p.values[0, 0] == 1.0

    def test_operations_return_new_panels(self) -> None:
        original = panel([[1.0], [2.0], [3.0]])

        shifted = original.shift(1)

        assert original.values[0, 0] == 1.0
        assert np.isnan(shifted.values[0, 0])


class TestInvestability:
    def test_nan_marks_a_cell_not_investable(self) -> None:
        p = panel([[1.0, np.nan], [2.0, 3.0]])

        assert p.investable.tolist() == [[True, False], [True, True]]
        assert p.coverage == 0.75

    def test_breadth_per_row_is_reported(self) -> None:
        """A cross-sectional strategy needs breadth. Deciles computed over three names are
        noise, however good the full-sample average looks."""
        p = panel([[1.0, np.nan, np.nan], [1.0, 2.0, 3.0]])

        assert p.count_per_row().tolist() == [1, 3]

    def test_an_empty_panel_has_zero_coverage(self) -> None:
        assert Panel.empty(assets=("A", "B")).coverage == 0.0


class TestShift:
    def test_shift_moves_data_forward(self) -> None:
        shifted = panel([[1.0], [2.0], [3.0]]).shift(1)

        assert np.isnan(shifted.values[0, 0])
        assert shifted.values[1, 0] == 1.0
        assert shifted.values[2, 0] == 2.0

    def test_a_negative_shift_is_refused(self) -> None:
        """The one-line version of lookahead. Refused outright rather than documented."""
        with pytest.raises(PanelError, match="future data into the present"):
            panel([[1.0], [2.0]]).shift(-1)

    def test_the_refusal_points_at_forward_returns(self) -> None:
        with pytest.raises(PanelError, match="forward_returns"):
            panel([[1.0], [2.0]]).shift(-1)

    def test_shifting_by_zero_is_the_identity(self) -> None:
        p = panel([[1.0], [2.0]])

        assert p.shift(0) is p

    def test_shifting_past_the_end_yields_all_nan(self) -> None:
        assert np.isnan(panel([[1.0], [2.0]]).shift(10).values).all()


class TestReturns:
    def test_pct_change_uses_only_the_past(self) -> None:
        p = panel([[100.0], [110.0], [121.0]])

        changes = p.pct_change(1)

        assert np.isnan(changes.values[0, 0])
        assert changes.values[1, 0] == pytest.approx(0.10)
        assert changes.values[2, 0] == pytest.approx(0.10)

    def test_a_zero_previous_price_yields_nan_not_infinity(self) -> None:
        changes = panel([[0.0], [10.0]]).pct_change(1)

        assert np.isnan(changes.values[1, 0])

    def test_forward_returns_look_ahead_by_design(self) -> None:
        """A target, never an input. Named so it is obvious in a diff."""
        forward = panel([[100.0], [110.0], [121.0]]).forward_returns(1)

        assert forward.values[0, 0] == pytest.approx(0.10)
        assert np.isnan(forward.values[2, 0])

    def test_log_returns_refuse_non_positive_prices(self) -> None:
        returns = panel([[-5.0], [10.0]]).log_return(1)

        assert np.isnan(returns.values[1, 0])


class TestRolling:
    def test_a_partial_window_emits_nothing_by_default(self) -> None:
        """Otherwise a 200-day average is quietly a 12-day average for the first ten months,
        and the early period looks unusually tradable."""
        rolled = panel([[1.0], [2.0], [3.0]]).rolling(3, "mean")

        assert np.isnan(rolled.values[0, 0])
        assert np.isnan(rolled.values[1, 0])
        assert rolled.values[2, 0] == pytest.approx(2.0)

    def test_min_periods_permits_a_partial_window_explicitly(self) -> None:
        rolled = panel([[1.0], [2.0], [3.0]]).rolling(3, "mean", min_periods=2)

        assert np.isnan(rolled.values[0, 0])
        assert rolled.values[1, 0] == pytest.approx(1.5)

    def test_nan_inputs_do_not_poison_the_window(self) -> None:
        rolled = panel([[1.0], [np.nan], [3.0]]).rolling(3, "mean", min_periods=2)

        assert rolled.values[2, 0] == pytest.approx(2.0)

    def test_an_unknown_statistic_is_refused(self) -> None:
        with pytest.raises(PanelError, match="unknown statistic"):
            panel([[1.0]]).rolling(1, "kurtosis")

    def test_min_periods_outside_the_window_is_refused(self) -> None:
        with pytest.raises(PanelError, match="min_periods"):
            panel([[1.0], [2.0]]).rolling(2, "mean", min_periods=5)


class TestArithmetic:
    def test_panels_combine_elementwise(self) -> None:
        left = panel([[1.0, 2.0]])
        right = panel([[10.0, 20.0]])

        assert (left + right).values.tolist() == [[11.0, 22.0]]
        assert (right / left).values.tolist() == [[10.0, 10.0]]

    def test_scalars_broadcast(self) -> None:
        assert (panel([[1.0, 2.0]]) * 3).values.tolist() == [[3.0, 6.0]]

    def test_mismatched_panels_refuse_to_combine(self) -> None:
        """Silent broadcasting is how a signal ends up paired with the wrong asset's returns."""
        left = panel([[1.0, 2.0]], name="left")
        right = Panel.from_rows(
            [[1.0, 2.0]], index=[date(2030, 1, 1)], assets=["A0", "A1"], name="right"
        )

        with pytest.raises(PanelError, match="Use align"):
            left + right

    def test_division_by_zero_yields_infinity_not_an_exception(self) -> None:
        result = panel([[1.0]]) / panel([[0.0]])

        assert np.isinf(result.values[0, 0])


class TestSelection:
    def test_columns_can_be_selected_and_reordered(self) -> None:
        p = panel([[1.0, 2.0, 3.0]])

        selected = p.select(["A2", "A0"])

        assert selected.assets == ("A2", "A0")
        assert selected.values.tolist() == [[3.0, 1.0]]

    def test_an_unknown_asset_is_refused(self) -> None:
        with pytest.raises(PanelError, match="unknown assets"):
            panel([[1.0]]).select(["NOPE"])

    def test_time_slicing_preserves_order(self) -> None:
        p = panel([[1.0], [2.0], [3.0], [4.0]])

        assert p.slice_time(1, 3).values.tolist() == [[2.0], [3.0]]

    def test_between_is_inclusive(self) -> None:
        p = panel([[1.0], [2.0], [3.0]])

        window = p.between(date(2025, 1, 2), date(2025, 1, 3))

        assert window.n_times == 2

    def test_a_window_outside_the_index_is_empty(self) -> None:
        assert panel([[1.0]]).between(date(2030, 1, 1), date(2030, 1, 2)).n_times == 0


class TestAlign:
    def test_common_times_and_assets_are_kept(self) -> None:
        left = Panel.from_rows(
            [[1.0, 2.0], [3.0, 4.0]],
            index=[date(2025, 1, 1), date(2025, 1, 2)],
            assets=["A", "B"],
            name="left",
        )
        right = Panel.from_rows(
            [[5.0, 6.0]], index=[date(2025, 1, 2)], assets=["B", "C"], name="right"
        )

        a, b = align(left, right)

        assert a.index == b.index == (date(2025, 1, 2),)
        assert a.assets == b.assets == ("B",)
        assert a.values.tolist() == [[4.0]]
        assert b.values.tolist() == [[5.0]]

    def test_disjoint_panels_are_refused(self) -> None:
        left = Panel.from_rows([[1.0]], index=[date(2025, 1, 1)], assets=["A"])
        right = Panel.from_rows([[1.0]], index=[date(2030, 1, 1)], assets=["A"])

        with pytest.raises(PanelError, match="no common"):
            align(left, right)

    def test_a_single_panel_passes_through(self) -> None:
        p = panel([[1.0]])

        assert align(p) == (p,)

    def test_aligning_nothing_is_refused(self) -> None:
        with pytest.raises(PanelError, match="at least one panel"):
            align()
