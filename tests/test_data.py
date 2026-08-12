"""Loading real data.

Reshaping long rows into matrices is ten lines. Everything tested here is the other part: each
vendor encodes decisions in its schema, and reading those columns at face value produces a
dataset that looks fine and is not.

The CRSP fixture below plants every trap deliberately, so the loader is tested against data
shaped like the real thing rather than against data shaped like the loader.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from crucible.costs import CostModel
from crucible.data import (
    DataError,
    Dataset,
    dataset_from_panels,
    load_crsp_ciz,
    load_crsp_csv,
    long_to_panel,
)
from crucible.engine import backtest
from crucible.panel import Panel

SCHEMA = {
    "PERMNO": pl.Int64,
    "date": pl.Utf8,
    "PRC": pl.Float64,
    "VOL": pl.Float64,
    "RET": pl.Utf8,
    "SHRCD": pl.Int64,
    "EXCHCD": pl.Int64,
    "TICKER": pl.Utf8,
    "DLRET": pl.Utf8,
}

#: permno -> (share code, exchange code, ticker). Each row plants a different trap.
SECURITIES = {
    "10001": (11, 1, "AAA"),  # ordinary, trades throughout
    "10002": (11, 3, "BBB"),  # delists mid-sample with a punitive DLRET
    "10003": (11, 1, "CCC"),  # intermittent non-trading days (negative PRC)
    "10004": (73, 1, "DDD"),  # closed-end fund — must be filtered out by SHRCD
    "10005": (11, 9, "EEE"),  # exchange 9 — must be filtered out by EXCHCD
}
DELIST_ROW = 150
N_SESSIONS = 250


@pytest.fixture
def crsp_file(tmp_path: Path) -> Path:
    rng = np.random.default_rng(2)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(N_SESSIONS)]
    rows = []
    for permno, (share_code, exchange, ticker) in SECURITIES.items():
        prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, N_SESSIONS))
        for i, when in enumerate(dates):
            if permno == "10002" and i > DELIST_ROW:
                continue
            price = prices[i]
            if permno == "10003" and i % 20 == 7:
                price = -price  # CRSP: negated midpoint, no trade occurred
            rows.append(
                {
                    "PERMNO": int(permno),
                    "date": when.isoformat(),
                    "PRC": round(float(price), 4),
                    "VOL": float(rng.integers(1e5, 1e7)),
                    # CRSP ships returns as strings and encodes missing ones as letters.
                    "RET": "C" if i == 3 else f"{rng.normal(0, 0.015):.6f}",
                    "SHRCD": share_code,
                    "EXCHCD": exchange,
                    "TICKER": ticker,
                    "DLRET": "-0.85" if (permno == "10002" and i == DELIST_ROW) else "",
                }
            )
    path = tmp_path / "crsp.csv"
    pl.DataFrame(rows, schema=SCHEMA).write_csv(path)
    return path


class TestCRSPTraps:
    def test_it_keys_on_permno_not_ticker(self, crsp_file: Path) -> None:
        """Tickers are recycled. Keying on them concatenates unrelated companies into one series
        and produces an enormous return at the splice."""
        dataset = load_crsp_csv(crsp_file)

        assert dataset.assets == ("10001", "10002", "10003")
        assert dataset.tickers["10001"] == "AAA"

    def test_a_negative_price_marks_a_non_trading_day(self, crsp_file: Path) -> None:
        """A negated midpoint is not a price anyone filled."""
        dataset = load_crsp_csv(crsp_file)
        column = dataset.assets.index("10003")

        investable = int(dataset.prices.investable[:, column].sum())

        assert investable == N_SESSIONS - len(range(7, N_SESSIONS, 20))

    def test_the_delisting_return_is_captured(self, crsp_file: Path) -> None:
        """DLRET is the whole point of using CRSP. Ignoring it and letting positions vanish at
        their last price is the largest remaining upward bias in a survivorship-free dataset."""
        assert load_crsp_csv(crsp_file).delisting_returns == {"10002": -0.85}

    def test_share_codes_filter_out_non_common_stock(self, crsp_file: Path) -> None:
        assert "10004" not in load_crsp_csv(crsp_file).assets

    def test_exchange_codes_filter_out_minor_venues(self, crsp_file: Path) -> None:
        assert "10005" not in load_crsp_csv(crsp_file).assets

    def test_filters_can_be_disabled(self, crsp_file: Path) -> None:
        dataset = load_crsp_csv(crsp_file, share_codes=None, exchange_codes=None)

        assert {"10004", "10005"} <= set(dataset.assets)

    def test_letter_coded_returns_become_nan_not_zero(self, crsp_file: Path) -> None:
        dataset = load_crsp_csv(crsp_file)

        assert np.isnan(dataset.returns.values[3, dataset.assets.index("10001")])

    def test_a_security_running_past_the_extract_is_not_marked_delisted(
        self, crsp_file: Path
    ) -> None:
        """Recording that as a delisting would invent an exit that never happened."""
        listings = load_crsp_csv(crsp_file).listings

        assert listings["10001"][1] is None
        assert listings["10002"][1] is not None

    def test_dollar_volume_uses_the_price_magnitude(self, crsp_file: Path) -> None:
        dataset = load_crsp_csv(crsp_file)

        assert np.nanmin(dataset.dollar_volume.values) > 0


class TestLoaderValidation:
    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="does not exist"):
            load_crsp_csv(tmp_path / "nope.csv")

    def test_a_file_without_required_columns_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong.csv"
        pl.DataFrame({"a": [1], "b": [2]}).write_csv(path)

        with pytest.raises(DataError, match="missing required column"):
            load_crsp_csv(path)

    def test_filters_that_exclude_everything_are_refused(self, crsp_file: Path) -> None:
        with pytest.raises(DataError, match="no rows survived"):
            load_crsp_csv(crsp_file, share_codes=(99,))

    def test_an_unknown_pivot_column_is_refused(self) -> None:
        frame = pl.DataFrame({"d": [date(2024, 1, 1)], "a": ["X"], "v": [1.0]})

        with pytest.raises(DataError, match="not in frame"):
            long_to_panel(frame, date_column="d", asset_column="a", value_column="missing")


class TestLongToPanel:
    def test_missing_combinations_become_nan(self) -> None:
        frame = pl.DataFrame(
            {
                "d": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2)],
                "a": ["X", "Y", "X"],
                "v": [1.0, 2.0, 3.0],
            }
        )

        panel = long_to_panel(frame, date_column="d", asset_column="a", value_column="v")

        assert panel.values[0].tolist() == [1.0, 2.0]
        assert panel.values[1, 0] == 3.0
        assert np.isnan(panel.values[1, 1])

    def test_a_forced_asset_set_keeps_absent_columns(self) -> None:
        """A security that never traded in this window keeps its place as an all-NaN column."""
        frame = pl.DataFrame({"d": [date(2024, 1, 1)], "a": ["X"], "v": [1.0]})

        panel = long_to_panel(
            frame, date_column="d", asset_column="a", value_column="v", assets=["X", "Z"]
        )

        assert panel.assets == ("X", "Z")
        assert np.isnan(panel.values[0, 1])


class TestDelistingIsPermanent:
    """A security that stops trading for a session and resumes has not delisted. Treating every
    gap as an exit applied the delisting return once per gap — a name with intermittent
    non-trading days and a DLRET of -0.85 was charged -85% several times over."""

    @staticmethod
    def panel_with_gaps_then_delisting() -> Panel:
        values = np.full((60, 2), 100.0)
        for row in range(5, 50, 10):
            values[row, 0] = np.nan  # five temporary gaps
        values[50:, 0] = np.nan  # then a permanent exit
        return Panel(
            values,
            tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(60)),
            ("A", "B"),
            "close",
        )

    def test_temporary_gaps_are_not_delistings(self) -> None:
        prices = self.panel_with_gaps_then_delisting()

        result = backtest(
            prices.with_values(np.full((60, 2), 0.5)),
            prices,
            costs=CostModel(),
            delisting_return={"A": -0.85},
        )

        assert result.delisting_events == 1

    def test_the_delisting_return_is_charged_once(self) -> None:
        prices = self.panel_with_gaps_then_delisting()

        result = backtest(
            prices.with_values(np.full((60, 2), 0.5)),
            prices,
            costs=CostModel(),
            delisting_return={"A": -0.85},
        )

        # -85% on a 50% position, once. Charged per gap it would approach a total loss.
        assert result.total_return == pytest.approx(-0.425, abs=0.05)

    def test_per_asset_returns_beat_a_single_assumption(self) -> None:
        prices = self.panel_with_gaps_then_delisting()
        weights = prices.with_values(np.full((60, 2), 0.5))

        optimistic = backtest(weights, prices, costs=CostModel())
        actual = backtest(weights, prices, costs=CostModel(), delisting_return={"A": -0.85})

        assert optimistic.total_return > actual.total_return

    def test_an_asset_absent_from_the_mapping_falls_back_to_zero(self) -> None:
        prices = self.panel_with_gaps_then_delisting()

        result = backtest(
            prices.with_values(np.full((60, 2), 0.5)),
            prices,
            costs=CostModel(),
            delisting_return={"UNKNOWN": -0.85},
        )

        assert result.total_return == pytest.approx(0.0, abs=0.01)


class TestDatasetSummary:
    def test_it_reports_delistings(self, crsp_file: Path) -> None:
        summary = load_crsp_csv(crsp_file).summary()

        assert "1 securities delisted in-sample" in summary
        assert "1 worse than -50%" in summary

    def test_a_survivorship_filtered_extract_is_flagged(self) -> None:
        """Over a multi-year US equity sample nothing delisting does not happen."""
        prices = Panel(
            np.full((100, 3), 100.0),
            tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(100)),
            ("A", "B", "C"),
            "close",
        )

        dataset = dataset_from_panels(prices, prices.with_values(np.full((100, 3), 1e7)))

        assert "survivorship-filtered somewhere upstream" in dataset.summary()

    def test_dataset_from_panels_derives_listings(self) -> None:
        values = np.full((50, 2), 100.0)
        values[30:, 1] = np.nan
        prices = Panel(
            values,
            tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(50)),
            ("A", "B"),
            "close",
        )

        dataset = dataset_from_panels(prices, prices.with_values(np.full((50, 2), 1e7)))

        assert dataset.listings["A"][1] is None
        assert dataset.listings["B"][1] == date(2024, 1, 30)


class TestEndToEnd:
    def test_a_loaded_dataset_drives_a_backtest(self, crsp_file: Path) -> None:
        from crucible.ops import cs_demean, cs_rank, cs_scale
        from crucible.universe import Universe, listing_mask

        dataset = load_crsp_csv(crsp_file)
        universe = Universe(
            listing_mask(dataset.prices.index, dataset.assets, dataset.listings),
            dataset.prices.index,
            dataset.assets,
        )
        signal = universe.apply(cs_rank(dataset.prices.pct_change(20), min_breadth=2))
        weights = cs_scale(cs_demean(signal, min_breadth=2), min_breadth=2)

        result = backtest(
            weights,
            dataset.prices,
            costs=CostModel(),
            dollar_volume=dataset.dollar_volume,
            delisting_return=dataset.delisting_returns,
        )

        assert result.traded
        assert result.delisting_events == 1


CIZ_SCHEMA = {
    "PERMNO": pl.Int64,
    "DlyCalDt": pl.Utf8,
    "DlyPrc": pl.Float64,
    "DlyVol": pl.Float64,
    "DlyPrcVol": pl.Float64,
    "DlyRet": pl.Utf8,
    "ShrOut": pl.Float64,
    "ShareType": pl.Utf8,
    "PrimaryExch": pl.Utf8,
    "Ticker": pl.Utf8,
}

#: permno offset -> (sharetype, primaryexch). CIZ uses text codes, not numeric SHRCD/EXCHCD.
CIZ_SECURITIES = [
    ("NS", "N"),  # ordinary common, NYSE
    ("NS", "Q"),  # ordinary common, Nasdaq — delists mid-sample
    ("NS", "N"),
    ("ADR", "N"),  # ADR — filtered out by share type
    ("NS", "X"),  # unlisted venue — filtered out by exchange
]
CIZ_EXIT_ROW = 120


@pytest.fixture
def ciz_files(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(4)
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(200)]
    rows = []
    for offset, (share_type, exchange) in enumerate(CIZ_SECURITIES):
        prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, 200))
        exit_at = CIZ_EXIT_ROW if offset == 1 else None
        for i, when in enumerate(dates):
            if exit_at is not None and i > exit_at:
                continue
            rows.append(
                {
                    "PERMNO": 20000 + offset,
                    "DlyCalDt": when.isoformat(),
                    "DlyPrc": round(float(prices[i]), 4),
                    "DlyVol": float(rng.integers(1e5, 9e6)),
                    "DlyPrcVol": float(rng.integers(1e7, 5e8)),
                    "DlyRet": f"{rng.normal(0, 0.015):.6f}",
                    "ShrOut": 1e6,
                    "ShareType": share_type,
                    "PrimaryExch": exchange,
                    "Ticker": f"T{offset}",
                }
            )
    daily = tmp_path / "ciz_daily.csv"
    pl.DataFrame(rows, schema=CIZ_SCHEMA).write_csv(daily)

    delist = tmp_path / "ciz_delist.csv"
    pl.DataFrame({"PERMNO": [20001], "DLRET": [-0.72]}).write_csv(delist)
    return daily, delist


class TestCIZFormat:
    """CRSP discontinued the legacy SIZ format after December 2024. CIZ is not a rename — the
    date column, price, volume and return columns are all renamed, share and exchange codes
    became text, and the delisting return moved to a separate table joined on PERMNO."""

    def test_it_loads_a_ciz_daily_file(self, ciz_files: tuple[Path, Path]) -> None:
        daily, delist = ciz_files

        dataset = load_crsp_ciz(daily, delisting_path=delist)

        assert dataset.assets == ("20000", "20001", "20002")

    def test_text_share_types_filter_out_adrs(self, ciz_files: tuple[Path, Path]) -> None:
        """CIZ `sharetype` is 'NS' for ordinary common, not the numeric SHRCD 10/11."""
        daily, _ = ciz_files

        assert "20003" not in load_crsp_ciz(daily).assets

    def test_text_exchange_codes_filter_out_minor_venues(
        self, ciz_files: tuple[Path, Path]
    ) -> None:
        daily, _ = ciz_files

        assert "20004" not in load_crsp_ciz(daily).assets

    def test_filters_can_be_disabled(self, ciz_files: tuple[Path, Path]) -> None:
        daily, _ = ciz_files

        dataset = load_crsp_ciz(daily, share_types=None, exchanges=None)

        assert {"20003", "20004"} <= set(dataset.assets)

    def test_the_delisting_return_comes_from_the_separate_table(
        self, ciz_files: tuple[Path, Path]
    ) -> None:
        """The structural difference that matters most: DLRET is no longer a column in the
        daily file."""
        daily, delist = ciz_files

        assert load_crsp_ciz(daily, delisting_path=delist).delisting_returns == {"20001": -0.72}

    def test_without_the_delisting_table_there_are_no_returns(
        self, ciz_files: tuple[Path, Path]
    ) -> None:
        """Runs, but falls back to the engine's optimistic exit-at-last-price assumption."""
        daily, _ = ciz_files

        assert load_crsp_ciz(daily).delisting_returns == {}

    def test_dollar_volume_uses_crsps_own_column(self, ciz_files: tuple[Path, Path]) -> None:
        """CIZ supplies dlyprcvol directly; the legacy loader had to multiply volume by price."""
        daily, _ = ciz_files

        assert np.nanmin(load_crsp_ciz(daily).dollar_volume.values) >= 1e7

    def test_delistings_are_detected_from_the_daily_file(
        self, ciz_files: tuple[Path, Path]
    ) -> None:
        daily, delist = ciz_files

        listings = load_crsp_ciz(daily, delisting_path=delist).listings

        assert listings["20000"][1] is None
        assert listings["20001"][1] is not None

    def test_a_legacy_file_is_refused_with_a_pointer(self, tmp_path: Path) -> None:
        """Handing a SIZ export to the CIZ loader should say which loader to use."""
        path = tmp_path / "legacy.csv"
        pl.DataFrame({"PERMNO": [1], "date": ["2020-01-01"], "PRC": [10.0]}).write_csv(path)

        with pytest.raises(DataError, match="use load_crsp_csv instead"):
            load_crsp_ciz(path)

    def test_a_delisting_table_without_returns_is_refused(
        self, ciz_files: tuple[Path, Path]
    ) -> None:
        """Silently returning nothing would reinstate the exit-at-last-price assumption without
        saying so."""
        daily, _ = ciz_files
        bad = daily.parent / "bad_delist.csv"
        pl.DataFrame({"PERMNO": [1], "DelReasonType": ["M"]}).write_csv(bad)

        with pytest.raises(DataError, match="no recognisable delisting-return column"):
            load_crsp_ciz(daily, delisting_path=bad)

    def test_filters_excluding_everything_are_refused(self, ciz_files: tuple[Path, Path]) -> None:
        daily, _ = ciz_files

        with pytest.raises(DataError, match="not the numeric SHRCD/EXCHCD"):
            load_crsp_ciz(daily, share_types=("NOPE",))

    def test_the_dataset_shape_matches_the_legacy_loader(
        self, ciz_files: tuple[Path, Path]
    ) -> None:
        """Everything downstream must be unaffected by which format the data arrived in."""
        daily, delist = ciz_files

        dataset = load_crsp_ciz(daily, delisting_path=delist)

        assert isinstance(dataset, Dataset)
        assert dataset.prices.shape == dataset.dollar_volume.shape == dataset.returns.shape
        assert set(dataset.listings) <= set(dataset.assets)
