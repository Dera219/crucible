"""Point-in-time universe construction.

The most consequential module and the least interesting to read. Two things carry it: membership
is decided as of each date from trailing data only, and the hysteresis band stops names near the
threshold from oscillating in and out. Measured without a buffer, boundary churn alone produced
~105x book per year of turnover that no signal asked for.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from crucible.panel import Panel, PanelError
from crucible.universe import Universe, liquidity_screen, listing_mask

N_TIMES, N_ASSETS = 300, 40
INDEX = tuple(date(2023, 1, 1) + timedelta(days=i) for i in range(N_TIMES))
ASSETS = tuple(f"A{i:02d}" for i in range(N_ASSETS))


def flat_prices(value: float = 50.0) -> Panel:
    return Panel(np.full((N_TIMES, N_ASSETS), value), INDEX, ASSETS, "close")


def volumes(*, boundary_names: int = 0, seed: int = 11) -> Panel:
    """Volumes with `boundary_names` packed into a narrow band so their ranks churn on noise."""
    rng = np.random.default_rng(seed)
    clear = N_ASSETS - boundary_names
    base = np.concatenate([np.full(clear, 5e7), np.linspace(1.05e7, 1.0e7, boundary_names)])
    return Panel(base * np.exp(rng.normal(0, 0.30, (N_TIMES, N_ASSETS))), INDEX, ASSETS, "adv")


class TestListingCalendar:
    def test_an_asset_is_not_investable_before_it_lists(self) -> None:
        mask = listing_mask(INDEX, ASSETS, {"A00": (date(2023, 3, 1), None)})

        assert not mask[0, 0]
        assert mask[INDEX.index(date(2023, 3, 1)), 0]

    def test_an_asset_is_not_investable_after_it_delists(self) -> None:
        mask = listing_mask(INDEX, ASSETS, {"A00": (None, date(2023, 3, 1))})

        assert mask[INDEX.index(date(2023, 3, 1)), 0]
        assert not mask[INDEX.index(date(2023, 3, 2)), 0]

    def test_an_unknown_symbol_is_never_investable(self) -> None:
        """Silence about an asset is not permission to trade it. Defaulting the other way is how
        an unknown ticker quietly becomes a nine-year position."""
        mask = listing_mask(INDEX, ASSETS, {"A00": (None, None)})

        assert mask[:, 0].all()
        assert not mask[:, 1:].any()

    def test_unbounded_windows_are_always_investable(self) -> None:
        mask = listing_mask(INDEX, ASSETS, dict.fromkeys(ASSETS, (None, None)))

        assert mask.all()

    def test_a_backwards_calendar_is_refused(self) -> None:
        """It would silently produce an empty universe rather than an error."""
        with pytest.raises(PanelError, match="before listing"):
            listing_mask(INDEX, ASSETS, {"A00": (date(2023, 6, 1), date(2023, 1, 1))})


class TestLiquidityScreen:
    def test_low_volume_names_are_excluded(self) -> None:
        volume = volumes()
        quiet = volume.values.copy()
        quiet[:, 0] = 1_000.0
        mask = liquidity_screen(
            volume.with_values(quiet), flat_prices(), top_n=None, min_dollar_volume=1e6
        )

        assert not mask[-1, 0]
        assert mask[-1, 1]

    def test_penny_stocks_are_excluded(self) -> None:
        """Sub-$5 names have wide relative spreads, are often un-shortable, and are where
        cross-sectional signals go to produce fictional returns."""
        prices = flat_prices().values.copy()
        prices[:, 0] = 2.0
        mask = liquidity_screen(
            volumes(), flat_prices().with_values(prices), top_n=None, min_price=5.0
        )

        assert not mask[-1, 0]

    def test_the_screen_uses_trailing_averages_not_point_observations(self) -> None:
        """A single day's volume spikes on news; a universe built from it churns on noise."""
        volume = volumes().values.copy()
        volume[:, 0] = 1_000.0
        volume[150, 0] = 1e12  # one enormous day
        mask = liquidity_screen(
            volumes().with_values(volume),
            flat_prices(),
            top_n=None,
            min_dollar_volume=1e6,
            window=20,
        )

        # The spike lifts the trailing mean for a while, but not the whole series.
        assert not mask[100, 0]
        assert not mask[-1, 0]

    def test_top_n_limits_membership(self) -> None:
        mask = liquidity_screen(
            volumes(), flat_prices(), top_n=10, min_dollar_volume=0, min_price=0
        )

        assert mask[-1].sum() <= 15  # top_n plus the hysteresis band

    def test_top_n_none_keeps_everything_clearing_the_floors(self) -> None:
        mask = liquidity_screen(
            volumes(), flat_prices(), top_n=None, min_dollar_volume=0, min_price=0
        )

        assert mask[-1].all()

    def test_inverted_hysteresis_is_refused(self) -> None:
        """A name would need to be more liquid to stay than to join, which makes churn worse."""
        with pytest.raises(PanelError, match="inverts the hysteresis"):
            liquidity_screen(volumes(), flat_prices(), top_n=30, entry_rank=40, exit_rank=20)

    def test_misaligned_inputs_are_refused(self) -> None:
        other = Panel(np.ones((N_TIMES, N_ASSETS)), INDEX, tuple(f"B{i}" for i in range(N_ASSETS)))

        with pytest.raises(PanelError, match="must be aligned"):
            liquidity_screen(volumes(), other)


class TestHysteresis:
    """Without a buffer, boundary churn alone produced ~105x book/year of turnover."""

    def test_a_buffer_dramatically_reduces_churn(self) -> None:
        volume = volumes(boundary_names=25)
        common = {
            "prices": flat_prices(),
            "top_n": 20,
            "min_dollar_volume": 0.0,
            "min_price": 0.0,
            "window": 5,
        }

        none = Universe(
            liquidity_screen(volume, entry_rank=20, exit_rank=20, **common), INDEX, ASSETS
        ).audit()
        wide = Universe(
            liquidity_screen(volume, entry_rank=14, exit_rank=30, **common), INDEX, ASSETS
        ).audit()

        assert none.churn_per_period > 2.0
        assert wide.churn_per_period < none.churn_per_period / 10

    def test_the_buffer_widens_monotonically(self) -> None:
        volume = volumes(boundary_names=25)
        common = {
            "prices": flat_prices(),
            "top_n": 20,
            "min_dollar_volume": 0.0,
            "min_price": 0.0,
            "window": 5,
        }

        churn = [
            Universe(
                liquidity_screen(volume, entry_rank=enter, exit_rank=leave, **common),
                INDEX,
                ASSETS,
            )
            .audit()
            .churn_per_period
            for enter, leave in ((20, 20), (18, 23), (14, 30))
        ]

        assert churn[0] > churn[1] > churn[2]

    def test_incumbency_wins_inside_the_band(self) -> None:
        """A member sitting between the two thresholds must stay a member."""
        volume = volumes(boundary_names=25)
        mask = liquidity_screen(
            volume,
            flat_prices(),
            top_n=20,
            min_dollar_volume=0.0,
            min_price=0.0,
            window=5,
            entry_rank=14,
            exit_rank=30,
        )

        # Membership should be near-constant once established.
        settled = mask[100:]
        changes = np.abs(settled[1:].astype(int) - settled[:-1].astype(int)).sum()
        assert changes < len(settled) * 0.05


class TestUniverse:
    def test_masks_intersect(self) -> None:
        left = np.zeros((N_TIMES, N_ASSETS), dtype=bool)
        right = np.zeros((N_TIMES, N_ASSETS), dtype=bool)
        left[:, :20] = True
        right[:, 10:] = True

        universe = Universe.from_masks(left, right, index=INDEX, assets=ASSETS)

        assert universe.mask[:, :10].sum() == 0
        assert universe.mask[:, 10:20].all()
        assert universe.mask[:, 20:].sum() == 0

    def test_apply_blanks_non_members(self) -> None:
        """Applied to the signal, not just the weights: a non-member carrying a signal value
        still takes up a rank and shifts every cross-sectional statistic in that row."""
        mask = np.ones((N_TIMES, N_ASSETS), dtype=bool)
        mask[:, 0] = False
        universe = Universe(mask, INDEX, ASSETS)
        signal = Panel(np.ones((N_TIMES, N_ASSETS)), INDEX, ASSETS, "signal")

        applied = universe.apply(signal)

        assert np.isnan(applied.values[:, 0]).all()
        assert np.isfinite(applied.values[:, 1:]).all()

    def test_applying_to_a_misaligned_panel_is_refused(self) -> None:
        universe = Universe(np.ones((N_TIMES, N_ASSETS), dtype=bool), INDEX, ASSETS)
        other = Panel(np.ones((N_TIMES, 2)), INDEX, ("X", "Y"))

        with pytest.raises(PanelError, match="not aligned"):
            universe.apply(other)

    def test_members_on_a_date(self) -> None:
        mask = np.zeros((N_TIMES, N_ASSETS), dtype=bool)
        mask[:, :3] = True

        assert Universe(mask, INDEX, ASSETS).members_on(INDEX[0]) == ("A00", "A01", "A02")

    def test_an_unknown_date_is_refused(self) -> None:
        universe = Universe(np.ones((N_TIMES, N_ASSETS), dtype=bool), INDEX, ASSETS)

        with pytest.raises(PanelError, match="not in the index"):
            universe.members_on(date(1999, 1, 1))

    def test_a_mismatched_mask_is_refused(self) -> None:
        with pytest.raises(PanelError, match="does not match"):
            Universe(np.ones((5, 5), dtype=bool), INDEX, ASSETS)


class TestSurvivorshipDetection:
    def test_a_universe_nothing_ever_leaves_is_flagged(self) -> None:
        """Over a multi-year window a real equity universe loses names constantly. Zero exits
        means a survivors' list, and no masking here can recover what the source omits."""
        universe = Universe(np.ones((N_TIMES, N_ASSETS), dtype=bool), INDEX, ASSETS)

        audit = universe.audit()

        assert audit.looks_survivorship_filtered
        assert "survivors' list" in audit.summary()
        assert "foreknowledge" in audit.summary()

    def test_a_universe_with_delistings_is_not_flagged(self) -> None:
        mask = np.ones((N_TIMES, N_ASSETS), dtype=bool)
        mask[150:, 0] = False

        audit = Universe(mask, INDEX, ASSETS).audit()

        assert not audit.looks_survivorship_filtered
        assert audit.exits == 1

    def test_excessive_churn_is_flagged(self) -> None:
        rng = np.random.default_rng(0)
        mask = rng.random((N_TIMES, N_ASSETS)) < 0.5

        assert "membership changes per period" in Universe(mask, INDEX, ASSETS).audit().summary()

    def test_entries_and_exits_are_counted(self) -> None:
        mask = np.zeros((N_TIMES, N_ASSETS), dtype=bool)
        mask[100:200, 0] = True

        audit = Universe(mask, INDEX, ASSETS).audit()

        assert audit.entries == 1
        assert audit.exits == 1


class TestEndToEnd:
    def test_a_realistic_universe_composes_listings_and_liquidity(self) -> None:
        listings = {
            symbol: (
                date(2023, 1, 1) if i % 7 else date(2023, 4, 1),
                date(2023, 9, 1) if i % 11 == 0 else None,
            )
            for i, symbol in enumerate(ASSETS)
        }
        universe = Universe.from_masks(
            listing_mask(INDEX, ASSETS, listings),
            liquidity_screen(
                volumes(), flat_prices(), top_n=20, min_dollar_volume=0.0, min_price=0.0
            ),
            index=INDEX,
            assets=ASSETS,
            name="us-liquid",
        )

        audit = universe.audit()

        assert audit.exits > 0
        assert not audit.looks_survivorship_filtered
        assert 0 < audit.mean_members <= 30

    def test_the_universe_reaches_the_backtest(self) -> None:
        from crucible.costs import CostModel
        from crucible.engine import backtest
        from crucible.ops import cs_demean, cs_rank, cs_scale

        rng = np.random.default_rng(4)
        prices = Panel(
            np.cumprod(1 + rng.normal(0.0003, 0.014, (N_TIMES, N_ASSETS)), axis=0) * 100,
            INDEX,
            ASSETS,
            "close",
        )
        mask = np.ones((N_TIMES, N_ASSETS), dtype=bool)
        mask[:, 20:] = False
        universe = Universe(mask, INDEX, ASSETS)

        signal = universe.apply(cs_rank(prices.pct_change(20)))
        result = backtest(cs_scale(cs_demean(signal)), prices, costs=CostModel())

        assert np.all(result.held_weights[:, 20:] == 0.0)
        assert result.traded
