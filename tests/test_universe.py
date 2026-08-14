"""Point-in-time universe construction.

The most consequential module and the least interesting to read. Two things carry it: membership
is decided as of each date from trailing data only, and the hysteresis band stops names near the
threshold from oscillating in and out. Measured without a buffer, boundary churn alone produced
~105x book per year of turnover that no signal asked for.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import numpy.typing as npt
import pytest

from crucible.panel import Panel, PanelError
from crucible.universe import Universe, liquidity_screen, listing_mask, price_floor_screen

N_TIMES, N_ASSETS = 300, 40
INDEX = tuple(date(2023, 1, 1) + timedelta(days=i) for i in range(N_TIMES))
ASSETS = tuple(f"A{i:02d}" for i in range(N_ASSETS))


def flat_prices(value: float = 50.0) -> Panel:
    return Panel(np.full((N_TIMES, N_ASSETS), value), INDEX, ASSETS, "close")


def raw_prices(values: npt.NDArray[np.float64]) -> Panel:
    """The unadjusted panel a price floor must read: `Dataset.raw_prices`, never `prices`."""
    return Panel(values, INDEX, ASSETS, "close_raw")


def wandering_prices(*, around: float = 5.4, seed: int = 7) -> Panel:
    """Prices loitering near the $5 line, which is what makes a bare floor churn.

    Mean-reverting rather than a random walk on purpose: a walk drifts away from the boundary
    and stops crossing it, which would test the buffer against the one population that does not
    need it.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, 0.05, (N_TIMES, N_ASSETS))
    log_price = np.zeros((N_TIMES, N_ASSETS))
    for row in range(1, N_TIMES):
        log_price[row] = 0.85 * log_price[row - 1] + shocks[row]
    return Panel(around * np.exp(log_price), INDEX, ASSETS, "close_raw")


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


class TestPriceFloor:
    """Below $5 on the tape, US common stock returned -0.64% annualised over 2015-2025 with a
    median 21-day return of -4.00%, survivorship-free. At or above it, +9.85%. The screen removes
    a value-destroying segment; that is hygiene, not an edge."""

    def test_names_below_the_floor_are_excluded_and_names_above_are_kept(self) -> None:
        prices = np.full((N_TIMES, N_ASSETS), 50.0)
        prices[:, 0] = 3.0

        mask = price_floor_screen(raw_prices(prices), min_price=5.0)

        assert not mask[:, 0].any()
        assert mask[:, 1:].all()

    def test_the_floor_is_inclusive_at_the_boundary(self) -> None:
        """Pinned deliberately: exactly $5.00 clears a $5.00 floor, matching the `>=` that
        liquidity_screen already uses, so the two floors cannot disagree about the same cent."""
        at_the_line = np.full((N_TIMES, N_ASSETS), 5.0)
        one_cent_under = np.full((N_TIMES, N_ASSETS), 4.99)

        assert price_floor_screen(raw_prices(at_the_line), min_price=5.0, entry_price=5.0).all()
        assert not price_floor_screen(
            raw_prices(one_cent_under), min_price=5.0, entry_price=5.0
        ).any()

    def test_at_the_boundary_an_incumbent_stays_and_a_newcomer_does_not_join(self) -> None:
        """The consequence of a buffered entry, pinned so it cannot surprise anyone: with the
        default 1.2x buffer the effective admission price is $6.00, and $5.00 exactly is enough
        to keep a name but not to admit one."""
        incumbent = np.full((N_TIMES, N_ASSETS), 5.0)
        incumbent[:100] = 7.0

        mask = price_floor_screen(raw_prices(incumbent), min_price=5.0)

        assert mask.all()
        assert not price_floor_screen(
            raw_prices(np.full((N_TIMES, N_ASSETS), 5.0)), min_price=5.0
        ).any()

    def test_a_name_that_never_reaches_the_entry_price_is_never_admitted(self) -> None:
        """A name sitting at $5.50 forever clears the floor and is still excluded, because it
        has never traded at the $6.00 admission price. The $5-6 zone is the weakest slice above
        the floor — +2.57% annualised, median -1.58% — so deferring it costs little."""
        assert not price_floor_screen(
            raw_prices(np.full((N_TIMES, N_ASSETS), 5.50)), min_price=5.0
        ).any()

    def test_a_missing_raw_price_is_excluded_rather_than_silently_kept(self) -> None:
        prices = np.full((N_TIMES, N_ASSETS), 50.0)
        prices[100:150, 0] = np.nan

        mask = price_floor_screen(raw_prices(prices))

        assert not mask[100:150, 0].any()

    def test_a_halt_does_not_eject_a_name_that_was_fine_either_side_of_it(self) -> None:
        """A missing price is not evidence about the floor in either direction. Ejecting on it
        would manufacture a round trip out of a data gap."""
        prices = np.full((N_TIMES, N_ASSETS), 50.0)
        prices[100:150, 0] = np.nan

        mask = price_floor_screen(raw_prices(prices))

        assert mask[99, 0]
        assert mask[150, 0]

    def test_an_absent_raw_price_panel_is_refused(self) -> None:
        """Dataset.raw_prices is None for loaders that supply no unadjusted series. Falling back
        to the adjusted panel would report a screen as applied that removes nothing."""
        with pytest.raises(PanelError, match="worse than no screen"):
            price_floor_screen(None)

    def test_a_dataset_with_no_raw_panel_reaches_that_refusal(self) -> None:
        """The path that actually happens: `dataset_from_panels` cannot know whether what it was
        handed is adjusted, so it carries no raw panel, and `dataset.raw_prices` passes straight
        into the screen and stops there rather than silently becoming the adjusted one."""
        from crucible.data import dataset_from_panels

        dataset = dataset_from_panels(flat_prices(), volumes())

        assert dataset.raw_prices is None
        with pytest.raises(PanelError, match="dataset_from_panels"):
            price_floor_screen(dataset.raw_prices)

    def test_the_screen_reads_the_raw_panel_and_not_the_adjusted_one(self) -> None:
        """The whole point. Both panels are handed the same $5 floor and they disagree, in both
        directions, exactly as they disagree on 195,447 liquid name-days of the real extract.

        A00 has split 4:1 — it trades at $3 and its total-return index still reads $12, so an
        adjusted floor keeps a penny stock. A01 has reverse-split 1:10, the move a failing
        company makes to hold its listing — it trades at $8 and its return index never moved,
        so an adjusted floor throws out a name that is genuinely above the line.
        """
        tape = np.full((N_TIMES, N_ASSETS), 50.0)
        adjusted = np.full((N_TIMES, N_ASSETS), 50.0)
        tape[:, 0], adjusted[:, 0] = 3.0, 12.0
        tape[:, 1], adjusted[:, 1] = 8.0, 0.80

        from_raw = price_floor_screen(raw_prices(tape), min_price=5.0)
        from_adjusted = price_floor_screen(raw_prices(adjusted), min_price=5.0)

        assert not from_raw[:, 0].any(), "a $3 stock was kept: the screen read the adjusted panel"
        assert from_raw[:, 1].all(), "an $8 stock was dropped: the screen read the adjusted panel"
        assert from_adjusted[:, 0].all()
        assert not from_adjusted[:, 1].any()

    def test_exit_is_not_buffered_and_takes_effect_the_same_session(self) -> None:
        """Entry is where patience is cheap; exit is where it is expensive. Every extra session
        a name is held below the floor is a session inside a -4.00% median."""
        prices = np.full((N_TIMES, N_ASSETS), 20.0)
        prices[150:, 0] = 3.0

        mask = price_floor_screen(raw_prices(prices), min_price=5.0)

        assert mask[149, 0]
        assert not mask[150:, 0].any()

    def test_a_trailing_average_would_have_held_the_crashed_name_and_this_does_not(self) -> None:
        """Stated as a contrast because it is the one place this screen deliberately breaks
        liquidity_screen's rule. Volume spikes and needs smoothing; a price moves, and a name
        that fell from $20 to $2 is a $2 stock today, not a $17 stock for another ten sessions."""
        prices = np.full((N_TIMES, N_ASSETS), 20.0)
        prices[150:, 0] = 2.0
        panel = raw_prices(prices)

        mask = price_floor_screen(panel, min_price=5.0)
        trailing = panel.rolling(20, "mean", min_periods=10).values[:, 0]

        assert not mask[150:, 0].any()
        assert (trailing[150:160] >= 5.0).all()

    def test_the_entry_buffer_cuts_boundary_churn(self) -> None:
        """A bare $5 line manufactured 61.2% of the book per year in forced round trips on the
        real extract — worse than the 37.7% that made the rank band too narrow to keep. Requiring
        $6.00 to join cut it to 22.9% for 1.0% of mean membership."""
        prices = wandering_prices(around=5.5)

        bare = price_floor_screen(prices, min_price=5.0, entry_price=5.0)
        buffered = price_floor_screen(prices, min_price=5.0)

        bare_churn = Universe(bare, INDEX, ASSETS).audit().churn_per_period
        buffered_churn = Universe(buffered, INDEX, ASSETS).audit().churn_per_period
        assert bare_churn > 0.5
        assert buffered_churn < bare_churn / 2

    def test_a_wider_buffer_churns_less(self) -> None:
        prices = wandering_prices(around=5.5)

        churn = [
            Universe(
                price_floor_screen(prices, min_price=5.0, entry_price=entry), INDEX, ASSETS
            )
            .audit()
            .churn_per_period
            for entry in (5.0, 5.5, 6.0, 7.5)
        ]

        assert churn == sorted(churn, reverse=True)

    def test_an_entry_price_below_the_floor_is_refused(self) -> None:
        """A name would have to be cheaper to join than to stay, which inverts the hysteresis."""
        with pytest.raises(PanelError, match="inverts the hysteresis"):
            price_floor_screen(raw_prices(np.full((N_TIMES, N_ASSETS), 50.0)), entry_price=1.0)

    def test_a_negative_floor_is_refused(self) -> None:
        """A price panel uses a negative number to mean 'no trade happened', so a negative floor
        admits exactly the rows nobody could have transacted at."""
        with pytest.raises(PanelError, match="no trade happened"):
            price_floor_screen(raw_prices(np.full((N_TIMES, N_ASSETS), 50.0)), min_price=-1.0)

    def test_it_composes_with_the_listing_calendar_and_the_liquidity_screen(self) -> None:
        """Screens intersect: an asset must pass every one. A liquid, listed, well-ranked name
        that trades at $2 is still not investable."""
        tape = np.full((N_TIMES, N_ASSETS), 50.0)
        tape[:, 0] = 2.0

        universe = Universe.from_masks(
            listing_mask(INDEX, ASSETS, dict.fromkeys(ASSETS, (None, None))),
            liquidity_screen(
                volumes(), flat_prices(), top_n=None, min_dollar_volume=0.0, min_price=0.0
            ),
            price_floor_screen(raw_prices(tape), min_price=5.0),
            index=INDEX,
            assets=ASSETS,
            name="us-liquid-above-5",
        )

        assert not universe.mask[:, 0].any()
        # From row 20, past liquidity_screen's rolling warmup, everything else is investable.
        assert universe.mask[20:, 1:].all()
        assert "A00" not in universe.members_on(INDEX[-1])

    def test_a_name_falling_through_the_floor_leaves_the_universe_visibly(self) -> None:
        """It shows up as an exit in the audit rather than as a silent hole in the panel."""
        tape = np.full((N_TIMES, N_ASSETS), 50.0)
        tape[150:, 0] = 1.0

        audit = Universe(
            price_floor_screen(raw_prices(tape), min_price=5.0), INDEX, ASSETS
        ).audit()

        assert audit.exits == 1
        assert not audit.looks_survivorship_filtered


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
