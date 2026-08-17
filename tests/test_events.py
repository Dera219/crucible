"""The event-study layer.

The central test is `TestAgainstAPlantedEffect`: returns are built with a known abnormal drift
on known dates, and the study must recover it at roughly the size it was planted. A measurement
that cannot find an effect you put there yourself will report "nothing here" for every real one
you ever have, and you will believe it — the same reasoning that shaped `test_diagnostics.py`
after an off-by-one measured a true IC of 0.13 at 0.008.

The second concern is the one that makes event studies specifically dangerous:
`TestClustering` builds forty events that all fall on the same day, where the honest number of
independent observations is closer to one. The t-statistic cannot see this, so `evaluate` has to.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from crucible.events import (
    Event,
    EventError,
    EventKillCriteria,
    evaluate,
    event_study,
    market_adjusted,
)
from crucible.panel import Panel

N_TIMES = 260
N_ASSETS = 60


def calendar(n: int = N_TIMES) -> list[date]:
    return [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]


def zero_returns() -> Panel:
    """A market with no noise at all.

    Used where the mechanics are being tested exactly rather than statistically. With random
    noise underneath, an assertion about a planted effect is really an assertion about the seed.
    """
    assets = [f"A{i:03d}" for i in range(N_ASSETS)]
    return Panel(np.zeros((N_TIMES, N_ASSETS)), calendar(), assets, "returns")


def flat_returns(seed: int = 0, scale: float = 0.01) -> Panel:
    """A market with no drift anywhere — the null the planted effect is measured against."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, scale, size=(N_TIMES, N_ASSETS))
    assets = [f"A{i:03d}" for i in range(N_ASSETS)]
    return Panel(values, calendar(), assets, "returns")


class TestAgainstAPlantedEffect:
    def test_recovers_a_planted_drift_at_roughly_its_size(self):
        returns = flat_returns()
        values = returns.values.copy()
        # 40 events, well separated, each followed by +20bp/day for five days.
        events = []
        for n in range(40):
            row = 30 + n * 5
            column = n % N_ASSETS
            values[row + 1 : row + 6, column] += 0.002
            events.append(Event(returns.assets[column], returns.index[row]))

        study = event_study(returns.with_values(values), events, pre=5, post=10)
        traded = study.car_over(1, 5)

        # Planted 5 x 20bp = 100bp. Noise is large per event, so assert the neighbourhood.
        assert study.n_events == 40
        assert 0.006 < float(np.nanmean(traded)) < 0.014

    def test_finds_nothing_when_nothing_was_planted(self):
        returns = flat_returns(seed=7)
        events = [Event(returns.assets[n % N_ASSETS], returns.index[30 + n * 5]) for n in range(40)]

        verdict = evaluate(event_study(returns, events), EventKillCriteria())

        # The point of the whole apparatus: pure noise must not survive.
        assert not verdict.survived
        assert abs(verdict.t_statistic) < 2.0

    def test_pre_event_window_is_excluded_from_the_traded_result(self):
        returns = flat_returns()
        values = returns.values.copy()
        events = []
        for n in range(40):
            row = 30 + n * 5
            column = n % N_ASSETS
            # Drift placed entirely BEFORE the event date.
            values[row - 4 : row, column] += 0.01
            events.append(Event(returns.assets[column], returns.index[row]))

        study = event_study(returns.with_values(values), events, pre=5, post=10)

        # Visible in the full-window CAR, absent from what a strategy could have traded. A study
        # that reported the former would be claiming profit on information it did not have.
        assert float(np.nanmean(study.car)) > 0.02
        assert abs(float(np.nanmean(study.car_over(0, 10)))) < 0.01


class TestAbnormalReturns:
    def test_market_adjustment_removes_a_common_shock(self):
        returns = flat_returns()
        values = returns.values.copy()
        values[100, :] += 0.05  # every asset falls or rises together
        adjusted = market_adjusted(returns.with_values(values))

        # A day when everything moved together is not an abnormal day for anybody.
        assert abs(float(np.nanmean(adjusted.values[100, :]))) < 1e-9

    def test_a_market_wide_crash_does_not_become_an_event_result(self):
        returns = zero_returns()
        values = returns.values.copy()
        # Events clustered in a drawdown: without adjustment this reads as a discovery.
        for row in range(100, 140):
            values[row, :] -= 0.02
        events = [Event(returns.assets[n % N_ASSETS], returns.index[100 + n]) for n in range(35)]

        study = event_study(returns.with_values(values), events, pre=2, post=5)

        # Every asset moved identically, so nothing is abnormal for anybody.
        assert abs(study.mean_car) < 1e-9


class TestClustering:
    def test_events_sharing_a_date_are_reported(self):
        returns = flat_returns()
        same_day = returns.index[100]
        events = [Event(returns.assets[i], same_day) for i in range(40)]

        study = event_study(returns, events, pre=2, post=5)

        assert study.clustering == 1.0

    def test_evaluate_refuses_a_study_that_is_one_day_wearing_forty_hats(self):
        returns = flat_returns()
        values = returns.values.copy()
        same_day = 100
        # A single day on which these 40 names happen to jump. One observation, not forty.
        values[same_day + 1, :40] += 0.05
        events = [Event(returns.assets[i], returns.index[same_day]) for i in range(40)]

        study = event_study(returns.with_values(values), events, pre=2, post=5)
        verdict = evaluate(study, EventKillCriteria())

        assert not verdict.survived
        assert any("share a date" in reason for reason in verdict.reasons)

    def test_well_separated_events_are_not_penalised(self):
        returns = flat_returns()
        events = [Event(returns.assets[n % N_ASSETS], returns.index[30 + n * 5]) for n in range(40)]
        assert event_study(returns, events, pre=2, post=5).clustering == 0.0


class TestDroppedEvents:
    def test_events_outside_the_panel_are_counted_not_silently_ignored(self):
        returns = flat_returns()
        events = [
            Event(returns.assets[0], returns.index[100]),
            Event("NOT_IN_UNIVERSE", returns.index[100]),
            Event(returns.assets[0], date(1990, 1, 1)),
        ]

        study = event_study(returns, events, pre=2, post=5)

        assert study.n_events == 1
        assert study.dropped == 2

    def test_a_window_running_off_the_end_is_dropped_rather_than_truncated(self):
        returns = flat_returns()
        events = [
            Event(returns.assets[0], returns.index[1]),  # not enough history before
            Event(returns.assets[0], returns.index[-2]),  # not enough future after
        ]

        study = event_study(returns, events, pre=5, post=20)

        # A truncated window is a different measurement wearing the same name.
        assert study.n_events == 0
        assert study.dropped == 2

    def test_evaluate_refuses_when_too_many_events_were_dropped(self):
        returns = flat_returns()
        real = [Event(returns.assets[n % N_ASSETS], returns.index[30 + n * 5]) for n in range(40)]
        missing = [Event("NOT_IN_UNIVERSE", returns.index[100]) for _ in range(40)]

        study = event_study(returns, [*real, *missing], pre=2, post=5)
        verdict = evaluate(
            study, EventKillCriteria(min_events=10, min_t_statistic=0.0, min_win_rate=0.0)
        )

        assert not verdict.survived
        assert any("dropped" in reason for reason in verdict.reasons)


class TestKillCriteria:
    def test_too_few_events_is_fatal_however_good_the_number_looks(self):
        returns = flat_returns()
        values = returns.values.copy()
        events = []
        for n in range(5):
            row = 30 + n * 10
            values[row + 1, n] += 0.5  # enormous, and on five events
            events.append(Event(returns.assets[n], returns.index[row]))

        verdict = evaluate(event_study(returns.with_values(values), events), EventKillCriteria())

        assert not verdict.survived
        assert any("below the minimum" in reason for reason in verdict.reasons)

    def test_a_mean_carried_by_two_outliers_fails_on_win_rate(self):
        returns = zero_returns()
        values = returns.values.copy()
        events = []
        for n in range(40):
            row = 30 + n * 5
            column = n % N_ASSETS
            if n < 2:
                values[row + 1, column] += 1.0  # two vast winners
            else:
                values[row + 1, column] -= 0.001  # everything else slightly negative
            events.append(Event(returns.assets[column], returns.index[row]))

        verdict = evaluate(
            event_study(returns.with_values(values), events),
            EventKillCriteria(min_t_statistic=0.0),
        )

        assert not verdict.survived
        assert any("win rate" in reason for reason in verdict.reasons)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_events": 1},
            {"min_win_rate": 1.5},
            {"max_clustering": -0.1},
            {"max_dropped_fraction": 2.0},
            {"min_t_statistic": -1.0},
        ],
    )
    def test_nonsense_criteria_are_refused_at_construction(self, kwargs):
        with pytest.raises(ValueError):
            EventKillCriteria(**kwargs)


class TestWindowValidation:
    def test_a_zero_length_window_is_refused(self):
        with pytest.raises(EventError):
            event_study(flat_returns(), [], pre=0, post=0)

    def test_negative_windows_are_refused(self):
        with pytest.raises(EventError):
            event_study(flat_returns(), [], pre=-1, post=5)

    def test_an_inverted_traded_window_is_refused(self):
        returns = flat_returns()
        study = event_study(returns, [Event(returns.assets[0], returns.index[100])])
        with pytest.raises(EventError):
            study.car_over(5, 1)

    def test_an_unnamed_asset_is_refused(self):
        with pytest.raises(EventError):
            Event("", date(2020, 1, 1))

    def test_an_empty_study_reports_rather_than_crashes(self):
        returns = flat_returns()
        verdict = evaluate(event_study(returns, []), EventKillCriteria())

        assert not verdict.survived
        assert verdict.n_events == 0


class TestRegistration:
    """An event claim has to be registrable, or the instrument is unreachable.

    `Hypothesis` hashes its criteria so that moving a bar after seeing a result changes the
    fingerprint. Criteria that cannot serialise cannot be committed to, and criteria the type
    system rejects cannot be registered at all — either way the discipline is bypassed by simply
    not using it.
    """

    def hypothesis(self, **overrides):
        from crucible.preregistration import EdgeSource, Hypothesis

        defaults = dict(
            name="spin-off orphan selling",
            claim="Index funds sell spun-off shares they cannot hold, at any price.",
            edge_source=EdgeSource.STRUCTURAL_CONSTRAINT,
            mechanism="x" * 200,
            universe="US common stock, CRSP CIZ",
            horizon_bars=20,
            warmup_bars=0,
            parameters=(),
            kill=EventKillCriteria(),
            trial_budget=6,
        )
        return Hypothesis(**{**defaults, **overrides})

    def test_event_criteria_can_stand_where_cross_sectional_ones_do(self):
        assert isinstance(self.hypothesis().kill, EventKillCriteria)

    def test_the_criteria_are_serialised_into_the_fingerprint(self):
        loose = self.hypothesis(kill=EventKillCriteria(min_t_statistic=2.0))
        relaxed = self.hypothesis(kill=EventKillCriteria(min_t_statistic=0.5))

        # Lowering a bar after seeing a result must be visible as a different claim.
        assert loose.fingerprint != relaxed.fingerprint

    def test_the_parameter_cap_is_still_enforced(self):
        with pytest.raises(ValueError):
            self.hypothesis(parameters=("a", "b", "c", "d"), kill=EventKillCriteria())

    def test_assess_refuses_an_event_claim_instead_of_scoring_absent_fields(self):
        from crucible.preregistration import Evidence, PreregistrationError, assess

        with pytest.raises(PreregistrationError, match="events.evaluate"):
            assess(self.hypothesis(), Evidence(trials_used=1))


class TestStatisticalPower:
    """ "Could not be seen" and "was not there" are different findings.

    Only one of them retires an idea. A study whose sample could never have reached the required
    t-statistic, whatever the true effect, has measured its own size — grading that as a kill
    discards a claim that was never tested, which is why `preregistration` keeps INVALID separate
    from KILLED.
    """

    def test_detectable_effect_shrinks_with_the_square_root_of_events(self):
        from crucible.events import minimum_detectable_car

        # Four times the events, half the detectable effect.
        assert minimum_detectable_car(100, 0.20) == pytest.approx(0.04)
        assert minimum_detectable_car(400, 0.20) == pytest.approx(0.02)

    def test_a_single_event_can_detect_nothing(self):
        from crucible.events import minimum_detectable_car

        assert minimum_detectable_car(1, 0.20) == float("inf")

    def test_an_underpowered_study_is_uninformative_rather_than_killed(self):
        returns = flat_returns(seed=3)
        # 40 events against noisy returns: the resolvable effect is far above a 1% bar.
        events = [Event(returns.assets[n % N_ASSETS], returns.index[30 + n * 5]) for n in range(40)]

        verdict = evaluate(
            event_study(returns, events, pre=2, post=10),
            EventKillCriteria(min_events=30, min_mean_car=0.01),
        )

        assert not verdict.survived
        assert verdict.underpowered
        assert verdict.detectable_car > 0.01
        assert any("underpowered" in reason for reason in verdict.reasons)
        assert "UNINFORMATIVE" in str(verdict)

    def test_a_well_powered_failure_is_still_a_kill(self):
        returns = zero_returns()
        values = returns.values.copy()
        events = []
        for n in range(40):
            row = 30 + n * 5
            column = n % N_ASSETS
            # A precise, tiny, consistent effect: resolvable, and below the bar.
            values[row + 1, column] += 0.001
            events.append(Event(returns.assets[column], returns.index[row]))

        verdict = evaluate(
            event_study(returns.with_values(values), events, pre=2, post=10),
            EventKillCriteria(min_events=30, min_mean_car=0.01),
        )

        # The sample could see the effect perfectly well; the effect is simply too small.
        assert not verdict.survived
        assert not verdict.underpowered
        assert "KILLED" in str(verdict)
