"""An event claim, shown end to end — as a TEMPLATE, not a recommendation.

    PYTHONPATH=$PWD .venv/bin/python scripts/example_event_hypothesis.py

`example_hypothesis.py` shows the shape for a **cross-sectional** claim: a number that ranks
thousands of names every day, judged by information coefficient and quantile monotonicity. This
file shows the other shape, because a claim about corporate actions and forced flows is not a
ranking at all. It is forty discrete bets a year, nothing held most days, and passing it through
the cross-sectional criteria means relaxing them until they admit anything.

The example below uses **index reconstitution**, deliberately, for the same reason the other file
uses a decayed factor: it is the most studied forced flow in existence, everybody can see the
same calendar, and nobody should mistake it for a tip. It is here to show what every field looks
like when taken seriously.

## Why this file cannot contain YOUR hypothesis

The same reason as the other one. The mechanism is a claim about how the world works that you
have to actually believe, because you are the one holding the position while it is not working.
A mechanism you did not reason your way to is one you abandon at the first drawdown, and every
real edge has one.

## The check to run before you spend a trial

An event claim dies of capacity more often than of signal, and capacity is computable on paper
before any data is touched. Look 4 in SEARCH_LOG.md — odd-lot tender provisions — had a real
mechanism, a documented rule, and a genuine moat. It died on ~18 reachable events a year capped
at 99 shares, roughly $5,400 a year, and that number needed no query at all.

The sharper form of the question is not "how much does this pay" but **is the ceiling set by the
mechanism or by me?** Ninety-nine shares is a regulatory cap that no amount of capital lifts.
A forced seller distributing a $500M spin-off is limited only by how much you can deploy.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from crucible.events import (
    Event,
    EventKillCriteria,
    evaluate,
    event_study,
)
from crucible.panel import Panel
from crucible.preregistration import EdgeSource, Hypothesis

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


HYPOTHESIS = Hypothesis(
    name="index-reconstitution-forced-selling",
    claim=(
        "Names deleted from a small-cap index at reconstitution are sold by mandate-bound "
        "funds and are cheap for a short window afterwards."
    ),
    edge_source=EdgeSource.STRUCTURAL_CONSTRAINT,
    mechanism=(
        "An index fund tracking a small-cap benchmark must hold the index and nothing else. "
        "When a name is deleted at reconstitution the fund has to sell its entire position "
        "regardless of price, on a schedule published in advance, because tracking error is the "
        "only thing it is measured on and a cheap stock it is not permitted to own is worth "
        "exactly nothing to it. The seller is not expressing a view; it is discharging an "
        "obligation. Whoever takes the other side is paid for absorbing an urgent, "
        "price-insensitive flow. NOTE: this is a TEMPLATE. The flow is enormous, the calendar is "
        "public, and the effect has been documented for decades, so the counterparty here is not "
        "a constrained fund but every other participant who read the same paper."
    ),
    universe="US common stock, CRSP CIZ; small-cap index deletions at reconstitution",
    horizon_bars=10,
    warmup_bars=0,
    parameters=(),
    kill=EventKillCriteria(
        min_events=30,
        min_mean_car=0.005,
        min_t_statistic=2.0,
        min_win_rate=0.5,
        # Reconstitution is one day a year. This bar is the one this hypothesis should fail on,
        # and it is set here on purpose to show what an honest criterion looks like when it
        # rules out the very claim being registered.
        max_clustering=0.5,
    ),
    # Five exploratory trials are already spent — see SEARCH_LOG.md. A budget that ignores them
    # deflates against a search that did not happen.
    trial_budget=6,
    notes="Template only. See SEARCH_LOG.md before adding a sixth trial.",
)


def synthetic_events() -> tuple[Panel, list[Event]]:
    """Fabricated data, so the script runs without the licensed extract.

    Forty deletions that all fall on the same reconstitution day — which is exactly the shape
    that makes a t-statistic across events dishonest, and exactly what the criteria should catch.
    """
    rng = np.random.default_rng(0)
    n_times, n_assets = 260, 60
    index = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_times)]
    assets = [f"DELETED{i:02d}" for i in range(n_assets)]
    values = rng.normal(0.0, 0.01, size=(n_times, n_assets))

    recon_day = 120
    values[recon_day + 1 : recon_day + 6, :40] += 0.012  # the post-deletion bounce

    returns = Panel(values, index, assets, "returns")
    events = [Event(assets[i], index[recon_day]) for i in range(40)]
    return returns, events


def main() -> None:
    print(HYPOTHESIS.name)
    print(f"fingerprint: {HYPOTHESIS.fingerprint}\n")

    returns, events = synthetic_events()
    study = event_study(returns, events, pre=5, post=10)

    traded = study.car_over(1, 5)
    print(f"events placed: {study.n_events}  dropped: {study.dropped}")
    print(f"mean CAR (full window):   {study.mean_car:+.4f}")
    print(f"mean CAR (traded 1..5):   {float(np.nanmean(traded)):+.4f}")
    print(f"t across events:          {study.t_statistic:+.2f}")
    print(f"win rate:                 {study.win_rate:.2f}")
    print(f"clustering:               {study.clustering:.0%}\n")

    verdict = evaluate(study, HYPOTHESIS.kill, traded_window=(1, 5))
    print(verdict)
    print()
    print(
        "Killed on clustering alone, and correctly. Every other number passes: the mean CAR is\n"
        "comfortably positive, the t-statistic clears 2, the win rate clears half. That is the\n"
        "point. Forty events sharing one date is closer to one observation than to forty, and\n"
        "nothing in the headline numbers can tell you so."
    )


if __name__ == "__main__":
    main()
