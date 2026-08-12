"""A completed hypothesis, shown end to end — as a TEMPLATE, not a recommendation.

    PYTHONPATH=$PWD .venv/bin/python scripts/example_hypothesis.py

The mechanism below is deliberately one whose edge is **widely believed to have largely decayed
since the 1990s**, precisely so that nobody mistakes this file for a tip. It is here to show what
a complete pre-registration looks like when every field is taken seriously, because "write a
mechanism" is much easier to act on with one worked example in front of you than with a
description of the form.

Read it, then throw the content away and keep the shape.

## Why this file cannot contain YOUR hypothesis

The mechanism is a claim about how the world works that you have to actually believe, because you
are the one who will hold the position through the months when it is not working. Nobody can
supply conviction on your behalf, and a mechanism you did not reason your way to is one you will
abandon at exactly the wrong moment — the first drawdown, which every real edge has.

There is also a plainer reason. A mechanism written from a worldview months out of date, by
something with no stake in the outcome, is not research. It is a plausible paragraph, and a
plausible paragraph is exactly what this entire library exists to distinguish from an edge.

## How to use it

Copy `HYPOTHESIS` into a scratch file. Replace `mechanism` with your own, and keep replacing it
until you can name the counterparty without hand-waving. Then set the kill criteria at numbers you
will actually honour when the run comes back below them — that is the part everybody softens
afterwards, which is why the fingerprint exists.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from crucible.preregistration import (
    EdgeSource,
    Evidence,
    Hypothesis,
    KillCriteria,
    ResearchJournal,
    Verdict,
    assess,
)

# ── The mechanism. The only field that cannot be filled in mechanically. ─────────────────────
#
# What makes this one complete rather than a slogan: it names who trades, why they trade, why
# they do not care about price, and therefore what you are being paid for. Every clause is a
# claim that could be checked and could be wrong — which is the point.
MECHANISM = """
Index funds and ETFs tracking a benchmark are obliged to hold its constituents in benchmark
weight. When the index provider adds a company, those funds must buy it by the effective date
regardless of the price at that moment, because their mandate is minimising tracking error and
not achieving a good execution price. Demand that is both large and price-insensitive arrives on
a known date, and it has to be filled by somebody willing to sell into it.

That somebody is compensated. The compensation is the gap between the price drift from
announcement to effective date and the partial reversion afterwards, once the forced buying has
cleared and the security trades on its own merits again. The seller is providing liquidity to a
buyer who cannot wait and cannot negotiate, which is a service with a price.

The effect is expected to be strongest where forced demand is largest relative to normal traded
volume — smaller additions, thinner names — and to have weakened substantially since the 1990s as
the trade became widely known and index providers moved to less predictable, less pre-announced
reconstitution schedules. Any surviving edge should therefore be small, concentrated in the least
liquid additions, and capacity-constrained.
""".strip()


HYPOTHESIS = Hypothesis(
    name="index-inclusion-drift",
    claim=(
        "Additions to the benchmark drift up between announcement and effective date and "
        "partially revert afterwards, by enough to beat an equal-weight baseline in a majority "
        "of out-of-sample folds, net of costs."
    ),
    edge_source=EdgeSource.STRUCTURAL_CONSTRAINT,
    mechanism=MECHANISM,
    universe="US common shares (SHRCD 10/11) on NYSE/AMEX/NASDAQ, top 1500 by trailing 60-day "
    "dollar volume, price above $5",
    horizon_bars=10,
    # Above the longest lookback the signal uses. Setting this too low is what made Apex's
    # walk-forward report UNTESTABLE, correctly, on a signal that had simply never warmed up.
    warmup_bars=90,
    parameters=("announcement_window", "reversion_window"),
    kill=KillCriteria(
        min_ic=0.015,
        min_ic_t_statistic=2.5,
        min_monotonicity=0.6,
        min_oos_fold_win_rate=0.55,
        max_annual_turnover=8.0,
        require_deflation_survival=True,
        max_parameters=2,
        # An edge with no capacity is a hobby. This is the number the square-root cost model
        # exists to answer, and it belongs in the kill criteria rather than in a footnote.
        min_capacity=250_000.0,
    ),
    # Chosen knowing the arithmetic: more trials raise the bar the winner must clear. Against a
    # 50-trial search, an annualised Sharpe near 1.27 needs roughly twenty years of daily data to
    # separate itself from luck. A large budget is not generosity, it is a debt.
    trial_budget=8,
    notes="Template only. Effect widely believed largely arbitraged since the 1990s.",
)


def main() -> int:
    print(__doc__.split("\n\n")[0])
    print()
    print("── the hypothesis, as registered ──")
    print(f"  name        {HYPOTHESIS.name}")
    print(f"  edge source {HYPOTHESIS.edge_source.value}")
    print(f"  universe    {HYPOTHESIS.universe}")
    print(f"  horizon     {HYPOTHESIS.horizon_bars} bars, warmup {HYPOTHESIS.warmup_bars}")
    print(f"  parameters  {list(HYPOTHESIS.parameters)}")
    print(f"  budget      {HYPOTHESIS.trial_budget} trials")
    print(f"  fingerprint {HYPOTHESIS.fingerprint}")

    journal = ResearchJournal(Path(tempfile.mkdtemp()) / "journal.jsonl")
    fingerprint = journal.register(HYPOTHESIS)
    print("\n── registered before any data was touched ──")
    print(f"  {journal.summarize()}")

    print("\n── what the fingerprint is for ──")
    softened = Hypothesis(
        **{
            **{f: getattr(HYPOTHESIS, f) for f in HYPOTHESIS.__slots__},
            "kill": KillCriteria(
                min_ic=0.0,
                min_ic_t_statistic=0.0,
                min_monotonicity=-1.0,
                min_oos_fold_win_rate=0.0,
                max_annual_turnover=1e6,
                require_deflation_survival=False,
                max_parameters=2,
            ),
        }
    )
    print(f"  original bar : {HYPOTHESIS.fingerprint}")
    print(f"  softened bar : {softened.fingerprint}")
    print("  -> a different claim, not the original one that happened to survive.")
    try:
        journal.record_trial(softened.fingerprint, sharpe=9.9)
        print("  evidence for the softened claim: ACCEPTED (bad)")
    except Exception as exc:  # PreregistrationError
        print(f"  evidence for the softened claim: refused — {str(exc).split('.')[0]}")

    print("\n── three ways the run can come back ──")
    baseline = {
        "ic_mean": 0.021,
        "ic_t_statistic": 3.1,
        "monotonicity": 0.72,
        "oos_fold_win_rate": 0.62,
        "folds_scored": 9,
        "folds_testable": True,
        "annual_turnover": 5.4,
        "survived_deflation": True,
        "trials_used": 6,
        "capacity": 1_100_000.0,
        "test_window_bars": 252,
    }
    outcomes = {
        "a clean result": Evidence(**baseline),
        "no edge (IC below the bar)": Evidence(
            **{**baseline, "ic_mean": 0.004, "ic_t_statistic": 0.9}
        ),
        "never warmed up": Evidence(**{**baseline, "test_window_bars": 60}),
        "budget blown by a sweep": Evidence(**{**baseline, "trials_used": 200}),
    }
    for label, evidence in outcomes.items():
        verdict, violations = assess(HYPOTHESIS, evidence)
        print(f"\n  {label}: {verdict.value.upper()}")
        for violation in violations[:2]:
            print(f"    - {violation.criterion}: {violation.detail[:96]}")
        if verdict is Verdict.SURVIVED:
            print("    - nothing to report. Worth paper-forwarding, which is not the same as")
            print("      worth funding.")
        journal.record_verdict(fingerprint, verdict, violations)

    print("\n" + "─" * 78)
    print("The shape is the deliverable. The content has to be yours — you are the one who will")
    print("hold the position through the months it is not working, and a mechanism you did not")
    print("reason your way to is one you will abandon at the first drawdown every edge has.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
