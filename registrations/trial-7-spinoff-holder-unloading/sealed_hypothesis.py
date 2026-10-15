"""The sealed registration: spin-off holder unloading. Trial 7 against the CRSP universe.

    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/spinoff-holder-unloading/sealed_hypothesis.py

Running this file SEALS the registration: it writes registration.json (every field that defines
the claim, its fingerprint, and the SHA-256 of the event set and of every script that touches
it) and records the hypothesis in this folder's journal.jsonl. Only the SHA-256 of
registration.json is published (crucible's SEARCH_LOG.md); everything here stays in this private
repository until it is revealed.

The mechanism was drafted by Claude on 2026-10-15 at Chidera's instruction, critiqued by him, and
ruled research rather than search by him the same day, in the knowledge that trial 6 (the S&P
500 orphan subset of this population, a 2-day window) had been killed the day before.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from crucible.events import EventKillCriteria
from crucible.preregistration import EdgeSource, Hypothesis, ResearchJournal

HERE = Path(__file__).resolve().parent

#: t >= 2.69: Bonferroni over the 7 lifetime trials (5 exploratory looks, trial 6, this one),
#: two-sided 5%, z = 2.690 — crucible's event `evaluate` never applies deflation itself, so the
#: bar is corrected by hand, as trial 6 did at 2.64. min_mean_car = 0.07 is the smallest effect
#: ~290 events can resolve at that bar if the per-event dispersion over 120 sessions is near 40%;
#: a wider dispersion makes the study UNINFORMATIVE, which is registered as the risk it is.
KILL = EventKillCriteria(
    min_events=30,
    min_mean_car=0.07,
    min_t_statistic=2.69,
    min_win_rate=0.5,
    max_clustering=0.5,
    max_dropped_fraction=0.2,
)

HYPOTHESIS = Hypothesis(
    name="spinoff-holder-unloading",
    claim=(
        "A spun-off company's shares are distributed to the parent's holders, most of whom "
        "cannot keep them, and are sold down over the first weeks of trading; bought at the "
        "close of the 30th session after the first trade and held to the close of the 150th, "
        "they earn a positive abnormal return."
    ),
    edge_source=EdgeSource.STRUCTURAL_CONSTRAINT,
    mechanism=(
        "The parent's holders receive the spinco whether they want it or not, and a large "
        "share of them cannot keep it: index funds hold it only if it qualifies for their "
        "index, which the small ones do not; large-cap and dividend mandates hold a stock of "
        "the parent's size and yield, not a smaller one with none; many holders never chose "
        "it at all. They sell it over the first several weeks, into a name with no coverage "
        "and no natural owner yet, so the price is pressed or flat while they unload and "
        "recovers over the following months as the sellers finish and the first owners who "
        "chose it arrive. Funds cannot deploy there: the names are small, the holding is "
        "months, and the position is one a committee would not sign off; a small account "
        "with no committee can."
    ),
    universe=(
        "CRSP CIZ daily US common stock (crucible's CIZ common-stock filter), 2005-2025. "
        "Events: distributions with DisType SP and DisDetailType SECSO (a spin-off) whose "
        "spinco is a new listing whose first daily observation (its first regular-way trade, "
        "offset 0) is within 5 days of the ex-date, is US common stock that day, and first "
        "traded on or after 2005-01-01; the parent's index membership is not a condition. "
        "298 events, 33% sharing a day-0 date. Abnormal = return minus the equal-weighted "
        "investable mean of the slice. A listed day without a trade carries zero abnormal "
        "return (CRSP's next valid return carries the move); a spinco delisted inside the "
        "window takes CRSP's delisting return on the session after its last trade and the "
        "proceeds earn zero raw return, i.e. minus the market, for the rest of the window. "
        "Traded window: offsets 31-150, i.e. buy at the session-30 close, sell at the "
        "session-150 close, 120 sessions held."
    ),
    horizon_bars=150,
    warmup_bars=0,
    parameters=(),
    kill=KILL,
    trial_budget=7,
    notes=(
        "Sealed 2026-10-15. Window from the mechanism's predicted PATH (pressed or flat for "
        "weeks, then recovery over months), per trial 6's lesson, not from its duration. The "
        "entry offset 30 and exit 150 are fixed in code, not searched. Registered on the "
        "2005-2025 regime only. Trial 6 (S&P 500 orphans, offsets 2-3) found -1.35% in a "
        "window this claim says is still the unloading; that result is consistent with this "
        "claim and is not evidence for it. Run once with run_event_study.py --final; "
        "RUN.lock enforces it."
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(events_csv: Path) -> Path:
    registration = {
        "sealed_at": datetime.now().astimezone().isoformat(),
        "fingerprint": HYPOTHESIS.fingerprint,
        "hypothesis": HYPOTHESIS.canonical(),
        "notes": HYPOTHESIS.notes,
        "provenance": {
            "events_csv_sha256": sha256(events_csv),
            "events_csv_rows": sum(1 for _ in events_csv.open(encoding="utf-8")) - 1,
            "build_events_py_sha256": sha256(HERE / "build_events.py"),
            "run_event_study_py_sha256": sha256(HERE / "run_event_study.py"),
            "sealed_hypothesis_py_sha256": sha256(HERE / "sealed_hypothesis.py"),
        },
        "lifetime_trials_before": 6,
    }
    out = HERE / "registration.json"
    if out.exists():
        raise SystemExit(f"{out} exists: this registration is sealed and cannot be rewritten")
    out.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ResearchJournal(HERE / "journal.jsonl").register(HYPOTHESIS)
    return out


if __name__ == "__main__":
    sealed = seal(Path.home() / "Downloads" / "spinoff_holder_events.csv")
    print(f"sealed {sealed}\nfingerprint {HYPOTHESIS.fingerprint}\nSHA-256 {sha256(sealed)}")
