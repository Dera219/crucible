"""The sealed registration: spin-off orphan forced selling. Trial 6 against the CRSP universe.

    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/spinoff-orphans/sealed_hypothesis.py

Running this file SEALS the registration: it writes registration.json (every field that defines
the claim, its fingerprint, and the SHA-256 of the event set and of every script that touches
it) and records the hypothesis in this folder's journal.jsonl. Only the SHA-256 of
registration.json is published (crucible's SEARCH_LOG.md); everything here stays in this private
repository until it is revealed. Revealing it means publishing registration.json, whose hash
anyone can then check against the published one.

Approved by Chidera 2026-10-14. The mechanism is his sentence, verbatim.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from crucible.events import EventKillCriteria
from crucible.preregistration import EdgeSource, Hypothesis, ResearchJournal

HERE = Path(__file__).resolve().parent

#: t >= 2.64 rather than the default 2.0. EventKillCriteria carries require_deflation_survival,
#: but crucible's event `evaluate` never applies it, so the bar is corrected by hand instead:
#: Bonferroni over the 6 lifetime trials (5 exploratory + this one), two-sided 5%, gives
#: z = 2.638. min_mean_car = 0.025 was committed 2026-08-24, before any spin-off CAR existed.
KILL = EventKillCriteria(
    min_events=30,
    min_mean_car=0.025,
    min_t_statistic=2.64,
    min_win_rate=0.5,
    max_clustering=0.5,
    max_dropped_fraction=0.2,
)

HYPOTHESIS = Hypothesis(
    name="spinoff-orphan-forced-selling",
    claim=(
        "Spin-offs of S&P 500 companies that the index does not keep are sold by index funds at "
        "their first close, and earn a positive abnormal return from the next session's close "
        "through two sessions after."
    ),
    edge_source=EdgeSource.LIQUIDITY_PROVISION,
    mechanism=(
        "The traders who buy the shares the index funds are forced to dump at the spin-off's "
        "first close are middlemen who can't afford to hold a small, risky new stock, so they "
        "sell it to me below fair value for about 2 days until they've unloaded."
    ),
    universe=(
        "CRSP CIZ daily US common stock (crucible's CIZ common-stock filter), 1990-2025. Events: "
        "distributions with DisType SP and DisDetailType SECSO whose parent is an S&P 500 member "
        "on the ex-date; the spinco is a new listing whose first daily observation (its first "
        "regular-way trade, offset 0) is within 5 days of the ex-date, is US common stock that "
        "day, and is NOT an S&P 500 member on the next session. 105 events: 79 before "
        "2015-10-01, 26 after (14 one-day members under S&P DJI's post-2015 spin-off rule, 12 "
        "never added). Abnormal = return minus the equal-weighted investable mean. Traded "
        "window: offsets 2-3, i.e. buy at the day-1 close, sell at the day-3 close."
    ),
    horizon_bars=3,
    warmup_bars=0,
    parameters=(),
    kill=KILL,
    trial_budget=6,
    notes=(
        "Sealed 2026-10-14. Two regimes pooled (pre-2015 dispersed selling, post-2015 a single "
        "closing sale) because the 26 post-2015 events alone resolve only ~4.9% at the 11-day "
        "sigma and could not reach the 2.5% bar. Window shortened from 2-10 to 2-3 on his "
        "mechanism (unloading lasts ~2 days); the 2.5% bar was deliberately NOT moved with it. "
        "Run once with run_event_study.py --final; RUN.lock enforces it."
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
        "lifetime_trials_before": 5,
    }
    out = HERE / "registration.json"
    if out.exists():
        raise SystemExit(f"{out} exists: this registration is sealed and cannot be rewritten")
    out.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ResearchJournal(HERE / "journal.jsonl").register(HYPOTHESIS)
    return out


if __name__ == "__main__":
    sealed = seal(Path.home() / "Downloads" / "spinoff_orphan_events.csv")
    print(f"sealed {sealed}\nfingerprint {HYPOTHESIS.fingerprint}\nSHA-256 {sha256(sealed)}")
