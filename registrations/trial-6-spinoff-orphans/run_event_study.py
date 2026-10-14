"""Score the sealed spin-off orphan registration, ONCE.

    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/spinoff-orphans/run_event_study.py --self-test
    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/spinoff-orphans/run_event_study.py --final

`--self-test` exercises the whole path on fabricated data and touches nothing real: a planted
2-day effect must survive, and pure noise must not. `--final` is the trial. Before it reads a
single return it refuses unless:

1. `spinoff_orphan_events.csv` hashes to the SHA-256 sealed in registration.json, so the event
   set cannot have been edited after sealing;
2. the hypothesis's fingerprint is registered in this folder's journal.jsonl, so the claim was
   written down before the evidence;
3. RUN.lock can be created exclusively (O_EXCL), so this runs once. The lock is taken after the
   data is loaded (which reveals nothing, so a crash there may be re-run) and before the study is
   computed. A second `--final` that reaches that point fails, whatever happened to the first.

The computation is crucible's own, not a reimplementation: each era's slice is loaded with
`load_crsp_ciz` (common-stock filter, price flags, delisting returns), market-adjusted with
`market_adjusted` (equal-weighted investable mean), and the spinco columns are stitched across
eras so an event near a slice boundary keeps its full window instead of being dropped. The
stitched panel goes to `event_study(pre=0, post=3)` and the verdict to
`evaluate(..., traded_window=(2, 3))`: buy at the day-1 close, sell at the day-3 close.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from crucible.data import load_crsp_ciz
from crucible.events import (
    Event,
    EventKillCriteria,
    EventVerdict,
    evaluate,
    event_study,
    market_adjusted,
)
from crucible.panel import Panel
from crucible.preregistration import ResearchJournal, Verdict

HERE = Path(__file__).resolve().parent
SLICES = ((1990, 1994), (1995, 1999), (2000, 2004), (2005, 2009), (2010, 2014), (2015, 2025))
TRADED_WINDOW = (2, 3)
POST = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stitched_panels(data: Path, delist: Path, spincos: list[str]) -> tuple[Panel, Panel]:
    """(returns, abnormal) for the spinco columns only, across every era, in one index."""
    index: list[date] = []
    returns_rows: list[np.ndarray] = []
    abnormal_rows: list[np.ndarray] = []
    for start, end in SLICES:
        dataset = load_crsp_ciz(data / f"crsp_{start}_{end}.csv.gz", delisting_path=delist)
        abnormal = market_adjusted(dataset.returns)
        position = {asset: column for column, asset in enumerate(dataset.returns.assets)}
        take = [position.get(asset) for asset in spincos]
        for source, sink in ((dataset.returns, returns_rows), (abnormal, abnormal_rows)):
            block = np.full((len(source.index), len(spincos)), np.nan)
            for column, found in enumerate(take):
                if found is not None:
                    block[:, column] = source.values[:, found]
            sink.append(block)
        index.extend(
            stamp.date() if isinstance(stamp, datetime) else stamp
            for stamp in dataset.returns.index
        )
        present = sum(found is not None for found in take)
        print(f"  {start}-{end}: {len(dataset.returns.index)} sessions, "
              f"{present} of {len(spincos)} spincos present", flush=True)
        del dataset, abnormal
    returns = Panel(np.vstack(returns_rows), tuple(index), tuple(spincos), "returns")
    abnormal = Panel(np.vstack(abnormal_rows), tuple(index), tuple(spincos), "returns_abnormal")
    return returns, abnormal


def judge(
    returns: Panel, abnormal: Panel, events: list[Event], kill: EventKillCriteria
) -> EventVerdict:
    study = event_study(returns, events, pre=0, post=POST, benchmark=abnormal)
    print(f"events placed: {study.n_events}  dropped: {study.dropped}")
    return evaluate(study, kill, traded_window=TRADED_WINDOW)


def self_test() -> int:
    """Fabricated data only. A planted effect must survive; noise must be killed."""
    from sealed_hypothesis import KILL  # this folder's sealed criteria

    rng = np.random.default_rng(7)
    sessions = [date(2001, 1, 2) + timedelta(days=i) for i in range(400)]
    assets = [f"S{i:03d}" for i in range(105)]
    outcomes = {}
    for label, bump in (("planted 3% over days 2-3", 0.015), ("pure noise", 0.0)):
        values = rng.normal(0.0, 0.02, size=(len(sessions), len(assets)))
        events = []
        for column, asset in enumerate(assets):
            day = 20 + 3 * column
            values[day + 2 : day + 4, column] += bump
            events.append(Event(asset, sessions[day]))
        panel = Panel(values, tuple(sessions), tuple(assets), "returns")
        verdict = judge(panel, panel, events, KILL)
        outcomes[label] = verdict.survived
        print(f"{label}: {verdict}\n")
    ok = outcomes["planted 3% over days 2-3"] and not outcomes["pure noise"]
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def final(data: Path, delist: Path) -> int:
    from sealed_hypothesis import HYPOTHESIS, KILL

    registration = json.loads((HERE / "registration.json").read_text(encoding="utf-8"))
    events_csv = data / "spinoff_orphan_events.csv"
    if sha256(events_csv) != registration["provenance"]["events_csv_sha256"]:
        sys.exit(f"{events_csv} does not match the sealed SHA-256; refusing to run")
    journal = ResearchJournal(HERE / "journal.jsonl")
    fingerprint = HYPOTHESIS.fingerprint
    if not journal.is_registered(fingerprint) or fingerprint != registration["fingerprint"]:
        sys.exit(f"{fingerprint} is not the registered, sealed hypothesis; refusing to run")

    frame = pl.read_csv(events_csv, try_parse_dates=True)
    events = [Event(str(row["spinco"]), row["day0"]) for row in frame.iter_rows(named=True)]
    spincos = sorted({event.asset for event in events})
    # Loading reveals nothing — no abnormal return is aggregated, printed, or compared yet — so a
    # crash here costs no trial and may be re-run. The lock is taken AFTER it and BEFORE the
    # study: that boundary is where the one permitted look begins.
    returns, abnormal = stitched_panels(data, delist, spincos)
    descriptor = os.open(HERE / "RUN.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        stamp = datetime.now().astimezone().isoformat()
        handle.write(json.dumps({"fingerprint": fingerprint, "started": stamp}) + "\n")
    verdict = judge(returns, abnormal, events, KILL)
    print(f"\n{verdict}")

    journal.record_trial(
        fingerprint,
        sharpe=verdict.t_statistic,
        label="event study; the sharpe field holds the t-statistic across events",
    )
    if verdict.survived:
        outcome = Verdict.SURVIVED
    elif verdict.underpowered:
        outcome = Verdict.INVALID
    else:
        outcome = Verdict.KILLED
    journal.record_verdict(fingerprint, outcome, ())
    result = {
        "fingerprint": fingerprint,
        "finished": datetime.now().astimezone().isoformat(),
        "verdict": outcome.value,
        "mean_car": verdict.mean_car,
        "t_statistic": verdict.t_statistic,
        "win_rate": verdict.win_rate,
        "n_events": verdict.n_events,
        "clustering": verdict.clustering,
        "detectable_car": verdict.detectable_car,
        "underpowered": verdict.underpowered,
        "reasons": list(verdict.reasons),
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--data", type=Path, default=Path.home() / "Downloads")
    parser.add_argument(
        "--delist", type=Path, default=Path.home() / "Grind" / "crucible" / "crsp_delist.csv"
    )
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    return self_test() if args.self_test else final(args.data, args.delist)


if __name__ == "__main__":
    sys.exit(main())
