"""Score the sealed spin-off holder-unloading registration, ONCE.

    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/spinoff-holder-unloading/run_event_study.py --self-test
    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/spinoff-holder-unloading/run_event_study.py --final

`--self-test` exercises the whole path on fabricated data and touches nothing real: a planted
effect over the traded window must survive, pure noise must not, and the two holding rules below
must do what the registration says. `--final` is the trial. Before it reads a single return it
refuses unless:

1. `spinoff_holder_events.csv` hashes to the SHA-256 sealed in registration.json;
2. the hypothesis's fingerprint is registered in this folder's journal.jsonl;
3. RUN.lock can be created exclusively (O_EXCL), so this runs once. The lock is taken after the
   data is loaded (which reveals nothing) and before the study is computed.

The computation is crucible's own: each era's slice is loaded with `load_crsp_ciz` (common-stock
filter, price flags, delisting table), market-adjusted with `market_adjusted`, and the spinco
columns are stitched across eras so a window near a slice boundary keeps its full length.

## The two holding rules, registered

`crucible.events` (bug #14, fixed 2026-10-15) excludes an event with a missing day inside the
traded window, and that is right for a library that cannot know why the day is missing. This
script can, from CRSP's own conventions, and fills before the study runs:

- a **listed day without a trade** carries zero abnormal return: CRSP's next valid return is
  measured from the last trade, so the move across the gap lands there;
- a **spinco delisted inside the window** takes CRSP's delisting return on the session after its
  last trade (zero if CRSP records none: exit at the last price, the optimistic assumption, said
  out loud) and the proceeds earn zero raw return — minus the market — for the rest of the window.
  Dropping such events would select the sample on survival, in the direction that flatters it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
SLICES = ((2005, 2009), (2010, 2014), (2015, 2025))
POST = 150
TRADED_WINDOW = (31, 150)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fill_holding(
    abnormal: np.ndarray,
    market: np.ndarray,
    first: list[int | None],
    last: list[int | None],
    delisting: list[float | None],
) -> np.ndarray:
    """Apply the two holding rules to a (time, spinco) abnormal-return block. Pure.

    `first`/`last` are the index positions of each column's first and last tradable session,
    `delisting` CRSP's delisting return where it recorded one. Between first and last a NaN is
    a listed day without a trade: zero. After last, the delisting return minus the market, then
    minus the market: cash. Before first the column is left as it is; no window starts there.
    """
    out = abnormal.copy()
    n = out.shape[0]
    for column in range(out.shape[1]):
        start, stop = first[column], last[column]
        if start is None or stop is None:
            continue
        held = out[start : stop + 1, column]
        held[np.isnan(held)] = 0.0
        out[start : stop + 1, column] = held
        if stop + 1 < n:
            paid = delisting[column]
            recovered = paid if paid is not None and math.isfinite(paid) else 0.0
            out[stop + 1, column] = recovered - market[stop + 1]
            out[stop + 2 :, column] = -market[stop + 2 :]
    return out


def stitched_panels(data: Path, delist: Path, spincos: list[str]) -> tuple[Panel, Panel]:
    """(returns, abnormal) for the spinco columns only, across every era, holding rules applied."""
    index: list[date] = []
    returns_rows: list[np.ndarray] = []
    abnormal_rows: list[np.ndarray] = []
    market_parts: list[np.ndarray] = []
    first_dates: dict[str, date] = {}
    last_dates: dict[str, date] = {}
    delisting: dict[str, float] = {}
    for start, end in SLICES:
        dataset = load_crsp_ciz(data / f"crsp_{start}_{end}.csv.gz", delisting_path=delist)
        abnormal = market_adjusted(dataset.returns)
        with np.errstate(invalid="ignore"):
            market_parts.append(np.nanmean(dataset.returns.values, axis=1))
        position = {asset: column for column, asset in enumerate(dataset.returns.assets)}
        take = [position.get(asset) for asset in spincos]
        for source, sink in ((dataset.returns, returns_rows), (abnormal, abnormal_rows)):
            block = np.full((len(source.index), len(spincos)), np.nan)
            for column, found in enumerate(take):
                if found is not None:
                    block[:, column] = source.values[:, found]
            sink.append(block)
        for asset in spincos:
            listing = dataset.listings.get(asset)
            if listing is None:
                continue
            opened, closed = listing
            if opened is not None:
                first_dates[asset] = min(first_dates.get(asset, opened), opened)
            if closed is not None:
                last_dates[asset] = max(last_dates.get(asset, closed), closed)
            if asset in dataset.delisting_returns:
                delisting[asset] = dataset.delisting_returns[asset]
        index.extend(
            stamp.date() if isinstance(stamp, datetime) else stamp
            for stamp in dataset.returns.index
        )
        present = sum(found is not None for found in take)
        print(f"  {start}-{end}: {len(dataset.returns.index)} sessions, "
              f"{present} of {len(spincos)} spincos present", flush=True)
        del dataset, abnormal
    positions = {stamp: row for row, stamp in enumerate(index)}
    first = [positions.get(first_dates.get(a, date.min)) for a in spincos]
    last = [positions.get(last_dates.get(a, date.min)) for a in spincos]
    paid = [delisting.get(a) for a in spincos]
    market = np.concatenate(market_parts)
    filled = fill_holding(np.vstack(abnormal_rows), market, first, last, paid)
    returns = Panel(np.vstack(returns_rows), tuple(index), tuple(spincos), "returns")
    abnormal_panel = Panel(filled, tuple(index), tuple(spincos), "returns_abnormal")
    return returns, abnormal_panel


def judge(
    returns: Panel, abnormal: Panel, events: list[Event], kill: EventKillCriteria
) -> EventVerdict:
    study = event_study(returns, events, pre=0, post=POST, benchmark=abnormal)
    print(f"events placed: {study.n_events}  dropped: {study.dropped}")
    return evaluate(study, kill, traded_window=TRADED_WINDOW)


def self_test() -> int:
    """Fabricated data only. A planted effect must survive; noise must be killed; the holding
    rules must do what the registration says."""
    from sealed_hypothesis import KILL  # this folder's sealed criteria

    rng = np.random.default_rng(11)
    sessions = [date(2006, 1, 2) + timedelta(days=i) for i in range(1_400)]
    assets = [f"S{i:03d}" for i in range(120)]
    outcomes: dict[str, bool] = {}
    for label, bump in (("planted 12% over sessions 31-150", 0.001), ("pure noise", 0.0)):
        values = rng.normal(0.0, 0.02, size=(len(sessions), len(assets)))
        events = []
        for column, asset in enumerate(assets):
            day = 20 + 10 * column
            values[day + 31 : day + 151, column] += bump
            events.append(Event(asset, sessions[day]))
        panel = Panel(values, tuple(sessions), tuple(assets), "returns")
        verdict = judge(panel, panel, events, KILL)
        outcomes[label] = verdict.survived
        print(f"{label}: {verdict}\n")
    ok = outcomes["planted 12% over sessions 31-150"] and not outcomes["pure noise"]

    # The holding rules, exactly. One column, ten sessions, market +1% a day.
    market = np.full(10, 0.01)
    block = np.full((10, 1), np.nan)
    block[2:7, 0] = 0.02  # listed sessions 2..6; session 4 has no trade
    block[4, 0] = np.nan
    filled = fill_holding(block, market, first=[2], last=[6], delisting=[-0.5])
    expect = [np.nan, np.nan, 0.02, 0.02, 0.0, 0.02, 0.02, -0.5 - 0.01, -0.01, -0.01]
    rules = all(
        (math.isnan(a) and math.isnan(b)) or math.isclose(a, b)
        for a, b in zip(filled[:, 0].tolist(), expect, strict=True)
    )
    none_recorded = fill_holding(block, market, first=[2], last=[6], delisting=[None])
    rules = rules and math.isclose(none_recorded[7, 0], -0.01)  # exit at the last price
    still_listed = fill_holding(block, market, first=[2], last=[9], delisting=[None])
    rules = rules and all(math.isnan(x) is False for x in still_listed[2:, 0].tolist())
    print("holding rules:", "OK" if rules else "FAILED")
    ok = ok and rules
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def final(data: Path, delist: Path) -> int:
    from sealed_hypothesis import HYPOTHESIS, KILL

    registration = json.loads((HERE / "registration.json").read_text(encoding="utf-8"))
    events_csv = data / "spinoff_holder_events.csv"
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
