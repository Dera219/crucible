"""Score the sealed deal-break-unwind registration, ONCE.

    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/deal-break-unwinds/run_event_study.py --self-test
    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/deal-break-unwinds/run_event_study.py --final

`--self-test` exercises the whole path on fabricated data and touches nothing real: a planted
effect in the high-block half must survive while the low-block half stays flat; pure noise must
not; the same effect planted in BOTH halves must die on the mechanism bar; the holding rules and
the day-0 placement must do what the registration says. `--final` is the trial. Before it reads
a single return it refuses unless:

1. `deal_break_registration_set.csv` hashes to the SHA-256 sealed in registration.json;
2. the hypothesis's fingerprint is registered in this folder's journal.jsonl;
3. RUN.lock can be created exclusively (O_EXCL), so this runs once. The lock is taken after the
   data is loaded (which reveals nothing) and before the study is computed.

The computation is crucible's own: each era's slice is loaded with `load_crsp_ciz` (common-stock
filter, price flags, delisting table), market-adjusted with `market_adjusted`, and the targets'
columns are stitched across eras so a window near a slice boundary keeps its full length. The
two halves of the registration set are scored as two event studies with the same holding rules;
the high-block half carries the sealed KILL bars, the low-block half is the control, and the
difference of their traded-window means is graded against MECHANISM_GAP.

## Registered rules

- **day 0** is the first session on or after the 8-K's period of report (a termination filed
  on a weekend or holiday lands on the next session; none is shifted earlier);
- a **listed day without a trade** carries zero abnormal return: CRSP's next valid return is
  measured from the last trade, so the move across the gap lands there;
- a **target delisted inside the window** takes CRSP's delisting return on the session after its
  last trade (zero if CRSP records none: exit at the last price, the optimistic assumption, said
  out loud) and the proceeds earn zero raw return — minus the market — for the rest of the
  window. A target taken out by a new bidder inside the window is therefore kept, at its
  delisting return: that is part of what a buyer at +5 gets;
- the **pressure window** (offsets 0-5) is printed for the record and never graded.
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
POST = 60
PRESSURE_WINDOW = (0, 5)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fill_holding(
    abnormal: np.ndarray,
    market: np.ndarray,
    first: list[int | None],
    last: list[int | None],
    delisting: list[float | None],
) -> np.ndarray:
    """Apply the holding rules to a (time, target) abnormal-return block. Pure.

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


def place(day0: date, sessions: list[date]) -> date | None:
    """The first session on or after day 0, or None past the end of the data."""
    for stamp in sessions:  # sessions are sorted; the list is a few thousand long
        if stamp >= day0:
            return stamp
    return None


def stitched_panels(data: Path, delist: Path, targets: list[str]) -> tuple[Panel, Panel]:
    """(returns, abnormal) for the target columns only, across every era, holding rules applied."""
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
        take = [position.get(asset) for asset in targets]
        for source, sink in ((dataset.returns, returns_rows), (abnormal, abnormal_rows)):
            block = np.full((len(source.index), len(targets)), np.nan)
            for column, found in enumerate(take):
                if found is not None:
                    block[:, column] = source.values[:, found]
            sink.append(block)
        for asset in targets:
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
              f"{present} of {len(targets)} targets present", flush=True)
        del dataset, abnormal
    positions = {stamp: row for row, stamp in enumerate(index)}
    first = [positions.get(first_dates.get(a, date.min)) for a in targets]
    last = [positions.get(last_dates.get(a, date.min)) for a in targets]
    paid = [delisting.get(a) for a in targets]
    market = np.concatenate(market_parts)
    filled = fill_holding(np.vstack(abnormal_rows), market, first, last, paid)
    returns = Panel(np.vstack(returns_rows), tuple(index), tuple(targets), "returns")
    abnormal_panel = Panel(filled, tuple(index), tuple(targets), "returns_abnormal")
    return returns, abnormal_panel


def judge(
    returns: Panel, abnormal: Panel, events: list[Event], kill: EventKillCriteria,
    traded_window: tuple[int, int], label: str,
) -> EventVerdict:
    study = event_study(returns, events, pre=0, post=POST, benchmark=abnormal)
    pressure = study.car_over(*PRESSURE_WINDOW)
    pressure = pressure[np.isfinite(pressure)]
    pressed = float(pressure.mean()) if pressure.size else float("nan")
    print(f"{label}: events placed {study.n_events}  dropped {study.dropped}  "
          f"pressure window {PRESSURE_WINDOW} mean CAR {pressed:+.2%} (not graded)")
    return evaluate(study, kill, traded_window=traded_window)


def grade(high: EventVerdict, low: EventVerdict, gap: float) -> tuple[bool, str]:
    """The registered conjunction: the high-block half survives its bars AND beats the low-block
    half by at least `gap` over the traded window."""
    if not (math.isfinite(high.mean_car) and math.isfinite(low.mean_car)):
        return False, "a half has no measurable mean; the mechanism bar cannot be graded"
    spread = high.mean_car - low.mean_car
    if spread < gap:
        return False, (f"mechanism bar failed: high {high.mean_car:+.2%} minus low "
                       f"{low.mean_car:+.2%} = {spread:+.2%} < {gap:+.2%}")
    if not high.survived:
        return False, "the high-block half did not survive its bars: " + "; ".join(high.reasons)
    return True, f"high {high.mean_car:+.2%} minus low {low.mean_car:+.2%} = {spread:+.2%}"


def self_test() -> int:
    """Fabricated data only."""
    from sealed_hypothesis import KILL, MECHANISM_GAP, TRADED_WINDOW

    rng = np.random.default_rng(11)
    sessions = [date(2006, 1, 2) + timedelta(days=i) for i in range(1_400)]
    assets = [f"T{i:03d}" for i in range(120)]
    lo, hi = TRADED_WINDOW
    outcomes: dict[str, bool] = {}
    cases = (
        ("planted 15% in the high half only", 0.15 / (hi - lo + 1), 0.0),
        ("pure noise", 0.0, 0.0),
        ("planted 15% in BOTH halves", 0.15 / (hi - lo + 1), 0.15 / (hi - lo + 1)),
    )
    for label, bump_high, bump_low in cases:
        values = rng.normal(0.0, 0.02, size=(len(sessions), len(assets)))
        high_events, low_events = [], []
        for column, asset in enumerate(assets):
            day = 20 + 10 * column
            is_high = column % 2 == 0
            values[day + lo : day + hi + 1, column] += bump_high if is_high else bump_low
            (high_events if is_high else low_events).append(Event(asset, sessions[day]))
        panel = Panel(values, tuple(sessions), tuple(assets), "returns")
        high = judge(panel, panel, high_events, KILL, TRADED_WINDOW, "high")
        low = judge(panel, panel, low_events, KILL, TRADED_WINDOW, "low")
        ok, why = grade(high, low, MECHANISM_GAP)
        outcomes[label] = ok
        print(f"{label}: {'SURVIVED' if ok else 'KILLED'} — {why}\n")
    ok = (outcomes["planted 15% in the high half only"] and not outcomes["pure noise"]
          and not outcomes["planted 15% in BOTH halves"])

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
    print("holding rules:", "OK" if rules else "FAILED")
    # Day-0 placement: a Saturday lands on the following Monday, never earlier.
    week = [date(2010, 5, 3) + timedelta(days=i) for i in (0, 1, 2, 3, 4, 7, 8)]
    placed = (place(date(2010, 5, 8), week) == date(2010, 5, 10)
              and place(date(2010, 5, 5), week) == date(2010, 5, 5)
              and place(date(2010, 6, 1), week) is None)
    print("day-0 placement:", "OK" if placed else "FAILED")
    ok = ok and rules and placed
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def final(data: Path, delist: Path) -> int:
    from sealed_hypothesis import BLOCK_LINE, HYPOTHESIS, KILL, MECHANISM_GAP, TRADED_WINDOW

    registration = json.loads((HERE / "registration.json").read_text(encoding="utf-8"))
    events_csv = data / "deal_break_registration_set.csv"
    if sha256(events_csv) != registration["provenance"]["events_csv_sha256"]:
        sys.exit(f"{events_csv} does not match the sealed SHA-256; refusing to run")
    journal = ResearchJournal(HERE / "journal.jsonl")
    fingerprint = HYPOTHESIS.fingerprint
    if not journal.is_registered(fingerprint) or fingerprint != registration["fingerprint"]:
        sys.exit(f"{fingerprint} is not the registered, sealed hypothesis; refusing to run")

    frame = pl.read_csv(events_csv, try_parse_dates=True)
    rows = list(frame.iter_rows(named=True))
    targets = sorted({str(row["PERMNO"]) for row in rows})
    # Loading reveals nothing — no abnormal return is aggregated, printed, or compared yet — so a
    # crash here costs no trial and may be re-run. The lock is taken AFTER it and BEFORE the
    # study: that boundary is where the one permitted look begins.
    returns, abnormal = stitched_panels(data, delist, targets)
    sessions = [stamp.date() if isinstance(stamp, datetime) else stamp for stamp in returns.index]
    high_events: list[Event] = []
    low_events: list[Event] = []
    unplaced = 0
    for row in rows:
        when = place(row["day0"], sessions)
        if when is None:
            unplaced += 1
            continue
        event = Event(str(row["PERMNO"]), when)
        (high_events if float(row["block_new"]) >= BLOCK_LINE else low_events).append(event)
    print(f"high-block events {len(high_events)}, low-block {len(low_events)}, past the data "
          f"{unplaced}")
    descriptor = os.open(HERE / "RUN.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        stamp = datetime.now().astimezone().isoformat()
        handle.write(json.dumps({"fingerprint": fingerprint, "started": stamp}) + "\n")
    high = judge(returns, abnormal, high_events, KILL, TRADED_WINDOW, "high-block half")
    low = judge(returns, abnormal, low_events, KILL, TRADED_WINDOW, "low-block half (control)")
    survived, why = grade(high, low, MECHANISM_GAP)
    print(f"\nhigh-block half: {high}\nlow-block half:  {low}\n"
          f"{'SURVIVED' if survived else 'KILLED'}: {why}")

    journal.record_trial(
        fingerprint,
        sharpe=high.t_statistic,
        label="event study; the sharpe field holds the high-block half's t-statistic",
    )
    if survived:
        outcome = Verdict.SURVIVED
    elif high.underpowered:
        outcome = Verdict.INVALID
    else:
        outcome = Verdict.KILLED
    journal.record_verdict(fingerprint, outcome, ())
    result = {
        "fingerprint": fingerprint,
        "finished": datetime.now().astimezone().isoformat(),
        "verdict": outcome.value,
        "why": why,
        "high": {
            "mean_car": high.mean_car, "t_statistic": high.t_statistic,
            "win_rate": high.win_rate, "n_events": high.n_events,
            "clustering": high.clustering, "detectable_car": high.detectable_car,
            "underpowered": high.underpowered, "reasons": list(high.reasons),
        },
        "low": {
            "mean_car": low.mean_car, "t_statistic": low.t_statistic,
            "win_rate": low.win_rate, "n_events": low.n_events,
            "clustering": low.clustering,
        },
        "unplaced_past_data": unplaced,
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
