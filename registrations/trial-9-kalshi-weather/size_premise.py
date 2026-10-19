"""Sizing the premise of trial nine (NOT the trial): Kalshi daily temperature markets vs the
National Blend of Models, in the window before the day's high can be observed.

What this reads: Kalshi's PUBLIC market list and trade tape (no key), with every outcome field
(result, settlement value, expiration value, last/previous prices) dropped at ingestion, and
only trades BEFORE the clean-window cutoff (12:00Z on the market's day for highs: before the
daytime maximum exists anywhere); the NBM short-range forecast archive from the Iowa
Environmental Mesonet (txn = 12-h max temperature, xnd = its spread), only runs available
before each trade. Nothing here looks at what the temperature turned out to be.

What it measures: how much was actually traded at prices 10+ points from the forecast-implied
probability, after Kalshi's fee, per day, across stations; and who took those trades (a
taker on the forecast's side is competition, a taker against it is the flow a maker could
have been on the other side of).

The bar, fixed before the data was read (2026-10-19): median day >= $2,000 notional of
qualifying volume after fee, highs only, last 90 days. Below: retire on paper.

    size_premise.py fetch      # resumable; ~1 h at 6 requests/s
    size_premise.py measure    # writes sizing_result.md next to this file
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

DATA = Path.home() / "Downloads" / "kalshi_weather"
HERE = Path(__file__).resolve().parent
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
DAYS = 90
PACE_S = 0.17  # ~6 Kalshi requests a second
FEE_RATE = 0.07  # quadratic fee, multiplier 1, rounded up to the cent per contract
MIN_SD = 1.5  # never trust a forecast spread under this (degrees F)
NBS_LAG_S = 100 * 60  # an NBM run is on the wire about 100 minutes after its run time
GAP_BAR = 0.10
DAILY_BAR = 2_000.0
last_brackets: dict = {}

# series -> MOS station, highs only (phase 1); the CLI site named in each market's rules.
HIGHS = {
    "KXHIGHNY": "KNYC", "KXHIGHLAX": "KLAX", "KXHIGHTDC": "KDCA", "KXHIGHTLV": "KLAS",
    "KXHIGHAUS": "KAUS", "KXHIGHTEWR": "KEWR", "KXHIGHTSAN": "KSAN", "KXHIGHTNOLA": "KMSY",
    "KXHIGHTSFO": "KSFO", "KXHIGHTATL": "KATL", "KXHIGHTMIN": "KMSP", "KXHIGHDEN": "KDEN",
    "KXHIGHTOKC": "KOKC", "KXHIGHTDAL": "KDFW", "KXHIGHTSDF": "KSDF", "KXHIGHPHIL": "KPHL",
    "KXHIGHCHI": "KMDW", "KXHIGHTPHX": "KPHX", "KXHIGHTSEA": "KSEA", "KXHIGHMIA": "KMIA",
    "KXHIGHTSATX": "KSAT", "KXHIGHTHOU": "KHOU", "KXHIGHTTTN": "KTTN", "KXHIGHTBOS": "KBOS",
}
# Each station's clock: the climate day The Weather Company reports runs local midnight to
# midnight, and the Blend's 12Z-00Z maximum misses a maximum that falls outside those hours.
TZ = {
    "KNYC": "America/New_York", "KDCA": "America/New_York", "KEWR": "America/New_York",
    "KPHL": "America/New_York", "KBOS": "America/New_York", "KTTN": "America/New_York",
    "KMIA": "America/New_York", "KATL": "America/New_York", "KSDF": "America/New_York",
    "KMDW": "America/Chicago", "KMSP": "America/Chicago", "KAUS": "America/Chicago",
    "KDFW": "America/Chicago", "KSAT": "America/Chicago", "KHOU": "America/Chicago",
    "KMSY": "America/Chicago", "KOKC": "America/Chicago", "KDEN": "America/Denver",
    "KPHX": "America/Phoenix", "KLAS": "America/Los_Angeles", "KLAX": "America/Los_Angeles",
    "KSAN": "America/Los_Angeles", "KSFO": "America/Los_Angeles", "KSEA": "America/Los_Angeles",
}
OUTCOME_FIELDS = ("result", "settlement_value_dollars", "expiration_value", "last_price_dollars",
                  "previous_price_dollars", "yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars",
                  "no_ask_dollars", "previous_yes_bid_dollars", "previous_yes_ask_dollars",
                  "settlement_ts")
MONTHS = {m: i for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
                                       "SEP", "OCT", "NOV", "DEC"), start=1)}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def get(url: str, tries: int = 12) -> dict:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "premise-sizing/0.1"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            wait = 2.0 * (attempt + 1) if exc.code in (429, 500, 502, 503, 504) else 0
            if not wait:
                raise
            log(f"  HTTP {exc.code}, waiting {wait:.0f}s")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            log(f"  {exc!r}, waiting")
            time.sleep(min(5.0 * (attempt + 1), 60.0))
    raise RuntimeError(f"gave up on {url}")


def market_day(event_ticker: str) -> dt.date:
    tail = event_ticker.rsplit("-", 1)[-1]  # 26OCT18
    return dt.date(2000 + int(tail[:2]), MONTHS[tail[2:5]], int(tail[5:7]))


def clean_market(m: dict) -> dict:
    """Keep only what the sizing needs; every outcome-bearing field is dropped here."""
    for k in OUTCOME_FIELDS:
        m.pop(k, None)
    return {k: m.get(k) for k in ("ticker", "event_ticker", "strike_type", "floor_strike",
                                  "cap_strike", "open_time", "close_time", "title", "yes_sub_title")}


def fetch_markets(series: str, since: dt.datetime, until: dt.datetime) -> list[dict]:
    path = DATA / "markets" / f"{series}.json"
    if path.exists():
        return json.loads(path.read_text())
    out: list[dict] = []
    cursor = ""
    while True:
        q = {"series_ticker": series, "status": "settled", "limit": 1000,
             "min_close_ts": int(since.timestamp()), "max_close_ts": int(until.timestamp())}
        if cursor:
            q["cursor"] = cursor
        d = get(f"{KALSHI}/markets?{urllib.parse.urlencode(q)}")
        out += [clean_market(m) for m in d.get("markets", [])]
        cursor = d.get("cursor") or ""
        time.sleep(PACE_S)
        if not cursor:
            break
    path.write_text(json.dumps(out))
    return out


def cutoff_for(day: dt.date) -> dt.datetime:
    """Highs: 12:00Z on the market's day, before any daytime maximum exists in the US."""
    return dt.datetime(day.year, day.month, day.day, 12, 0, tzinfo=dt.timezone.utc)


def fetch_trades(m: dict) -> list[dict]:
    path = DATA / "trades" / f"{m['ticker']}.json"
    if path.exists():
        return json.loads(path.read_text())
    opened = dt.datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
    cutoff = cutoff_for(market_day(m["event_ticker"]))
    out: list[dict] = []
    if cutoff > opened:
        cursor = ""
        while True:
            q = {"ticker": m["ticker"], "limit": 1000, "min_ts": int(opened.timestamp()),
                 "max_ts": int(cutoff.timestamp())}
            if cursor:
                q["cursor"] = cursor
            d = get(f"{KALSHI}/markets/trades?{urllib.parse.urlencode(q)}")
            for t in d.get("trades", []):
                out.append({"t": t["created_time"], "yes": float(t["yes_price_dollars"]),
                            "n": float(t["count_fp"]), "taker": t.get("taker_side")})
            cursor = d.get("cursor") or ""
            time.sleep(PACE_S)
            if not cursor:
                break
    path.write_text(json.dumps(out))
    return out


def fetch_nbs(station: str, since: dt.datetime, until: dt.datetime) -> Path:
    path = DATA / "nbs" / f"{station}.csv"
    if path.exists() and path.stat().st_size > 1000:
        return path
    q = {"station": station, "model": "NBS", "sts": since.strftime("%Y-%m-%dT%H:%MZ"),
         "ets": until.strftime("%Y-%m-%dT%H:%MZ"), "format": "csv"}
    for attempt in range(5):
        req = urllib.request.Request(f"{IEM}?{urllib.parse.urlencode(q)}",
                                     headers={"User-Agent": "premise-sizing/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
        if body.startswith(b"runtime,"):
            path.write_bytes(body)
            return path
        log(f"  IEM refused {station}: {body[:60]!r}; waiting")
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"no NBS for {station}")


def cmd_fetch() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    until = now.replace(hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(days=1)
    since = until - dt.timedelta(days=DAYS)
    log(f"markets closing {since.date()} .. {until.date()}, {len(HIGHS)} series")
    for i, (series, station) in enumerate(HIGHS.items()):
        fetch_nbs(station, since - dt.timedelta(days=2), now)
        time.sleep(12)
        markets = fetch_markets(series, since, until)
        n_tr = 0
        for m in markets:
            n_tr += len(fetch_trades(m))
        log(f"[{i + 1}/{len(HIGHS)}] {series} {station}: {len(markets)} markets, "
            f"{n_tr} trades in the clean windows")
    log("fetch done")


def load_nbs(station: str) -> dict[dt.date, list[tuple[dt.datetime, float, float]]]:
    """Per market day: (available-from time, forecast day-maximum, spread) for every NBM run.

    The forecast maximum is the larger of the Blend's 12Z-00Z maximum (txn at the 00Z valid
    time) and the highest 3-hourly temperature the same run puts inside the station's LOCAL
    calendar day, so a maximum that falls at midnight or in the evening, outside the Blend's
    daytime window, is not missed. The spread is the txn row's xnd.
    """
    zone = ZoneInfo(TZ[station])
    by_run: dict[dt.datetime, list[dict]] = defaultdict(list)
    with open(DATA / "nbs" / f"{station}.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("ftime") and row.get("runtime"):
                by_run[dt.datetime.fromisoformat(row["runtime"]).replace(tzinfo=dt.timezone.utc)].append(row)
    out: dict[dt.date, list[tuple[dt.datetime, float, float]]] = defaultdict(list)
    for run, rows in by_run.items():
        hourly: dict[dt.date, list[float]] = defaultdict(list)
        txn: dict[dt.date, tuple[float, float]] = {}
        for row in rows:
            ftime = dt.datetime.fromisoformat(row["ftime"]).replace(tzinfo=dt.timezone.utc)
            if row.get("tmp"):
                hourly[ftime.astimezone(zone).date()].append(float(row["tmp"]))
            if row.get("txn") and ftime.hour == 0:
                sd = float(row["xnd"]) if row.get("xnd") else 3.0
                txn[(ftime - dt.timedelta(days=1)).date()] = (float(row["txn"]), sd)
        for day, (daytime_max, sd) in txn.items():
            forecast = max([daytime_max] + hourly.get(day, []))
            out[day].append((run + dt.timedelta(seconds=NBS_LAG_S), forecast, max(sd, MIN_SD)))
    for day in out:
        out[day].sort()
    return out


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def implied_yes(m: dict, mean: float, sd: float) -> float | None:
    kind, lo, hi = m["strike_type"], m.get("floor_strike"), m.get("cap_strike")
    if kind == "greater" and lo is not None:  # "greater than 61" settles YES at 62 and up
        return 1.0 - phi((lo + 0.5 - mean) / sd)
    if kind == "less" and hi is not None:  # "less than 54" settles YES at 53 and below
        return phi((hi - 0.5 - mean) / sd)
    if kind == "between" and lo is not None and hi is not None:  # 54-55 inclusive
        return phi((hi + 0.5 - mean) / sd) - phi((lo - 0.5 - mean) / sd)
    return None


def fee(price: float) -> float:
    """Kalshi's quadratic taker fee per $1 contract, rounded up to the cent."""
    return math.ceil(FEE_RATE * price * (1.0 - price) * 100.0) / 100.0


def measure(sd_override: float | None) -> tuple[dict, dict, dict, dict]:
    """Totals, per day, per station, and the mean-disagreement rows, for one spread setting
    (None = the Blend's own xnd, floored). Reads the cached data only."""
    per_day: dict[dt.date, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    disagreement: dict[tuple[str, dt.date], tuple[float, float, float]] = {}
    per_station: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    totals: dict[str, float] = defaultdict(float)
    for series, station in HIGHS.items():
        mpath = DATA / "markets" / f"{series}.json"
        if not mpath.exists():
            continue
        try:
            forecasts = load_nbs(station)
        except FileNotFoundError:
            continue
        for m in json.loads(mpath.read_text()):
            day = market_day(m["event_ticker"])
            runs = forecasts.get(day, [])
            tpath = DATA / "trades" / f"{m['ticker']}.json"
            if not runs or not tpath.exists():
                continue
            last_price: tuple[dt.datetime, float] | None = None
            for t in json.loads(tpath.read_text()):
                when = dt.datetime.fromisoformat(t["t"].replace("Z", "+00:00"))
                avail = [r for r in runs if r[0] <= when]
                if not avail:
                    continue
                _, mean, sd = avail[-1]
                if sd_override is not None:
                    sd = sd_override
                if last_price is None or when > last_price[0]:
                    last_price = (when, t["yes"])
                p = implied_yes(m, mean, sd)
                if p is None:
                    continue
                yes, n = t["yes"], t["n"]
                if not 0.01 <= yes <= 0.99:
                    continue
                gap = p - yes  # > 0: YES was cheap; < 0: NO was cheap
                side_price = yes if gap > 0 else 1.0 - yes
                net = abs(gap) - fee(side_price)
                totals["volume"] += n
                per_day[day]["volume"] += n
                per_station[station]["volume"] += n
                if net < GAP_BAR:
                    continue
                cheap_side = "yes" if gap > 0 else "no"
                sharp_taker = t.get("taker") == cheap_side  # the taker took the cheap side
                key = "sharp" if sharp_taker else "flow"
                for bucket in (per_day[day], per_station[station], totals):
                    bucket["qual_notional"] += n
                    bucket["qual_cost"] += n * side_price
                    bucket["qual_edge"] += n * net
                    bucket[key] += n
            if last_price is not None and runs:
                # the market's centre vs the Blend's: last traded price per bracket in the window
                brackets = last_brackets.setdefault((station, day), {})
                brackets[m["ticker"]] = (m, last_price[1], runs[-1][1])
    for (station, day), brackets in last_brackets.items():
        centre = market_centre(brackets)
        if centre is not None:
            nbm = next(iter(brackets.values()))[2]
            disagreement[(station, day)] = (centre, nbm, abs(centre - nbm))
    return dict(totals), dict(per_day), dict(per_station), disagreement


def conditional_capacity(disagreement: dict, min_gap_f: float, sd: float) -> dict[str, float]:
    """On station-days where the crowd's centre sits >= min_gap_f from the Blend, the window's
    volume in brackets on the Blend's side of the crowd, traded under the Blend-implied
    probability by more than the fee. What a bot could have bought where the two disagree."""
    out: dict[str, float] = defaultdict(float)
    days = set()
    for (station, day), (centre, nbm, gap) in disagreement.items():
        if gap < min_gap_f:
            continue
        series = next(s for s, st in HIGHS.items() if st == station)
        mpath = DATA / "markets" / f"{series}.json"
        for m in json.loads(mpath.read_text()):
            if market_day(m["event_ticker"]) != day:
                continue
            tpath = DATA / "trades" / f"{m['ticker']}.json"
            if not tpath.exists():
                continue
            p = implied_yes(m, nbm, sd)
            if p is None:
                continue
            for t in json.loads(tpath.read_text()):
                yes, n = t["yes"], t["n"]
                if not 0.01 <= yes <= 0.99:
                    continue
                g = p - yes
                side_price = yes if g > 0 else 1.0 - yes
                if abs(g) - fee(side_price) < 0.05:
                    continue
                days.add((station, day))
                out["notional"] += n
                out["cost"] += n * side_price
                out["edge"] += n * (abs(g) - fee(side_price))
    out["station_days"] = float(len(days))
    return dict(out)


def market_centre(brackets: dict) -> float | None:
    """The crowd's expected high from the last traded prices of a day's brackets, when the
    brackets' prices add to something near one (0.7-1.3); else None."""
    weights = []
    for m, price, _ in brackets.values():
        kind, lo, hi = m["strike_type"], m.get("floor_strike"), m.get("cap_strike")
        if kind == "between" and lo is not None and hi is not None:
            mid = (lo + hi) / 2.0
        elif kind == "greater" and lo is not None:
            mid = lo + 2.0
        elif kind == "less" and hi is not None:
            mid = hi - 2.0
        else:
            continue
        weights.append((mid, price))
    total = sum(w for _, w in weights)
    if len(weights) < 4 or not 0.7 <= total <= 1.3:
        return None
    return sum(mid * w for mid, w in weights) / total


def cmd_measure() -> None:
    global last_brackets
    sweep: list[tuple[str, dict, dict]] = []
    for label, sd in (("NBM xnd (floored 1.5F)", None), ("3F", 3.0), ("4F", 4.0), ("5F", 5.0)):
        last_brackets = {}
        totals, per_day, per_station, disagreement = measure(sd)
        sweep.append((label, totals, per_day))
    totals, per_day, per_station, disagreement = measure(None)
    days = sorted(per_day)
    if not days:
        print("nothing measured yet")
        return
    q = sorted(per_day[d]["qual_notional"] for d in days)
    median = q[len(q) // 2]
    mean = sum(q) / len(q)
    verdict = "CLEARS" if median >= DAILY_BAR else "RETIRE ON PAPER"
    sweep_lines = ["", "## Robustness: the same count with a duller forecast spread", "",
                   "The Blend's stated spread (xnd, often 2F) is known to be underdispersive. If the "
                   "qualifying volume vanishes at 4-5F the 'mispricing' was our sharpness, not the "
                   "crowd's error.", "",
                   "| spread | qualifying median day | mean day | total | share of volume | "
                   "competition | flow |", "|---|---|---|---|---|---|---|"]
    for label, tot, pd_ in sweep:
        qq = sorted(pd_[d]["qual_notional"] for d in sorted(pd_)) or [0.0]
        sweep_lines.append(f"| {label} | ${qq[len(qq) // 2]:,.0f} | ${sum(qq) / len(qq):,.0f} | "
                           f"${tot.get('qual_notional', 0):,.0f} | "
                           f"{tot.get('qual_notional', 0) / max(tot.get('volume', 0), 1):.1%} | "
                           f"{tot.get('sharp', 0):,.0f} | {tot.get('flow', 0):,.0f} |")
    gaps = sorted(v[2] for v in disagreement.values())
    dis_lines = ["", "## The crowd's centre vs the Blend's (spread-free)", "",
                 "Per station-day: the expected high implied by the last traded price of each "
                 "bracket in the window, against the Blend's 12Z-00Z maximum. A mean "
                 "disagreement is where an edge would live whatever the spread.", ""]
    if gaps:
        n = len(gaps)
        dis_lines += [f"| station-days with a usable centre | {n} |", "|---|---|",
                      f"| median absolute disagreement | {gaps[n // 2]:.1f} F |",
                      f"| share >= 2 F | {sum(g >= 2 for g in gaps) / n:.1%} |",
                      f"| share >= 3 F | {sum(g >= 3 for g in gaps) / n:.1%} |",
                      f"| share >= 5 F | {sum(g >= 5 for g in gaps) / n:.1%} |"]
        dis_lines += ["", "On the station-days with a 3F+ disagreement, what traded on the Blend's "
                      "side under its implied probability by more than the fee plus 5 points:", "",
                      "| spread | station-days | notional | cost | implied edge (NOT a result) |",
                      "|---|---|---|---|---|"]
        for label, sd in (("3F", 3.0), ("4F", 4.0)):
            c = conditional_capacity(disagreement, 3.0, sd)
            dis_lines.append(f"| {label} | {c.get('station_days', 0):.0f} | "
                             f"${c.get('notional', 0):,.0f} | ${c.get('cost', 0):,.0f} | "
                             f"${c.get('edge', 0):,.0f} |")
    else:
        dis_lines.append("(no station-day had bracket prices adding to near one)")
    lines = [
        "# Sizing result: Kalshi daily highs vs the National Blend, clean window only",
        "",
        f"Measured {time.strftime('%Y-%m-%d %H:%MZ', time.gmtime())}: {len(days)} market days, "
        f"{len(per_station)} stations, {totals['volume']:,.0f} contracts traded in the clean "
        f"windows (open to 12:00Z of the market day). Outcomes never read.",
        "",
        f"Qualifying = traded at a price whose gap to the NBM-implied probability clears the "
        f"fee by {GAP_BAR:.0%}+. Bar (fixed 2026-10-19 before the read): median day >= "
        f"${DAILY_BAR:,.0f} notional.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| qualifying notional, median day | ${median:,.0f} |",
        f"| qualifying notional, mean day | ${mean:,.0f} |",
        f"| qualifying notional, total | ${totals['qual_notional']:,.0f} of "
        f"{totals['volume']:,.0f} ({totals['qual_notional'] / max(totals['volume'], 1):.1%}) |",
        f"| cost to buy the cheap side, total | ${totals['qual_cost']:,.0f} |",
        f"| forecast-implied edge after fee, total (NOT a result) | ${totals['qual_edge']:,.0f} |",
        f"| taken by a taker on the forecast's side (competition) | {totals['sharp']:,.0f} |",
        f"| taken by a taker against the forecast (flow a maker could meet) | {totals['flow']:,.0f} |",
        f"| **verdict against the bar** | **{verdict}** |",
        "",
        "## By station",
        "",
        "| station | traded | qualifying | share | competition | flow |",
        "|---|---|---|---|---|---|",
    ]
    for st in sorted(per_station, key=lambda s: -per_station[s]["qual_notional"]):
        b = per_station[st]
        lines.append(f"| {st} | {b['volume']:,.0f} | {b['qual_notional']:,.0f} | "
                     f"{b['qual_notional'] / max(b['volume'], 1):.1%} | {b['sharp']:,.0f} | "
                     f"{b['flow']:,.0f} |")
    lines += sweep_lines + dis_lines
    lines += ["", "## By day (last 20)", "", "| day | traded | qualifying | edge after fee |",
              "|---|---|---|---|"]
    for d in days[-20:]:
        b = per_day[d]
        lines.append(f"| {d} | {b['volume']:,.0f} | {b['qual_notional']:,.0f} | "
                     f"${b['qual_edge']:,.0f} |")
    (HERE / "sizing_result.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))
    print(f"\nwritten {HERE / 'sizing_result.md'}")


# ---- the registration set (the prints the sealed trial scores, outcomes still unread) ----

SPREAD_FLOOR = 2.0  # the Blend's xnd, never under this (F); fixed, not searched
MARGIN = 0.05  # the cheap side must clear the fee by this much; fixed, not searched
REGISTRATION_CSV = DATA / "trial9_registration_set.csv"


def cmd_register() -> None:
    """One opportunity per bracket per Blend run: the FIRST clean-window print after a run comes
    on the wire (and before the next one) at which the Blend-implied probability of a side
    exceeds its traded price by the fee plus MARGIN. The strategy buys one contract of that side
    at that price, once per run. One row per opportunity. Outcomes are not in this file and are
    not read to build it."""
    rows: list[dict] = []
    for series, station in HIGHS.items():
        mpath = DATA / "markets" / f"{series}.json"
        if not mpath.exists():
            continue
        forecasts = load_nbs(station)
        for m in json.loads(mpath.read_text()):
            day = market_day(m["event_ticker"])
            runs = forecasts.get(day, [])
            tpath = DATA / "trades" / f"{m['ticker']}.json"
            if not runs or not tpath.exists():
                continue
            taken: set[dt.datetime] = set()  # runs already acted on in this bracket
            prints = sorted(json.loads(tpath.read_text()), key=lambda t: t["t"])
            for t in prints:
                when = dt.datetime.fromisoformat(t["t"].replace("Z", "+00:00"))
                avail = [r for r in runs if r[0] <= when]
                if not avail:
                    continue
                run_avail, mean, sd = avail[-1]
                if run_avail in taken:
                    continue
                sd = max(sd, SPREAD_FLOOR)
                p = implied_yes(m, mean, sd)
                if p is None or not 0.01 <= t["yes"] <= 0.99:
                    continue
                gap = p - t["yes"]
                side = "yes" if gap > 0 else "no"
                price = t["yes"] if side == "yes" else 1.0 - t["yes"]
                implied_side = p if side == "yes" else 1.0 - p
                if implied_side - price - fee(price) < MARGIN:
                    continue
                taken.add(run_avail)
                rows.append({"ticker": m["ticker"], "day": day.isoformat(), "station": station,
                             "series": series, "time": t["t"], "side": side,
                             "price": round(price, 4), "yes_price": round(t["yes"], 4),
                             "n": t["n"], "implied_side": round(implied_side, 6),
                             "nbm_mean": mean, "nbm_sd": sd,
                             "run_available": run_avail.isoformat(),
                             "strike_type": m["strike_type"], "floor": m.get("floor_strike"),
                             "cap": m.get("cap_strike")})
    rows.sort(key=lambda r: (r["day"], r["station"], r["time"], r["ticker"]))
    with REGISTRATION_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["ticker"])
        w.writeheader()
        w.writerows(rows)
    days = {(r["station"], r["day"]) for r in rows}
    dates = {r["day"] for r in rows}
    print(f"registration set: {len(rows)} opportunities over {len(days)} station-days and "
          f"{len(dates)} calendar days -> {REGISTRATION_CSV}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "measure"
    {"fetch": cmd_fetch, "measure": cmd_measure, "register": cmd_register}[cmd]()
