"""Score the sealed Kalshi-highs-vs-NBM registration, ONCE.

    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/kalshi-weather/run_trial.py --self-test
    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/kalshi-weather/run_trial.py --final

`--self-test` exercises the whole grading path on fabricated prints and outcomes and touches
nothing real: a planted edge must survive, pure noise must die, and a third of the markets
lacking a result must fail the dropped-fraction bar. `--final` is the trial. Before it reads a
single outcome it refuses unless:

1. the registration set CSV hashes to the SHA-256 sealed in registration.json;
2. the hypothesis's fingerprint is registered in this folder's journal.jsonl;
3. RUN.lock can be created exclusively (O_EXCL), so this runs once. The lock is taken after the
   set is loaded (which reveals nothing) and before any result is fetched.

Outcomes come from Kalshi's public market objects (`result`: yes | no), one read per market,
paced. The grading is bespoke (crucible's `evaluate` is an event-study grader over CRSP
panels); the bars are the sealed EventKillCriteria read literally:

- unit: station-day; an opportunity's net = payoff(1 if its side won) - price - taker fee;
- a station-day's value = the equal-weighted mean net over its opportunities (one per bracket
  per Blend run, the first qualifying print after the run came on the wire);
- mean over station-days; t with the standard error clustered by CALENDAR DAY (a regional
  forecast bust hits every station that day); at least MIN_CALENDAR_DAYS clusters;
  win rate (station-days with positive value),
  clustering (share of the positive total carried by the top 10% of station-days), dropped
  fraction (station-days with any print whose market has no yes/no result; those are excluded);
- the mechanism bar: Brier(Blend) must be at least BRIER_GAP better than Brier(traded price)
  on the registered prints, both in YES terms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from crucible.preregistration import ResearchJournal, Verdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sealed_hypothesis import BRIER_GAP, HYPOTHESIS, KILL, MIN_CALENDAR_DAYS  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "Downloads" / "kalshi_weather"
REGISTRATION_CSV = DATA / "trial9_registration_set.csv"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
FEE_RATE = 0.07


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fee(price: float) -> float:
    return math.ceil(FEE_RATE * price * (1.0 - price) * 100.0) / 100.0


def load_prints(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["price"] = float(r["price"])
        r["yes_price"] = float(r["yes_price"])
        r["implied_side"] = float(r["implied_side"])
    return rows


def grade(prints: list[dict], results: dict[str, str | None]) -> dict:
    """The verdict and every number behind it. `results`: ticker -> "yes" | "no" | None."""
    by_day: dict[tuple[str, str], list[float]] = defaultdict(list)
    missing_days: set[tuple[str, str]] = set()
    brier_blend: list[float] = []
    brier_market: list[float] = []
    for p in prints:
        key = (p["station"], p["day"])
        result = results.get(p["ticker"])
        if result not in ("yes", "no"):
            missing_days.add(key)
            continue
        won = 1.0 if result == p["side"] else 0.0
        by_day[key].append(won - p["price"] - fee(p["price"]))
        yes_outcome = 1.0 if result == "yes" else 0.0
        blend_yes = p["implied_side"] if p["side"] == "yes" else 1.0 - p["implied_side"]
        brier_blend.append((blend_yes - yes_outcome) ** 2)
        brier_market.append((p["yes_price"] - yes_outcome) ** 2)
    all_days = set(by_day) | missing_days
    usable = {k: float(np.mean(v)) for k, v in by_day.items() if k not in missing_days}
    n = len(usable)
    values = np.array(sorted(usable.values(), reverse=True)) if n else np.array([])
    mean = float(values.mean()) if n else float("nan")
    sd = float(values.std(ddof=1)) if n > 1 else float("nan")
    # Station-days on the same calendar day share the weather, and a regional forecast bust
    # hits every station in it: the standard error is clustered by calendar day.
    clusters: dict[str, float] = defaultdict(float)
    for (station, day), value in usable.items():
        clusters[day] += value - mean
    n_clusters = len(clusters)
    cluster_se = math.sqrt(sum(v * v for v in clusters.values())) / n if n else float("nan")
    t = mean / cluster_se if n > 1 and cluster_se > 0 else float("nan")
    win_rate = float((values > 0).mean()) if n else float("nan")
    positive = values[values > 0]
    top = max(1, int(math.ceil(0.10 * n)))
    clustering = (float(values[:top].clip(min=0).sum() / positive.sum())
                  if n and positive.sum() > 0 else 0.0)
    dropped = len(missing_days) / len(all_days) if all_days else 1.0
    bb = float(np.mean(brier_blend)) if brier_blend else float("nan")
    bm = float(np.mean(brier_market)) if brier_market else float("nan")
    reasons: list[str] = []
    if n < KILL.min_events:
        reasons.append(f"{n} station-days < {KILL.min_events}")
    if n_clusters < MIN_CALENDAR_DAYS:
        reasons.append(f"{n_clusters} calendar days < {MIN_CALENDAR_DAYS}")
    if not (mean >= KILL.min_mean_car):
        reasons.append(f"mean net {mean:.4f} < {KILL.min_mean_car}")
    if not (t >= KILL.min_t_statistic):
        reasons.append(f"t {t:.2f} < {KILL.min_t_statistic}")
    if not (win_rate >= KILL.min_win_rate):
        reasons.append(f"win rate {win_rate:.3f} < {KILL.min_win_rate}")
    if clustering > KILL.max_clustering:
        reasons.append(f"clustering {clustering:.2f} > {KILL.max_clustering}")
    if dropped > KILL.max_dropped_fraction:
        reasons.append(f"dropped {dropped:.2f} > {KILL.max_dropped_fraction}")
    if not (bm - bb >= BRIER_GAP):
        reasons.append(f"Brier gap {bm - bb:+.4f} < {BRIER_GAP} (market {bm:.4f}, Blend {bb:.4f})")
    underpowered = (n >= KILL.min_events and not math.isnan(cluster_se)
                    and KILL.min_t_statistic * cluster_se > KILL.min_mean_car
                    and mean > 0 and not (t >= KILL.min_t_statistic))
    return {
        "survived": not reasons, "reasons": reasons, "station_days": n,
        "calendar_days": n_clusters, "cluster_se": cluster_se,
        "dropped_station_days": len(missing_days), "prints": len(prints),
        "prints_graded": len(brier_blend), "mean_net_per_contract": mean, "sd": sd,
        "t_statistic": t, "win_rate": win_rate, "clustering": clustering,
        "dropped_fraction": dropped, "brier_blend": bb, "brier_market": bm,
        "detectable_mean": (KILL.min_t_statistic * cluster_se) if n > 1 else float("nan"),
        "underpowered": underpowered,
    }


def fetch_results(tickers: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for i, tk in enumerate(tickers):
        req = urllib.request.Request(f"{KALSHI}/markets/{tk}", headers={"User-Agent": "trial9/1.0"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=40) as r:
                    m = json.load(r).get("market", {})
                out[tk] = (m.get("result") or "").lower() or None
                break
            except Exception as exc:  # noqa: BLE001 - retried, then recorded as missing
                time.sleep(2.0 * (attempt + 1))
                if attempt == 5:
                    out[tk] = None
                    print(f"  ! {tk}: {exc!r}")
        time.sleep(0.17)
        if (i + 1) % 200 == 0:
            print(f"  results {i + 1}/{len(tickers)}", flush=True)
    return out


def self_test() -> int:
    rng = random.Random(9)

    def fake(edge: float, days: int = 640, missing: float = 0.0) -> tuple[list[dict], dict]:
        prints: list[dict] = []
        results: dict[str, str | None] = {}
        for d in range(days):
            station = f"S{d % 8}"
            for k in range(3):
                tk = f"T{d}-{k}"
                price = rng.uniform(0.15, 0.45)
                implied = min(0.95, price + fee(price) + 0.05 + rng.uniform(0, 0.15))
                side = rng.choice(["yes", "no"])
                prints.append({"ticker": tk, "station": station, "day": f"day{d // 8}",
                               "side": side, "price": price,
                               "yes_price": price if side == "yes" else 1 - price,
                               "implied_side": implied})
                if rng.random() < missing:
                    results[tk] = None
                else:
                    # the planted truth: our side wins with probability price + edge
                    win = rng.random() < min(0.98, price + edge)
                    results[tk] = side if win else ("no" if side == "yes" else "yes")
        return prints, results

    planted = grade(*fake(edge=0.25))
    noise = grade(*fake(edge=0.0))
    holes = grade(*fake(edge=0.25, missing=0.35))
    ok = planted["survived"] and not noise["survived"] and not holes["survived"]
    print(f"planted edge: {'SURVIVED' if planted['survived'] else 'KILLED'} "
          f"(mean {planted['mean_net_per_contract']:.3f}, t {planted['t_statistic']:.1f}, win "
          f"{planted['win_rate']:.2f}, Brier gap {planted['brier_market'] - planted['brier_blend']:+.3f})")
    print(f"pure noise:   {'SURVIVED' if noise['survived'] else 'KILLED'} "
          f"(mean {noise['mean_net_per_contract']:.3f}; {'; '.join(noise['reasons'][:2])})")
    print(f"35% missing:  {'SURVIVED' if holes['survived'] else 'KILLED'} "
          f"(dropped {holes['dropped_fraction']:.2f}; {holes['reasons'][-1] if holes['reasons'] else ''})")
    print("self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def final() -> int:
    registration = json.loads((HERE / "registration.json").read_text(encoding="utf-8"))
    if sha256(REGISTRATION_CSV) != registration["provenance"]["registration_csv_sha256"]:
        sys.exit("the registration set does not hash to the sealed SHA-256; refusing to run")
    journal = ResearchJournal(HERE / "journal.jsonl")
    fingerprint = HYPOTHESIS.fingerprint
    if not journal.is_registered(fingerprint) or fingerprint != registration["fingerprint"]:
        sys.exit(f"{fingerprint} is not the registered, sealed hypothesis; refusing to run")
    prints = load_prints(REGISTRATION_CSV)
    tickers = sorted({p["ticker"] for p in prints})
    print(f"{len(prints)} registered prints over {len(tickers)} markets; taking RUN.lock")
    descriptor = os.open(HERE / "RUN.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    stamp = datetime.now().astimezone().isoformat()
    with os.fdopen(descriptor, "w") as handle:
        handle.write(json.dumps({"fingerprint": fingerprint, "started": stamp}) + "\n")
    results = fetch_results(tickers)
    verdict = grade(prints, results)
    print(json.dumps({k: v for k, v in verdict.items() if k != "reasons"}, indent=2))
    print("SURVIVED" if verdict["survived"] else "KILLED: " + "; ".join(verdict["reasons"]))
    journal.record_trial(fingerprint, sharpe=float(verdict["t_statistic"])
                         if not math.isnan(verdict["t_statistic"]) else 0.0,
                         label="prediction-market trial; the sharpe field holds the station-day t")
    if verdict["survived"]:
        outcome = Verdict.SURVIVED
    elif verdict["underpowered"] or verdict["station_days"] < KILL.min_events:
        outcome = Verdict.INVALID
    else:
        outcome = Verdict.KILLED
    journal.record_verdict(fingerprint, outcome, ())
    (HERE / "result.json").write_text(json.dumps({
        "fingerprint": fingerprint, "started": stamp,
        "finished": datetime.now().astimezone().isoformat(), "verdict": outcome.value,
        **verdict,
    }, indent=2, default=float) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--final", action="store_true")
    args = parser.parse_args()
    return self_test() if args.self_test else final()


if __name__ == "__main__":
    raise SystemExit(main())
