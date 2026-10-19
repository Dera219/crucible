"""The registration: Kalshi daily highs vs the National Blend. Trial 9, once sealed.

    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/kalshi-weather/sealed_hypothesis.py

Running this file SEALS the registration: it writes registration.json (every field that defines
the claim, its fingerprint, the SHA-256 of the registration set of prints and of every script
that touches it) and records the hypothesis in this folder's journal.jsonl. Only the SHA-256 of
registration.json is published (crucible's SEARCH_LOG.md); the rest stays private until revealed.

Drafted 2026-10-19 after Chidera brought the edge search back ("if people could do it why not
us") and accepted the reframe to capacity-limited niches. The premise was sized from public
Kalshi prints and the NBM archive without reading an outcome (size_premise.py beside this file);
the rule, the window, the spread floor, the margin and the bars below were fixed before the seal.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from crucible.events import EventKillCriteria
from crucible.preregistration import EdgeSource, Hypothesis, ResearchJournal

HERE = Path(__file__).resolve().parent

#: The Blend's own spread (xnd) is used, floored here: the ensemble is underdispersive and a
#: 1-2F spread over 2F brackets makes any price look stale. Fixed, not searched.
SPREAD_FLOOR = 2.0
#: A print qualifies when the Blend-implied probability of a side exceeds its traded price by
#: Kalshi's taker fee plus this margin. Fixed, not searched.
MARGIN = 0.05
#: The clean window: from the market's open (14:00Z the day before) to 12:00Z on its day, before
#: any daytime maximum exists at a US station. Prints after it embed observations, not forecasts.
CUTOFF_UTC_HOUR = 12
#: Station-days on one calendar day share the weather, so the t-statistic's standard error is
#: clustered by calendar day, and at least this many calendar days must carry the sample.
MIN_CALENDAR_DAYS = 50
#: The mechanism bar: on the registered prints the Blend's Brier score must beat the traded
#: price's by at least this much. If the crowd's price is the better forecast of the outcome on
#: the very prints we would take, the "edge" is noise around a sharper opponent, and this is dead.
BRIER_GAP = 0.01

#: Units are station-days (prints within one share the outcome). t >= 2.77 is Bonferroni over 9
#: lifetime trials, two-sided 5%. min_mean_car = 0.03 is net profit per $1 contract after the
#: fee; at a per-station-day dispersion near 0.25 and ~150 station-days that is what the bar can
#: resolve, and a smaller true edge comes back UNINFORMATIVE, registered as the risk it is.
KILL = EventKillCriteria(
    min_events=100,
    min_mean_car=0.03,
    min_t_statistic=2.77,
    min_win_rate=0.55,
    max_clustering=0.50,
    max_dropped_fraction=0.2,
    max_parameters=3,
)

HYPOTHESIS = Hypothesis(
    name="kalshi-daily-highs-vs-nbm",
    claim=(
        "In Kalshi's daily maximum-temperature markets for 24 US stations, a contract bought in "
        "the clean window (open to 12:00Z of the market day), once per bracket per National "
        "Blend run, at the first price the Blend's implied probability exceeds by the taker fee "
        "plus five points, earns net of that fee at least three cents per dollar of contract on "
        "average across station-days, with the standard error clustered by calendar day, and "
        "the Blend's probabilities beat the traded prices as forecasts of those outcomes."
    ),
    edge_source=EdgeSource.INFORMATION_PROCESSING,
    mechanism=(
        "The other side is a resting order placed by a retail trader quoting a temperature "
        "bracket on intuition, a glance at a phone forecast, or a position they want out of, "
        "and left standing overnight. The National Blend of Models is the official statistical "
        "blend of every operational model, published free four times a day with its own spread, "
        "and it is a sharper estimate of tomorrow's high than a guess. In the hours before the "
        "day's maximum can be observed, nothing but a forecast can price these brackets, so a "
        "price far from the Blend is a stale quote, not information the Blend lacks. The brackets "
        "are a few hundred dollars deep and the markets settle in a day, which is why no fund "
        "stands there to correct them, and why a bot with a small account can: the capacity is "
        "the limit, and the limit is well above what a small account needs."
    ),
    universe=(
        "Kalshi daily maximum-temperature series for 24 US stations (the series in "
        "size_premise.py HIGHS), markets closing 2026-07-20 to 2026-10-18, every print in each "
        "market between its open and 12:00Z on its day, from Kalshi's public trade tape. "
        "Forecast: NBM short-range (NBS) runs from the Iowa Environmental Mesonet archive; the "
        "forecast day-maximum is the larger of the 12Z-00Z maximum (txn) and the highest "
        "3-hourly temperature the same run places inside the station's local calendar day, so "
        "a maximum outside the Blend's daytime window is not missed; spread = the txn row's xnd, "
        "floored at 2.0F; only the latest run on the wire 100 minutes after its run time before "
        "the print. Implied probability per bracket from a normal at that mean and spread with "
        "the 0.5F rounding of integer thresholds. An opportunity is registered once per bracket "
        "per run: the first print after the run came on the wire at which the implied "
        "probability of a side exceeds its traded price by Kalshi's quadratic taker fee (0.07 x "
        "p x (1-p), rounded up to the cent) plus 0.05; the strategy buys one contract of that "
        "side at that price. The registration set is the CSV of those opportunities (hash "
        "sealed); outcomes come from Kalshi's result field read once, after sealing, by "
        "run_trial.py --final. Station-day = unit, its value the mean net over its "
        "opportunities (payoff - price - fee); the t-statistic's standard error is clustered by "
        "calendar day and at least 50 calendar days must carry the sample; win rate and "
        "clustering over station-days; opportunities whose market shows no yes/no result drop "
        "their station-day and count against max_dropped_fraction."
    ),
    horizon_bars=1,
    warmup_bars=0,
    parameters=("spread_floor_f", "margin_over_fee"),
    kill=KILL,
    trial_budget=9,
    notes=(
        "Drafted 2026-10-19; unsealed until Chidera runs this file. The premise sizing "
        "(sizing_result.md) could not retire the claim: its pre-fixed bar cleared at every "
        "spread, which showed the bar measured model sharpness, not the crowd; the spread-free "
        "check found the crowd's centre within 0.8F of the Blend on the median station-day and "
        "3F+ away on 5% (72 of 1,433), with $1.43M traded on the Blend's side of those days. Two bars, both required: KILL on station-day net profit, and the Blend's Brier "
        "better than the price's by BRIER_GAP on the registered prints. Highs only; lows are a "
        "separate claim with a harder clean window and are not part of this trial. Run once with "
        "run_trial.py --final; RUN.lock enforces it."
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(registration_csv: Path) -> Path:
    registration = {
        "sealed_at": datetime.now().astimezone().isoformat(),
        "fingerprint": HYPOTHESIS.fingerprint,
        "hypothesis": HYPOTHESIS.canonical(),
        "notes": HYPOTHESIS.notes,
        "spread_floor_f": SPREAD_FLOOR,
        "margin_over_fee": MARGIN,
        "cutoff_utc_hour": CUTOFF_UTC_HOUR,
        "brier_gap": BRIER_GAP,
        "min_calendar_days": MIN_CALENDAR_DAYS,
        "provenance": {
            "registration_csv_sha256": sha256(registration_csv),
            "registration_csv_rows": sum(1 for _ in registration_csv.open(encoding="utf-8")) - 1,
            "size_premise_py_sha256": sha256(HERE / "size_premise.py"),
            "run_trial_py_sha256": sha256(HERE / "run_trial.py"),
            "sealed_hypothesis_py_sha256": sha256(HERE / "sealed_hypothesis.py"),
        },
        "lifetime_trials_before": 8,
    }
    out = HERE / "registration.json"
    if out.exists():
        raise SystemExit(f"{out} exists: this registration is sealed and cannot be rewritten")
    out.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ResearchJournal(HERE / "journal.jsonl").register(HYPOTHESIS)
    return out


if __name__ == "__main__":
    sealed = seal(Path.home() / "Downloads" / "kalshi_weather" / "trial9_registration_set.csv")
    print(f"sealed {sealed}\nfingerprint {HYPOTHESIS.fingerprint}\nSHA-256 {sha256(sealed)}")
