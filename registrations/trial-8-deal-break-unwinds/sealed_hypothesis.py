"""The registration: deal-break unwinds. Trial 8 against the CRSP universe, once sealed.

    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/deal-break-unwinds/sealed_hypothesis.py

Running this file SEALS the registration: it writes registration.json (every field that defines
the claim, its fingerprint, and the SHA-256 of the event set and of every script that touches
it) and records the hypothesis in this folder's journal.jsonl. Only the SHA-256 of
registration.json is published (crucible's SEARCH_LOG.md); everything here stays in this private
repository until it is revealed.

The mechanism was drafted by Claude on 2026-10-15 at Chidera's instruction after two fund-side
sellers retired on paper the same night; its premise — the arbitrage block — was sized from 13F
holdings before any return was read (size_premise.py beside this file), and the event rule,
windows and bars below were critiqued by him before the seal.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from crucible.events import EventKillCriteria
from crucible.preregistration import EdgeSource, Hypothesis, ResearchJournal

HERE = Path(__file__).resolve().parent

#: The block line that splits the registration set: shares held at the last 13F public on day 0
#: by managers holding none at the quarter-end before the merger-agreement 8-K, over shares
#: outstanding. Fixed at the value the premise was sized against, not fitted.
BLOCK_LINE = 0.10
#: The mechanism bar: the high-block half's mean CAR over the traded window must exceed the
#: low-block half's by at least this much. Without it a recovery that every broken target shows
#: is the known revaluation of failed targets, not forced selling, and this claim is dead.
MECHANISM_GAP = 0.03
#: Buy at the close of session +5 after the termination (the arbitrageurs' selling week), sell
#: at the close of session +60: the traded window is offsets 6..60, 55 sessions held.
TRADED_WINDOW = (6, 60)

#: t >= 2.73: Bonferroni over the 8 lifetime trials (5 exploratory looks, trials 6 and 7, this
#: one), two-sided 5%, z = 2.734 — crucible's event `evaluate` never applies deflation itself, so
#: the bar is corrected by hand, as trials 6 and 7 did. min_mean_car = 0.10 is what ~54 events
#: can resolve at that bar if the per-event dispersion over 55 sessions is near 27%; a smaller
#: true effect comes back UNINFORMATIVE, which is registered as the risk it is. Applied to the
#: high-block half only; the low-block half is the control, graded by the gap above.
KILL = EventKillCriteria(
    min_events=40,
    min_mean_car=0.10,
    min_t_statistic=2.73,
    min_win_rate=0.55,
    max_clustering=0.30,
    max_dropped_fraction=0.2,
)

HYPOTHESIS = Hypothesis(
    name="deal-break-unwinds",
    claim=(
        "When an announced acquisition of a listed US target is terminated, the arbitrageurs "
        "holding the target's float must sell within days and press the price through its "
        "standalone value; bought at the close of the 5th session after the termination and "
        "held to the close of the 60th, targets whose arbitrage block was at least 10% of "
        "shares outstanding earn a positive abnormal return, and by at least 3 points more "
        "than targets whose block was smaller."
    ),
    edge_source=EdgeSource.STRUCTURAL_CONSTRAINT,
    mechanism=(
        "Once a deal is announced, merger arbitrageurs buy the target to collect the spread "
        "between the market price and the offer, hedged against the deal closing, until they "
        "hold a large block of the float. When the deal is terminated their mandate ends with "
        "it: a broken target is a directional equity position their rules do not allow, the "
        "loss on the break brings redemptions, and the hedge comes off at the same time. They "
        "sell the block inside days into a name with no catalyst left, so the price falls "
        "through the standalone value and recovers over the following months as value "
        "investors re-underwrite the company at the speed of a committee. Event funds buy "
        "catalysts and a broken deal has none for months; the names are mid and small caps, "
        "twenty to forty a year, so the block is too small and too slow for the funds that "
        "could otherwise absorb it, and the part that survives busted-deal desks is the weeks "
        "of waiting at a size a small account can hold."
    ),
    universe=(
        "CRSP CIZ daily US common stock (crucible's CIZ common-stock filter), 2005-2025. "
        "Events: deal breaks built from EDGAR indices (WRDS SEC Analytics Suite): a target-side "
        "filing (PREM14A, DEFM14A or SC 14D9; information statements excluded) with a "
        "merger-agreement exhibit (EX-2.*) on an 8-K within the year before it, no S-4 by the "
        "same filer (the acquirer's side), then an 8-K Item 1.02 termination within 540 days "
        "with no Form 25/15 delisting in the 120 days after it; no SPAC by name; the target "
        "mapped to a PERMNO through the CIK-CUSIP link or CIK-GVKEY-PERMNO, US common stock on "
        "the quarter-end before day 0, pre-deal market cap of at least $100M. Day 0 is the "
        "first session on or after the 8-K's period of report. 130 events; the block from s34 "
        "13F holdings at the last report public on day 0 (45-day filing lag) against the "
        "quarter-end before the agreement 8-K splits them 54 high (>= 10%) / 76 low. Abnormal = "
        "return minus the equal-weighted investable mean of the slice. A listed day without a "
        "trade carries zero abnormal return; a target delisted inside the window takes CRSP's "
        "delisting return on the session after its last trade and the proceeds earn zero raw "
        "return, i.e. minus the market, for the rest of the window; a target that receives a "
        "new bid inside the window stays in the sample. Traded window: offsets 6-60."
    ),
    horizon_bars=60,
    warmup_bars=0,
    parameters=(),
    kill=KILL,
    trial_budget=8,
    notes=(
        "Sealed 2026-10-15. The block line (10%) and the event rule were fixed when the premise "
        "was sized, before any return was read; the sized block's median was 10.2% (11.0% "
        "before the GVKEY link recovered 82 smaller targets), ordinary churn over the same span "
        "4.2%, the point-in-time block's median 6.9% on the clean set. Two bars, both required: "
        "KILL on the high-block half, and high minus low >= MECHANISM_GAP over the traded "
        "window; the pressure window (offsets 0-5) is reported and never graded. Entry +5 and "
        "exit +60 are fixed in code, not searched. Run once with run_event_study.py --final; "
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
        "block_line": BLOCK_LINE,
        "mechanism_gap": MECHANISM_GAP,
        "traded_window": list(TRADED_WINDOW),
        "provenance": {
            "events_csv_sha256": sha256(events_csv),
            "events_csv_rows": sum(1 for _ in events_csv.open(encoding="utf-8")) - 1,
            "build_breaks_py_sha256": sha256(HERE / "build_breaks.py"),
            "size_premise_py_sha256": sha256(HERE / "size_premise.py"),
            "run_event_study_py_sha256": sha256(HERE / "run_event_study.py"),
            "sealed_hypothesis_py_sha256": sha256(HERE / "sealed_hypothesis.py"),
        },
        "lifetime_trials_before": 7,
    }
    out = HERE / "registration.json"
    if out.exists():
        raise SystemExit(f"{out} exists: this registration is sealed and cannot be rewritten")
    out.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ResearchJournal(HERE / "journal.jsonl").register(HYPOTHESIS)
    return out


if __name__ == "__main__":
    sealed = seal(Path.home() / "Downloads" / "deal_break_registration_set.csv")
    print(f"sealed {sealed}\nfingerprint {HYPOTHESIS.fingerprint}\nSHA-256 {sha256(sealed)}")
