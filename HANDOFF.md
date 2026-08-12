# Handoff — 2026-08-12

State at the end of the session that loaded real CRSP data for the first time. Everything below
is verified against the actual extract, not against fixtures.

## Where things stand

crucible is complete as a platform and has real data in it. **No strategy has been tested and no
hypothesis has been registered** — deliberately, because a backtest run before the criteria are
fixed is a trial that raises the deflation bar for whatever is eventually claimed.

```
319 tests · ruff + mypy strict clean (package AND scripts) · public at github.com/Dera219/crucible
```

## The data

Two files in the repo root, both gitignored (`*.csv` is blocked — CRSP is licensed to UMD and
publishing an extract could cost the university's access for everyone):

| file | rows | contents |
|---|---|---|
| `crsp_daily.csv` | 23,101,820 | 2.0 GB · CIZ format · 2015-01-02 → 2025-12-31 |
| `crsp_delist.csv` | 6,227 | delisting table with `DelRet` |

Loads in ~206s into `(2766, 10551)` panels. Median breadth **4,984 names/day**.

Verified properties:
- **4,205 securities delisted in-sample**, 3.6%/yr attrition — survivorship-free is real
- **331 delisting returns worse than -50%**, worst -100% — the bankruptcies are present
- Breadth 4,885–6,337 every year, no thin periods
- Extreme moves are 91% concentrated in sub-$5 names — real microcaps, not corruption
- **Adjusted prices reproduce CRSP's own `DlyRet` exactly** (0 disagreements >1bp, 15.0M obs)

## Open items, in priority order

### 1. The mechanism — the only real blocker

`Hypothesis` will not construct without a ≥120-character mechanism naming who is on the other
side of the trade and why they lose money to you. This is Chidera's to write and has been the
gating item all along. `scripts/example_hypothesis.py` shows the shape using a deliberately
decayed effect so it cannot be mistaken for a recommendation.

### 2. ETF contamination in the universe

`sharetype='NS'` does **not** exclude ETFs, unlike legacy `SHRCD 10/11`. SPY, IWM, ARKK and QQQ
all carry it. Filtering to exchanges N/A/Q removed most (Arca and Cboe BZX are gone) but
Nasdaq-listed funds remain.

**Fix:** re-pull with four more columns — `securitytype`, `securitysubtype`, `usincflg`,
`issuertype`. The Rerun link restores every other setting:
`/pages/get-data/center-research-security-prices-crsp/annual-update/stock-version-2/daily-stock-file/?saved_query=7647821`

Then extend `load_crsp_ciz` to filter on them. The documented CIZ equivalent of US common stock is
roughly `securitytype='EQTY' AND securitysubtype='COM' AND usincflg='Y' AND issuertype IN
('ACOR','CORP')` — **verify against the actual values in the extract before trusting that**, the
way the exchange codes were identified empirically rather than assumed.

### 3. Residual return disagreements — RESOLVED (bug #17)

The 1,940 disagreements were **duplicate `(PERMNO, date)` rows**, which WRDS documents on the
query page: CRSP emits more than one row when a security reports multiple distribution events on
a day. 4,579 pairs across 23.1M rows, touching 2,381 securities.

`cum_prod` compounded every row while the pivot's `aggregate_function="first"` kept only one, so
the adjusted series carried a return the panel never showed and the two diverged from that day
onward. All five largest disagreements traced to a duplicate on the preceding session. The first
hypothesis — that `fill_null(0.0)` on missing returns was to blame — was **wrong**: zero of the
disagreements followed a missing return, which is what pointed at duplicates instead.

Fixed by deduplicating on `(_permno, _date)` before compounding, so the return series and the
pivot see identical rows. **Verified across all 14,993,678 observations: zero disagreements above
a basis point, max error 7.11e-15 — floating-point noise.** The progression was 128,781 → 1,940 →
0.

**DONE:** `backtest()` now takes an optional `returns=` panel, so the engine consumes the vendor's
authoritative series (`Dataset.returns` carries CRSP's `DlyRet`) instead of re-deriving it —
removing the class of error rather than correcting for it. Positions and delistings still key off
`prices`: a NaN price means non-investable regardless of what the return panel says. The
full-sample series passes through `walk_forward(**backtest_kwargs)` unsliced; `align()` inside the
engine intersects it down to each fold's test window. Tested with a planted split — raw prices
carrying a fake -75% day, vendor series carrying the truth — plus an exact-match check against the
derived path and a fold pass-through check.

### 4. Universe churn — RESOLVED

The band was widened empirically, not by taste — churn measured on the real extract at four
widths (top-1000, daily reconstitution, 2015-2025):

| enter/leave | mean members | churn/period | forced round trips |
|---|---|---|---|
| 800/1200 (old default) | 997 | 3.34 | 37.7% of book/yr |
| 700/1400 | 1038 | 2.05 | 20.2% |
| **600/1600 (new default)** | **1026** | **1.50** | **13.4%** |
| 500/2000 | 974 | 1.14 | 9.5% |

`liquidity_screen` now defaults to `0.6x/1.6x`: two-thirds less forced trading, membership still
on target. Wider than that buys little and lets members ride to rank 2000, at which point a
"top-1000" universe isn't one. The full table is in the docstring.

## The two bugs found in the last hour, for context on how to work here

**#16 — the split bug, reproduced in my own loader.** `DlyPrc` is raw. The engine derives returns
from it. AAPL read as -74.15% on its 2020 split, TSLA -66.78%, NVDA -89.93%. 3,244 split events
across the panel. This is the Alpaca defect that `apex/data/splits.py` exists to catch, on a
vendor that supplied the right answer in `DlyRet` and which the loader ignored.

**#15 — the verifier that hid it.** Compared a count capped at 400 against 25% of the full
11,478, a threshold that can never be reached. It printed "394 of the first 400 are NOT near a
recorded exit" and then reported "No defects found."

Both are the same shape as the other fourteen: **correct on a small fixture, silently wrong on
real data.** The technique that keeps working is to check against something with a known answer —
a planted signal, an accounting identity, a documented corporate action — rather than against
whether the number looks reasonable.

## Do not

- Run a signal search before a hypothesis is registered. It spends trial budget on curiosity and
  the deflation arithmetic is unforgiving: against 50 trials, an annualised Sharpe near 1.27 needs
  ~20 years of daily data to separate from luck.
- Commit the extract. `.gitignore` blocks it; do not `git add -f`.
- Enter WRDS credentials on Chidera's behalf.
