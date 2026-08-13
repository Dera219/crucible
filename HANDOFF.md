# Handoff — 2026-08-12

State at the end of the session that loaded real CRSP data for the first time. Everything below
is verified against the actual extract, not against fixtures.

## Where things stand

crucible is complete as a platform and has real data in it. **No strategy has been tested and no
hypothesis has been registered** — deliberately, because a backtest run before the criteria are
fixed is a trial that raises the deflation bar for whatever is eventually claimed.

```
341 tests · ruff + mypy strict clean (package AND scripts) · public at github.com/Dera219/crucible
```

## The data

Two files in the repo root, both gitignored (`*.csv` is blocked — CRSP is licensed to UMD and
publishing an extract could cost the university's access for everyone):

| file | rows | contents |
|---|---|---|
| `crsp_daily.csv` | 23,101,820 | 2.4 GB · CIZ format + classification columns · 2015-01-02 → 2025-12-31 |
| `crsp_delist.csv` | 6,227 | delisting table with `DelRet` |

Loads into `(2766, 6635)` panels after the common-stock filter. Median breadth **3,716
names/day** — US common stock only; ETFs, CEFs, REITs and ADRs are excluded by
`CIZ_COMMON_STOCK_FILTER` (see item 2).

Verified properties:
- **2,964 securities delisted in-sample**, 4.1%/yr attrition — inside the typical 4-8% band
- **331 delisting returns worse than -50%**, worst -100% — the bankruptcies are present
- SPY, QQQ, IWM, ARKK confirmed absent from the loaded panel
- **Adjusted prices reproduce CRSP's own `DlyRet` exactly** (0 disagreements >1bp)

## Open items, in priority order

### 1. The mechanism — the only real blocker

`Hypothesis` will not construct without a ≥120-character mechanism naming who is on the other
side of the trade and why they lose money to you. This is Chidera's to write and has been the
gating item all along. `scripts/example_hypothesis.py` shows the shape using a deliberately
decayed effect so it cannot be mistaken for a recommendation.

### 2. ETF contamination in the universe — RESOLVED

Re-pulled 2026-08-12 with the four classification columns (same 23,101,820 rows, Rerun link for
any future pull: `...daily-stock-file/?saved_query=7648132`). `load_crsp_ciz` now applies
`CIZ_COMMON_STOCK_FILTER` — `securitytype='EQTY' AND securitysubtype='COM' AND
issuertype='CORP' AND usincflg='Y'` — by default, and **refuses to load an extract missing those
columns** rather than silently skipping the filter, because silent-skip is how the contamination
survived `sharetype`.

**The instruction to verify empirically before trusting the docs was load-bearing.** The
documented guess included `issuertype='ACOR'` — which is what every probed ETF carries (SPY,
QQQ, IWM, ARKK are all FUND/ETF/ACOR). Trusting it would have excluded nothing. The cross-tab
on the real extract separates cleanly: common stock EQTY/COM/CORP, funds FUND/{ETF,CEF}/ACOR,
REITs carry issuertype REIT inside EQTY/COM, ADRs show usincflg N. Keeping CORP+Y reproduces
what legacy SHRCD 10/11 excluded.

Verified on the loaded panel: SPY (84398), QQQ (86755), IWM (88222), ARKK (14948) all absent.
The universe fell from 10,551 securities / 4,974 median names/day to **6,635 / 3,716** — the
old panel was ~1,250 funds, REITs and ADRs deep. Attrition *rose* to 4.1%/yr (funds rarely
delist, so they were diluting it into the suspicious range).

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
