# Handoff — 2026-08-12

State at the end of the session that loaded real CRSP data for the first time. Everything below
is verified against the actual extract, not against fixtures.

## Where things stand

crucible is complete as a platform and has real data in it. **No strategy has been tested and no
hypothesis has been registered** — deliberately, because a backtest run before the criteria are
fixed is a trial that raises the deflation bar for whatever is eventually claimed.

```
367 tests · ruff + mypy strict clean (package AND scripts) · public at github.com/Dera219/crucible
```

## The data

Two files in the repo root, both gitignored (`*.csv` is blocked — CRSP is licensed to UMD and
publishing an extract could cost the university's access for everyone):

| file | rows | contents |
|---|---|---|
| `crsp_daily.csv` | 23,101,820 | 2.4 GB · CIZ + classification columns **+ `DlyPrcFlg`** · 2015-01-02 → 2025-12-31 · WRDS query 11570520 |
| `crsp_delist.csv` | 6,227 | delisting table with `DelRet` |

Loads into `(2766, 6635)` panels after the common-stock filter. Median breadth **3,716
names/day** — US common stock only; ETFs, CEFs, REITs and ADRs are excluded by
`CIZ_COMMON_STOCK_FILTER` (see item 2).

Verified properties:
- **2,964 securities delisted in-sample**, 4.1%/yr attrition — inside the typical 4-8% band
- **331 delisting returns worse than -50%**, worst -100% — the bankruptcies are present
- SPY, QQQ, IWM, ARKK confirmed absent from the loaded panel
- **Adjusted prices reproduce CRSP's own `DlyRet` exactly** (0 disagreements >1bp)

## The search log

`SEARCH_LOG.md` records every exploratory look at the data, especially the dead ones. It exists
because `significance` deflates by trial count and `preregistration.summarize()` reports the
honest total — and both are defeated by searches that happened and were never written down. An
exploratory look is a trial the moment the data is queried with an outcome in mind, not when it
succeeds.

**Three trials are on the budget as of 2026-08-14** (short-horizon reversal, turn-of-month, the
sub-$5 segment — all dead). Add them to any hypothesis registered against this universe before
computing a deflated Sharpe.

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

### 5. CIZ price flags — RESOLVED (verified, not assumed)

`CIZ_QUOTE_ONLY_PRICE_FLAGS = ("BA",)` was taken from CRSP's documentation and was the last
unverified assumption in the loader. Settled by re-pulling a whole-market 2024 extract WITH
`DlyPrcFlg` (WRDS query 11568315, 2,404,143 rows; Rerun saved as `?saved_query=7651357`) and
cross-tabbing every flag value against volume and delisting status. Full table in WRDS.md.

`BA` held: 44,396 rows, no delistings, median volume zero. But the cross-tab also found what the
documentation reading missed — `DA` (501 rows) and `DP` (248) are 100% delisting rows with no
volume, and are in EVERY case the security's final row, `DA` with a literal price of 0.0. Those
terminal payouts were investable, so each became its security's last tradable session and
pre-empted the delisting return the delisting table exists to supply. Now withdrawn via
`CIZ_NON_TRADED_PRICE_FLAGS`.

**Extended to the full 2015-2025 history 2026-08-14** (query 11570520, 23,101,820 rows), which
found two more things a single year could not. `HA` (halted) exists and never occurred in 2024
at all. And `NT`/`SU`/`MP`/`HA` — 102,261 rows with null price, null volume AND null return —
were INVESTABLE, because the series compounds `DlyRet` with `fill_null(0.0)`, so a null return
carries the previous price forward and `investable` only asks whether the price is non-NaN. A
fabricated `NT` day loaded at 10.10, fully tradable. All eight non-`TR` flags are now withdrawn;
9 flag tests total.

**RESOLVED 2026-08-14 — the repo-root `crsp_daily.csv` now carries `DlyPrcFlg`.** It was replaced
with the WRDS query 11570520 extract (19 columns; the flag sits at position 14). The superseded
flagless file is preserved beside it as `crsp_daily_SUPERSEDED_no_flag.csv` and can be deleted —
both are gitignored by the blanket `*.csv` rule.

Measured on the swap, which is what the flag was worth: the old file marked **10,565,390** cells
investable and the new one **10,394,463**. The difference — **170,927 name-days** — is sessions a
strategy could have transacted on where the security did not trade. Twelve more securities are
also now correctly recognised as delisted, because the `DA`/`DP` payment rows no longer extend a
name's tradable life past its own delisting. The loader emits no warning on the new file.

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

### 6. Price floor on the RAW tape — ADDED 2026-08-14, uncommitted

`crucible.universe.price_floor_screen` composes with `listing_mask` and `liquidity_screen`
through `Universe.from_masks`. It is **not** wired into any default: nothing changes unless a
universe explicitly intersects it, so no existing result moves.

It reads `Dataset.raw_prices` and **raises when that is `None`** rather than falling back to
`Dataset.prices`. That is the load-bearing part. `prices` is a total-return index compounded from
each security's first observation, so a "$5 line" on it drifts with every split and dividend:
measured over 7,698,163 liquid name-days, the adjusted level runs at median 1.00x the raw price
but p99 3.16x and max 70x, and a $5 adjusted line disagrees with the tape on **195,447 name-days
across 1,556 securities** — 16,874 of them penny stocks it would admit, 178,573 of them
reverse-split names above $5 on the tape that it would throw out.

Forward 21-session returns by raw price bucket, >$500k/day, survivorship-free (delisted names
liquidated at CRSP's `DelRet` and held flat), measured on the loaded extract:

| raw price | annualised | median 21d | delisted within 21d |
|---|---|---|---|
| <$1 | −8.75% | −9.47% | 3.05% |
| $1-2 | +0.55% | −4.80% | 0.51% |
| $2-5 | +0.80% | −2.83% | 0.46% |
| $5-10 | +6.50% | −0.31% | 0.51% |
| $10-30 | +10.39% | +0.35% | 0.56% |
| $30-100 | +9.92% | +0.69% | 0.39% |
| >$100 | +11.04% | +0.84% | 0.24% |

Defaults are `min_price=5.0` (where the sign changes: +0.80% → +6.50%) and `entry_price` at
1.2x that, i.e. **exit at $5.00, join at $6.00**. The buffer is the same hysteresis argument as
the rank band and was sized the same way — a bare $5 line manufactured **61.2% of the book per
year** in forced round trips, worse than the 37.7% that got the rank band widened; $6.00 entry
cuts it to 22.9% for 1.0% of mean membership, and the $5-6 zone it defers is the weakest slice
above the floor (+2.57% annualised, median −1.58%). Exit is deliberately unbuffered.

Not decayed: splitting at 2020-10-06, every cheap bucket is worse in the second half (<$1 goes
+1.45% → −13.78%, $2-5 goes +9.02% → −4.70%) while $10-30 is flat at ~+10%.

**This is hygiene, not an edge**, and the docstring says so at length. It removes a segment that
destroyed value; it does not add anything. A strategy that only works with this screen applied
was short the penny-stock complex, which is expensive to borrow and often un-shortable.

A $10 floor scores better on every line of the table and was refused: cutting a further 10.3% of
breadth out of a +6.50%/yr bucket is a factor tilt, and a tilt belongs in a registered hypothesis
where the deflation arithmetic charges for it, not in the universe definition where nothing does.

Left open: `liquidity_screen`'s own `min_price` still applies a trailing average to whatever panel
it is handed, which for a backtest is the adjusted one. Its default of 5.0 therefore drifts. It is
documented rather than changed, because changing it would move existing results silently; the
honest fix is to compose `price_floor_screen` and pass `min_price=0.0` to `liquidity_screen`.

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
