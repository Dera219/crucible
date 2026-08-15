# Getting data out of WRDS and into crucible

Written while access was pending. Everything here is ready to run the day it lands.

## Before anything: do not commit the data

CRSP and Compustat are **licensed to the subscribing institution**. Redistributing an extract —
including pushing one to a public repository — violates the WRDS subscriber agreement and can cost
you, and everyone else at the university, access.

`.gitignore` blocks `*.csv`, `*.parquet`, `*.dta` and a `data/` directory for exactly this reason,
and the patterns are deliberately broad rather than precise. Keep it that way. If you need to share
a dataset, share the **query** that produces it.

## The one query you need first

WRDS has a web query builder (**Get Data → CRSP → Stock / Security Files → Daily Stock File**).
Use it before bothering with the Python client — a CSV export is enough to be running the same
afternoon, and `crucible.data.load_crsp_csv` reads it directly.

Ask for these variables, and no others to begin with:

| variable | why |
|---|---|
| `PERMNO` | the identifier. Not the ticker — tickers are recycled between unrelated companies |
| `date` | |
| `PRC` | price. **Negative means the security did not trade**; the value is a negated bid/ask midpoint |
| `VOL` | share volume, for the liquidity screen and the impact model |
| `RET` | holding-period return, dividends included. Ships as a string with letter codes for missing |
| `SHROUT` | shares outstanding, for market cap |
| `SHRCD` | share code. 10 and 11 are US ordinary common shares |
| `EXCHCD` | exchange code. 1/2/3 are NYSE / AMEX / NASDAQ |
| `TICKER` | a label, never a key |
| `DLRET` | **the delisting return.** The single most valuable column in the file |

### If WRDS serves the new (CIZ) format instead

Extracts pulled after CRSP retired the legacy SIZ format (December 2024) use Flat File Format
2.0 and `crucible.data.load_crsp_ciz`. The variables above map to `PERMNO`, `DlyCalDt`, `DlyPrc`,
`DlyVol`, `DlyPrcVol`, `DlyRet`, `ShrOut`, `ShareType`, `PrimaryExch`, `Ticker` — plus the four
classification columns `SecurityType`, `SecuritySubType`, `IssuerType`, `USIncFlg` (required by
default: `sharetype` alone cannot exclude ETFs), and **`DlyPrcFlg`**. That last one matters more
than it looks: CIZ prices are unsigned, so the legacy "negative PRC means no trade" convention is
gone and `DlyPrcFlg` is the only way to tell a traded price from a bid/ask midpoint nobody
filled. Without it every quoted session looks tradable, and the loader warns about exactly that.

#### What `DlyPrcFlg` actually contains

Cross-tabulated against the whole-market 2015-2025 extract (WRDS query 11570520, 23,101,820
rows) rather than taken from the documentation, because the loader's behaviour hangs off these
values. An earlier single-year check (2024) missed one flag entirely — see `HA` below.

| Flag | Rows | Delisting | Volume | Price | Return | Verdict |
|---|---|---|---|---|---|---|
| `TR` | 22,397,840 | none | median 132,476 | yes | yes | a trade. Tradable |
| `BA` | 595,088 | none | median 0 | yes | yes | bid/ask average — quote, not a fill |
| `NT` | 95,765 | none | none | none | none | no trade |
| `DA` | 5,393 | **100%** | none | **all 0.0** | yes | delisting amount |
| `SU` | 4,662 | none | none | none | none | suspended |
| `MP` | 1,634 | none | none | none | none | missing price |
| `DP` | 1,238 | **100%** | none | yes | yes | delisting payment |
| `DM` | 125 | 100% | none | none | none | delisting, no price |
| `HA` | 75 | none | none | none | none | halted |

All eight non-`TR` values are withdrawn from investability by `CIZ_NON_TRADED_PRICE_FLAGS`.

Three things this table settled that reading the documentation did not:

**`BA` was right.** The original assumption held at eleven-year scale: 595,088 rows, not one of
them a delisting, median volume zero.

**`DA` and `DP` are payouts, not prices.** Both are 100% delisting rows with no volume, and both
are in every single case the security's final row — `DA` with a literal price of `0.0`. Left
investable, that terminal payout became the security's last tradable session and pre-empted the
delisting return the delisting table exists to supply.

**`NT`, `SU`, `MP` and `HA` were investable, and that is not obvious.** All four carry null
price, null volume AND null return. They look untradable and were not: the adjusted series is
compounded from `DlyRet` with `fill_null(0.0)`, so a null return reads as "no move", the previous
price carries forward, and `investable` — which asks only whether the price is non-NaN — answers
yes. `DlyPrc` never enters into it. A fabricated `NT` session came out of the loader priced at
10.10 and fully tradable.

`HA` appears 75 times in eleven years and **zero times in 2024**. It is the argument for pulling
the full history rather than a representative year: a single year cannot show you a flag that
year did not happen to use.
Delisting returns are a separate query in CIZ (**Stock Delisting Information**, joined on
PERMNO) — pass it as `delisting_path`.

Date range: start as early as the subscription allows. The evidence requirement measured in
`crucible.significance` is blunt — against a 50-trial search, an edge with an annualised Sharpe
near 1.27 needs roughly 20 years of daily data to distinguish itself from luck, and one near 0.79
needs about 40. Depth is not a luxury here, it is the difference between being able to conclude
anything and not.

A whole-market daily extract over 20+ years is large. If the web interface times out, pull it in
five-year slices and concatenate — `load_crsp_csv` does not care how the file was assembled.

## Then

```python
from crucible.data import load_crsp_csv
from crucible.universe import Universe, listing_mask, liquidity_screen, price_floor_screen

data = load_crsp_csv("crsp_daily.csv")
print(data.summary())          # read this before anything else
```

`summary()` reports coverage, breadth, and how many securities delisted in-sample. **If it says
nothing delisted, stop** — the extract has been filtered somewhere upstream and no code
downstream can recover the missing names.

```python
universe = Universe.from_masks(
    listing_mask(data.prices.index, data.assets, data.listings),
    liquidity_screen(data.dollar_volume, top_n=1000),
    price_floor_screen(data.raw_prices, min_price=5.0),
    index=data.prices.index,
    assets=data.assets,
)
print(universe.audit().summary())
```

Each screen reads one panel and answers one question. `liquidity_screen` used to take a `prices`
panel and a `min_price` floor as well; it applied a trailing mean to whatever panel it was
handed, and what a backtest hands it is the **adjusted** one, so the "$5 line" drifted with every
split. **That is now a `TypeError`, deliberately** — see the note in `HANDOFF.md`. The floor
comes from `price_floor_screen` against the raw tape instead, and `load_crsp_csv` returns
`raw_prices=None`, so on a legacy SIZ extract that third line raises with a message saying why,
rather than quietly measuring the wrong panel. Pull in CIZ format (the section above) and use
`load_crsp_ciz` if you want the floor; drop the line if you accept a universe without one.

Then a signal, checked for causality before it is ever backtested:

```python
from crucible.causality import assert_causal
from crucible.diagnostics import diagnose
from crucible.ops import cs_demean, cs_rank, cs_scale

def momentum(prices):
    return cs_scale(cs_demean(cs_rank(prices.pct_change(252).shift(21))))

assert_causal(momentum, data.prices)                    # milliseconds
print(diagnose(universe.apply(momentum(data.prices)), data.prices))
```

`diagnose` kills most ideas in under a second. Only what survives it is worth a backtest, and
only what survives a backtest is worth walking forward.

```python
from crucible.costs import CostModel
from crucible.walkforward import rolling_folds, walk_forward

folds = rolling_folds(data.prices, train_size=1260, test_size=252, warmup=300)
result = walk_forward(
    folds, momentum, data.prices,
    costs=CostModel(),
    delisting_return=data.delisting_returns,   # use CRSP's record, not an assumption
)
print(result.summary())
```

Set `warmup` above the signal's longest lookback. A 252-day momentum with `warmup=0` reports
UNTESTABLE, which is correct and is the bug that broke Apex's baseline.

## Compustat, when you get to fundamentals

Join on **`RDQ`**, the report date — never on `datadate`, the fiscal period end. A quarter ending
31 March is not knowable on 31 March; it becomes knowable when the filing appears, typically six
to eight weeks later. Joining on the fiscal date hands the strategy weeks of foreknowledge on
every position, every quarter.

`crucible.data.check_point_in_time` measures the lag in your extract and reports it. This is the
one lookahead `crucible.causality` cannot catch — it is baked into the data before any code runs,
so no amount of perturbation testing will see it.

## If WRDS is declined

`load_crsp_csv` accepts a `columns` mapping, so Sharadar's SEP/TICKERS export can be remapped
without touching the loader. The parts that matter are: a permanent identifier that is not the
ticker, delisted securities present, and a recorded delisting return. Anything with those three
works. Anything without them makes `NaN means not investable` decorative, however correct the
code downstream.
