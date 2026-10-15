# WRDS pull for the fire-sale pressure candidate

Written 2026-10-15, before any of this data existed on disk and before anything was registered.
Read `WRDS.md` first: the licence rules there apply to every file named here. Nothing below is
committed; the queries are what get shared.

## The claim being sized, in one paragraph

Mutual funds hit by heavy redemptions must sell what they hold, regardless of price, inside the
quarter. The stocks those funds hold in size relative to the stock's float are pressed during the
quarter and recover over the following one to three quarters as the selling ends (Coval &
Stafford, *Asset fire sales (and purchases) in equity markets*, JFE 2007). Structural constraint:
the seller is a mandate, not an opinion. The durable part is where a fund cannot go — the pressure
concentrates in small names that a distressed fund cannot leave quietly, the recovery takes months,
and the capacity is what a committee would never approve. The decay risk is stated up front: the
pressure quarter itself has been front-run by hedge funds since the paper (Shive & Yun 2013), so
what is left, if anything, is the reversal after it, in the smaller names.

**It dies on paper if the premise is small.** The premise is a number: what fraction of a stock's
shares outstanding do funds with large outflows hold. That is computed from the holdings and the
flows below without reading a single stock return, which is the lesson of trial 7 applied.

## Five pulls, four of them small

Date range everywhere: **2005-01-01 to 2025-12-31** (the regime the claim would be registered on).
Export as CSV (gzip if offered). Name the files as shown; the sizing script will look for them in
`~/Downloads`.

### 1. Thomson/Refinitiv s12 — fund holdings (the big one)

**Get Data → Thomson/Refinitiv → Mutual Fund Holdings (s12) → Holdings** (table `tfn.s12type3`).

| variable | why |
|---|---|
| `FUNDNO` | the Thomson fund identifier |
| `FDATE` | the vintage date of the report; join key to the header and to MFLINKS |
| `RDATE` | the report date (the holdings' as-of date) — keep both, they differ |
| `CUSIP` | 8-character; maps to PERMNO through `stocknames` by date |
| `SHARES` | shares held |
| `CHANGE` | change in shares since the previous report, when supplied |

Pull it in **four slices** like the CRSP daily file, because a single export can be ~1.5 GB and the
laptop has 8 GB: `s12_holdings_2005_2009.csv.gz`, `s12_holdings_2010_2014.csv.gz`,
`s12_holdings_2015_2019.csv.gz`, `s12_holdings_2020_2025.csv.gz`. Roughly 50 million rows in all.

### 2. Thomson/Refinitiv s12 — fund header

**Same menu → Fund Information** (table `tfn.s12type1`): `FUNDNO`, `FDATE`, `RDATE`, `ASSETS`,
`IOC` (investment objective code), `COUNTRY`. Small. Name it `s12_header_2005_2025.csv.gz`.

### 3. MFLINKS — the bridge between Thomson funds and CRSP funds

**Get Data → MFLINKS**:

- `mfl.mflink1`: `FUNDNO`, `FDATE`, `WFICN` → `mflink1.csv.gz`
- `mfl.mflink2`: `CRSP_FUNDNO`, `WFICN` → `mflink2.csv.gz`

Both tiny. Without them a fund's holdings and its flows cannot be joined, and the whole measure is
undefined. No date range needed; take everything.

### 4. CRSP Mutual Fund Database — flows come from here

**Get Data → CRSP → Mutual Funds → Monthly Total Net Assets, Returns and NAV** (table
`crsp.monthly_tna_ret_nav`): `CRSP_FUNDNO`, `CALDT`, `MTNA`, `MRET`, `MNAV`. About 8 million
rows. Name it `crsp_mf_monthly_2005_2025.csv.gz`.

**Same menu → Fund Summary** (table `crsp.fund_summary`): `CRSP_FUNDNO`, `CALDT`, `CRSP_OBJ_CD`,
`INDEX_FUND_FLAG`, `ET_FLAG`, `FUND_NAME`. This is how equity funds are kept (`CRSP_OBJ_CD`
beginning `ED`, domestic equity) and index funds and ETFs are excluded — an index fund's selling is
a different mechanism with its own literature, and mixing it in would be the two-regimes mistake
of trial 6. Name it `crsp_mf_summary_2005_2025.csv.gz`.

Flow for fund *f* in month *t* is `MTNA_t − MTNA_{t−1} × (1 + MRET_t)`, summed over the fund's
share classes through `WFICN`, and scaled by `MTNA_{t−1}`. Quarterly flow is the three months
compounded. A fund with a quarterly flow at or below −5% is a forced seller; the −5% is the paper's
line and is fixed here before any return is read.

### 5. CRSP stock names — CUSIP to PERMNO, by date

**Get Data → CRSP → Stock / Security Files → Stock Header Info** or **Names** (table
`crsp.stocknames`): `PERMNO`, `PERMCO`, `NAMEDT`, `NAMEENDT`, `NCUSIP`, `CUSIP`, `TICKER`,
`SHRCD`, `EXCHCD`. Whole history, no date range. Name it `crsp_stocknames.csv.gz`. The 8-character
`NCUSIP` with its validity dates is the only correct way to map a 2008 holding's CUSIP to a
PERMNO — the daily file's `HdrCUSIP` is the latest header CUSIP and drifts.

Shares outstanding come from the CIZ daily file already on disk (`ShrOut`), at each quarter end.

## What happens before anything is registered

A sizing script, identifier-level, reads holdings, flows, links and shares outstanding — never a
stock return — and reports, per quarter and per size decile:

- the share of a stock's shares outstanding held by funds whose quarterly flow is at or below −5%,
  weighted by those funds' flows (the paper's expected forced sales);
- how many stocks each quarter carry forced sales above 1%, 2% and 5% of shares outstanding;
- how much of that sits in names a small account could trade (dollar volume from the daily file).

The decision rule, written down now: if the median stock in the top pressure decile has expected
forced sales under 1% of its shares outstanding, the premise cannot move a price and the candidate
is retired on paper, trial count unchanged. If it clears, the registration is a quarterly
cross-sectional signal — not an event — ranked by pressure at each quarter end, held for the
following quarter(s), and graded by crucible's cross-sectional harness (IC, monotonicity,
walk-forward, deflation against **8** lifetime trials, t ≥ 2.73). Point-in-time rule for the
signal: holdings from the latest report at least 45 days old on the trade date (the filing lag),
flows through the month end before it. `crucible.causality` polices that and nothing else does.

## Volumes, for the laptop

| file | rows (approx.) | gzip size (approx.) |
|---|---|---|
| s12 holdings, four slices | 50 M total | 1.5 GB total |
| s12 header | 1 M | 20 MB |
| MFLINKS, both | < 1 M | < 10 MB |
| CRSP MF monthly | 8 M | 150 MB |
| CRSP MF summary | 3 M | 100 MB |
| stocknames | 0.1 M | 5 MB |

Load one holdings slice at a time; the sizing never needs two in memory.
