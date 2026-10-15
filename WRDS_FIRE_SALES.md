# WRDS pull for the fire-sale pressure candidate

Written 2026-10-15, before any of this data existed on disk and before anything was registered;
corrected the same night to what the WRDS web forms actually serve, once every file was on disk.
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

Date range: **2005-01-01 to 2025-12-31** for the holdings and the header; the CRSP mutual-fund
files start one month earlier (2004-12) so January 2005 has a prior month to compute a flow from;
the link tables and the names history are taken whole. Everything is CSV, gzip, ISO dates. The
files live in `~/Downloads` under the names below; WRDS hands each one out under a random name
and it is renamed on arrival.

### 1. LSEG (Thomson/Refinitiv) s12 — fund holdings (the big one)

**Get Data → LSEG → Mutual Fund Holdings (S12) → Type 3: Stock Holdings** (table `tfn.s12type3`).
The web form offers exactly four variables; `RDATE` and `CHANGE` are not on it (the report date
comes from the Type 1 header through `FUNDNO, FDATE`).

| variable | why |
|---|---|
| `fundno` | the Thomson fund identifier |
| `fdate` | the vintage date of the report; join key to the header and to MFLINKS |
| `cusip` | 8-character; maps to PERMNO through the names history by date |
| `shares` | shares held; the form's label says millions but the values look like raw shares (`27000`, `2100`) — the script settles the unit against `ShrOut` before anything else |

Four slices, because a single export is over 1.5 GB and the laptop has 8 GB:

| file | rows | gzip |
|---|---|---|
| `s12_holdings_2005_2009.csv.gz` | 22.5 M | 155 MB |
| `s12_holdings_2010_2014.csv.gz` | 21.1 M | 150 MB |
| `s12_holdings_2015_2019.csv.gz` | 61.0 M | 461 MB |
| `s12_holdings_2020_2025.csv.gz` | 146.2 M | 1.12 GB |

The 2020–2025 slice is the whole of the rest put together: LSEG's coverage widened sharply after
2019 (many more funds, many of them non-US). The sizing script streams a slice in chunks and keeps
only US equity funds' rows (header `country`, CRSP `crsp_obj_cd`), so the 146 M rows never sit in
memory at once.

### 2. LSEG s12 — fund header

**Same menu → Type 1: Fund Characteristics** (table `tfn.s12type1`), all nine variables: `fundno,
rdate, fdate, fundname, mgrcoab, country, assets, ioc, prdate`. Report Date 2005-01-01 to
2025-12-31. 3.25 M rows, 64 MB. `s12_header_2005_2025.csv.gz`. `assets` is in $10,000s.

### 3. MFLINKS — the bridge between Thomson funds and CRSP funds

**Get Data → MFLINKS** (vendor page `mutual-fund-links`). The two tables are the other way round
from how the first draft of this file named them; the files follow the WRDS tables:

- `mfl.mflink1` — **CRSP_FUNDNO – WFICN**: `crsp_fundno, wficn, flag_missing_group, cusip8,
  fund_name, ticker, ncusip, merge_fundno`. No date range on the form. 50,550 rows, 1.2 MB.
  `mflink1.csv.gz`.
- `mfl.mflink2` — **FUNDNO – WFICN**: `fundno, fdate, wficn, rdate`. The form's date picker is
  month-granular and bounded by the table (1980-03 to 2025-12); the whole range was taken.
  3.81 M rows, 15 MB, and most rows carry an empty `wficn` — the table lists every Thomson
  fund-vintage, linked or not. `mflink2.csv.gz`.

Without them a fund's holdings and its flows cannot be joined, and the whole measure is undefined.

### 4. CRSP Mutual Fund Database — flows come from here

**Get Data → CRSP → Quarterly Update → Mutual Funds → Monthly Returns** (table
`crsp_q_mutualfunds.monthly_tna_ret_nav`): `caldt, crsp_fundno, mtna, mret, mnav`. 2004-12 to
2025-12. 7.50 M rows, 83 MB. `crsp_mf_monthly_2005_2025.csv.gz`.

**Same menu → Fund Summary** (table `crsp_q_mutualfunds.fund_summary`), **quarterly rows only**
(the form offers Annual and Quarterly as two checkboxes; Annual is ticked by default and was
unticked), 2004-12 to 2025-12, eighteen variables: `crsp_portno, fund_name, ticker, ncusip,
mgmt_cd, retail_fund, inst_fund, dead_flag, delist_cd, index_fund_flag, vau_fund, et_flag,
merge_fundno, per_com, si_obj_cd, crsp_obj_cd, lipper_class, lipper_asset_cd`. 2.46 M rows,
19 MB. `crsp_mf_summary_2005_2025.csv.gz`. The delivered file carries a leading `summary_period2`
column (`AQ` on quarter-ends that are also year-ends) ahead of those.

This is how equity funds are kept (`crsp_obj_cd` beginning `ED`, domestic equity) and index funds
and ETFs are excluded (`index_fund_flag`, `et_flag`) — an index fund's selling is a different
mechanism with its own literature, and mixing it in would be the two-regimes mistake of trial 6.
`per_com` is a second check that the fund actually holds common stock.

Flow for fund *f* in month *t* is `mtna_t − mtna_{t−1} × (1 + mret_t)`, summed over the fund's
share classes through `wficn`, and scaled by `mtna_{t−1}`. Quarterly flow is the three months
compounded. A fund with a quarterly flow at or below −5% is a forced seller; the −5% is the paper's
line and is fixed here before any return is read.

### 5. CRSP names history — CUSIP to PERMNO, by date

**Get Data → CRSP → Annual Update → Stock - Version 2 (CIZ) → Names** (the security information
history, `crsp_a_stock.stksecurityinfohist`), all 24 variables, whole history. 191 K rows, 3.6 MB.
`crsp_stocknames.csv.gz`. The columns are the CIZ ones, not the legacy `stocknames` ones:

| legacy | CIZ column in this file |
|---|---|
| `NCUSIP` (8-char, valid over a date range) | `cusip` |
| `NAMEDT`, `NAMEENDT` | `secinfostartdt`, `secinfoenddt` |
| `SHRCD` | `sharetype` + `securitytype` + `securitysubtype` (`NS` + `EQTY` + `COM` is a common share) |
| `EXCHCD` | `primaryexch` (`N`, `A`, `Q`) |
| `COMNAM`, `TICKER` | `issuernm`, `ticker` |

A holding's 8-character CUSIP on `fdate` maps to the PERMNO whose `cusip` row has
`secinfostartdt ≤ fdate ≤ secinfoenddt`. That is the only correct mapping: the daily file's
`HdrCUSIP` is the latest header CUSIP and drifts.

Shares outstanding come from the CIZ daily file already on disk (`ShrOut`), at each quarter end.

## Query record (shareable; the data is not)

| WRDS query | what | result |
|---|---|---|
| 11787987 | s12 Type 3, 2005–2009 | `s12_holdings_2005_2009.csv.gz` |
| 11787991 | s12 Type 3, 2010–2014 | `s12_holdings_2010_2014.csv.gz` |
| 11788003 | s12 Type 3, 2015–2019 | `s12_holdings_2015_2019.csv.gz` |
| 11788005 | s12 Type 3, 2020–2025 | `s12_holdings_2020_2025.csv.gz` |
| 11788011 | s12 Type 1, 2005–2025 | `s12_header_2005_2025.csv.gz` |
| 11788022 | MFLINKS CRSP_FUNDNO – WFICN | `mflink1.csv.gz` |
| 11788020 | MFLINKS FUNDNO – WFICN | `mflink2.csv.gz` |
| 11788030 | CRSP MF monthly, 2004-12–2025-12 | `crsp_mf_monthly_2005_2025.csv.gz` |
| 11788033 | CRSP MF summary, quarterly, 2004-12–2025-12 | `crsp_mf_summary_2005_2025.csv.gz` |
| 11788040 | CRSP CIZ Names, whole history | `crsp_stocknames.csv.gz` |

All ten ran 2026-10-14 (US evening); WRDS keeps the results two days, after which the query is
rerun from the manager, not re-specified.

Web-form notes for the next pull: the mutual-fund forms' date pickers take months (`YYYY-MM`) and
silently reject anything outside the table's own range; the variable picker's "All" button does
nothing under automation, so each box is ticked; the conditional-statement builder, once opened on
any form, reappears on every later form and must be removed before submitting or the query carries
an empty condition.

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

## Volumes, as delivered

| file | rows | gzip |
|---|---|---|
| s12 holdings, four slices | ~250 M total | 1.9 GB total |
| s12 header | 3.25 M | 64 MB |
| MFLINKS, both | 3.86 M | 16 MB |
| CRSP MF monthly | 7.50 M | 83 MB |
| CRSP MF summary (quarterly) | 2.46 M | 19 MB |
| names history | 191 K | 3.6 MB |

Load one holdings slice at a time, in chunks; the sizing never needs two in memory.
