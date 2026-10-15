# WRDS pull for the deal-break unwind candidate

Written 2026-10-15, the morning after the fire-sale and liquidation sizings retired on paper.
Read `WRDS.md` first: the licence rules there apply to every file named here. Nothing below is
committed; the queries are what get shared.

## Why these files

UMD's WRDS subscription does **not** carry SDC M&A (`tr_sdc_ma`), so the withdrawn-deal list is
built from the EDGAR indices that the WRDS SEC Analytics Suite serves, and the arbitrage block
from Thomson 13F holdings. Both are identifier-level; the builder and the sizing never read a
return. The rule that turns filings into deal breaks is in the private registrations folder
(`deal-break-unwinds/build_breaks.py`) and summarised in `SEARCH_LOG.md`.

## The pulls (all run 2026-10-15, US morning)

| WRDS query | form | what | file |
|---|---|---|---|
| 11789403 | SEC Analytics Suite → SEC Filings Queries → List of 8K Items | Section 1 items only (1.01–1.05), filing date 2004-01-01 to 2025-12-31, all 14 variables | `sec_8k_items_section1_2004_2025.csv.gz` (334,706 rows) |
| 11789405 | → SEC Filings Index | the whole index 2004-01-01 to 2025-12-31: `cik, fdate, form, coname, fname` | `sec_filings_index_2004_2025.csv.gz` (23.0 M rows, 213 MB); cut locally to the deal forms as `sec_filings_index_deal_forms_2004_2025.csv` |
| 11789407 | → SEC Linking Tables → CIK-CUSIP Link Table | valid links only (validation 2 and 3), all 11 variables | `sec_cik_cusip_link.csv.gz` (33,758 rows) |
| 11789409 | → List of Filing Exhibits | form 8-K only, 2004-01 to 2025-12: `fdate, accession, form, rdate, cik, coname, fname, sequence, type, description, filename` | `sec_8k_exhibits_2004_2025.csv.gz` (12.8 M rows, 166 MB); `EX-2.*` rows are the merger agreements |
| 11789423 | LSEG → Institutional (13f) Holdings S34 → Type 3: Stock Holdings | the 156 target CUSIPs typed into the Codes box (8-digit, space-separated; the box takes 4,096 characters), file date 2004-01-01 to 2025-12-31: `fdate, cusip, mgrno, type, shares` | `s34_deal_targets_2004_2025.csv.gz` (1.76 M rows) |

The 13F pull depends on the event list, so it is rerun whenever the deal rule changes: the
CUSIP list is `deal_break_cusips.txt`, written by the builder beside the events.

Form notes beyond those in `WRDS_FIRE_SALES.md`: the 8-K items form chooses sections with radio
buttons, not checkboxes; the exhibits form's date pickers are month-granular and its form
picker's values are quoted (`'8-K'`); every form's submit button is `wrds-query-form-submit-button`
and clicking it from script is more reliable than clicking by position; a tender-offer statement
(SC TO-T) is indexed under the target's CIK as well as the bidder's.

## The deal-break rule, as sized

- **pendency**: PREM14A, DEFM14A, PREM14C, DEFM14C or SC 14D9 by a CIK; filings no more than 270
  days apart are one episode, t1 the first.
- **bidder**: an S-4 by the same CIK between t1 − 120 days and t1 + 540 days drops the episode
  (acquirers file PREM14A/DEFM14A too, for a share-issuance vote).
- **break**: an 8-K Item 1.02 by the same CIK in (t1, t1 + 540 days] with no Form 25 / 25-NSE /
  15-12B / 15-12G / 15-15D in the 120 days after it. Day 0 is the 8-K's period of report when it
  is within 10 days before the filing, else the filing date.
- **target**: CIK → CUSIP (link table) → PERMNO (names history on day 0), US common stock on the
  quarter-end before day 0, pre-deal market cap ≥ $100M on the quarter-end before t1.
- **block**: 13F shares at the last quarter-end before day 0 held by managers with none at the
  quarter-end at or before t1 − 150 days, over shares outstanding.

Funnel: 6,548 episodes → 446 dropped as bidders → 517 breaks → 391 linked → 264 mapped → 248 US
common → **160** with a pre-deal cap of at least $100M (156 CUSIPs). Of the 160, 133 carry a
merger-agreement exhibit within a year before t1 and 7 are SPACs by name.

## What the registration will need beyond this

A better CIK → PERMNO link (the 13D/G-based table loses 126 of 517 breaks and the names history
another 127): Compustat's `cik` through the CRSP/Compustat link, which UMD carries. The event
rule for registration should require the merger-agreement exhibit, exclude SPAC names and the
information-statement forms (PREM14C/DEFM14C: controlled companies, no arbitrage block), and
state whether the events are all breaks or only those with a block above the 10% line.
