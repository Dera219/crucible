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

Added the same morning, to recover the breaks the 13D/G-based link could not map:

| WRDS query | form | what | file |
|---|---|---|---|
| 11789715 | CRSP → Annual Update → CRSP/Compustat Merged → Compustat CRSP Link | the 242 unmapped CIKs typed as codes (format `cik`), link types LC/LU/LS, link + identifying variables | `ccm_link.csv.gz` (150 rows) |
| 11789720 | SEC Analytics Suite → SEC Linking Tables → CIK-GVKEY Link Table | the same 242 CIKs, all four sources, valid and name-mismatch links | `sec_cik_gvkey_link.csv.gz` (913 rows) |
| 11789880 | Compustat CRSP Link again | the 68 GVKEYs from the CIK-GVKEY link that the first CCM pull lacked, typed as codes (format `gvkey`) | `ccm_link_gvkey.csv.gz` (12 rows) |
| 11789892 | s34 Type 3 again | the 32 CUSIPs the GVKEY route added | `s34_deal_targets_extra.csv.gz` (102,102 rows) |

Whole-database versions of the two link tables (11789680, 11789684) sat in the WRDS queue for
twelve minutes and were killed; the narrow ones took 23 to 53 minutes to start that morning,
against seconds earlier in the day. Type codes into the box rather than pull a whole table.

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

Funnel: 6,548 episodes → 446 dropped as bidders → 517 breaks → 391 linked → 264 mapped through
the CUSIP link + 82 through CIK → GVKEY → PERMNO → 308 US common → **193** with a pre-deal cap of
at least $100M (187 CUSIPs, 184 with 13F rows). Of the 193, 164 carry a merger-agreement exhibit
within a year before t1 and 9 are SPACs by name.

## The registration set

The event rule for registration requires the merger-agreement exhibit and excludes SPAC names
and the information-statement forms (PREM14C/DEFM14C: controlled companies, no arbitrage block):
**130** events with 13F coverage on both dates. The block for the split is point in time — the
last 13F public on day 0, a report being due 45 days after its quarter-end — and at the 10% line
the set divides **54 high-block / 76 low-block**. The sized (non-public) block has a median of
10.2% over 172 covered breaks; the public one 6.0%, and 6.9% on the clean set; ordinary churn
over the same span is 4.2%.
