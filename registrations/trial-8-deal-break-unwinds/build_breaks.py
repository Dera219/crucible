"""Build the deal-break event list from SEC filings. Identifier-level: no return is read.

    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/deal-break-unwinds/build_breaks.py [DATA_DIR] [CSV_DIR]

UMD's WRDS subscription does not carry SDC M&A, so the withdrawn-deal list is built from the
WRDS SEC Analytics Suite (EDGAR indices) and CRSP identifiers:

  sec_filings_index_deal_forms_2004_2025.csv   the EDGAR filings index 2004-2025 (WRDS query
                                               11789405) cut to the form types used below
  sec_8k_items_section1_2004_2025.csv.gz       8-K Section 1 items (query 11789403)
  sec_cik_cusip_link.csv.gz                    WRDS CIK-CUSIP link, valid links (query 11789407)
  sec_cik_gvkey_link.csv.gz                    WRDS CIK-GVKEY link, valid links (query 11789684)
  ccm_link*.csv.gz                             CRSP/Compustat link history + header CIK, by the
                                               unmapped CIKs (11789715) then by GVKEY (11789880)
  crsp_stocknames.csv.gz, crsp daily slices    as in crucible/WRDS_FIRE_SALES.md

## The rule, fixed before the count

pendency   a target-side deal filing: PREM14A, DEFM14A, PREM14C, DEFM14C or SC 14D9. Filings by
           one CIK no more than 270 days apart belong to one deal episode; t1 is the first.
completion a Form 25, 25-NSE, 15-12B, 15-12G or 15-15D by the same CIK (the target leaves the
           exchange when the deal closes).
bidder     an S-4 by the same CIK between t1 - 120 days and t1 + 540 days marks the filer as
           the acquirer in a stock deal; the episode is dropped. (Acquirers file PREM14A/DEFM14A
           too when they need a share-issuance vote. SC TO-T cannot serve: EDGAR indexes the
           bidder's tender-offer statement under the target's CIK as well.)
break      an 8-K carrying Item 1.02 (Termination of a Material Definitive Agreement) by the
           same CIK, filed in (t1, t1 + 540 days], with no completion filing in the 120 days
           after it. Day 0 is the 8-K's period of report when that is within 10 days before the
           filing date, else the filing date. One break per episode: the first qualifying 1.02.
target     the CIK's CUSIP from the link table (valid links, in force on day 0 with a year of
           slack), mapped to a PERMNO through the names history on day 0; where that fails,
           CIK -> GVKEY (WRDS CIK-GVKEY link, valid links from any source, or Compustat's own
           header CIK on the CCM table) -> PERMNO through the CRSP/Compustat link history
           (LC/LU/LS links in force on day 0, primary links first), with the CUSIP read back
           from the names history. Then US common stock on the quarter-end before day 0 by
           crucible's CIZ classification with a market cap of at least $100M on the quarter-end
           before t1.

Outputs (never committed): deal_break_events.csv and deal_break_cusips.txt in DATA_DIR, the
latter being the 8-digit CUSIP list for the 13F pull. Aggregates print to stdout.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

_FIRE = Path(__file__).resolve().parent.parent / "fire-sale-pressure" / "size_premise.py"
_spec = importlib.util.spec_from_file_location("fire_sale_sizing", _FIRE)
assert _spec is not None and _spec.loader is not None
fire = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fire)

PENDENCY = ("PREM14A", "DEFM14A", "PREM14C", "DEFM14C", "SC 14D9")
COMPLETION = ("25", "25-NSE", "15-12B", "15-12G", "15-15D")
ACQUIRER_SIDE = ("S-4",)  # the bidder registers the shares it will issue; a tender
#   offer statement (SC TO-T) is indexed under the TARGET's CIK too, so it cannot be used
EPISODE_GAP_DAYS = 270
BREAK_WINDOW_DAYS = 540
QUIET_AFTER_BREAK_DAYS = 120
MIN_CAP_K = 100_000.0  # $100M, DlyCap is in thousands
FIRST_T1 = date(2004, 6, 1)
LAST_DAY0 = date(2025, 12, 31)


def filings(data: Path) -> pl.DataFrame:
    f = pl.read_csv(data / "sec_filings_index_deal_forms_2004_2025.csv", infer_schema_length=0)
    return f.select("cik", "fdate", "form", "coname").with_columns(
        pl.col("fdate").str.to_date()).unique()


def episodes(f: pl.DataFrame) -> pl.DataFrame:
    """One row per deal episode: cik, t1, t_last, filings, first form, company name."""
    p = f.filter(pl.col("form").is_in(PENDENCY)).sort(["cik", "fdate"])
    p = p.with_columns(gap=(pl.col("fdate") - pl.col("fdate").shift(1)).dt.total_days()
                       .over("cik"))
    p = p.with_columns(new=(pl.col("gap").is_null() | (pl.col("gap") > EPISODE_GAP_DAYS)))
    p = p.with_columns(episode=pl.col("new").cast(pl.Int32).cum_sum().over("cik"))
    e = p.group_by(["cik", "episode"]).agg(
        t1=pl.col("fdate").min(), t_last=pl.col("fdate").max(), filings=pl.len(),
        first_form=pl.col("form").sort_by("fdate").first(), coname=pl.col("coname").first())
    e = e.filter(pl.col("t1") >= FIRST_T1)
    bid = f.filter(pl.col("form").is_in(ACQUIRER_SIDE)).select("cik", bdate=pl.col("fdate"))
    jb = e.join(bid, on="cik", how="inner").filter(
        (pl.col("bdate") >= pl.col("t1") - timedelta(days=120))
        & (pl.col("bdate") <= pl.col("t1") + timedelta(days=BREAK_WINDOW_DAYS)))
    bidders = jb.select("cik", "episode").unique().with_columns(bidder=pl.lit(True))
    e = e.join(bidders, on=["cik", "episode"], how="left").with_columns(
        pl.col("bidder").fill_null(False))
    print(f"  episodes dropped as the bidder's side (S-4): "
          f"{e.filter(pl.col('bidder')).height:,}")
    return e.filter(~pl.col("bidder")).drop("bidder")


def breaks(data: Path, e: pl.DataFrame, f: pl.DataFrame) -> pl.DataFrame:
    k = fire.text_csv(data / "sec_8k_items_section1_2004_2025.csv.gz")
    k = k.filter(pl.col("nitem") == "1.02").select("cik", "fdate", "rdate", "accession")
    k = k.with_columns(pl.col("fdate").str.to_date(), pl.col("rdate").str.to_date(strict=False))
    k = k.with_columns(day0=pl.when((pl.col("rdate").is_not_null())
                                    & ((pl.col("fdate") - pl.col("rdate")).dt.total_days()
                                       .is_between(0, 10)))
                       .then(pl.col("rdate")).otherwise(pl.col("fdate")))
    comp = f.filter(pl.col("form").is_in(COMPLETION)).select("cik", cdate=pl.col("fdate"))
    # every (episode, 1.02) pair inside the window
    j = e.join(k, on="cik", how="inner")
    j = j.filter((pl.col("fdate") > pl.col("t1"))
                 & (pl.col("fdate") <= pl.col("t1") + timedelta(days=BREAK_WINDOW_DAYS)))
    # a completion filing inside 120 days after the 1.02 means the deal closed anyway
    jc = j.join(comp, on="cik", how="left")
    jc = jc.with_columns(closed_after=(pl.col("cdate").is_not_null()
                                       & (pl.col("cdate") >= pl.col("fdate"))
                                       & (pl.col("cdate") <= pl.col("fdate")
                                          + timedelta(days=QUIET_AFTER_BREAK_DAYS))))
    jc = jc.with_columns(closed_before=(pl.col("cdate").is_not_null()
                                        & (pl.col("cdate") >= pl.col("t1"))
                                        & (pl.col("cdate") < pl.col("fdate"))))
    g = jc.group_by(["cik", "episode", "fdate"]).agg(
        pl.col("t1").first(), pl.col("t_last").first(), pl.col("coname").first(),
        pl.col("first_form").first(), pl.col("day0").first(), pl.col("accession").first(),
        closed_after=pl.col("closed_after").any(), closed_before=pl.col("closed_before").any())
    g = g.filter(~pl.col("closed_after") & ~pl.col("closed_before"))
    b = g.sort("fdate").group_by(["cik", "episode"]).first()
    return b.filter(pl.col("day0") <= LAST_DAY0)


def completed(e: pl.DataFrame, f: pl.DataFrame) -> pl.Series:
    comp = f.filter(pl.col("form").is_in(COMPLETION)).select("cik", cdate=pl.col("fdate"))
    j = e.join(comp, on="cik", how="inner")
    j = j.filter((pl.col("cdate") >= pl.col("t1"))
                 & (pl.col("cdate") <= pl.col("t1") + timedelta(days=BREAK_WINDOW_DAYS)))
    return j.select("cik", "episode").unique().with_columns(done=pl.lit(True))


def link_cusip(data: Path, b: pl.DataFrame) -> pl.DataFrame:
    lk = fire.text_csv(data / "sec_cik_cusip_link.csv.gz")
    lk = lk.filter(pl.col("validated").is_in(["2", "3"])).select(
        "cik", "cusip", "validated", d1=pl.col("cikdate1").str.to_date(strict=False),
        d2=pl.col("cikdate2").str.to_date(strict=False))
    j = b.join(lk, on="cik", how="inner")
    j = j.with_columns(in_force=(pl.col("d1") <= pl.col("day0") + timedelta(days=365))
                       & (pl.col("d2") >= pl.col("day0") - timedelta(days=365)))
    j = j.sort(["in_force", "validated", "d2"], descending=[True, True, True])
    return j


def to_permno(j: pl.DataFrame, names: pl.DataFrame) -> pl.DataFrame:
    n = j.join(names, on="cusip", how="inner")
    n = n.filter((pl.col("secinfostartdt") <= pl.col("day0"))
                 & (pl.col("day0") <= pl.col("secinfoenddt")))
    n = n.sort(["in_force", "validated"], descending=[True, True])
    return n.unique(subset=["cik", "episode"], keep="first").rename({"permno": "PERMNO"})


FAR_FUTURE = date(2099, 12, 31)


def via_gvkey(data: Path, b: pl.DataFrame, names: pl.DataFrame) -> pl.DataFrame:
    """Breaks mapped CIK -> GVKEY -> PERMNO on day 0, with the CUSIP read back from the names."""
    g = fire.text_csv(data / "sec_cik_gvkey_link.csv.gz")
    g = g.filter(pl.col("link_desc").fill_null("").str.contains("Valid")).select("cik", "gvkey")
    parts = [fire.text_csv(f) for f in sorted(data.glob("ccm_link*.csv.gz"))]
    ccm = pl.concat([q.rename({c: c.lower() for c in q.columns}) for q in parts]).unique()
    # WRDS upper-cases the link columns; the pulls were by CIK (11789715) then by GVKEY (11789880)
    ccm = ccm.filter(pl.col("linktype").is_in(["LC", "LU", "LS"]) & pl.col("lpermno").is_not_null())
    ccm = ccm.with_columns(
        pl.col("linkdt").str.to_date(strict=False),
        linkend=pl.col("linkenddt").str.to_date(strict=False).fill_null(FAR_FUTURE),
        PERMNO=pl.col("lpermno").cast(pl.Int64, strict=False),
        prim=pl.col("linkprim").is_in(["P", "C"]))
    header = ccm.filter(pl.col("cik").is_not_null() & (pl.col("cik") != "")).select(
        pl.col("cik").str.zfill(10), "gvkey").unique()
    pairs = pl.concat([g, header]).unique()
    j = b.join(pairs, on="cik", how="inner").join(
        ccm.select("gvkey", "PERMNO", "linkdt", "linkend", "prim"), on="gvkey", how="inner")
    j = j.filter((pl.col("linkdt") <= pl.col("day0")) & (pl.col("day0") <= pl.col("linkend")))
    j = j.sort("prim", descending=True).unique(subset=["cik", "episode"], keep="first")
    n = names.rename({"permno": "PERMNO"})
    j = j.join(n, on="PERMNO", how="inner")
    j = j.filter((pl.col("secinfostartdt") <= pl.col("day0"))
                 & (pl.col("day0") <= pl.col("secinfoenddt")))
    j = j.unique(subset=["cik", "episode"], keep="first")
    return j.with_columns(validated=pl.lit("ccm"), in_force=pl.lit(True)).select(
        b.columns + ["cusip", "validated", "in_force", "PERMNO", "secinfostartdt",
                     "secinfoenddt"])


def main() -> int:
    data = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "Downloads"
    csv_dir = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else data / "wrds_csv"

    f = filings(data)
    e = episodes(f)
    done = completed(e, f)
    b = breaks(data, e, f)
    print(f"deal episodes with a target-side filing: {e.height:,}; completed inside 540 days: "
          f"{done.height:,}; broken by the 1.02 rule: {b.height:,}")
    print("  broken, by first pendency form:")
    print(b.group_by("first_form").len().sort("len", descending=True))

    j = link_cusip(data, b)
    print(f"  with a valid CIK-CUSIP link: {j.select('cik', 'episode').unique().height:,}")
    names = fire.names_history(data)
    m = to_permno(j, names)
    print(f"  mapped to a PERMNO on day 0 through the CIK-CUSIP link: {m.height:,}")
    rest = b.join(m.select("cik", "episode"), on=["cik", "episode"], how="anti")
    m2 = via_gvkey(data, rest, names)
    print(f"  mapped through CIK -> GVKEY -> PERMNO: {m2.height:,} more of {rest.height:,}")
    m = pl.concat([m.select(m2.columns), m2])

    m = m.with_columns(P=fire.quarter_end(pl.col("day0").dt.offset_by("-3mo")),
                       P1=fire.quarter_end(pl.col("t1").dt.offset_by("-3mo")))
    ps = sorted(set(m["P"].to_list()) | set(m["P1"].to_list()))
    stocks = fire.quarter_end_stocks(csv_dir, ps)
    uni = fire.common_stock(stocks)
    ev = m.join(uni, on=["PERMNO", "P"], how="inner")
    ev = ev.join(uni.select("PERMNO", P1=pl.col("P"), cap_pre=pl.col("cap")),
                 on=["PERMNO", "P1"], how="left")
    print(f"  US common stock on the quarter-end before day 0: {ev.height:,}")
    ev = ev.filter(pl.col("cap_pre") >= MIN_CAP_K)
    print(f"  with a pre-deal market cap of at least $100M: {ev.height:,}")

    ev = ev.with_columns(year=pl.col("day0").dt.year(), month=pl.col("day0").dt.strftime("%Y-%m"),
                         pendency_days=(pl.col("fdate") - pl.col("t1")).dt.total_days())
    with pl.Config(tbl_rows=30):
        print(ev.group_by("year").len().sort("year"))
    bm = ev.group_by("month").len().sort("len", descending=True)
    print(f"busiest month: {bm['month'][0]} with {bm['len'][0] / ev.height:.1%} of events; "
          f"months with any: {bm.height}")
    print(f"pre-deal cap: median ${ev['cap_pre'].median() / 1e3:,.0f}M, p25 "
          f"${ev['cap_pre'].quantile(0.25) / 1e3:,.0f}M, "
          f"p75 ${ev['cap_pre'].quantile(0.75) / 1e3:,.0f}M; "
          f"dollar volume median ${ev['dollar_vol'].median() / 1e6:,.1f}M/day; "
          f"days from first filing to break: median {ev['pendency_days'].median():.0f}")

    out = ev.select("cik", "episode", "coname", "first_form", "t1", "t_last", "fdate", "day0",
                    "accession", "cusip", "PERMNO", "P", "P1", "shrout", "cap", "cap_pre",
                    "dollar_vol").sort("day0")
    out.write_csv(data / "deal_break_events.csv")
    cusips = sorted(set(out["cusip"].to_list()))
    (data / "deal_break_cusips.txt").write_text("\n".join(cusips) + "\n")
    print(f"events -> {data / 'deal_break_events.csv'}; {len(cusips)} CUSIPs -> "
          f"{data / 'deal_break_cusips.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
