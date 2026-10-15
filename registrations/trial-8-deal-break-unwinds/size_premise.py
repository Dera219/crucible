"""Size the deal-break premise: the arbitrage block. Identifier-level: no return is read.

    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/deal-break-unwinds/size_premise.py [DATA_DIR] [CSV_DIR]

Reads `deal_break_events.csv` (from build_breaks.py) and the 13F holdings of the targets,
`s34_deal_targets*.csv.gz` (WRDS s34 Type 3, the targets' CUSIPs, 2004-2025; query 11789423
for the first 156 CUSIPs, 11789892 for the 32 recovered through the GVKEY link).

## The measure, fixed before it was computed

For each break with day 0 and first target-side filing t1:
  Qb   the last quarter-end before day 0 (the 13F report the arbitrageurs are still in)
  Qa   the quarter-end at or before t1 - 150 days: before the deal was announced for a proxy
       filed 30-120 days after announcement; one quarter earlier than needed for a tender
  block_new   shares held at Qb by 13F managers holding none at Qa, over shares outstanding at
              Qb (the paper's "new holders since announcement": the arbitrage block)
  block_up    the sum over all managers of max(0, shares_Qb - shares_Qa), over shares
              outstanding (new holders plus additions by existing holders)
  inst_Qa, inst_Qb   total 13F ownership at each date (the units check: below one)

Checks printed beside the verdict, not part of it: the same block with Qa dated from the
merger-agreement 8-K (an EX-2.* exhibit) where one exists; a baseline block over a window of
the same span ending before the deal (ordinary holder churn); the agreement-confirmed,
non-SPAC subset, by first filing form.

## The decision rule (the draft of 2026-10-15, before any of this was pulled)

Proceed to registration only if there are at least 150 breaks, no calendar month holds more
than 15% of them, and the median target's block_new is at least 10% of shares outstanding.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import polars as pl

_FIRE = Path(__file__).resolve().parent.parent / "fire-sale-pressure" / "size_premise.py"
_spec = importlib.util.spec_from_file_location("fire_sale_sizing", _FIRE)
assert _spec is not None and _spec.loader is not None
fire = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fire)

MIN_EVENTS = 150
MAX_MONTH_SHARE = 0.15
MIN_MEDIAN_BLOCK = 0.10
PRE_ANNOUNCEMENT_LEAD_DAYS = 150
FILING_LAG_DAYS = 45  # a 13F is due 45 days after quarter-end; before that it is not public


def events(data: Path) -> pl.DataFrame:
    e = pl.read_csv(data / "deal_break_events.csv", infer_schema_length=0)
    e = e.with_columns(pl.col("t1", "day0", "P", "P1").str.to_date(),
                       pl.col("shrout", "cap", "cap_pre", "dollar_vol").cast(pl.Float64),
                       pl.col("PERMNO").cast(pl.Int64))
    lead = timedelta(days=PRE_ANNOUNCEMENT_LEAD_DAYS)
    e = e.with_columns(Qb=pl.col("P"), Qa=fire.quarter_end(pl.col("t1") - lead))
    # the last 13F that was PUBLIC on day 0: a report is due 45 days after its quarter-end
    e = e.with_columns(Qp=fire.quarter_end((pl.col("day0") - timedelta(days=FILING_LAG_DAYS))
                                           .dt.offset_by("-3mo")))
    # Qa must precede Qb: a break inside the first quarter after the filing gets the quarter before
    e = e.with_columns(Qa=pl.when(pl.col("Qa") >= pl.col("Qb"))
                       .then(fire.quarter_end(pl.col("Qb").dt.offset_by("-3mo")))
                       .otherwise(pl.col("Qa")))
    return e


SPAC_NAME = r"ACQUISITION (CORP|CO\b|COMPANY)|ACQUISITION$|CAPITAL ACQUISITION"


def with_agreements(data: Path, e: pl.DataFrame) -> pl.DataFrame:
    """Add the merger-agreement 8-K (an EX-2.* exhibit by the same CIK within a year before t1),
    an announcement-dated Qa2, a SPAC flag, and a baseline window of the same span ending at Qa2."""
    ex = pl.scan_csv(data / "sec_8k_exhibits_2004_2025.csv.gz", infer_schema_length=0).filter(
        pl.col("type").str.to_uppercase().str.starts_with("EX-2")).select(
        "cik", "fdate").collect().with_columns(pl.col("fdate").str.to_date())
    j = e.select("cik", "episode", "t1").join(ex, on="cik", how="inner")
    j = j.filter((pl.col("fdate") >= pl.col("t1") - timedelta(days=365))
                 & (pl.col("fdate") <= pl.col("t1") + timedelta(days=30)))
    ann = j.group_by(["cik", "episode"]).agg(ann=pl.col("fdate").max())
    e = e.join(ann, on=["cik", "episode"], how="left")
    e = e.with_columns(Qa2=pl.when(pl.col("ann").is_not_null())
                       .then(fire.quarter_end(pl.col("ann").dt.offset_by("-3mo")))
                       .otherwise(pl.col("Qa")))
    e = e.with_columns(Qa2=pl.when(pl.col("Qa2") >= pl.col("Qb"))
                       .then(fire.quarter_end(pl.col("Qb").dt.offset_by("-3mo")))
                       .otherwise(pl.col("Qa2")))
    e = e.with_columns(spac=pl.col("coname").str.to_uppercase().str.contains(SPAC_NAME),
                       span=(pl.col("Qb") - pl.col("Qa2")).dt.total_days())
    return e.with_columns(Qa0=fire.quarter_end(pl.col("Qa2") - pl.duration(days=pl.col("span"))
                                               - pl.duration(days=1)))


def holdings(data: Path) -> pl.DataFrame:
    files = sorted(data.glob("s34_deal_targets*.csv.gz"))
    h = pl.concat([fire.text_csv(f) for f in files])
    h = h.select("fdate", "cusip", "mgrno", "shares").with_columns(
        pl.col("fdate").str.to_date(), pl.col("shares").cast(pl.Float64, strict=False))
    h = h.filter(pl.col("shares") > 0)
    return h.group_by(["fdate", "cusip", "mgrno"]).agg(pl.col("shares").sum())


def block(e: pl.DataFrame, h: pl.DataFrame) -> pl.DataFrame:
    hb = h.rename({"fdate": "Qb", "shares": "s_b"})
    ha = h.rename({"fdate": "Qa", "shares": "s_a"})
    # all managers appearing at either date, per event
    kb = e.select("cik", "episode", "cusip", "Qb").join(hb, on=["cusip", "Qb"], how="inner")
    ka = e.select("cik", "episode", "cusip", "Qa").join(ha, on=["cusip", "Qa"], how="inner")
    m = kb.select("cik", "episode", "mgrno", "s_b").join(
        ka.select("cik", "episode", "mgrno", "s_a"), on=["cik", "episode", "mgrno"], how="full",
        coalesce=True).with_columns(pl.col("s_a", "s_b").fill_null(0.0))
    agg = m.group_by(["cik", "episode"]).agg(
        inst_a=pl.col("s_a").sum(), inst_b=pl.col("s_b").sum(),
        new_holders=((pl.col("s_a") == 0) & (pl.col("s_b") > 0)).sum(),
        managers_b=(pl.col("s_b") > 0).sum(),
        block_new_sh=pl.col("s_b").filter(pl.col("s_a") == 0).sum(),
        block_up_sh=(pl.col("s_b") - pl.col("s_a")).clip(lower_bound=0).sum(),
    )
    has_a = ka.select("cik", "episode").unique().with_columns(has_a=pl.lit(True))
    has_b = kb.select("cik", "episode").unique().with_columns(has_b=pl.lit(True))
    out = e.join(agg, on=["cik", "episode"], how="left").join(has_a, on=["cik", "episode"],
                                                              how="left")
    out = out.join(has_b, on=["cik", "episode"], how="left")
    so = pl.col("shrout") * 1e3
    return out.with_columns(
        covered=pl.col("has_a").fill_null(False) & pl.col("has_b").fill_null(False),
        inst_a_pct=pl.col("inst_a") / so, inst_b_pct=pl.col("inst_b") / so,
        block_new=pl.col("block_new_sh") / so, block_up=pl.col("block_up_sh") / so,
    )


def _blk(ev: pl.DataFrame, h: pl.DataFrame, qa: str, qb: str) -> pl.DataFrame:
    x = ev.select("cik", "episode", "cusip", "shrout", "cap_pre", "coname", "PERMNO", "t1",
                  "day0", "first_form", Qa=pl.col(qa), Qb=pl.col(qb))
    return block(x, h).filter(pl.col("covered"))


def diagnostics(e: pl.DataFrame, h: pl.DataFrame) -> list[tuple[str, int, float, float, float]]:
    """The same block under the checks a sceptic would ask for; printed and recorded, not
    part of the verdict."""
    confirmed = e.filter(pl.col("ann").is_not_null() & ~pl.col("spac"))
    runs = [
        ("announcement-dated window (EX-2.* 8-K)", e, "Qa2", "Qb"),
        ("POINT-IN-TIME: last 13F public on day 0", e, "Qa2", "Qp"),
        ("  point-in-time, agreement-confirmed, no SPAC, no 14C",
         confirmed.filter(~pl.col("first_form").is_in(["PREM14C", "DEFM14C"])), "Qa2", "Qp"),
        ("baseline: same span, ending before the deal", e, "Qa0", "Qa2"),
        ("agreement-confirmed, no SPAC", confirmed, "Qa2", "Qb"),
    ]
    for form in ("PREM14A", "DEFM14A", "SC 14D9", "PREM14C"):
        runs.append((f"  confirmed, first filing {form}", confirmed.filter(pl.col("first_form")
                                                                            == form), "Qa2", "Qb"))
    out = []
    print(f"breaks with a merger-agreement exhibit: {e.filter(pl.col('ann').is_not_null()).height}"
          f" of {e.height}; SPAC-named: {e.filter(pl.col('spac')).height}")
    for label, ev, qa, qb in runs:
        if ev.height == 0:
            continue
        r = _blk(ev, h, qa, qb)
        q = (r["block_new"].quantile(0.25), r["block_new"].median(), r["block_new"].quantile(0.75))
        print(f"  {label:<46} n={r.height:>3}  block_new p25 {q[0]:.1%}  p50 {q[1]:.1%}  "
              f"p75 {q[2]:.1%}")
        out.append((label, r.height, q[0], q[1], q[2]))
        if label.startswith("  point-in-time"):
            hi = r.filter(pl.col("block_new") >= MIN_MEDIAN_BLOCK).height
            print(f"    split at the {MIN_MEDIAN_BLOCK:.0%} line: {hi} high-block, "
                  f"{r.height - hi} low-block")
    return out


def main() -> int:
    data = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "Downloads"
    e = events(data)
    h = holdings(data)
    print(f"breaks: {e.height:,}; 13F rows for the targets: {h.height:,} "
          f"({h['fdate'].min()} to {h['fdate'].max()}, {h['cusip'].n_unique()} CUSIPs)")
    e = with_agreements(data, e)
    b = block(e, h)
    c = b.filter(pl.col("covered"))
    print(f"breaks with 13F holdings at both Qa and Qb: {c.height:,} "
          f"(Qa only {b.filter(pl.col('has_a').fill_null(False)).height:,}, "
          f"Qb only {b.filter(pl.col('has_b').fill_null(False)).height:,})")
    for col in ("inst_a_pct", "inst_b_pct", "block_new", "block_up"):
        q = [c[col].quantile(x) for x in (0.1, 0.25, 0.5, 0.75, 0.9)]
        print(f"  {col:>11}: p10 {q[0]:.1%}  p25 {q[1]:.1%}  p50 {q[2]:.1%}  p75 {q[3]:.1%}  "
              f"p90 {q[4]:.1%}")
    print(f"  new holders at Qb: median {c['new_holders'].median():.0f} of "
          f"{c['managers_b'].median():.0f} managers")
    for lo, hi in ((0, 500_000), (500_000, 2_000_000), (2_000_000, 1e12)):
        s = c.filter((pl.col("cap_pre") >= lo) & (pl.col("cap_pre") < hi))
        if s.height:
            print(f"  pre-deal cap ${lo / 1e3:,.0f}M-{hi / 1e3:,.0f}M: {s.height} breaks, "
                  f"median block_new {s['block_new'].median():.1%}, block_up "
                  f"{s['block_up'].median():.1%}")
    diag = diagnostics(e, h)
    by_month = e.with_columns(month=pl.col("day0").dt.strftime("%Y-%m")).group_by("month").len()
    busiest = by_month["len"].max() / e.height
    med = c["block_new"].median()
    ok = e.height >= MIN_EVENTS and busiest <= MAX_MONTH_SHARE and med >= MIN_MEDIAN_BLOCK
    verdict = (f"{e.height} breaks (bar {MIN_EVENTS}); busiest month {busiest:.1%} (bar "
               f"{MAX_MONTH_SHARE:.0%}); median new-holder block {med:.1%} of shares outstanding "
               f"(bar {MIN_MEDIAN_BLOCK:.0%}) over {c.height} breaks with 13F coverage. "
               + ("PREMISE CLEARS: proceed to registration." if ok
                  else "PREMISE FAILS: retire on paper, trial count unchanged."))
    print(verdict)
    clean = e.filter(pl.col("ann").is_not_null() & ~pl.col("spac")
                     & ~pl.col("first_form").is_in(["PREM14C", "DEFM14C"]))
    reg = _blk(clean, h, "Qa2", "Qp").select("cik", "episode", "coname", "first_form", "PERMNO",
                                             "t1", "day0", "Qa", "Qb", "block_new", "block_up",
                                             "cap_pre")
    reg = reg.rename({"Qa": "Qa_announce", "Qb": "Qb_public"}).sort("day0")
    reg.write_csv(data / "deal_break_registration_set.csv")
    print(f"registration set (clean, point-in-time block): {reg.height} events, "
          f"{reg.filter(pl.col('block_new') >= MIN_MEDIAN_BLOCK).height} high-block -> "
          f"{data / 'deal_break_registration_set.csv'}")
    out = data / "deal_break_blocks.csv"
    b.select("cik", "episode", "coname", "t1", "day0", "Qa", "Qb", "PERMNO", "cap_pre",
             "covered", "inst_a_pct", "inst_b_pct", "block_new", "block_up", "new_holders",
             "managers_b").write_csv(out)
    here = Path(__file__).resolve().parent
    lines = ["# Deal-break unwinds: the premise, sized", "",
             "Run of `size_premise.py` over `build_breaks.py`'s events; aggregates only.", "",
             f"**Verdict: {verdict}**", "", "| statistic | p10 | p25 | p50 | p75 | p90 |",
             "|---|---|---|---|---|---|"]
    for col in ("inst_a_pct", "inst_b_pct", "block_new", "block_up"):
        q = [c[col].quantile(x) for x in (0.1, 0.25, 0.5, 0.75, 0.9)]
        lines.append(f"| {col} | " + " | ".join(f"{v:.1%}" for v in q) + " |")
    lines += ["", "| check | n | p25 | p50 | p75 |", "|---|---|---|---|---|"]
    lines += [f"| {lab.strip()} | {n} | {a:.1%} | {m:.1%} | {z:.1%} |" for lab, n, a, m, z in diag]
    by_year = e.group_by(pl.col("day0").dt.year().alias("year")).len().sort("year")
    lines += ["", "| year | breaks |", "|---|---|"]
    lines += [f"| {r['year']} | {r['len']} |" for r in by_year.iter_rows(named=True)]
    (here / "sizing_result.md").write_text("\n".join(lines) + "\n")
    print(f"per-break table -> {out}; aggregate record -> {here / 'sizing_result.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
