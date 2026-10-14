"""Build the spin-off orphan event set. Outcome-blind: dates, membership and classification only.

    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/spinoff-orphans/build_events.py [DATA_DIR]

DATA_DIR defaults to ~/Downloads and must hold the CRSP extracts named below. The output,
`spinoff_orphan_events.csv`, is written there too: it is derived from licensed CRSP data and is
never committed anywhere. Its SHA-256 is recorded in registration.json instead.

## The rule, fixed 2026-10-14 before any outcome existed

An event is a distribution with DisType 'SP' and DisDetailType 'SECSO' (a spin-off) such that:

1. the PARENT is an S&P 500 member on the ex-date;
2. the SPINCO is a new listing whose first daily observation falls within 5 days of the ex-date
   (crucible's 2026-08-26 identification rule). That observation is its first regular-way trade,
   and it is the event date, offset 0;
3. the spinco is US common stock on that day by crucible's own CIZ classification
   (CIZ_COMMON_SHARE_TYPES, CIZ_MAJOR_EXCHANGES, CIZ_COMMON_STOCK_FILTER), so REITs, ADRs and
   foreign-incorporated issuers are out, exactly as they are out of every crucible panel;
4. the spinco is NOT an S&P 500 member on the session after its first trade, i.e. it was never
   added, or held for one day under S&P DJI's post-2015 rule and then removed.

No price, volume or return column is ever read.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from crucible.data import CIZ_COMMON_SHARE_TYPES, CIZ_COMMON_STOCK_FILTER, CIZ_MAJOR_EXCHANGES

SLICES = ((1990, 1994), (1995, 1999), (2000, 2004), (2005, 2009), (2010, 2014), (2015, 2025))
FIRST_TRADE_TOLERANCE_DAYS = 5
POST_2015_RULE = date(2015, 10, 1)
CLASSIFICATION = {
    "securitytype": "SecurityType",
    "securitysubtype": "SecuritySubType",
    "issuertype": "IssuerType",
    "usincflg": "USIncFlg",
}


def membership(data: Path) -> pl.DataFrame:
    return (
        pl.read_csv(
            data / "crsp_sp500_constituents_1985_2025.csv.gz",
            columns=["MbrStartDt", "MbrEndDt", "PERMNO"],
            schema_overrides={"MbrStartDt": pl.Date, "MbrEndDt": pl.Date},
        )
        .unique()
        .with_columns(pl.col("PERMNO").cast(pl.Int64))
    )


def spin_offs_of_members(data: Path, spans: pl.DataFrame) -> pl.DataFrame:
    dist = pl.read_csv(
        data / "crsp_distributions_1985_2025.csv.gz",
        columns=["PERMNO", "DisExDt", "DisType", "DisDetailType", "DisPERMNO"],
        infer_schema_length=0,
    )
    events = (
        dist.filter((pl.col("DisType") == "SP") & (pl.col("DisDetailType") == "SECSO"))
        .select(
            parent=pl.col("PERMNO").cast(pl.Int64),
            ex=pl.col("DisExDt").str.to_date(),
            spinco=pl.col("DisPERMNO").cast(pl.Int64),
        )
        .filter(pl.col("spinco") > 0)
        .unique()
    )
    return (
        events.join(spans.rename({"PERMNO": "parent"}), on="parent")
        .filter((pl.col("MbrStartDt") <= pl.col("ex")) & (pl.col("ex") <= pl.col("MbrEndDt")))
        .select("parent", "ex", "spinco")
        .unique()
        .sort(["ex", "spinco"])
    )


def first_rows(data: Path, spincos: list[int]) -> pl.DataFrame:
    """Every daily row of the spincos: date plus classification, never a price or return."""
    columns = ["PERMNO", "DlyCalDt", "ShareType", "PrimaryExch", *CLASSIFICATION.values()]
    frames = []
    for start, end in SLICES:
        frame = pl.read_csv(
            data / f"crsp_{start}_{end}.csv.gz", columns=columns, infer_schema_length=0
        )
        frames.append(
            frame.with_columns(
                pl.col("PERMNO").cast(pl.Int64), pl.col("DlyCalDt").str.to_date()
            ).filter(pl.col("PERMNO").is_in(spincos))
        )
    return pl.concat(frames).unique(subset=["PERMNO", "DlyCalDt"]).sort(["PERMNO", "DlyCalDt"])


def is_common(row: dict[str, object]) -> bool:
    if row["ShareType"] not in CIZ_COMMON_SHARE_TYPES:
        return False
    if row["PrimaryExch"] not in CIZ_MAJOR_EXCHANGES:
        return False
    return all(
        row[CLASSIFICATION[key]] in allowed for key, allowed in CIZ_COMMON_STOCK_FILTER.items()
    )


def is_member(spans: pl.DataFrame, permno: int, on: date) -> bool:
    hits = spans.filter(
        (pl.col("PERMNO") == permno) & (pl.col("MbrStartDt") <= on) & (pl.col("MbrEndDt") >= on)
    )
    return hits.height > 0


def build(data: Path) -> pl.DataFrame:
    spans = membership(data)
    candidates = spin_offs_of_members(data, spans)
    rows = first_rows(data, candidates["spinco"].unique().to_list())
    kept: list[dict[str, object]] = []
    for parent, ex, spinco in candidates.iter_rows():
        own = rows.filter(pl.col("PERMNO") == spinco)
        if own.is_empty():
            continue
        tolerance = timedelta(days=FIRST_TRADE_TOLERANCE_DAYS)
        first = own.row(0, named=True)
        if abs(first["DlyCalDt"] - ex) > tolerance:  # pre-existing, or first traded too late
            continue
        if not is_common(first):
            continue
        if own.height < 2:
            continue
        day0, day1 = first["DlyCalDt"], own.row(1, named=True)["DlyCalDt"]
        if is_member(spans, spinco, day1):
            continue
        kept.append(
            {
                "parent": parent,
                "ex": ex,
                "spinco": spinco,
                "day0": day0,
                "day1": day1,
                "one_day_member": is_member(spans, spinco, day0),
                "regime": "post-2015" if ex >= POST_2015_RULE else "pre-2015",
            }
        )
    return pl.DataFrame(kept).sort(["day0", "spinco"])


def main() -> int:
    data = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "Downloads"
    events = build(data)
    out = data / "spinoff_orphan_events.csv"
    events.write_csv(out)
    print(f"{events.height} events → {out}")
    print(events.group_by(["regime", "one_day_member"]).len().sort(["regime", "one_day_member"]))
    shared = events.group_by("day0").len().filter(pl.col("len") > 1)["len"].sum()
    print(f"share of events sharing a day-0 date: {shared / events.height:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
