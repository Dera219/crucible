"""Build the spin-off holder-unloading event set. Outcome-blind: dates and classification only.

    cd ~/Grind/crucible && PYTHONPATH=$PWD .venv/bin/python \\
        ~/Grind/apex-live/registrations/spinoff-holder-unloading/build_events.py [DATA_DIR]

DATA_DIR defaults to ~/Downloads and must hold the CRSP extracts named below. The output,
`spinoff_holder_events.csv`, is written there too: it is derived from licensed CRSP data and is
never committed anywhere. Its SHA-256 is recorded in registration.json instead.

## The rule, fixed 2026-10-15 before any outcome existed

An event is a distribution with DisType 'SP' and DisDetailType 'SECSO' (a spin-off) such that:

1. the SPINCO is a new listing whose first daily observation falls within 5 days of the ex-date
   (crucible's 2026-08-26 identification rule, the one trial 6 used). That observation is its
   first regular-way trade, and it is the event date, offset 0;
2. the spinco is US common stock on that day by crucible's own CIZ classification
   (CIZ_COMMON_SHARE_TYPES, CIZ_MAJOR_EXCHANGES, CIZ_COMMON_STOCK_FILTER), so REITs, ADRs and
   foreign-incorporated issuers are out, exactly as they are out of every crucible panel;
3. offset 0 is on or after 2005-01-01: the regime the claim is registered on.

Every spin-off qualifies, whatever the parent's index membership: the sellers this claim is about
are the parent's holders of every kind, not index funds at one close. A spinco distributed by
two parents counts once, on its first trade.

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
ERA_START = date(2005, 1, 1)
CLASSIFICATION = {
    "securitytype": "SecurityType",
    "securitysubtype": "SecuritySubType",
    "issuertype": "IssuerType",
    "usincflg": "USIncFlg",
}


def spin_offs(data: Path) -> pl.DataFrame:
    dist = pl.read_csv(
        data / "crsp_distributions_1985_2025.csv.gz",
        columns=["PERMNO", "DisExDt", "DisType", "DisDetailType", "DisPERMNO"],
        infer_schema_length=0,
    )
    return (
        dist.filter((pl.col("DisType") == "SP") & (pl.col("DisDetailType") == "SECSO"))
        .select(
            parent=pl.col("PERMNO").cast(pl.Int64),
            ex=pl.col("DisExDt").str.to_date(),
            spinco=pl.col("DisPERMNO").cast(pl.Int64),
        )
        .filter((pl.col("spinco") > 0) & (pl.col("ex") >= ERA_START - timedelta(days=30)))
        .unique()
        .sort(["ex", "spinco"])
    )


def first_rows(data: Path, spincos: list[int]) -> pl.DataFrame:
    """Every daily row of the spincos: date plus classification, never a price or return."""
    columns = ["PERMNO", "DlyCalDt", "ShareType", "PrimaryExch", *CLASSIFICATION.values()]
    frames = []
    for start, end in SLICES:
        if end < ERA_START.year - 1:
            continue
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


def build(data: Path) -> pl.DataFrame:
    candidates = spin_offs(data)
    rows = first_rows(data, candidates["spinco"].unique().to_list())
    kept: dict[int, dict[str, object]] = {}
    for parent, ex, spinco in candidates.iter_rows():
        if spinco in kept:
            continue
        own = rows.filter(pl.col("PERMNO") == spinco)
        if own.is_empty():
            continue
        first = own.row(0, named=True)
        if abs(first["DlyCalDt"] - ex) > timedelta(days=FIRST_TRADE_TOLERANCE_DAYS):
            continue  # pre-existing, or first traded too late to be this distribution
        if not is_common(first) or first["DlyCalDt"] < ERA_START:
            continue
        kept[spinco] = {
            "parent": parent,
            "ex": ex,
            "spinco": spinco,
            "day0": first["DlyCalDt"],
            "sessions_in_data": own.height,
        }
    return pl.DataFrame(list(kept.values())).sort(["day0", "spinco"])


def main() -> int:
    data = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "Downloads"
    events = build(data)
    out = data / "spinoff_holder_events.csv"
    events.write_csv(out)
    print(f"{events.height} events → {out}")
    print(events.with_columns(pl.col("day0").dt.year().alias("year")).group_by("year").len()
          .sort("year"))
    shared = events.group_by("day0").len().filter(pl.col("len") > 1)["len"].sum()
    print(f"share of events sharing a day-0 date: {shared / events.height:.1%}")
    print(f"events with at least 151 sessions in the data: "
          f"{events.filter(pl.col('sessions_in_data') >= 151).height}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
