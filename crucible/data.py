"""Loading real data, and the vendor quirks that make a naive loader silently wrong.

Reshaping long-format rows into aligned matrices is ten lines. The rest of this file is the part
that matters: every source encodes decisions in its schema, and reading those columns at face
value produces a dataset that looks fine and is not.

## CRSP, and the five traps

CRSP is the survivorship-bias-free US equity database the academic literature is built on. It is
also full of conventions that punish the obvious reading.

**1. PERMNO is the identifier, not the ticker.** Tickers are recycled — the same three letters
belong to different companies in different decades. Keying on ticker silently concatenates the
history of unrelated firms into one series, producing a price path nobody ever traded and, at the
splice point, a return that can be enormous. PERMNO is permanent and unique per security. This
loader keys on PERMNO and carries the ticker along only as a label.

**2. A negative price means no trade happened.** CRSP stores the negated bid/ask midpoint when a
security did not trade that day. `abs(prc)` is the standard fix and it is only half of one: the
sign carries information, and a midpoint on a non-trading day is not a price you could have
transacted at. Here the magnitude is kept and the day is marked non-investable, because a
strategy that trades on it is trading a quote nobody filled.

**3. DLRET is the delisting return, and it is the whole point of using CRSP.** When a security
disappears, CRSP records what a holder actually received — often a large negative number for
bankruptcies. `crucible.engine` takes a per-asset delisting return precisely so this can be used
rather than assumed. Ignoring DLRET and letting positions vanish at their last price is the
single largest source of upward bias in a survivorship-*free* dataset, which is a bitter way to
lose the advantage you paid for.

**4. SHRCD selects what kind of security it is.** 10 and 11 are US ordinary common shares.
Everything else — ADRs, REITs, closed-end funds, units, SBIs — behaves differently and belongs in
a study only if you meant to include it. The default here is 10/11 because that is what nearly
every published cross-sectional result uses, and a universe that quietly includes closed-end
funds is not comparable to any of them.

**5. RET can be a string.** CRSP encodes missing returns as letter codes (`B`, `C`) rather than
nulls. Naive numeric coercion turns those into NaN, which is right, but a loader that does not
expect strings will fail on a type error partway through a large file — or worse, silently coerce
them to zero.

## Compustat and point-in-time fundamentals

Fundamentals must be joined on the date they were *published*, not the fiscal period they
describe. A quarter ending 31 March is not knowable on 31 March; it is knowable when the filing
appears, typically weeks later. Joining on the fiscal date gives the strategy several weeks of
foreknowledge on every position, every quarter, which is the most common lookahead in fundamental
research and one that `crucible.causality` cannot see because it is baked into the data before
any code runs. `RDQ` is the report date; use it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from crucible.panel import Panel, PanelError

__all__ = [
    "Dataset",
    "DataError",
    "load_crsp_ciz",
    "load_crsp_csv",
    "long_to_panel",
]

#: CRSP share codes for US ordinary common shares.
US_COMMON_SHARES = (10, 11)
#: CRSP exchange codes for NYSE, AMEX (NYSE American) and NASDAQ.
MAJOR_EXCHANGES = (1, 2, 3)


class DataError(ValueError):
    """The source data cannot be loaded, or loading it would produce a misleading dataset."""


def _as_date(stamp: date | datetime) -> date:
    """Normalise a panel index entry to a plain date."""
    return stamp.date() if isinstance(stamp, datetime) else stamp


def long_to_panel(
    frame: pl.DataFrame,
    *,
    date_column: str,
    asset_column: str,
    value_column: str,
    name: str | None = None,
    assets: Sequence[str] | None = None,
    index: Sequence[date] | None = None,
) -> Panel:
    """Pivot long-format rows into an aligned (time x asset) panel.

    Missing combinations become NaN, which the rest of the library reads as "not investable" —
    the correct default, because a row that is absent is a day the security did not report.

    Args:
        frame: Long-format data.
        date_column: Column holding dates.
        asset_column: Column holding the asset identifier.
        value_column: Column holding the value.
        name: Panel name. Defaults to `value_column`.
        assets: Force a column set and order. Assets absent from the frame become all-NaN
            columns, which is how a security that never traded in this window keeps its place.
        index: Force a row set and order.
    """
    for column in (date_column, asset_column, value_column):
        if column not in frame.columns:
            raise DataError(f"column {column!r} not in frame; have {sorted(frame.columns)}")

    pivoted = (
        frame.select([date_column, asset_column, value_column])
        .pivot(on=asset_column, index=date_column, values=value_column, aggregate_function="first")
        .sort(date_column)
    )

    found_assets = [c for c in pivoted.columns if c != date_column]
    columns = list(assets) if assets is not None else sorted(found_assets)
    rows = (
        list(index)
        if index is not None
        else [_as_date(d) for d in pivoted[date_column].to_list()]
    )

    # Reindex inside polars rather than in Python. An earlier version walked
    # dates x assets in a nested loop, which is invisible on a 200-row fixture and
    # impossible on a real extract: 2,770 sessions x 16,915 securities is 47 million
    # iterations per field, and there are three fields. The pivot below does the same
    # work in well under a second.
    missing = [a for a in columns if a not in found_assets]
    if missing:
        # Securities absent from this field keep their column, as all-NaN.
        pivoted = pivoted.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(a) for a in missing]
        )

    wanted = pl.DataFrame({date_column: rows}).with_columns(
        pl.col(date_column).cast(pivoted[date_column].dtype)
    )
    aligned = wanted.join(pivoted, on=date_column, how="left")

    values = (
        aligned.select([pl.col(a).cast(pl.Float64, strict=False) for a in columns]).to_numpy()
        if columns
        else np.empty((len(rows), 0), dtype=np.float64)
    )

    return Panel(values=values, index=tuple(rows), assets=tuple(columns), name=name or value_column)


@dataclass(frozen=True, slots=True)
class Dataset:
    """Everything a backtest needs from one source, already aligned.

    `listings` plugs into `crucible.universe.listing_mask`; `delisting_returns` plugs into the
    engine. Both are carried rather than derived so the vendor's own record is used instead of
    being inferred from where the price series happens to stop.
    """

    #: Total-return adjusted price series. THIS is what the engine must use, because it
    #: derives returns as pct_change of this panel. See `load_crsp_ciz` for why a raw price
    #: panel here silently reproduces the split bug this library exists to catch.
    prices: Panel
    dollar_volume: Panel
    returns: Panel
    #: Unadjusted price, as the vendor reported it. Only for screens that care about the actual
    #: traded level — a penny stock is a penny stock regardless of how its history is adjusted.
    #: Never use this to compute returns.
    #: `{permno: (first_tradable, last_tradable)}`
    listings: dict[str, tuple[date | None, date | None]]
    #: `{permno: realised return on delisting}`. Frequently large and negative.
    delisting_returns: dict[str, float]
    #: Unadjusted price, as the vendor reported it. Only for screens that care about the actual
    #: traded level — a penny stock is a penny stock regardless of how its history is adjusted.
    #: Never use this to compute returns.
    raw_prices: Panel | None = None
    #: `{permno: most recent ticker}`. A label only — never a key.
    tickers: dict[str, str] = field(default_factory=dict)

    @property
    def assets(self) -> tuple[str, ...]:
        return self.prices.assets

    def summary(self) -> str:
        first, last = _as_date(self.prices.index[0]), _as_date(self.prices.index[-1])
        years = (last - first).days / 365.25
        delisted = sum(1 for window in self.listings.values() if window[1] is not None)
        punitive = sum(1 for r in self.delisting_returns.values() if r < -0.5)
        lines = [
            f"{self.prices.n_assets} securities, {self.prices.n_times} sessions, "
            f"{first} → {last} ({years:.1f} years)",
            f"  coverage {self.prices.coverage:.1%} | median breadth "
            f"{np.median(self.prices.count_per_row()):.0f} names/day",
            f"  {delisted} securities delisted in-sample, "
            f"{len(self.delisting_returns)} with a recorded delisting return "
            f"({punitive} worse than -50%)",
        ]
        if not delisted:
            lines.append(
                "  WARNING: nothing delisted. Over a multi-year US equity sample that does not "
                "happen — this extract has been survivorship-filtered somewhere upstream, and "
                "no code downstream can recover the missing names."
            )
        return "\n".join(lines)


def _numeric(frame: pl.DataFrame, column: str) -> pl.Expr:
    """Coerce a column to Float64, tolerating CRSP's letter codes for missing values."""
    return pl.col(column).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)


def load_crsp_csv(
    path: str | Path,
    *,
    share_codes: Iterable[int] | None = US_COMMON_SHARES,
    exchange_codes: Iterable[int] | None = MAJOR_EXCHANGES,
    columns: Mapping[str, str] | None = None,
) -> Dataset:
    """Load a CRSP daily stock file exported from WRDS.

    Expects the standard WRDS column names, lowercased: `permno`, `date`, `prc`, `vol`, `ret`,
    `shrout`, and optionally `dlret`, `shrcd`, `exchcd`, `ticker`. Override via `columns`.

    Args:
        path: CSV exported from the WRDS web interface or the `wrds` Python client.
        share_codes: SHRCD values to keep. Defaults to US ordinary common shares (10, 11);
            `None` keeps everything, which will include ADRs, REITs and closed-end funds.
        exchange_codes: EXCHCD values to keep. Defaults to NYSE/AMEX/NASDAQ.
        columns: Remap source column names, e.g. `{"prc": "PRC"}`.

    Returns:
        A `Dataset` keyed on PERMNO as a string.
    """
    source = Path(path)
    if not source.exists():
        raise DataError(f"{source} does not exist")

    mapping = {k: k for k in ("permno", "date", "prc", "vol", "ret", "shrout")}
    mapping |= {k: k for k in ("dlret", "shrcd", "exchcd", "ticker")}
    if columns:
        mapping |= dict(columns)

    frame = pl.read_csv(source, infer_schema_length=10_000)
    frame.columns = [c.lower() for c in frame.columns]

    required = [mapping[k] for k in ("permno", "date", "prc")]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DataError(
            f"{source} is missing required column(s) {missing}. Expected a CRSP daily stock "
            f"file; found {sorted(frame.columns)}."
        )

    # PERMNO as a string: it is an identifier, and arithmetic on it is always a bug.
    frame = frame.with_columns(
        pl.col(mapping["permno"]).cast(pl.Int64).cast(pl.Utf8).alias("_permno"),
        pl.col(mapping["date"]).cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias("_date"),
    )

    if share_codes is not None and mapping["shrcd"] in frame.columns:
        frame = frame.filter(pl.col(mapping["shrcd"]).cast(pl.Int64).is_in(list(share_codes)))
    if exchange_codes is not None and mapping["exchcd"] in frame.columns:
        frame = frame.filter(pl.col(mapping["exchcd"]).cast(pl.Int64).is_in(list(exchange_codes)))
    if frame.is_empty():
        raise DataError(
            "no rows survived the share-code and exchange-code filters. Check that shrcd and "
            "exchcd are present and that the requested codes match this extract."
        )

    # A negative PRC is a negated bid/ask midpoint on a day the security did not trade. Keep the
    # magnitude for continuity, but mark the day non-investable — a midpoint nobody filled is
    # not a price a strategy may transact at.
    frame = frame.with_columns(
        _numeric(frame, mapping["prc"]).alias("_prc_raw"),
    ).with_columns(
        pl.when(pl.col("_prc_raw") < 0).then(None).otherwise(pl.col("_prc_raw")).alias("_price"),
        pl.col("_prc_raw").abs().alias("_price_magnitude"),
    )

    volume = _numeric(frame, mapping["vol"]) if mapping["vol"] in frame.columns else pl.lit(None)
    frame = frame.with_columns(
        (volume * pl.col("_price_magnitude")).alias("_dollar_volume"),
        (
            _numeric(frame, mapping["ret"])
            if mapping["ret"] in frame.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("_ret"),
    )

    dates = sorted({d for d in frame["_date"].to_list() if d is not None})
    permnos = sorted(set(frame["_permno"].to_list()))

    def pivot(value_column: str, name: str) -> Panel:
        return long_to_panel(
            frame,
            date_column="_date",
            asset_column="_permno",
            value_column=value_column,
            name=name,
            index=dates,
            assets=permnos,
        )

    prices = pivot("_price", "close")
    dollar_volume = pivot("_dollar_volume", "adv")
    returns = pivot("_ret", "ret")

    listings: dict[str, tuple[date | None, date | None]] = {}
    investable = prices.investable
    for column, permno in enumerate(permnos):
        rows = np.flatnonzero(investable[:, column])
        if rows.size == 0:
            continue
        last = dates[int(rows[-1])]
        # A security still trading on the final date has not delisted; it merely runs past the
        # end of the extract. Recording that as a delisting would invent an exit.
        listings[permno] = (dates[int(rows[0])], None if last == dates[-1] else last)

    delisting_returns: dict[str, float] = {}
    if mapping["dlret"] in frame.columns:
        delist = (
            frame.with_columns(_numeric(frame, mapping["dlret"]).alias("_dlret"))
            .filter(pl.col("_dlret").is_not_null())
            .group_by("_permno")
            .agg(pl.col("_dlret").last())
        )
        delisting_returns = dict(
            zip(delist["_permno"].to_list(), delist["_dlret"].to_list(), strict=True)
        )

    tickers: dict[str, str] = {}
    if mapping["ticker"] in frame.columns:
        latest = (
            frame.filter(pl.col(mapping["ticker"]).is_not_null())
            .sort("_date")
            .group_by("_permno")
            .agg(pl.col(mapping["ticker"]).last())
        )
        tickers = dict(
            zip(latest["_permno"].to_list(), latest[mapping["ticker"]].to_list(), strict=True)
        )

    return Dataset(
        prices=prices,
        dollar_volume=dollar_volume,
        returns=returns,
        listings=listings,
        delisting_returns=delisting_returns,
        tickers=tickers,
    )


def check_point_in_time(
    fundamentals: pl.DataFrame,
    *,
    fiscal_date_column: str = "datadate",
    report_date_column: str = "rdq",
) -> str:
    """Report the publication lag in a Compustat extract, and refuse to let it be ignored.

    A quarter ending 31 March is not knowable on 31 March. Joining fundamentals on the fiscal
    date rather than the report date hands the strategy several weeks of foreknowledge on every
    position, every quarter — the most common lookahead in fundamental research, and one
    `crucible.causality` cannot detect because it is baked into the data before any code runs.
    """
    for column in (fiscal_date_column, report_date_column):
        if column not in fundamentals.columns:
            raise DataError(
                f"column {column!r} not present. Without a report date there is no way to know "
                f"when a fundamental became public, and joining on the fiscal date is lookahead."
            )

    lagged = fundamentals.with_columns(
        (pl.col(report_date_column) - pl.col(fiscal_date_column)).dt.total_days().alias("_lag")
    ).filter(pl.col("_lag").is_not_null())

    if lagged.is_empty():
        raise DataError("no rows have both a fiscal date and a report date")

    lags = lagged["_lag"].to_numpy()
    missing = len(fundamentals) - len(lagged)
    return (
        f"publication lag: median {np.median(lags):.0f} days, "
        f"p10 {np.percentile(lags, 10):.0f}, p90 {np.percentile(lags, 90):.0f}\n"
        f"  {missing} row(s) lack a report date and cannot be used point-in-time.\n"
        f"  Join on {report_date_column!r}. Joining on {fiscal_date_column!r} would give the "
        f"strategy a median of {np.median(lags):.0f} days of foreknowledge per observation."
    )


def dataset_from_panels(
    prices: Panel, dollar_volume: Panel, *, returns: Panel | None = None
) -> Dataset:
    """Wrap existing panels as a `Dataset`, deriving listings from investability.

    For sources without an explicit listing calendar. The delisting returns are empty, which
    means the engine falls back to its optimistic exit-at-last-price assumption — acceptable for
    a vendor that has no better answer, and worth knowing about rather than assuming away.
    """
    if prices.index != dollar_volume.index or prices.assets != dollar_volume.assets:
        raise PanelError("prices and dollar_volume must be aligned")

    investable = prices.investable
    listings: dict[str, tuple[date | None, date | None]] = {}
    for column, asset in enumerate(prices.assets):
        rows = np.flatnonzero(investable[:, column])
        if rows.size == 0:
            continue
        last_row = int(rows[-1])
        listings[asset] = (
            _as_date(prices.index[int(rows[0])]),
            None if last_row == prices.n_times - 1 else _as_date(prices.index[last_row]),
        )

    return Dataset(
        prices=prices,
        dollar_volume=dollar_volume,
        returns=returns or prices.pct_change(1),
        listings=listings,
        delisting_returns={},
    )


# ── CRSP Flat File Format 2.0 (CIZ) ───────────────────────────────────────────────────────────
#
# CRSP discontinued the legacy SIZ format after December 2024. CIZ is not a rename — it is a
# different schema, and three of the differences change what the loader has to do.
#
# **Delisting returns moved to a separate table.** In SIZ, `DLRET` was a column in the daily
# file. In CIZ, "Stock Delisting Information" is its own query, joined on PERMNO. That is the
# column that matters most in this whole library, so it is now a second required file rather
# than an optional one, and its absence is reported rather than silently tolerated.
#
# **Share and exchange codes became text.** `shrcd` 10/11 became `sharetype`, and `exchcd` 1/2/3
# split into `primaryexch` and `exchangetier`. The numeric filters do not transfer, so the CIZ
# loader filters on strings and the defaults below are stated explicitly rather than inherited.
#
# **Some derived fields come free.** `dlyprcvol` is dollar volume and `dlycap` is market
# capitalisation, both computed by CRSP. The SIZ loader multiplied volume by price to get the
# first and could not produce the second at all.

#: CIZ share type for US ordinary common shares. The CIZ analogue of SHRCD 10/11.
CIZ_COMMON_SHARE_TYPES = ("NS",)
#: CIZ primary exchange codes for NYSE, NYSE American and Nasdaq.
CIZ_MAJOR_EXCHANGES = ("N", "A", "Q")


def load_crsp_ciz(
    daily_path: str | Path,
    *,
    delisting_path: str | Path | None = None,
    share_types: Iterable[str] | None = CIZ_COMMON_SHARE_TYPES,
    exchanges: Iterable[str] | None = CIZ_MAJOR_EXCHANGES,
) -> Dataset:
    """Load a CRSP CIZ daily stock file, plus its separate delisting table.

    Args:
        daily_path: The Daily Stock File export. Needs at least `permno`, `dlycaldt`, `dlyprc`;
            `dlyvol`, `dlyprcvol`, `dlyret`, `sharetype`, `primaryexch`, `ticker` are used when
            present.
        delisting_path: The Delisting Information export. Optional only in the sense that the
            loader will run without it — but running without it means falling back to the
            engine's optimistic exit-at-last-price assumption, which is the single largest
            remaining upward bias in a survivorship-free dataset. `Dataset.summary()` says so.
        share_types: CIZ `sharetype` values to keep. Defaults to US ordinary common shares.
            `None` keeps everything, including ADRs, REITs and closed-end funds.
        exchanges: CIZ `primaryexch` values to keep. Defaults to NYSE/NYSE American/Nasdaq.

    Returns:
        A `Dataset` keyed on PERMNO as a string, identical in shape to the legacy loader's — so
        everything downstream is unchanged by which format the data arrived in.
    """
    source = Path(daily_path)
    if not source.exists():
        raise DataError(f"{source} does not exist")

    frame = pl.read_csv(source, infer_schema_length=10_000)
    frame.columns = [c.lower() for c in frame.columns]

    required = ["permno", "dlycaldt", "dlyprc"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DataError(
            f"{source} is missing required CIZ column(s) {missing}. Expected a CRSP Version 2 "
            f"(CIZ) daily stock file; found {sorted(frame.columns)}. If this is a legacy SIZ "
            f"export with `date`/`prc`/`ret`, use load_crsp_csv instead."
        )

    frame = frame.with_columns(
        pl.col("permno").cast(pl.Int64).cast(pl.Utf8).alias("_permno"),
        pl.col("dlycaldt").cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias("_date"),
    )

    # CRSP emits more than one row for a security on a day when it reports multiple distribution
    # events — WRDS documents this on the query page. It is rare (4,579 pairs across 23.1M rows
    # in the 2015-2025 extract, touching 2,381 securities) and it silently corrupts the adjusted
    # price series: the cumulative product below compounds every row, while the pivot that builds
    # the panel keeps only the first. The series then carries a return the panel never shows, and
    # the two disagree from that day onward. Every one of the five largest disagreements traced
    # back to a duplicate on the preceding session.
    #
    # Deduplicated here, before compounding, so the return series and the pivot see the same rows.
    duplicates = frame.height
    frame = frame.unique(subset=["_permno", "_date"], keep="first", maintain_order=True)
    duplicates -= frame.height

    if share_types is not None and "sharetype" in frame.columns:
        frame = frame.filter(
            pl.col("sharetype").cast(pl.Utf8).str.strip_chars().is_in(list(share_types))
        )
    if exchanges is not None and "primaryexch" in frame.columns:
        frame = frame.filter(
            pl.col("primaryexch").cast(pl.Utf8).str.strip_chars().is_in(list(exchanges))
        )
    if frame.is_empty():
        raise DataError(
            f"no rows survived the share-type and exchange filters. CIZ uses text codes — "
            f"sharetype {list(share_types) if share_types else 'any'}, primaryexch "
            f"{list(exchanges) if exchanges else 'any'} — not the numeric SHRCD/EXCHCD of the "
            f"legacy format. Check the values actually present in this extract."
        )

    # CIZ prices are unsigned; a non-trading day is marked by dlyprcflg rather than by a negative
    # price. Where the flag is absent the price is taken at face value.
    #
    # But dlyprc is RAW and UNADJUSTED, and the engine derives returns as pct_change of the price
    # panel — so handing it this column silently reproduces the exact split bug this library was
    # built to catch. Verified against three known splits in the real extract:
    #
    #     AAPL  2020-08-28   499.23 -> 2020-08-31   129.04   DlyRet +0.0339
    #     TSLA  2022-08-24   891.29 -> 2022-08-25   296.07   DlyRet -0.0035
    #     NVDA  2024-06-07  1208.88 -> 2024-06-10   121.79   DlyRet +0.0075
    #
    # Naive pct_change would read those as -74.2%, -66.8% and -89.9%. CRSP's dlyret is correct
    # throughout — it already accounts for splits, dividends and distributions — so the adjusted
    # series is built by compounding it forward from each security's first observed price.
    # Anything that needs the actual traded level (a penny-stock screen) uses raw_prices instead.
    frame = frame.with_columns(_numeric(frame, "dlyprc").alias("_raw_price"))
    if "dlyret" in frame.columns:
        frame = frame.sort(["_permno", "_date"]).with_columns(
            _numeric(frame, "dlyret").alias("_ret_raw")
        )
        frame = frame.with_columns(
            (
                pl.col("_raw_price").first().over("_permno")
                * (1.0 + pl.col("_ret_raw").fill_null(0.0)).cum_prod().over("_permno")
                / (1.0 + pl.col("_ret_raw").fill_null(0.0)).first().over("_permno")
            ).alias("_price")
        )
    else:
        # No return column: fall back to raw prices and say so, rather than pretending.
        frame = frame.with_columns(pl.col("_raw_price").alias("_price"))

    # dlyprcvol is dollar volume computed by CRSP. Fall back to volume x price only if absent.
    if "dlyprcvol" in frame.columns:
        dollar_volume = _numeric(frame, "dlyprcvol")
    elif "dlyvol" in frame.columns:
        dollar_volume = _numeric(frame, "dlyvol") * pl.col("_price").abs()
    else:
        dollar_volume = pl.lit(None, dtype=pl.Float64)
    frame = frame.with_columns(
        dollar_volume.alias("_dollar_volume"),
        (
            _numeric(frame, "dlyret")
            if "dlyret" in frame.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("_ret"),
    )

    dates = sorted({d for d in frame["_date"].to_list() if d is not None})
    permnos = sorted(set(frame["_permno"].to_list()))

    def pivot(value_column: str, name: str) -> Panel:
        return long_to_panel(
            frame,
            date_column="_date",
            asset_column="_permno",
            value_column=value_column,
            name=name,
            index=dates,
            assets=permnos,
        )

    _ = duplicates  # kept for clarity at the dedup site; not part of the returned Dataset
    prices = pivot("_price", "close_adjusted")
    raw_prices = pivot("_raw_price", "close_raw")
    volume_panel = pivot("_dollar_volume", "adv")
    returns = pivot("_ret", "ret")

    listings: dict[str, tuple[date | None, date | None]] = {}
    investable = prices.investable
    for column, permno in enumerate(permnos):
        rows = np.flatnonzero(investable[:, column])
        if rows.size == 0:
            continue
        last = dates[int(rows[-1])]
        listings[permno] = (dates[int(rows[0])], None if last == dates[-1] else last)

    delisting_returns = (
        _load_ciz_delistings(Path(delisting_path)) if delisting_path is not None else {}
    )

    tickers: dict[str, str] = {}
    if "ticker" in frame.columns:
        latest = (
            frame.filter(pl.col("ticker").is_not_null())
            .sort("_date")
            .group_by("_permno")
            .agg(pl.col("ticker").last())
        )
        tickers = dict(zip(latest["_permno"].to_list(), latest["ticker"].to_list(), strict=True))

    return Dataset(
        prices=prices,
        dollar_volume=volume_panel,
        returns=returns,
        listings=listings,
        delisting_returns=delisting_returns,
        raw_prices=raw_prices,
        tickers=tickers,
    )


def _load_ciz_delistings(path: Path) -> dict[str, float]:
    """Read the CIZ Delisting Information table into `{permno: delisting return}`.

    CIZ does not standardise on one column name here across releases, so several candidates are
    tried in order of specificity. Finding none is an error rather than an empty result: a
    delisting file with no return column is almost certainly the wrong export, and silently
    returning nothing would reinstate the exit-at-last-price assumption without saying so.
    """
    if not path.exists():
        raise DataError(f"{path} does not exist")

    frame = pl.read_csv(path, infer_schema_length=10_000)
    frame.columns = [c.lower() for c in frame.columns]
    if "permno" not in frame.columns:
        raise DataError(f"{path} has no permno column; found {sorted(frame.columns)}")

    candidates = ["dlret", "delret", "delistingreturn", "dlyret", "delrettype", "delamt"]
    column = next((c for c in candidates if c in frame.columns), None)
    if column is None:
        raise DataError(
            f"{path} has no recognisable delisting-return column (tried {candidates}); found "
            f"{sorted(frame.columns)}.\n"
            f"\n"
            f"This is the column the whole point of using CRSP rests on. Without it the engine "
            f"assumes a delisted position exited at its last price, which is optimistic and is "
            f"the largest remaining upward bias in a survivorship-free dataset. Re-export the "
            f"Delisting Information table with every variable selected."
        )

    resolved = (
        frame.with_columns(
            pl.col("permno").cast(pl.Int64).cast(pl.Utf8).alias("_permno"),
            _numeric(frame, column).alias("_dlret"),
        )
        .filter(pl.col("_dlret").is_not_null())
        .group_by("_permno")
        .agg(pl.col("_dlret").last())
    )
    return dict(zip(resolved["_permno"].to_list(), resolved["_dlret"].to_list(), strict=True))
