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

__all__ = ["Dataset", "DataError", "load_crsp_csv", "long_to_panel"]

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
        else [d.date() if isinstance(d, datetime) else d for d in pivoted[date_column].to_list()]
    )

    lookup = {
        (d.date() if isinstance(d, datetime) else d): i
        for i, d in enumerate(pivoted[date_column].to_list())
    }
    values = np.full((len(rows), len(columns)), np.nan)
    for column_index, asset in enumerate(columns):
        if asset not in found_assets:
            continue
        series = pivoted[asset].to_numpy()
        for row_index, when in enumerate(rows):
            source = lookup.get(when)
            if source is not None:
                values[row_index, column_index] = series[source]

    return Panel(values=values, index=tuple(rows), assets=tuple(columns), name=name or value_column)


@dataclass(frozen=True, slots=True)
class Dataset:
    """Everything a backtest needs from one source, already aligned.

    `listings` plugs into `crucible.universe.listing_mask`; `delisting_returns` plugs into the
    engine. Both are carried rather than derived so the vendor's own record is used instead of
    being inferred from where the price series happens to stop.
    """

    prices: Panel
    dollar_volume: Panel
    returns: Panel
    #: `{permno: (first_tradable, last_tradable)}`
    listings: dict[str, tuple[date | None, date | None]]
    #: `{permno: realised return on delisting}`. Frequently large and negative.
    delisting_returns: dict[str, float]
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
