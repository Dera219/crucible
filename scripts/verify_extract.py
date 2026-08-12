"""Check the data before trusting anything computed from it.

    PYTHONPATH=$PWD .venv/bin/python scripts/verify_extract.py crsp_daily.csv

This is **not** a strategy test, and running one now would be a mistake. Without a registered
hypothesis there is nothing to test against, and every backtest run before the criteria are fixed
is a trial that raises the deflation bar for whatever you eventually claim. The trial budget is a
debt; do not spend it on curiosity.

What this does instead is ask whether the extract is what it says it is. That question has to be
settled first, because every defect below is one that produces plausible numbers rather than an
error — and the previous vendor shipped two of them.

## What it checks and why each one has burned somebody

**Survivorship.** If nothing delists across a multi-year US equity sample, the extract has been
filtered upstream and no code downstream can recover the missing names. This is the single check
worth stopping for.

**Delisting severity.** CRSP's `DLRET` records what a holder actually received. If most delistings
are near zero the extract may carry only the benign ones; real bankruptcies are large and negative,
and their absence quietly inflates every strategy that touches distressed names.

**Breadth over time.** A cross-sectional statistic computed across eleven names is noise wearing a
number. If breadth collapses in the early years, the folds there cannot support a decile spread
however good the full-sample average looks.

**Unexplained jumps.** A single-day move beyond roughly ±50% that is *not* a recorded delisting is
usually an unadjusted split or a bad print. Alpaca shipped exactly this: `adjustment=all` applied
dividends but not splits, so AAPL appeared to fall 74.1% in one session.

**Non-trading days.** A negative `PRC` is a negated bid/ask midpoint, meaning the security did not
trade. The loader already marks those non-investable; this reports how common they are, because a
universe where a tenth of observations never traded is not the liquid universe you think it is.

**Duplicate identifiers.** One PERMNO appearing twice on a date means the extract was assembled
from overlapping pulls, and every cross-sectional statistic on those rows is double-counting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from crucible.data import DataError, load_crsp_csv
from crucible.universe import Universe, liquidity_screen, listing_mask


def rule(title: str) -> None:
    print(f"\n{title}\n{'─' * len(title)}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().split("\n\n")[1])
        return 2

    path = Path(argv[1])
    print(f"reading {path} …")
    try:
        data = load_crsp_csv(path)
    except DataError as exc:
        print(f"\nrefused: {exc}")
        return 1

    findings: list[str] = []

    rule("1. What arrived")
    print(f"  {data.summary()}")

    rule("2. Survivorship — the one worth stopping for")
    delisted = sum(1 for window in data.listings.values() if window[1] is not None)
    span_years = (data.prices.index[-1] - data.prices.index[0]).days / 365.25  # type: ignore[operator]
    rate = delisted / max(1, data.prices.n_assets) / max(0.1, span_years)
    print(f"  {delisted} of {data.prices.n_assets} securities delisted over {span_years:.1f} years")
    print(f"  attrition {rate:.1%}/year")
    if delisted == 0:
        findings.append(
            "FATAL: nothing delisted. The extract is survivorship-filtered upstream and no "
            "analysis on it means anything. Re-pull without a 'currently active' filter."
        )
    elif rate < 0.02:
        findings.append(
            f"attrition of {rate:.1%}/year is low for US equities, where 4-8% is typical. "
            f"Some exits may be missing."
        )

    rule("3. Delisting severity — are the bankruptcies present?")
    if not data.delisting_returns:
        findings.append(
            "no DLRET values at all. Without them the engine falls back to assuming you exited "
            "at the last price, which is optimistic and is the largest remaining upward bias."
        )
        print("  none recorded")
    else:
        values = np.array(list(data.delisting_returns.values()))
        catastrophic = int((values < -0.5).sum())
        print(f"  {len(values)} recorded | median {np.median(values):+.2%}")
        print(f"  {catastrophic} worse than -50% ({catastrophic / len(values):.0%})")
        print(f"  worst {values.min():+.2%} | best {values.max():+.2%}")
        if catastrophic == 0:
            findings.append(
                "no delisting return worse than -50%. Real bankruptcies are large and negative; "
                "their absence inflates any strategy that touches distressed names."
            )

    rule("4. Cross-sectional breadth over time")
    counts = data.prices.count_per_row()
    years = [str(d.year) for d in data.prices.index]  # type: ignore[union-attr]
    print(f"  {'year':<8}{'median names':>14}{'min':>8}")
    thin_years = []
    for year in sorted(set(years)):
        rows = [i for i, y in enumerate(years) if y == year]
        median, low = int(np.median(counts[rows])), int(counts[rows].min())
        flag = "  <- thin" if median < 30 else ""
        if median < 30:
            thin_years.append(year)
        print(f"  {year:<8}{median:>14}{low:>8}{flag}")
    if thin_years:
        findings.append(
            f"breadth below 30 names in {', '.join(thin_years)}. Decile spreads computed there "
            f"are noise; consider starting the sample later."
        )

    rule("5. Unexplained single-day moves")
    returns = data.prices.pct_change(1).values
    with np.errstate(invalid="ignore"):
        extreme = np.abs(returns) > 0.5
    n_extreme = int(np.nansum(extreme))
    print(f"  {n_extreme} moves beyond ±50%")
    if n_extreme:
        rows, cols = np.where(extreme)
        unexplained = 0
        for row, col in zip(rows[:400], cols[:400], strict=True):
            asset = data.assets[col]
            window = data.listings.get(asset, (None, None))[1]
            near_exit = window is not None and abs((data.prices.index[row] - window).days) <= 5  # type: ignore[operator]
            if not near_exit:
                unexplained += 1
        print(f"  {unexplained} of the first {min(400, n_extreme)} are NOT near a recorded exit")
        if unexplained > n_extreme * 0.25:
            findings.append(
                f"{unexplained} large moves are not explained by a delisting. The usual cause is "
                f"an unadjusted split — the exact defect that made AAPL appear to fall 74.1% in "
                f"one session on the previous vendor. Check a few by hand before trusting this."
            )

    rule("6. Non-trading days and data holes")
    coverage = data.prices.coverage
    print(f"  overall coverage {coverage:.1%} of (date × security) cells investable")
    if coverage < 0.5:
        findings.append(
            f"coverage of {coverage:.1%} is low. Much of the panel is securities outside their "
            f"listed window, which is expected — but confirm it is not missing sessions."
        )

    rule("7. A usable universe, built from this extract")
    universe = Universe.from_masks(
        listing_mask(data.prices.index, data.assets, data.listings),
        liquidity_screen(data.dollar_volume, data.prices, top_n=1000),
        index=data.prices.index,
        assets=data.assets,
        name="us-liquid-1000",
    )
    print(f"  {universe.audit().summary()}")

    rule("Verdict")
    if not findings:
        print("  No defects found. The extract looks like what it claims to be.")
        print("  Next step is a registered hypothesis — NOT a signal search.")
        return 0

    for i, finding in enumerate(findings, 1):
        marker = "FATAL" if finding.startswith("FATAL") else "check"
        print(f"  [{marker}] {finding.removeprefix('FATAL: ')}")
        if i < len(findings):
            print()
    print()
    print("  Every item above produces plausible numbers rather than an error, which is why")
    print("  they are worth settling before anything is computed on this data.")
    return 1 if any(f.startswith("FATAL") for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
