# Trial nine candidate: Kalshi daily highs vs the National Blend of Models

*Drafted 2026-10-19 for his critique. NOT sealed. The sizing below reads public data only
and never an outcome; a trial is spent only when `--final` runs on a sealed registration.*

## Why this arena

Eight sealed trials in US equities are dead. The edges we have actually found were venue
microstructure (the Cup's nets and sniper) and an information filter (rugwatch). The people
who make money as individuals do it in capacity-limited niches a fund will not enter. Kalshi's
daily temperature markets are that: roughly $3M a day across stations, $500-sized brackets,
settled on an official station reading The Weather Company reports, priced by a crowd that
does not read the National Blend.

## The claim

In the window before the day's high can be observed (market open at 14:00Z the day before,
until 12:00Z on the day), resting prices in the thinner brackets sit further from the
NBM-implied probability than the fee, often enough and deep enough that a bot taking the
cheap side earns a positive expectancy per contract. Mechanism: the counterparty is a retail
quote placed on gut and left standing; the pros do not bother with $500 of depth.

## The sizing (identifiers and liquidity, no outcomes)

- Kalshi public API: settled daily-high markets for 24 US stations, last 90 days, every
  outcome field dropped at ingestion (`clean_market`); trades only between open and the
  12:00Z cutoff.
- Forecast: NBM short-range (NBS) from the IEM archive, `txn` at the 00Z valid time = the
  12Z-00Z maximum, `xnd` = its spread (floored at 1.5F); only runs on the wire 100 minutes
  after run time, before the trade.
- Implied probability per bracket from a normal(txn, xnd), with the 0.5F rounding the
  integer thresholds imply.
- Qualifying trade: |implied - price| minus Kalshi's quadratic fee (rounded up to the cent)
  >= 10 points. Counted in notional, cost, forecast-implied edge, and by who took it: a taker
  on the forecast's side is competition already harvesting; a taker against it is the flow a
  maker could have stood opposite.

## The bar, fixed before the read

Median market day >= $2,000 of qualifying notional across the 24 stations. Below it the
premise is retired on paper, as fire-sale pressure and liquidation unwinds were. Above it a
registration is written (sealed hypothesis, kill criteria, one `--final`) and he seals.

## What this cannot tell us

- Whether the forecast is right: that is the trial, run once, after sealing.
- Our share of the volume: the tape shows what was taken, not what rested untaken.
- Lows: the clean window for a daily low is harder to draw (the minimum can fall late in the
  day). Highs first; lows only if highs clear.

Data lives in `~/Downloads/kalshi_weather/` and is never committed.

## Sizing outcome (2026-10-19, outcomes unread)

- 24 stations, 65 market days, 9,198 markets, 18.9M contracts in the clean windows.
- The pre-fixed bar (median day >= $2,000 qualifying notional) CLEARED at $115k, but the spread
  sweep showed the count rises as the forecast is made duller (42% of volume at the Blend's own
  spread, 56% at 5F): that statistic measured the model's sharpness, not the crowd's error. Size
  can neither retire nor support the claim. Said plainly, so it is not read as support.
- Spread-free: the crowd's centre sits within 0.8F of the Blend on the median station-day, 2F+
  away on 13.8% and 3F+ on 5.0% (72 of 1,433 station-days); on those 72 days $1.43M traded on
  the Blend's side under its implied probability ($573k at the printed prices).
- Registration set built by `size_premise.py register`: 1,017,995 prints over 1,478 station-days
  (11.7M contracts, $4.1M at the printed prices), SHA-256
  bfe5c778fbee8e3c6f55369aa875b0793ddfd10c4f2617d2378535a1621e073c. Unsealed.
- The trial (`run_trial.py --final`, once) is the only test: station-day net profit after fee
  against the sealed bars, and the Blend's Brier against the traded price's.

## Two weaknesses, and what was changed for them (2026-10-19, before the seal)

1. **The Blend's 12Z-00Z maximum is not the calendar-day maximum.** The Weather Company reports
   the local midnight-to-midnight maximum; a warm night before a front, or an evening peak in
   the west, falls outside the Blend's daytime window. Fix: the forecast maximum is now the
   larger of txn and the highest 3-hourly temperature the same run places inside the station's
   local calendar day (per-station time zones in `TZ`). The archive is 3-hourly, so an
   off-window peak is still read slightly low; the error that remains is against us, not for us.
2. **Prints are not independent.** The same resting order hit twenty times is one opportunity,
   and every station on the east coast shares one weather day. Fix: the registered unit is one
   opportunity per bracket per Blend run (the first qualifying print after the run came on the
   wire); a station-day's value is the mean over its opportunities; the t-statistic's standard
   error is clustered by calendar day, and at least 50 calendar days must carry the sample.
