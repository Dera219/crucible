# Sizing result: Kalshi daily highs vs the National Blend, clean window only

Measured 2026-10-19 14:33Z: 65 market days, 24 stations, 18,856,119 contracts traded in the clean windows (open to 12:00Z of the market day). Outcomes never read.

Qualifying = traded at a price whose gap to the NBM-implied probability clears the fee by 10%+. Bar (fixed 2026-10-19 before the read): median day >= $2,000 notional.

| metric | value |
|---|---|
| qualifying notional, median day | $114,997 |
| qualifying notional, mean day | $121,592 |
| qualifying notional, total | $7,903,476 of 18,856,119 (41.9%) |
| cost to buy the cheap side, total | $2,778,217 |
| forecast-implied edge after fee, total (NOT a result) | $1,730,702 |
| taken by a taker on the forecast's side (competition) | 3,224,242 |
| taken by a taker against the forecast (flow a maker could meet) | 4,679,234 |
| **verdict against the bar** | **CLEARS** |

## By station

| station | traded | qualifying | share | competition | flow |
|---|---|---|---|---|---|
| KLAX | 5,045,611 | 2,428,234 | 48.1% | 1,072,345 | 1,355,889 |
| KNYC | 1,962,300 | 848,151 | 43.2% | 345,811 | 502,339 |
| KMDW | 1,329,998 | 605,771 | 45.5% | 286,594 | 319,177 |
| KMIA | 1,702,740 | 522,025 | 30.7% | 218,964 | 303,060 |
| KAUS | 950,567 | 413,353 | 43.5% | 144,480 | 268,873 |
| KPHL | 696,092 | 311,899 | 44.8% | 119,914 | 191,984 |
| KDFW | 707,911 | 276,328 | 39.0% | 98,018 | 178,310 |
| KPHX | 612,519 | 256,727 | 41.9% | 112,447 | 144,280 |
| KATL | 677,651 | 230,952 | 34.1% | 87,596 | 143,356 |
| KSEA | 594,087 | 225,631 | 38.0% | 92,285 | 133,346 |
| KDEN | 592,102 | 220,301 | 37.2% | 81,621 | 138,681 |
| KSFO | 564,275 | 220,067 | 39.0% | 82,306 | 137,760 |
| KHOU | 515,716 | 203,820 | 39.5% | 83,637 | 120,183 |
| KBOS | 432,089 | 193,424 | 44.8% | 70,034 | 123,389 |
| KOKC | 501,003 | 175,848 | 35.1% | 52,983 | 122,865 |
| KLAS | 430,226 | 162,133 | 37.7% | 49,856 | 112,277 |
| KDCA | 400,851 | 149,970 | 37.4% | 54,648 | 95,322 |
| KMSP | 360,084 | 149,848 | 41.6% | 45,485 | 104,363 |
| KSAT | 282,053 | 117,689 | 41.7% | 35,445 | 82,244 |
| KMSY | 281,620 | 83,790 | 29.8% | 27,421 | 56,369 |
| KSDF | 70,574 | 35,624 | 50.5% | 26,687 | 8,937 |
| KTTN | 61,077 | 30,667 | 50.2% | 18,728 | 11,939 |
| KEWR | 48,269 | 23,491 | 48.7% | 11,570 | 11,921 |
| KSAN | 36,701 | 17,734 | 48.3% | 5,367 | 12,367 |

## Robustness: the same count with a duller forecast spread

The Blend's stated spread (xnd, often 2F) is known to be underdispersive. If the qualifying volume vanishes at 4-5F the 'mispricing' was our sharpness, not the crowd's error.

| spread | qualifying median day | mean day | total | share of volume | competition | flow |
|---|---|---|---|---|---|---|
| NBM xnd (floored 1.5F) | $114,997 | $121,592 | $7,903,476 | 41.9% | 3,224,242 | 4,679,234 |
| 3F | $129,745 | $135,256 | $8,791,653 | 46.6% | 3,406,524 | 5,385,129 |
| 4F | $152,167 | $151,336 | $9,836,834 | 52.2% | 3,828,612 | 6,008,222 |
| 5F | $159,494 | $161,476 | $10,495,940 | 55.7% | 4,200,980 | 6,294,960 |

## The crowd's centre vs the Blend's (spread-free)

Per station-day: the expected high implied by the last traded price of each bracket in the window, against the Blend's 12Z-00Z maximum. A mean disagreement is where an edge would live whatever the spread.

| station-days with a usable centre | 1433 |
|---|---|
| median absolute disagreement | 0.8 F |
| share >= 2 F | 13.8% |
| share >= 3 F | 5.0% |
| share >= 5 F | 1.0% |

On the station-days with a 3F+ disagreement, what traded on the Blend's side under its implied probability by more than the fee plus 5 points:

| spread | station-days | notional | cost | implied edge (NOT a result) |
|---|---|---|---|---|
| 3F | 72 | $1,431,062 | $573,308 | $376,691 |
| 4F | 72 | $1,396,413 | $536,721 | $339,877 |

## By day (last 20)

| day | traded | qualifying | edge after fee |
|---|---|---|---|
| 2026-09-27 | 288,900 | 138,170 | $27,901 |
| 2026-09-28 | 271,858 | 107,728 | $22,967 |
| 2026-09-29 | 307,723 | 82,461 | $13,745 |
| 2026-09-30 | 287,712 | 106,743 | $31,602 |
| 2026-10-01 | 307,693 | 143,585 | $34,775 |
| 2026-10-02 | 317,341 | 128,985 | $25,986 |
| 2026-10-03 | 321,901 | 105,872 | $23,798 |
| 2026-10-04 | 274,999 | 134,886 | $30,037 |
| 2026-10-05 | 342,635 | 186,513 | $49,661 |
| 2026-10-06 | 358,923 | 221,358 | $48,526 |
| 2026-10-07 | 315,883 | 142,514 | $30,688 |
| 2026-10-08 | 316,095 | 140,152 | $33,833 |
| 2026-10-09 | 337,016 | 132,711 | $32,959 |
| 2026-10-10 | 282,732 | 117,885 | $24,516 |
| 2026-10-11 | 338,520 | 193,879 | $44,889 |
| 2026-10-12 | 299,872 | 122,855 | $23,758 |
| 2026-10-13 | 237,885 | 118,448 | $25,939 |
| 2026-10-14 | 305,599 | 154,429 | $35,021 |
| 2026-10-15 | 389,290 | 180,425 | $45,463 |
| 2026-10-16 | 418,990 | 225,172 | $55,699 |
