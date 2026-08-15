# Search log

Every exploratory look at the data, recorded before it can be forgotten.

## Why this file exists

`crucible.significance` deflates a Sharpe ratio by the number of trials it took to find it, and
`preregistration.summarize()` reports every hypothesis ever registered so that "deflation can be
run against the honest total rather than against today's". Both are defeated by the same thing:
searches that happened and were never written down.

An exploratory look is a trial. It does not become one when it succeeds — it was one the moment
the data was queried with an outcome in mind. A file of dead ideas is the only defence against
quietly computing a deflated Sharpe against a trial count of one, having actually looked eleven
times.

So: **every look goes here, especially the ones that died.** The count at the bottom is the
number that belongs in the next deflation.

## Ground rules

- Write the mechanism and the prediction **before** running the query. A result you can explain
  afterwards is not a prediction.
- Record the death honestly, including the ones that died of costs rather than of signal — "the
  effect is real but 2bps" is a different fact from "there is no effect", and only one of them
  says anything about where to look next.
- A look that was abandoned half-run still counts. The trial budget is spent by asking, not by
  finishing.

---

## 2026-08-14 — first exploratory session

Universe: verified 2015-2025 CRSP CIZ extract (WRDS query 11570520, 23,101,820 rows), loaded
through `load_crsp_ciz` with delisting returns. 6,635 securities, 2,766 sessions, 2,976
delistings in-sample. This is the first extract with `DlyPrcFlg`, so it is also the first in
which non-traded sessions are correctly excluded — see WRDS.md.

### Look 1 — short-horizon reversal, conditioned on volume

**Edge source:** liquidity provision.
**Mechanism:** a large one-day move on unusually heavy volume means somebody transacted because
they had to. They paid for immediacy, so part of the move should revert, and the reversal should
concentrate where the volume says the urgency was.
**Prediction:** decile-0-minus-decile-9 next-day spread, materially larger in high-volume names.

**Result:** the signature is present and correctly shaped — in the extreme-volume bucket the
biggest losers bounce and the biggest winners fade — but the tradable spread is **+2.0 bps/day**
against a book that must rebalance the entire cross-section daily. Round-trip costs in the top
30% by dollar volume are 5-15bps.

**Verdict: DEAD on costs.** The mechanism is real and already owned by people with better
execution. Informative: liquidity provision is not available in liquid names, and in illiquid
ones the cost model correctly refuses to trade.

### Look 2 — turn-of-month

**Edge source:** structural constraint.
**Mechanism:** index funds and mandated portfolios must trade at month and quarter boundaries
regardless of price, because their obligation is tracking error rather than execution quality.
Pension contributions arrive on a payroll calendar. Nobody in that group is choosing the moment.
**Prediction:** turn-of-month sessions (last trading day plus first three) outperform the rest,
and — critically — still do so in the back half of the sample, since the effect has been
published since the 1980s.

**Result:** turn-of-month returned **11.70 bps/day** against **15.25** for the rest of the
month — the wrong sign, by 3.56 bps/day, consistently in both halves (-3.22 early, -3.89 late).

**Verdict: DEAD, and reversed.** No decay story: it is simply not present in this universe over
this period. (Absolute levels are inflated by daily-rebalanced equal weighting; both buckets use
identical construction so the comparison stands.)

### Look 3 — the sub-$5 segment

**Edge source:** structural constraint, against a competing behavioural story.
**Mechanism, two-sided on purpose:** institutional mandates commonly forbid holding stocks under
$5, which are also index-excluded and often margin-ineligible. A fund crossing that line sells
because a document says so. That predicts a discount a solo book could collect. Against it:
retail overpays for cheap high-variance lottery tickets, which predicts the opposite.
**Prediction:** whichever dominates, the sign is decisive either way.

**Result:** forward 21-day returns are near-monotone in raw price. Annualised: `<$1` **-0.4%**,
`$1-2` +1.3%, `$2-5` +1.3%, `$5-10` +6.8%, `$10-30` +10.4%, `>$100` **+11.0%**. Median 21-day
return in `<$1` is **-9.19%** against a mean near zero — the lottery distribution in one number.
Sub-$1 names delist at ~5-10x the rate of the rest, and the second half is worse than the first.

**Verdict: DEAD as a long, unavailable as a short.** The constraint and the badness are aligned,
not opposed: funds cannot hold these names, and the names are genuinely value-destroying. There
is no discount to collect. Shorting sub-$5 borrow is expensive, squeeze-prone and effectively
closed to a small account.

**Kept from it:** empirical justification for a price floor in universe construction. Removing a
value-destroying segment is hygiene, not alpha, but the number is now evidence-backed rather
than folklore.

### Look 4 — odd-lot tender provisions

**Edge source:** structural constraint, with capacity as the moat.
**Mechanism:** when a company tenders for its own shares it frequently accepts holders of fewer
than 100 shares IN FULL, exempt from proration. The provision exists because small holders are
administratively expensive and buying them out cleanly is worth a concession — so the rule is
written to benefit exactly the size of holder a fund cannot be. Counterparty and motive are both
stated in the filing rather than inferred.
**Prediction:** a workable number of offers per year carry the provision, and the tender price
exceeds market at expiration by more than costs.

**Result, measured against EDGAR rather than folklore:**

| stage | count |
|---|---|
| SC TO-I filings, 2025 | ~578 |
| carrying odd-lot language | ~52 |
| mapping to an exchange ticker | 25 |
| listed AND not a fund repurchase programme | **18** |

Leg 1 confirmed, and not weakly: one 2025 filing disclosed a proration factor of **49.3%** —
a normal holder had half their shares accepted, an odd-lot holder all of them. The rule works
exactly as written.

Leg 2 was never reached, because the population is not what the folklore says. Of ~52 filings
with odd-lot language, **21 are non-traded BDCs and interval funds** running periodic repurchase
programmes — `N/A (CUSIP Number)`, no exchange price to transact against, and "odd lot priority"
meaning fairness among existing holders rather than an opportunity. Of the listed remainder,
several are closed-end funds, warrants (`RUMBW`, `SQFTW`) or dual-class B shares (`LEN-B`).

**Verdict: DEAD on capacity, not on mechanism.** ~18 reachable events a year, capped at 99
shares. Even a generous 10% premium on a $3,000 position is ~$300 an event, ~$5,400 a year gross,
before per-deal filing reads and election paperwork — and before several of the 18 prove
unattractive or unreachable.

**Kept from it:** the transferable finding, which is sharper than the result. "Too small for a
fund" and "big enough for me" is a narrower band than it sounds. A capacity advantage only pays
where the per-event prize scales with something OTHER than the constraint creating the moat, and
an odd-lot provision caps the prize by construction — the rule that grants the edge is the rule
that bounds it. The next candidate wants to be small-capacity but NOT small-per-event.

### Look 5 — the illiquidity premium, at a size that can actually collect it

**Edge source:** risk premium, capacity as a moat that does NOT cap the prize — deliberately
chosen to answer Look 4's failure.
**Mechanism:** a fund's mandate requires it to be able to exit without moving the price, so it
pays up for liquidity it will mostly never use. That payment is the return. The constraint limits
how much can be deployed, not how much each dollar earns.
**Prediction:** illiquid names out-earn liquid ones gross, and the question worth answering is
whether the spread survives spread-and-impact at a small book's size — which the literature does
not address and this repository's cost model exists to.

**Result:** monotone, and the wrong way.

| quintile | annualised | median $vol/day |
|---|---|---|
| 0 most liquid | **+12.2%** | $182M |
| 1 | +10.8% | $37.6M |
| 2 | +10.7% | $12.2M |
| 3 | +8.3% | $3.7M |
| 4 most illiquid | **+4.1%** | $602k |

Illiquid minus liquid: **-7.3%/yr GROSS**, before a cent of trading cost. Already guarded with
the $5 raw price floor (so it is not rediscovering penny stocks) and with delisting returns
included (so the dead names are not dropped).

**Verdict: DEAD, and it died a third way.** Looks 1-3 died because the effect was already owned.
Look 4 died on capacity arithmetic. This one died because the premium is not in the sample at
all — investors were paid LESS, monotonically, at every step down the liquidity ladder.

**Kept from it — and this outranks the result: the sample has a regime problem.** Price quintiles
(Look 3) and liquidity quintiles (Look 5) both rank monotonically in the same direction: big,
liquid and expensive won, small, illiquid and cheap lost, across the whole decade. 2015-2025
contains one of the most extreme large-cap concentrations in market history, so ANY factor
correlated with size or quality inherits it. Over this window the data cannot distinguish "this
factor does not work" from "this factor was on the wrong side of one regime" — which means five
trials have been spent on an instrument not yet sensitive enough to answer them. See the note
below on sample depth.

---

## What the pattern says

Four looks, four deaths — the instrumentation is working. Three of them shared a shape: every
candidate visible in daily US equity bars is visible to everyone holding the same bars. Reversal
was already owned. Turn-of-month was arbitraged. Low-price pointed the wrong way.

Look 4 broke that pattern and is the more interesting death. It was NOT visible in the bars — it
required assembling filing data nobody hands you, the mechanism was documented rather than
inferred, and the moat was genuine. It died on arithmetic instead: the opportunity was real and
too small. That is a different lesson and a more useful one.

The data will not hand over a mechanism. It can only confirm or kill one brought to it, and it
kills fastest exactly those an outsider could think of by looking at the same screen. The next
candidate should come from something not in a CRSP daily bar: a market with fewer participants,
domain knowledge from outside finance, a holding period nobody else tolerates, or a dataset that
has to be built rather than downloaded.

---

## Running trial count

**5** exploratory trials as of 2026-08-15. Add these to the trial budget of any hypothesis
registered against this universe, and pass the honest total to `deflated_sharpe`.


## The sample depth problem — read this before spending trial 6

Two independent sorts today (raw price, Amihud illiquidity) ranked monotonically in the same
direction. That is not two findings; it is one regime showing up twice. 2015-2025 US equities
were dominated by a handful of very large, very liquid names, and any cross-sectional factor
tilting small, cheap or illiquid was short that concentration for the entire sample.

The consequence is methodological rather than economic: **this window cannot separate a dead
factor from a factor caught on the wrong side of one regime.** Five trials have been charged to
the deflation budget for questions the data was never able to answer.

`WRDS.md` already says this in its own words — "start as early as the subscription allows...
depth is not a luxury here, it is the difference between being able to conclude anything and
not" — and `significance` quantifies it: against a 50-trial search, an edge near Sharpe 1.27
needs roughly 20 years to separate from luck. The extract in hand is 11.

CRSP daily data begins in 1925 and the subscription almost certainly reaches back decades before
2015. The cheapest possible improvement to every future trial is one more WRDS query with an
earlier start date, covering the dot-com peak and bust, the financial crisis, the ZIRP decade and
the 2022 inflation shock. Until then, a negative result here means "not in this regime" and
should be recorded as such rather than as "does not work".

---

## 2026-08-15 — the regime question, answered

Look 5 was re-run per era on deeper history (WRDS slices 1990-1994, 1995-1999, 2005-2009 plus the
existing 2015-2025), with two corrections that both cut AGAINST the result worth wanting:

- **A real-terms price floor.** The original $5 was nominal, so early eras screened out roughly
  twice as much of the market as late ones and the samples were never comparable. The floor is
  now deflated by CPI to constant 2025 dollars — $2.18 in 1990-1994 against $4.23 in 2015-2025.
- **Newey-West t-statistics, 21 lags.** Overlapping 21-day windows are autocorrelated, and an
  uncorrected t overstates significance by roughly the square root of the overlap.

| era | Q0 liquid | Q4 illiquid | Q4-Q0 | t |
|---|---|---|---|---|
| 1990-1994 | 12.9% | **23.1%** | **+9.1pp** | 2.07 |
| 1995-1999 | 23.8% | 22.8% | -0.8pp | -0.17 |
| 2005-2009 | 4.7% | 5.3% | +0.6pp | 0.15 |
| 2015-2025 | 13.0% | **4.8%** | **-7.3pp** | **-2.87** |

**The real floor changed nothing material** (+9.7pp to +9.1pp), so the original screen was not
driving the early result — worth knowing, since that was the flaw most likely to have manufactured
it.

**Conclusion, stated at the strength the evidence supports.** The modern penalty is robust: t of
-2.87 on the largest sample, monotone across all five quintiles. The 1990s premium is suggestive
only: t of 2.07 clears 1.96, but FOUR eras were tested and one significant result in four is
roughly what chance produces from nothing. Under a Bonferroni bar of about 2.5 it does not
survive; the modern penalty does.

So: suggestive that illiquidity once paid, confident that it now costs.

**What this settles, and what it retires.** The regime hypothesis was right that the sample
mattered and wrong about what depth would show. Deeper history did not rescue the thesis — it
revealed a premium that existed, decayed through the 1990s, and inverted. Trading it today means
betting that 1990-1994 conditions return, which is not a mechanism.

It also retires the broader worry that the five dead trials were an artefact of one hostile
decade. Extending to 1990 changed one result's INTERPRETATION and no result's VERDICT. The
sample-depth problem was real and is now measured rather than suspected.

**Still unmeasured:** 1985-1989, 2000-2004 and 2010-2014 were queried but not downloaded. Three
more windows would firm up the decay path, and 1985-1989 carries the 1987 crash. Nothing above
depends on them.

