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

---

## What the pattern says

Three looks, three deaths, each in under a minute — the instrumentation is working. But note the
shape: every candidate visible in daily US equity bars is visible to everyone holding the same
bars. Reversal was already owned. Turn-of-month was arbitraged. Low-price pointed the wrong way.

The data will not hand over a mechanism. It can only confirm or kill one brought to it, and it
kills fastest exactly those an outsider could think of by looking at the same screen. The next
candidate should come from something not in a CRSP daily bar: a market with fewer participants,
domain knowledge from outside finance, a holding period nobody else tolerates, or a dataset that
has to be built rather than downloaded.

---

## Running trial count

**3** exploratory trials as of 2026-08-14. Add these to the trial budget of any hypothesis
registered against this universe, and pass the honest total to `deflated_sharpe`.
