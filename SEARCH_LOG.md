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

**7** trials as of 2026-10-15: the five exploratory looks above, plus trials 6 and 7, each
registered, sealed and killed (see the end of this file). Add these to the trial budget of any
hypothesis registered against this universe, and pass the honest total to `deflated_sharpe`.


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

---

## 2026-08-17 — two methodological notes. NOT trials: no data was queried.

Recorded here because both change how the next trial should be chosen, and because the count at
the bottom must stay honest in both directions — inflating it costs real statistical power.

### The ceiling test

Look 4 is usually remembered as "the opportunity was too small". That is the wrong lesson. It
died because **$5.4k was a ceiling**: 99 shares per event is a regulatory cap, and no amount of
capital, skill or patience lifts it. Every other candidate considered since is limited by how
much capital is behind it, not by a rule.

So the question to ask before spending a trial is not "how much would this pay today" but
**is the ceiling set by the mechanism, or by me?** It is answerable on paper.

Order-of-magnitude arithmetic on the current candidates, crowd-adjusted, at retail scale:
post-reorg equity ~16k/yr, spin-off orphan selling ~15k/yr, odd-lot ~3.2k/yr (the model
reproduces the known death, which is the only reason to trust the others), index reconstitution
~3k/yr despite ~$6M/yr of deployable flow — crowding eats it. Weakest inputs are the capturable
share and the per-event edge; change either by 3x and the top two swap.

### The instrument was missing, and that was the actual blocker

Every kill criterion in `preregistration.KillCriteria` describes a **cross-sectional** signal:
information coefficient across thousands of names, quantile monotonicity, annual turnover. None
of them is computable for a claim about forty discrete events. An IC across a cross-section where
3,698 of 3,700 weights are zero is degenerate rather than small; monotonicity across two
positions means nothing.

Note what that implies about the five dead trials. Looks 1, 2, 3 and 5 were all cross-sectional
and all died of being already owned. **Look 4 was the only event-driven one — and the only one
recorded above as breaking that pattern.** The conclusion in "What the pattern says" is that the
next candidate must come from a dataset built rather than downloaded, and every such candidate is
event-shaped. The platform could not have graded any of them.

`crucible/events.py` now supplies the missing instrument: abnormal returns aligned in event time,
CAR per event, and a t-statistic **across events**. `EventKillCriteria` is a separate type rather
than a relaxed `KillCriteria`, because loosening cross-sectional bars until an event claim passes
removes the discipline without replacing it. It adds the guard the headline cannot supply —
**clustering**, since events bunch and a t-statistic assumes independence, so forty events sharing
a reconstitution date are nearer one observation than forty. `scripts/example_event_hypothesis.py`
shows the shape, and is killed by exactly that criterion with every other number passing.

**Trial count unchanged at 5.** Reasoning about candidates is free; the budget is spent when the
data is queried with an outcome in mind. Querying EDGAR for event dates almost certainly does
count — Look 4 did.

### Third note: the spin-off candidate is very likely untestable on 11 years

Still not a trial — no data queried. Chidera chose spin-off orphan selling over post-reorg
equity, and then chose the tightest available version of it: index funds only as the forced
seller; **index non-addition** as the condition, which is binary and therefore costs **zero
parameters**; the first regular-way trade as the event; a 1-10 day window, chosen so the test is
of the forced flow rather than of the 3-6 month spin-off premium published in 1993.

Good choices. The problem is arithmetic, and it is the same kind that killed Look 4.

`minimum_detectable_car` (added to `crucible/events.py` for this) says a sample of N events with
per-event CAR volatility σ cannot resolve anything below `2σ/√N`. A newly listed small-cap over
ten days is plausibly 20-25% volatile. So:

| events | detectable at σ=20% |
|---|---|
| 50 | 5.7% |
| 150 | 3.3% |
| 400 | 2.0% |

Now count the events. ~40 US spin-offs a year; from S&P 500 parents perhaps 10-15; over the
2015-2025 extract that is ~110-165 before conditioning, and **40-55 after**. Requiring a 5.7%
abnormal move over ten days is not a plausible bar, so on this extract the design would return
"no effect" as a statement about its own sample size.

Across the ~35 years already downloaded (1990-94, 1995-99, 2005-09, 2015-25) it is ~150 events
and ~3.3% — demanding but no longer absurd. **The deeper history is not an improvement to this
hypothesis, it is a precondition for it.** The two slices that expired unretrieved, 2000-2004 and
2010-2014, are exactly the gaps in the middle of that span.

Two prerequisites before registering, both cheap and neither a trial:

1. **The distributions table is not in the current extract.** Its 19 columns carry no
   distribution codes, so a spin-off cannot be distinguished from any other newly listed PERMNO.
   WRDS.md now documents the query.
2. **Index constituent history may not be available.** S&P 500 membership commonly is; Russell
   is licensed separately and often is not. If only S&P is reachable, the universe narrows as
   above. If neither is, the conditioning variable has to change and the mechanism should be
   rewritten rather than patched.

**Trial count still 5.**


---

## 2026-10-14 — Trial 6 registered and SEALED before it was run

The spin-off orphan claim above is now registered, with every field that defines it fixed: event
rule, window, mechanism, kill criteria, and the code that runs it. It is **sealed** rather than
published, by commit-reveal: the full registration lives in a private repository, and only its
hash is published here, so this entry proves the claim existed in exactly that form before any
result did, without saying what it is.

- **SHA-256 of `registration.json`:** `f1cd884f75997fcc1186c7351569a42525efb2e054756adb76d7301b4c713543`
- **`Hypothesis.fingerprint`:** `758a2901b006df48`

Revealing it means publishing `registration.json`. Anyone can then hash it and compare. If the
claim is killed, it will be revealed here in full, because a dead idea has nothing left to
protect. If it survives, it may stay sealed while it is traded; the hash above still dates it.

Not yet run when this was written. **Trial count: 5, becoming 6 the moment it runs.**

---

## 2026-10-14 — Trial 6 RUN and KILLED. Revealed in full, as committed.

Scored once at 16:16:50 UTC, twelve minutes after the sealing commit above. The registration is
now public at `registrations/trial-6-spinoff-orphans/`. Verify it against the hash published
before the run:

```
shasum -a 256 registrations/trial-6-spinoff-orphans/registration.json
# f1cd884f75997fcc1186c7351569a42525efb2e054756adb76d7301b4c713543
```

The three scripts in that folder match the SHA-256s recorded inside `registration.json`. The event
list itself is derived from licensed CRSP data and is not published; its hash is in the
registration.

**The claim.** Spin-offs of S&P 500 companies that the index does not keep are sold by index
funds at their first close, and earn a positive abnormal return from the next session's close
through two sessions after. The mechanism is in Chidera's words: *the traders who buy the shares
the index funds are forced to dump at the spin-off's first close are middlemen who can't afford
to hold a small, risky new stock, so they sell it to me below fair value for about 2 days until
they've unloaded.*

**The events.** 105 new listings from 1990 to 2025, the common-stock spin-offs of S&P 500 members
that were not in the index on the session after their first trade. 79 are before 2015-10-01, when
funds sold at their own discretion. 26 are after, when S&P DJI's rule adds a spinco at a zero
price and removes it after one regular-way day, so the sale is a single closing trade (14 of the
26 were one-day members; 12 were never added). Two things were found while building the set, and
both were fixed before sealing. The August count of 115 orphans had classified the one-day
members as "added", which is exactly backwards for this claim. It had also included pre-existing
securities and twelve REITs, ADRs and foreign issuers that no crucible panel contains.

**The result.**

| criterion | bar | result |
|---|---|---|
| mean abnormal return, offsets 2-3 | >= +2.50% | **-1.35%** |
| \|t\| across events | >= 2.64 (Bonferroni over 6 trials) | **2.28**, negative |
| win rate | >= 50% | **47%** |
| events placed / dropped | >= 30 / <= 20% | 105 / 0 |
| clustering | <= 50% | 13% |

Not underpowered: the sample could resolve 1.56%, below the bar. **Killed**,
and in the opposite direction: buying at the day-1 close lost 1.35% against the equal-weighted
market over the next two sessions. That is not a significant short either, since |t| 2.28 is below
the same corrected bar.

**What this says, and what it does not.** The registered window held through the period the
mechanism says the middlemen are still unloading, and the price fell there. The window was set by
translating "the unloading lasts about two days" into "hold for those two days", when the
mechanism more naturally predicts a price pushed DOWN during the unloading and a recovery after
it. So the result is consistent with selling pressure that persists past the first close, and
inconsistent with the registered claim that it reverses inside the window. Whether it reverses
afterwards is a different claim, motivated by having seen this one, and would be trial 7. It is
**not being run**. At about three events a year in small names, a real edge there pays a few
hundred dollars a year at this account's size, and trying a new window immediately after a kill
is how a search starts presenting itself as research.

**Lesson for the next registration:** derive the traded window from the mechanism's predicted
price PATH (down, then up), not from its duration.

---

## 2026-10-15 — Post-reorganization equity, retired on paper. NOT a trial: no return was read.

The candidate after trial 6 was the creditor-forced selling of new equity after a Chapter 11
emergence (the Eberhart–Altman–Aggarwal effect). Before any registration, the event rule was built
from identifiers alone — CRSP's delisting table (`DelReasonType`, `DelActionType`) and the daily
slices' PERMNO/PERMCO/classification columns, never a price or a return — and the count is the
verdict:

| rule | 1990–2025 | 2005–2025 |
|---|---|---|
| old equity delisted with reason `BKPY`, same PERMCO relists on a CRSP exchange 1–30 months later, US common | 31 | 15 |
| any dropped-by-exchange reason (`BKPY`, `INSC`, `INSF`, `FING`, `DELQ`, `LP`, `EQRQ`, `CORQ`, `MVOT`), same window | 84 | 38 |
| the same, 1–60 months | 105 | 49 |

Ticker and issuer-CUSIP matching added nothing beyond PERMCO. Two things set the ceiling, and
neither is capital:

1. **Frequency.** Even pooling 1990–2025, about a hundred emergences are visible. At a per-event
   volatility near 40% over a 100-session window the smallest effect a hundred events can resolve
   at the seven-trial bar (t ≥ 2.69) is about 11%; the registered claim was 8%, and the
   2005–2025 regime alone (≤49 events) resolves ~15%. The study would come back UNINFORMATIVE,
   and an uninformative trial still spends one.
2. **Selection.** Most post-reorg equity trades OTC first. CRSP sees only the emergences that
   listed, or relisted, on an exchange — the ones that went well enough to uplist — so the
   measurable sample is selected on the outcome the claim is about, in the direction that would
   flatter it. No window fixes that; only OTC return data would, and there is none here.

The Florida–UCLA–LoPucki Bankruptcy Research Database (free, ~1,000 large public bankruptcies
1979–2022) would give true emergence dates but not OTC returns, so it changes the count by a few
and the selection problem not at all.

Retired without a registration. **Trial count unchanged at 6.** The event builder is kept in the
private registrations folder, marked retired, because the funnel numbers above are the kind of
thing a later session re-derives at a cost of an afternoon.

**Lesson for the next candidate:** an event must be exchange-listed from offset 0 — the spin-off
set had that property and this one does not — and frequent enough that the deflated bar can be
resolved at a plausible dispersion. Both are answerable from identifiers before anything is
registered, and they should be answered first.

---

## 2026-10-15 — Trial 7 registered and SEALED before it was run

A spin-off claim, registered with every field that defines it fixed — event rule, window, the
two holding rules, mechanism, kill criteria, and the code that runs it — and **sealed** by
commit-reveal exactly as trial 6 was: the full registration lives in a private repository, and
only its hash is published here, so this entry proves the claim existed in exactly this form
before any result did, without saying what it is.

- **SHA-256 of `registration.json`:** `b34026c32b32019299125802a426487dfb9d770fcb98d558dfd42988f602e82c`
- **`Hypothesis.fingerprint`:** `85c09064ec2d4ad0`

What can be said now: it is a different claim from trial 6 — different sellers, a different
horizon, a different population — and it was proposed the day after that kill. Whether that is
research or search was put to Chidera in those words, with trial 6's own warning quoted back,
and he ruled research; the registration records that ruling. The mechanism paragraph was drafted
by Claude at his instruction and critiqued by him, under a rule he lifted the same day; the
earlier reasoning about why that is dangerous (training data is the most-public information in
finance, and a menu of mechanisms is a search) still governs how it was done: one claim, no menu,
its decay risk stated inside it, and nothing queried before this seal except identifiers.

If the claim is killed, it will be revealed here in full. Not yet run when this was written.
**Trial count: 6, becoming 7 the moment it runs.**

---

## 2026-10-15 — Trial 7 RUN and KILLED. Revealed in full, as committed.

Scored once at 01:29:58 UTC on 2026-10-15, about three hours after the sealing entry above. The
registration is now public at `registrations/trial-7-spinoff-holder-unloading/` in this
repository. Verify it against the hash published before the run:

```
shasum -a 256 registrations/trial-7-spinoff-holder-unloading/registration.json
# b34026c32b32019299125802a426487dfb9d770fcb98d558dfd42988f602e82c
```

The three scripts in that folder match the SHA-256s recorded inside `registration.json`. The event
list is derived from licensed CRSP data and is not published; its hash is in the registration.

**The claim.** A spun-off company's shares are distributed to the parent's holders, most of whom
cannot keep them, and are sold down over the first weeks of trading; bought at the close of the
30th session after the first trade and held to the close of the 150th, they earn a positive
abnormal return. The mechanism: index funds hold the spinco only if it qualifies for their index,
large-cap and dividend mandates hold a stock of the parent's size and yield, many holders never
chose it; they sell over the first several weeks into a name with no coverage and no natural
owner, and the price recovers over the following months as the sellers finish.

**The events.** 298 spin-offs with a new listing within five days of the ex-date, US common
stock on the day of the first trade, first traded 2005-2025, any parent. 293 placed, 5 dropped
(their 150-session window runs past the end of the data). Two holding rules were registered:
a listed day with no trade carries zero abnormal return, and a spinco delisted inside the
window takes CRSP's delisting return and then holds cash at minus the market.

**The result.**

| criterion | bar | result |
|---|---|---|
| mean abnormal return, offsets 31-150 | >= +7.00% | **+0.29%** |
| \|t\| across events | >= 2.69 (Bonferroni over 7 trials) | **0.14** |
| win rate | >= 50% | 52% |
| events placed / dropped | >= 30 / <= 20% | 293 / 2% |
| clustering | <= 50% | 33% |

Not underpowered: the sample could resolve 5.65%, below the 7% bar, so a real effect of the
registered size would have been seen. **Killed**, and cleanly: a mean of three tenths of a
percent over 120 sessions, with half the events up and half down, is the absence of the effect,
not a sample too small to find it.

**What this says.** The mechanism predicted a price pressed by holders unloading and recovering
once they finish. Whatever pressure exists is not followed by a recovery in sessions 31 to 150:
either the unloading is priced at once, or the holders who cannot keep a spinco are a smaller
share of the float than the story assumes, or the buyers who can hold it are already there. The
published spin-off drift of the 1990s is not present in this population at this horizon.

**What it does not say, and what is retired with it.** Trial 6 found the first days of an S&P 500
orphan losing to the market; this trial finds the following months flat for all spin-offs. The
spin-off population has now carried two registered claims and killed both. Any further window on
it — earlier, later, a different subset — is a search, and is not being run. The next candidate
has to come from a different forced seller.

**Lesson for the next registration:** the identifier-level checks introduced this week (listed
from offset 0, frequent enough to resolve the deflated bar) worked exactly as intended — the
post-reorg candidate died on paper, this one died in the data — and the cost of the second was
one trial. A mechanism's premise ("holders who cannot keep it") should be sized before it is
registered: what fraction of the float does the constrained holder actually own? That number is
answerable from 13F data before any return is read, and it was not asked here.

---

## 2026-10-15 — Mutual-fund fire-sale pressure, retired on paper. NOT a trial: no return was read.

The candidate after trial 7 was Coval & Stafford's forced selling: funds hit by heavy redemptions
must sell what they hold inside the quarter, so the stocks they hold in size are pressed and then
recover. The lesson written at the end of trial 7 was applied before anything was registered —
size the premise from holdings, never from returns. The pull is specified in `WRDS_FIRE_SALES.md`
(ten WRDS queries, listed there); the measure and the line were fixed in that file before the
data existed, and the script that computed it is `size_premise.py` in the private registrations
folder, with its aggregate output beside it.

**The measure.** For each quarter 2005Q1–2025Q4 and each US common stock (crucible's CIZ
classification), expected forced sales = Σ over domestic-equity, non-index, non-ETF funds whose
compounded flow over the quarter is at or below −5% of shares held at the quarter start × |flow|,
divided by shares outstanding at the quarter start. Holdings are the fund's latest s12 report in
the two quarters before, first vintage; flows are CRSP monthly TNA and returns summed over share
classes through MFLINKS. The pre-stated line: if the median stock in the top pressure decile has
expected forced sales under 1% of shares outstanding, the premise cannot move a price.

**The numbers** (83 quarters; 2,467 eligible funds a quarter at the median, 507 of them distressed —
one in five; 10th-percentile quarterly flow −8.2%; eligible funds own 10.4% of the median stock in
2005 and 7.0% in 2025, so the unit check passes and the share is falling):

| statistic | value |
|---|---|
| median over quarters of the top-decile median expected forced sales | **0.63%** of shares outstanding |
| inter-quartile range of that number across quarters | 0.50% – 0.78% |
| worst quarter (2020Q1) | 1.59%; then 2007Q3 1.41%, 2015Q3 1.33%, 2008Q3 1.21% |
| quarters at or above the 1% line | 9 of 83 |
| stocks with expected forced sales ≥ 2% in a quarter | median 3, mean 9, max 112 (2020Q1), 723 stock-quarters in all |
| stocks at or above 5% | 25 stock-quarters in twenty-one years |
| share of the top decile held by distressed funds (unweighted) | 6.7% |
| median size decile of the top-decile names (1 = smallest) | 7; 27% sit in the small half |
| top-decile names with average daily dollar volume ≥ $1M | 89% |

**Verdict.** The line was 1% and the number is 0.63%; the premise fails on its own terms. The
distressed funds together own about seven percent of a top-decile stock and are redeeming eight
to ten percent of themselves, so the selling they are forced into is six tenths of a percent of
the company over a quarter — a day's turnover spread across sixty sessions. It is also not where
the story said it would be: the pressure concentrates in deciles 6–8, in names any fund can
trade, not in small illiquid ones a distressed fund cannot leave quietly. The strong form (two
percent or more of a company forced out) exists only in crisis quarters, a few dozen names at a
time and three in an ordinary quarter: an event set of ~700 stock-quarters in twenty-one years,
nearly all sharing a handful of dates, which is the clustering that killed the example event
claim before a single trial was spent.

Retired without a registration. **Trial count unchanged at 7.** Nothing in the daily file beyond
shares outstanding, market capitalisation and dollar volume was read; no return or price series
was opened.

**Lesson for the next candidate.** The premise-sizing step cost an evening and no trial, and it
settled the question the paper leaves open for this era: by 2005 the mutual-fund channel is too
thin and too arbitraged to force a price. The structural seller for the next claim has to own a
larger share of the company than a distressed mutual fund does, or be forced to sell faster.

---

## 2026-10-15 — Mutual-fund liquidation unwinds, retired on paper. NOT a trial: no return was read.

The candidate after the fire-sale sizing was the same seller forced faster: a fund being
liquidated must sell every position by a date. The claim was that where such a fund owns a few
percent of a small company, that selling is many days of volume forced through in weeks, the
price is pressed into the liquidation date and recovers over the following one to three months.
The rule and the line were fixed before the count (private registrations folder,
`liquidation-unwinds/size_premise.py`, aggregate record beside it): proceed only with at least 150
event stocks 2005–2025 — a liquidating domestic-equity fund's position of at least 2% of shares
outstanding and at least 5 days of average dollar volume at its last report — with no calendar
month holding more than a quarter of them.

**The count.** 590 liquidated active domestic-equity funds 2005–2025 (every share class dead, the
last one coded L), 566 with an s12 report inside twelve months of the last NAV (median 31 days
before it), 40,266 positions in US common stock on those reports.

| statistic | value |
|---|---|
| median position | 0.00% of the company, 0.0 days of volume |
| 99th-percentile position | 0.18% of the company, 0.4 days of volume |
| largest position per fund, median / 90th percentile | 0.02% / 0.30% of the company |
| positions ≥ 1% of the company and ≥ 3 days of volume | 11, in 6 funds |
| positions at the registered line (≥ 2%, ≥ 5 days) | **1** (June 2020, a $46M company) |

**Verdict.** The seller is forced, but it owns nothing. The funds that get liquidated are small
(the one qualifying fund ended at $18M) and diversified into liquid names; their unwinding is
invisible against normal volume. One event in twenty-one years against a bar of 150.

Retired without a registration. **Trial count unchanged at 7.** Same data as the fire-sale
sizing; nothing new was pulled and no return or price series was opened.

**What the two sizings together say.** The mutual-fund channel is closed on both sides: a
distressed fund holds about seven percent of the stocks it presses and sells a tenth of itself; a
liquidating fund holds a few hundredths of a percent and sells all of it. Neither moves a price
in this era. The next structural seller has to own the company in blocks — tens of percent of the
float — and be forced out on a date: merger arbitrageurs at a deal break, or holders bound by an
index rule, are the sizes that remain. Either is sizeable from holdings before a return is read.
