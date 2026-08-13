# crucible

A cross-sectional equity research platform — panel data, a signal algebra, a vectorised
backtester with square-root market impact, and validation machinery that tries to kill your
results before you believe them.

**321 tests. ruff + mypy strict clean. And thirteen bugs it caught in its own code, every one
with a green test suite already passing.**

That last number is what this README is about.

---

## The finding

I kept a record of every bug found after the tests were green. When I lined them up, the pattern
was not subtle:

> **Twelve of the thirteen made results look *better* than reality. None made them look worse.**

That is not a coincidence, and it is not bad luck. In quantitative research the errors are
**asymmetric by construction**. A bug that loses information gets noticed — the strategy stops
working, someone investigates. A bug that *adds* information looks like a discovery, gets
celebrated, and gets funded. Nobody debugs a good result.

So the interesting engineering question stopped being "does the code work" and became **"how do
you find the bugs that are hiding behind a passing test and a plausible number?"**

Every technique below came out of trying to answer that.

---

## The thirteen

| # | Bug | Measured impact | Found by |
|---|---|---|---|
| 1 | Delisting return computed, then discarded when equity was rebuilt from returns | `-100%` and `exit flat` were **identical** | Testing that an assumption changes the answer |
| 2 | Costs batched against a starting equity the strategy no longer had | compounding error, silent | Reasoning about data dependencies |
| 3 | Market impact used annualised volatility where the √-law is per-period | inflated **16×**; capacity read **$6.4k** instead of **$1.6m** | Sanity-checking a number against intuition |
| 4 | IC and the engine both shifted by `execution_lag`, one period too many | a planted edge of **IC 0.13** measured **0.008** | Planting a signal of known strength |
| 5 | A perfectly leaked target reported *"no detectable edge"* | zero IC variance ⇒ t-stat 0 | Feeding in the answer as the feature |
| 6 | "Driven by one tail" warning fired on a *perfectly linear* signal | false positive on the ideal case | Testing the warning against the best case |
| 7 | Delisting return charged once **per non-trading gap** | **-96.4%** instead of **-42.5%** | Loading realistically-shaped data |
| 8 | `net = gross − costs` silently false; delisting drag was a hidden third term | failed **885 / 3000** fuzzed runs | Fuzzing an accounting identity |
| 9 | Daily loss limit re-baselined on every restart | **9.15%** lost with a **2%** limit that never fired | Simulating a crash loop |
| 10 | Bar feed emitted a still-forming bar pre-market | live lookahead | Enumerating clock states |
| 11 | Spread guard written, documented, and never wired in | one-sided books traded through | Dead-code sweep |
| 12 | Partial fills recorded no position at all | 50 real shares, 0 recorded | Asking "what if it half-fills?" |
| 13 | Reconciliation halted on routine partial fills | system stops on normal operation | Following #12 downstream |

Numbers 9–13 are in the sibling execution repo; 1–8 are here.

---

## Three worth the detail

### #4 — The diagnostic could not find an edge I planted on purpose

The information coefficient is the first question you ask a signal: *does this predict anything?*
Mine said no to everything.

I could have shipped that. Instead I generated prices where a hidden score **drives** the next
period's return at a strength I chose, so the true IC was computable in closed form:

```
theoretical IC   0.1275
measured IC      0.0084      ← 15× too low
```

The cause was an off-by-one shared between two modules. Both the engine and the diagnostic
shifted by `execution_lag`, but a position established at *t* already earns the *t → t+1* return
— so lag 1 needed no shift at all. Worse, the standard academic convention was **actively
refused** by an error message claiming it "scores the past." It doesn't.

```
measured IC after the fix    0.1261     (theoretical 0.1275)
```

Both now share one parameter with one meaning, so an IC and a backtest cannot silently disagree
about what was measured.

**The lesson:** a diagnostic that cannot find an edge you planted yourself will report "no edge"
for every real one you ever find — and you will believe it. Test instruments against known
inputs, not against plausibility.

### #8 — The report hid its own largest number

Everything was green. So I fuzzed an *identity* rather than an output:

```python
assert net_returns == gross_returns - costs      # over 3000 randomised backtests
```

**885 failed.** No arithmetic error — the delisting drag was a third term appearing in neither
column. A run dominated by one bankruptcy printed:

```
gross 0.00% − costs 0.01%          ...against an actual return of -40%
```

Every equity number was correct. The *explanation* of them wasn't, and an explanation that can't
be reconciled to the result is worse than none, because it gets believed. The drag is now a
first-class series; re-fuzzed at **0 / 3000**.

**The lesson:** assert on invariants, not just outputs. Outputs need a known-correct answer to
compare against. Identities have to hold for *every* input, so a fuzzer can attack them.

### #9 — A 2% loss limit that let 9.15% through

The most dangerous one, and it needed no exotic input — just a restart.

The runner set the daily-loss baseline from *current* equity, so a mid-day restart re-baselined
it. Each session stayed comfortably inside its own limit:

```
restart 1: equity 98,100  halted=False  cumulative loss 1.90%
restart 2: equity 96,236  halted=False  cumulative loss 3.76%
restart 3: equity 94,408  halted=False  cumulative loss 5.59%
restart 4: equity 92,614  halted=False  cumulative loss 7.39%
restart 5: equity 90,854  halted=False  cumulative loss 9.15%
```

A crash loop or a five-minute cron does this for free — and it's worst exactly when the strategy
is losing, which is when restarts happen.

**The lesson:** safety limits need a lifetime, not just a value. Ask what happens to every piece
of state when the process dies at the worst possible moment.

---

## The techniques, generalised

What actually found things, roughly in order of yield:

**1. Plant a known answer.** Generate data with an effect of computable strength and require the
instrument to recover it. Caught #4. This is the single highest-value technique here and almost
nobody does it — it is much easier to check that a number looks reasonable than that it is right.

**2. Fuzz invariants, not outputs.** `net == gross − costs` must hold for *every* input, so 3000
random cases can attack it. Caught #8. Also used to verify the risk engine never increases a
position (20,000 randomised evaluations, zero violations).

**3. Feed in the answer.** Pass forward returns as the feature. Any honest system must scream.
Mine said "no detectable edge" — caught #5.

**4. Test the warning against the ideal case.** A warning that fires on a perfect signal trains
you to ignore warnings, which is worse than not having it. Caught #6.

**5. Use realistically-shaped data, not test-shaped data.** Every synthetic test passed. Loading a
CRSP-shaped file — negative prices meaning "did not trade", letter-coded returns, recycled tickers
— immediately caught #7.

**6. Simulate the crash.** Not "does it work" but "what if it dies here, and here, and here."
Caught #9 and #12.

**7. Sweep for dead code.** A guard that exists and is never called is worse than no guard,
because you believe you have it. Caught #11.

---

## What the platform does

The bugs are the interesting part, but the thing has a job.

Apex, its predecessor, could only express **absolute** statements — buy when *this* does *that* —
because its core type was one symbol's bar series. Nearly every documented equity anomaly is
**relative**: momentum, value, quality, low-volatility are all statements about one asset
*compared to its peers at the same instant*. You cannot say "long the top decile, short the
bottom" to a backtester that sees one symbol at a time. Not slowly — at all.

So the core type here is a matrix: rows are timestamps, columns are assets.

| module | does |
|---|---|
| `panel.py` | Aligned (time × asset) matrices. **NaN means "not investable"** — so survivorship bias is something you opt into, not something that happens to you. Arrays are read-only, because numpy views alias and one in-place edit silently rewrites history. |
| `causality.py` | **Proves a signal cannot see the future.** Perturb all data after *t*, recompute, assert nothing at or before *t* moved. Catches full-sample z-scoring, global min-max, quantile clipping, centred windows, leaked targets — the bugs that survive code review because they look like the textbook. |
| `ops.py` | Nine composable cross-sectional and time-series operators. Every one row-wise or trailing-window, so full-sample normalisation is hard to reach for by accident. |
| `engine.py` | Vectorised backtest. Turnover measured against **drifted** weights — target-to-target differencing invented **65%** of turnover that never happened. |
| `costs.py` | **Square-root** market impact, not linear. Makes capacity a real constraint: a strategy profitable at $10k and ruinous at $1m looks identical under a linear model. |
| `diagnostics.py` | IC, IC decay, quantile monotonicity, and Grinold's fundamental law — which bounds the Sharpe your breadth can even support. A backtest Sharpe above that ceiling is a bug, not a discovery. |
| `universe.py` | Point-in-time membership with hysteresis. Without a buffer, universe churn alone generated **~105× book/year** of turnover no signal asked for. |
| `walkforward.py` | Folds carry warmup from the training window, so a slow signal is never scored on its own silence. Apex reported a 150-bar signal in 100-bar folds as having *lost* every fold; it had never traded. |
| `significance.py` | Probabilistic and deflated Sharpe against an honest lifetime trial count. |
| `preregistration.py` | Kill criteria fixed and hashed before the run. Loosen a threshold and the fingerprint changes, so the softened claim is visibly a different claim. |
| `data.py` | CRSP loader handling five schema traps that make a naive read silently wrong — PERMNO is the key because tickers get recycled, a negative price means *no trade occurred*, and `DLRET` carries the real delisting return. |

```python
weights = cs_scale(cs_demean(cs_rank(prices.pct_change(252))))
assert_causal(momentum, prices)          # milliseconds
print(diagnose(weights, prices))         # kills most ideas in under a second
```

---

## What it does not claim

**It has not found an edge.** It has not been pointed at real data yet — point-in-time US equity
data with delisted securities is pending. Every number in this README is from synthetic data,
planted signals, or the code's own behaviour.

That is the honest state, and the platform is built to keep it honest. The deflation arithmetic
here is blunt: against a fifty-trial search, an edge with an annualised Sharpe near **1.27** needs
roughly **twenty years** of daily data to distinguish itself from luck. Most ideas will die in
`diagnose()` in under a second. **That is the system working.**

A tool that only tells you yes is a tool that will eventually tell you yes about nothing.

---

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
PYTHONPATH=$PWD .venv/bin/python -m pytest -q
PYTHONPATH=$PWD .venv/bin/python scripts/example_hypothesis.py
```

Python 3.12+, numpy, polars. `WRDS.md` documents the path from data access to a first result.
