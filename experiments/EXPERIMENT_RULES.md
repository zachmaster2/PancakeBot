# Experiment Rules & Backtesting Constraints

**Every experiment and backtest MUST account for all constraints below.**
Violating any of these will produce misleading results that don't translate to live performance.

---

## 1. Market Impact (Payout Dilution)

**The #1 mistake we keep making.**

Our bet gets added to our side's pool, which dilutes the payout multiplier.
The payout we see before betting is NOT the payout we get.

```
pre_bet_payout  = pool_total * 0.97 / our_side
post_bet_payout = (pool_total + bet_size) * 0.97 / (our_side + bet_size)
```

### Hard numbers (from 20k-round data):
| Pool stat     | P10  | P25  | P50  | P75  | P90  |
|---------------|------|------|------|------|------|
| Pool total    | 1.34 | 1.81 | 2.45 | 3.29 | 4.29 |
| Our side      | 0.60 | 0.82 | 1.11 | 1.48 | —    |

- 33% of rounds have pool < 2 BNB total
- Median our-side is only 1.11 BNB

### Impact by bet size:
| Bet Size | Avg Pre Payout | Avg Post Payout | Drop  | Net EV/bet |
|----------|---------------|-----------------|-------|------------|
| 0.25     | 2.03          | 1.85            | -8.9% | +0.011     |
| 0.50     | 2.03          | 1.73            | -15%  | -0.011     |
| 1.00     | 2.03          | 1.57            | -23%  | -0.112     |
| 2.00     | 2.03          | 1.40            | -31%  | -0.420     |

**Rule: ALL backtests must compute post-bet payout and use it for settlement.**
Never evaluate a sizing strategy using pre-bet payout.

**Rule: Maximum practical bet is ~0.25-0.50 BNB for a typical pool.**
Bet sizing must be a function of pool size, not just conviction.

### How to compute correctly:
```python
# WRONG - what we used to do
payout = pool_total * 0.97 / our_side
profit_if_win = bet_size * payout - bet_size

# RIGHT - what we must do  
post_payout = (pool_total + bet_size) * 0.97 / (our_side + bet_size)
profit_if_win = bet_size * post_payout - bet_size
```

### Pool-relative sizing guideline:
To keep payout degradation under 10%, bet should be < 10% of our_side:
```python
max_safe_bet = our_side * 0.10
```

---

## 2. Kelly Criterion Does NOT Apply Naively

Kelly assumes fixed odds. In our case:
- Odds change as a function of bet size (see #1)
- Pool size varies 10x across rounds
- WR varies by signal tier, hour, and regime

**Rule: Never quote raw Kelly fractions as actionable bet sizes.**
Always compute the "pool-constrained Kelly" which optimizes:
```
f* = argmax_f [ WR * log(1 + f * post_payout(f) - f) + (1-WR) * log(1 - f) ]
```
where post_payout(f) depends on f through market impact.

---

## 3. Gas Costs

- Bet gas: 0.0002 BNB
- Claim gas: 0.00025 BNB
- Total round-trip: ~0.00045 BNB

Gas is negligible for bets > 0.10 BNB. But at scale (6000+ bets), it adds up:
- 6000 bets * 0.00045 = 2.7 BNB in gas

**Rule: Always include gas in backtest PnL calculations.**

---

## 4. Settlement Must Match On-Chain Exactly

The backtest settlement function must be identical to on-chain:
- Treasury fee: 3% 
- Pool amounts use ALL bets in the round (no timestamp filter) — this is what the contract uses
- Our bet added to the pool before computing payout (market impact)
- Failed rounds: refund bet minus claim gas
- Loss: bet is gone, no claim gas

**Two different pool views for two different purposes:**
- **Decision time** (signal, sizing, payout filter): use bets with `created_at <= lock_at` — this is what we can see when placing our bet
- **Settlement time** (computing actual profit/loss): use ALL bets — this is what the contract does

**Rule: Use the shared `settle()` function. Never reimplement settlement.**

---

## 5. Data Integrity

- Closed rounds, BNB klines, and BTC klines must be perfectly aligned 1:1:1
- Every stored round MUST have matching klines in BOTH stores
- Never delete valid data to fix alignment — always fetch missing data first
- If data genuinely doesn't exist (OKX retention limit), trim the round
- Sync must be resumable/interruptible (staging files, atomic writes)

**Rule: Before running any experiment, verify store alignment:**
```python
assert round_epochs == bnb_epochs == btc_epochs
```

---

## 6. Regime Awareness

Strategy performance varies dramatically by time period:
| Period          | WR    | PnL/1k |
|-----------------|-------|--------|
| Oct-Nov 2025    | 53.5% | +0.89  |
| Nov-Dec 2025    | 51.3% | -0.98  |
| Jan-Feb 2026    | 55.3% | +2.27  |
| Feb-Mar 2026    | 56.2% | +3.48  |

**Rule: Always report results across multiple time windows, not just aggregate.**
A strategy that only works in recent data may be overfit.

**Rule: Use 8-segment analysis minimum. All segments should be examined.**

---

## 7. The Payout Asymmetry Trap

Momentum signals tend to align with the crowd (other bettors see the same price move).
This means:
- Our signal direction usually has the LARGER pool (lower payout)
- Contrarian bets get HIGHER payouts but with lower WR
- A 57% WR at 1.9x payout is worth less than a 52% WR at 3.0x payout

**Rule: Always evaluate strategies by EV (WR * payout - 1), not WR alone.**

---

## 8. Simulation Size vs Stat Significance

- 50 bets: WR confidence interval = +/- 14%. Useless.
- 200 bets: WR CI = +/- 7%. Marginal.
- 1000 bets: WR CI = +/- 3%. Acceptable.
- 5000 bets: WR CI = +/- 1.4%. Good.

**Rule: Don't make decisions based on subgroups with < 200 bets.**
Any "72% WR" on 68 bets has a CI of [60%, 82%] — could easily be 55% in reality.

---

## 9. Backtest Implementation Checklist

Every new experiment script must:

- [ ] Use post-bet payout (market impact) for settlement
- [ ] Use the shared `settle()` function, not a reimplementation
- [ ] Include gas costs (bet + claim)
- [ ] Report multi-segment breakdown (not just aggregate PnL)
- [ ] Report per-1k-round PnL for consistency comparison
- [ ] Cap bet size relative to pool (not just absolute cap)
- [ ] Verify data alignment before running
- [ ] Test across at least 3 different window sizes
- [ ] Flag any subgroup analysis with < 200 samples
- [ ] Compute post-bet payout when evaluating sizing changes

---

## 10. Common Mistakes Log

| Date       | Mistake                                          | Consequence                         |
|------------|--------------------------------------------------|-------------------------------------|
| 2026-04-11 | Kelly analysis used pre-bet payout               | Suggested 7.5 BNB bets that would lose money |
| 2026-04-11 | Evaluated "72% WR" on 68 bets                   | Likely just 55-60% with noise       |
| 2026-04-11 | Sync collected all results in memory             | Not resumable, lost progress on kill |
| 2026-04-11 | Proposed deleting rounds without klines          | Would destroy valid historical data |
| 2026-04-11 | BNB/BTC kline counts differed by 55              | Stores weren't aligned, breaks invariant |
| 2026-04-11 | 4 concurrent workers without rate-limit backoff  | Mass 429 errors from OKX            |
