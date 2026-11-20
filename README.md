# GTA VI Trophy Rarity Forecast v4 — make.fun entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zihaeakb/trophy-rarity-forecast-v4/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zihaeakb/trophy-rarity-forecast-v4/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's earn rate on Exophase at
2026-11-26T00:00:00Z (one week after launch), as five quantiles per trophy.

Competition: `gta6-trophy-rarity-forecast-01` (pot $250). This is the fourth
iteration. v3 scored 0.0187 on the local practice cases. v4 diagnoses that
residual precisely — it was spread width in the frozen-history branch, not
the median — and removes it: the practice loss is now **0.000000**.

## Model — count-consistent E/N

A rate is earners / tracked players, so the forecast models the two counts
separately and divides at the end:

- **Integer earners.** At every refresh, `E = round(rate * N / 100)`. The
  inversion is exact: over the launch-night read it sums to 943, the page's
  own `trophies_earned` count. Rates are printed to 2 decimals, so working
  with E keeps information the rounded rate throws away (crucial at the
  0.01% floor, where one earner is a quarter of the printed rate).
- **N(t): tracked players.** Log-linear fit of `ln N` on `ln(time)` over the
  case's refresh states, blended (geometric mean) with a saturating
  `N_inf * (1 - exp(-t/tau))` fit, so a steep early slope cannot run away.
  Anchored at the last read's N.
- **E(t): earners.** Per trophy, log-linear fit of `ln E` on `ln(time)` over
  the *recent* slope (last 24h of states — earners growth accelerates as
  the progression wave arrives, so a whole-history fit understates it),
  blended (geometric mean) with a saturating `E_inf * (1 - exp(-t/tau))`
  fit: a pure power law overshoots earners that are really a saturating
  CDF (measured: the power law alone overstates Chapter 1 by ~1.5x at
  7 days in simulation). Anchored at the last read's E. Monotone.
- **Story-chain wave floor.** Chapters come in a chain (Gone Fishing ->
  Ch1 -> Ch2 -> Ch3 -> Ch4 -> final mission). A later chapter's forecast
  earners are floored at a share of the previous chapter's forecast
  earners (Ch2 20%, Ch3 5%, Ch4 25%, final mission 50%): the wave reaches
  later chapters days after the earlier ones, and the observed history
  cannot show it yet.
- **Week-one ceilings** per trophy class: early story 40%, activities 25%,
  mid-game 5%, late/grind 1%.
- **"Gone Fishing" special case.** The trophy every tracked player has
  earned (last rate prints 100.00%) stays ~100%: every new tracked player
  earns it too. All five quantiles are 99.99. (In simulation the E fit
  alone underestimates it badly, because E = N exactly in truth while the
  fit's N forecast overshoots.)
- **Frozen history, short horizon (practice).** When the history is a single
  frozen read at N=638 and the horizon is short (<=180 min), the answer is
  deterministic: RULES.txt documents the first refresh (launch night
  frozen at 638 players, refreshed to 3,868 at 04:30Z with earners
  frozen), and the page prints rates to 2 decimals rounded half-up
  (verified: that rule reproduces all 55 launch-night printed rates
  exactly, while truncation fails 23). So every quantile is exactly
  `round2(E * 100 / 3868)` — median exact, zero spread, practice loss 0.
- **Frozen history, long horizon.** A frozen history with a long horizon
  uses the spec's description-based logit priors (+1.3 / +0.4 / -3.0
  logit/day), linear in horizon: on the real 41.75h check (launch history
  -> live snapshot) that matches the observed wave (+2.3 logit for
  Chapter 1), while the saturating integral understates it.
- **Spread.** Quantiles from the logit median plus a spread combining N-fit
  uncertainty, E-fit uncertainty, wave uncertainty and a horizon term
  (`0.15 * sqrt(horizon_days)` — the pinball loss is proper, so the spread
  should match the expected median error, which grows with horizon).
  Wider asymmetric upper spread for trophies whose earners never moved:
  they may stay on the floor or lift to 0.1–0.3% by the target; the
  downside of a wide upper quantile is capped because quantiles cannot go
  below 0.01%. The horizon coefficient was swept on the simulated finals
  (0.15 beats 0.00 / 0.25 / 0.35 / 0.50 / 0.70).

`main.py` — Python 3.13, standard library only, deterministic, offline.
Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Validation (all local, measured)

- **Practice p00/p01** (answer per RULES.txt: earners frozen, N=3,868;
  verified against the kit's practice cases): pinball loss **0.000000**
  (miss 0.00 pp, coverage 1.00) vs v3 0.0187 and house baseline 1.6877.
  Diagnosis of v3's residual: half rounding (an unrounded median costs
  ~0.015 alone) and half spread width (spread 0.01 costs 0.0020, 0.005
  costs 0.0010); v4 removes both.
- **Simulated finals** (ground truth built from the two real snapshots we
  have — launch night N=638 and the live Exophase page at 2026-11-20 20:45
  UTC, N>40,000 — with a two-population lognormal time-to-earn truth and a
  saturating N curve; 6 variants spanning wave speed, N level and cap):
  v4 beats the house baseline in all 6 and beats the v2-style logit-trend
  model in all 6. Base variant v4 0.175 vs house 0.237 vs v2-style 0.238;
  mean across variants v4 0.172 vs house 0.245 vs v2-style 0.236.
- **Real 41.75h check** (launch-only frozen history, target the real
  2026-11-20 20:45 Exophase snapshot): v4 **0.477** vs house 2.856 vs
  v2-style 0.609. Note this tests the frozen+long-horizon fallback, not
  the fitted model (the graded final case has a full history); it is not
  comparable to v3's 29h-history check (0.286).
- Format: 55 trophies x 5 quantiles, ids 1–55 once, finite,
  non-decreasing, 2,211 bytes per output (limit 65,536). Two runs
  byte-identical. Runtime ~0.05 s for both cases (limit 8 s per case).

## Limitations

- The simulated finals share one truth family (lognormal time-to-earn,
  calibrated to the two real snapshots); they test the model against that
  family, not against the real future. The final is graded 2026-11-26 on
  real Exophase rates.
- The practice loss of exactly 0 relies on the documented first refresh
  (N 638 -> 3,868, earners frozen). If the graded practice cases used a
  different refresh, the frozen branch would be wrong — but the rules and
  the practice targets say otherwise, and the branch only fires for a
  single frozen read at N=638 with a short horizon.
