# Data findings (Phase 2, 2026-10-07)

Samples: 100 random train sequences (non-holdout), first 200 valid sequences; mask AUC used valid 0-149 (train) vs 1000-1059 (test).

## Features
- No NaN/inf in features or targets.
- Already Gaussianised (looks like a rank -> normal quantile transform): most columns have p1/p99 = -2.326/+2.326
  (normal 1%/99% quantiles); `a5`, `a6`, `a7` have a floor at -5.199 (normal quantile of 1e-7), and about 43% of `a7` rows sit there.
  `a7` has only ~87 distinct values (categorical-like).
- Group-level mean ~0, std ~0.7-1.0. No extra normalisation needed; the baseline feeds raw features.
- Individual price-like columns have very different means (-1.8 ... +2.0) and small within-sample std (0.06-0.33): levels are
  roughly stable over time, so changes (diffs) are probably more informative than levels -> test in Phase 3.

## Targets
- std ~1.0, mean ~-0.26, 22% exactly zero (zero metric weight), ~3% beyond the +/-2 clip, heavy right tail (p99.9 ~ 5.3).
- corr(t0, t1) = -0.73 (strongly negative): a shared representation is natural.
- Autocorrelation: lag 1 0.97, lag 10 0.88, lag 100 0.45, lag 1000 ~0 -> forward-looking windows of a few hundred steps.
- Per-sequence target mean varies (std ~0.2-0.25 across sequences): sequence-level offsets matter for the global metric.

## Scoring mask (valid)
- 9.35% of rows scored; uniform over steps; per-sequence fraction varies widely (p5 1.7%, p50 7.4%, p95 25%).
- Short bursts: ~480 runs per sequence, run length median 2, p95 13.
- Scored rows: higher target std (1.00 vs 0.87), similar zero fraction.
- **Mask is largely predictable from the current row:** gradient-boosting AUC 0.87 (0.81 from a0-a7 alone).
  Most shifted features on scored rows: a3 (+0.66 sd), a2 (+0.48), a4 (-0.37), i0_dp0 (+0.29).
- Baseline: V1 (mask) 0.617, V2 (all rows) 0.440, LB 0.5935. The LB is far closer to V1 than V2, so the test mask
  very likely selects the same kind of moments as the valid mask.

## Implications
1. V1 is the primary view; V2 still guards against degrading all-rows quality (FAQ warns about mask overfitting).
2. A mask-proxy model (AUC 0.87) can give a pseudo-mask for the train holdout -> "V3m", a second V1-like view.
3. Candidate improvement: weight the training loss towards likely-scored rows (proxy probability).
4. Features look pre-normalised; try diffs/EMAs of levels in the linear diagnostic.
