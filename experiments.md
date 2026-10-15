# Experiment log

Decision rule: keep a change only if it beats the current best on V1 and V2 by > 2 sigma of the paired
bootstrap (`python src/evaluate.py compare A B`) and does not hurt V3 beyond noise. LB is for confirmation only.

Views: V1 = valid, official mask · V2 = valid, all need_prediction rows · V3 = train holdout (splits/holdout.json, 800 seqs), all rows.
Runtime: M2 us/row from `src/check_submission.py`; projected scorer minutes = us/row x 1.55 x 39.4M / 60e6 (limit 60, target <= 48).

| date | name | model | V1 | V2 | V3 | M2 us/row (proj. min) | LB | notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-07 | baseline | starterpack GRU 2x128, raw features | 0.6171 (t0 .5879, t1 .6462) | 0.4402 | 0.4035* | 27.3 (26) | 0.5935 (27m06s) | *baseline likely trained on these sequences, so V3 is not held out for it |

Bootstrap sigma (single score, baseline): V1 0.0060, V2 0.0031, V3 0.0056. Paired-delta sigma is smaller; use `compare`.

## Phase 3: linear diagnostic (models/linear.py; |y|-weighted ridge, 1000 train seqs, every 5th row; not for submission)

| date | name | features | V1 | V2 | V3 | paired delta vs baseline (V1 / V2) |
|---|---|---|---|---|---|---|
| 2026-10-07 | linear_raw | raw | 0.5878 | 0.4211 | 0.3385 | -0.029 (z -9) / -0.019 (z -8) |
| 2026-10-07 | linear_raw+diff | + 1-step diff | 0.5979 | 0.4271 | 0.3458 | -0.019 / -0.013 |
| 2026-10-07 | linear_raw+diff+ema | + x - EMA(10,100,1000) | 0.6333 | 0.4555 | 0.3763 | **+0.016 (z +5.4) / +0.015 (z +7.1)** |
| 2026-10-07 | linear_all | + EMA100(abs diff) | 0.6352 | 0.4563 | 0.3801 | **+0.018 (z +6.1) / +0.016 (z +7.6)** · submitted as submissions/linear_all.zip (9.6 us/row, ~9 min proj.), LB: **0.6089** (10m16s) |

- Multi-scale EMA deviations are the big win (+0.035 V1 over raw+diff). Volatility block adds +0.0019 V1 (z 5.2).
- V3 is lower than the baseline (-0.023) while V1/V2 are higher: the baseline was very likely trained on the holdout
  sequences too, so V3 is in-sample for it. V3 stays valid for comparing our own models.
- Paired-delta sigma: ~0.003 vs a different model family, ~0.0004-0.001 between close variants.

## Phase 4: GRU on engineered features

Model: 672 causal features (src/features.py) -> standardise -> Linear 672->64 -> GRU 64->128 -> GRU 128->128 -> head, + linear skip head on features.
218k params (0.87 MB, kept < ~1 MB so weights stay in the scorer's 1 MB L2 per core). Loss: |y|-weighted MSE on clipped targets.
TBPTT chunks of 1000 steps over full 20k sequences, batch 16, AdamW 1e-3 OneCycle, seed 0. Model selection on holdout[:200] (all rows).

| date | name | train data | V1 | V2 | V3 | delta vs linear_all (V1 / V2 / V3) | M2 us/row (proj. min) | LB |
|---|---|---|---|---|---|---|---|---|
| 2026-10-07 | gru_eng_p64_s2000 | 2000 seqs x 3 epochs (~55 min on M2 CPU) | **0.6634** (t0 .6590, t1 .6678) | 0.4906 | 0.4235 | +0.028 (z 12) / +0.034 (z 19) / +0.043 (z 16) | 38.8 (37; actual 39m21s) | **0.6373** |
| 2026-10-08 | gru_eng_p64_full | all 9807 train seqs x 3 epochs (~4.5 h on M2 CPU) | **0.6714** (t0 .6680, t1 .6748) | 0.5002 | 0.4405 | vs s2000: +0.0080 (z 10) / +0.0096 (z 12) / +0.0171 (z 9) | 41.7 (42; actual 39m42s) | **0.6433** (rank 571/4478) |

Monitor (holdout[:200]) by epoch: s2000 0.3608, 0.4157, 0.4182; full 0.4047, 0.4306, 0.4393 (still rising) (linear_all 0.3718, baseline 0.4021).
Inference note: ONNX Runtime per-call cost dominates (baseline GRU alone 27 us); a numpy GRU step was 28 us vs ORT 41 us for an unprojected 672-input model.

### LB calibration (2026-10-08)
| model | V1 | LB | LB - V1 | V1 gain vs baseline | LB gain vs baseline |
|---|---|---|---|---|---|
| baseline | 0.6171 | 0.5935 | -0.0236 | - | - |
| linear_all | 0.6352 | 0.6089 | -0.0263 | +0.0181 | +0.0154 |
| gru_eng_p64_s2000 | 0.6634 | 0.6373 | -0.0261 | +0.0463 | +0.0438 |
| gru_eng_p64_full | 0.6714 | 0.6433 | -0.0281 | +0.0543 | +0.0498 |
| ens2_full_e3_e5 (numba) | 0.6742 | 0.6459 | -0.0283 | +0.0571 | +0.0524 |
Local gains transfer at ~85-95%; LB ~ V1 - 0.025. Scorer/M2 runtime ratio 1.51-1.63 -> use 1.55.

### 2026-10-08: epochs, seed spread, 2-model average
| name | setup | V1 | V2 | V3 | notes |
|---|---|---|---|---|---|
| gru_eng_p64_full_e5 | full data, 5 epochs, Colab T4 (~100 min) | 0.6715 | 0.4987 | 0.4398 | vs 3 epochs: V1 +0.0001 (z 0.2), V2 -0.0015 (z -2.3), V3 -0.0007 -> **more epochs: no gain** |
| ens_full_e3_e5 | average of full (3ep) + full_e5 | **0.6742** | **0.5042** | **0.4452** | vs single full: **+0.0028 (z 8) / +0.0040 (z 12) / +0.0046 (z 7)** |

Monitor (holdout[:200]) finals: full seed0 0.4393 (Mac), full_e5 0.4356, full seed2 0.4332 (Colab, 3ep); s2000 seed0 0.4182, seed1 0.4214.
Run-to-run spread on the monitor ~0.003-0.006 -> single-run differences below ~0.005 need repeat seeds.
Infra: Colab T4 trains at ~168k rows/s (4.4x M2); numba inference 22 us/model vs ONNX 45 us under the same load.
Mac lesson: one heavy job at a time (8 GB RAM; two jobs thrash into swap).

### 2026-10-08: averaging and numba packaging
| name | setup | V1 | V2 | V3 | runtime | notes |
|---|---|---|---|---|---|---|
| ens3_full | average of full (s0, 3ep) + full_e5 + full_s2 | 0.6751 | 0.5056 | 0.4457 | numba kernel 53.6 us/row -> ~59 min proj. **too slow** | vs single: +0.0037 (z 9) / +0.0054 / +0.0052; vs 2-model: +0.0009 (z 4) |
| ens2_full_e3_e5 (numba) | submissions/ens2_full_e3_e5.zip | **0.6742** (official row-by-row scorer: 0.674232) | 0.5042 | 0.4452 | 41.6 us/row -> ~42 min proj. (actual 42m57s) | **LB 0.6459** (+0.0026 vs 0.6433; local +0.0028) |
| x_base_s4000 | 4000 train seqs, 3 ep, seed 0 (Colab, eval on GPU) | 0.6703 | 0.4957 | 0.4325 | - | only -0.001 V1 vs all 9807 seqs -> more data is not the lever |

numba kernel cost is linear: ~17 us per 218k-param model (19.5 / 36.8 / 53.6 us for K = 1 / 2 / 3).

### 2026-10-08: s4000 experiment queue (Colab, 4000 train seqs, 3 epochs, eval on GPU)
| name | change vs x_base_s4000 | V1 | V2 | V3 | paired delta vs base (V1 / V2 / V3) |
|---|---|---|---|---|---|
| x_base_s4000 | - (seed 0) | 0.6703 | 0.4957 | 0.4325 | - |
| x_pearson_s4000 | Pearson loss (+0.05 wMSE anchor) | 0.6708 | 0.4965 | 0.4307 | +0.0005 (z 0.7) / +0.0007 / -0.0018 -> no gain |
| x_base_s4000_seed1 | seed 1 (different subset + init) | 0.6682 | 0.4940 | 0.4339 | -0.0021 (z -2.7) / -0.0017 (z -2.5) / +0.0014 |

**Seed noise floor:** identical setups differ by ~0.002 on V1 and the paired-bootstrap z (data noise only) reports it as z ~2.7.
So a single-run change needs > ~0.004 on V1/V2 (or a replicate seed) before it counts as real.
Not run yet (cancelled at user's request, ready in runs/queue_s4000_rest.sh minus the seed1 line): x_h160_s4000 (hidden 160), x_p128_s4000 (proj 128).

## 2026-10-09: diagnosis of the plateau (read-only probes on gru_eng_p64_full / ens_full_e3_e5)
| # | probe | result | reading |
|---|---|---|---|
| D1 | model on 96 of its own train seqs vs 96 unseen holdout seqs (holdout[200:296]), all rows | 0.453 vs 0.462 | no overfitting: bias/capacity-limited; explains why data/epochs/seeds/averaging/loss gave <= 0.003 |
| D2 | weighted ridge on [GRU pred + block] (fit 400 holdout seqs all rows, every 10th; scored on valid 0-159 masked rows) | log step +0.0001; x-EMA3000 -0.002; x-EMA10000 -0.003; vol10 -0.001; vol1000 -0.001; x-cummean -0.002; raw -0.002..-0.004; all -0.01..-0.02 | no linear residual signal left in hand-made long-horizon/vol/step features |
| D3 | HistGB stage 2 [pred, log step, raw] on valid masked rows, 2-fold by sequence, per-fold | fold0 0.6634 -> 0.6156; fold1 0.6692 -> 0.6325 | mask-only stacking overfits; mask info must enter through full-data training |
| D4 | mask AUC (HistGB, valid 0-79 train / 80-129 test) | raw 0.866; +diff 0.873; +x-EMA10 0.887; +step 0.887; raw+diff+next-row diff 0.877; abs(y) alone 0.674 | soft function of current state (a3, a2, i0_dp0); not future- or target-driven |
| D5 | V1 by step bucket (valid 0-159, single model) | 99-500 0.633; 500-2k 0.640; 2k-5k 0.640; 5k-10k 0.680; 10k-20k 0.675 | weak first quarter; long context matters (TBPTT = 1000) |
| D6 | V1 moments: between-sequence share of var(y) | 18-20%; corr(seq mean pred, seq mean y) 0.87-0.89; oracle perfect seq offsets 0.706 (upper bound, uses future) | regime component large and mostly captured |
| D7 | V3 vs V2 (both all rows) | 0.445 vs 0.504; seq mean y -0.46 vs -0.34; sum abs(y)/seq 13.8k vs 12.0k | train and valid are different regimes -> final model on train+valid |
| D8 | calibration slope of y on pred (weighted) | masked 0.98/0.96, unmasked 0.92/0.99; pred sd masked 0.80 vs unmasked 0.52 | model already recognises scored moments |

Conclusion: the model is capacity-limited and capacity is capped by the inference budget. Next: capacity ceiling test
(runs/queue_capacity.sh: h384/p192 and h192/p96 on s4000), int8 numba kernel, then zero-inference-cost levers
(long TBPTT, weight EMA, auxiliary targets, mask-proxy weights). Plan: ~/.claude/plans/ok-now-we-got-peppy-bonbon.md.
Reference for s4000 screening = mean of seed 0/1: V1 0.6693, V2 0.4949, V3 0.4332; keep a change at dV1 >= +0.004 and dV2 >= 0.

### 2026-10-09: first Step-1/Step-4 results and the realised-past-target finding
| name | change vs s4000 reference (V1 0.6693 / V2 0.4949 / V3 0.4332) | V1 | V2 | V3 | reading |
|---|---|---|---|---|---|
| x_h192_p96_s4000 | hidden 192, proj 96 (450k params, 2.1x) | 0.6680 | 0.4938 | 0.4312 | -0.0013 / -0.0011 / -0.0020: **2x width gives nothing** |

Non-causal diagnostic: HistGB on future-window summaries reconstructs y_t with weighted corr 0.40 (5 future steps), 0.55 (20),
0.66 (50), 0.73 (100), 0.77 (200), 0.78 (all up to 1000) -> the targets look ~100-300 steps ahead.
Oracle residual probe (D2 setup) with TRUE lagged targets next to the GRU prediction: y[t-300] **+0.0168**,
EMA1000(y[t-300]) +0.0107, cummean +0.0021; y[t-1000] +0.0009. The realised recent past target carries signal the GRU
does not extract. Runs queued: x_lag_s4000 (training-only head predicting y[t-100], y[t-300], weight 0.5), then aux,
mask-weighting, h384. A realistic probe with HistGB-reconstructed y[t-H] is running.

Realistic version of that probe (HistGB reconstructor of y[s] from x[s..s+H], fitted on holdout[300:360]; feature
yhat[t-H] is causal): **+0.0000 for H=100 and H=300**, while the true y[t-100] gives +0.079 and y[t-300] +0.017 on the
same rows. The useful part of past targets is exactly the part not recoverable from the features -> explicit
"realised target" features are a dead end.
**D5 retracted:** on all 1873 valid seqs the step buckets show no trend (99-300 0.701, 300-1k 0.675, 1k-2k 0.651,
2k-3k 0.680, 3k-5k 0.677, 5k-7.5k 0.649, 7.5k-10k 0.671, 10k-15k 0.681, 15k-20k 0.672); the 160-seq pattern was noise.
Nonlinear residual probe (HistGB fitted on 240 holdout seqs, all rows; scored on valid 0-199 masked rows): GRU pred 0.6776,
HistGB[pred] 0.6751, HistGB[pred, raw, diff, vol100] 0.6736 -> no instantaneous nonlinear signal left. Calibration slope
of y on pred by sequence-volatility tercile 0.92-1.00 -> no regime miscalibration.

| name | change vs s4000 reference (V1 0.6693 / V2 0.4949 / V3 0.4332) | V1 | V2 | V3 | reading |
|---|---|---|---|---|---|
| x_lag_s4000 | training-only head predicting y[t-100], y[t-300] (weight 0.5) | **0.6725** | **0.5009** | **0.4366** | **+0.0032 / +0.0060 / +0.0034** (vs seed0: z 4.1 / 8.7 / 4.0); replicate queued |
| x_lag_s4000_ema | same run, weight EMA 0.999 | 0.6726 | 0.5010 | 0.4365 | EMA = final weights (+0.0001): drop EMA |
| x_aux_s4000 | training-only head predicting future i0 changes mean(x[t+1..t+h]) - x[t], h = 10, 100 (weight 0.2); evaluated on the Mac (Colab session lost before mirroring results) | **0.6728** | **0.4991** | **0.4369** | **+0.0035 / +0.0042 / +0.0037** (vs seed0: z 5.4 / 6.1 / 4.3) |

**Auxiliary supervision is the first lever that works:** two independent training-only heads each give about +0.003 V1,
+0.004-0.006 V2, +0.003-0.004 V3 at zero inference cost (heads are dropped at export). Width did not help (h192), so the
gain comes from a richer training signal, not capacity.
Colab free GPU quota ran out at ~20:53 after ~2.5 h (session reclaimed, new T4 'Service Unavailable').
Queued on the Mac CPU instead: x_lagaux_s4000_seed1 (both heads, seed 1 -> replicate + combination in one run).
Pending when Colab is back: x_lag5_s4000 (lags 20/50/100/200/300, weight 1.0), x_maskw_s4000, x_h384_p192_s4000 (low priority).

### 2026-10-10: Colab queue (runs/queue_day2.sh, Mac only orchestrated)
| name | change | V1 | V2 | V3 | paired delta vs same-seed base (V1 / V2 / V3) |
|---|---|---|---|---|---|
| x_lagaux_s4000_seed1 | lag head (100, 300; w 0.5) + future-change head (w 0.2), seed 1 | 0.6719 | 0.4982 | 0.4375 | **+0.0038 (z 7.9) / +0.0042 / +0.0036** vs x_base_s4000_seed1 -> aux gain replicates; the two heads are not clearly additive |
| x_lag5_s4000 | lag head with 20/50/100/200/300, weight 1.0 | 0.6716 | 0.4998 | 0.4373 | +0.0013 / +0.0041 / +0.0048; vs x_lag_s4000 -0.0009 V1 -> more lags/weight is not better |
| x_maskw_s4000 | row weight 0.1 + p(scored) from the mask MLP (fitted on valid half A) | **0.6761** | 0.4994 | 0.4348 | **+0.0058 (z 11) / +0.0037 / +0.0023**; on clean valid half B: **+0.0053 (z 5.9)** / +0.0033 -> biggest single lever; V2 does not drop |
Next: combine mask weighting + both aux heads, then the full-data / train+valid final model.

## 2026-10-15: data reinterpretation (all analysis on a Colab CPU VM)
- **Targets are integers x a per-sequence step**, step in {0.12543, 0.24133, 0.46854, 0.65687} (same for t0 and t1;
  exact to 1e-4). 4 "asset classes" (cls 0..3). Class is predictable from warm-up mean levels with 99.9% accuracy.
  Labels: splits/valid_classes.json, splits/train_classes.json.
- Class mix: train 12/12/33/43 %, holdout 12/13/35/41 %, valid 21/19/23/37 % -> explains V2 (valid) > V3 (train holdout).
- Per-class V1 of ens_full_e3_e5: cls0 0.786 (28% of weight), cls1 0.752 (16%), cls2 0.714 (9%), **cls3 0.422 (46%)**;
  per-class calibration slopes ~1 and class means are right -> the model knows the class; class 3 (coarse ticks,
  target n in about -4..4) is the bottleneck.
- n0+n1 (S) is a small-integer "cost/spread-like" part (cls3: -1 70%, -2 27%), n0-n1 (D) the direction part.
  On cls3 masked rows the model's p0+p1 has corr 0.03 with S (current features reach 0.54 pooled), corr(D, pD) 0.39;
  oracle S lifts cls3 t0 WP 0.392 -> 0.420; oracle D -> 0.98.
- Price-like columns: 11 bid columns update together (69% of rows unchanged); not monotone copies of one tick price
  (Spearman 0.6-0.98, mixed signs); future changes of single columns explain the targets only weakly (corr ~0.16).
Code added (not yet run on GPU): train_gru --dist-weight (cross-entropy head over the 63 clipped target atoms, decoded
as E[|y|y]/E[|y|]); eval_torch --output reg|dist|blend; train_gru --valid-groups.

### 2026-10-15: bundled run A (Colab; orchestrator died with the session, re-evaluated on a fresh GPU session)
| name | recipe | V1 | V2 | V3 | delta vs x_base_s4000 / vs x_maskw_s4000 (V1) |
|---|---|---|---|---|---|
| x_all_s4000 (regression output) | mask weights + lag & future-change heads + distributional head (CE 0.5), s4000 | **0.6789** | **0.5035** | **0.4410** | **+0.0086 (z 13.9) / +0.0028 (z 5.6)**; half B +0.0023; V2 +0.0078 / +0.0041; V3 +0.0085 / +0.0062 |
| x_all_s4000_dist | decoded distributional head as output | 0.6743 | 0.4931 | 0.4313 | worse than regression output |
| x_all_s4000_blend | average of both | 0.6791 | 0.5018 | 0.4396 | = regression on V1, worse V2/V3 -> use regression output |
Monitor (holdout[:200]) 0.4405, highest so far. Full-data run gru_all_full (same recipe) launched on the same session.

### 2026-10-15: full-data model with the bundled recipe (Colab)
| name | V1 | V2 | V3 | delta vs gru_eng_p64_full / vs ens_full_e3_e5 (V1) |
|---|---|---|---|---|
| gru_all_full (regression output) | **0.6810** | **0.5085** | **0.4466** | **+0.0096 (z 14) / +0.0068 (z 11)**; V2 +0.0083 / +0.0042; V3 +0.0061 / +0.0014 |
| gru_all_full_dist / _blend | 0.6774 / 0.6812 | 0.4999 / 0.5071 | 0.4397 / 0.4460 | regression output kept |
Packaged as submissions/gru_all_full (single model, numba, 24.3 us/row -> ~25 min projected; matches torch to 2e-6).
**LB 0.6522** (24m46s, 707 MB): +0.0063 over 0.6459, 93% of the local gain.

### 2026-10-15: capacity with the new recipe (Kaggle job x-all-h384-s4000, 1x T4, ~116k rows/s, ~50 min)
| name | change vs x_all_s4000 | V1 | V2 | V3 | paired delta (V1 / V2 / V3) |
|---|---|---|---|---|---|
| x_all_h384_s4000 | hidden 384 / proj 192 (8x params), same recipe | 0.6784 | 0.5036 | 0.4419 | -0.0004 / +0.0001 / +0.0009 -> **no gain; every class unchanged (c3 0.4242 vs 0.4254)** |
Capacity is ruled out for good (h192 and h384, old and new recipe). No teacher/distillation.
Kaggle note: the CLI's stored OAuth login stops working for API calls; kaggle/push.sh now exports a fresh
KAGGLE_API_TOKEN from `kaggle auth print-access-token`.

### 2026-10-15: Kaggle job wunder-seed1-sum (2x T4)
| name | change | V1 | V2 | V3 | paired delta |
|---|---|---|---|---|---|
| gru_all_full_seed1 | gru_all_full recipe, seed 1 | 0.6815 | 0.5081 | 0.4483 | vs seed 0: +0.0005 / -0.0003 / +0.0017 (consistent) |
| x_all_sum_s4000 | + MSE on predicted t0+t1 (weight 0.5) | 0.6775 | 0.4986 | 0.4386 | vs x_all_s4000: **-0.0014 / -0.0049 / -0.0024**, class 3 0.4229 vs 0.4254 -> drop |
Packaged submissions/ens2_all_full (gru_all_full + seed1, numba, 42.6 us/row -> ~43 min projected, all checks pass).
Local score (Kaggle job wunder-eval-ens2): **V1 0.6829 / V2 0.5114 / V3 0.4506** (+0.0019 V1 over gru_all_full).
**LB 0.6543** (44m52s, 710 MB): +0.0021 over 0.6522; gap V1 - LB 0.0286 (same as before).
Per-class V1 of ens2: c0 0.798, c1 0.765, c2 0.726, **c3 0.430**.

### 2026-10-15: class-3 specialist (Kaggle job wunder-c3ft, 2x T4, ~45 min)
Fine-tune each full model on class-3 train sequences only (4.2k seqs, 2 epochs, lr 3e-4, same recipe; monitor on
class-3 holdout seqs), then route class-3 sequences to the specialist (class known from the warm-up rows).
c3ft_s0 from gru_all_full/last.pt (heads restored), c3ft_s1 from gru_all_full_seed1/best.pt (training heads fresh),
plus eval of their average. Scored with tools/route_moments.py (true class labels) vs ens2_all_full.
Prior is modest: h384 (8x capacity) left class 3 unchanged, so a gain would have to come from removing
interference between classes, not from capacity.

| name | class-3 V1 (base -> specialist) | routed V1 / V2 / V3 | paired delta vs base (V1 / V2 / V3) |
|---|---|---|---|
| route_c3ft_s0 (gru_all_full; c3 -> c3ft_s0) | 0.4276 -> 0.4276 | 0.6810 / 0.5085 / 0.4471 | +0.0000 / +0.0001 / +0.0006 |
| route_c3ft_s1 (seed1; c3 -> c3ft_s1) | 0.4279 -> 0.4296 | 0.6820 / 0.5083 / 0.4482 | +0.0005 V1 |
| route_c3ft_avg (ens2; c3 -> avg of both) | 0.4298 -> 0.4312 | 0.6833 / 0.5116 / 0.4510 | **+0.0004 (z 1.0) / +0.0003 / +0.0005** |
Specialists lose 0.01-0.03 on the other classes (as expected) and gain nothing on class 3 -> **class 3 is not limited
by interference between classes** (nor by capacity, h384). Not deployed.
Per-class affine calibration of ens2 (scale + offset per class, fitted on valid half A from the saved moments):
+0.0003 on half B, -0.0010 V3 -> calibration is already right; dead end.
Code: train_gru --lag-targets now also takes negative H = future targets y[t+|H|] (training-only head), not yet run.

### 2026-10-15: Kaggle job wunder-fut-maskeps (2x T4, s4000, vs x_all_s4000)
| name | change | V1 | V2 | V3 | paired delta (V1 / V2 / V3; V1 half B) |
|---|---|---|---|---|---|
| x_all_fut_s4000 | lag head also predicts future targets y[t+100], y[t+300] | 0.6790 | 0.5049 | 0.4417 | +0.0001 / **+0.0014 (z 3.5)** / +0.0007; half B +0.0003 -> flat on V1 |
| x_all_eps03_s4000 | mask eps 0.1 -> 0.03 | 0.6787 | 0.5014 | 0.4384 | -0.0002 / **-0.0022 (z -6)** / -0.0026 -> drop; mask weighting is saturated |
Decision (screening dropped from here on, GPU is no longer scarce): future targets get one full-data run
(gru_all_fut_full, seed 0, paired with gru_all_full), alongside gru_all_full_seed2 for the ensemble pool.
Job wunder-tva-tbptt (gru_all_tvA, x_all_tbptt4k_s4000) still running.

### 2026-10-15: Kaggle job wunder-tva-tbptt (2x T4)
| name | change | V1 | V2 | V3 | paired delta |
|---|---|---|---|---|---|
| x_all_tbptt4k_s4000 | batch 4 x TBPTT chunk 4000 (same rows per step), s4000 | 0.6794 | 0.5053 | 0.4407 | vs x_all_s4000: +0.0005 (z 1.6) / **+0.0018 (z 5)** / -0.0003; trains 2.75x slower (60k vs 165k rows/s) -> not worth it |
| gru_all_tvA | full recipe on all train + valid half A (937 seqs, +9.5% data) | (0.6839, half A seen) | | 0.4484 | vs gru_all_full on clean valid half B: V1 +0.0007 (sigma 0.0014) / **V2 +0.0037 (z 3.5)** / V3 +0.0018 (z 1.7) |
Valid data helps on unseen valid sequences (V2 half B) and does not hurt the train holdout -> the final models train on
train + all valid. Launched gru_all_tv / gru_all_tv_seed1 (job wunder-tv-final; V1/V2 no longer clean for these,
judge on V3 + the half-B evidence above, LB as confirmation).

### 2026-10-15: Kaggle job wunder-fut-full (2x T4, full data, paired with gru_all_full = seed 0)
| name | change | V1 | V2 | V3 | paired delta (V1 / V2 / V3) |
|---|---|---|---|---|---|
| gru_all_fut_full | lag head also predicts future targets y[t+100], y[t+300] | **0.6824** | 0.5086 | 0.4474 | **+0.0014 (z 4.0)** / +0.0002 / +0.0008; class 3 0.4308 (best single so far) |
| gru_all_full_seed2 | seed 2 | 0.6808 | 0.5064 | 0.4466 | -0.0002 / -0.0021 / +0.0000 |
Seed spread of the full recipe, V1: 0.6810 / 0.6815 / 0.6808 (sd ~0.0004). Future targets: +0.0014 is ~3 sd, but at
s4000 the gain showed on V2 instead of V1 -> needs a seed-1 replicate before adoption.

### 2026-10-15: next round (user request: aux learning, hidden-state analysis, CNN-GRU)
Job wunder-probe-cnn: tools/probe_hidden.py on gru_all_full and x_base_s4000 (no aux heads) -- decodability of past /
future targets, regimes and volatility from the top GRU state vs from the inputs, oracle value of each, residual
probes (is the head underfit?), class accuracy; and gru_all_cnn = full recipe + causal CNN front-end on the raw block
(--conv 32 4 1 4 16: 3 dilated layers, 64-row receptive field, +23k params, ~+16% per-row cost).
