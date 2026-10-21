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

### 2026-10-15: final-type models on train + all valid (Kaggle job wunder-tv-final)
| name | V1 (seen) | V2 (seen) | V3 | V3 delta vs same-seed train-only model |
|---|---|---|---|---|
| gru_all_tv | 0.6871 | 0.5219 | 0.4477 | +0.0011 (z 1.1) |
| gru_all_tv_seed1 | 0.6881 | 0.5216 | 0.4492 | +0.0009 (z 0.9) |
V1 rises only ~0.006 although valid is in training (no memorisation, consistent with D1). Packaged
submissions/ens2_all_tv (both, numba, 43.4 us/row -> ~44 min projected, all checks pass). Expected LB: ens2_all_full
0.6543 plus the valid-data gain (half-B V2 +0.0037 at +9.5% data; V1 effect not measurable locally).
Next job wunder-aux1: gru_all_fut_full_seed1 (replicate), gru_all_fut_lw1_full (lag/future-target weight 1.0).

### 2026-10-15: hidden-state probes (tools/probe_hidden.py, valid seqs 0-599, every 20th row, 2-fold by sequence)
Held-out corr of a ridge decode of each quantity from the top GRU state h vs from the 672 inputs at t:
| quantity | gru_all_full h | x_base_s4000 h (no aux heads) | inputs | oracle WP gain if known exactly (gru_all_full) |
|---|---|---|---|---|
| y[t-1] / y[t-10] / y[t-30] | 0.43 / 0.46 / 0.56 | 0.38 / 0.41 / 0.47 | 0.40 / 0.43 / 0.47 | +0.30 / +0.26 / +0.19 |
| y[t-100] / y[t-300] (lag-head targets) | **0.72 / 0.80** | 0.48 / 0.35 | 0.48 / 0.33 | +0.065 / +0.004 |
| y[t-1000] / y[t-3000] | 0.22 / 0.18 | 0.22 / 0.17 | 0.23 / 0.18 | -0.010 / -0.010 |
| y[t+10] / y[t+100] / y[t+300] | 0.30 / 0.20 / 0.18 | 0.27 / 0.18 / 0.17 | 0.28 / 0.18 / 0.17 | - |
| mean y over last 100 / 1000 | 0.63 / 0.69 | 0.52 / 0.53 | 0.53 / 0.52 | +0.19 / +0.036 |
| future mean abs(y) over 100 / abs(y) | 0.25 / 0.37 | 0.22 / 0.32 | 0.24 / 0.33 | +0.051 / - |
| sequence mean of y (regime) | 0.77 | 0.74 | 0.76 | +0.005 |
| asset class (accuracy) | 0.9998 | 0.979 | - | - |
Residual probes on h (ridge, MLP; also with raw inputs), WP gain on held-out: -0.011 to -0.024 for both models.
Reading:
- **The readout is not underfit**: no extra signal for the current target is extractable from h. Gains must come from
  changing what h encodes (objectives / inputs), not from a bigger head.
- Aux heads reshape the state strongly: the lag head lifts the y[t-100] / y[t-300] decode from 0.48 / 0.35 to 0.72 / 0.80.
  Without explicit targets the GRU state holds little beyond the engineered inputs (decode h ~= inputs).
- Past-target value decays with the lag: large for k <= 30 (the target windows overlap with y[t]; mostly unrealised
  future, so unknowable), +0.065 at 100, ~0 from 300 on. Our lag-300 target carries almost no direct value; the 30-100
  zone (and the past-100 mean, +0.19, decode 0.63) is where the value is and where the state is weaker.
- The regime (sequence mean) and the class are already known; future volatility is barely decodable (0.25) by either.

### 2026-10-15: CNN-GRU (gru_all_cnn, full recipe + --conv 32 4 1 4 16, seed 0)
V1 0.6804 / V2 0.5069 / V3 0.4461; vs gru_all_full -0.0006 / **-0.0016 (z -3.1)** / -0.0005 -> learned 64-row filters on
the raw inputs find nothing the engineered features miss (matches D2). Dropped; no kernel work.

### Next: job wunder-aux2 (both paired with gru_all_fut_full, seed 0)
gru_all_fut_mid_full (lags 30 60 100 300 + future 100 300) and gru_all_fut_pmean_full (+ past-mean targets over 100 / 1000).

### 2026-10-16: aux-target round (jobs wunder-aux1, wunder-aux2; full data)
| name | change | V1 | V2 | V3 | paired delta vs reference (V1 / V2 / V3) |
|---|---|---|---|---|---|
| gru_all_fut_full_seed1 | future targets, seed 1 (replicate) | 0.6817 | 0.5082 | 0.4484 | vs gru_all_full_seed1: +0.0002 / +0.0001 / +0.0001 -> **the seed-0 +0.0014 did not replicate** |
| gru_all_fut_lw1_full | lag/future weight 1.0 | 0.6808 | 0.5080 | 0.4478 | vs gru_all_fut_full: -0.0016 / -0.0006 / +0.0004 |
| gru_all_fut_mid_full | + lags 30, 60 | 0.6812 | 0.5085 | 0.4496 | -0.0012 / -0.0001 / +0.0022 |
| gru_all_fut_pmean_full | + past-target means over 100 / 1000 | 0.6808 | 0.5077 | 0.4493 | -0.0016 / -0.0009 / +0.0019 |
**Run-to-run spread**: the 9 full-data runs of this recipe family (3 seeds, 5 aux variants, CNN) span V1 0.6804-0.6824,
sd 0.0006 (V2 0.0008, V3 0.0012). The paired bootstrap sigma (~0.0004) only covers sequence sampling, not training
noise, so single-run z-scores overstate significance; the seed-0 future-target "gain" was a high draw and the
negative deltas above are mostly regression from it. All aux variants are within noise of the base recipe ->
**aux-target tuning is saturated**. Decision rule from here: a change must beat the multi-seed mean by >= 2x the run
spread (~+0.0012 V1) in two seeds.

### 2026-10-16: LSTM and pair diversity (job wunder-lstm-div)
| name | V1 | V2 | V3 | note |
|---|---|---|---|---|
| gru_all_lstm (LSTM cells, same recipe, seed 0) | 0.6810 | 0.5069 | 0.4454 | vs gru_all_full +0.0000 / -0.0015 / -0.0012 |
| pair_s0_lstm (gru_all_full + LSTM) | 0.6828 | 0.5110 | 0.4494 | vs ens2_all_full (seed pair): -0.0001 / -0.0004 / -0.0012 |
| pair_s0_fut (gru_all_full + future-target GRU) | 0.6832 | 0.5112 | 0.4497 | vs ens2_all_full: +0.0003 / -0.0002 / -0.0009 |
Pair gain over the mean of its two singles: seed pair +0.0017, GRU+LSTM +0.0018, GRU+fut-GRU +0.0015 -> **a different
cell type or aux recipe averages exactly like another seed.** The diversity premise fails: no LSTM kernel work.
(pair_s0_s2 / pair_s0_cnn hit CUDA OOM running concurrently on one GPU; not needed for the decision.)

### 2026-10-16: LB of ens2_all_tv (train + all valid, 2 numba models)
**LB 0.6562** (43m36s, 714 MB): +0.0019 over ens2_all_full (0.6543) -> the valid-data gain transfers to test. Current best.

### 2026-10-16: inference profile (M2, one valid sequence, best of 3)
Kernel us/row: K=1 22.0, K=2 37.6, K=3 54.6 -> ~16.3 per model + ~5.5 fixed (features); solution.predict adds ~4.
Fix: _matvec wrote np.dot(W, x) + b into temporaries; now sgemv into the output buffer + bias loop (outputs bit-identical
over 20k rows). New: K=1 20.4, K=2 34.8, K=3 50.6 (~15 per model). ens2_all_tv: 40.1 us/row -> ~41 min projected (was 44).
A 3-model ensemble now projects to ~56 min (real/projected has been 0.97-1.04) -> possible but tight.
Job wunder-ens3: gru_all_tv_seed2 (third train+valid model) and the local score of ens3_all_full (seeds 0/1/2) to
measure the current recipe's 3rd-model gain (old recipe: +0.0009 V1).

### 2026-10-16: third model (job wunder-ens3)
| name | V1 | V2 | V3 | note |
|---|---|---|---|---|
| ens3_all_full (seeds 0/1/2, train only) | 0.6833 | 0.5119 | 0.4513 | vs ens2_all_full: **+0.0004 (z 3.6) / +0.0005 / +0.0007** -> small but consistent |
| gru_all_tv_seed2 (train + valid) | (0.6867 seen) | (0.5221 seen) | 0.4476 | V3 vs gru_all_full_seed2 +0.0010 (same as seeds 0/1) |
Packaged submissions/ens3_all_tv (3 train+valid models): all correctness checks pass; 53.7 us/row -> **~55 min projected**
(fails only our own 48-min safety threshold; real/projected has been 0.99-1.04 -> 54-57 min of the 60-min limit).
Expected LB ~ 0.6562 + 0.0004.

### 2026-10-16: external EDA review + decoded-path gate (job wunder-gate-paths, tools/gate_paths.py)
Verified on 48 valid seqs (12 per class): targets change only on i0 price-shape transitions (100%; 52-65% of
transitions change a target); shapes per seq cls3 8 / cls2 27 / cls1 295 / cls0 1034; dp1-3 and dv1-3 are functions of
dp0 / dv0; a6 ~ time since the last i0 transition (Spearman 0.75-0.92); cls3 transition rows scored 2.8% vs 13.8% (no
effect in other classes) -- and the mask MLP already predicts it (0.059 vs 0.183). Transition -> tick table accuracy
(fit / test halves): cls3 98.5%, cls2 82%, cls1 81%, cls0 85% (coverage 89-100%).
Gate: tables from 3000 train seqs (i0 decode 396k types, i1 encodings 649k; coverage valid 98% / 91%), causal window
sums (10/100/1000) of decoded i0 moves, i0 and i1 lead-lag target encodings (H 100), timers and transition counts.
Per-class |y|-weighted ridge stacked on the ens2_all_full (train-only) predictions, vs ridge on the predictions alone:
| setting | +i0 | +i1 | +clock | +all | cls3 (+all) |
|---|---|---|---|---|---|
| G1 holdout -> valid, V1 | -0.0006 | +0.0001 | -0.0008 | -0.0010 | -0.0014 |
| G2 half A -> B, V1 | +0.0002 | +0.0002 | -0.0007 | -0.0008 | -0.0034 |
| G2 half B -> A, V1 | +0.0000 | +0.0005 | -0.0009 | -0.0010 | -0.0052 |
V2 likewise <= +0.0002. HistGB on everything (G1): V1 0.6792 vs 0.6829 pred. **Gate failed** (needed +0.002): decoded
i0 paths, i1 lead-lag and event-clock features add nothing on top of the GRU ensemble. Decoded-path / event-clock GRU
(#3) not built. Search closed; remaining time goes to final selection.

### 2026-10-16: LB attempt of ens3_all_tv
**SCORE: TIMEOUT** at 61m56s (projected ~55 min from 53.7 us/row x 1.55). Real/projected >= 1.13 for 3 models vs 0.99-1.04
for 2 models -> the scorer slowdown grows with the number of models (weights 3 x ~0.86 MB spill out of cache), so the
1.55 factor is only valid up to 2 models. The 48-min check threshold in check_submission was right; keep it.
Final picks: ens2_all_tv (LB 0.6562, 43m36s) primary; ens2_all_full (LB 0.6543) backup.

### 2026-10-16: Discord "frozen random filter bank" baseline (job wunder-reservoir, tools/reservoir.py)
Recipe as posted: P = randn(112, C)/sqrt(112) (seed 0), causal EMA per channel, half-lives geomspace(1, 3000, C),
|y|-weighted ridge on 2000 train seqs (every 2nd row), ridge strength picked on the holdout.
| model | V1 | V2 | V3 |
|---|---|---|---|
| reservoir C=1024 | 0.6335 | 0.4586 | 0.3795 |
| reservoir C=512 | 0.6283 | 0.4551 | 0.3759 |
| (ref) our 672-feature ridge linear_all | 0.6352 | 0.4563 | 0.3801 |
| (ref) ens2_all_full | 0.6829 | 0.5114 | 0.4506 |
The claimed ~0.655 does not reproduce on our V1 (0.634 = our old linear model; LB-equivalent ~0.605). Blending it into
ens2_all_full only loses (w 0.05: -0.0001, 0.1: -0.0005, 0.2: -0.0019); w chosen on half A = 0. Dropped.
Kernel bench on the Kaggle CPU (bench_kaggle_cpu): K=1 44.6, K=2 99.5, K=3 146.0 us/row -> superlinear already at K=2
there, unlike the scorer (fine at K=2, timeout at K=3) -> not a usable proxy for scorer runtime.

### 2026-10-16: model soup, attempt 1 (job wunder-soup-tm)
Four trainings on 2 GPUs (two per T4) -> ft1 and gru_all_tvA_tm hit CUDA OOM; one training per GPU from now on.
| name | change | V1 | V2 | V3 |
|---|---|---|---|---|
| ft2 | gru_all_full/last.pt fine-tuned 1 epoch, lr 3e-4, seed 12 | 0.6815 | 0.5092 | 0.4477 |
| ft3 | same, seed 13 | 0.6806 | 0.5079 | 0.4471 |
(base gru_all_full 0.6810 / 0.5085 / 0.4466: single fine-tunes are within noise, as expected.)
Relaunched: wunder-soup2 (ft1, gru_all_tvA_tm, soup_ft3, ens_ft3) and wunder-soup23 (soup / ensemble of ft2 + ft3).

### 2026-10-16: 2-way soup (job wunder-soup23)
| name | V1 | V2 | V3 | vs gru_all_full (V1 / V2 / V3) |
|---|---|---|---|---|
| soup_ft23 (weights of ft2 + ft3 averaged; 1 model) | 0.6814 | 0.5091 | 0.4481 | +0.0004 / +0.0006 / +0.0015 |
| ens_ft23 (predictions of ft2 + ft3 averaged; 2 models) | 0.6815 | 0.5093 | 0.4483 | +0.0005 / +0.0009 / +0.0017 |
**The soup works** (soup - ensemble: -0.0001 V1, the fine-tunes stay linearly connected), but 1-epoch lr-3e-4
fine-tunes are too similar: their ensemble adds only +0.0005 (a seed pair adds +0.0017). Next: lr 1e-3 fine-tunes
(job wunder-souphi) for more diversity, checking that the soup still tracks the ensemble.

### 2026-10-16: soup and true-mask results (jobs wunder-soup2, wunder-souphi)
| name | V1 | V2 | V3 | V1 vs gru_all_full |
|---|---|---|---|---|
| ft1 (lr 3e-4 fine-tune, seed 11) | 0.6810 | 0.5087 | 0.4470 | +0.0000 |
| soup_ft3 (weights of ft1-3 averaged; 1 model) | 0.6814 | 0.5093 | 0.4482 | +0.0004 (z 1.9) |
| ens_ft3 (predictions of ft1-3; 3 models) | 0.6816 | 0.5097 | 0.4484 | +0.0006 |
| fthi1 / fthi2 (lr 1e-3 fine-tunes) | 0.6799 / 0.6795 | 0.5064 / 0.5063 | 0.4470 / 0.4461 | -0.0011 / -0.0015 |
| soup_fthi / ens_fthi | 0.6803 / 0.6808 | 0.5074 / 0.5085 | 0.4483 / 0.4488 | -0.0007 / -0.0002 |
Soup verdict: weight averaging matches prediction averaging at lr 3e-4, but fine-tunes from one checkpoint are too
similar (+0.0004); at lr 1e-3 they drift and get worse. Ceiling ~+0.0004 per model, under the +0.0012 bar -> dropped.
True mask on valid rows (gru_all_tvA_tm vs gru_all_tvA, both train + valid A, clean half B): V1 +0.0009 (sigma
0.0013), V2 +0.0001, V3 -0.0008 -> noise; not adopted.

### 2026-10-16: bf16 weight storage (fast_gru pack/export --bf16)
Five large matrices stored as bfloat16 bits (2-model weights 1.75 MB -> 0.89 MB), widened to fp32 in explicit numba
loops; matches the torch bf16 simulation to 9e-5. Speed (us/row, K = 1 / 2 / 3):
| machine | fp32 (BLAS sgemv) | bf16 (loops) |
|---|---|---|
| M2 (2 models) | 38.5 | 56.5 |
| Kaggle Intel Xeon 2.0 GHz, AVX-512 F/BW/DQ/VL (no VNNI / BF16) | 62.1 / 115.7 / 167.5 | 67.9 / 127.4 / 189.5 |
bf16 is 10-13% slower even on an AVX-512 x86 CPU: halving the weight bytes does not pay for the widening + loop
overhead against BLAS, so memory bandwidth is not the limit there. Not uploaded; the numba speed-up route is closed.
A C/int8-VNNI kernel would be the only remaining route (1-2 days, scorer CPU flags unverified, ~+0.0004) -> not pursued.
Accuracy of reduced-precision weights (torch simulation, ens2_all_full; all matrices incl. heads rounded):
bf16 V1 0.6829 / V2 0.5114 / V3 0.4506 = unchanged; int8 (per-row symmetric) V1 0.6822 / V2 0.5100 / V3 0.4488 =
-0.0007 / -0.0014 / -0.0018. bf16 is free accuracy-wise; int8 costs more than a third model would add.

## 2026-10-21: autonomous round 1 (new hypotheses; reference = gru_all_full seeds 0/1/2, V1 mean 0.6811, run sd 0.0006)
**H1 privileged-teacher distillation.** A teacher GRU also sees the future (mean(x[t+1..t+h]) - x[t], h = 1..1000, all
112 columns; `--future-inputs`), so its prediction is a much less noisy estimate of the target. The student (deployable,
unchanged architecture) adds a |y|-weighted MSE to the clipped teacher prediction (`--teacher`, weight 1). If the teacher
is calibrated, E[|y| teacher | past] = E[|y| y | past], so the student's optimum is unchanged while label noise drops
-- equivalent to more data (s4000 -> full gave +0.0021 V1). Teachers are cross-fitted on train halves (group % 2) so
no student sequence gets an in-sample teacher prediction. Zero inference cost. Job wunder-teach: teacher_h0/h1.
**H2 valid emphasis.** Valid data transferred to LB at ~5x the per-sequence value of train data (likely closer to the
test period/mix). gru_all_tvA_vr3 = train + valid A repeated 3x; gru_all_tvA_vft = gru_all_tvA fine-tuned on valid A
only (2 epochs, lr 3e-4). Both judged against gru_all_tvA on clean valid half B. Job wunder-validw.

### 2026-10-21: privileged teachers (job wunder-teach, 2x T4, ~70 min)
| name | data | monitor WP (holdout[:200], all rows) by epoch |
|---|---|---|
| teacher_h0 | train groups % 2 == 0, + future means h 1..1000 (1456 inputs, proj 128) | 0.792 / 0.823 / **0.856** |
| teacher_h1 | train groups % 2 == 1 | 0.787 / 0.829 / **0.847** |
(student models: ~0.445 on the same view.) The future-window inputs explain far more of the target than any causal
model, so the teacher signal is much cleaner than y. Teacher V1/V2/V3 eval hit CUDA OOM at batch 32 (not needed).
Smoke student (32 seqs, both teachers, full recipe) ran at 137k rows/s vs ~165k without teacher.
Launched job wunder-distil: distil_w1_full / distil_w3_full (full recipe + teacher weight 1 / 3, seed 0; paired with
gru_all_full seed 0).

### 2026-10-21: valid emphasis (job wunder-validw), paired vs gru_all_tvA (seed 0) on clean valid half B
| name | change | V1 half B | V2 half B | V3 |
|---|---|---|---|---|
| gru_all_tvA_vr3 | train + valid A repeated 3x (0.6886 / 0.5244 / 0.4479 overall, A seen) | **+0.0016** (sigma 0.0019) | **+0.0031 (z 2.3)** | -0.0005 |
| gru_all_tvA_vft | gru_all_tvA fine-tuned on valid A only, 2 ep lr 3e-4 | -0.0007 | +0.0009 | **-0.0041 (z -4.1)** |
Fine-tuning on valid alone forgets the train distribution -> dropped. 3x valid weighting points the same way as the
original valid-data gain (tvA vs full: V2 half B +0.0037); single seed, so a paired seed-1 replicate is running
(job wunder-validw2: gru_all_tvA_seed1 vs gru_all_tvA_vr3_seed1).

### 2026-10-21: wunder-distil invalid (bug)
The distillation term was overwritten by the aux-loss line (`total = loss + aux`), so distil_w1_full / distil_w3_full
trained bit-identically to gru_all_full (same monitor curve, zero paired delta) -- which also confirms full training
determinism (same seed => identical model). Fixed (commit "Fix: distillation term..."); relaunched as job wunder-distil2.

### 2026-10-21: distillation from privileged teachers (job wunder-distil2, fixed), paired vs gru_all_full (seed 0)
| name | V1 | V2 | V3 | paired delta (V1 / V2 / V3) |
|---|---|---|---|---|
| distil_w1_full | 0.6814 | 0.5094 | 0.4476 | +0.0004 (z 0.8) / +0.0010 (z 2.0) / +0.0011 (z 1.2) |
| distil_w3_full | 0.6810 | 0.5088 | 0.4475 | +0.0000 / +0.0003 / +0.0009 |
Training is deterministic, so these deltas are the pure effect of distillation, yet they are within the run spread
(sd 0.0006 V1). A teacher at holdout WP 0.85 (vs 0.44) does not make the student better: the student is not limited by
label noise (consistent with D1: no overfitting; more data helps little). **H1 rejected** -- cleaner targets do not
break the plateau; not pursued further (w1 at most +0.0004 V1).

### 2026-10-21: valid emphasis replicate (job wunder-validw2), clean valid half B
| pair (seed 1) | V1 half B | V2 half B | V3 |
|---|---|---|---|
| gru_all_tvA_vr3_seed1 vs gru_all_tvA_seed1 | **+0.0015 (z 2.3)** | **+0.0026 (z 3.5)** | +0.0005 |
| (seed 0, earlier) gru_all_tvA_vr3 vs gru_all_tvA | +0.0016 (z 0.8) | +0.0031 (z 2.3) | -0.0005 |
| (seed noise) gru_all_tvA_seed1 vs gru_all_tvA | +0.0003 | -0.0003 | +0.0011 |
**H2 confirmed in two seeds:** repeating the valid sequences 3x adds ~+0.0015 V1 / +0.0028 V2 on unseen valid
sequences without hurting the train holdout. Valid-like data is worth more than train data for valid-like (and, by the
LB transfer of the plain valid gain, test-like) sequences. Next: final models train + all valid x3 (2 seeds), and
whether more emphasis (x6) helps further.
Jobs: wunder-vr-final (gru_all_tv_vr3, _seed1: train + all valid x3) and wunder-vr6 (gru_all_tvA_vr6, _seed1).

### 2026-10-21: domain shift train vs valid (CPU job wunder-domain, tools/domain_shift.py)
Per-sequence summaries (mean, std, mean |diff| of the 112 raw columns), HistGB valid-vs-train, 5-fold CV:
**AUC 0.996 overall and 0.994-0.999 within every asset class** -> valid sequences are a different period/regime,
almost perfectly separable from train. Strongest single separators (mean within-class AUC): i1 volume levels and i1
volume activity (mean / mean |diff| of columns 74-95: 0.70-0.74), some i0 volume levels (mean_40, mean_25).
Out-of-fold train p(valid): median 7e-5, 95th pct 0.03 -> importance weights p/(1-p) would collapse onto a handful of
train sequences, so explicit importance weighting is not usable; repeating valid (H2) is the practical form.
**H3 (shift-robust inputs):** the raw-level block and the |diff|-activity block carry exactly the sequence-level levels
that shift between periods. Models trained on train only and scored on valid (out of period) measure robustness
directly: `--drop-blocks raw` and `--drop-blocks raw vol` (zeroed after standardisation; export zeroes their weights,
kernel unchanged). Keep only if V1/V2 rise (V3 is in-period and may drop).

### 2026-10-21: how much valid emphasis (job wunder-vr6) and final x3 models (job wunder-vr-final)
| comparison | V1 half B | V2 half B | V3 |
|---|---|---|---|
| gru_all_tvA_vr6 vs gru_all_tvA_vr3 (seed 0) | **-0.0026 (z -2.6)** | **-0.0046** | **-0.0045** |
| gru_all_tvA_vr6_seed1 vs gru_all_tvA_vr3_seed1 | **-0.0041 (z -2.6)** | **-0.0045** | **-0.0041** |
| gru_all_tv_vr3 vs gru_all_tv (V3 only; valid all seen) | | | -0.0024 (z -1.4) |
| gru_all_tv_vr3_seed1 vs gru_all_tv_seed1 | | | -0.0027 (z -2.0) |
x6 overfits the repeated sequences and loses everywhere -> the optimum is at or below x3 (x2 running: wunder-vr2).
Final x3 models (train + all valid x3; V1/V2 seen): gru_all_tv_vr3 0.6955 / 0.5371 / 0.4453, _seed1 0.6969 / 0.5389 /
0.4465. With all of valid repeated (36% of each epoch) the train holdout drops ~0.0025 (with half A only it was flat):
x3 trades train-period accuracy for valid-period accuracy. Whether the test period is closer to valid is not known;
the plain valid refit's LB gain (+0.0019) suggests it is at least partly.
Packaged **submissions/ens2_all_tv_vr3** (same architecture/shape as ens2_all_tv -> same scorer runtime, ~44 min);
checks running on a Kaggle CPU (job wunder-check-vr3: check_submission + tools/verify_numba.py vs torch).
Expected LB if valid-like: ens2_all_tv 0.6562 + ~0.0013 (90% of +0.0015).
Checks (Kaggle CPU, job wunder-check-vr3 v2): ZIP 1.56 MB, imports, output contract, determinism, no state leakage,
causality all PASS; numba vs torch max |diff| 1.7e-6 over 4 valid seqs (tools/verify_numba.py). The runtime projection
there (151 us/row) is not meaningful (Kaggle CPU is not a scorer proxy, see bench_kaggle_cpu); the package has exactly
the shapes of ens2_all_tv (43m36s on the scorer) with the newer allocation-free kernel -> expected ~43-44 min.
First attempt failed only because macOS tar shipped ._ AppleDouble files (push.sh now sets COPYFILE_DISABLE=1).
**Upload candidate: submissions/ens2_all_tv_vr3.zip.**
**LB 0.65621** (45m17s, 683 MB) = ens2_all_tv 0.6562 -> no transfer. The valid-period gain (+0.0015 V1 half B) and the
train-period loss (V3 -0.0025) cancel on the test set: the test period is not simply "valid-like" (consistent with
AUC 0.996 separability: it is likely a third period). Valid emphasis beyond x1 is neutral for test -> ens2_all_tv stays
primary (same score, better V3). Lesson: half-B gains measure valid-period fit, not test fit; for shift questions,
judge on V3 and V1/V2 together. Runtime 45m17s vs 43m36s for identical shapes -> scorer timing noise ~+-2 min.

### 2026-10-22: shift-robust inputs (job wunder-dropblk, train only, seed 0) and valid x2 (job wunder-vr2)
| name | V1 | V2 | V3 | paired delta vs gru_all_full (V1 / V2 / V3) |
|---|---|---|---|---|
| drop_raw_full | 0.6791 | 0.5060 | 0.4445 | -0.0019 / **-0.0024 (z -2.5)** / -0.0021 |
| drop_rawvol_full | 0.6789 | 0.5079 | 0.4458 | -0.0021 / -0.0006 / -0.0008 |
**H3 rejected:** removing the sequence-level levels loses even out of period (valid) -> they are signal, not shift noise.
gru_all_tvA_vr2 / _seed1 vs vr3 on half B: V1 -0.0001 / +0.0002, V2 -0.0009 / -0.0003 -> x2 = x3; with x3 neutral on
the LB, the valid-emphasis line is closed.
Next (job wunder-tune): tbptt2k_full (batch 8 x chunk 2000, same rows per step), lr2e3_full (OneCycle max lr 2e-3),
and eval-only pairs gru_all_full + drop_raw(vol)_full vs the seed pair (input-view diversity).
