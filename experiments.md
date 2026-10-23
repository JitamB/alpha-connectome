# Experiment log

Decision rule: keep a change only if it beats the current best on V1 and V2 by > 2 sigma of the paired
bootstrap (`python src/evaluate.py compare A B`) and does not hurt V3 beyond noise. LB is for confirmation only.

Views: V1 = valid, official mask · V2 = valid, all need_prediction rows · V3 = train holdout (splits/holdout.json, 800 seqs), all rows.
Runtime: M2 us/row from `src/check_submission.py`; projected scorer minutes = us/row x 1.55 x 39.4M / 60e6 (limit 60, target <= 48).

**Since 2026-10-22, the decision rule and runtime model are in `.claude/skills/connectome-experiment/SKILL.md`.**
- `tools/judge.py` gives same-seed paired deltas with training noise included.
- Scorer minutes ≈ 5.3 + 19.5·K.
- LB ≈ 0.93·V1 + 0.0195.

## Closed lines (check before proposing; reopen only with a materially different hypothesis)
| line | verdict | evidence | would reopen it |
|---|---|---|---|
| width / capacity (h192, h384) | no gain, both recipes | 10-09 x_h192, 10-15 x_all_h384 | new inputs that change the bottleneck |
| LSTM cell; CNN front-end; TBPTT 2k / 4k | no gain | 10-16, 10-15, 10-15 / 10-22 | — |
| weight EMA; model soups; Pearson loss | no gain (≤ +0.0004) | 10-09, 10-16, 10-08 | — |
| t0+t1 sum loss; distributional head as *output* | worse | 10-15 x_all_sum, x_all_s4000_dist | (the dist head as an aux term was never ablated) |
| class-3 specialists; per-class calibration | no gain | 10-15 c3ft, calibration | — |
| privileged-teacher distillation | no gain (student not label-noise-limited) | 10-21 distil_w1/w3 | — |
| reservoir random filter bank | 0.6335, blending hurts | 10-16 | — |
| valid emphasis ×3 / ×6 | LB = ×1; ×6 overfits | 10-21 | confounded: LB-tested share 36% vs validated 22% |
| drop raw / vol blocks | worse out of period | 10-22 | — |
| lr 2e-3, batch 32, 5 epochs, mask eps 0.03 | worse | 10-22, 10-15 | — |
| lag / future / past-mean aux variants | within noise | 10-16 | fut targets: +0.0014 / +0.0002, never a 3rd seed |
| decoded tick paths, i1 lead-lag, event clock (linear stacking gate) | ≤ +0.0005 | 10-16 gate_paths | exact target definition → features as GRU inputs |
| reconstructed past targets (HistGB on 60 seqs) | +0.000 | 10-09 | exact target definition (realized labels) |
| K=3 ensemble; bf16 / int8 kernels | timeout / slower / −0.0007 | 10-16, 10-18 | — |
| causal-ensemble distillation | not run: teachers are in-sample, so it would distil memorisation | 10-22 review | cross-fitted teachers |
| best-exit target (H3) + anchored-window clock (B) | rejected: S falls with activity, floor broken, no length threshold | 10-22 forensics_a1 | — |
| online per-sequence calibration / class-3 regime features | closed: seq-mean y does not persist (half-vs-half corr ≈ 0) | 10-22 forensics_a1 | — |
| reconstructed-mask training weights (B3) | closed: mask AUC 0.889 even with next-row features (gate 0.93) | 10-22 forensics_a1 | — |
| regularisation: dropout 0.1, weight decay 1e-3 | dropout −0.0009 V1 (2 seeds), wd flat | 10-22 round 2 | — |
| activity/range aux head (future excursions, price-change counts, mean abs y) | V1 −0.0002 main effect (2×2, 2 seeds) | 10-22 round 2 | — |
| exact-shape lookup (S/D per exact i0 shape): post-hoc ridge, stacking gate, HistGB | V1 +0.00004 / +0.00001 (G1), HistGB worse | 10-22 corr, shape_gate | the GRU already has it |
| class-conditional joint tick-pair CE and metric-aware joint decoder | joint loss V1 −0.00012 / c3 −0.00063; decoded output V1 −0.00214 | 10-22 jointpair (2 seeds) | causal c3 transition representation, not a larger output distribution |
| explicit book/trade timers, transition rates, and row-position inputs | V1 −0.00058 / c3 −0.00064 (2 seeds); V3 +0.00105 does not transfer | 10-23 clock | materially different causal state-transition inputs |
| class-3 target-innovation (one-row delta) auxiliary loss | V1 −0.00003 / c3 −0.00005 (2 seeds) | 10-23 delta_c3 | a different causal target, not loss-weight tuning |
| class-3 ordered entry-transition target table | c3 V1 −0.00145 G1 / −0.00031, −0.00086 G2 | 10-23 entry gate | a causal target definition beyond i0 state history |
| volume-queue imbalance (role-agnostic: 366 candidates, screen + stacking gate + HistGB) | screen below null and not replicated; gate V1 −0.0012 / −0.0023 / −0.0012 (c3 −0.0020 / −0.0087 / −0.0050) | 10-23 imbalance_probe | volume columns are a decorrelated basis, not a queue ladder |
| i1 lead-lag as GRU input (out-of-fold i1 transition-type tables, 6 window-sum features) | V1 +0.00016 (seeds −0.00023 / +0.00055), c3 +0.00022, V3 +0.00097 | 10-23 i1lead (2 seeds) | label-derived i1 decoding adds nothing on clean V1 |

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

### 2026-10-22: tuning round (job wunder-tune, full data, seed 0, paired vs gru_all_full)
| name | change | V1 | V2 | V3 | paired delta (V1 / V2 / V3) |
|---|---|---|---|---|---|
| tbptt2k_full | batch 8 x TBPTT chunk 2000 (same rows/step), 1.45x slower | 0.6809 | 0.5087 | 0.4473 | -0.0001 / +0.0002 / +0.0007 -> nothing |
| lr2e3_full | OneCycle max lr 2e-3 (was 1e-3) | 0.6744 | 0.4988 | 0.4365 | **-0.0066 / -0.0096 / -0.0101** |
| pair_full_droprv (gru_all_full + drop_rawvol_full) | input-view diversity, vs the seed pair ens2_all_full | 0.6825 | 0.5126 | 0.4505 | -0.0004 / +0.0012 (z 3.3) / -0.0001 |
| pair_full_dropraw (gru_all_full + drop_raw_full) | same | 0.6823 | 0.5110 | 0.4493 | -0.0006 / -0.0003 / -0.0013 |
Long TBPTT is closed (s4000 x4 and full x2 both flat). Input-view pairs average like seed pairs (no V1 gain).
The lr has never been tuned: 2e-3 costs 0.007, so the optimum may be below 1e-3 -> job wunder-lr: lr5e4_full, lr7e4_full.

### 2026-10-22: lower learning rates (job wunder-lr, full data, seed 0, paired vs gru_all_full at 1e-3)
| name | V1 | V2 | V3 | paired delta (V1 / V2 / V3) |
|---|---|---|---|---|
| lr5e4_full | 0.6815 | 0.5086 | 0.4460 | +0.0005 (z 1.6) / +0.0001 / -0.0006 |
| lr7e4_full | 0.6818 | 0.5089 | 0.4468 | +0.0008 (z 1.9) / +0.0005 / +0.0002 |
The lr curve is flat below 1e-3 and steep above (2e-3: -0.0066). 7e-4 is marginally best but within ~1.3 run sd;
replicates running (job wunder-lr2: lr7e4_full_seed1 / _seed2 vs gru_all_full_seed1 / _seed2).

### 2026-10-22: lr 7e-4 replicates (job wunder-lr2)
| pair (lr 7e-4 vs 1e-3, same seed) | V1 | V2 | V3 |
|---|---|---|---|
| seed 0 | +0.0008 (z 1.9) | +0.0005 | +0.0002 |
| seed 1 | +0.0007 (z 2.2) | +0.0005 | -0.0000 |
| seed 2 | **+0.0014 (z 3.4)** | **+0.0027 (z 6.3)** | +0.0010 |
| mean | **+0.0010** | **+0.0012** | +0.0004 |
lr 7e-4 singles: V1 0.6818 / 0.6822 / 0.6822 (mean 0.6821, vs 0.6811 at 1e-3). Consistent sign in 3/3 seeds on every view,
zero inference cost -> **adopted** (just under the +0.0012 V1 bar on the mean, but no run went the other way).
Next: job wunder-tv-lr7e4 (final pair on train + all valid at 7e-4 + eval of the train-only 7e-4 pair vs ens2_all_full)
and job wunder-opt (batch 32 and 4 epochs at 7e-4, paired vs lr7e4_full).

### 2026-10-22: lr 7e-4 final pair (job wunder-tv-lr7e4)
| name | V1 | V2 | V3 | paired delta |
|---|---|---|---|---|
| pair_lr7e4 (lr7e4_full + _seed1, train only) | 0.6833 | 0.5115 | 0.4502 | vs ens2_all_full: +0.0005 (z 2.1) / +0.0001 / -0.0004 |
| gru_all_tv_lr7e4 (train + all valid) | (0.6872 seen) | (0.5211 seen) | 0.4482 | V3 vs gru_all_tv +0.0005 |
| gru_all_tv_lr7e4_seed1 | (0.6878 seen) | (0.5218 seen) | 0.4496 | V3 vs gru_all_tv_seed1 +0.0004 |
The single-model gain (+0.0010 V1) shrinks to ~+0.0005 after 2-model averaging (part of what lr 7e-4 fixes is
seed-level variance that averaging already removes). Packaged submissions/ens2_all_tv_lr7e4 (same shapes as
ens2_all_tv); expected LB ~ +0.0004 over 0.6562 -- inside scorer noise, optional upload. Checks: job wunder-check-lr7e4.
Checks (Kaggle CPU, job wunder-check-lr7e4): all correctness checks PASS, numba = torch to 1.8e-6 (runtime projection
not meaningful there; same shapes as ens2_all_tv -> ~44 min). submissions/ens2_all_tv_lr7e4.zip is ready.

### 2026-10-22: optimisation at lr 7e-4 (job wunder-opt, seed 0, paired vs lr7e4_full)
| name | change | V1 | V2 | V3 | paired delta (V1 / V2 / V3) |
|---|---|---|---|---|---|
| lr7e4_b32_full | batch 32 (half the optimiser steps) | 0.6806 | 0.5071 | 0.4458 | **-0.0012 / -0.0018 / -0.0011** |
| lr7e4_e4_full | 4 epochs | 0.6825 | 0.5110 | 0.4490 | **+0.0008 / +0.0020 (z 4.9) / +0.0021 (z 2.7)** |
More optimiser steps help, fewer hurt -- at the lower lr the recipe is under-trained (monitor 0.4478 at epoch 3 and 4;
at lr 1e-3 / old recipe 5 epochs gave nothing). Next (job wunder-epochs): lr7e4_e4_full_seed1 (replicate vs
lr7e4_full_seed1) and lr7e4_e5_full (vs e4).

### 2026-10-22: LB of ens2_all_tv_lr7e4
**LB 0.65655** (43m22s, 683 MB): +0.00035 over ens2_all_tv (0.6562), as expected from the pair gain (+0.0005 V1,
~90% transfer). New best by a hair; within scorer noise on its own, but consistent with the 3-seed local evidence.

### 2026-10-22: epochs at lr 7e-4 (job wunder-epochs)
| comparison | V1 | V2 | V3 |
|---|---|---|---|
| lr7e4_e4_full_seed1 vs lr7e4_full_seed1 (4 vs 3 epochs) | -0.0006 | +0.0010 | +0.0020 |
| (seed 0, earlier) lr7e4_e4_full vs lr7e4_full | +0.0008 | +0.0020 | +0.0021 |
| mean of 2 seeds | **+0.0001** | **+0.0015** | **+0.0020** |
| lr7e4_e5_full vs lr7e4_e4_full (5 vs 4) | -0.0004 | -0.0007 | -0.0008 |
4 epochs: all-rows views (V2, V3) up in both seeds, the masked V1 flat; 5 epochs past the peak. Not a V1 win, but V3
is the only clean view for train + valid models and moves +0.002 in both seeds -> one 4-epoch final pair as a
candidate (job wunder-tv-e4: gru_all_tv_lr7e4_e4 / _seed1 + eval of the train-only e4 pair vs pair_lr7e4).

### 2026-10-22: 4-epoch final pair (job wunder-tv-e4)
| name | V1 | V2 | V3 | paired delta |
|---|---|---|---|---|
| pair_lr7e4_e4 (train-only 4-epoch pair) | 0.6836 | 0.5134 | 0.4527 | vs pair_lr7e4: +0.0003 / **+0.0019 (z 5.7)** / **+0.0025 (z 3.6)** |
| gru_all_tv_lr7e4_e4 / _seed1 (train + all valid) | (0.6897 seen) | (0.5245 / 0.5252 seen) | 0.4504 / 0.4504 | V3 vs 3-epoch tv twins +0.0022 / +0.0008 |
Packaged submissions/ens2_all_tv_lr7e4_e4 (same shapes -> ~44 min); checks (job wunder-check-e4): all correctness checks
PASS, numba = torch to 1.7e-6 (runtime projection on the Kaggle CPU not meaningful). V1 is flat at pair level,
all-rows views clearly up -> expected LB +0.0000 to +0.0010 over 0.65655.
Seen-vs-unseen check from saved moments: gru_all_tvA vs gru_all_full on valid half A (seen) +0.0052 / half B (unseen)
+0.0007; seed 1 +0.0061 / +0.0004 (class 3: +0.009 / +0.015 seen) -> a real memorisation gap; regularisation untested.

## 2026-10-22: research review, round 2 (new session; workflow in .claude/skills/connectome-experiment/SKILL.md)

## 2026-10-22: preregistration — structured joint tick-pair head (Experiment 1)
- **Hypothesis:** a class-conditional joint distribution over the clipped target pair will teach/use the discrete tick-grid and t0/t1 coupling better than independent per-target atom CE. Decode the marginals with the metric-optimal `E[|y|y] / E[|y|]` rule; retain the regression head as the other candidate output. This is a different hypothesis from the rejected sum-MSE and marginal-distribution output: the unit is the *joint pair*, and scoring is judged on the decoded pair distribution.
- **Primary comparison:** two train-only runs, seeds 0 and 1, paired against `lr7e4_full` / `lr7e4_full_seed1`; same base recipe (lr 7e-4, 3 epochs, mask proxy, lag/feature auxiliaries and marginal distribution auxiliary) plus a joint-pair auxiliary loss. Also evaluate joint-decoded output vs regression output from each candidate. Primary view V1 on clean valid sequences; class-3 V1 is the mechanism check. V2 and V3 can veto; report each class and V1t sensitivity.
- **Decision rule:** candidate only if the two-seed mean paired V1 gain is at least +0.0009, neither seed is below −0.0006, class-3 V1 improves, and V2/V3 are not below −2 SE. A joint-decoded output must beat the regression output on clean V1 to justify any inference-path cost. If the joint loss does not improve class 3 or the pair decode is flat, close this exact formulation.
- **No leakage:** use train-only runs for the primary comparison; all tuning/model selection stays sequence-grouped. Do not use valid labels in fitting. Only refit on train+valid after the screen passes.
- **Cost:** code change and CPU-free orchestration on the Mac; Kaggle fetch ~12 min, two 3-epoch T4 training runs in parallel ~75 min, GPU evaluation ~10 min, total target ~1h45m. No local parquet reads, training, evaluation, or GPU work. Inference output size/runtime to be measured on Kaggle only if the joint-decoded output wins.

### Result — REJECT (Kaggle jobs `theconqueror01/wunder-jointpair-exp1`, `theconqueror01/wunder-jointpair-eval`)

| output / comparison | V1 | V2 | V3 | paired result vs base |
|---|---:|---:|---:|---|
| regression head, seed 0 / 1 | 0.68163 / 0.68212 | 0.50926 / 0.50941 | 0.44777 / 0.44893 | V1 **−0.00012** mean (SE 0.00052); class 3 **−0.00063** |
| joint-pair decoded, seed 0 / 1 | 0.67886 / 0.68084 | 0.50410 / 0.50628 | 0.44554 / 0.44609 | vs candidate regression: V1 **−0.00203**, V2 **−0.00414**, V3 **−0.00253**; vs base V1 **−0.00214** |

The class router is accurate on 99.41% / 99.15% of V1 sequences (98.75% / 98.63% V3), so wrong class routing does not explain the failure. The joint CE changes neither the regression solution nor class 3 in the required direction, while metric-aware marginal decoding from the learned joint distribution is strongly worse on all views. Close the class-conditional joint tick-pair loss and this exact decoder; any future use of target coupling needs a representation tied to a causal c3 state transition, rather than a larger output distribution.

## 2026-10-22: preregistration — explicit causal state-clock inputs (Experiment 2)
- **Hypothesis:** the class-3 residual has slow *row-time* dependence (row-lag-100 autocorrelation 0.44 versus event-lag-20 0.08), while its exact book state flickers at a median segment length of one. The GRU currently has to infer and preserve elapsed-time counters from raw changes. Supplying exact causal book/trade timers, trailing transition rates, and row position lets it learn state × elapsed-time interactions through its projection and recurrent layers. This materially differs from the closed linear event-clock stacker: the features are inputs to the nonlinear causal model, not a post-hoc correction to its predictions.
- **Features:** log elapsed rows since exact i0/i1 22-price-book changes and i0/i1 trade changes; i0/i1 transition rates over trailing 100 and 1000 rows; normalized row position and its square. No target-derived table, future row, or valid label is used.
- **Primary comparison:** two train-only seeds (`clock_s0`, `clock_s1`) against paired `lr7e4_full` / `lr7e4_full_seed1`. The recipe remains lr 7e-4, 3 epochs, mask proxy, lag and future-feature auxiliary losses, and marginal distribution auxiliary loss. Primary view is clean V1; class-3 V1 is the mechanism check. V2/V3 can veto.
- **Decision rule:** candidate only at mean paired V1 ≥ +0.0009, neither seed below −0.0006, class 3 improves, and V2/V3 are not below −2 SE. If class 3 does not improve, close this exact state-clock input set rather than searching timer scales.
- **Cost:** code only; Kaggle data fetch ~12 min, two T4 runs in parallel ~85 min (the small clock block adds minor feature work), V1/V2/V3 evaluation ~10 min; total target ~1h50m. The Mac performs orchestration only. Export/runtime work is deferred until the train-only screen passes.

### Result — REJECT (Kaggle jobs `theconqueror01/wunder-clock-exp2`, `theconqueror01/wunder-clock-eval`)

| comparison | V1 | V2 | V3 | class 3 V1 |
|---|---:|---:|---:|---:|
| `clock_s0` / `clock_s1` scores | 0.68109 / 0.68174 | 0.50889 / 0.50851 | 0.44772 / 0.44948 | — |
| paired vs `lr7e4_full` / `_seed1` | **−0.00058** (SE 0.00056) | −0.00006 | +0.00105 | **−0.00064** |

Both seeds miss the V1 criterion (−0.00069 / −0.00046) and both reduce class 3 (−0.00101 / −0.00028). The V3 gain shows the timers can fit train-period temporal structure, but its failure on clean V1 rules out this direct elapsed-time representation. Close these ten clock inputs as a class-3 route; do not tune timer scales or row-position parameterizations.

## 2026-10-23: preregistration — class-3 target-innovation auxiliary (Experiment 3)
- **Hypothesis:** class 3 has substantial row-time persistence (target autocorrelation 0.44 at 100 rows) but little event-time persistence. The standard level-only loss can match the dominant static cost component while leaving directional changes underconstrained. Add an auxiliary, |y|-weighted MSE on one-row target innovations `y[t] − y[t−1]`, only for class 3, so the recurrent representation must respond correctly to causal state transitions. This differs from the rejected sum loss (a level constraint), lag head (a target reconstruction task), pair CE (an output distribution), and clock features (inputs only).
- **Primary comparison:** two train-only runs (`delta_c3_s0`, `delta_c3_s1`) paired against `lr7e4_full` / `_seed1`. Keep the adopted base recipe and add `--delta-weight 0.1 --delta-classes 3`. Primary view is clean V1; class-3 V1 is the required mechanism check. V2/V3 can veto.
- **Decision rule:** candidate only at mean paired V1 ≥ +0.0009, neither seed below −0.0006, class 3 improves, and V2/V3 are not below −2 SE. Reject if class 3 is flat or down. No loss-weight sweep after rejection.
- **No leakage and cost:** the innovation label uses only adjacent training targets; it is a training-only loss and has zero inference cost. Kaggle fetch ~12 min, two 3-epoch T4 runs in parallel ~75 min, evaluation ~10 min; target ~1h45m. The Mac performs orchestration only.

### Result — REJECT (Kaggle jobs `theconqueror01/wunder-delta-c3-exp3`, `theconqueror01/wunder-delta-c3-eval`)

| comparison | V1 | V2 | V3 | class 3 V1 |
|---|---:|---:|---:|---:|
| `delta_c3_s0` / `delta_c3_s1` scores | 0.68177 / 0.68217 | 0.50891 / 0.50902 | 0.44720 / 0.44801 | — |
| paired vs `lr7e4_full` / `_seed1` | **−0.00003** (SE 0.00046) | +0.00021 | +0.00006 | **−0.00005** |

The innovation loss produces no reliable movement overall or on class 3 (seed deltas −0.00041 / +0.00031 for c3). The recurrent base already encodes the usable one-row target dynamics; increasing the weight or broadening the loss would be an unmotivated weight sweep. Close this target-innovation formulation.

## 2026-10-23: preregistration — class-3 entry-transition lookup gate (Experiment 4)
- **Hypothesis:** the closed shape table used only the current i0 state. In class 3, targets change only at i0 state transitions and scored rows usually occur *after* a transition. The target may therefore depend on the ordered entry transition `(prior state at entry, current state)`, held causally throughout the resulting segment. This is a finite-state transition hypothesis, distinct from the rejected current-state table and from `gate_paths`, which encoded one-step transition deltas and window sums rather than the state level conditioned on its entry.
- **Method:** build a train-only, class-3 table of count, mean tick sum `S`, and mean tick difference `D` for the causal entry-transition key. Append its frozen statistics to the train-only base pair predictions and fit a per-class ridge only on the designated G1/G2 fitting split. Test G1 holdout→valid plus both valid half directions. Non-c3 rows receive zeros, so the gate measures class-3 information without changing other classes.
- **Decision rule:** advance to an input or table-based training experiment only if class-3 V1 improves by at least +0.002 in G1 and both G2 directions, with the same sign overall. Otherwise close the entry-transition table family. This is a discovery gate, not an eligible submission model.
- **Cost:** Kaggle GPU inference for the saved base pair and CPU lookup/ridge work; fetch ~12 min and gate ~20 min, about 35 min total. No local parquet reads, model inference, or evaluation.

### Result — REJECT (Kaggle job `theconqueror01/wunder-entry-gate-exp4`)

| protocol | overall V1 change | class-3 V1 change | class-3 V2 change |
|---|---:|---:|---:|
| G1: holdout → valid | −0.00037 | **−0.00145** | −0.00219 |
| G2: valid A → B | −0.00009 | **−0.00031** | −0.00118 |
| G2: valid B → A | −0.00023 | **−0.00086** | −0.00117 |

The table had 1,017 recurring class-3 entry-transition keys across 8.66M sampled train rows, so data sparsity is not the limiting explanation. Conditioning on the state of entry worsens every out-of-sample split and fails the required +0.002 c3 gate. Close entry-transition lookup features and do not turn them into a training input.

The review (three independent specialists plus cross-review) had these key findings:
- class 3 holds 66% of the residual error;
- t0+t1 is not a pure cost (class 0 has positive |y|-weighted means for both targets);
- the target is likely a "best exit" measure tied to i0 price changes;
- the memorisation gap is +0.005 V1;
- scorer time ≈ 5.3 + 19.5·K min;
- LB ≈ 0.93·V1 + 0.0195.

Pre-registered (running):
- **2×2 screen.** Seeds 0 and 1; base cell = lr7e4_full / _seed1.
  - Factor R = `--dropout 0.1`, motivated by the memorisation gap.
  - Factor A = `--act-weight 0.2`: a training-only head for future i0 excursions, i0 price-change counts and mean |y| over the next 100/300 rows. The hidden state decodes future volatility at only 0.25.
  - Extra run: lr7e4_wd1e3_full (weight decay 1e-3, seed 0).
  - repro_lr7e4_full must reproduce lr7e4_full bit-identically; otherwise the base cell is retrained.
  - Rule: `tools/judge.py --cells`; adopt a factor at a main effect ≥ +0.0006 V1 with V2/V3 not below −2 SE. Kill: both effects < +0.0003.
  - Jobs: theconqueror01/wunder-r2-repro-drop, wunder-r2-act-s0; jitamb/wunder-r2-s1a, wunder-r2-s1b.
- **A1 forensic probes** (CPU, theconqueror01/wunder-r2-forensics; tools/forensics.py): target anatomy (floor test, S purity), clock test, class-3 persistence, a0–a7, mask AUC.
  - Gate G1: clear best-exit structure plus an identified clock → A2, the exact-definition search.
  - Mask gate: AUC ≥ 0.93 with next-row features → reconstructed-mask training weights.
- **K=1 runtime fallback** gru_all_tv_lr7e4_s1 (check job jitamb/wunder-r2-check-k1).

### 2026-10-22: A1 forensic probes (job theconqueror01/wunder-r2-forensics, tools/forensics.py; 60 train seqs per class + 240 valid)
| probe | class 0 | class 1 | class 2 | class 3 | reading |
|---|---|---|---|---|---|
| segments / distinct i0 shapes per seq | 8584 / 1514 | 6108 / 306 | 2369 / 45 | 4067 / 11 | class 3 flickers among ~11 shapes; median segment = 1 row |
| S = n0+n1: top values | −2, −3, −4, −1 … (P(S>0) 0.26) | −2, −1, −3 (0.18) | −1, −2 (0.08) | **−1 66%, −2 27%** (0.03) | cost-like only in coarse classes |
| S purity given the exact shape: within seq / global (held-out seqs, coverage) | 0.67 / 0.61 (95%) | 0.66 / 0.65 (99.4%) | 0.76 / 0.75 (99.8%) | **0.92 / 0.92 (99.99%)** | **S is a lookup of the current exact shape in class 3** |
| "no move" rate by segment length (clock-B test) | ~0.1, flat | → 0 | ~0.12, flat | 0.05 → 0.27, no jump | **no anchored-window threshold → clock B rejected** |
| S vs next-300-row i0 activity | S falls with activity (+5.7 quiet → −0.56) | same | flat | flat | **opposite of best-exit (H3)**; n0 ≤ −3 on 10% of class-3 rows breaks its floor → **H3 rejected** |
| label autocorr: event lag 20 vs row lag 100 | 0.55 / 0.52 | 0.37 / 0.34 | 0.29 / 0.57 | **0.08 / 0.44** | class 3 = shape-determined cost part + slow row-time part |
| persistence of seq-mean y, 1st vs 2nd half (y0 / y1) | −0.09 / 0.17 | 0.28 / 0.18 | −0.04 / −0.06 | 0.09 / −0.01 | **no regime persistence → online per-sequence calibration closed** |

Other results:
- **a-features.** a6 ≈ time since the last i0 transition (Spearman 0.74; it only decreases at i0 transitions). a5 ≈ time since the last i0 trade (0.50). a7 ≈ time since the last i1 event (130 values, 44% at the floor). a2/a3 are high right after i1 events (Spearman −0.41/−0.36 with time since the last i1 trade/transition).
- **Mask.** P(scored | i1 transition at t) 0.16 vs 0.08; P(scored | i0 transition at t) 0.07 vs 0.10. HistGB AUC is 0.886 causal and 0.889 with the next 2 rows added. **The mask cannot be reconstructed for training weights (gate 0.93), so B3 is closed.**
- **Raw examples (class 3).** Shapes cluster into best-price states that share labels. In one sequence, n1 flips 39 ↔ −1 between consecutive transitions, so the label depends on the current quote state in a jumpy way. The exact definition is not found and the realized-label line is parked.

Next: tools/shape_gate.py (job theconqueror01/wunder-r2-shape-gate) stacks per-shape S/D lookup features on the train-only pair. Pass mark: +0.002 V1 in G1 and both G2 directions, against the ridge-on-predictions baseline.

### 2026-10-22: round-2 training levers (jobs theconqueror01/wunder-r2-repro-drop; jitamb/wunder-r2-s1a, -s1b)
**Reproducibility.** repro_lr7e4_full is lr7e4_full re-trained with the new code (all new flags off) on the second account. Its best.pt sha256 is **identical** (d2e62450…) and the monitor curve is identical (0.415582 / 0.437756 / 0.443733). So the default path is unchanged and training reproduces exactly across accounts and days. That covers the prize-verification reproducibility check. Provenance is now recorded in config.json (commit 4f199a3, torch 2.11.0+cu128, cuDNN 91900, T4).

| name | change | V1 | V2 | V3 | same-seed paired delta (V1 / V2 / V3), tools/judge.py |
|---|---|---|---|---|---|
| lr7e4_drop10_full / _seed1 | dropout 0.1 on both GRU inputs | 0.6811 / 0.6811 | 0.5082 / 0.5080 | 0.4463 / 0.4477 | **−0.0009 (z −2.1) / −0.0007 / −0.0006**, class 3 −0.0015 → **REJECT** |
| lr7e4_wd1e3_full | weight decay 1e-4 → 1e-3 (seed 0) | 0.6818 | 0.5093 | 0.4466 | +0.0000 / +0.0003 / −0.0003 → flat |

The memorisation gap is real (+0.005 on seen valid), but dropout and weight decay do not turn it into generalisation. Regularisation is closed.

**2×2 screen complete** (job theconqueror01/wunder-r2-act-s0 for the seed-0 cells). Base = lr7e4_full / _seed1; `tools/judge.py --cells`:

| effect (2 seeds) | V1 | V2 | V3 | class 3 V1 | verdict |
|---|---|---|---|---|---|
| activity/range head `--act-weight 0.2` (main effect) | −0.0002 (SE 0.0004) | −0.0006 | +0.0001 | −0.0004 | REJECT (flat) |
| dropout 0.1 (main effect) | −0.0007 (z −2.1) | −0.0008 | −0.0008 | −0.0014 | REJECT (hurts) |
| interaction | +0.0002 | −0.0002 | −0.0003 | +0.0001 | — |
| both vs base | −0.0010 | −0.0014 | −0.0007 | −0.0018 | REJECT |

The activity head by itself (paired) gives V1 −0.0004 / V2 −0.0004 / V3 +0.0004. Supervising future volatility and excursions adds nothing, even though the state decodes future volatility at only 0.25. Aux-target supervision is saturated, and **Track B is closed.**

### 2026-10-22: exact-shape lookup correction (job jitamb/wunder-r2-corr; tools/fit_shape_correction.py, tables tools/build_shape_table.py)
Method:
- **Tables:** train non-holdout gives 85k shapes (count ≥ 2, 99.85% row coverage, 1.4 MB); train+valid gives 95k (1.6 MB). Only 2 of 236k shapes are mixed-class.
- **Correction:** a per-class |y|-weighted ridge on [p0, p1, hit, log count, E[S·step], E[D·step], P(S bucket) × 4], fitted on the train holdout.

| protocol (train-only pair lr7e4_full + _seed1) | V1 | delta |
|---|---|---|
| G1: fit on holdout, score valid | 0.68334 → 0.68339 | **+0.00004** (V2 +0.00012; each class −0.0001 to −0.0005) |
| G2: fit on valid A, score B / fit on B, score A | | +0.00010 / +0.00004 |
| in-sample on holdout (all rows) | 0.4502 → 0.4508 | +0.0006 (in-sample only) |

The shape-determined cost part S is already captured by the GRU pair's predictions; the old "corr(p0+p1, S) = 0.03 on class-3 scored rows" did not mean the information was missing. **The shape-lookup line is closed.** It is not packaged: the train+valid coefficients (results/corr_tv_lr7e4.npz) are kept but not shipped.

### 2026-10-22: LB of the K=1 fallback gru_all_tv_lr7e4_s1
**LB 0.65539** (26m37s, 680 MB). Predicted from the clean twin: 0.93 × 0.6822 + 0.0195 + ~0.002 (train+valid refit) ≈ 0.6559, so it lands within ~0.0005. The K=2 pair ens2_all_tv_lr7e4 adds +0.0012 LB over this single model. As a runtime hedge, the K=1 pick fits a final set up to ~2.2× the public test size; K=2 (~44 min) fits up to ~1.3×.
Independent confirmation (job theconqueror01/wunder-r2-shape-gate, tables from 3000 seqs, tools/shape_gate.py):

| protocol | V1 | V2 |
|---|---|---|
| G1 holdout → valid | +0.00001 | +0.00003 |
| G2 A→B | +0.00007 | −0.00018 |
| G2 B→A | −0.00001 | −0.00007 |
| HistGB (pred + shape + class) | 0.6805, below pred 0.6833 | |

On class-3 *scored* valid rows, the modal-S lookup is right only 77% of the time (92% on all segments). corr(lookup E[S], S) = 0.22 and corr(pred sum, S) = 0.085. Scored rows avoid transitions, and there S is less shape-determined. **Track A is closed.**

### 2026-10-22: round-2 outcome
None of the round-2 levers moved V1 beyond noise:
- the best-exit target and the anchored clock (rejected);
- online / regime calibration (no persistence);
- reconstructed-mask weights (AUC 0.889);
- exact-shape lookup (+0.00004);
- dropout (−0.0009);
- weight decay (flat);
- the activity/range head (−0.0002).

What carries forward:
- **bit-identical reproducibility** across accounts and days (prize verification);
- a **scored K=1 runtime fallback** (LB 0.65539, 26m37s);
- the experiment workflow (skill, judge.py, CANDIDATES.md, closed lines);
- the second Kaggle account.

ens2_all_tv_lr7e4 (LB 0.65655) stays primary.

### 2026-10-22: LB of ens2_all_tv_lr7e4_e4 (4-epoch pair)
**LB 0.65702** (44m27s, 683 MB), +0.00047 over ens2_all_tv_lr7e4 (0.65655): a new best by a hair. On its own the gap is within the LB noise for near-identical packages, but it agrees in sign with the local pair evidence (V1 +0.0003, V2 +0.0019 (z 5.7), V3 +0.0025 (z 3.6)). It becomes co-primary.

## 2026-10-23: default-path reproducibility check after the Codex edits (job theconqueror01/wunder-r3-repro-check)
- **Why:** commits 15ec2c3..eb39f21 changed train/train_gru.py, train/eval_torch.py, train/gpu_features.py and src/features.py (new opt-in flags: pair/delta losses, clock inputs). A flag must leave the default path bit-identical.
- **Run:** `repro2_lr7e4_full` = lr7e4_full recipe, seed 0, all new flags off, code at commit eb39f21 (provenance dirty_files 0).
- **Pass:** `best.pt` sha256 = d2e62450b9cf6d05dec276794fa0e8b23ce71ae181ca128f4711440fcf61cc21 (same as lr7e4_full and repro_lr7e4_full). Fail: find the diff in the default path before any further run.
- **Cost:** fetch ~12 min, 3-epoch train ~75 min, eval ~10 min; launched 09:58 IST, ready ~11:40 IST. One GPU only.

## 2026-10-23: preregistration — queue-imbalance probe for the class-3 direction part (tools/imbalance_probe.py)
- **Hypothesis:** queue imbalance at the best levels is the textbook predictor of the next tick direction for coarse-tick assets, and the direction part (corr ~0.39 on class 3) is the bottleneck. The GRU sees every volume column and its diff linearly, but never a role-aware, nonlinear imbalance feature. Caveat found while designing it: docs say indices are identifiers, not depth levels, and the price columns have mixed-sign correlations (rotated-basis look), so the best-level queues may not be literal columns. The probe is role-agnostic.
- **Not a repeat:** closed lines used price/trade paths, shape lookups, timers, target tables and loss variants. None built volume-imbalance candidates or screened volume columns against the GRU residual.
- **Method (Kaggle GPU, base = lr7e4_full + lr7e4_full_seed1 pair, train-only):**
  1. Residual screen: 366 candidates (44 raw volumes, 44 five-row changes, 2x121 bid-minus-ask pairs, 36 permutation-invariant side aggregates) vs the pair's direction and cost residual, |y|-weighted, on holdout (V3 rows) and valid (scored rows), per class. Null = the same statistic with (y, p, mask) rolled by a random offset (30 replicates).
  2. Transition lift of each volume column (P(i0 shape change in next 5 rows | z>1.28 vs z<-1.28)) and the within-sequence correlation structure of the 22 i0 volume and price columns.
  3. Stacking gate (ridge on [pred, 36 aggregates], G1 holdout->valid, G2 both halves) and HistGB, per class.
- **Decision rule:** advance to a GRU input experiment only if class-3 V1 improves >= +0.002 in G1 and both G2 directions, or a screen candidate beats the null max by >= 2x with the same sign on holdout and valid. Otherwise close volume-imbalance as a class-3 route (the information is then already in the GRU).
- **Unit test (Mac, random data, seconds):** a planted pair signal is recovered at corr 0.383 vs null max 0.034.
- **Cost:** CPU-light code; Kaggle fetch ~12 min + probe ~30 min; launched after the repro job, account theconqueror01 (second GPU).

### Result — REJECT (Kaggle job theconqueror01/wunder-r3-imbalance, finished ~10:30 IST; results/imbalance_probe.json)
**Residual screen.** Nothing clears the permutation null, and nothing replicates between views.
| view, class | null max \|corr\| (30 rolls) | best direction candidate | best cost candidate |
|---|---:|---|---|
| holdout, c3 | 0.069 | i0 v8-v17 +0.045 | i1 ask mean +0.064 |
| valid scored, c3 | 0.064 | i0 bid-ask mean −0.051 | i0 v0-v21 +0.067 |
| holdout, c0 | 0.052 | i0 bid-ask min −0.074 | i1 bid std −0.060 |
| valid scored, c0 | 0.030 | i0 bid-ask std +0.049 | i0 v18 +0.041 |
The top names differ between holdout and valid, and the sign flips for the aggregate bid-ask mean. The only candidates above the null are class 0, with |corr| <= 0.074 and no replication, so they are not an imbalance effect. Rule needed 2x the null max with the same sign on both views: not met.

**Stacking gate (ridge on [pred, 36 aggregates] vs ridge on pred):**
| protocol | V1 all | V1 c3 | V2 c3 |
|---|---:|---:|---:|
| G1 holdout -> valid | −0.00123 | −0.0020 | −0.0017 |
| G2 A -> B | −0.00230 | −0.0087 | −0.0133 |
| G2 B -> A | −0.00115 | −0.0050 | −0.0172 |
HistGB on [pred, aggregates, class]: V1 0.68175 vs 0.68334 for the bare pair (−0.0016). The gate needed +0.002 on class 3; it is negative in all three protocols and overfits hardest within valid halves.

**Structure.** The 22 i0 volume columns are uncorrelated within a sequence (mean pairwise corr 0.01; range −0.33…+0.29; 51% negative within the bid block), so they are not a ladder of neighbouring queue levels (that would give positive neighbour correlations). The 11 price columns have pairwise corr from −0.99 to +1.00 within the bid block. This looks like a rotated/decorrelated basis, consistent with the docs' "index is an identifier, not a depth level". Literal best-level queue imbalance cannot be built from single columns. The transition-lift table is too noisy to use: z > 1.28 is rare for most columns, so rates like 1.0 vs 0.0 rest on a handful of rows.

**Conclusion.** Close volume imbalance as a class-3 route: role-agnostic imbalance aggregates add nothing to the GRU pair, and the column structure says the best-level queues are not recoverable as single columns. Remaining untested: i1 lead-lag as a GRU input (old gate +0.0010 to +0.0016 on class 3), and recovering a hidden linear basis of the volume/price columns (not pursued).

### Result — PASS: default path bit-identical after the Codex edits (job theconqueror01/wunder-r3-repro-check, finished 11:13 IST)
`repro2_lr7e4_full` (commit eb39f21, all new flags off, seed 0) has best.pt sha256 **d2e62450b9cf6d05dec276794fa0e8b23ce71ae181ca128f4711440fcf61cc21**, identical to lr7e4_full and repro_lr7e4_full. Scores V1 0.681786 / V2 0.508943 / V3 0.446826 as before. The pair/delta/clock changes to train_gru.py, eval_torch.py, gpu_features.py and features.py do not touch the default training or evaluation path. best.pt not re-committed (byte-identical to runs/lr7e4_full/best.pt); config/history/results are in runs/repro2_lr7e4_full and results/.

## 2026-10-23: preregistration — i1 lead-lag as GRU input (Experiment 5; tools/build_i1_table.py, src/i1_lead.py, --i1-table)
- **Hypothesis:** i1 is a related instrument whose book changes may lead i0. The 2026-10-16 linear stacking gate (the 'i1enc' group: per-i1-transition-type mean future i0 target change over H = 100 rows, causal window sums over 10 / 100 / 1000 rows) gave V1 +0.0001 / +0.0002 / +0.0005 overall and class 3 +0.0010 / +0.0009 / +0.0016, but class-3 V2 −0.0002…−0.0019, so it did not pass its +0.002 bar. A GRU input could use the same event-decoded information nonlinearly and conditioned on state. Raw i1 prices and diffs are already inputs; what is new is the label-derived decoding of what an i1 transition of this exact type usually does to i0.
- **Not a repeat:** the gate was linear and post-hoc, on the pair's outputs. Here the 6 features enter the GRU. Closed lines (shape lookup, entry transitions, clocks) are all about i0 state; this one is i1 → i0.
- **Honest expectation:** the old gate bounds the gain at about +0.0005 V1 overall (class 3 ~ +0.001). That is below the +0.0009 candidate bar, so a pass would be a surprise. It is run because the user asked for it and the information is genuinely new (cross-instrument).
- **Leakage control:** the table is label-derived, so a training sequence must not read a table that contains its own future. Tables 0..3 are out-of-fold (table k excludes train groups g with g % 4 = k); holdout groups are never in any table; validation and holdout use table 4 (all non-holdout train groups). Types seen fewer than 2 times are dropped; values are shrunk by 20. Features are in target units (no class needed): causal window sums of the looked-up values at i1 transition rows, 2 targets × 3 windows = 6 columns appended after the 672.
- **Primary comparison:** two train-only seeds `i1lead_s0`, `i1lead_s1` (base recipe + `--i1-table results/i1_table.npz`) paired against `lr7e4_full` / `lr7e4_full_seed1`. Primary view V1, class-3 V1 is the mechanism check, V2/V3 can veto.
- **Decision rule:** candidate only if the 2-seed mean paired V1 is >= +0.0009, neither seed is below −0.0006, class 3 improves, and V2/V3 are not below −2 SE. No timer-scale or horizon sweeps after a rejection. If it passes, the table (~600k keys) must also be packaged and timed for the scorer before any submission; this is not an eligible model yet.
- **Mac checks (seconds, random data):** `i1_lead.features` equals gate_paths' 'i1enc' features exactly (max abs diff 0.0); the fold tables exclude their own fold; syntax of the edited train/eval scripts. Default path: every new branch is gated on `--i1-table`; the default-path repro from 10-23 11:13 IST (eb39f21) covers the code before this edit, and these two seeds do not re-prove it.
- **Cost:** table build job (CPU only, no GPU quota): fetch ~12 min + 4-worker stream ~10 min = ~25 min. Then the training job (2 T4, one seed each): fetch ~12 min + 3-epoch run ~80 min + eval ~10 min = ~1h45m. Total about 2h15m end to end.

### Result — REJECT (Kaggle jobs theconqueror01/wunder-r3-i1-table, wunder-r3-i1lead, wunder-r3-i1lead-eval; judge: runs/i1lead_judge.log)
Table: results/i1_table.npz (19 MB, sha256 68c46e8a07e73cd6faa38fba6a09909b58570b2db6f18907b3e70e4b4d1787fe, not committed; rebuild with `tools/build_i1_table.py --out results/i1_table.npz`). Full table 369k i1 transition types from 20.4M transitions; each out-of-fold table ~290k types.

| comparison (paired to lr7e4_full / _seed1) | V1 | V2 | V3 | class-3 V1 |
|---|---:|---:|---:|---:|
| i1lead_s0 / i1lead_s1 scores | 0.68156 / 0.68275 | 0.50857 / 0.51040 | 0.44822 / 0.44880 | — |
| per seed delta | −0.00023 / +0.00055 | −0.00038 / +0.00183 | +0.00140 / +0.00053 | −0.00079 / +0.00123 |
| mean (SE) | **+0.00016** (0.00054) | +0.00072 (0.00059) | +0.00097 (0.00071) | **+0.00022** (0.00088) |
Per class V1: c0 −0.00011, c1 −0.00038, c2 **+0.00129** (z 2.2), c3 +0.00022, V1t +0.00003. The candidate bar was +0.0009 with class 3 improving; both seeds fail it (mean +0.00016, class 3 +0.00022 with the seeds on opposite sides). Verdict REJECT.

Reading: the i1 → i0 information gave the GRU nothing measurable on the class that matters (class 3). The V3 gain (+0.0010) is again a train-period gain that does not appear on clean V1, the same pattern as the clock inputs. The one positive cell, class 2 (+0.0013, z 2.2 in a family of 8 views), is not a pre-registered claim and is not acted on. Together with the 10-16 linear gate (class 3 +0.0009…+0.0016, V2 inconsistent), this closes i1 lead-lag as a class-3 route.

Operational: in-job eval at batch 32 ran out of GPU memory once the 6 extra columns made a second copy of the 678-column input; training and checkpoints were fine. Evaluation was rerun with `--batch 16` (kaggle/run_job.py now lets an `eval:` run pass its own --batch).
