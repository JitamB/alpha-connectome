# Submission candidates

The rules are in `.claude/skills/connectome-experiment/SKILL.md`, section 5.

- Never delete or overwrite a scored package.
- "Clean twin" = the same recipe trained on train only, so V1/V2 are unseen.
- Expected LB ≈ 0.93·V1(twin) + 0.0195, plus ~0.002 for a train+valid refit.
- Runtime (scorer, 39.4M rows) ≈ 5.3 + 19.5·K min.

| package | models (K) | train data | clean twin V1 / V2 / V3 | LB (scorer time) | status | zip sha256 |
|---|---|---|---|---|---|---|
| ens2_all_tv_lr7e4 | gru_all_tv_lr7e4 + _seed1 (2) | train + all valid | pair_lr7e4 0.6833 / 0.5115 / 0.4502 | **0.65655** (43m22s) | **current best, primary pick** | ef90e4ad… |
| ens2_all_tv | gru_all_tv + _seed1 (2) | train + all valid | ens2_all_full 0.6829 / 0.5114 / 0.4506 | 0.6562 (43m36s) | scored | d22f6df9… |
| ens2_all_tv_vr3 | gru_all_tv_vr3 + _seed1 (2) | train + all valid ×3 | (no clean twin) | 0.65621 (45m17s) | scored; period hedge | a0f6be83… |
| ens2_all_full | gru_all_full + _seed1 (2) | train only | itself 0.6829 / 0.5114 / 0.4506 | 0.6543 (44m52s) | scored | c35da0f4… |
| gru_all_full | gru_all_full (1) | train only | itself 0.6810 / 0.5085 / 0.4466 | 0.6522 (24m46s) | scored; K=1 | 51f55fe4… |
| ens2_all_tv_lr7e4_e4 | gru_all_tv_lr7e4_e4 + _seed1 (2) | train + all valid, 4 epochs | pair_lr7e4_e4 0.6836 / 0.5134 / 0.4527 | — | checks pass; not scored (expected +0.000 to +0.001) | f2ad4019… |
| gru_all_tv_lr7e4_s1 | gru_all_tv_lr7e4_seed1 (1) | train + all valid | lr7e4_full_seed1 0.6822 / 0.5086 / 0.4483 | 0.65539 (26m37s, 680 MB) | **scored K=1 runtime fallback** (fits a final set up to ~2.2× the public test) | b63c2fb9… |
| ens3_all_tv | 3 tv models (3) | train + all valid | ens3_all_full 0.6833 / 0.5119 / 0.4513 | TIMEOUT (61m56s) | rejected: K=3 does not fit | 803a2d97… |

Older scored packages (gru_eng_p64_s2000 0.6373, gru_eng_p64_full 0.6433, ens2_full_e3_e5 0.6459, linear_all 0.6089) are superseded.
