---
name: connectome-experiment
description: Run, judge, log and ship a Wunder Alpha Connectome experiment - pre-register in experiments.md, launch on Kaggle (two accounts), evaluate V1/V2/V3, decide with the seed-aware tools/judge.py, commit, and keep submission candidates in submissions/CANDIDATES.md. Use whenever launching, evaluating, accepting or packaging a Connectome model change.
---

# Connectome experiment workflow

Standing rules (user instructions and memory):
- The Mac only orchestrates. No training, eval, smoke tests or parquet reads locally. Pure numpy on results/*.npz, small edits, exports and unit checks on random tensors (seconds) are fine.
- Before every launch, state the approximate wall time of each stage.
- After every completed experiment:
  - update experiments.md first;
  - commit runs/<name>/{best.pt,config.json,history.json}, results/<name>.*, and the log;
  - write scores and the verdict in the commit message, with no Co-Authored-By trailer.
- Keep everything private: Kaggle datasets and notebooks, and the GitHub repo. Do not push to GitHub before Dec 1.
- At most 5 LB submissions per day. The user uploads them.

## 1. Pre-register (in experiments.md, before launching)
Write a card with:
- the hypothesis and mechanism;
- why it is not a repeat (check "Closed lines" at the top of experiments.md);
- the primary view and the seeds;
- the kill criterion;
- quota and wall time.

## 2. Launch
- **Account 1 (jitamb):** `kaggle/push.sh SLUG 'RUN|train args' ['RUN2|args']`
- **Account 2 (theconqueror01):** `KUSER=theconqueror01 KAGGLE_API_TOKEN=<token the user gave in chat> kaggle/push.sh ...`
  - Never write the token to a file or to memory.
  - Each account can run 2 GPU jobs at once.
  - Quota is 30 h/week per account and resets Saturday 00:00 UTC.
- **Job layout:**
  - One run per GPU, two runs per job.
  - `eval:NAME|runs...` averages the runs listed. `py:` and `post:` run any script.
  - `GPU=false` for CPU probes; it uses no GPU quota.
  - `EXTRA='files'` ships extra files, e.g. checkpoints or new tools.
- **Pairing:** use the same seeds as the base (0, 1, optionally 2) so the deltas pair.
  - Base recipe (runs/lr7e4_full/config.json): `--aux-weight 0.2 --aux-horizons 10 100 --lag-targets 100 300 --lag-weight 0.5 --mask-model models/mask_mlp.pt --mask-eps 0.1 --dist-weight 0.5 --lr 7e-4 --epochs 3 --seed S`
- **Default path:** a new training flag must leave the default path bit-identical. Verify once per code change with a retrain whose best.pt hash matches (e.g. repro_lr7e4_full vs lr7e4_full).
- **Timings (T4):**
  - fetch ~12 min;
  - 3-epoch full run ~75 min (+10% with the act head);
  - V1/V2/V3 eval ~10 min;
  - whole job ~1h45m.
- **Status:** `kaggle kernels status USER/SLUG`. Output: `kaggle kernels output USER/SLUG -p DIR`, then copy runs/<name>/* and results/<name>.* into the repo.

## 3. Judge (`tools/judge.py`, numpy only)
- **Commands:**
  - Paired: `python tools/judge.py --base B B_seed1 --new N N_seed1`. Add `--valid-from 937` when valid half A is in training.
  - 2x2: `python tools/judge.py --cells C00s0,C00s1 C10s0,C10s1 C01s0,C01s1 C11s0,C11s1`
- **Noise model:**
  - The unit of evidence is the same-seed paired delta.
  - SE = sqrt(σ_t²/n + σ_b²), with σ_t(V1) = 0.00062 (training noise) and σ_b from the paired bootstrap.
- **Verdicts (V1 for train-only runs; V1B, i.e. half B, when valid is in training):**
  - **1 seed:** STOP below −0.0006; CONTINUE above +0.0006.
  - **2 seeds:** CANDIDATE if the mean is ≥ +0.0009, no seed is below −0.0006, and V2/V3 are not below −2 SE. For a 2x2 main effect the bar is +0.0006.
  - **Adopt:** after a fresh seed with mean ≥ +0.0007, or when the gain holds at pair level.
- **Views:**
  - Never decide on V1/V2 for sequences the model has already seen, and never on V3 alone.
  - V1t (V1 reweighted to the test class mix) is only a sign check for class-targeted changes.
- **Pair-level shrinkage:** variance-type changes (lr, epochs, regularisation) shrink about 2× after averaging two models. Information-type changes (aux heads, mask weights, new inputs) keep ~90%. Decide shipping on the pair-level delta, using an `eval:` stage that averages both seeds.
- **LB conversion:** LB ≈ 0.93·V1(train-only) + 0.0195. A train+valid refit adds about +0.002.

## 4. Document
- **experiments.md:** a section per job with a table (name, change, V1/V2/V3, paired deltas, verdict) and a short reading. Rejections also go into "Closed lines": hypothesis, verdict, evidence, and what would reopen it.
- **Commit:** `git add` the runs, results and log; message "<what>: V1 +x / V2 +y / V3 +z (n seeds); <verdict>".

## 5. Ship (submissions/CANDIDATES.md)
- **Build:** train+valid refit pair → `train/export_fast.py runs/a runs/b --name NAME` → check job on a Kaggle CPU (`py:check_X|src/check_submission.py submissions/NAME --seqs 6` and `py:verify_X|tools/verify_numba.py submissions/NAME runs/a runs/b --seqs 4 --name verify_X`).
- **Record:** commit the package, add a row to CANDIDATES.md (sha256 of the zip, clean train-only twin scores), and tag scored packages `lb/<name>`. Never delete or overwrite a scored package.
- **Runtime:** scorer minutes ≈ 5.3 + 19.5·K for 39.4M rows (h128 models). Keep ≥ 10 min margin. The final-set size is unknown, so keep a scored K=1 fallback.
- **LB use:** submit only for runtime or correctness checks, an expected gain ≥ +0.0015, or scoring final picks. Never rank candidates on LB gaps under 0.001.
