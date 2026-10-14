# HW5 — Matrix Factorization in TensorFlow.js (MovieLens 100K)

LLM4Rec HW5: train a 16-dimensional matrix factorization (`r_hat = mu + p_u . q_i`)
on MovieLens 100K **in the browser** with TensorFlow.js, produce Top-10 lists, and
evaluate Recall@10 — with every number the page prints cross-checked against an
independent numpy implementation. The deployed page lives in [`site/`](site/)
(published to GitHub Pages by [.github/workflows/pages.yml](.github/workflows/pages.yml)).

Four implementations of the same specification exist in this repository's history:

1. the **starter** (`script.js`) — six throwing `TODO(hw5)` stubs, kept for the diff;
2. **`fixed/`** — the stubs implemented from the pinned design (§2 below) and
   corrected against the numpy reference (`fixed/` is *not* deployed);
3. **`site/`** — the stubs re-implemented from scratch, following only
   [`prompt_fixed.md`](prompt_fixed.md), and deployed;
4. **numpy** (`mf_core.py` / `mf_reference.py`) — no TensorFlow.js at all, used to
   replay and cross-check both browser implementations.

`fixed/` and `site/` produce **bit-identical** factors and identical 20-epoch
histories (compared in the session that wrote them), so the pinned specification
of §2 is exact enough to be implementation-independent.

---

## 1. What the starter contained — and what the checks found

The starter contained a complete scaffolding: `index.html` + `style.css` (page),
`data.js` (parse `u.item`/`u.data`, rating matrix, chronological 80/10/10 split),
and a `script.js` with the TensorFlow.js bootstrap, RMSE/baseline helpers, all DOM
renderers, the hyper-parameter reader, the training-pipeline wrapper and an
11-check self-test harness — plus **six throwing stubs** (`predictRating`,
`initializeFactors`, `trainMF`, `selectBestCheckpoint`, `recommendTopK`,
`recallAtK`). The harness deliberately reports unimplemented stubs as PENDING, so
the starter showed **4 passed / 0 failed / 7 pending**.

Findings (each traced to an artifact in `results/`; see §4 for the commands):

| # | Finding | Evidence |
|---|---|---|
| 1 | The `trainMF` sketch calls `.dispose()` on `optimizer.minimize(f, false, varList)` — with `returnCost = false`, TF.js 4.22 applies the gradients, disposes the cost, and **returns `null`**, so the sketch throws `TypeError` on the very first batch (all 7 stub checks fail or stay pending). | `check-3` (and the shipped TF.js source: `value.dispose(); return null;`) |
| 2 | The sketch's L2 is `reg * (|P|^2 + |Q|^2)` over the **whole matrices**, added to every batch. It shrinks every row at every step (the data gradient on `p` is exactly 0 while `q = 0`, so the origin is absorbing), and the model collapses to the training mean — **while all 11 self-checks pass**. Test RMSE 1.2289 = baseline 1.2289, |P| = 0. | `check-4` (flat curve 1.1052/1.1830), `check-5` (2 FAILs), `check-6` (sweep: only the per-rating penalty learns; matrix rows collapse at every reg) |
| 3 | `trainingHistory` is never reset — a second Train click appends (20-epoch run + two 1-epoch clicks → 22 rows with duplicate epochs, best spanning runs). | `check-4` `[retrain]` (21 then 22 rows) |
| 4 | The selected checkpoint is not the model in memory: `selectBestCheckpoint` reports an epoch while the factors of the *last* epoch stay in `model`, so a test RMSE computed after training belongs to the wrong epoch. | `check-4` `[best]` ("NOT the selected checkpoint") |
| 5 | The `recommendTopK` sketch ranks by **unclamped** `mu + p_u . q_i` — fine as a ranking score, but unmarked against §6's clamp on predicted ratings (a listed score can exceed 5, e.g. 5.1224). | audit NOTE, `check-7` `[top10]` |
| 6 | `runPipeline` trains synchronously after setting "Training…", so the status never paints and the page freezes for the whole training. | audit NOTE, `check-2`/`check-4` timing lines |
| 7 | The page shows **no MF test RMSE anywhere** (only the baseline test RMSE in the split hint) although the readme requires reporting it. | `check-2` vs `check-7` `[test]` |
| 8 | The page shows **no matrix sparsity** and names the padded matrix `944 x 1,683` (row/column 0 unused) — the reportable figure is `943 x 1,682`, density 6.3047%, sparsity 93.6953%. | `check-1` `[matrix]` |
| 9 | `data.js` carries a stray `TODO(hw5)` comment above the already-implemented `splitByTimestamp` (and the readme says `data.js` must not be modified). | `check-1` `[data.js]` NOTE |
| 10 | Readme §6 describes a relevant test movie "the user already rated in training" — impossible: `u.data` has **0 repeated (user, movie) pairs**. Unseen ids are relevant test movies nobody rated in training. | `check-1` `[readme]` NOTE |
| 11 | The slide/§5.1 say "predict ratings using their **dot product**"; §6 defines `mu + p_u . q_i` clamped to [1, 5]. The harness settles it (test #5 expects `3 + 11 -> 5`). | `check-2` tests, prompt_fixed.md §5.1 |
| 12 | The readme's citation has no volume, article number or DOI (the audit flags `10.1145/2827872` as MISSING). The archive README's CITATION block and Crossref agree on the full entry. | `check-1` `[readme]`; `data/ml-100k/README` |
| 13 | §8/§9 point to `docs/homework-guidelines/guidelines.md`, which is not part of this repository. | `check-1` `[readme]` claims; `check-16` |
| 14 | `tf.train.adam(lr)` takes its epsilon from the **backend** (1e-4 on webgl, 1e-7 on cpu in TF.js 4.22): the same code gives epoch-1 RMSE 0.8894/1.0203 on webgl and 0.8865/1.0233 on cpu. The replay identifies the active epsilon. | `check-12b`, `check-12c` |

Items 1–4, 8, 10, 11, 12 and 13 are repaired in [`prompt_fixed.md`](prompt_fixed.md)
(§4.3, §5.1, §6, §6.1, §7 item 4, §7.1, §8/§9, §3); items 1–6 are repaired in the
`fixed/` and `site/` implementations.

---

## 2. The pinned design (prompt_fixed.md §6.1)

Every value below is mandatory; a fresh implementation following only
`prompt_fixed.md` reproduces all numbers (verified: `fixed/` vs `site/` are
bit-identical; see `check-17`–`check-21`).

| Decision | Pinned value |
|---|---|
| Model | `r_hat(u, i) = mu + p_u . q_i`, 16 factors; `mu` = training-split mean (plain JS, not a tensor) |
| Clamp | `predictRating` clamps to [1, 5]; **the Top-K ranking score is unclamped** |
| Initial factors | `tf.randomNormal(shape, 0, 0.05, "float32", seed)` — seed **42** (`userFactors`), **43** (`itemFactors`), each created inside `tf.tidy` |
| Shuffle | mulberry32, seed **7** (exact generator body in prompt_fixed.md §6.1), created once per `trainMF` call with the index array `[0..n-1]`; Fisher-Yates in place at every epoch |
| Batches | consecutive slices of `min(max(1, batchSize), n)` in shuffled order |
| Optimizer | one `tf.train.adam(lr)` per `trainMF` call, kept across epochs, **backend default epsilon** (webgl 1e-4 / cpu 1e-7) |
| Loss | batch MSE of the unclamped prediction **+** `reg *` batch mean of `(|p_u|^2 + |q_i|^2)` on the batch rows only (Koren, Bell & Volinsky 2009, equation 2) — never `reg * (|P|^2 + |Q|^2)` |
| Minimize | `optimizer.minimize(f, false, [userFactors, itemFactors])` — returns `null`, nothing chained |
| History | `trainingHistory = []` on every Train click; one `{epoch, trainRMSE, valRMSE}` row per epoch |
| Checkpoint | strict-improvement clones of both factor matrices; after the last epoch the earliest best-val factors are assigned back into the variables |
| Top-10 | one `matMul` over all item rows; exclude ids absent from `movies` and ids rated in train/val; `tf.topk` over masked scores; titles via `movieTitle` |
| Recall@10 | candidates = movie ids anywhere in the training split; relevant = test ratings >= 4.0 among candidates; macro-mean over evaluated users; report unseen ids and users without a relevant candidate |

Reference numbers (webgl/SwiftShader, 20 epochs, lr 0.01, reg 0.05, batch 64):
best epoch **2**, val RMSE **1.0023**, test RMSE **1.0426** (baseline 1.2289),
Recall@10 **0.0151** (13 unseen ids, 82 users without a relevant candidate),
11/11 self-checks.

---

## 3. Repository layout

| Path | What it is |
|---|---|
| `index.html`, `style.css`, `data.js`, `u.data`, `u.item` | Provided page, styling and data layer — unchanged (byte-identical copies also live in `fixed/` and `site/`) |
| `script.js` | The **starter** with the six throwing stubs — kept unchanged as the diff base |
| `prompt.md` | The original assignment readme, as received |
| `prompt_fixed.md` | The same document repaired in this session: full citation + license note, sparsity and split exclusions, Adam-epsilon warning, `mu +` dot product, §6 Recall fix + unclamped Top-K score, new §6.1 pinned training design, §7 item 4 checkpoint requirement, new §7.1 verification checklist, guidelines reference (see `check-16` for the exact diff) |
| `fixed/` | The six stubs implemented in this session from the pinned design and **checked against the numpy reference** (`check-4`–`check-9`, `check-11`–`check-13`); also the page trained at reg 0.1 and on the cpu backend. Not deployed |
| `site/` | The six stubs **re-generated from `prompt_fixed.md` only** (starter + fresh implementation, no code shared with `fixed/`); passes the §7.1 verification and is **deployed to GitHub Pages** |
| `.github/workflows/pages.yml` | Publishes `./site` to GitHub Pages on push to `main` and on workflow_dispatch |
| `audit_data.py` | Dataset + static audit of the readme/script claims (file hashes, shapes, split, baseline, stub counter, sketch-pitfall notes) |
| `app_probe.py` | Drives the real page in headless Chromium (serves the folder, presses buttons, reads every panel, exports the model) |
| `crosscheck.py` | Recomputes every number the page reported, in numpy, from the page's exported factors |
| `mf_core.py` | numpy reference: parsing, split, baseline, RMSE, Top-K, Recall, Adam replay, ALS-WR |
| `mf_reference.py` | Training-side experiments: step-level replay, reg sweep, ALS-WR, training-curve plot |
| `user_card.py` | One user's history next to their Top-10 (familiar/other/new labels, test-period hits, pairs with optional answers) |
| `coverage_check.py` | All-user business comparison: MF vs popularity vs evidence-floored MF (hits, catalogue use, thin evidence, film age) |
| `requirements.txt` | Python dependencies for the tools above |
| `data/` | **git-ignored** — the official GroupLens `ml-100k` archive used for the byte-identity check |
| `results/` | **git-ignored** — all `check-*.txt` artifacts and the exported models (`page_model.json`, `site_model.json`, …); the report carries them |
| `figures/` | **git-ignored** — `fig1_training.png`, `fig2_fixed.png`, `fig2_site.png` |

---

## 4. Reproducing the artifacts

Every `results/check-*.txt` is produced by one command from the repository root
(the tools need the `.venv` — `pip install -r requirements.txt` — and
`playwright install chromium` for the probes; the page itself loads TensorFlow.js
from the CDN, so the first probe run needs network access):

| Artifact | Command |
|---|---|
| `check-1-audit-starter.txt` | `.venv/bin/python3 audit_data.py --official data/ml-100k \| tee results/check-1-audit-starter.txt` |
| `check-2-probe-starter.txt` | `.venv/bin/python3 app_probe.py --page . --inspect --train --epochs 1 --users 1 --recall \| tee results/check-2-probe-starter.txt` |
| `check-3-probe-sketch.txt` | `.venv/bin/python3 app_probe.py --page fixed --inspect --api --train --epochs 1 \| tee results/check-3-probe-sketch.txt` *(historical: `fixed/` was sketch-literal at the time and has since been corrected in place, so re-running no longer reproduces the TypeError)* |
| `check-4-probe-sketch-nodispose.txt` | `.venv/bin/python3 app_probe.py --page fixed --inspect --train --capture-init --users 1 2 --recall --export results/page_sketch.json --retrain \| tee results/check-4-probe-sketch-nodispose.txt` |
| `check-5-crosscheck-sketch.txt` | `.venv/bin/python3 crosscheck.py --model results/page_sketch.json \| tee results/check-5-crosscheck-sketch.txt` |
| `check-6-penalty-form.txt` | `.venv/bin/python3 mf_reference.py --sweep-reg 0 0.001 0.05 --epochs 3 --init-from results/page_sketch.json \| tee results/check-6-penalty-form.txt` |
| `check-7-probe-fixed.txt` | `.venv/bin/python3 app_probe.py --page fixed --inspect --api --train --capture-init --users 1 2 --recall --export results/page_model.json --screenshot figures/fig2_fixed.png --retrain \| tee results/check-7-probe-fixed.txt` |
| `check-8-crosscheck.txt` | `.venv/bin/python3 crosscheck.py --model results/page_model.json \| tee results/check-8-crosscheck.txt` |
| `check-9-replay.txt` | `.venv/bin/python3 mf_reference.py --replay results/page_model.json \| tee results/check-9-replay.txt` |
| `check-10-user1.txt` | `.venv/bin/python3 user_card.py --user 1 --pairs 3 > results/check-10-user1.txt` *(add `--answer` to print the pair RMSE and recall_u)* |
| `check-11-sweep.txt` | `.venv/bin/python3 mf_reference.py --sweep-reg 0 0.01 0.05 0.1 0.2 --init-from results/page_model.json \| tee results/check-11-sweep.txt` |
| `check-12-probe-reg01.txt` | `.venv/bin/python3 app_probe.py --page fixed --train --reg 0.1 --recall --export results/page_reg01.json \| tee results/check-12-probe-reg01.txt` |
| `check-12b-probe-cpu.txt` | `.venv/bin/python3 app_probe.py --page fixed --backend cpu --train --epochs 1 --capture-init --export results/page_cpu.json \| tee results/check-12b-probe-cpu.txt` |
| `check-12c-replay-cpu.txt` | `.venv/bin/python3 mf_reference.py --replay results/page_cpu.json \| tee results/check-12c-replay-cpu.txt` |
| `check-13-als.txt` | `.venv/bin/python3 mf_reference.py --als \| tee results/check-13-als.txt` |
| `check-14-users.txt` | `.venv/bin/python3 user_card.py --user 1 --user 2 \| tee results/check-14-users.txt` |
| `check-15-coverage.txt` | `.venv/bin/python3 coverage_check.py --min-pop 20 \| tee results/check-15-coverage.txt` |
| `check-16-prompt-diff.txt` | `diff prompt.md prompt_fixed.md \| tee results/check-16-prompt-diff.txt` |
| `check-17-probe-site.txt` | `.venv/bin/python3 app_probe.py --page site --inspect --api --train --capture-init --users 1 2 --recall --export results/site_model.json --screenshot figures/fig2_site.png --retrain \| tee results/check-17-probe-site.txt` |
| `check-18-audit-site.txt` | `.venv/bin/python3 audit_data.py --script site/script.js \| tee results/check-18-audit-site.txt` |
| `check-19-crosscheck-site.txt` | `.venv/bin/python3 crosscheck.py --model results/site_model.json \| tee results/check-19-crosscheck-site.txt` |
| `check-20-replay-site.txt` | `.venv/bin/python3 mf_reference.py --replay results/site_model.json \| tee results/check-20-replay-site.txt` |
| `check-21-diff-site.txt` | `git diff --no-index script.js site/script.js \| tee results/check-21-diff-site.txt` |
| `check-22-static-files.txt` | `(for f in index.html style.css data.js u.data u.item; do cmp $f fixed/$f; cmp $f site/$f; done; echo "static files compared") 2>&1 \| tee results/check-22-static-files.txt` |
| `figures/fig1_training.png` | `.venv/bin/python3 mf_reference.py --plot figures/fig1_training.png --page-json results/page_model.json` |

Commands that say `--page fixed` / `--page site` train the page (about 5 minutes
on SwiftShader webgl; pass `--chrome` to use a real GPU). To run the page by hand:
`python3 -m http.server 8000` inside `fixed/` or `site/`, then open
`http://localhost:8000/`.

---

## 5. Data attribution

The dataset is **MovieLens 100K**, collected by the GroupLens Research Project at
the **University of Minnesota**
(<https://grouplens.org/datasets/movielens/100k/>). Required citation:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
> and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
> Article 19 (December 2015), 19 pages. <https://doi.org/10.1145/2827872>

License (from the archive README): the data may be used for research purposes
with the conditions that no endorsement by the University of Minnesota or
GroupLens may be stated or implied, **use of the dataset must be acknowledged via
the citation above**, the data **may not be redistributed without separate
permission**, and commercial use requires prior permission from GroupLens. The
copies in this repository (`u.data`, `u.item` at the root and in `fixed/`,
`site/`) are byte-identical to the official release — sha256 recorded in
`results/check-1-audit-starter.txt`.
