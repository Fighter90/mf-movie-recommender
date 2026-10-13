#!/usr/bin/env python3
"""
crosscheck.py — recompute in numpy, without TensorFlow.js, every number the page reported, from the
page's own trained factors (results/page_model.json written by app_probe.py --export).

    python3 crosscheck.py --model results/page_model.json

Checks: training mean and split sizes; baseline RMSE; RMSE of the exported factors on validation and
test (and that the model in memory is the selected checkpoint); the Top-10 lists requested in the
page (ids, order, scores, no train/val movie); Recall@10 with both exclusion lists, user by user.
"""
import argparse, json
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

import mf_core as mc

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default="results/page_model.json")
ap.add_argument("--data", default=".", help="folder with u.data and u.item")
a = ap.parse_args()

m = json.load(open(a.model, encoding="utf-8"))
movies, ratings = mc.load(a.data)
train, val, test, _ = mc.split(ratings)
bad = 0


def verdict(ok, text):
    global bad
    bad += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {text}")


def to_fixed(x, digits):
    # Number.prototype.toFixed rounds the exact binary value half-up; Python's format rounds ties to even
    return str(Decimal(x).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


f = m.get("factors")
if not f:
    raise SystemExit("no factors in the export: run app_probe.py with --train --export")
P = np.array(f["users"], dtype=np.float32).reshape(f["userShape"])
Q = np.array(f["items"], dtype=np.float32).reshape(f["itemShape"])
mu = mc.train_mean(train)
print(f"[model] userFactors {list(P.shape)}, itemFactors {list(Q.shape)}; |P| {np.linalg.norm(P):.4f}, |Q| {np.linalg.norm(Q):.4f}")
verdict(list(P.shape) == [944, 16] and list(Q.shape) == [1683, 16], "factor shapes [944, 16] and [1683, 16] (raw ids as row indices)")
verdict(abs(m["globalMean"] - mu) < 1e-9, f"globalMean {m['globalMean']:.6f} = training mean recomputed {mu:.6f}")

bv, bt = mc.baseline_rmse(train, val), mc.baseline_rmse(train, test)
verdict(abs(m["baseVal"] - bv) < 1e-9 and abs(m["baseTest"] - bt) < 1e-9,
        f"baseline RMSE validation {bv:.4f}, test {bt:.4f} (page {m['baseVal']:.4f}, {m['baseTest']:.4f})")

rv, rt, rtr = mc.rmse(P, Q, mu, val), mc.rmse(P, Q, mu, test), mc.rmse(P, Q, mu, train)
verdict(abs(m["valNow"] - rv) < 1e-5, f"validation RMSE of the exported factors {rv:.6f} (page {m['valNow']:.6f})")
verdict(abs(m["testRMSE"] - rt) < 1e-5, f"test RMSE of the exported factors {rt:.6f} (page {m['testRMSE']:.6f})")
best = m.get("best")
if best:
    verdict(abs(best["valRMSE"] - rv) < 1e-5,
            f"selected checkpoint (epoch {best['epoch']}, val {best['valRMSE']:.6f}) is the model that was exported (val {rv:.6f}); "
            f"its train RMSE {rtr:.6f} (page {best['trainRMSE']:.6f})")
    hist = m.get("history") or []
    first_min = min(hist, key=lambda r: (r["valRMSE"], r["epoch"])) if hist else None
    verdict(first_min is not None and first_min["epoch"] == best["epoch"],
            f"epoch {best['epoch']} has the smallest validation RMSE in the history (earliest on ties)")
verdict(bt - rt > 0.001, f"MF test RMSE {rt:.4f} beats the training-mean baseline {bt:.4f} by more than 0.001 (difference {bt - rt:+.4f})")
tu, ti, _ = mc.arrays(test)
spread = float(np.std(np.einsum("ij,ij->i", P[tu], Q[ti])))
verdict(spread > 0.01, f"the factors carry a signal: std of p_u . q_i over the test pairs {spread:.4f} > 0.01 "
        f"(0 means every prediction is the mean {mu:.4f})")
seen = {r[1] for r in train}
t_in, t_out = [r for r in test if r[1] in seen], [r for r in test if r[1] not in seen]
unseen_rows = sorted({r[1] for r in t_out})
same = ""
if m.get("init") and unseen_rows:
    Q0 = np.array(m["init"]["items"], dtype=np.float32).reshape(m["init"]["itemShape"])
    moved = float(np.abs(Q[unseen_rows] - Q0[unseen_rows]).max())
    same = (f"; their factor rows changed during training by at most {moved:.1e} (no training rating reaches them"
            + (", so only a penalty on the whole matrix can move them)" if moved > 0 else ")"))
print(f"[info] test ratings of movies absent from training: {len(t_out)} ({len(unseen_rows)} movies){same}; RMSE on them "
      f"{mc.rmse(P, Q, mu, t_out):.4f} (baseline {mc.baseline_rmse(train, t_out):.4f}); on the other {len(t_in)}: "
      f"{mc.rmse(P, Q, mu, t_in):.4f} (baseline {mc.baseline_rmse(train, t_in):.4f})")

rated = mc.rated_by_user(train, val)
ids = list(movies)
for u, rec in (m.get("topk") or {}).items():
    u = int(u)
    raw = rec["raw"]
    ref = mc.top_k(P, Q, mu, u, 10, ids, rated)
    same_ids = [r[0] for r in raw] == [r[0] for r in ref]
    if not same_ids:  # allow swaps between scores equal to float32 precision
        same_ids = sorted(r[0] for r in raw) == sorted(r[0] for r in ref) and all(abs(x[2] - y[1]) < 1e-5 for x, y in zip(raw, ref))
    max_diff = max((abs(x[2] - y[1]) for x, y in zip(raw, ref)), default=0.0)
    verdict(same_ids and len(raw) == len(ref) and max_diff < 1e-4,
            f"user {u}: Top-10 ids and order equal to numpy, max |score diff| {max_diff:.1e}")
    clash = [r[0] for r in raw if r[0] in rated.get(u, set())]
    verdict(not clash, f"user {u}: none of the 10 movies was rated by the user in train or validation ({len(rated.get(u, set()))} rated)")
    shown = [s for _, s in rec.get("rendered", [])]
    verdict(shown == [to_fixed(x[2], 4) for x in raw],
            f"user {u}: the page renders recommendTopK's scores ({', '.join(shown[:3])}, ...) with fmt(score, 4)")

r = m.get("recall")
if r:
    overall, per_user, unseen, without = mc.recall_at_k(P, Q, mu, train, val, test, ids, 10)
    verdict(abs(r["overall"] - overall) < 1e-9, f"Recall@10 {overall:.6f} (page {r['overall']:.6f})")
    page_users = {(x["userId"], x["relevant"], x["hit"]) for x in r["perUser"]}
    verdict(page_users == set(per_user), f"per-user relevant and hit counts equal for all {len(per_user)} evaluated users"
            + ("" if page_users == set(per_user) else f" (differ for {len(page_users ^ set(per_user))} rows)"))
    verdict(sorted(r["unseen"]) == unseen, f"unseen relevant test ids: {len(unseen)} (page {len(r['unseen'])})")
    verdict(sorted(r["without"]) == without, f"users without a relevant candidate: {len(without)} (page {len(r['without'])})")
    print(f"[recall] hits {sum(h for _, _, h in per_user)} over {len(per_user)} users; users with at least one hit {sum(h > 0 for _, _, h in per_user)}")
print(f"[summary] failed checks: {bad}")
