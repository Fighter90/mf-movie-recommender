#!/usr/bin/env python3
"""
mf_reference.py — the training side of HW5 in numpy (mf_core.py), to check what the page's trainMF
does and to see what the loss and the hyper-parameters change. Page defaults: 20 epochs, Adam lr 0.01,
L2 0.05, batch 64, 16 factors.

    python3 mf_reference.py --replay results/page_short.json   # page's initial factors + batch order, epoch by epoch
    python3 mf_reference.py --l2 batch --init-from results/page_model.json    # per-rating penalty from the page's start
    python3 mf_reference.py --l2 matrix --init-from results/page_model.json   # whole-matrix penalty (the readme sketch)
    python3 mf_reference.py --sweep-reg 0 0.01 0.05 0.1 0.2 --init-from results/page_model.json
    python3 mf_reference.py --als                               # ALS-WR, another algorithm for the per-rating objective
    python3 mf_reference.py --plot figures/fig1_training.png --page-json results/page_model.json

Adam's epsilon: tf.train.adam(lr) takes it from the TensorFlow.js backend (TF.js 4.22: 1e-4 on webgl,
1e-7 on cpu). --replay reads the backend's value from the export; elsewhere --eps sets it (default
1e-4, the value on webgl). --init-from takes the page's initial factors from an export made with
app_probe.py --capture-init, so a run here starts exactly where the page started.
"""
import argparse, json, time

import numpy as np

import mf_core as mc

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", default=".", help="folder with u.data and u.item")
ap.add_argument("--replay", help="app_probe.py export made with --capture-init (short run)")
ap.add_argument("--l2", choices=["batch", "matrix"])
ap.add_argument("--als", action="store_true")
ap.add_argument("--sweep-reg", nargs="*", type=float)
ap.add_argument("--plot", help="training-curve figure (PNG)")
ap.add_argument("--page-json", help="app_probe.py export whose history is drawn in --plot")
ap.add_argument("--init-from", help="app_probe.py export with captured initial factors (--capture-init)")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--reg", type=float, default=0.05)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--eps", type=float, default=1e-4, help="Adam epsilon (1e-4 = tf.train.adam on webgl, 1e-7 on cpu)")
ap.add_argument("--seed", type=int, default=0, help="numpy seed of the initial factors when --init-from is not given")
a = ap.parse_args()

movies, ratings = mc.load(a.data)
train, val, test, _ = mc.split(ratings)
ids = list(movies)
NU, NI = max(r[0] for r in ratings) + 1, max(movies) + 1
bt = mc.baseline_rmse(train, test)
bv = mc.baseline_rmse(train, val)
print(f"[data] train {len(train)}, validation {len(val)}, test {len(test)}; baseline RMSE validation {bv:.4f}, test {bt:.4f}")


def factors(block):
    return (np.array(block["users"], dtype=np.float32).reshape(block["userShape"]),
            np.array(block["items"], dtype=np.float32).reshape(block["itemShape"]))


def page_init(path):
    m = json.load(open(path, encoding="utf-8"))
    if not m.get("init"):
        raise SystemExit(f"{path} has no initial factors: run app_probe.py --train --capture-init --export ...")
    return m, factors(m["init"])


start, start_note = None, f"numpy N(0, 0.05^2) start, seed {a.seed}"
if a.init_from:
    _, start = page_init(a.init_from)
    start_note = f"the page's initial factors from {a.init_from}"


def summary(tag, P, Q, mu, hist, t0):
    best = min(hist, key=lambda r: (r[2], r[0]))
    rec = mc.recall_at_k(P, Q, mu, train, val, test, ids, 10)
    print(f"[{tag}] best epoch {best[0]}: train {best[1]:.4f}, val {best[2]:.4f}; test {mc.rmse(P, Q, mu, test):.4f} "
          f"(baseline {bt:.4f}); Recall@10 {rec[0]:.4f} over {len(rec[1])} users; |P| {np.linalg.norm(P):.3f}, |Q| {np.linalg.norm(Q):.3f}; "
          f"{time.time() - t0:.0f} s")
    return best, rec


if a.replay:
    m, (P0, Q0) = page_init(a.replay)
    hp = m["hyper"]
    tfinfo = m.get("tf") or {}
    eps_page = tfinfo.get("epsilon") or (1e-4 if "webgl" in str(m.get("backend", "")) else 1e-7)
    eps_other = 1e-7 if eps_page != 1e-7 else 1e-4
    print(f"[replay] page hyper-parameters {hp}; backend {tfinfo.get('backend', '?')}, Adam epsilon {eps_page:g}; "
          f"initial |P| {np.linalg.norm(P0):.4f}, |Q| {np.linalg.norm(Q0):.4f}")
    t0 = time.time()
    page_hist = m["history"]
    exported = factors(m["factors"]) if m.get("factors") else None
    for l2, eps in (("batch", eps_page), ("batch", eps_other), ("matrix", eps_page)):
        keep = {}
        _, _, mu, hist = mc.adam_mf(train, val, NU, NI, epochs=hp["epochs"], lr=hp["lr"], reg=hp["reg"], batch=hp["batchSize"],
                                    l2=l2, init=(P0, Q0), restore_best=False, eps=eps, keep=keep)
        worst = max(max(abs(tr - row["trainRMSE"]), abs(va - row["valRMSE"])) for (e, tr, va), row in zip(hist, page_hist))
        line = (f"[replay] numpy, {l2} penalty, epsilon {eps:g}: "
                + "; ".join(f"epoch {e} train {tr:.4f} val {va:.4f}" for e, tr, va in hist[:3])
                + (" ..." if len(hist) > 3 else "") + f"  -> max |RMSE difference| from the page {worst:.1e}")
        if exported is not None:
            be, Pb, Qb = keep["best"]
            d_last = max(np.abs(keep["last"][0] - exported[0]).max(), np.abs(keep["last"][1] - exported[1]).max())
            d_best = max(np.abs(Pb - exported[0]).max(), np.abs(Qb - exported[1]).max())
            line += f"; exported factors vs numpy epoch {be} (best) {d_best:.1e}, vs epoch {hist[-1][0]} (last) {d_last:.1e}"
        print(line)
    print("[replay] page: " + "; ".join(f"epoch {r['epoch']} train {r['trainRMSE']:.4f} val {r['valRMSE']:.4f}" for r in page_hist[:3])
          + (" ..." if len(page_hist) > 3 else ""))
    print(f"[replay] {time.time() - t0:.0f} s. The variant whose differences are smaller than the others' by orders of magnitude "
          "(float32 rounding: about 1e-7 in RMSE and 1e-5 in the factors; other GPUs may give up to about 1e-4) is the computation "
          "the page performs, and the epoch whose factors match is the one the page keeps in memory.")

if a.l2:
    t0 = time.time()
    P, Q, mu, hist = mc.adam_mf(train, val, NU, NI, epochs=a.epochs, lr=a.lr, reg=a.reg, batch=a.batch, l2=a.l2,
                                seed=a.seed, init=start, eps=a.eps)
    print(f"[{a.l2}] {start_note}; Adam epsilon {a.eps:g}; val RMSE by epoch: " + " ".join(f"{r[2]:.4f}" for r in hist))
    summary(a.l2, P, Q, mu, hist, t0)

if a.als:
    t0 = time.time()
    P, Q, mu, hist = mc.als_wr(train, NU, NI, reg=a.reg, iters=15, seed=a.seed, val=val)
    print("[als] val RMSE by iteration: " + " ".join(f"{r[2]:.4f}" for r in hist))
    best = min(hist, key=lambda r: (r[2], r[0]))
    rec = mc.recall_at_k(P, Q, mu, train, val, test, ids, 10)
    print(f"[als] ALS-WR lambda {a.reg}: best iteration {best[0]} val {best[2]:.4f}; after 15 iterations train {hist[-1][1]:.4f}, "
          f"val {hist[-1][2]:.4f}, test {mc.rmse(P, Q, mu, test):.4f}; Recall@10 {rec[0]:.4f}; {time.time() - t0:.0f} s")

if a.sweep_reg:
    print(f"[sweep] {start_note}; Adam epsilon {a.eps:g}; best epoch by validation RMSE, factors of that epoch")
    print("[sweep] penalty  L2       best epoch  val     test    Recall@10  |P|")
    for l2 in ("batch", "matrix"):
        for reg in a.sweep_reg:
            P, Q, mu, hist = mc.adam_mf(train, val, NU, NI, epochs=a.epochs, lr=a.lr, reg=reg, batch=a.batch, l2=l2,
                                        seed=a.seed, init=start, eps=a.eps)
            best = min(hist, key=lambda r: (r[2], r[0]))
            rec = mc.recall_at_k(P, Q, mu, train, val, test, ids, 10)
            print(f"[sweep] {l2:7s}  {reg:<7g}  {best[0]:>10d}  {best[2]:.4f}  {mc.rmse(P, Q, mu, test):.4f}  {rec[0]:.4f}     {np.linalg.norm(P):.3f}")

if a.plot:
    import pathlib

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    if a.page_json:
        m = json.load(open(a.page_json, encoding="utf-8"))
        h = m["history"]
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["trainRMSE"] for r in h], "k-", marker="o", ms=3, lw=1.2, label="page: train RMSE")
        ax.plot(ep, [r["valRMSE"] for r in h], "k--", marker="s", ms=3, lw=1.2, label="page: validation RMSE")
        b = m.get("best")
        if b:
            ax.plot([b["epoch"]], [b["valRMSE"]], "o", ms=9, mfc="none", mec="black", label=f"selected checkpoint (epoch {b['epoch']})")
    ax.axhline(bv, color="0.55", ls=":", lw=1, label=f"mean baseline, validation {bv:.4f}")
    ax.axhline(bt, color="0.3", ls="-.", lw=1, label=f"mean baseline, test {bt:.4f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE")
    ax.set_xticks(range(0, a.epochs + 1, 2))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(0.99, 0.54))  # between the validation curve and the train curve
    fig.tight_layout()
    pathlib.Path(a.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.plot)
    print(f"[plot] {a.plot}")
