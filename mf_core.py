#!/usr/bin/env python3
"""
mf_core.py — independent numpy reference for HW5 (MovieLens 100K matrix factorization).

It re-implements, without TensorFlow.js, everything the page computes, so the page can be checked
against it: parsing of u.item / u.data (as data.js does), the per-user 80/10/10 timestamp split,
the training-mean baseline, RMSE with clamping to [1, 5], Top-K with train/val exclusions,
Recall@K with its two exclusion lists, and two ways to fit r_hat = mu + p_u . q_i:
  * adam_mf  — mini-batch Adam with the same loss as the page (TF.js Adam update rule, dense
               moments for every row), optionally from the page's own initial factors and the
               page's batch order (mulberry32 shuffle), so a short run can be compared step for step;
  * als_wr   — alternating least squares with weighted-lambda regularization (Zhou et al. 2008),
               a different algorithm for the same per-rating L2 objective.
Imported by the check scripts; running it directly prints a short self-test.
"""
import math
import pathlib
import re

import numpy as np

GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
          "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


# ---------------------------------------------------------------- data (mirrors data.js)
def load(folder="."):
    """Parse u.item and u.data exactly as data.js does. Returns (movies, ratings).
    movies: dict id -> {"title", "genres", "release", "year"}; ratings: list of (user, movie, rating, ts) in file order.
    "year" is the film's year from the title, e.g. "Pather Panchali (1955)" -> 1955; the release-date field is often a
    later US (re-)release (22-Mar-1996 for that film), so it is used only when the title has no year."""
    folder = pathlib.Path(folder)
    movies = {}
    for line in folder.joinpath("u.item").read_bytes().decode("latin-1").split("\n"):
        if not line.strip():
            continue
        f = line.split("|")
        if len(f) < 24:
            continue
        flags = [int(x) if x.strip().lstrip("-").isdigit() else 0 for x in f[6:24]]
        ty = re.search(r"\((\d{4})\)\s*$", f[1])
        year = int(ty.group(1)) if ty else (int(f[2][-4:]) if f[2][-4:].isdigit() else None)
        movies[int(f[0])] = {"title": f[1], "release": f[2], "year": year, "genres": [g for g, v in zip(GENRES, flags) if v == 1]}
    ratings = []
    for line in folder.joinpath("u.data").read_text(encoding="latin-1").split("\n"):
        if not line.strip():
            continue
        f = line.split("\t")
        if len(f) < 4:
            continue
        ratings.append((int(f[0]), int(f[1]), float(f[2]), int(f[3])))
    return movies, ratings


def split(ratings, min_ratings=5):
    """Per-user chronological split of data.js: users in order of first appearance, each user's ratings
    sorted by (timestamp, movieId); nTrain = floor(0.8 n), nVal = floor(0.1 n), test = the rest.
    Python floats are IEEE doubles, so floor(0.8 * n) is the same number JavaScript computes."""
    by_user = {}
    for r in ratings:
        by_user.setdefault(r[0], []).append(r)
    train, val, test, skipped = [], [], [], []
    for u, rs in by_user.items():
        n = len(rs)
        if n < min_ratings:
            skipped.append((u, n))
            continue
        rs = sorted(rs, key=lambda r: (r[3], r[1]))
        n_train, n_val = math.floor(0.8 * n), math.floor(0.1 * n)
        train += rs[:n_train]
        val += rs[n_train:n_train + n_val]
        test += rs[n_train + n_val:]
    return train, val, test, skipped


def arrays(rs):
    """(users int32, items int32, ratings float32) arrays of a split."""
    a = np.array([(r[0], r[1]) for r in rs], dtype=np.int64).reshape(-1, 2)
    return a[:, 0].astype(np.int32), a[:, 1].astype(np.int32), np.array([r[2] for r in rs], dtype=np.float32)


def train_mean(train):
    """Mean training rating, summed left to right in double precision like Array.reduce in JavaScript."""
    s = 0.0
    for r in train:
        s += r[2]
    return s / len(train) if train else 0.0


def baseline_rmse(train, rs):
    """RMSE of predicting the training mean for every rating of rs (computeBaselineRMSE)."""
    if not rs:
        return 0.0
    mu = train_mean(train)
    s = 0.0
    for r in rs:
        s += (mu - r[2]) ** 2
    return math.sqrt(s / len(rs))


# ---------------------------------------------------------------- model evaluation
def predict(P, Q, mu, users, items, clamp=True):
    """mu + p_u . q_i in float32 (as the page's tensors), clamped to [1, 5] unless clamp=False."""
    dot = np.einsum("ij,ij->i", P[users].astype(np.float32), Q[items].astype(np.float32), dtype=np.float32)
    pred = dot.astype(np.float64) + mu
    return np.clip(pred, 1.0, 5.0) if clamp else pred


def rmse(P, Q, mu, rs):
    """RMSE of the factor model over a split (the page's computeRMSEFor)."""
    if not rs:
        return 0.0
    u, i, y = arrays(rs)
    return float(np.sqrt(np.mean((predict(P, Q, mu, u, i) - y) ** 2)))


def rated_by_user(*splits):
    out = {}
    for rs in splits:
        for r in rs:
            out.setdefault(r[0], set()).add(r[1])
    return out


def top_k(P, Q, mu, user, k, movie_ids, rated):
    """Top-k movie ids for user by unclamped score mu + p_u . q_i, skipping movies rated in train/val and
    ids that are not in u.item. Returns [(movie_id, score)], ties broken by the smaller movie id."""
    scores = (Q.astype(np.float32) @ P[user].astype(np.float32)).astype(np.float64) + mu
    eligible = np.array(sorted(set(movie_ids) - rated.get(user, set())), dtype=np.int64)
    if k <= 0 or eligible.size == 0:
        return []
    s = scores[eligible]
    order = np.lexsort((eligible, -s))[:k]
    return [(int(eligible[j]), float(s[j])) for j in order]


def recall_at_k(P, Q, mu, train, val, test, movie_ids, k=10):
    """Recall@k as defined in the readme (section 6): relevant = test rating >= 4.0 among movies seen anywhere
    in train; mean over users with at least one such movie. Returns (overall, per_user, unseen_ids, users_without)."""
    cand = {r[1] for r in train}
    rated = rated_by_user(train, val)
    relevant, test_users, unseen = {}, set(), set()
    for r in test:
        test_users.add(r[0])
        if r[2] < 4.0:
            continue
        if r[1] not in cand:
            unseen.add(r[1])
            continue
        relevant.setdefault(r[0], set()).add(r[1])
    per_user, without = [], []
    for u in sorted(test_users):
        rel = relevant.get(u)
        if not rel:
            without.append(u)
            continue
        hits = sum(m in rel for m, _ in top_k(P, Q, mu, u, k, movie_ids, rated))
        per_user.append((u, len(rel), hits))
    overall = sum(h / n for _, n, h in per_user) / len(per_user) if per_user else 0.0
    return overall, per_user, sorted(unseen), without


# ---------------------------------------------------------------- training
def mulberry32(seed):
    """The page's seeded shuffle generator (same 32-bit integer arithmetic as makeRandom in script.js)."""
    a = seed & 0xFFFFFFFF

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return nxt


def adam_mf(train, val, n_users, n_items, *, epochs=20, lr=0.01, reg=0.05, batch=64, factors=16, l2="batch",
            seed=0, init=None, shuffle_seed=7, restore_best=True, eps=1e-7, log=None, keep=None):
    """Mini-batch Adam on mean squared error + L2, the update rule of tf.train.adam (beta1 0.9, beta2 0.999,
    bias-corrected, dense moments: rows that are not in the batch still move by momentum). tf.train.adam(lr)
    takes epsilon from the backend: 1e-7 on cpu, 1e-4 on webgl (TF.js 4.22 webgl reports float precision 16
    whenever float32 rendering is enabled), so pass the page's tf.backend().epsilon() as eps.
      l2 = "batch":  reg * mean over the batch of (|p_u|^2 + |q_i|^2)       (per-rating penalty)
      l2 = "matrix": reg * (|P|_F^2 + |Q|_F^2) added to every batch loss    (the readme's sketch)
    init = (P0, Q0) starts from given factors (e.g. the page's); otherwise N(0, 0.05^2) from numpy's seed.
    Shuffle: mulberry32(shuffle_seed) Fisher-Yates over the rating order each epoch, as in script.js.
    Returns (P, Q, mu, history) with history rows (epoch, train_rmse, val_rmse); with restore_best the returned
    factors are those of the epoch with the smallest validation RMSE (earliest on ties). A dict passed as keep
    receives keep["last"] = (P, Q) after the last epoch and keep["best"] = (epoch, P, Q) of the selected epoch."""
    mu = train_mean(train)
    if init is None:
        rng = np.random.default_rng(seed)
        P = rng.normal(0, 0.05, (n_users, factors)).astype(np.float32)
        Q = rng.normal(0, 0.05, (n_items, factors)).astype(np.float32)
    else:
        P, Q = (np.array(x, dtype=np.float32).copy() for x in init)
    mP, vP, mQ, vQ = (np.zeros_like(x) for x in (P, P, Q, Q))
    b1, b2 = 0.9, 0.999
    acc1, acc2 = np.float32(b1), np.float32(b2)  # tf.train.adam keeps beta^t as float32 variables
    u_all, i_all, y_all = arrays(train)
    n = len(train)
    size = min(max(1, batch), n)
    order = list(range(n))
    rnd = mulberry32(shuffle_seed)
    history, best = [], None
    for epoch in range(1, epochs + 1):
        for k in range(n - 1, 0, -1):
            j = math.floor(rnd() * (k + 1))
            order[k], order[j] = order[j], order[k]
        o = np.array(order)
        us, its, ys = u_all[o], i_all[o], y_all[o]
        for start in range(0, n, size):
            u, i, y = us[start:start + size], its[start:start + size], ys[start:start + size]
            m = len(y)
            pu, qi = P[u], Q[i]
            pred = np.einsum("ij,ij->i", pu, qi, dtype=np.float32) + np.float32(mu)
            g = (np.float32(2.0 / m) * (pred - y))[:, None]
            gP = np.zeros_like(P)
            gQ = np.zeros_like(Q)
            if l2 == "batch":
                np.add.at(gP, u, g * qi + np.float32(2.0 * reg / m) * pu)
                np.add.at(gQ, i, g * pu + np.float32(2.0 * reg / m) * qi)
            elif l2 == "matrix":
                np.add.at(gP, u, g * qi)
                np.add.at(gQ, i, g * pu)
                gP += np.float32(2.0 * reg) * P
                gQ += np.float32(2.0 * reg) * Q
            else:
                raise ValueError(l2)
            c1, c2 = np.float32(1) - acc1, np.float32(1) - acc2
            for W, G, M, V in ((P, gP, mP, vP), (Q, gQ, mQ, vQ)):
                M *= np.float32(b1)
                M += G * np.float32(1 - b1)
                V *= np.float32(b2)
                V += (G * G) * np.float32(1 - b2)
                W += ((M / c1) / (np.sqrt(V / c2) + np.float32(eps))) * np.float32(-lr)
            acc1 = np.float32(acc1 * np.float32(b1))
            acc2 = np.float32(acc2 * np.float32(b2))
        row = (epoch, rmse(P, Q, mu, train), rmse(P, Q, mu, val))
        history.append(row)
        if log:
            log(row)
        if best is None or row[2] < best[0]:
            best = (row[2], P.copy(), Q.copy(), epoch)
    if keep is not None:
        keep["last"] = (P.copy(), Q.copy())
        keep["best"] = (best[3], best[1], best[2]) if best else None
    if restore_best and best is not None:
        P, Q = best[1], best[2]
    return P, Q, mu, history


def als_wr(train, n_users, n_items, *, reg=0.05, factors=16, iters=15, seed=0, val=None, log=None):
    """Alternating least squares with weighted-lambda regularization: minimizes
    sum over train of (r - mu - p_u . q_i)^2 + reg * (sum_u n_u |p_u|^2 + sum_i n_i |q_i|^2),
    i.e. the per-rating ("batch") L2 objective of adam_mf multiplied by N. A row with no training rating is never
    solved for and keeps its random start."""
    mu = train_mean(train)
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.05, (n_users, factors))
    Q = rng.normal(0, 0.05, (n_items, factors))
    by_u, by_i = {}, {}
    for (u, i, r, _) in train:
        by_u.setdefault(u, []).append((i, r - mu))
        by_i.setdefault(i, []).append((u, r - mu))
    eye = np.eye(factors)
    hist = []
    for it in range(1, iters + 1):
        for rows, fixed, groups in ((P, Q, by_u), (Q, P, by_i)):
            for a, obs in groups.items():
                idx = np.array([o[0] for o in obs])
                y = np.array([o[1] for o in obs])
                F = fixed[idx]
                rows[a] = np.linalg.solve(F.T @ F + reg * len(obs) * eye, F.T @ y)
        row = (it, rmse(P, Q, mu, train), rmse(P, Q, mu, val) if val is not None else float("nan"))
        hist.append(row)
        if log:
            log(row)
    return P.astype(np.float32), Q.astype(np.float32), mu, hist


if __name__ == "__main__":
    movies, ratings = load(pathlib.Path(__file__).parent if pathlib.Path(__file__).parent.joinpath("u.data").exists() else ".")
    tr, va, te, sk = split(ratings)
    print(f"movies {len(movies)}, ratings {len(ratings)}, users {len({r[0] for r in ratings})}")
    print(f"split train {len(tr)}, val {len(va)}, test {len(te)}, skipped users {len(sk)}")
    print(f"mu {train_mean(tr):.6f}; baseline RMSE val {baseline_rmse(tr, va):.4f}, test {baseline_rmse(tr, te):.4f}")
