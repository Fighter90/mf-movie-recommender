#!/usr/bin/env python3
"""
user_card.py — one user's history next to the Top-10 the page recommended (familiar choices vs new
discoveries), from the page's own trained factors (results/page_model.json, app_probe.py --export).

    python3 user_card.py --model results/page_model.json --user 1 --user 2
    python3 user_card.py --model results/page_model.json --user 1 --pairs 3            # 3 test ratings, no RMSE, no recall_u
    python3 user_card.py --model results/page_model.json --user 1 --pairs 3 --answer   # ... plus their RMSE and recall_u

History = the user's train + validation ratings (everything before the test period).
Labels for a recommended movie (genres from u.item):
  familiar  at least one of its genres is among the user's three most-rated genres;
  new       none of its genres occurs anywhere in the user's history;
  other     the rest (genres the user has rated, but not among the top three).
"popularity" = number of ratings the movie has in the training split.
"""
import argparse, collections, datetime as dt, json, statistics

import numpy as np

import mf_core as mc

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default="results/page_model.json")
ap.add_argument("--data", default=".")
ap.add_argument("--user", type=int, action="append", required=True)
ap.add_argument("--pairs", type=int, default=0, help="print this many of the user's last test ratings with the model's predictions")
ap.add_argument("--answer", action="store_true", help="also print the RMSE over those pairs and the user's recall_u")
a = ap.parse_args()

m = json.load(open(a.model, encoding="utf-8"))
f = m["factors"]
P = np.array(f["users"], dtype=np.float32).reshape(f["userShape"])
Q = np.array(f["items"], dtype=np.float32).reshape(f["itemShape"])
movies, ratings = mc.load(a.data)
train, val, test, _ = mc.split(ratings)
mu = mc.train_mean(train)
pop = collections.Counter(r[1] for r in train)
rated = mc.rated_by_user(train, val)
cand = {r[1] for r in train}
day = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d")
year = {mid: v["year"] for mid, v in movies.items() if v["year"]}  # film year from the title
lists = {}

for u in a.user:
    hist = [r for r in train + val if r[0] == u]
    tst = sorted((r for r in test if r[0] == u), key=lambda r: (r[3], r[1]))
    g = collections.Counter(x for r in hist for x in movies[r[1]]["genres"])
    liked = [r for r in hist if r[2] >= 4]
    top3 = [x for x, _ in g.most_common(3)]
    print(f"=== user {u}: {len(hist)} ratings before the test period ({day(min(r[3] for r in hist))} .. {day(max(r[3] for r in hist))}), "
          f"mean {statistics.mean(r[2] for r in hist):.2f}, liked (>= 4) {len(liked)}; test period {len(tst)} ratings "
          f"({day(tst[0][3])} .. {day(tst[-1][3])})")
    print("[history] genres (share of the user's rated movies): " + ", ".join(f"{x} {c / len(hist):.0%}" for x, c in g.most_common(6)))
    print(f"[history] three most-rated genres: {', '.join(top3)}; genres never rated: "
          + (", ".join(x for x in mc.GENRES if g[x] == 0) or "none"))
    print(f"[history] median popularity of the movies the user rated: {statistics.median(pop[r[1]] for r in hist):g} training ratings; "
          f"median film year {statistics.median(year[r[1]] for r in hist if r[1] in year):g}, made before 1980: "
          f"{sum(year.get(r[1], 9999) < 1980 for r in hist)} of {len(hist)}")
    best = sorted(hist, key=lambda r: (-r[2], -pop[r[1]], r[1]))[:5]
    print("[history] highest-rated: " + "; ".join(f"{movies[r[1]]['title']} ({r[2]:g})" for r in best))

    recs = m["topk"][str(u)]["raw"] if str(u) in (m.get("topk") or {}) else \
        [(mid, movies[mid]["title"], s) for mid, s in mc.top_k(P, Q, mu, u, 10, list(movies), rated)]
    test_r = {r[1]: r[2] for r in tst}
    labels = collections.Counter()
    print("[top10] rank  id     score   pop  label     test  title | genres")
    for k, (mid, title, score) in enumerate(recs, 1):
        gs = movies[mid]["genres"]
        lab = "familiar" if any(x in top3 for x in gs) else ("new" if gs and all(g[x] == 0 for x in gs) else "other")
        labels[lab] += 1
        out = f"{test_r[mid]:g}" if mid in test_r else "-"
        print(f"        {k:4d}  {mid:5d}  {score:6.3f}  {pop[mid]:4d}  {lab:8s}  {out:>4s}  {title} | {', '.join(gs) or 'no genre'}")
    pops = [pop[mid] for mid, _, _ in recs]
    lists[u] = [mid for mid, _, _ in recs]
    print(f"[top10] familiar {labels['familiar']}, other {labels['other']}, new {labels['new']}; median popularity {statistics.median(pops):g} "
          f"training ratings; movies with fewer than 5 training ratings: {sum(p < 5 for p in pops)}; scores above 5: {sum(s > 5 for _, _, s in recs)}")
    print(f"[top10] median film year {statistics.median(year[mid] for mid, _, _ in recs if mid in year):g}, made before 1980: "
          f"{sum(year.get(mid, 9999) < 1980 for mid, _, _ in recs)} of {len(recs)}")

    rel = sorted(r[1] for r in tst if r[2] >= 4.0 and r[1] in cand)
    hits = [mid for mid, _, _ in recs if mid in rel]
    print(f"[recall] relevant test movies (>= 4.0, in the training pool): {len(rel)} -> {rel}")
    tail = (f" -> recall_u = {len(hits)}/{len(rel)} = {len(hits) / len(rel):.4f}" if a.answer else "") if rel else " (user not evaluated)"
    print(f"[recall] of them in the Top-10: {len(hits)} {hits}{tail}")

    if a.pairs:
        pairs = tst[-a.pairs:]
        preds = mc.predict(P, Q, mu, np.array([r[0] for r in pairs]), np.array([r[1] for r in pairs]))
        print(f"[pairs] the user's last {len(pairs)} test ratings and the model's predictions (mu + p_u . q_i, clamped to [1, 5]):")
        for r, p in zip(pairs, preds):
            print(f"        movie {r[1]:5d}  actual {r[2]:g}  predicted {p:.6f}  ({movies[r[1]]['title']})")
        if a.answer:
            err = np.sqrt(np.mean((preds - np.array([r[2] for r in pairs])) ** 2))
            print(f"[pairs] RMSE over these {len(pairs)} pairs: {err:.4f}")

if len(lists) >= 2:
    us = list(lists)
    common = set(lists[us[0]]).intersection(*map(set, lists.values()))
    print(f"[overlap] movies in the Top-10 of every user above ({', '.join(map(str, us))}): {len(common)}"
          + (" -> " + "; ".join(movies[mid]["title"] for mid in sorted(common)) if common else ""))
