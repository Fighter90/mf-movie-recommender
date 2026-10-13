#!/usr/bin/env python3
"""
coverage_check.py — the Top-10 lists of ALL users, from the page's trained factors, next to a popularity
recommender (the 10 most-rated training movies the user has not rated yet). Business view of
"familiar choices vs new discoveries": hits in the test period, how much of the catalogue the lists
use, how thin the evidence behind the recommended movies is, and how old they are.

    python3 coverage_check.py --model results/page_model.json
    python3 coverage_check.py --model results/page_model.json --min-pop 20   # + MF restricted to movies with >= 20 ratings

Definitions (all from the training split unless stated):
  popularity of a movie      number of training ratings it has;
  short head                 the most-rated movies that together hold one third of all training ratings;
                             every other movie is the long tail;
  thin evidence              fewer than 5 training ratings;
  familiar slot              the movie shares a genre with the user's three most-rated genres (train + val);
  hit                        a recommended movie the user rated >= 4.0 in the test period (as in Recall@10).
Recommendations exclude movies the user rated in train or validation, as recommendTopK does.
"""
import argparse, collections, json, statistics

import numpy as np

import mf_core as mc

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default="results/page_model.json")
ap.add_argument("--data", default=".")
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--min-pop", type=int, default=0, help="also show MF limited to movies with at least this many training ratings")
a = ap.parse_args()

m = json.load(open(a.model, encoding="utf-8"))
f = m["factors"]
P = np.array(f["users"], dtype=np.float32).reshape(f["userShape"])
Q = np.array(f["items"], dtype=np.float32).reshape(f["itemShape"])
movies, ratings = mc.load(a.data)
train, val, test, _ = mc.split(ratings)
mu = mc.train_mean(train)
ids = sorted(movies)
rated = mc.rated_by_user(train, val)
pop = collections.Counter(r[1] for r in train)
cand = set(pop)
users = sorted({r[0] for r in train})

# short head: most-rated movies holding one third of the training ratings
head, acc = set(), 0
for mid, c in sorted(pop.items(), key=lambda x: (-x[1], x[0])):
    if acc >= len(train) / 3:
        break
    head.add(mid)
    acc += c
year = {mid: v["year"] for mid, v in movies.items() if v["year"]}  # film year from the title

relevant = collections.defaultdict(set)
for r in test:
    if r[2] >= 4.0 and r[1] in cand:
        relevant[r[0]].add(r[1])

top3 = {}
for u in users:
    g = collections.Counter(x for mid in rated.get(u, ()) for x in movies[mid]["genres"])
    top3[u] = {x for x, _ in g.most_common(3)}

pop_order = [mid for mid, _ in sorted(((mid, pop.get(mid, 0)) for mid in ids), key=lambda x: (-x[1], x[0]))]


def pop_top(u):
    out = []
    for mid in pop_order:
        if mid not in rated.get(u, set()):
            out.append(mid)
            if len(out) == a.k:
                break
    return out


lists = {
    "MF (page factors)": {u: [mid for mid, _ in mc.top_k(P, Q, mu, u, a.k, ids, rated)] for u in users},
    "popularity": {u: pop_top(u) for u in users},
}
if a.min_pop:
    floor_ids = [mid for mid in ids if pop.get(mid, 0) >= a.min_pop]
    lists[f"MF, movies with >= {a.min_pop} ratings"] = {u: [mid for mid, _ in mc.top_k(P, Q, mu, u, a.k, floor_ids, rated)] for u in users}

print(f"[data] users {len(users)}, movies in u.item {len(ids)}; short head = {len(head)} most-rated movies "
      f"({len(head) / len(ids):.1%} of the catalogue) holding {acc / len(train):.1%} of the training ratings")
hist_years = [year[mid] for u in users for mid in rated.get(u, ()) if mid in year]
print(f"[data] films users rated in train + val: median film year {statistics.median(hist_years):g}, "
      f"made before 1980 {sum(y < 1980 for y in hist_years) / len(hist_years):.1%}")
print(f"[data] users evaluated for Recall@{a.k} (at least one relevant test movie in the training pool): {len(relevant)}")

rows = collections.OrderedDict()
for name, L in lists.items():
    slots = [mid for u in users for mid in L[u]]
    per_user = [(len(set(L[u]) & relevant[u]), len(relevant[u])) for u in users if relevant[u]]
    count = collections.Counter(slots)
    rows[name] = {
        f"Recall@{a.k}": f"{sum(h / n for h, n in per_user) / len(per_user):.4f}",
        "hits (relevant movies recommended)": f"{sum(h for h, _ in per_user)}",
        "users with at least one hit": f"{sum(h > 0 for h, _ in per_user)} of {len(per_user)}",
        "distinct movies in all lists": f"{len(count)} ({len(count) / len(ids):.1%} of the catalogue)",
        "slots taken by the 10 most-recommended movies": f"{sum(c for _, c in count.most_common(10)) / len(slots):.1%}",
        "slots on long-tail movies": f"{sum(mid not in head for mid in slots) / len(slots):.1%}",
        "slots on thin-evidence movies (< 5 ratings)": f"{sum(pop.get(mid, 0) < 5 for mid in slots) / len(slots):.1%}",
        "median popularity of a recommended movie": f"{statistics.median(pop.get(mid, 0) for mid in slots):g} ratings",
        "median film year of a recommended movie": f"{statistics.median(year[mid] for mid in slots if mid in year):g}",
        "slots on films made before 1980": f"{sum(year.get(mid, 9999) < 1980 for mid in slots) / len(slots):.1%}",
        "familiar slots (genre among user's top 3)": f"{sum(bool(set(movies[mid]['genres']) & top3[u]) for u in users for mid in L[u]) / len(slots):.1%}",
    }
    rows[name]["_top"] = count.most_common(5)

names = list(rows)
w = max(len(k) for k in rows[names[0]] if not k.startswith("_"))
print(f"[compare] {'':{w}s}  " + "  ".join(f"{n:>30s}" for n in names))
for key in rows[names[0]]:
    if key.startswith("_"):
        continue
    print(f"[compare] {key:{w}s}  " + "  ".join(f"{rows[n][key]:>30s}" for n in names))
for n in names:
    print(f"[top] {n}: most-recommended movies (in how many of the {len(users)} lists): "
          + "; ".join(f"{movies[mid]['title']} {c}" for mid, c in rows[n]["_top"]))
