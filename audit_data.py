#!/usr/bin/env python3
"""
audit_data.py — what the HW5 starter really contains, checked against prompt.md (the week5 readme)
and script.js. Reads files only; changes nothing.

    python3 audit_data.py --readme prompt.md --script script.js
    python3 audit_data.py --official ~/Downloads/ml-100k        # also compare u.data / u.item byte for byte

The split, the training mean and the baseline RMSE are recomputed in Python (mf_core.py), not read
from the page, so they can be compared with what the page prints.
"""
import argparse, collections, datetime as dt, hashlib, pathlib, re, statistics

import mf_core as mc

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", default=".", help="folder with u.data and u.item")
ap.add_argument("--readme", default="prompt.md")
ap.add_argument("--script", default="script.js")
ap.add_argument("--datajs", default="data.js")
ap.add_argument("--official", help="folder of the unzipped GroupLens ml-100k release, to compare the two files")
a = ap.parse_args()

folder = pathlib.Path(a.data)
for name in ("u.data", "u.item"):
    b = folder.joinpath(name).read_bytes()
    print(f"[file] {name}: {len(b):,} bytes, sha256 {hashlib.sha256(b).hexdigest()}")
    if a.official:
        o = pathlib.Path(a.official, name).read_bytes()
        print(f"[file] {name} vs {a.official}/{name}: byte-identical: {o == b}")

movies, ratings = mc.load(folder)
users = {r[0] for r in ratings}
items = {r[1] for r in ratings}
n = len(ratings)
print(f"[data] ratings {n:,}; users {len(users)} (ids {min(users)}..{max(users)}); movies in u.item {len(movies)} "
      f"(ids {min(movies)}..{max(movies)}); movies with at least one rating {len(items)}")
pairs = collections.Counter((r[0], r[1]) for r in ratings)
print(f"[data] repeated (user, movie) pairs: {sum(c > 1 for c in pairs.values())}")
dist = collections.Counter(int(r[2]) for r in ratings)
print("[data] rating distribution: " + ", ".join(f"{k}: {dist[k]:,} ({dist[k] / n:.1%})" for k in sorted(dist))
      + f"; mean {sum(r[2] for r in ratings) / n:.4f}")
ts = [r[3] for r in ratings]
day = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d")
print(f"[data] timestamps {day(min(ts))} .. {day(max(ts))} (UTC)")
per_user = collections.Counter(r[0] for r in ratings)
per_item = collections.Counter(r[1] for r in ratings)
print(f"[data] ratings per user: min {min(per_user.values())}, median {statistics.median(per_user.values()):g}, max {max(per_user.values())}; "
      f"per movie: min {min(per_item.values())}, median {statistics.median(per_item.values()):g}, max {max(per_item.values())}")
print(f"[data] movies with exactly one rating: {sum(c == 1 for c in per_item.values())}; with fewer than 5: {sum(c < 5 for c in per_item.values())}")
no_genre = [m for m, v in movies.items() if not v["genres"]]
print(f"[data] movies with none of the 18 genres (only the 'unknown' flag or nothing): {len(no_genre)} -> "
      + ", ".join(f"{m} {movies[m]['title']!r}" for m in no_genre[:4]))

U, I = len(users), len(movies)
print(f"[matrix] users x movies {U} x {I} = {U * I:,} cells, {n:,} observed -> density {n / (U * I):.4%}, sparsity {1 - n / (U * I):.4%}")
print(f"[matrix] the page's padded matrix {U + 1} x {I + 1} = {(U + 1) * (I + 1):,} cells (row and column 0 unused) "
      f"would give sparsity {1 - n / ((U + 1) * (I + 1)):.4%}; report the {U} x {I} figure")

train, val, test, skipped = mc.split(ratings)
print(f"[split] train {len(train):,} ({len(train) / n:.1%}), validation {len(val):,} ({len(val) / n:.1%}), test {len(test):,} "
      f"({len(test) / n:.1%}); users skipped (< 5 ratings) {len(skipped)}")
tr_users, tr_items = {r[0] for r in train}, {r[1] for r in train}
for name, rs in (("validation", val), ("test", test)):
    miss_u = {r[0] for r in rs} - tr_users
    miss_i = {r[1] for r in rs} - tr_items
    print(f"[split] {name}: users absent from train {len(miss_u)}; movies absent from train {len(miss_i)} "
          f"({sum(r[1] in miss_i for r in rs)} ratings)")
rel_unseen = {r[1] for r in test if r[2] >= 4.0 and r[1] not in tr_items}
users_no_rel = sorted({r[0] for r in test} - {r[0] for r in test if r[2] >= 4.0 and r[1] in tr_items})
print(f"[split] relevant (>= 4.0) test movies absent from train: {len(rel_unseen)}; users without a relevant test movie in the train pool: {len(users_no_rel)}")
mu = mc.train_mean(train)
print(f"[baseline] training mean {mu:.6f}; RMSE train {mc.baseline_rmse(train, train):.4f}, validation {mc.baseline_rmse(train, val):.4f}, "
      f"test {mc.baseline_rmse(train, test):.4f}")

rd = pathlib.Path(a.readme).read_text(encoding="utf-8") if pathlib.Path(a.readme).exists() else ""
if rd:
    print("[readme] claims:")
    for pat in [r"\*\*943 users\*\*", r"\*\*1,682 movies\*\*", r"\*\*100,000 ratings\*\*", r"4 passed / 0 failed / 7 pending",
                r"TensorFlow\.js", r"rating >= 4\.0", r"Harper", r"10\.1145/2827872", r"docs/homework-guidelines/guidelines\.md"]:
        print(f"    {'found' if re.search(pat, rd) else 'MISSING':7s} {pat}")
    if re.search(r"already rated in\s+training", rd):
        print("[readme] NOTE section 6 says a relevant test movie may be one 'the user already rated in training'; u.data has no repeated"
              f" (user, movie) pair ({sum(c > 1 for c in pairs.values())} above), so that case cannot occur. The unseen ids are relevant"
              " test movies that nobody rated in training.")
js = pathlib.Path(a.script).read_text(encoding="utf-8") if pathlib.Path(a.script).exists() else ""
if js:
    stubs = re.findall(r'throw new Error\("TODO\(hw5\): (\w+) is not implemented yet\."\)', js)
    print(f"[script] TODO(hw5) stubs that throw: {len(stubs)} -> {', '.join(stubs) if stubs else 'none (implemented)'}")
    if re.search(r"false, \[model\.userFactors, model\.itemFactors\]\)\.dispose\(\)", js):
        print("[script] NOTE the trainMF sketch calls .dispose() on optimizer.minimize(f, false, varList); with returnCost = false"
              " minimize returns null, so the sketch throws when copied as is.")
    if re.search(r"model\.userFactors\.square\(\)\.sum\(\)\s*(?:\n\s*\*\s*)?\.add\(model\.itemFactors\.square\(\)\.sum\(\)\)\.mul\(reg\)", js):
        print("[script] NOTE the trainMF sketch adds reg * (|P|^2 + |Q|^2) over the WHOLE factor matrices to every batch loss"
              " (every row is shrunk at every step, not only the rows in the batch).")
    if "trainingHistory.push(" in js and not re.search(r"trainingHistory\s*=\s*\[\]", js[js.find("function trainMF"):js.find("function selectBestCheckpoint")]):
        print("[script] NOTE the trainMF sketch appends to the module-level trainingHistory and never clears it.")
    if "matMul(userRow, false, true)" in js and re.search(r"\.squeeze\(\)\.add\(model\.globalMean\)", js):
        print("[script] NOTE the recommendTopK sketch ranks by mu + p_u . q_i without the clamp to [1, 5] that the readme (section 6)"
              " puts on a predicted rating, so a listed score can be above 5.")
    body = js[js.find("function runPipeline"):js.find("function init()")]
    if "Training" in body and "trainMF(" in body and "await" not in body:
        print("[script] NOTE runPipeline sets the status to 'Training...' and trains in the same synchronous call: the browser repaints"
              " only after training ends, so that text never shows and the page does not respond while it trains.")
dj = pathlib.Path(a.datajs).read_text(encoding="utf-8") if pathlib.Path(a.datajs).exists() else ""
if dj and "TODO(hw5)" in dj:
    print("[data.js] NOTE data.js carries a 'TODO(hw5)' comment above splitByTimestamp, but the function is implemented and the"
          " readme says data.js must not be modified; the six stubs are all in script.js.")
