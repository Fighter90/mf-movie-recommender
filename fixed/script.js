/**
 * week5/script.js — Matrix Factorization with TensorFlow.js (HW5).
 *
 * This is a plain (classic) script, NOT an ES module. `index.html` loads the
 * TensorFlow.js runtime from a CDN before this file, exposing the global `tf`:
 *
 *   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
 *   <script src="data.js"></script>
 *   <script src="script.js"></script>
 *
 * `week5/data.js` (loaded before this file) exposes the MovieLens 100k data
 * layer as globals: `movies`, `ratings`, `numUsers`, `numMovies`,
 * `ratingMatrix`, `SPLIT_EXCLUSIONS`, `loadData()` and `splitByTimestamp()`.
 * This file reads those globals, renders a dataset / split summary, and wires
 * the page controls. The factorization itself is built on `tf.Variable`s and
 * tensor ops so it can run on the WebGL (GPU) backend.
 *
 * BACKEND: `initTf()` prefers WebGL, falls back to CPU, and writes the chosen
 * backend into `#tf-status`. If the CDN did not load (`typeof tf ===
 * 'undefined'`) the page shows a clear message and stops instead of throwing.
 *
 * WHAT YOU MUST IMPLEMENT (`TODO(hw5)` — each stub throws until you write it):
 *   1. `predictRating`        — scalar gather + dot + globalMean, clamp [1, 5].
 *   2. `initializeFactors`    — two `tf.Variable`s + the global mean.
 *   3. `trainMF`              — batched tensor training loop with L2 / Adam.
 *   4. `selectBestCheckpoint` — earliest entry with the smallest val RMSE.
 *   5. `recommendTopK`        — `matMul` + `tf.topk`, exclude train/val items.
 *   6. `recallAtK`            — Recall@K against relevant (>= 4.0) test items.
 *
 * PROVIDED for you (scaffolding): `initTf`, the shared metric helpers
 * (`computeRMSEFor`, `computeBaselineRMSE`), formatting helpers, every DOM
 * renderer, the hyper-parameter reader, the training pipeline wrapper, and the
 * automated test harness. Leave the provided code as-is and implement only the
 * six stubs.
 *
 * The harness reports unimplemented stubs as PENDING rather than FAIL, so the
 * page stays usable before and after the assignment is done.
 *
 * @module week5/script
 */

// ---------------------------------------------------------------------------
// Typedefs
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} Movie
 * @property {number} id
 * @property {string} title
 * @property {string[]} genres
 */

/**
 * @typedef {Object} Rating
 * @property {number} userId
 * @property {number} movieId
 * @property {number} rating
 * @property {number} timestamp
 */

/**
 * @typedef {Object} Model
 * @property {tf.Variable} userFactors Dense factor matrix `[numUsers + 1, F]`,
 *   row index = raw `userId`.
 * @property {tf.Variable} itemFactors Dense factor matrix `[maxMovieId + 1, F]`,
 *   row index = raw `movieId`.
 * @property {number} globalMean       Mean rating of the training set.
 */

/**
 * @typedef {Object} HistoryRow
 * @property {number} epoch
 * @property {number} trainRMSE
 * @property {number} valRMSE
 */

/**
 * @typedef {Object} TrainOptions
 * @property {number} epochs    Number of full passes over the training set.
 * @property {number} lr        Learning rate.
 * @property {number} reg       L2 regularization strength.
 * @property {number} batchSize Mini-batch size.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/** Number of latent factors used by the default pipeline. */
const NUM_FACTORS = 16;

/** Training split (80% of each user's ratings, oldest first). */
let trainSet = [];

/** Validation split (10% of each user's ratings). */
let valSet = [];

/** Test split (10% of each user's ratings, newest). */
let testSet = [];

/**
 * The fitted factorization, or `null` before `initializeFactors` runs.
 *
 * @type {Model|null}
 */
let model = null;

/**
 * Metric history produced by `trainMF`.
 *
 * @type {HistoryRow[]}
 */
let trainingHistory = [];

/**
 * The history entry returned by `selectBestCheckpoint`, or `null`.
 *
 * @type {HistoryRow|null}
 */
let bestCheckpoint = null;

/** Name of the TensorFlow.js backend selected by `initTf` (`"webgl"` / `"cpu"`). */
let tfBackend = null;

// ---------------------------------------------------------------------------
// TensorFlow.js bootstrap (provided)
// ---------------------------------------------------------------------------

/**
 * Select the fastest available TensorFlow.js backend and report it.
 *
 * Tries WebGL (GPU) first and falls back to the pure-JS CPU backend. Writes
 * `backend: <name>` into `#tf-status` when that element exists. If the CDN
 * failed, `tf` is undefined: this renders a readable message and returns
 * `null` instead of letting the page throw an uncaught `ReferenceError`.
 *
 * @returns {Promise<string|null>} the selected backend, or `null` when TF.js is missing.
 */
async function initTf() {
  const statusEl = document.getElementById("tf-status");
  if (typeof tf === "undefined") {
    const message =
      "TensorFlow.js failed to load. Check your network connection (the CDN script) and reload the page.";
    if (statusEl) {
      statusEl.textContent = message;
      statusEl.className = "error";
    } else if (document.body) {
      const banner = document.createElement("p");
      banner.className = "error";
      banner.textContent = message;
      document.body.insertBefore(banner, document.body.firstChild);
    }
    return null;
  }

  try {
    await tf.setBackend("webgl");
  } catch (error) {
    await tf.setBackend("cpu");
  }
  await tf.ready();

  tfBackend = tf.getBackend();
  if (statusEl) statusEl.textContent = `TensorFlow.js backend: ${tfBackend}`;
  return tfBackend;
}

/**
 * True when `candidate` is a TF.js factor model (dense `tf.Variable` rows).
 *
 * Used to route `computeRMSEFor` through the batched tensor path and to keep
 * the fixture-based tests independent of how the model was produced.
 *
 * @param {*} candidate
 * @returns {boolean}
 */
function isTfModel(candidate) {
  return (
    !!candidate &&
    !!candidate.userFactors &&
    typeof candidate.userFactors.dataSync === "function" &&
    !!candidate.itemFactors &&
    typeof candidate.itemFactors.dataSync === "function"
  );
}

// ---------------------------------------------------------------------------
// Formatting helpers (provided)
// ---------------------------------------------------------------------------

/**
 * Format a number to a fixed number of decimals. Non-finite and missing values
 * render as an em dash so the tables never show `NaN` or `undefined`.
 *
 * @param {number|null|undefined} x
 * @param {number} [d=4] decimal places
 * @returns {string}
 */
function fmt(x, d = 4) {
  if (x === null || x === undefined) return "\u2014";
  const n = Number(x);
  if (!Number.isFinite(n)) return "\u2014";
  return n.toFixed(d);
}

/**
 * Look up a movie title by id, with a readable fallback for unknown ids.
 *
 * @param {number|string} id
 * @returns {string}
 */
function movieTitle(id) {
  const numericId = Number(id);
  const movie = movies.find((m) => m.id === numericId);
  return movie ? movie.title : `Movie #${id}`;
}

/**
 * Escape text before inserting it into HTML.
 *
 * @param {string} text
 * @returns {string}
 */
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Clamp a number into the inclusive range `[lo, hi]`.
 *
 * @param {number} value
 * @param {number} lo
 * @param {number} hi
 * @returns {number}
 */
function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

// ---------------------------------------------------------------------------
// Shared metric helpers (provided)
// ---------------------------------------------------------------------------

/**
 * Root-mean-square error of the current model over a split.
 *
 * Provided so that `trainMF` and the test harness share one implementation.
 *
 * When `model` holds TF.js variables this builds the index tensors ONCE, uses
 * `tf.gather` + element-wise multiply to score every rating in a single batch,
 * and clamps to `[1, 5]` — the same contract as `predictRating`, but fast
 * enough to call every epoch. Out-of-range ids fall back to `globalMean`, and
 * the whole computation runs inside `tf.tidy` so the intermediate tensors are
 * released automatically.
 *
 * When no TF.js model is available it falls back to the scalar
 * `predictRating` loop. That path returns `NaN` instead of throwing while
 * `predictRating` is still a `TODO(hw5)` stub, so the helper never crashes the
 * page or the harness.
 *
 * @param {Rating[]} split
 * @returns {number} `0` for an empty split; `NaN` if the scalar path throws.
 */
function computeRMSEFor(split) {
  if (!split || split.length === 0) return 0;

  if (isTfModel(model)) {
    return tf.tidy(() => {
      const n = split.length;
      const uRows = model.userFactors.shape[0];
      const iRows = model.itemFactors.shape[0];
      const userIds = new Int32Array(n);
      const movieIds = new Int32Array(n);
      const actual = new Float32Array(n);
      const valid = new Float32Array(n);

      for (let k = 0; k < n; k++) {
        const r = split[k];
        userIds[k] = r.userId;
        movieIds[k] = r.movieId;
        actual[k] = r.rating;
        const uOk = r.userId >= 0 && r.userId < uRows;
        const iOk = r.movieId >= 0 && r.movieId < iRows;
        valid[k] = uOk && iOk ? 1 : 0;
      }

      const uIdx = tf.clipByValue(tf.tensor1d(userIds, "int32"), 0, Math.max(0, uRows - 1));
      const iIdx = tf.clipByValue(tf.tensor1d(movieIds, "int32"), 0, Math.max(0, iRows - 1));
      const u = tf.gather(model.userFactors, uIdx);
      const i = tf.gather(model.itemFactors, iIdx);
      const dot = tf.sum(tf.mul(u, i), 1);
      const mask = tf.tensor1d(valid, "float32");
      const pred = tf.clipByValue(dot.mul(mask).add(model.globalMean), 1, 5);
      const diff = pred.sub(tf.tensor1d(actual, "float32"));
      return Math.sqrt(tf.mean(diff.square()).dataSync()[0]);
    });
  }

  let sum = 0;
  try {
    for (const r of split) {
      const diff = predictRating(r.userId, r.movieId) - r.rating;
      sum += diff * diff;
    }
  } catch (error) {
    return Number.NaN;
  }
  return Math.sqrt(sum / split.length);
}

/**
 * Baseline RMSE: predict the training-set mean rating for every pair and
 * measure the error on `split`. This is the number your factorization must beat.
 *
 * @param {Rating[]} split
 * @returns {number} `0` for an empty split (never divides by zero).
 */
function computeBaselineRMSE(split) {
  if (!split || split.length === 0) return 0;
  const mean = trainSet.length
    ? trainSet.reduce((acc, r) => acc + r.rating, 0) / trainSet.length
    : 0;
  let sum = 0;
  for (const r of split) {
    const diff = mean - r.rating;
    sum += diff * diff;
  }
  return Math.sqrt(sum / split.length);
}

// ---------------------------------------------------------------------------
// Student stubs (TODO(hw5))
//
// Implemented. Helpers shared by the stubs live below; the JSDoc notes on
// each stub describe the design where it differs from the original sketch.
// ---------------------------------------------------------------------------

/** Seed for the two `tf.randomNormal` initializations (reproducible runs). */
const MF_INIT_SEED = 42;

/** Seed for the per-epoch Fisher-Yates shuffle of the training order. */
const MF_SHUFFLE_SEED = 7;

/**
 * Mulberry32 seeded PRNG returning floats in [0, 1) — the same generator as
 * `mf_core.mulberry32` in the numpy replay, so the page and Python shuffle
 * the training order identically.
 *
 * @param {number} seed
 * @returns {() => number}
 */
function makeRandom(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Predict the rating a user would give a movie (scalar).
 *
 * TODO(hw5): implement the prediction with TensorFlow.js. The model stores the
 * factors as two dense `tf.Variable`s:
 *   - `model.userFactors` shape `[numUsers + 1, F]`, row index = `userId`
 *   - `model.itemFactors` shape `[maxMovieId + 1, F]`, row index = `movieId`
 *
 * Suggested TF.js path (wrap it in `tf.tidy` and pull out a JS number):
 * ```js
 * const f = model.userFactors.shape[1];
 * return tf.tidy(() => {
 *   const u = model.userFactors.slice([userId, 0], [1, f]); // [1, F]
 *   const i = model.itemFactors.slice([movieId, 0], [1, f]); // [1, F]
 *   const dot = u.mul(i).sum().dataSync()[0];                // scalar
 *   return clamp(model.globalMean + dot, 1, 5);
 * });
 * ```
 * `dataSync()` copies the value to the CPU as a plain number, which survives
 * `tf.tidy` (the tensors do not). Keep this function scalar: `computeRMSEFor`
 * uses its own batched tensor path, and the harness calls it once per pair.
 *
 * Contract:
 *  - Return `model.globalMean` when the user or movie row is out of range
 *    (an id that has no allocated factor row). Never throw for an unknown id.
 *  - Return `0` when the model is not initialized (`model === null`) so callers
 *    can guard cheaply.
 *  - The result is clamped to `[1, 5]`.
 *
 * @param {number} userId
 * @param {number} movieId
 * @returns {number} predicted rating in `[1, 5]`
 */
function predictRating(userId, movieId) {
  // Contract: 0 when no model; globalMean when an id is not an integer row of
  // the factor matrices; otherwise globalMean + dot of the two slices inside
  // tf.tidy, read with dataSync, clamped to [1, 5].
  if (!model) return 0;
  const f = model.userFactors.shape[1];
  if (
    !Number.isInteger(userId) || !Number.isInteger(movieId) ||
    userId < 0 || userId >= model.userFactors.shape[0] ||
    movieId < 0 || movieId >= model.itemFactors.shape[0]
  ) {
    return model.globalMean;
  }
  return tf.tidy(() => {
    const u = model.userFactors.slice([userId, 0], [1, f]); // [1, F]
    const i = model.itemFactors.slice([movieId, 0], [1, f]); // [1, F]
    const dot = u.mul(i).sum().dataSync()[0]; // scalar
    return clamp(model.globalMean + dot, 1, 5);
  });
}

/**
 * Allocate the latent factor variables and the global mean.
 *
 * Implemented design (differs from the original sketch):
 *  - `globalMean` is the mean rating over `train`, computed in plain
 *    JavaScript (`0` for an empty training set) — not with a tensor.
 *  - Both dense variables are created inside `tf.tidy`, so the intermediate
 *    `tf.randomNormal` tensor is released when the scope exits while the
 *    variable keeps the data (TF.js refcounts the shared storage).
 *  - The draws are seeded and reproducible: `tf.randomNormal(shape, 0, 0.05,
 *    "float32", MF_INIT_SEED)` for users (row index = userId) and seed
 *    `MF_INIT_SEED + 1` for items (row index = movieId), shapes
 *    `[numUsers + 1, numFactors]` and `[maxMovieId + 1, numFactors]`.
 *
 * Contract:
 *  - `numFactors` defaults to the module-level `NUM_FACTORS`.
 *  - If a previous `model` exists, dispose its variables (`tf.dispose`) before
 *    replacing it so repeated "Train" clicks do not leak GPU memory.
 *  - Store the result on the module-level `model` object and return it.
 *
 * @param {Rating[]} train
 * @param {number} [numFactors=NUM_FACTORS]
 * @returns {Model}
 */
function initializeFactors(train, numFactors = NUM_FACTORS) {
  // Contract: dispose a previous model first; globalMean = the mean rating
  // over `train` in plain JavaScript (0 when empty); dense tf.Variables
  // indexed by raw ids, created inside tf.tidy from seeded tf.randomNormal
  // draws, stored on the module-level `model` and returned.
  if (model) tf.dispose([model.userFactors, model.itemFactors]);
  const globalMean = train.length
    ? train.reduce((acc, r) => acc + r.rating, 0) / train.length
    : 0;
  const maxMovieId = movies.reduce((m, x) => Math.max(m, x.id), 0);
  model = {
    userFactors: tf.tidy(() => tf.variable(
      tf.randomNormal([numUsers + 1, numFactors], 0, 0.05, "float32", MF_INIT_SEED),
    )),
    itemFactors: tf.tidy(() => tf.variable(
      tf.randomNormal([maxMovieId + 1, numFactors], 0, 0.05, "float32", MF_INIT_SEED + 1),
    )),
    globalMean,
  };
  return model;
}

/**
 * Train the factorization with batched gradient descent.
 *
 * Implemented design (differs from the original sketch; its pitfalls fixed):
 *  - `trainingHistory` is reset at the start of every call; an empty history
 *    is returned when there is no model or no training data.
 *  - The training order is a seeded Fisher-Yates shuffle (mulberry32 via
 *    `makeRandom(MF_SHUFFLE_SEED)`, the same generator as the numpy replay),
 *    re-drawn every epoch; mini-batches of `batchSize` in shuffled order
 *    (full-batch when `batchSize >= train.length`).
 *  - Per-batch loss, inside `tf.tidy`:
 *      tf.losses.meanSquaredError(y, pred)
 *         .add(u.square().sum(1).add(i.square().sum(1)).mean().mul(reg))
 *    — the per-rating L2 penalty of Koren, Bell and Volinsky (2009), Equation 2
 *    as minimized by their per-rating SGD rules: each batch rating drags the
 *    factor norms of its own user and item rows exactly once. The sketch's
 *    whole-matrix `reg * (|P|^2 + |Q|^2)` penalty is gone (it shrunk every row
 *    of both matrices at every step and collapsed the model to the mean).
 *  - `optimizer.minimize(f, false, [model.userFactors, model.itemFactors])`
 *    with nothing chained after it: with `returnCost = false` it returns
 *    `null`, and the loss tensor never materializes.
 *  - After each epoch one history row is appended; whenever `valRMSE`
 *    improves, clones of both factor matrices replace the previous best
 *    clones. After the last epoch the best clones are assigned back into the
 *    variables, so the model in memory IS the selected checkpoint; clones and
 *    optimizer are disposed.
 *
 * Contract:
 *  - `opts = { epochs, lr, reg, batchSize }`.
 *  - Update `model` in place. Do NOT reallocate it.
 *  - Append one `HistoryRow` per epoch to the module-level `trainingHistory`
 *    and return it: `{ epoch, trainRMSE: computeRMSEFor(train),
 *    valRMSE: computeRMSEFor(val) }`. `epoch` is 1-based.
 *  - Guard `train.length === 0` (and a missing model) by returning an empty
 *    history.
 *
 * @param {Rating[]} train
 * @param {Rating[]} val
 * @param {TrainOptions} opts
 * @returns {HistoryRow[]}
 */
function trainMF(train, val, opts) {
  // Contract: reset the history first; empty history without a model or data.
  trainingHistory = [];
  if (!model || train.length === 0) return trainingHistory;

  const { epochs, lr, reg } = opts;
  const n = train.length;
  const size = Math.min(Math.max(1, opts.batchSize), n);
  const order = new Array(n);
  for (let k = 0; k < n; k++) order[k] = k;
  const random = makeRandom(MF_SHUFFLE_SEED);
  const optimizer = tf.train.adam(lr);

  const users = new Int32Array(n);
  const items = new Int32Array(n);
  const targets = new Float32Array(n);

  let bestValRMSE = Number.POSITIVE_INFINITY;
  let bestUserClone = null;
  let bestItemClone = null;

  for (let epoch = 1; epoch <= epochs; epoch++) {
    // Seeded Fisher-Yates over the rating order, in place.
    for (let k = n - 1; k >= 1; k--) {
      const j = Math.floor(random() * (k + 1));
      const t = order[k];
      order[k] = order[j];
      order[j] = t;
    }
    for (let k = 0; k < n; k++) {
      const r = train[order[k]];
      users[k] = r.userId;
      items[k] = r.movieId;
      targets[k] = r.rating;
    }
    for (let start = 0; start < n; start += size) {
      const end = Math.min(start + size, n);
      tf.tidy(() => {
        const userIdx = tf.tensor1d(users.slice(start, end), "int32");
        const itemIdx = tf.tensor1d(items.slice(start, end), "int32");
        const y = tf.tensor1d(targets.slice(start, end), "float32");
        // returnCost = false: minimize returns null — nothing to chain.
        optimizer.minimize(() => tf.tidy(() => {
          const u = tf.gather(model.userFactors, userIdx);
          const i = tf.gather(model.itemFactors, itemIdx);
          const pred = tf.sum(u.mul(i), 1).add(model.globalMean);
          const loss = tf.losses.meanSquaredError(y, pred)
            .add(u.square().sum(1).add(i.square().sum(1)).mean().mul(reg));
          return loss;
        }), false, [model.userFactors, model.itemFactors]);
      });
    }
    trainingHistory.push({
      epoch,
      trainRMSE: computeRMSEFor(train),
      valRMSE: computeRMSEFor(val),
    });
    const row = trainingHistory[trainingHistory.length - 1];
    if (row.valRMSE < bestValRMSE) {
      bestValRMSE = row.valRMSE;
      const newUserClone = model.userFactors.clone();
      const newItemClone = model.itemFactors.clone();
      if (bestUserClone) bestUserClone.dispose();
      if (bestItemClone) bestItemClone.dispose();
      bestUserClone = newUserClone;
      bestItemClone = newItemClone;
    }
  }

  // Restore the selected checkpoint so the model in memory matches the
  // reported best epoch (assign shares the clone's storage; disposing the
  // clone afterwards releases the remaining reference).
  if (bestUserClone && bestItemClone) {
    model.userFactors.assign(bestUserClone);
    model.itemFactors.assign(bestItemClone);
    bestUserClone.dispose();
    bestItemClone.dispose();
  }
  optimizer.dispose();
  return trainingHistory;
}

/**
 * Pick the best epoch from a training history.
 *
 * TODO(hw5): return the entry with the smallest `valRMSE`. On a tie return the
 * entry with the EARLIEST `epoch` (do not simply take the last minimum). This
 * one is plain JavaScript — no TF.js needed.
 *
 * Contract:
 *  - Return `null` for an empty or missing history.
 *  - Rows whose `valRMSE` is not finite are skipped.
 *  - Do not mutate the input array.
 *
 * @param {HistoryRow[]} history
 * @returns {HistoryRow|null}
 */
function selectBestCheckpoint(history) {
  // Contract: null for an empty or missing history; rows whose valRMSE is not
  // finite are skipped; the strict < comparison keeps the earliest epoch on
  // ties; the input array is not mutated.
  if (!history || history.length === 0) return null;
  let best = null;
  for (const row of history) {
    if (!Number.isFinite(row.valRMSE)) continue;
    if (best === null || row.valRMSE < best.valRMSE) best = row;
  }
  return best;
}

/**
 * Per-user sets of movie ids rated in `trainSet` or `valSet`, cached across
 * `recommendTopK` / `recallAtK` calls and rebuilt whenever either split array
 * is replaced (module-level identity check).
 *
 * @returns {Map<number, Set<number>>}
 */
let ratedCache = { train: null, val: null, byUser: new Map() };
function ratedMoviesInTrainAndVal() {
  if (ratedCache.train !== trainSet || ratedCache.val !== valSet) {
    const byUser = new Map();
    for (const r of trainSet.concat(valSet)) {
      let set = byUser.get(r.userId);
      if (!set) {
        set = new Set();
        byUser.set(r.userId, set);
      }
      set.add(r.movieId);
    }
    ratedCache = { train: trainSet, val: valSet, byUser };
  }
  return ratedCache.byUser;
}

/**
 * Recommend the top-K movies for a user.
 *
 * TODO(hw5): score every candidate movie and return the K highest-scoring
 * movies the user has NOT already rated in `trainSet` or `valSet`. Test-set
 * ratings are allowed as candidates. **Prefer the tensor path**:
 * ```js
 * const userRow = model.userFactors.slice([userId, 0], [1, F]); // [1, F]
 * const scores = model.itemFactors.matMul(userRow, false, true)  // [rows, 1]
 *   .squeeze().add(model.globalMean);                            // [rows]
 * // mask excluded ids to a large negative value, then:
 * const { values, indices } = tf.topk(maskedScores, Math.min(K, rows));
 * ```
 * Because `itemFactors` is indexed by raw `movieId`, the `tf.topk` indices ARE
 * the movie ids — no reverse mapping needed. Excluded rows must be masked (for
 * example multiply by a 0/1 mask and add `(1 - mask) * -1e9`); re-apply the
 * same mask when mapping results back so masked ids are dropped.
 *
 * Contract:
 *  - Return `[{ movieId, title, score }, ...]` sorted by descending `score`,
 *    truncated to `K` (fewer if not enough candidates). `score` is the
 *    UNCLAMPED ranking score `mu + p_u . q_i` — clamping belongs to a
 *    predicted rating, not to a ranking score.
 *  - Exclude any movie id that is absent from the global `movies` list, and
 *    any id the user rated in `trainSet` or `valSet` (per-user exclusion sets
 *    are cached and rebuilt when a split array is replaced).
 *  - Return `[]` when the model is not initialized, when `K <= 0`, when
 *    `userId` is not an integer row of `userFactors`, or when the user has no
 *    eligible candidates — never throw.
 *  - `title` comes from `movieTitle(movieId)`.
 *
 * @param {number} userId
 * @param {number} [K=10]
 * @returns {Array<{movieId: number, title: string, score: number}>}
 */
function recommendTopK(userId, K = 10) {
  // Contract: [] when the model is missing, K <= 0, or userId is not an
  // integer row of userFactors; one matMul scores every item row (NO clamp —
  // the score is a ranking score); ids absent from `movies` or rated by the
  // user in trainSet/valSet are masked to -1e9; tf.topk returns
  // min(K, eligible) rows and masked ids are dropped when mapping back.
  if (!model || K <= 0) return [];
  if (!Number.isInteger(userId) || userId < 0 || userId >= model.userFactors.shape[0]) {
    return [];
  }
  const rated = ratedMoviesInTrainAndVal().get(userId) || new Set();
  const known = new Set(movies.map((m) => m.id));
  const rows = model.itemFactors.shape[0];
  const eligible = new Int32Array(rows);
  let eligibleCount = 0;
  for (let id = 0; id < rows; id++) {
    eligible[id] = known.has(id) && !rated.has(id) ? 1 : 0;
    eligibleCount += eligible[id];
  }
  if (eligibleCount === 0) return [];

  return tf.tidy(() => {
    const F = model.userFactors.shape[1];
    const userRow = model.userFactors.slice([userId, 0], [1, F]); // [1, F]
    const scores = model.itemFactors.matMul(userRow, false, true) // [rows, 1]
      .squeeze()
      .add(model.globalMean); // [rows] ranking scores, unclamped
    const mask = tf.tensor1d(eligible, "float32");
    const maskedScores = scores.mul(mask).add(tf.scalar(1).sub(mask).mul(-1e9));
    const { indices, values } = tf.topk(maskedScores, Math.min(K, eligibleCount));
    const ids = indices.dataSync();
    const topScores = values.dataSync();
    const results = [];
    for (let j = 0; j < ids.length; j++) {
      const movieId = ids[j];
      if (eligible[movieId] === 1) {
        results.push({ movieId, title: movieTitle(movieId), score: topScores[j] });
      }
    }
    return results;
  });
}

/**
 * Compute Recall@K for the current model.
 *
 * TODO(hw5): retrieval evaluation built on `recommendTopK`.
 *   - relevant = test ratings with `rating >= 4.0`.
 *   - Candidate items = the set of movie ids that appear anywhere in `trainSet`
 *     (a global candidate pool, NOT just the user's own training items).
 *   - For every user with at least one relevant test movie among the
 *     candidates, take `recommendTopK(userId, K)` and count how many of that
 *     user's relevant candidate movies appear in it.
 *
 * Contract:
 *  - Return
 *    `{ overallRecall, perUser, excluded }` where:
 *      `overallRecall` = mean over reported users of `hit / relevant`
 *        (`0` when there are no reported users — never divide by zero);
 *      `perUser` = `[{ userId, relevant, hit }, ...]` (reported users only);
 *      `excluded` = `{ unseenIds, usersWithoutRelevant }`.
 *  - `unseenIds` = distinct relevant test movie ids that are NOT present in
 *    `trainSet` (they can never be recommended from a training-only pool).
 *  - `usersWithoutRelevant` = distinct users that appear in `testSet` but have
 *    no relevant test movie among the candidates.
 *  - Report exclusions explicitly so the recall number is interpretable.
 *
 * @param {number} [K=10]
 * @returns {{overallRecall: number, perUser: Array<{userId: number, relevant: number, hit: number}>, excluded: {unseenIds: number[], usersWithoutRelevant: number[]}}}
 */
function recallAtK(K = 10) {
  // Contract: relevant = test ratings >= 4.0; candidates = every movie id in
  // trainSet (global pool). Report per-user hits for users with a relevant
  // candidate, plus the two exclusions: relevant test movies absent from the
  // training pool and test users without a relevant candidate.
  const candidates = new Set(trainSet.map((r) => r.movieId));

  const testUsers = new Set();
  const relevantByUser = new Map();
  const unseenIds = new Set();
  for (const r of testSet) {
    testUsers.add(r.userId);
    if (r.rating < 4.0) continue;
    if (!candidates.has(r.movieId)) {
      unseenIds.add(r.movieId);
      continue;
    }
    if (!relevantByUser.has(r.userId)) relevantByUser.set(r.userId, new Set());
    relevantByUser.get(r.userId).add(r.movieId);
  }

  const perUser = [];
  const usersWithoutRelevant = [];
  for (const userId of Array.from(testUsers).sort((a, b) => a - b)) {
    const relevant = relevantByUser.get(userId);
    if (!relevant) {
      usersWithoutRelevant.push(userId);
      continue;
    }
    const listed = new Set(recommendTopK(userId, K).map((rec) => rec.movieId));
    let hit = 0;
    for (const movieId of relevant) {
      if (listed.has(movieId)) hit++;
    }
    perUser.push({ userId, relevant: relevant.size, hit });
  }

  const overallRecall = perUser.length
    ? perUser.reduce((acc, u) => acc + u.hit / u.relevant, 0) / perUser.length
    : 0;

  return {
    overallRecall,
    perUser,
    excluded: {
      unseenIds: Array.from(unseenIds).sort((a, b) => a - b),
      usersWithoutRelevant,
    },
  };
}

// ---------------------------------------------------------------------------
// UI rendering (provided)
// ---------------------------------------------------------------------------

/**
 * Render the dataset shape summary.
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderDatasetSummary(container) {
  const target = container || document.getElementById("dataset-summary-body");
  if (!target) return;
  const maxMovieId = movies.reduce((max, m) => Math.max(max, m.id), 0);
  target.innerHTML = `
    <dl class="summary-list">
      <dt>Movies</dt><dd>${movies.length.toLocaleString("en-US")}</dd>
      <dt>Ratings</dt><dd>${ratings.length.toLocaleString("en-US")}</dd>
      <dt>Users</dt><dd>${numUsers.toLocaleString("en-US")}</dd>
      <dt>Rating matrix</dt><dd>${(numUsers + 1).toLocaleString("en-US")} &times; ${(maxMovieId + 1).toLocaleString("en-US")}</dd>
    </dl>`;
}

/**
 * Render the chronological 80/10/10 split summary.
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderSplitSummary(container) {
  const target = container || document.getElementById("split-summary-body");
  if (!target) return;
  const total = trainSet.length + valSet.length + testSet.length;
  const pct = (n) => (total ? ((n / total) * 100).toFixed(1) : "0.0");
  const exclusions = typeof SPLIT_EXCLUSIONS === "undefined" ? [] : SPLIT_EXCLUSIONS;
  target.innerHTML = `
    <table class="data-table">
      <thead><tr><th scope="col">Split</th><th scope="col">Ratings</th><th scope="col">Share</th></tr></thead>
      <tbody>
        <tr><td>Train</td><td class="num">${trainSet.length}</td><td class="num">${pct(trainSet.length)}%</td></tr>
        <tr><td>Validation</td><td class="num">${valSet.length}</td><td class="num">${pct(valSet.length)}%</td></tr>
        <tr><td>Test</td><td class="num">${testSet.length}</td><td class="num">${pct(testSet.length)}%</td></tr>
      </tbody>
    </table>
    <p class="hint">${exclusions.length} user(s) skipped while splitting; baseline test RMSE ${fmt(computeBaselineRMSE(testSet))}.</p>`;
}

/**
 * Render the training history as a plain table (no SVG, no chart library).
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderTrainingCurve(container) {
  const target = container || document.getElementById("training-curve");
  if (!target) return;
  if (!trainingHistory || trainingHistory.length === 0) {
    target.innerHTML =
      '<p class="empty-state">No training history yet. Implement the <code>TODO(hw5)</code> functions and press &ldquo;Train&rdquo;.</p>';
    return;
  }
  let bestIndex = 0;
  for (let i = 1; i < trainingHistory.length; i++) {
    if (trainingHistory[i].valRMSE < trainingHistory[bestIndex].valRMSE) bestIndex = i;
  }
  const rows = trainingHistory
    .map(
      (row, i) => `
      <tr${i === bestIndex ? ' class="best-epoch"' : ""}>
        <td class="num">${row.epoch}</td>
        <td class="num">${fmt(row.trainRMSE)}</td>
        <td class="num">${fmt(row.valRMSE)}</td>
      </tr>`,
    )
    .join("");
  target.innerHTML = `
    <table class="data-table">
      <thead><tr><th scope="col">Epoch</th><th scope="col">Train RMSE</th><th scope="col">Val RMSE</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

/**
 * Render the top-K recommendations for a user.
 *
 * @param {number} userId
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderTopK(userId, container) {
  const target = container || document.getElementById("recommendations");
  if (!target) return;
  if (!model) {
    target.innerHTML = '<p class="empty-state">Train the model before requesting recommendations.</p>';
    return;
  }
  try {
    const recs = recommendTopK(Number(userId), 10);
    if (!recs || recs.length === 0) {
      target.innerHTML = '<p class="empty-state">No eligible recommendations for this user.</p>';
      return;
    }
    target.innerHTML =
      '<ol class="recommendation-list">' +
      recs
        .map(
          (r) =>
            `<li><span class="rec-title">${escapeHtml(r.title)}</span> <span class="rec-score">${fmt(r.score)}</span></li>`,
        )
        .join("") +
      "</ol>";
  } catch (error) {
    target.innerHTML = `<p class="empty-state">Recommendations unavailable — implement the <code>TODO(hw5)</code> functions. <code>${escapeHtml(error.message)}</code></p>`;
  }
}

/**
 * Render the Recall@K report, including explicit exclusions.
 *
 * @param {number} [K=10]
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderRecall(K = 10, container) {
  const target = container || document.getElementById("recall-results");
  if (!target) return;
  try {
    const result = recallAtK(K);
    const rows = result.perUser
      .map(
        (u) => `
        <tr>
          <td>${u.userId}</td>
          <td class="num">${u.relevant}</td>
          <td class="num">${u.hit}</td>
          <td class="num">${u.relevant ? fmt(u.hit / u.relevant) : "\u2014"}</td>
        </tr>`,
      )
      .join("");
    target.innerHTML = `
      <p><strong>Recall@${K}:</strong> <span class="num">${fmt(result.overallRecall)}</span></p>
      <table class="data-table">
        <thead><tr><th scope="col">User</th><th scope="col">Relevant</th><th scope="col">Hits</th><th scope="col">Recall</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="4">No evaluable users.</td></tr>'}</tbody>
      </table>
      <p class="hint">Excluded: ${result.excluded.unseenIds.length} unseen relevant test item(s); ${result.excluded.usersWithoutRelevant.length} user(s) without a relevant candidate.</p>`;
  } catch (error) {
    target.innerHTML = `<p class="empty-state">Recall unavailable — implement the <code>TODO(hw5)</code> functions. <code>${escapeHtml(error.message)}</code></p>`;
  }
}

/**
 * Read the training hyper-parameters from the DOM, falling back to defaults when
 * the inputs are absent or invalid.
 *
 * @returns {TrainOptions}
 */
function readHyperparameters() {
  const readNumber = (id, fallback) => {
    const el = document.getElementById(id);
    if (!el) return fallback;
    const value = Number(el.value);
    return Number.isFinite(value) ? value : fallback;
  };
  return {
    epochs: Math.max(1, Math.round(readNumber("epochs", 20))),
    lr: readNumber("lr", 0.01),
    reg: readNumber("reg", 0.05),
    batchSize: Math.max(1, Math.round(readNumber("batch-size", 64))),
  };
}

/**
 * Populate the user `<select>` from the training split.
 *
 * @returns {void}
 */
function populateUserSelect() {
  const select = document.getElementById("user-select");
  if (!select) return;
  const ids = Array.from(new Set(trainSet.map((r) => r.userId))).sort((a, b) => a - b);
  select.innerHTML = ids.length
    ? ids.map((id) => `<option value="${id}">User ${id}</option>`).join("")
    : '<option value="">No users</option>';
}

/**
 * Run the full training pipeline: initialize, train, select the best checkpoint,
 * and refresh the training curve.
 *
 * @returns {void}
 */
function runPipeline() {
  const status = document.getElementById("status");
  if (!movies.length || trainSet.length === 0) {
    if (status) status.textContent = "Data not loaded yet — reload the page.";
    return;
  }
  const opts = readHyperparameters();
  try {
    if (status) status.textContent = "Training\u2026";
    initializeFactors(trainSet, NUM_FACTORS);
    trainingHistory = trainMF(trainSet, valSet, opts);
    bestCheckpoint = selectBestCheckpoint(trainingHistory);
    renderTrainingCurve();
    if (status) {
      status.textContent = bestCheckpoint
        ? `Trained ${opts.epochs} epoch(s) on ${tfBackend || "tf"} — best val RMSE ${fmt(bestCheckpoint.valRMSE)} at epoch ${bestCheckpoint.epoch}.`
        : `Trained ${opts.epochs} epoch(s) — no history recorded.`;
    }
  } catch (error) {
    const message = String(error && error.message ? error.message : error);
    if (status) status.textContent = message;
    const target = document.getElementById("training-curve");
    if (target) {
      target.innerHTML = `<p class="empty-state">Training failed — implement the <code>TODO(hw5)</code> functions, then press &ldquo;Train&rdquo;. <code>${escapeHtml(message)}</code></p>`;
    }
  }
}

/**
 * Wire the controls once the data has been loaded. Rendering is defensive so an
 * unimplemented stub can never take the whole page down at load time.
 *
 * @returns {void}
 */
function init() {
  populateUserSelect();
  renderDatasetSummary();
  renderSplitSummary();
  renderTrainingCurve();

  const testButton = document.getElementById("run-tests");
  if (testButton) testButton.addEventListener("click", () => runTests());

  const trainButton = document.getElementById("train");
  if (trainButton) trainButton.addEventListener("click", runPipeline);

  const recommendButton = document.getElementById("recommend-btn");
  if (recommendButton) {
    recommendButton.addEventListener("click", () => {
      const select = document.getElementById("user-select");
      renderTopK(select ? select.value : 1);
    });
  }

  const recallButton = document.getElementById("recall-btn");
  if (recallButton) recallButton.addEventListener("click", () => renderRecall(10));

  const status = document.getElementById("status");
  if (status) {
    status.textContent = `Dataset ready: ${movies.length.toLocaleString("en-US")} movies, ${trainSet.length.toLocaleString("en-US")} train ratings on ${tfBackend || "tf"}. Implement the TODO(hw5) functions, then press \u201cTrain\u201d.`;
  }
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

/** Marker carried by every unimplemented stub error message. */
const TODO_MARKER = "TODO(hw5)";

/** A five-rating toy training set used by several checks. */
const TOY_RATINGS = [
  { userId: 1, movieId: 1, rating: 5, timestamp: 1 },
  { userId: 1, movieId: 2, rating: 4, timestamp: 2 },
  { userId: 1, movieId: 3, rating: 3, timestamp: 3 },
  { userId: 2, movieId: 1, rating: 2, timestamp: 1 },
  { userId: 2, movieId: 2, rating: 5, timestamp: 2 },
];

/** Movies matching `TOY_RATINGS`. */
const TOY_MOVIES = [
  { id: 1, title: "Toy One", genres: [] },
  { id: 2, title: "Toy Two", genres: [] },
  { id: 3, title: "Toy Three", genres: [] },
];

/** A tiny train/val/test fixture for the recommendation checks. */
const REC_FIXTURE = {
  movies: [
    { id: 1, title: "Alpha", genres: [] },
    { id: 2, title: "Bravo", genres: [] },
    { id: 3, title: "Charlie", genres: [] },
    { id: 4, title: "Delta", genres: [] },
    { id: 5, title: "Echo", genres: [] },
  ],
  train: [{ userId: 1, movieId: 1, rating: 5, timestamp: 1 }],
  val: [{ userId: 1, movieId: 2, rating: 4, timestamp: 2 }],
  test: [{ userId: 1, movieId: 3, rating: 5, timestamp: 3 }],
};

/**
 * Build a TF.js factor model from sparse score maps.
 *
 * `userScores[id]` / `itemScores[id]` may be a number (written into factor 0,
 * so the dot product is just the product) or an array (written across factors).
 * Rows are dense and indexed by raw id, exactly like the real model, so the
 * recommendation / recall fixtures exercise the same tensor path as training.
 *
 * @param {Object<number, number|number[]>} userScores
 * @param {Object<number, number|number[]>} itemScores
 * @param {number} [globalMean=0]
 * @returns {Model}
 */
function fixtureModel(userScores, itemScores, globalMean = 0) {
  const F = NUM_FACTORS;
  const userIds = Object.keys(userScores).map(Number);
  const itemIds = Object.keys(itemScores).map(Number);

  // A number goes into factor 0 (so the dot product is the product); an array
  // is copied across the first factors.
  const rowsFor = (value) => {
    if (Array.isArray(value) || ArrayBuffer.isView(value)) return Array.from(value);
    const row = new Array(F).fill(0);
    row[0] = Number(value);
    return row;
  };

  const maxUser = Math.max(numUsers, ...userIds);
  const maxMovieFromMovies = movies.reduce((m, x) => Math.max(m, x.id), 0);
  const maxItem = Math.max(maxMovieFromMovies, ...itemIds);

  const uData = new Float32Array((maxUser + 1) * F);
  for (const id of userIds) {
    const row = rowsFor(userScores[id]);
    for (let f = 0; f < Math.min(F, row.length); f++) uData[id * F + f] = row[f];
  }
  const iData = new Float32Array((maxItem + 1) * F);
  for (const id of itemIds) {
    const row = rowsFor(itemScores[id]);
    for (let f = 0; f < Math.min(F, row.length); f++) iData[id * F + f] = row[f];
  }

  return {
    userFactors: tf.variable(tf.tensor2d(uData, [maxUser + 1, F])),
    itemFactors: tf.variable(tf.tensor2d(iData, [maxItem + 1, F])),
    globalMean,
  };
}

/**
 * A recall fixture with 12 movies, one user whose single relevant test movie
 * (movie 2) is placed at a chosen rank by the hand-made model.
 *
 * @param {number} targetRank 1 to put movie 2 first, 11 to put it last.
 * @returns {{movies: Movie[], train: Rating[], val: Rating[], test: Rating[], model: Model}}
 */
function recallFixture(targetRank) {
  const movies = [];
  for (let id = 1; id <= 12; id++) {
    movies.push({ id, title: `Movie ${id}`, genres: [] });
  }
  const train = [
    { userId: 1, movieId: 1, rating: 5, timestamp: 1 },
    { userId: 2, movieId: 2, rating: 5, timestamp: 1 },
    { userId: 2, movieId: 3, rating: 4, timestamp: 2 },
  ];
  const val = [];
  const test = [{ userId: 1, movieId: 2, rating: 5, timestamp: 2 }];
  const itemScores = { 1: 3.0 };
  if (targetRank === 1) {
    itemScores[2] = 5.0;
    for (let id = 3; id <= 12; id++) itemScores[id] = 1.0 + 0.01 * id;
  } else {
    itemScores[2] = 1.0;
    for (let id = 3; id <= 12; id++) itemScores[id] = 1.1 + 0.01 * id;
  }
  return { movies, train, val, test, model: fixtureModel({ 1: 1 }, itemScores) };
}

/**
 * Run `fn` with the module-level data globals temporarily replaced by a fixture,
 * then restore them. Keeps the checks independent of the loaded dataset.
 *
 * @param {{movies?: Movie[], train?: Rating[], val?: Rating[], test?: Rating[], model?: Model|null, history?: HistoryRow[]}} fixture
 * @param {() => any} fn
 * @returns {any}
 */
function withFixture(fixture, fn) {
  const saved = { movies, trainSet, valSet, testSet, model, trainingHistory };
  try {
    movies = fixture.movies !== undefined ? fixture.movies : movies;
    trainSet = fixture.train || [];
    valSet = fixture.val || [];
    testSet = fixture.test || [];
    model = fixture.model !== undefined ? fixture.model : null;
    trainingHistory = fixture.history || [];
    return fn();
  } finally {
    movies = saved.movies;
    trainSet = saved.trainSet;
    valSet = saved.valSet;
    testSet = saved.testSet;
    model = saved.model;
    trainingHistory = saved.trainingHistory;
  }
}

/**
 * Run the automated self-checks. Checks that call an unimplemented
 * `TODO(hw5)` stub are reported as PENDING rather than FAIL, so the harness is
 * useful before and after the assignment is implemented.
 *
 * @param {HTMLElement|null} [logElement] element that receives the text log
 * @returns {{passed: number, failed: number, pending: number, checks: Array<Object>}}
 */
function runTests(logElement) {
  const checks = [];
  const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

  const pass = (name, detail) => checks.push({ name, status: "PASS", detail });
  const fail = (name, detail) => checks.push({ name, status: "FAIL", detail });
  const pending = (name, detail) => checks.push({ name, status: "PENDING", detail });

  const check = (name, fn) => {
    try {
      const outcome = fn();
      if (outcome && outcome.pending) pending(name, outcome.pending);
      else pass(name, outcome === undefined ? "" : String(outcome));
    } catch (error) {
      if (String(error && error.message).includes(TODO_MARKER)) {
        pending(name, "TODO(hw5): not implemented yet.");
      } else {
        fail(name, String(error && error.message ? error.message : error));
      }
    }
  };

  // 1. Shared helper: baseline RMSE on a toy split.
  check("computeBaselineRMSE: toy split matches the hand-computed value", () =>
    withFixture(
      {
        train: [
          { userId: 1, movieId: 1, rating: 2, timestamp: 1 },
          { userId: 1, movieId: 2, rating: 4, timestamp: 2 },
        ],
      },
      () => {
        const rmse = computeBaselineRMSE(trainSet);
        if (!close(rmse, 1)) throw new Error(`rmse = ${rmse}, expected 1`);
        return `rmse=${fmt(rmse)}`;
      },
    ),
  );

  // 2. Shared helper: zero-denominator guard.
  check("computeBaselineRMSE: empty split returns 0 (no division by zero)", () =>
    withFixture({ train: [{ userId: 1, movieId: 1, rating: 3, timestamp: 1 }] }, () => {
      if (computeBaselineRMSE([]) !== 0) throw new Error("expected 0");
      return "guarded";
    }),
  );

  // 3. Formatting helper.
  check("fmt: formats numbers and guards non-finite input", () => {
    if (fmt(1.23456) !== "1.2346") throw new Error(`fmt(1.23456) = ${fmt(1.23456)}`);
    if (fmt(2, 2) !== "2.00") throw new Error(`fmt(2, 2) = ${fmt(2, 2)}`);
    if (fmt(Number.NaN) !== "\u2014") throw new Error("NaN must render as an em dash");
    return "ok";
  });

  // 4. Formatting helper.
  check("movieTitle: resolves ids and falls back for unknown ids", () =>
    withFixture({ movies: [{ id: 7, title: "Seven Samurai", genres: [] }] }, () => {
      if (movieTitle(7) !== "Seven Samurai") throw new Error("known id failed");
      if (movieTitle(999) !== "Movie #999") throw new Error(`fallback = ${movieTitle(999)}`);
      return "ok";
    }),
  );

  // 5. Student stub: prediction and clamp (TF.js gather + dot).
  check("predictRating: clamps to [1, 5] and falls back to the global mean", () =>
    withFixture(
      {
        movies: TOY_MOVIES,
        train: TOY_RATINGS,
        model: fixtureModel({ 1: [1, 2] }, { 3: [3, 4] }, 3),
      },
      () => {
        const pred = predictRating(1, 3);
        if (pred !== 5) throw new Error(`predictRating = ${pred}, expected 5 (3 + 11 clamped)`);
        const fallback = predictRating(100000, 3);
        if (fallback !== 3) throw new Error(`out-of-range user must return globalMean, got ${fallback}`);
        return `clamped=${pred}, fallback=${fallback}`;
      },
    ),
  );

  // 6. Student stub: TF.js variable allocation on the toy 5-rating set.
  check("initializeFactors: allocates tf.Variable factors and the global mean", () =>
    withFixture({ movies: TOY_MOVIES, train: TOY_RATINGS, model: null }, () => {
      const created = initializeFactors(TOY_RATINGS, 4);
      if (!created || !model) throw new Error("model was not assigned");
      if (!close(model.globalMean, 3.8)) {
        throw new Error(`globalMean = ${model.globalMean}, expected 3.8`);
      }
      if (!isTfModel(model)) throw new Error("userFactors/itemFactors are not tf tensors");
      if (model.userFactors.shape[1] !== 4 || model.itemFactors.shape[1] !== 4) {
        throw new Error(
          `factor width = ${model.userFactors.shape[1]} / ${model.itemFactors.shape[1]}, expected 4`,
        );
      }
      if (model.itemFactors.shape[0] !== 4) {
        throw new Error(`item rows = ${model.itemFactors.shape[0]}, expected 4 (max movie id 3 + 1)`);
      }
      if (model.userFactors.shape[0] < 3) {
        throw new Error(`user rows = ${model.userFactors.shape[0]}, expected at least 3`);
      }
      const row = model.userFactors.slice([1, 0], [1, 4]).dataSync();
      if (!Array.from(row).every(Number.isFinite)) {
        throw new Error("user factor row is not finite");
      }
      return `user [${model.userFactors.shape}], item [${model.itemFactors.shape}]`;
    }),
  );

  // 7. Student stub: batched training loop.
  check("trainMF: returns one finite history row per epoch", () =>
    withFixture({ movies: TOY_MOVIES, train: TOY_RATINGS, val: TOY_RATINGS, model: null }, () => {
      initializeFactors(TOY_RATINGS, 4);
      const history = trainMF(TOY_RATINGS, TOY_RATINGS, {
        epochs: 3,
        lr: 0.02,
        reg: 0.05,
        batchSize: 2,
      });
      if (!Array.isArray(history) || history.length !== 3) {
        throw new Error(`history length = ${history && history.length}, expected 3`);
      }
      history.forEach((row, i) => {
        if (row.epoch !== i + 1 && row.epoch !== i) {
          throw new Error(`row ${i} epoch = ${row.epoch}`);
        }
        if (!Number.isFinite(row.trainRMSE) || !Number.isFinite(row.valRMSE)) {
          throw new Error(`row ${i} RMSE is not finite`);
        }
      });
      return history.map((r) => `e${r.epoch}:${fmt(r.trainRMSE)}`).join(" ");
    }),
  );

  // 8. Student stub: checkpoint selection and tie-break.
  check("selectBestCheckpoint: smallest valRMSE wins, ties keep the earliest epoch", () => {
    const history = [
      { epoch: 2, trainRMSE: 0.9, valRMSE: 0.4 },
      { epoch: 1, trainRMSE: 1.0, valRMSE: 0.3 },
      { epoch: 3, trainRMSE: 0.8, valRMSE: 0.3 },
    ];
    const best = selectBestCheckpoint(history);
    if (!best || best.epoch !== 1) {
      throw new Error(`best epoch = ${best && best.epoch}, expected 1 (tie -> earliest)`);
    }
    return `best epoch ${best.epoch}`;
  });

  // 9. Student stub: recommendation filtering (tensor matMul + topk).
  check("recommendTopK: excludes train+val items but allows test-rated items", () =>
    withFixture(
      {
        movies: REC_FIXTURE.movies,
        train: REC_FIXTURE.train,
        val: REC_FIXTURE.val,
        test: REC_FIXTURE.test,
        model: fixtureModel({ 1: 1 }, { 1: 5, 2: 4, 3: 3, 4: 2, 5: 1 }),
      },
      () => {
        const recs = recommendTopK(1, 10);
        const ids = recs.map((r) => r.movieId);
        if (ids.includes(1)) throw new Error("movie 1 is in train and must be excluded");
        if (ids.includes(2)) throw new Error("movie 2 is in val and must be excluded");
        if (!ids.includes(3)) throw new Error("movie 3 is test-rated and must be eligible");
        for (let i = 1; i < recs.length; i++) {
          if (recs[i - 1].score < recs[i].score) {
            throw new Error("results are not sorted by descending score");
          }
        }
        return ids.join(", ");
      },
    ),
  );

  // 10. Student stub: recall at rank 1 vs rank 11.
  check("recallAtK: relevant movie at rank 1 scores 1.0; at rank 11 scores 0.0", () => {
    const rankOne = withFixture({ ...recallFixture(1), history: [] }, () => recallAtK(10));
    if (!close(rankOne.overallRecall, 1)) {
      throw new Error(`rank-1 recall = ${rankOne.overallRecall}, expected 1`);
    }
    const rankEleven = withFixture({ ...recallFixture(11), history: [] }, () => recallAtK(10));
    if (!close(rankEleven.overallRecall, 0)) {
      throw new Error(`rank-11 recall = ${rankEleven.overallRecall}, expected 0`);
    }
    return `rank1=${fmt(rankOne.overallRecall)} rank11=${fmt(rankEleven.overallRecall)}`;
  });

  // 11. Student stub: empty-result guard.
  check("recallAtK: empty test set returns 0 without dividing by zero", () =>
    withFixture(
      {
        movies: REC_FIXTURE.movies,
        train: REC_FIXTURE.train,
        val: [],
        test: [],
        model: fixtureModel({ 1: 1 }, { 1: 1, 2: 1, 3: 1, 4: 1, 5: 1 }),
      },
      () => {
        const result = recallAtK(10);
        if (result.overallRecall !== 0) {
          throw new Error(`expected 0, got ${result.overallRecall}`);
        }
        if (!Array.isArray(result.perUser) || result.perUser.length !== 0) {
          throw new Error("perUser must be empty");
        }
        if (!result.excluded) throw new Error("excluded report missing");
        return "guarded";
      },
    ),
  );

  const passed = checks.filter((c) => c.status === "PASS").length;
  const failed = checks.filter((c) => c.status === "FAIL").length;
  const pendingCount = checks.filter((c) => c.status === "PENDING").length;

  const lines = [
    `HW5 self-checks — pass ${passed}, fail ${failed}, pending ${pendingCount} (of ${checks.length})`,
    "",
    ...checks.map((c) => `${c.status.padEnd(7)} ${c.name}${c.detail ? ` \u2014 ${c.detail}` : ""}`),
  ];
  const text = lines.join("\n");

  const target = logElement || document.getElementById("testLog");
  if (target) target.textContent = text;

  return { passed, failed, pending: pendingCount, checks };
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

/**
 * Select the TF.js backend, load the MovieLens data, split it chronologically,
 * and wire the page. `data.js` must be loaded before this script.
 *
 * If TensorFlow.js is missing, `initTf` renders a clear message and this
 * function stops without throwing.
 *
 * @returns {Promise<void>}
 */
window.onload = async function () {
  const status = document.getElementById("status");
  const backend = await initTf();
  if (!backend) return;

  try {
    if (status) status.textContent = "Loading data\u2026";
    await loadData();
    const split = splitByTimestamp();
    trainSet = split.train;
    valSet = split.val;
    testSet = split.test;
    init();
  } catch (error) {
    console.error("Initialization failed:", error);
    if (status) status.textContent = `Error: ${error.message}`;
  }
};
