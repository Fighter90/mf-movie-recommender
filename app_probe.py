#!/usr/bin/env python3
"""
app_probe.py — drive the real HW5 page in headless Chromium and record what it shows.

The page needs HTTP (data.js fetches u.data and u.item), so a folder is served by a small built-in
web server on a free local port; an http(s) URL (for example the GitHub Pages site) is opened as is.
Nothing in the folder is changed.

    python3 app_probe.py --page . --inspect                       # starter: backend, summaries, self-tests
    python3 app_probe.py --page site --inspect --train --users 1 2 --recall \\
        --export results/page_model.json --screenshot figures/fig2_page.png
    python3 app_probe.py --page site --train --epochs 2 --capture-init --export results/page_short.json
    python3 app_probe.py --page site --retrain                    # two more Train clicks (1 epoch): history, tensors
    python3 app_probe.py --page site --backend cpu --train --epochs 2 --capture-init --export results/page_short_cpu.json

--train presses Train with the values in the page's inputs (20 / 0.01 / 0.05 / 64) unless --epochs, --lr,
--reg or --batch are given. --capture-init also stores the factors right after initializeFactors, so
mf_reference.py --replay can repeat the same run in numpy. --tfjs FILE serves a local copy of
tf.min.js instead of the CDN (only for machines that cannot reach cdn.jsdelivr.net). --backend switches
TensorFlow.js to that backend after the page has loaded (the page itself prefers webgl). Playwright's headless
Chromium usually renders WebGL in software (SwiftShader, see the "[tf] WebGL renderer" line), so training is
slower than in your own Chrome; --chrome uses the installed Google Chrome instead.
"""
import argparse, functools, http.server, json, os, pathlib, re, socketserver, sys, threading, time

from playwright.sync_api import sync_playwright

CDN = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--page", required=True, help="folder with index.html, or http(s) URL")
ap.add_argument("--inspect", action="store_true", help="print backend, dataset and split summaries, run the self-tests")
ap.add_argument("--train", action="store_true", help="press Train and wait for it")
ap.add_argument("--epochs", type=int)
ap.add_argument("--lr", type=float)
ap.add_argument("--reg", type=float)
ap.add_argument("--batch", type=int)
ap.add_argument("--capture-init", action="store_true", help="keep the factors produced by initializeFactors (for --replay)")
ap.add_argument("--retrain", action="store_true", help="press Train twice with 1 epoch; report history length and tensors")
ap.add_argument("--users", nargs="*", type=int, default=[], help="users whose Top-10 is requested through the page")
ap.add_argument("--recall", action="store_true", help="press Evaluate Recall@10")
ap.add_argument("--export", help="write the trained model and the page's results to this JSON file")
ap.add_argument("--screenshot", help="PNG of the whole page after the run (the per-user Recall table keeps its first 10 rows)")
ap.add_argument("--tfjs", help="local tf.min.js to serve instead of the CDN")
ap.add_argument("--backend", choices=["webgl", "cpu"], help="switch TensorFlow.js to this backend before anything else")
ap.add_argument("--chrome", action="store_true", help="use the installed Google Chrome instead of Playwright's Chromium "
                "(its headless mode can use the real GPU, which makes training faster)")
ap.add_argument("--api", action="store_true", help="list the TensorFlow.js calls in the page's script.js lines that differ from "
                "the starter and check in the page that each one exists")
ap.add_argument("--starter", default="script.js", help="starter script.js for --api (default: script.js in the current folder)")
a = ap.parse_args()

server = None
if re.match(r"https?://", a.page):
    url = a.page
else:
    folder = pathlib.Path(a.page).resolve()
    if not folder.joinpath("index.html").exists():
        sys.exit(f"no index.html in {folder}")

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(folder)))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/index.html"

out = {"page": a.page}

with sync_playwright() as p:
    exe = os.environ.get("PW_CHROMIUM")
    br = p.chromium.launch(**({"channel": "chrome"} if a.chrome else {"executable_path": exe} if exe else {}))
    pg = br.new_page(viewport={"width": 1400, "height": 1000})
    errors = []
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text}  [{(m.location or {}).get('url', '')}]"))
    if a.tfjs:
        pg.route(CDN, lambda route: route.fulfill(path=a.tfjs, content_type="application/javascript"))
    t0 = time.time()
    pg.goto(url)
    pg.wait_for_function("() => /Dataset ready|Error|failed/i.test(document.getElementById('status').textContent)", timeout=120000)
    print(f"[page] {a.page} ({url}) loaded in {time.time() - t0:.1f} s")
    print(f"[status] {pg.inner_text('#status')}")
    out["backend"] = pg.inner_text("#tf-status")
    print(f"[tf] {out['backend']}")
    if a.backend:
        now = pg.evaluate("async (b) => { await tf.setBackend(b); await tf.ready(); return tf.getBackend(); }", a.backend)
        print(f"[tf] switched by this script to: {now} (the page's own status line still names the backend it chose)")
    out["tf"] = pg.evaluate("""() => {
        const gl = document.createElement('canvas').getContext('webgl2');
        const info = gl && gl.getExtension('WEBGL_debug_renderer_info');
        return { version: tf.version.tfjs, backend: tf.getBackend(), epsilon: tf.backend().epsilon(),
                 renderer: info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : (gl ? gl.getParameter(gl.RENDERER) : 'no WebGL2') }; }""")
    print(f"[tf] TensorFlow.js {out['tf']['version']} on {out['tf']['backend']}; backend epsilon {out['tf']['epsilon']:g} "
          "(tf.train.adam(lr) uses it as Adam's epsilon when none is passed)")
    print(f"[tf] WebGL renderer of this browser: {out['tf']['renderer']}")

    if a.api:
        page_js = pg.evaluate("async () => (await fetch('script.js', { cache: 'no-store' })).text()")
        start_lines = set(l.strip() for l in pathlib.Path(a.starter).read_text(encoding="utf-8").splitlines())
        added = [l for l in page_js.splitlines() if l.strip() and l.strip() not in start_lines and not l.strip().startswith(("//", "*", "/*"))]
        names = sorted(set(re.findall(r"\btf\.([A-Za-z_][\w.]*)\s*\(", "\n".join(added))))
        methods = sorted(set(re.findall(r"\.(mul|sum|square|mean|matMul|squeeze|assign|clone|dataSync|arraySync|minimize|sub|div|reshape|transpose|expandDims|neg)\s*\(", "\n".join(added))))
        check = pg.evaluate("""([names, methods]) => {
            const t = tf.tensor1d([1, 2]); const v = tf.variable(tf.zeros([2])); const o = tf.train.adam(0.01);
            const out = { tf: {}, methods: {} };
            for (const n of names) {
              let x = tf; for (const part of n.split('.')) x = x == null ? undefined : x[part];
              out.tf[n] = typeof x;
            }
            for (const m of methods) out.methods[m] = [typeof t[m] === 'function' ? 'Tensor' : null, typeof v[m] === 'function' ? 'Variable' : null,
                                                      typeof o[m] === 'function' ? 'Optimizer' : null].filter(Boolean).join('/') || 'none';
            tf.dispose([t, v]); o.dispose();
            return out; }""", [names, methods])
        print(f"[api] {len(added)} non-comment lines of script.js differ from the starter ({a.starter}); TensorFlow.js calls in them:")
        for n in names:
            kind = check["tf"][n]
            print(f"    tf.{n:32s} " + ("exists (function)" if kind == "function" else f"DOES NOT EXIST as a function (typeof: {kind})"))
        print("[api] tensor / variable / optimizer methods called in those lines: "
              + (", ".join(f".{m} ({check['methods'][m]})" for m in methods) or "none"))
        out["api"] = {"tf": check["tf"], "methods": check["methods"], "added_lines": len(added)}

    if a.inspect:
        print("[dataset] " + pg.inner_text("#dataset-summary-body").replace("\n", " | "))
        print("[split] " + re.sub(r"\s+", " ", pg.inner_text("#split-summary-body")))
        pg.click("#run-tests")
        log = pg.inner_text("#testLog")
        print("[tests]\n" + "\n".join("    " + line for line in log.splitlines() if line.strip()))
        out["tests"] = log.splitlines()[0]

    if a.capture_init:
        pg.evaluate("""() => {
            const original = initializeFactors;
            window.initializeFactors = function (...args) {
              const m = original.apply(this, args);
              window.__initFactors = { users: Array.from(m.userFactors.dataSync()), userShape: m.userFactors.shape,
                                       items: Array.from(m.itemFactors.dataSync()), itemShape: m.itemFactors.shape,
                                       globalMean: m.globalMean };
              return m;
            };
        }""")

    def set_inputs():
        for key, el in (("epochs", "epochs"), ("lr", "lr"), ("reg", "reg"), ("batch", "batch-size")):
            v = getattr(a, key)
            if v is not None:
                pg.fill(f"#{el}", str(v))
        return pg.evaluate("() => readHyperparameters()")

    if a.train:
        hp = set_inputs()
        print(f"[train] hyper-parameters read by the page: {hp}")
        t = time.time()
        pg.evaluate("() => document.getElementById('train').click()")  # synchronous: returns when training ends
        secs = time.time() - t
        status = pg.inner_text("#status")
        print(f"[train] {secs:.0f} s ({secs / max(1, hp['epochs']):.1f} s per epoch); status: {status}")
        res = pg.evaluate("""() => ({
            history: trainingHistory.map(r => ({ epoch: r.epoch, trainRMSE: r.trainRMSE, valRMSE: r.valRMSE })),
            best: bestCheckpoint, ok: !!model,
            valNow: model ? computeRMSEFor(valSet) : null, testRMSE: model ? computeRMSEFor(testSet) : null,
            baseVal: computeBaselineRMSE(valSet), baseTest: computeBaselineRMSE(testSet),
            globalMean: model ? model.globalMean : null, tensors: tf.memory().numTensors })""")
        rows = pg.eval_on_selector_all("#training-curve tbody tr", "rs => rs.map(r => [...r.cells].map(c => c.textContent.trim()))")
        print(f"[curve] {len(rows)} rows as rendered (epoch, train RMSE, val RMSE):")
        for r in rows:
            print("    " + "  ".join(r))
        b = res["best"]
        if b:
            print(f"[best] epoch {b['epoch']}: train {b['trainRMSE']:.4f}, val {b['valRMSE']:.4f} (selectBestCheckpoint)")
            same = res["valNow"] is not None and abs(res["valNow"] - b["valRMSE"]) < 1e-9
            print(f"[best] val RMSE of the model now in memory {res['valNow']:.4f}: "
                  + ("the selected checkpoint is the model in memory" if same else
                     "NOT the selected checkpoint (the factors of another epoch are in memory)"))
        if res["testRMSE"] is not None:
            print(f"[test] RMSE of the model in memory {res['testRMSE']:.4f}; training-mean baseline {res['baseTest']:.4f} "
                  f"(difference {res['baseTest'] - res['testRMSE']:+.4f}); validation baseline {res['baseVal']:.4f}")
            print(f"[mean] globalMean {res['globalMean']:.6f}; live tensors {res['tensors']}")
        out.update(train_seconds=round(secs, 1), status=status, hyper=hp, curve=rows, **res)
        if a.capture_init:
            out["init"] = pg.evaluate("() => window.__initFactors || null")
            print(f"[init] captured initial factors: {'yes' if out['init'] else 'no (initializeFactors was not called)'}")

    if a.users:
        out["topk"] = {}
        for u in a.users:
            pg.select_option("#user-select", str(u))
            pg.click("#recommend-btn")
            shown = pg.eval_on_selector_all("#recommendations li", "ls => ls.map(l => [l.querySelector('.rec-title').textContent, l.querySelector('.rec-score').textContent])")
            raw = pg.evaluate("(u) => { try { return recommendTopK(u, 10).map(r => [r.movieId, r.title, r.score]); } catch (e) { return String(e && e.message || e); } }", u)
            rated = pg.evaluate("(u) => trainSet.concat(valSet).filter(r => r.userId === u).map(r => r.movieId)", u)
            if isinstance(raw, str):
                print(f"[top10] user {u}: recommendTopK threw: {raw}; the page shows: {pg.inner_text('#recommendations')[:200]}")
                out.setdefault("topk", {})[str(u)] = {"error": raw}
                continue
            clash = [m for m, _, _ in raw if m in set(rated)]
            print(f"[top10] user {u} ({len(rated)} movies rated in train + val; listed movies among them: {len(clash)}):")
            for (mid, title, score), (t_shown, s_shown) in zip(raw, shown):
                print(f"    {mid:5d}  {s_shown:>7s}  {t_shown}")
            if not shown:
                print("    " + pg.inner_text("#recommendations")[:200])
            out["topk"][str(u)] = {"rendered": shown, "raw": raw, "rated_train_val": len(rated), "rated_in_list": clash}

    if a.recall:
        t = time.time()
        pg.click("#recall-btn")
        text = pg.inner_text("#recall-results")
        head = re.search(r"Recall@10:\s*([\d.]+)", text)
        hint = re.search(r"Excluded:.*", text)
        print(f"[recall] page shows Recall@10 {head.group(1) if head else '?'} ({time.time() - t:.1f} s); {hint.group(0) if hint else text[:200]}")
        res = pg.evaluate("() => { try { const r = recallAtK(10); return { overall: r.overallRecall, perUser: r.perUser, unseen: r.excluded.unseenIds, without: r.excluded.usersWithoutRelevant }; } catch (e) { return { error: String(e && e.message || e) }; } }")
        if "error" in res:
            print(f"[recall] recallAtK threw: {res['error']}")
            out["recall"] = res
            res = None
    if a.recall and res:
        hits = sum(u["hit"] for u in res["perUser"])
        print(f"[recall] recallAtK(10): overall {res['overall']:.6f} over {len(res['perUser'])} users; hits {hits}; users with >= 1 hit "
              f"{sum(u['hit'] > 0 for u in res['perUser'])}; unseen relevant ids {len(res['unseen'])}; users without a relevant candidate {len(res['without'])}")
        out["recall"] = res

    if a.export:
        if a.train:
            f = pg.evaluate("() => model ? { users: Array.from(model.userFactors.dataSync()), userShape: model.userFactors.shape, items: Array.from(model.itemFactors.dataSync()), itemShape: model.itemFactors.shape } : null")
            out["factors"] = f
        pathlib.Path(a.export).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.export).write_text(json.dumps(out), encoding="utf-8")
        print(f"[export] {a.export} ({pathlib.Path(a.export).stat().st_size / 1e6:.1f} MB)")

    if a.screenshot:
        pathlib.Path(a.screenshot).parent.mkdir(parents=True, exist_ok=True)
        # the per-user Recall table has a row for every evaluated user (861): only its first 10 rows go into the picture
        rows = pg.evaluate("() => document.querySelectorAll('#recall-results tbody tr').length")
        cut = pg.add_style_tag(content="#recall-results tbody tr:nth-child(n+11) { display: none; }") if rows > 10 else None
        pg.screenshot(path=a.screenshot, full_page=True)
        if cut:
            cut.evaluate("e => e.remove()")
        print(f"[screenshot] {a.screenshot}" + (f" (whole page; the per-user Recall table shows its first 10 of {rows} rows)" if cut else ""))

    if a.retrain:  # last on purpose: it changes the model and the history that the steps above report
        pg.fill("#epochs", "1")
        counts = []
        for k in range(2):
            pg.evaluate("() => document.getElementById('train').click()")
            counts.append(pg.evaluate("() => ({ history: trainingHistory.length, tensors: tf.memory().numTensors, status: document.getElementById('status').textContent })"))
        print(f"[retrain] two Train clicks with epochs = 1: history rows {counts[0]['history']} then {counts[1]['history']}; "
              f"live tensors {counts[0]['tensors']} then {counts[1]['tensors']}")
        print(f"[retrain] status after the second click: {counts[1]['status']}")
        out["retrain"] = counts

    print(f"[console] errors: {len(errors)}" + "".join("\n    " + e for e in errors[:5]))
    if errors and all(e.rstrip("]").endswith("/favicon.ico") for e in errors):
        print("[console] note: the only error is a 404 for /favicon.ico; index.html declares no icon, so the browser asks the server root for one.")
    br.close()
if server:
    server.shutdown()
