#!/usr/bin/env python3
"""Local web UI for icon2svg.  Run:  python app.py   then open http://127.0.0.1:5000"""
import io
import os
import threading
import webbrowser
import zipfile

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

from icon2svg import Options, trace

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(HERE, "web"))
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


def _opt_from(form):
    def num(key, default):
        try:
            return float(form.get(key, default))
        except (TypeError, ValueError):
            return default
    return Options(
        mode=form.get("mode", "auto"),
        tolerance=num("tolerance", 0.30),
        stroke_width=num("stroke_width", 0),
        size=num("size", 0),
        color=form.get("color") or "auto",
        detect_circles=form.get("circles", "1") == "1",
    )


@app.post("/api/convert")
def convert():
    opt = _opt_from(request.form)
    results = []
    for f in request.files.getlist("files"):
        name = os.path.splitext(os.path.basename(f.filename or "icon"))[0]
        try:
            data = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("unsupported image")
            svg, info = trace(img, opt)
            results.append({"name": name, "svg": svg, **{k: (float(v) if isinstance(v, np.floating) else v)
                                                         for k, v in info.items()}})
        except Exception as e:
            results.append({"name": name, "error": str(e)})
    return jsonify(results)


# ------------------------------------------------------------ AI redraw --
from icon2svg import ai as _ai  # noqa: E402

_providers = {}


def _provider(form):
    cfg = _ai.load_config()
    name = form.get("provider") or cfg.get("provider") or "gemini"
    key = form.get("key") or _ai.key_for(name, cfg)
    model = form.get("model") or "auto"
    base = form.get("base_url") or None
    ck = (name, key, model, base)
    if ck not in _providers:
        _providers[ck] = _ai.make_provider(name, key, model, base)
    return _providers[ck]


@app.get("/api/ai/config")
def ai_config():
    cfg = _ai.load_config()
    return jsonify({"provider": cfg.get("provider", "gemini"),
                    "has_key": {k: bool(v) for k, v in (cfg.get("keys") or {}).items()}})


@app.post("/api/ai/config")
def ai_config_save():
    data = request.get_json(force=True) or {}
    cfg = _ai.load_config()
    if data.get("provider"):
        cfg["provider"] = data["provider"]
    if data.get("key"):
        cfg.setdefault("keys", {})[data.get("provider", "gemini")] = data["key"]
    _ai.save_config(cfg)
    return jsonify({"ok": True})


@app.post("/api/ai/models")
def ai_models():
    try:
        prov = _provider(request.form)
        return jsonify({"models": prov.list_models(), "auto": prov.resolve_model()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/ai/convert")
def ai_convert():
    f = request.files.get("file")
    if f is None:
        return jsonify({"error": "no file"}), 400
    name = os.path.splitext(os.path.basename(f.filename or "icon"))[0]
    logs = []
    try:
        prov = _provider(request.form)
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("unsupported image")
        n_ex = int(request.form.get("examples", 2))
        examples = _ai.load_examples(os.path.join(HERE, "examples"), n_ex) if n_ex > 0 else []
        sw = float(request.form.get("stroke_width") or 0) or None
        svg, info = _ai.redraw(img, prov, rounds=int(request.form.get("rounds", 2)), examples=examples,
                               stroke_width=sw, log=logs.append, n_examples=n_ex)
        return jsonify({"name": name, "svg": svg, "log": logs, **info})
    except Exception as e:
        return jsonify({"name": name, "error": str(e), "log": logs})


@app.post("/api/zip")
def make_zip():
    items = request.get_json(force=True) or []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        used = set()
        for it in items:
            base = it.get("name", "icon")
            name, k = base, 1
            while name in used:
                k += 1
                name = f"{base}_{k}"
            used.add(name)
            z.writestr(name + ".svg", it.get("svg", ""))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="svg_icons.zip")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    url = f"http://127.0.0.1:{port}"
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\n  icon2svg is running at {url}  (press Ctrl+C to stop)\n")
    app.run(host="127.0.0.1", port=port, debug=False)
