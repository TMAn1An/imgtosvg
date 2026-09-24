"""Test how well a local vision model (Ollama) helps with icon vectorizing.

For every icon and model it runs two experiments:
  guide  : the model only LISTS the shapes (JSON, 0-100 grid); math draws them
           and snaps them onto the ink (refine_shapes)
  redraw : the model writes the whole SVG (the normal AI redraw, 1 round)
and prints a score (F1 vs the input, 1.0 = perfect), the time taken, and saves
a comparison sheet  <out>/<model>.png  (input | guide raw | guide snapped | redraw).

Usage (Ollama running, models pulled):
    python tools/local_model_test.py icons_folder --models gemma3:4b qwen2.5vl:3b
"""
import argparse
import json
import os
import re
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icon2svg.ai import ProviderError, make_provider, redraw, render_svg, score_svg, _input_png  # noqa: E402
from icon2svg.geometry import ellipse_shape, polygon_shape, rect_shape  # noqa: E402
from icon2svg.refine import refine_shapes  # noqa: E402
from icon2svg.tracer import BASE, Options, fmt, load_image, shapes_to_d, to_ink, trace  # noqa: E402

GUIDE_SYSTEM = "You are an expert icon designer. You describe line icons as a list of simple geometric shapes."
GUIDE_PROMPT = """Look at this black line icon. Describe it as the simple shapes a designer would draw
with a pen tool, on a 100 x 100 grid (x to the right, y down, 0,0 = top-left corner).
Answer with JSON only, no explanation:
{"shapes": [
  {"type": "circle", "cx": 50, "cy": 30, "r": 10},
  {"type": "ellipse", "cx": 50, "cy": 80, "rx": 20, "ry": 6},
  {"type": "rect", "x": 20, "y": 40, "w": 60, "h": 30, "rx": 4},
  {"type": "line", "points": [[10, 90], [90, 90]]},
  {"type": "polyline", "points": [[20, 20], [30, 10], [40, 20]]},
  {"type": "polygon", "points": [[45, 5], [55, 5], [50, 0]]},
  {"type": "curve", "points": [[10, 50], [30, 40], [50, 50]]}
]}
Every line of the icon must be covered by a shape. Use "curve" for free-form lines (points along the line).
The icon is {W} x {H} pixels; the grid maps its longer side to 100."""


def parse_json(text):
    text = re.sub(r"```(json)?", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < 0:
        a, b = text.find("["), text.rfind("]")
    raw = text[a:b + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))  # trailing commas
    return data.get("shapes", []) if isinstance(data, dict) else data


def guide_shapes(items, k):
    """Model JSON (0-100 grid) -> internal shapes in pixel units (k px per unit)."""
    from icon2svg.bezier import fit_curve  # noqa: F401  (smooth curves)
    out = []
    num = lambda d, key, dv=0.0: float(d.get(key, dv)) * k
    for it in items:
        try:
            t = str(it.get("type", "")).lower()
            if t == "circle":
                out.append(("circle", (num(it, "cx"), num(it, "cy"), max(num(it, "r"), 0.5))))
            elif t == "ellipse":
                out.append(ellipse_shape(num(it, "cx"), num(it, "cy"), max(num(it, "rx"), 0.5), max(num(it, "ry"), 0.5)))
            elif t == "rect":
                x, y, w, h = num(it, "x"), num(it, "y"), num(it, "w"), num(it, "h")
                r = min(num(it, "rx"), w / 2, h / 2)
                if w > 0 and h > 0:
                    out.append(rect_shape(x, y, x + w, y + h, max(r, 0)))
            elif t in ("line", "polyline", "polygon", "curve"):
                P = [np.array([float(p[0]), float(p[1])]) * k for p in it.get("points", []) if len(p) >= 2]
                if len(P) < 2:
                    continue
                if t == "curve" and len(P) >= 3:
                    out.append(_smooth_open(P))
                else:
                    out.append(polygon_shape(P, t == "polygon"))
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def _smooth_open(P):
    """Catmull-Rom through the points -> cubic segments."""
    P = [P[0]] + P + [P[-1]]
    segs = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        segs.append(("C", p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6, p2))
    return ("open", (P[1], segs))


def to_svg(shapes, W, H, sw):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {fmt(W, 2)} {fmt(H, 2)}">'
            f'<path fill="none" stroke="#000" stroke-width="{fmt(sw, 2)}" stroke-linecap="round" '
            f'stroke-linejoin="round" d="{shapes_to_d(shapes, 2)}"/></svg>')


def tile(svg, size, label):
    im = (render_svg(svg, size, size) * 255).astype(np.uint8) if svg else np.full((size, size), 235, np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    cv2.putText(im, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--models", nargs="+", default=["gemma3:4b"])
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--n", type=int, default=16, help="how many icons")
    ap.add_argument("--out", default="local_model_test")
    ap.add_argument("--skip-redraw", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in os.listdir(a.folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))[: a.n]
    summary = []
    for model in a.models:
        prov = make_provider("openai", api_key="ollama", model=model, base_url=a.base_url)
        rows, res = [], []
        for fn in files:
            img = load_image(os.path.join(a.folder, fn))
            ink, _ = to_ink(img)
            H, W = ink.shape
            _, info = trace(img, Options(mode="stroke"))
            sw = info["stroke_width"]
            k = max(H, W) / 100.0
            row = {"icon": fn}
            tiles = [tile(None, 256, fn)]
            tiles[0][:] = cv2.resize(cv2.cvtColor(((1 - ink) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), (256, 256))
            # ---- guide
            t0 = time.time()
            try:
                txt = prov.generate(GUIDE_SYSTEM, [{"image": _input_png(img)},
                                                   {"text": GUIDE_PROMPT.replace("{W}", str(W)).replace("{H}", str(H))}],
                                    log=lambda *x: None)
                sh = guide_shapes(parse_json(txt), k)
                raw = to_svg(sh, W, H, sw)
                snapped = to_svg(refine_shapes(sh, ink, sw, iters=6), W, H, sw) if sh else raw
                row["guide_raw"] = round(score_svg(raw, ink, sw)[0], 3)
                row["guide_snap"] = round(score_svg(snapped, ink, sw)[0], 3)
                row["guide_shapes"] = len(sh)
                tiles += [tile(raw, 256, f"guide {row['guide_raw']}"), tile(snapped, 256, f"snapped {row['guide_snap']}")]
            except (ProviderError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                row["guide_error"] = str(e)[:120]
                tiles += [tile(None, 256, "guide failed"), tile(None, 256, "")]
            row["guide_s"] = round(time.time() - t0, 1)
            # ---- full redraw
            if not a.skip_redraw:
                t0 = time.time()
                try:
                    svg, rinfo = redraw(img, prov, rounds=0, examples=None, log=lambda *x: None, fallback=False)
                    row["redraw"] = round(score_svg(svg, ink, sw)[0], 3)
                    tiles.append(tile(svg, 256, f"redraw {row['redraw']}"))
                except Exception as e:  # noqa: BLE001
                    row["redraw_error"] = str(e)[:120]
                    tiles.append(tile(None, 256, "redraw failed"))
                row["redraw_s"] = round(time.time() - t0, 1)
            print(model, json.dumps(row), flush=True)
            res.append(row)
            rows.append(np.hstack(tiles))
        cv2.imwrite(os.path.join(a.out, model.replace(":", "_").replace("/", "_") + ".png"), np.vstack(rows))
        mean = lambda key: np.mean([r.get(key, 0.0) for r in res])
        summary.append((model, mean("guide_raw"), mean("guide_snap"), mean("redraw"), mean("guide_s"), mean("redraw_s")))
    print("\nmodel               guide  snapped  redraw   s/guide  s/redraw")
    for m, g, s, r, gs, rs in summary:
        print(f"{m:18s} {g:6.3f} {s:8.3f} {r:7.3f} {gs:9.1f} {rs:9.1f}")


if __name__ == "__main__":
    main()
