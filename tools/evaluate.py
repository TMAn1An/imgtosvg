"""Trace a folder, render the SVGs back and report fidelity + anchor counts.
Usage: python tools/evaluate.py INPUT_DIR OUT_DIR"""
import glob, io, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
import cairosvg
from PIL import Image
from icon2svg import Options, trace, load_image
from icon2svg.tracer import to_ink

inp, out = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "auto"
os.makedirs(out, exist_ok=True)
Z = 6
rows = []
sheet = []
for f in sorted(glob.glob(os.path.join(inp, "*")), key=lambda p: (len(p), p)):
    img = load_image(f)
    t = time.time()
    svg, info = trace(img, Options(mode=mode))
    dt = time.time() - t
    name = os.path.splitext(os.path.basename(f))[0]
    open(os.path.join(out, name + ".svg"), "w").write(svg)
    H, W = img.shape[:2]
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=W * Z, output_height=H * Z, background_color="white")
    r = np.asarray(Image.open(io.BytesIO(png)).convert("L"), np.float32) / 255
    ink, _ = to_ink(img)
    big = cv2.resize(ink, (W * Z, H * Z), interpolation=cv2.INTER_CUBIC)
    a, b = big > 0.5, (1 - r) > 0.5
    iou = (a & b).sum() / max((a | b).sum(), 1)
    rows.append((name, info["anchors"], iou, dt))
    sheet.append(r)
    print(f"{name:10s} {info['mode']:7s} anchors={info['anchors']:4d}  IoU={iou:.3f}  {dt:.2f}s")
    side = np.hstack([(1 - np.clip(big, 0, 1)) * 255, np.full((H * Z, 10), 128), r * 255]).astype(np.uint8)
    Image.fromarray(side).save(os.path.join(out, name + "_cmp.png"))
print("mean IoU %.3f  total anchors %d" % (np.mean([r[2] for r in rows]), sum(r[1] for r in rows)))

from PIL import Image as I
tiles = [I.fromarray((t * 255).astype(np.uint8)).resize((260, 260), I.LANCZOS) for t in sheet]
S = I.new("L", (260 * 4, 260 * ((len(tiles) + 3) // 4)), 255)
for k, t in enumerate(tiles):
    S.paste(t, ((k % 4) * 260, (k // 4) * 260))
S.save(os.path.join(out, "_sheet.png"))
