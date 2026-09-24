"""Render one icon: input | filled result | outline with anchors (red) and handles (blue).
Usage: python tools/debug_view.py IMAGE OUT.png [zoom]"""
import io, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, cv2, cairosvg
from PIL import Image
from icon2svg import Options, trace, load_image
from icon2svg.bezier import bez

f, out = sys.argv[1], sys.argv[2]
Z = int(sys.argv[3]) if len(sys.argv) > 3 else 8
img = load_image(f)
opt = Options()
opt.mode = sys.argv[4] if len(sys.argv) > 4 else "auto"
svg, info, shapes = trace(img, opt, return_shapes=True)
H, W = img.shape[:2]
png = cairosvg.svg2png(bytestring=svg.encode(), output_width=W * Z, output_height=H * Z, background_color="white")
r = cv2.cvtColor(np.asarray(Image.open(io.BytesIO(png)).convert("RGB")), cv2.COLOR_RGB2BGR)
src = cv2.resize(img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (W * Z, H * Z), interpolation=cv2.INTER_NEAREST)
ov = np.full_like(r, 255)
ov = (0.25 * src + 0.75 * ov).astype(np.uint8)
P = lambda p: (int(round(p[0] * Z * 4)), int(round(p[1] * Z * 4)))
for kind, data in shapes:
    if kind == "circle":
        cx, cy, rr = data
        cv2.circle(ov, P((cx, cy)), int(rr * Z * 4), (0, 0, 0), 1, cv2.LINE_AA, 2)
        for a in range(4):
            q = (cx + rr * np.cos(a * np.pi / 2), cy + rr * np.sin(a * np.pi / 2))
            cv2.circle(ov, P(q), 3 * 4, (0, 0, 255), -1, cv2.LINE_AA, 2)
        continue
    cur, segs = data
    for sg in segs:
        if sg[0] == "L":
            cv2.line(ov, P(cur), P(sg[1]), (0, 0, 0), 1, cv2.LINE_AA, 2)
        else:
            c = np.array([cur, sg[1], sg[2], sg[3]])
            pts = bez(c, np.linspace(0, 1, 30))
            cv2.polylines(ov, [np.array([P(p) for p in pts])], False, (0, 0, 0), 1, cv2.LINE_AA, 2)
            cv2.line(ov, P(cur), P(sg[1]), (255, 150, 0), 1, cv2.LINE_AA, 2)
            cv2.line(ov, P(sg[3]), P(sg[2]), (255, 150, 0), 1, cv2.LINE_AA, 2)
        cur = sg[-1]
        cv2.circle(ov, P(cur), 3 * 4, (0, 0, 255), -1, cv2.LINE_AA, 2)
cv2.imwrite(out, np.hstack([src, r, ov]))
print(info)
