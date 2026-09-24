"""Raster icon -> clean vector outline.

Pipeline
  1. load + normalise ink (auto polarity, alpha aware, contrast stretch)
  2. bicubic super-sampling + light blur, sub-pixel iso-contours
  3. per contour: corner detection, straight-run detection, circle detection
  4. primitive fitting: lines (axis snapped) + tangent-continuous cubic Beziers
     with a tolerance-driven split that prefers x/y extrema for anchors
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage import measure

from .bezier import fit_curve, line_fit, intersect, snap_dir, unit, simplify, fillet, snap_axis

BASE = 130.0  # all length parameters are tuned for a 130 px icon


@dataclass
class Options:
    tolerance: float = 0.30       # max curve deviation (px @130)
    line_tolerance: float = 0.09  # max deviation of a straight segment
    corner_angle: float = 50.0    # min turning (deg) to count as a corner
    min_line: float = 2.4         # min length of a straight segment
    axis_snap: float = 5.0        # snap lines within N degrees to H/V
    min_area: float = 1.5         # drop specks smaller than this (px^2)
    threshold: float = 0.5        # iso level on the normalised ink map
    blur: float = 0.35            # pre-blur sigma (px)
    detect_circles: bool = True
    color: str = "auto"           # "auto" or a css colour
    mode: str = "auto"            # "auto" | "stroke" | "outline"
    decimals: int = 2
    scale: float = 1.0            # output scale factor
    size: float = 0               # >0: scale output so the longest side is `size`
    stroke_width: float = 0       # >0: force this stroke width (output units)
    symmetry: bool = True         # make mirror-symmetric parts exactly symmetric
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- image ----
def load_image(path):
    data = np.fromfile(path, np.uint8)  # works with unicode paths on Windows
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def to_ink(img):
    """Return (ink map in [0,1] where 1 = ink, ink colour hex)."""
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = None
    elif img.shape[2] == 4:
        rgb = img[:, :, :3]
        alpha = img[:, :, 3].astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
    else:
        rgb = img[:, :, :3]
        alpha = None
    if rgb.dtype == np.uint16:
        rgb = (rgb / 257).astype(np.uint8)
    rgbf = rgb.astype(np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    if alpha is not None and alpha.min() < 0.9:
        # transparent icon: the alpha channel is the shape
        ink = alpha
    else:
        border = np.r_[gray[0], gray[-1], gray[:, 0], gray[:, -1]]
        bg = np.median(border)
        ink = np.abs(gray - bg)
        # use colour distance too, so coloured ink on white works
        bgc = np.median(np.r_[rgbf[0], rgbf[-1], rgbf[:, 0], rgbf[:, -1]], axis=0)
        cdist = np.linalg.norm(rgbf - bgc, axis=2) / np.sqrt(3)
        ink = np.maximum(ink, cdist)

    lo = np.percentile(ink, 1)
    strong = ink[ink > (lo + ink.max()) / 2]
    hi = np.percentile(strong, 90) if strong.size else ink.max()
    ink = np.clip((ink - lo) / max(hi - lo, 1e-6), 0, 1)

    mask = ink > 0.85
    if mask.sum() > 0:
        c = np.median(rgb[mask], axis=0).astype(int)  # BGR
        if c.max() < 40 and c.max() - c.min() < 12:
            c[:] = 0  # near-black JPEG ink -> pure black
        color = "#{:02x}{:02x}{:02x}".format(c[2], c[1], c[0])
    else:
        color = "#000000"
    return ink, color


def extract_contours(ink, opt, s):
    """Sub-pixel closed contours in pixel coordinates of `ink`."""
    h, w = ink.shape
    up = int(np.clip(np.ceil(1200 / max(h, w)), 2, 8))
    big = cv2.resize(ink, (w * up, h * up), interpolation=cv2.INTER_CUBIC)
    sig = opt.blur * s * up
    if sig > 0.3:
        big = cv2.GaussianBlur(big, (0, 0), sig)
    big = np.pad(big, 1, constant_values=0)
    raw = measure.find_contours(big, opt.threshold)
    out = []
    for c in raw:
        if len(c) < 8:
            continue
        pts = np.c_[c[:, 1] - 1, c[:, 0] - 1]
        pts = (pts + 0.5) / up  # to source pixel units (pixel i spans [i,i+1])
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        area = 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
        if area < opt.min_area * s * s:
            continue
        out.append(pts)
    return out


def _mirror(img, c):
    """Mirror img horizontally about the vertical line x = c (pixel units)."""
    h, w = img.shape
    M = np.float32([[-1, 0, 2 * c - 1], [0, 1, 0]])  # pixel centres at i+0.5
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def symmetrize(ink, min_score=0.80):
    """If the icon (or a large part of it) is mirror-symmetric about a vertical
    axis, average both halves wherever they agree. Asymmetric details are kept."""
    h, w = ink.shape
    cols = ink.sum(0)
    if cols.sum() <= 0:
        return ink
    cx = (cols * (np.arange(w) + 0.5)).sum() / cols.sum()
    best = (-1, cx)
    for c in np.arange(cx - 0.12 * w, cx + 0.12 * w, 0.25):
        m = _mirror(ink, c)
        inter = np.minimum(ink, m).sum()
        union = np.maximum(ink, m).sum()
        sc = inter / max(union, 1e-6)
        if sc > best[0]:
            best = (sc, c)
    for c in np.arange(best[1] - 0.25, best[1] + 0.25, 0.05):
        m = _mirror(ink, c)
        sc = np.minimum(ink, m).sum() / max(np.maximum(ink, m).sum(), 1e-6)
        if sc > best[0]:
            best = (sc, c)
    score, c = best
    if score < min_score:
        return ink
    m = _mirror(ink, c)
    # agree = both sides have (roughly) the same ink here, allowing a small shift
    near = cv2.dilate(m, np.ones((3, 3), np.uint8))
    near_i = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    agree = (np.abs(ink - m) < 0.5) | ((ink > 0.5) & (near > 0.5) & (near_i > 0.5) & (m > 0.15))
    agree = cv2.GaussianBlur(agree.astype(np.float32), (0, 0), 0.8) > 0.5
    out = ink.copy()
    avg = (ink + m) / 2
    out[agree] = avg[agree]
    return out


# -------------------------------------------------------------- geometry ---
def resample_closed(pts, ds):
    p = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    d = np.r_[0, np.cumsum(seg)]
    L = d[-1]
    n = max(12, int(round(L / ds)))
    t = np.linspace(0, L, n, endpoint=False)
    return np.c_[np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])], L / n


def smooth_closed(p, sigma):
    if sigma <= 0:
        return p.copy()
    return np.c_[gaussian_filter1d(p[:, 0], sigma, mode="wrap"),
                 gaussian_filter1d(p[:, 1], sigma, mode="wrap")]


def turning(p, k):
    a = np.roll(p, k, axis=0) - p
    b = np.roll(p, -k, axis=0) - p
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return 180 - np.degrees(np.arccos(np.clip(cos, -1, 1)))


def find_corners(pd, ds, opt, s, closed=True):
    n = len(pd)
    k = max(2, int(round(opt.extra.get("corner_k", 0.9) * s / ds)))
    k2 = max(1, k // 2)
    if n < 4 * k:
        return []
    tk = turning(pd, k)
    th = turning(pd, k2)
    cand = []
    rng = range(n) if closed else range(k, n - k)
    for i in rng:
        if tk[i] < opt.corner_angle:
            continue
        idx = np.arange(i - k, i + k + 1) % n
        if tk[i] < tk[idx].max() or (tk[idx] == tk[i]).sum() > 1 and i != idx[np.argmax(tk[idx])]:
            continue
        # sharp corners keep most of their turning at half the scale,
        # smooth arcs lose about half of it
        if th[i] / max(tk[i], 1e-6) < 0.72 and tk[i] < opt.extra.get("sure_corner", 100):
            continue
        cand.append(i)
    # suppress corners that are too close
    res = []
    for i in cand:
        if res and (i - res[-1]) < k:
            if tk[i] > tk[res[-1]]:
                res[-1] = i
            continue
        res.append(i)
    if closed and len(res) > 1 and (res[0] + n - res[-1]) < k:
        if tk[res[0]] >= tk[res[-1]]:
            res.pop()
        else:
            res.pop(0)
    return res


def fit_circle(p):
    x, y = p[:, 0], p[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(max(sol[2] + cx * cx + cy * cy, 0))
    dev = np.abs(np.hypot(x - cx, y - cy) - r)
    return cx, cy, r, dev.max()


def find_lines(S, ds, opt, s):
    """Straight runs inside an open polyline. Returns [(i0, i1, c, d)]."""
    n = len(S)
    w = max(3, int(round(0.9 * s / ds)))
    minlen = max(2 * w + 1, int(round(opt.min_line * s / ds)))
    if n < minlen:
        return []
    flat = np.zeros(n, bool)
    for i in range(w, n - w):
        a, b = S[i - w], S[i + w]
        d = unit(b - a)
        q = S[i - w:i + w + 1] - a
        dev = np.abs(q[:, 0] * d[1] - q[:, 1] * d[0])
        flat[i] = dev.max() < 0.045 * s
    runs = []
    i = 0
    while i < n:
        if not flat[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flat[j + 1]:
            j += 1
        runs.append([max(0, i - w), min(n - 1, j + w)])
        i = j + 1
    lines = []
    tol = opt.line_tolerance * s

    def split(a, b):
        """Douglas-Peucker style split of a flat run into truly straight parts."""
        if b - a + 1 < minlen:
            return []
        c, d, dev = line_fit(S[a:b + 1])
        if dev <= tol:
            return [(a, b)]
        nrm = np.array([-d[1], d[0]])
        k = a + int(np.argmax(np.abs((S[a:b + 1] - c) @ nrm)))
        k = min(max(k, a + 1), b - 1)
        return split(a, k) + split(k, b)

    cand = []
    for a, b in runs:
        cand += split(a, b)
    for a, b in cand:
        c, d, dev = line_fit(S[a:b + 1])
        # reject gentle arcs: fit quadratic in line coordinates
        nrm = np.array([-d[1], d[0]])
        u = (S[a:b + 1] - c) @ d
        v = (S[a:b + 1] - c) @ nrm
        L = u.max() - u.min()
        if L > 0:
            qa = np.polyfit(u, v, 2)[0]
            if abs(qa) * L * L / 4 > 0.6 * tol:
                continue
        # grow while the line still fits
        while a > 0 and abs((S[a - 1] - c) @ nrm) < tol:
            a -= 1
        while b < n - 1 and abs((S[b + 1] - c) @ nrm) < tol:
            b += 1
        c, d, dev = line_fit(S[a:b + 1])
        d = snap_dir(d, opt.axis_snap)
        c = S[a:b + 1].mean(0)
        if lines and a <= lines[-1][1]:
            pa, pb, pc, pd = lines[-1]
            if abs(abs(d @ pd) - 1) < 1e-3 and abs((c - pc) @ np.array([-pd[1], pd[0]])) < tol:
                lines[-1] = (pa, b, pc, pd)
                continue
            mid = (a + lines[-1][1]) // 2
            lines[-1] = (pa, mid, pc, pd)
            a = mid
        if (b - a) * ds >= opt.min_line * s * 0.8:
            lines.append((a, b, c, d))
    return lines


def project(p, c, d):
    return c + ((p - c) @ d) * d


# ------------------------------------------------------------- per path ----
def resample_open(pts, ds):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d = np.r_[0, np.cumsum(seg)]
    L = d[-1]
    n = max(4, int(round(L / ds)) + 1)
    t = np.linspace(0, L, n)
    return np.c_[np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])], L / max(n - 1, 1)


def smooth_open(p, sigma):
    if sigma <= 0 or len(p) < 3:
        return p.copy()
    q = np.c_[gaussian_filter1d(p[:, 0], sigma, mode="nearest"),
              gaussian_filter1d(p[:, 1], sigma, mode="nearest")]
    q[0], q[-1] = p[0], p[-1]
    return q


def vectorize_loop(raw, opt, s):
    return fit_path(raw, True, opt, s)


def fit_path(raw, closed, opt, s, circles=True):
    """Fit a dense polyline (closed loop or open stroke) with lines + Beziers.
    Returns ("circle", (cx, cy, r)) | ("path", (start, segs)) | ("open", (start, segs))."""
    ds = 0.2 * s
    if closed:
        P, ds = resample_closed(raw, ds)
        Pf = smooth_closed(P, 0.3 * s / ds)
        Pd = smooth_closed(P, 0.6 * s / ds)
    else:
        P, ds = resample_open(raw, ds)
        Pf = smooth_open(P, 0.3 * s / ds)
        Pd = smooth_open(P, 0.6 * s / ds)
    n = len(P)

    if closed and circles and opt.detect_circles:
        cx, cy, r, dev = fit_circle(Pf)
        rms = np.sqrt(np.mean((np.hypot(Pf[:, 0] - cx, Pf[:, 1] - cy) - r) ** 2))
        if r > 0.8 * s and rms < max(0.12 * s, 0.03 * r) and dev < max(0.4 * s, 0.07 * r):
            return ("circle", (cx, cy, r))

    corners = find_corners(Pd, ds, opt, s, closed)
    closed_smooth = closed and not corners
    if closed:
        if closed_smooth:
            # start at a point of high curvature so a straight side is never cut
            start = int(np.argmax(turning(Pd, max(2, int(round(1.0 * s / ds))))))
            corners = [start]
        order = np.r_[corners[0]:n, 0:corners[0]]
        Pf = Pf[order]
        Praw = P[order]
        cs = [(c - corners[0]) % n for c in corners] + [n]
        Pw = np.vstack([Pf, Pf[:1]])
    else:
        Praw = P
        cs = [0] + [c for c in corners if 2 < c < n - 3] + [n - 1]
        Pw = Pf

    pieces = []
    for a, b in zip(cs[:-1], cs[1:]):
        S = Pw[a:b + 1]
        lines = find_lines(S, ds, opt, s)
        cur = 0
        for (la, lb, c, d) in lines:
            if la - cur >= 3:
                pieces.append(["curve", a + cur, a + la, None])
                cur = la
            pieces.append(["line", a + cur, a + lb, (c, d)])
            cur = lb
        if len(S) - 1 - cur >= 3 or not lines:
            pieces.append(["curve", a + cur, b, None])
        else:
            pieces[-1][2] = b
    corner_set = set(cs[:-1]) if not closed_smooth else set()
    if not closed:
        corner_set |= {0, n - 1}
    m = len(pieces)
    if closed and m == 1 and pieces[0][0] == "line":  # degenerate
        pieces.append(["curve", pieces[0][2], pieces[0][2], None])
        m = 2

    def node_pos(k):
        """Position of the junction at the start of piece k (k == m: open end)."""
        if not closed and k == 0:
            return Pw[0]
        if not closed and k == m:
            return Pw[-1]
        A = pieces[k - 1]
        B = pieces[k % m]
        idx = B[1] % (n if closed else n + 1)
        praw = Pw[idx]
        if A[0] == "line" and B[0] == "line":
            (c1, d1), (c2, d2) = A[3], B[3]
            x = intersect(c1, d1, c2, d2)
            if x is not None and np.linalg.norm(x - praw) < 2.0 * s:
                return x
            return praw
        if A[0] == "line":
            return project(praw, *A[3])
        if B[0] == "line":
            return project(praw, *B[3])
        return Praw[idx] if idx in corner_set else praw

    nodes = [node_pos(k) for k in range(m if closed else m + 1)]
    segs = []
    win = max(2, int(round(1.2 * s / ds)))
    for k in range(m):
        typ, i0, i1, data = pieces[k]
        p0, p3 = nodes[k], nodes[(k + 1) % len(nodes)] if closed else nodes[k + 1]
        if typ == "line":
            segs.append(("L", p3))
            continue
        pts = Pw[i0:i1 + 1].copy()
        pts[0], pts[-1] = p0, p3
        prev = pieces[k - 1] if (closed or k > 0) else None
        nxt = pieces[(k + 1) % m] if (closed or k < m - 1) else None
        t1 = t2 = None
        if prev is not None and prev[0] == "line" and (i0 % n) not in corner_set:
            t1 = prev[3][1] * np.sign((pts[min(win, len(pts) - 1)] - p0) @ prev[3][1] or 1)
        elif closed_smooth and k == 0:
            t1 = snap_dir(unit(Pw[min(win, n)] - Pw[(n - win) % n]), 4.0)
        if nxt is not None and nxt[0] == "line" and (i1 % n) not in corner_set:
            t2 = nxt[3][1] * np.sign((pts[max(0, len(pts) - 1 - win)] - p3) @ nxt[3][1] or 1)
        elif closed_smooth and k == m - 1:
            t2 = -snap_dir(unit(Pw[min(win, n)] - Pw[(n - win) % n]), 4.0)
        if t1 is None:
            t1 = unit(pts[min(win, len(pts) - 1)] - p0)
        if t2 is None:
            t2 = unit(pts[max(0, len(pts) - 1 - win)] - p3)
        curves = fit_curve(pts, p0, p3, t1, t2, opt.tolerance * s, ds)
        if len(curves) >= 2:
            poly = _dp(pts, opt.tolerance * s)
            if len(poly) - 1 <= len(curves) and all(
                    np.linalg.norm(poly[j + 1] - poly[j]) > 1.2 * s for j in range(len(poly) - 1)):
                for q in poly[1:]:
                    segs.append(("L", q))
                continue
        for cv in curves:
            # nearly straight Bezier -> line
            ch = cv[3] - cv[0]
            L = np.linalg.norm(ch)
            if L > 1e-6:
                nrm = np.array([-ch[1], ch[0]]) / L
                if abs((cv[1] - cv[0]) @ nrm) < 0.08 * s and abs((cv[2] - cv[0]) @ nrm) < 0.08 * s:
                    segs.append(("L", cv[3]))
                    continue
            segs.append(("C", cv[1], cv[2], cv[3]))
    segs = merge_lines(nodes[0], segs)
    keep = opt.extra.get("keep_line", 2.6) * s
    segs = simplify(nodes[0], segs, closed, opt.tolerance * s, straight_tol=0.08 * s, keep_line=keep)
    start, segs = fillet(nodes[0], segs, closed, opt.tolerance * s * 1.5, keep,
                         opt.extra.get("fillet_len", 8.0) * s, opt.extra.get("sharp_r", 0.45) * s,
                         opt.extra.get("wide_factor", 1.0))
    start, segs = snap_axis(start, segs, closed, opt.axis_snap)
    return ("path" if closed else "open", (start, segs))


def _dp(pts, tol):
    """Douglas-Peucker; keeps the first and last point."""
    a, b = pts[0], pts[-1]
    if len(pts) < 3:
        return [a, b]
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-9:
        dist = np.linalg.norm(pts - a, axis=1)
    else:
        dist = np.abs((pts[:, 0] - a[0]) * d[1] - (pts[:, 1] - a[1]) * d[0]) / L
    i = int(np.argmax(dist))
    if dist[i] <= tol:
        return [a, b]
    return _dp(pts[:i + 1], tol)[:-1] + _dp(pts[i:], tol)


def merge_lines(start, segs):
    """Merge consecutive collinear line segments."""
    out = []
    cur = start
    for sg in segs:
        if sg[0] == "L" and out and out[-1][0] == "L":
            a = prev_pt
            b = out[-1][1]
            c = sg[1]
            d1, d2 = unit(b - a), unit(c - b)
            if d1 @ d2 > 0.9995:
                out[-1] = ("L", c)
                cur = c
                continue
        prev_pt = cur
        out.append(sg)
        cur = sg[-1]
    return out


# ---------------------------------------------------------------- output ---
def fmt(v, dec):
    s = f"{v:.{dec}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def shapes_to_d(shapes, dec, k=1.0):
    parts = []
    f = lambda v: fmt(v * k, dec)
    for kind, data in shapes:
        if kind == "circle":
            cx, cy, r = data
            # four arcs -> 4 anchors, exactly like a hand-drawn ellipse tool circle
            parts.append(f"M{f(cx - r)} {f(cy)}a{f(r)} {f(r)} 0 1 0 {f(2 * r)} 0a{f(r)} {f(r)} 0 1 0 {f(-2 * r)} 0z")
            continue
        start, segs = data
        closed = kind == "path"
        d = [f"M{f(start[0])} {f(start[1])}"]
        cur = start
        for i, sg in enumerate(segs):
            if sg[0] == "L":
                p = sg[1]
                if closed and i == len(segs) - 1 and np.allclose(p, start, atol=1e-6):
                    break
                if abs(p[1] - cur[1]) * k < 0.5 * 10 ** -dec:
                    d.append(f"H{f(p[0])}")
                elif abs(p[0] - cur[0]) * k < 0.5 * 10 ** -dec:
                    d.append(f"V{f(p[1])}")
                else:
                    d.append(f"L{f(p[0])} {f(p[1])}")
                cur = p
            else:
                _, c1, c2, p = sg
                d.append(f"C{f(c1[0])} {f(c1[1])} {f(c2[0])} {f(c2[1])} {f(p[0])} {f(p[1])}")
                cur = p
        if closed:
            d.append("Z")
        parts.append("".join(d))
    return "".join(parts)


def count_anchors(shapes):
    n = 0
    for kind, data in shapes:
        n += 4 if kind == "circle" else len(data[1]) + (kind == "open")
    return n


def trace(img, opt=None, return_shapes=False):
    """img: numpy image (as returned by cv2.imread / load_image).
    Returns (svg_text, info dict)."""
    opt = opt or Options()
    H, W = img.shape[:2]
    ink, color = to_ink(img)
    # work at a sane resolution
    work = ink
    f = 1.0
    if max(H, W) > 600:
        f = 600.0 / max(H, W)
        work = cv2.resize(ink, (int(round(W * f)), int(round(H * f))), interpolation=cv2.INTER_AREA)
    s = max(work.shape) / BASE
    s = max(s, 0.5)
    if opt.symmetry:
        work = symmetrize(work)
    fill = color if opt.color == "auto" else opt.color
    scale = opt.size / max(H, W) if opt.size > 0 else opt.scale
    k = scale / f
    Wo, Ho = W * scale, H * scale
    # viewBox only (no width/height): browsers then scale the icon to fit the
    # window, like SVGs exported from Illustrator with "responsive" on
    head = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {fmt(Wo, 2)} {fmt(Ho, 2)}">\n'


    if opt.mode in ("stroke", "auto"):
        from .stroke import trace_strokes
        strokes, fills, sw = trace_strokes(work, opt, s)
        use = opt.mode == "stroke"
        if opt.mode == "auto" and strokes:
            # accept the stroke result only if it reproduces the icon well
            use = _stroke_fidelity(work, strokes, fills, sw) >= 0.83
        if opt.stroke_width > 0:
            sw = opt.stroke_width / k
        if use:
            body = ""
            if strokes:
                body += (f'  <path fill="none" stroke="{fill}" stroke-width="{fmt(sw * k, 2)}" '
                         f'stroke-linecap="round" stroke-linejoin="round" '
                         f'd="{shapes_to_d(strokes, opt.decimals, k)}"/>\n')
            if fills:
                body += f'  <path fill="{fill}" fill-rule="evenodd" d="{shapes_to_d(fills, opt.decimals, k)}"/>\n'
            svg = head + body + "</svg>\n"
            allsh = strokes + fills
            info = {"mode": "stroke", "anchors": count_anchors(allsh), "shapes": len(allsh),
                    "color": fill, "stroke_width": round(float(sw * k), 3)}
            if return_shapes:
                return svg, info, [(kd, _scale_shape(kd, dt, k)) for kd, dt in allsh]
            return svg, info

    # outline tuning (compared against Vector Magic): only real corners stay
    # corners, straight pieces must be long, gentle bends become one smooth
    # curve, and narrow tips blurred by the raster are rebuilt sharp
    import dataclasses
    ex = dict(opt.extra)
    ex.setdefault("sharp_r", 1.3)
    ex.setdefault("keep_line", 2.2)
    ex.setdefault("wide_factor", 0.4)
    oopt = dataclasses.replace(opt, corner_angle=max(opt.corner_angle, 55), min_line=max(opt.min_line, 3.0),
                               tolerance=opt.tolerance * 1.33, extra=ex)
    shapes = []
    for c in extract_contours(work, opt, s):
        try:
            shapes.append(fit_path(c, True, oopt, s))
        except Exception:  # never lose a shape: fall back to a polygon
            pts = c[:: max(1, len(c) // 64)]
            shapes.append(("path", (pts[0], [("L", p) for p in pts[1:]] + [("L", pts[0])])))
    d = shapes_to_d(shapes, opt.decimals, k)
    svg = head + f'  <path fill="{fill}" fill-rule="evenodd" d="{d}"/>\n</svg>\n'
    info = {"mode": "outline", "anchors": count_anchors(shapes), "shapes": len(shapes), "color": fill}
    if return_shapes:
        return svg, info, [(kd, _scale_shape(kd, dt, k)) for kd, dt in shapes]
    return svg, info


def _render_shapes(shape_list, size, z, stroke_w=None):
    """Tiny rasteriser (OpenCV) used to score a result against the input."""
    from .bezier import bez
    img = np.zeros((size[0] * z, size[1] * z), np.uint8)
    sh = 4
    P = lambda p: (int(round(p[0] * z * 16)), int(round(p[1] * z * 16)))
    polys = []
    for kind, data in shape_list:
        if kind == "circle":
            cx, cy, r = data
            t = np.linspace(0, 2 * np.pi, 64)
            polys.append((np.c_[cx + r * np.cos(t), cy + r * np.sin(t)], True))
            continue
        cur, segs = data
        pts = [np.asarray(cur)]
        for sg in segs:
            if sg[0] == "L":
                pts.append(np.asarray(sg[1]))
            else:
                pts.extend(bez(np.array([cur, sg[1], sg[2], sg[3]]), np.linspace(0, 1, 16))[1:])
            cur = sg[-1]
        polys.append((np.array(pts), kind != "open"))
    for pts, closed in polys:
        arr = np.array([P(p) for p in pts], np.int32)
        if stroke_w is not None:
            cv2.polylines(img, [arr], closed, 255, max(1, int(round(stroke_w * z))), cv2.LINE_8, sh)
    if stroke_w is None:
        img2 = np.zeros_like(img)
        cv2.fillPoly(img2, [np.array([P(p) for p in pts], np.int32) for pts, _ in polys], 255, cv2.LINE_8, sh)
        return img2
    return img


def _stroke_fidelity(work, strokes, fills, sw):
    z = 4
    h, w = work.shape
    ref = cv2.resize(work, (w * z, h * z), interpolation=cv2.INTER_CUBIC) > 0.5
    got = _render_shapes(strokes, (h, w), z, sw) > 0
    if fills:
        got |= _render_shapes(fills, (h, w), z) > 0
    return (ref & got).sum() / max((ref | got).sum(), 1)


def _scale_shape(kind, data, k):
    if kind == "circle":
        return tuple(v * k for v in data)
    start, segs = data
    return (np.asarray(start) * k, [(sg[0],) + tuple(np.asarray(p) * k for p in sg[1:]) for sg in segs])
