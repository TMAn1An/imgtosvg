"""Render -> compare -> fix.

After the stroke engine has built clean shapes, this module measures, all along
every shape, where the centre of the ink really is (the middle between the two
ink edges, measured across the line) and moves the anchors and handle lengths
so the drawing sits exactly on it.  Topology, anchor count, straight lines,
horizontal/vertical lines and handle directions are kept, so the result stays
as clean as before; it only becomes more accurate.  Samples near junctions and
corners (where the ink is wider than one line) are ignored.

refine_width() then picks the line weight that best reproduces the ink.
"""
import numpy as np
from scipy.ndimage import map_coordinates

STEP = 0.05


def _bez(c, t):
    t = t[:, None]
    mt = 1 - t
    return mt ** 3 * c[0] + 3 * mt ** 2 * t * c[1] + 3 * mt * t ** 2 * c[2] + t ** 3 * c[3]


def _dbez(c, t):
    t = t[:, None]
    mt = 1 - t
    return 3 * mt ** 2 * (c[1] - c[0]) + 6 * mt * t * (c[2] - c[1]) + 3 * t ** 2 * (c[3] - c[2])


def _scan(ink, P, N, w):
    """Signed offset from P (along N) to the middle of the ink run, or nan
    where the run is not one clean line of width ~w."""
    ts = np.arange(-1.6 * w, 1.6 * w + STEP / 2, STEP)
    X = P[:, None, :] + ts[None, :, None] * N[:, None, :]
    v = map_coordinates(ink, [X[..., 1] - 0.5, X[..., 0] - 0.5], order=1, mode="constant")
    off = np.full(len(P), np.nan)
    c = len(ts) // 2
    reach = int(0.5 * w / STEP)
    for i in range(len(P)):
        r = v[i] > 0.5
        if not r[c - reach:c + reach + 1].any():
            continue
        idx = np.flatnonzero(r)
        j = idx[np.argmin(np.abs(idx - c))]
        a = j
        while a > 0 and r[a - 1]:
            a -= 1
        b = j
        while b < len(r) - 1 and r[b + 1]:
            b += 1
        if a == 0 or b == len(r) - 1:
            continue
        ea = ts[a - 1] + (0.5 - v[i, a - 1]) / (v[i, a] - v[i, a - 1] + 1e-9) * STEP
        eb = ts[b] + (v[i, b] - 0.5) / (v[i, b] - v[i, b + 1] + 1e-9) * STEP
        if abs((eb - ea) - w) < 0.3 * w:
            off[i] = (ea + eb) / 2
    return off


def _segments(shape):
    cur, segs = shape[1]
    out = []
    for sg in segs:
        p = np.asarray(sg[-1], float)
        if sg[0] == "L":
            out.append(("L", np.array([cur, p], float)))
        else:
            out.append(("C", np.array([cur, sg[1], sg[2], p], float)))
        cur = p
    return out


def _step(shapes, ink, w, lam, n_per):
    key = lambda p: (round(float(p[0]), 3), round(float(p[1]), 3))
    anchors = {}
    for kind, data in shapes:
        if kind == "circle":
            continue
        for _, c in _segments((kind, data)):
            for p in (c[0], c[-1]):
                anchors.setdefault(key(p), len(anchors))
    na = len(anchors)
    nvar = 2 * na
    handle_var, circle_var = {}, {}
    for si, (kind, data) in enumerate(shapes):
        if kind == "circle":
            circle_var[si] = nvar
            nvar += 3
            continue
        for gi, (tp, c) in enumerate(_segments((kind, data))):
            if tp == "C":
                handle_var[(si, gi)] = nvar
                nvar += 2
    rows, rhs = [], []

    def add(coef, val):
        rows.append(coef)
        rhs.append(val)

    for si, (kind, data) in enumerate(shapes):
        if kind == "circle":
            cx, cy, r = data
            m = max(16, int(2 * np.pi * r / (0.8 * w)))
            t = np.linspace(0, 2 * np.pi, m, endpoint=False)
            N = np.c_[np.cos(t), np.sin(t)]
            off = _scan(ink, np.array([cx, cy]) + r * N, N, w)
            v = circle_var[si]
            for n, o in zip(N, off):
                if np.isfinite(o):
                    row = np.zeros(nvar)
                    row[v:v + 2] = n
                    row[v + 2] = 1
                    add(row, o)
            continue
        for gi, (tp, c) in enumerate(_segments((kind, data))):
            L = np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
            if L < 2.0 * w:
                # rounded corners and other small pieces follow their neighbours;
                # the ink around them is not a single clean line to measure
                continue
            m = int(np.clip(L / (0.8 * w), 3, n_per))
            t = np.linspace(0.06, 0.94, m)
            P = _bez(c, t) if tp == "C" else c[0] + t[:, None] * (c[1] - c[0])
            D = _dbez(c, t) if tp == "C" else np.repeat((c[1] - c[0])[None], m, 0)
            D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-12
            N = np.c_[-D[:, 1], D[:, 0]]
            off = _scan(ink, P, N, w)
            i0, i1 = anchors[key(c[0])], anchors[key(c[-1])]
            if tp == "C":
                h = handle_var[(si, gi)]
                u0 = c[1] - c[0]
                u3 = c[2] - c[3]
                u0 = u0 / (np.linalg.norm(u0) + 1e-12)
                u3 = u3 / (np.linalg.norm(u3) + 1e-12)
            for k in range(m):
                if not np.isfinite(off[k]):
                    continue
                tk, n = t[k], N[k]
                row = np.zeros(nvar)
                if tp == "C":
                    mt = 1 - tk
                    a0, a1 = mt ** 3 + 3 * mt ** 2 * tk, 3 * mt * tk ** 2 + tk ** 3
                    row[h] = 3 * mt ** 2 * tk * (n @ u0)
                    row[h + 1] = 3 * mt * tk ** 2 * (n @ u3)
                else:
                    a0, a1 = 1 - tk, tk
                row[2 * i0:2 * i0 + 2] += a0 * n
                row[2 * i1:2 * i1 + 2] += a1 * n
                add(row, off[k])
            if tp == "L":
                # keep horizontal / vertical lines exactly so
                for ax in (0, 1):
                    if abs(c[1][ax] - c[0][ax]) < 1e-6 and i0 != i1:
                        row = np.zeros(nvar)
                        row[2 * i0 + ax], row[2 * i1 + ax] = 10.0, -10.0
                        add(row, 0.0)
    if not rows:
        return shapes, 0
    A = np.vstack(rows + [np.sqrt(lam) * np.eye(nvar)])
    b = np.r_[np.array(rhs), np.zeros(nvar)]
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    x = np.clip(x, -0.35 * w, 0.35 * w)  # damped: never jump across a line

    def mv(p):
        i = anchors[key(p)]
        return np.asarray(p, float) + x[2 * i:2 * i + 2]

    out = []
    for si, (kind, data) in enumerate(shapes):
        if kind == "circle":
            v = circle_var[si]
            cx, cy, r = data
            out.append(("circle", (cx + x[v], cy + x[v + 1], max(r + x[v + 2], 0.1))))
            continue
        start = mv(data[0])
        segs = []
        for gi, (tp, c) in enumerate(_segments((kind, data))):
            p0, p3 = mv(c[0]), mv(c[-1])
            if tp == "L":
                segs.append(("L", p3))
                continue
            h = handle_var[(si, gi)]
            v0, v3 = c[1] - c[0], c[2] - c[3]
            l0, l3 = np.linalg.norm(v0), np.linalg.norm(v3)
            n0 = max(l0 + x[h], 0.15 * l0) if l0 > 1e-9 else 0.0
            n3 = max(l3 + x[h + 1], 0.15 * l3) if l3 > 1e-9 else 0.0
            c1 = p0 + (v0 / l0 * n0 if l0 > 1e-9 else 0)
            c2 = p3 + (v3 / l3 * n3 if l3 > 1e-9 else 0)
            segs.append(("C", c1, c2, p3))
        out.append((kind, (start, segs)))
    return out, len(rhs)


def refine_shapes(shapes, ink, w, iters=3, lam=0.4, n_per=24):
    """Move `shapes` (in `ink` pixel units) onto the centre of the ink."""
    cur = shapes
    for _ in range(iters):
        cur, n = _step(cur, ink, w, lam, n_per)
        if n == 0:
            break
    return cur


def iou(svg, ref):
    from .ai import render_svg
    h, w = ref.shape
    b = (1 - render_svg(svg, w, h)) > 0.5
    a = ref > 0.5
    return (a & b).sum() / max((a | b).sum(), 1)


def refine_width(strokes, fills, ink, w, dec=2, zoom=4):
    """The line weight (in `ink` pixels) whose rendering best matches the ink."""
    try:
        import cv2
        from .tracer import shapes_to_d
        H, W = ink.shape
        ref = cv2.resize(ink, (W * zoom, H * zoom), interpolation=cv2.INTER_CUBIC)
        d = shapes_to_d(strokes, dec + 1)
        fd = f'<path d="{shapes_to_d(fills, dec + 1)}"/>' if fills else ""

        def svg(sw):
            return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">{fd}'
                    f'<path fill="none" stroke="#000" stroke-width="{sw:.4f}" stroke-linecap="round" '
                    f'stroke-linejoin="round" d="{d}"/></svg>')
        best = max((iou(svg(w * f), ref), -abs(f - 1), w * f) for f in np.arange(0.9, 1.201, 0.025))
        return float(best[2])
    except Exception:  # no renderer available: keep the measured weight
        return w
