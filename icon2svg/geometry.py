"""Geometric primitive recognition ("redraw it like a designer would").

Given a traced closed boundary or an open stroke, decide whether it is really a
rectangle, rounded rectangle, circle, ellipse, polygon, straight line or
polyline, and if so return the exact primitive. Everything here works in
work-pixel units; `s` is the scale factor relative to a 130 px icon and `w` is
the stroke width.
"""
import numpy as np

from .bezier import arc_to_cubics, unit

KAPPA = 0.5522847498


# ------------------------------------------------------------ helpers ----
def _dp(pts, tol):
    a, b = pts[0], pts[-1]
    if len(pts) < 3:
        return [0, len(pts) - 1]
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-9:
        dist = np.linalg.norm(pts - a, axis=1)
    else:
        dist = np.abs((pts[:, 0] - a[0]) * d[1] - (pts[:, 1] - a[1]) * d[0]) / L
    i = int(np.argmax(dist))
    if dist[i] <= tol:
        return [0, len(pts) - 1]
    left = _dp(pts[:i + 1], tol)
    right = _dp(pts[i:], tol)
    return left[:-1] + [i + k for k in right]


def dp_closed(P, tol):
    """Douglas-Peucker for a closed loop; returns vertex indices."""
    n = len(P)
    c = P.mean(0)
    i0 = int(np.argmax(np.linalg.norm(P - c, axis=1)))
    i1 = int(np.argmax(np.linalg.norm(P - P[i0], axis=1)))
    a, b = sorted((i0, i1))
    A = _dp(P[a:b + 1], tol)
    Bq = np.vstack([P[b:], P[:a + 1]])
    Bi = _dp(Bq, tol)
    idx = [a + k for k in A[:-1]] + [(b + k) % n for k in Bi[:-1]]
    return idx


def dist_to_polyline(P, V, closed=True):
    """Distance of each point of P to the polyline through vertices V."""
    segs = list(zip(V, V[1:] + ([V[0]] if closed else [])))
    best = np.full(len(P), np.inf)
    for a, b in segs:
        a, b = np.asarray(a), np.asarray(b)
        d = b - a
        L2 = d @ d
        t = np.clip(((P - a) @ d) / L2, 0, 1) if L2 > 0 else np.zeros(len(P))
        q = a + t[:, None] * d
        best = np.minimum(best, np.linalg.norm(P - q, axis=1))
    return best


def _ellipse_pts(cx, cy, rx, ry, n=64):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[cx + rx * np.cos(t), cy + ry * np.sin(t)]


def _nearest_dist(P, Q):
    d = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
    return d.min(1)


# ------------------------------------------------------- closed shapes ----
def _ellipse_ls(Q):
    m = Q.mean(0)
    x, y = Q[:, 0] - m[0], Q[:, 1] - m[1]
    A = np.c_[x * x, y * y, x, y]
    sol, *_ = np.linalg.lstsq(A, np.ones_like(x), rcond=None)
    a, b, c, d = sol
    if a <= 0 or b <= 0:
        return None
    cx, cy = -c / (2 * a), -d / (2 * b)
    k = 1 + a * cx * cx + b * cy * cy
    if k <= 0:
        return None
    return cx + m[0], cy + m[1], np.sqrt(k / a), np.sqrt(k / b)


def fit_circle_ellipse(P, s, w, robust=True, min_cover=0.85):
    """Axis-aligned ellipse (or circle) through a closed boundary.
    Robust: parts of the boundary that belong to something touching the
    ellipse (a notch, a joining line) are ignored if most of it is on it.
    Returns (shape, inlier_rms) or None."""
    inl = np.ones(len(P), bool)
    fit = None
    for it in range(4 if robust else 1):
        fit = _ellipse_ls(P[inl])
        if fit is None:
            return None
        cx, cy, rx, ry = fit
        E = _ellipse_pts(cx, cy, rx, ry, 180)
        dev = _nearest_dist(P, E)
        thr = 0.4 * s + 0.03 * max(rx, ry)
        new = dev < thr
        if new.sum() < 12 or (new == inl).all():
            inl = new if new.sum() >= 12 else inl
            break
        inl = new
    cx, cy, rx, ry = fit
    if min(rx, ry) < 1.2 * s:
        return None
    ratio = max(rx, ry) / min(rx, ry)
    if ratio > 3:
        return None
    if ratio < 1.1:
        rx = ry = (rx + ry) / 2
    r = max(rx, ry)
    E = _ellipse_pts(cx, cy, rx, ry, 180)
    dev = _nearest_dist(P, E)
    inl = dev < 0.4 * s + 0.03 * r
    frac = inl.mean()
    # how much of the ellipse is actually drawn
    dE = _nearest_dist(E, P[inl]) if inl.sum() else np.full(len(E), 1e9)
    cover = (dE < 0.6 * s + 0.05 * r).mean()
    rms = np.sqrt(np.mean(dev[inl] ** 2)) if inl.any() else 1e9
    if frac < (0.85 if robust else 0.98) or cover < min_cover or rms > 0.22 * s + 0.02 * r:
        return None
    # tolerated outliers must be notches *into* the ellipse; anything that
    # sticks out means the region is not an ellipse
    outside = ((P[:, 0] - cx) / rx) ** 2 + ((P[:, 1] - cy) / ry) ** 2 > 1
    if (outside & ~inl).mean() > 0.01:
        return None
    shape = ("circle", (cx, cy, rx)) if rx == ry else ellipse_shape(cx, cy, rx, ry)
    return shape, rms


def fit_rect(P, s, w):
    """Axis aligned rectangle / rounded rectangle."""
    V = dp_closed(P, 0.55 * s + 0.2 * w)
    if not (4 <= len(V) <= 8):
        return None
    x0, y0 = np.percentile(P[:, 0], 1), np.percentile(P[:, 1], 1)
    x1, y1 = np.percentile(P[:, 0], 99), np.percentile(P[:, 1], 99)
    W, H = x1 - x0, y1 - y0
    if min(W, H) < 1.2 * s:
        return None
    # sides: points close to each bbox edge
    band = 0.6 * s + 0.15 * w
    sides = []
    for m in (np.abs(P[:, 0] - x0) < band, np.abs(P[:, 0] - x1) < band,
              np.abs(P[:, 1] - y0) < band, np.abs(P[:, 1] - y1) < band):
        sides.append(m)
    cover = np.zeros(len(P), bool)
    for m in sides:
        cover |= m
    # refine edges with medians of their side points
    xs0 = np.median(P[sides[0], 0]) if sides[0].sum() > 3 else x0
    xs1 = np.median(P[sides[1], 0]) if sides[1].sum() > 3 else x1
    ys0 = np.median(P[sides[2], 1]) if sides[2].sum() > 3 else y0
    ys1 = np.median(P[sides[3], 1]) if sides[3].sum() > 3 else y1
    # corner radius from how far the boundary stays away from the bbox corners
    cr = []
    for cx, cy in ((xs0, ys0), (xs1, ys0), (xs1, ys1), (xs0, ys1)):
        d = np.min(np.hypot(P[:, 0] - cx, P[:, 1] - cy))
        cr.append(d / (np.sqrt(2) - 1))
    r = float(np.median(cr))
    if r < 0.55 * w:
        r = 0.0
    r = min(r, min(xs1 - xs0, ys1 - ys0) / 2)
    shape = rect_shape(xs0, ys0, xs1, ys1, r)
    Q = shape_points(shape)
    dev = _nearest_dist(P, Q)
    dev2 = _nearest_dist(Q, P)
    tol = 0.5 * s + 0.12 * w
    if dev.max() < 2 * tol and np.sqrt(np.mean(dev ** 2)) < 0.6 * tol and dev2.max() < 2 * tol:
        return shape
    return None


def fit_polygon(P, s, w, closed=True, max_vertices=8):
    """Sharp polygon / polyline with few vertices and reasonably long edges."""
    tol = 0.45 * s + 0.1 * w
    if closed:
        idx = dp_closed(P, tol)
    else:
        idx = _dp(P, tol)
    V = [P[i] for i in idx]
    nseg = len(V) if closed else len(V) - 1
    if nseg > max_vertices or nseg < (3 if closed else 1):
        return None
    V = refine_vertices(P, idx, closed)
    segs = list(zip(V, V[1:] + ([V[0]] if closed else [])))
    if min(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in segs) < 1.1 * s:
        return None
    dev = dist_to_polyline(P, V, closed)
    if dev.max() > 1.6 * tol or np.sqrt(np.mean(dev ** 2)) > 0.55 * tol:
        return None
    return V


def refine_vertices(P, idx, closed):
    """Move polygon vertices to the intersections of least-squares edge lines
    (a crisp corner instead of the rounded tip the raster gives)."""
    n = len(P)
    runs = []
    k = len(idx)
    for j in range(k if closed else k - 1):
        a, b = idx[j], idx[(j + 1) % k]
        seg = P[a:b + 1] if b > a else np.vstack([P[a:], P[:b + 1]])
        m = len(seg)
        core = seg[m // 5: m - m // 5] if m > 10 else seg
        c = core.mean(0)
        _, _, vt = np.linalg.svd(core - c)
        runs.append((c, vt[0]))
    V = []
    for j in range(k):
        if not closed and (j == 0 or j == k - 1):
            V.append(P[idx[j]].copy())
            continue
        (c1, d1), (c2, d2) = runs[j - 1], runs[j % len(runs)]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        p = P[idx[j]]
        if abs(den) > 0.05:
            t = ((c2[0] - c1[0]) * d2[1] - (c2[1] - c1[1]) * d2[0]) / den
            x = c1 + t * d1
            if np.linalg.norm(x - p) < 3.0:
                p = x
        V.append(np.asarray(p, float))
    return V


# ---------------------------------------------------------- open strokes ----
def fit_straight(P, s, w):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    d = vt[0]
    nrm = np.array([-d[1], d[0]])
    dev = np.abs((P - c) @ nrm)
    L = np.ptp((P - c) @ d)
    if dev.max() < 0.45 * s + 0.1 * w and L > 0.8 * s:
        return [P[0], P[-1]]
    return None


# ---------------------------------------------------------- shape output ----
def rect_shape(x0, y0, x1, y1, r):
    if r <= 0:
        start = np.array([x0, y0])
        segs = [("L", np.array([x1, y0])), ("L", np.array([x1, y1])),
                ("L", np.array([x0, y1])), ("L", np.array([x0, y0]))]
        return ("path", (start, segs))
    segs = []
    start = np.array([x0 + r, y0])
    segs.append(("L", np.array([x1 - r, y0])))
    segs += arc_to_cubics(np.array([x1 - r, y0 + r]), r, -np.pi / 2, 0)
    segs.append(("L", np.array([x1, y1 - r])))
    segs += arc_to_cubics(np.array([x1 - r, y1 - r]), r, 0, np.pi / 2)
    segs.append(("L", np.array([x0 + r, y1])))
    segs += arc_to_cubics(np.array([x0 + r, y1 - r]), r, np.pi / 2, np.pi)
    segs.append(("L", np.array([x0, y0 + r])))
    segs += arc_to_cubics(np.array([x0 + r, y0 + r]), r, np.pi, 1.5 * np.pi)
    segs = [sg for k, sg in enumerate(segs) if not (sg[0] == "L" and k > 0 and
                                                   np.allclose(sg[1], segs[k - 1][-1]))]
    return ("path", (start, segs))


def ellipse_shape(cx, cy, rx, ry):
    start = np.array([cx + rx, cy])
    pts = [np.array([cx, cy + ry]), np.array([cx - rx, cy]), np.array([cx, cy - ry]), start]
    segs = []
    cur = start
    for p in pts:
        # quarter ellipse between axis points
        t0 = cur - np.array([cx, cy])
        t1 = p - np.array([cx, cy])
        c1 = cur + KAPPA * t1
        c2 = p + KAPPA * t0
        segs.append(("C", c1, c2, p))
        cur = p
    return ("path", (start, segs))


def polygon_shape(V, closed):
    V = [np.asarray(v, float) for v in V]
    segs = [("L", v) for v in V[1:]]
    if closed:
        segs.append(("L", V[0]))
    return ("path" if closed else "open", (V[0], segs))


def shape_points(shape, n=24):
    from .bezier import bez
    kind, data = shape
    if kind == "circle":
        cx, cy, r = data
        return _ellipse_pts(cx, cy, r, r, 96)
    start, segs = data
    out = [np.asarray(start)]
    cur = np.asarray(start)
    for sg in segs:
        if sg[0] == "L":
            p = np.asarray(sg[1])
            L = np.linalg.norm(p - cur)
            k = max(2, int(L * 2))
            out.extend(np.linspace(cur, p, k)[1:])
        else:
            out.extend(bez(np.array([cur, sg[1], sg[2], sg[3]]), np.linspace(0, 1, n))[1:])
        cur = np.asarray(sg[-1])
    return np.array(out)


def _err(P, shape):
    Q = shape_points(shape)
    return np.sqrt(np.mean(_nearest_dist(P, Q) ** 2))


def primitive_closed(P, s, w, loose=1.0):
    """Best exact primitive for a closed boundary, or None.
    Occam: the candidate with the fewest anchors that fits well enough wins."""
    cands = []  # (shape, error)
    size = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    if size < 3.5 * w:
        # tiny regions (a collar flap, the gap in a tie knot): the fixed
        # tolerances would let almost any shape "fit"; only allow the simple
        # polygons such details really are, else leave it to the curve fitter
        V = fit_polygon(P, s, w, True, max_vertices=4)
        return polygon_shape(V, True) if V is not None else None
    e = fit_circle_ellipse(P, s, w)
    if e is not None:
        cands.append(e)
    r = fit_rect(P, s, w)
    if r is not None:
        cands.append((r, _err(P, r)))
    rp = fit_rounded_polygon(P, s, w)
    if rp is not None:
        segs = rp[1][1]
        n_round = sum(1 for sg in segs if sg[0] == "C")
        n_edge = sum(1 for sg in segs if sg[0] == "L")
        # mostly-rounded many-sided "polygons" are really smooth curves
        if n_edge <= 6 or (n_round <= n_edge / 2 and n_edge <= 12):
            cands.append((rp, _err(P, rp)))
    V = fit_polygon(P, s, w, True)
    if V is not None:
        pg = polygon_shape(V, True)
        cands.append((pg, _err(P, pg)))
    td = fit_teardrop(P, s, w)
    if td is not None:
        cands.append(td)
    if not cands:
        return None

    def anchors(sh):
        return 4 if sh[0] == "circle" else len(sh[1][1])
    good_tol = 0.3 * s + 0.07 * w
    def cost(sh):
        n = anchors(sh)
        if sh[0] == "path" and all(sg[0] == "L" for sg in sh[1][1]) and n > 4:
            # sharp polygons with short diagonal "chamfer" edges are usually
            # rounded corners that the raster lost; prefer the rounded shape
            pts = [np.asarray(sh[1][0])] + [np.asarray(sg[1]) for sg in sh[1][1]]
            lens = [np.linalg.norm(pts[k + 1] - pts[k]) for k in range(len(pts) - 1)]
            short = sum(1 for L in lens if L < 3.0 * s + 0.5 * w)
            n += 2 * short
        return n
    good = [(cost(sh), e, sh) for sh, e in cands if e <= good_tol]
    if good:
        good.sort(key=lambda t: (t[0], t[1]))
        return good[0][2]
    best = min(cands, key=lambda t: t[1] * (1 + 0.05 * anchors(t[0])))
    return best[0] if best[1] <= 1.5 * good_tol else None


def primitive_open(P, s, w):
    L = fit_straight(P, s, w)
    if L is not None:
        return polygon_shape(L, False)
    return None


# ------------------------------------------------------ rounded polygon ----
def _edge_line(P, a, b):
    n = len(P)
    seg = P[a:b + 1] if b >= a else np.vstack([P[a:], P[:b + 1]])
    m = len(seg)
    core = seg[m // 4: m - m // 4] if m > 12 else seg
    c = core.mean(0)
    _, _, vt = np.linalg.svd(core - c)
    return c, vt[0], m


def _isect(c1, d1, c2, d2):
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    t = ((c2[0] - c1[0]) * d2[1] - (c2[1] - c1[1]) * d2[0]) / den
    return c1 + t * d1


def fit_rounded_polygon(P, s, w, max_vertices=18, snap_deg=4.0):
    """Straight edges joined by sharp or individually filleted corners."""
    from .bezier import snap_dir
    idx = dp_closed(P, 0.9 * s + 0.15 * w)
    if len(idx) < 3:
        return None
    changed = True
    lines = None
    while changed and len(idx) >= 3:
        changed = False
        k = len(idx)
        lines = [_edge_line(P, idx[j], idx[(j + 1) % k]) for j in range(k)]
        lens = [np.linalg.norm(P[idx[(j + 1) % k]] - P[idx[j]]) for j in range(k)]
        # 1) merge nearly collinear neighbours
        for j in range(k):
            d1, d2 = lines[j][1], lines[(j + 1) % k][1]
            if abs(d1 @ d2) > np.cos(np.radians(9)):
                idx.pop((j + 1) % k)
                changed = True
                break
        if changed:
            continue
        # 2) short edges between two long ones are fillets/chamfers of a corner
        order = np.argsort(lens)
        for j in order:
            if lens[j] > 2.6 * s + 0.6 * w or k <= 3:
                break
            prev_, next_ = lines[(j - 1) % k], lines[(j + 1) % k]
            x = _isect(prev_[0], prev_[1], next_[0], next_[1])
            if x is None:
                continue
            if np.min(np.linalg.norm(P - x, axis=1)) < 3.5 * s:
                idx.pop((j + 1) % k)
                changed = True
                break
    k = len(idx)
    if not (3 <= k <= max_vertices):
        return None
    lines = [_edge_line(P, idx[j], idx[(j + 1) % k]) for j in range(k)]
    lines = [(c, snap_dir(d, snap_deg), m) for c, d, m in lines]
    V = []
    for j in range(k):
        x = _isect(lines[j - 1][0], lines[j - 1][1], lines[j][0], lines[j][1])
        if x is None:
            return None
        V.append(x)
    # fillet radius per corner from how far the data stays from the vertex
    R = []
    for j in range(k):
        a, b, c = V[j - 1], V[j], V[(j + 1) % k]
        u1, u2 = unit(a - b), unit(c - b)
        phi = np.arccos(np.clip(u1 @ u2, -1, 1))  # interior angle
        if phi < np.radians(15) or phi > np.radians(172):
            return None
        d = np.min(np.linalg.norm(P - b, axis=1))
        r = d / (1 / np.sin(phi / 2) - 1)
        tmax = 0.5 * min(np.linalg.norm(a - b), np.linalg.norm(c - b))
        r = min(r, tmax * np.tan(phi / 2))
        R.append(r if r > 0.5 * w else 0.0)
    shape = rounded_polygon_shape(V, R)
    Q = shape_points(shape)
    dev = _nearest_dist(P, Q)
    dev2 = _nearest_dist(Q, P)
    tol = 0.5 * s + 0.12 * w
    if dev.max() < 2 * tol and np.sqrt(np.mean(dev ** 2)) < 0.6 * tol and dev2.max() < 2 * tol:
        return shape
    return None


def rounded_polygon_shape(V, R):
    k = len(V)
    pts = []  # per corner: (t_in, t_out, handle_in, handle_out) or sharp vertex
    for j in range(k):
        a, b, c = V[j - 1], V[j], V[(j + 1) % k]
        r = R[j]
        if r <= 0:
            pts.append((b, b, None, None))
            continue
        u1, u2 = unit(a - b), unit(c - b)
        phi = np.arccos(np.clip(u1 @ u2, -1, 1))
        t = r / np.tan(phi / 2)
        p1, p2 = b + u1 * t, b + u2 * t
        theta = np.pi - phi  # turning angle of the arc
        hl = 4 / 3 * np.tan(theta / 4) * r
        pts.append((p1, p2, p1 - u1 * hl, p2 - u2 * hl))
    start = pts[0][1]
    segs = []
    for j in range(1, k + 1):
        p_in, p_out, h1, h2 = pts[j % k]
        if np.linalg.norm(p_in - (segs[-1][-1] if segs else start)) > 1e-6:
            segs.append(("L", p_in))
        if h1 is not None:
            segs.append(("C", h1, h2, p_out))
    return ("path", (start, segs))


# ------------------------------------------------------------ teardrop ----
def _circle_ls(Q):
    m = Q.mean(0)
    q = Q - m
    M = np.c_[2 * q, np.ones(len(q))]
    sol, *_ = np.linalg.lstsq(M, (q ** 2).sum(1), rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    return np.array([cx, cy]) + m, r


def teardrop_shape(C, r, T):
    """Circle C,r with two straight tangents meeting at the tip T (a map pin)."""
    v = T - C
    d = np.linalg.norm(v)
    phi = np.arctan2(v[1], v[0])
    a = np.arccos(np.clip(r / d, -1, 1))
    t1, t2 = phi + a, phi - a + 2 * np.pi  # go round the far side of the circle
    P1 = C + r * np.array([np.cos(t1), np.sin(t1)])
    segs = [("L", P1)]
    segs += arc_to_cubics(C, r, t1, t2)
    segs.append(("L", T.copy()))
    return ("path", (T.copy(), segs))


def fit_teardrop(P, s, w):
    best = None
    x0, y0 = P.min(0)
    x1, y1 = P.max(0)
    W, H = x1 - x0, y1 - y0
    # the round end may point up, down, left or right
    for sel in (P[:, 1] < y0 + 0.8 * W, P[:, 1] > y1 - 0.8 * W,
                P[:, 0] < x0 + 0.8 * H, P[:, 0] > x1 - 0.8 * H):
        if sel.sum() < 12:
            continue
        C, r = _circle_ls(P[sel])
        for _ in range(3):
            dev = np.abs(np.linalg.norm(P - C, axis=1) - r)
            inl = dev < 0.35 * s + 0.03 * r
            if inl.sum() < 12:
                break
            C, r = _circle_ls(P[inl])
        if r < 2 * s:
            continue
        dist = np.linalg.norm(P - C, axis=1)
        T = P[int(np.argmax(dist))].copy()
        d = dist.max()
        if not (1.3 * r < d < 3.5 * r):
            continue
        shape = teardrop_shape(C, r, T)
        Q = shape_points(shape)
        # the tip itself usually sits inside a junction (it rests on the base):
        # judge the fit away from it
        far_P = np.linalg.norm(P - T, axis=1) > 0.45 * (d - r) + 1.5 * s
        far_Q = np.linalg.norm(Q - T, axis=1) > 0.45 * (d - r) + 1.5 * s
        dev = _nearest_dist(P[far_P], Q)
        dev2 = _nearest_dist(Q[far_Q], P)
        rms = np.sqrt(np.mean(dev ** 2))
        if rms < 0.3 * s + 0.07 * w and dev2.max() < 0.9 * s + 0.2 * w:
            if best is None or rms < best[0]:
                best = (rms, shape)
    return None if best is None else (best[1], best[0])
