"""No line on top of another line.

A designer never draws a stroke twice: where two shapes would run along the
same line (a door's bottom edge on the ground line, the end of an arrow shaft
along a ring), the shorter shape simply stops at the other one; where a shape
would pass behind another one (a head behind a head) it is cut where the ink
shows no line; and a line that meets another ends exactly on its centre line.

Pieces of a shape are cut out when they
  * lie on another shape (within ~half a line width) for a stretch, or
  * run where the image has no ink.
Shapes are processed shortest first, so the longer, more important shape
keeps the shared piece.  Free ends created by a cut snap onto the line they
meet.  Every change is checked against the ink.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from .geometry import shape_points


def _ctrl(cur, sg):
    if sg[0] == "L":
        p = np.asarray(sg[1], float)
        return np.array([cur, cur + (p - cur) / 3, cur + 2 * (p - cur) / 3, p]), True
    return np.array([cur, sg[1], sg[2], sg[3]], float), False


def _split(c, t):
    """de Casteljau: (left, right) cubic control polygons at t."""
    p01 = c[0] + t * (c[1] - c[0])
    p12 = c[1] + t * (c[2] - c[1])
    p23 = c[2] + t * (c[3] - c[2])
    a = p01 + t * (p12 - p01)
    b = p12 + t * (p23 - p12)
    m = a + t * (b - a)
    return np.array([c[0], p01, a, m]), np.array([m, b, p23, c[3]])


def _sub(c, t0, t1):
    if t1 <= t0:
        return None
    if t0 > 0:
        _, c = _split(c, t0)
        t1 = (t1 - t0) / (1 - t0) if t0 < 1 else 1.0
    if t1 < 1:
        c, _ = _split(c, t1)
    return c


def _bez(c, t):
    t = np.asarray(t)[:, None]
    mt = 1 - t
    return mt ** 3 * c[0] + 3 * mt ** 2 * t * c[1] + 3 * mt * t ** 2 * c[2] + t ** 3 * c[3]


def _pieces(shape):
    kind, (start, segs) = shape
    out, cur = [], np.asarray(start, float)
    for sg in segs:
        c, ln = _ctrl(cur, sg)
        out.append((c, ln))
        cur = c[3]
    return out


def _samples(pieces, step):
    S = []  # (piece index, t, point)
    for k, (c, ln) in enumerate(pieces):
        L = np.linalg.norm(np.diff(_bez(c, np.linspace(0, 1, 9)), axis=0), axis=1).sum()
        n = max(2, int(np.ceil(L / step)))
        ts = np.linspace(0, 1, n + 1)[:-1] if k < len(pieces) - 1 else np.linspace(0, 1, n + 1)
        for t, p in zip(ts, _bez(c, ts)):
            S.append((k, float(t), p))
    return S


def _build(pieces, S, i0, i1):
    """Open path from sample i0 to sample i1 (i0 < i1, indices into S)."""
    k0, t0, p0 = S[i0]
    k1, t1, p1 = S[i1]
    segs = []
    for k in range(k0, k1 + 1):
        c, ln = pieces[k]
        a = t0 if k == k0 else 0.0
        b = t1 if k == k1 else 1.0
        sub = _sub(c, a, b)
        if sub is None:
            continue
        segs.append(("L", sub[3].copy()) if ln else ("C", sub[1].copy(), sub[2].copy(), sub[3].copy()))
    if not segs:
        return None
    return ("open", (np.asarray(p0, float).copy(), segs))


def _length(shape):
    P = shape_points(shape, 16)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def _cut_shape(shape, bad, S, pieces, closed, min_mid, min_end, w):
    """Remove runs of bad samples; returns list of open shapes (or None if
    nothing changes)."""
    n = len(S)
    bad = bad.copy()
    # runs
    runs = []
    i = 0
    while i < n:
        if bad[i]:
            j = i
            while j + 1 < n and bad[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    if closed and len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == n - 1:
        last = runs.pop()
        runs[0] = [last[0] - n, runs[0][1]]
    pts = np.array([s[2] for s in S])
    seg_len = np.r_[np.linalg.norm(np.diff(pts, axis=0), axis=1), 0]

    def run_len(a, b):
        idx = np.arange(a, b + 1) % n
        return seg_len[idx[:-1]].sum() if len(idx) > 1 else 0.0
    keep_runs = []
    for a, b in runs:
        L = run_len(a, b)
        at_end = (not closed) and (a <= 0 or b >= n - 1)
        if L >= (min_end if at_end else min_mid):
            keep_runs.append((a, b))
    if not keep_runs:
        return None
    cut = np.zeros(n, bool)
    for a, b in keep_runs:
        cut[np.arange(a, b + 1) % n] = True
    if cut.all():
        return []
    # kept intervals
    out = []
    if closed:
        s0 = int(np.flatnonzero(cut)[0])
        order = [(s0 + k) % n for k in range(n)]
    else:
        order = list(range(n))
    cur = []
    for i in order + ([order[0]] if closed else []):
        if not cut[i]:
            cur.append(i)
        else:
            if len(cur) >= 2:
                out.append(cur)
            cur = []
    if len(cur) >= 2:
        out.append(cur)
    shapes = []
    for idx in out:
        # indices may wrap for closed paths: split at the wrap
        parts, part = [], [idx[0]]
        for a, b in zip(idx, idx[1:]):
            if b == a + 1:
                part.append(b)
            else:
                parts.append(part)
                part = [b]
        parts.append(part)
        segs_all, start = [], None
        for part in parts:
            if len(part) < 2:
                continue
            sh = _build(pieces, S, part[0], part[-1])
            if sh is None:
                continue
            if start is None:
                start = sh[1][0]
            segs_all += sh[1][1]
        if start is not None and segs_all:
            sh = ("open", (start, segs_all))
            if _length(sh) >= 0.8 * w:
                shapes.append(sh)
    return shapes


def _snap_ends(shape, others_tree, others_pts, w):
    kind, (start, segs) = shape
    if kind != "open" or others_tree is None:
        return shape
    start = np.asarray(start, float).copy()
    segs = [tuple(np.asarray(p, float).copy() if not isinstance(p, str) else p for p in sg) for sg in segs]
    d, i = others_tree.query(start)
    if d < 0.7 * w:
        q = others_pts[i]
        dl = q - start
        start = q.copy()
        if segs[0][0] == "C":
            segs[0] = ("C", segs[0][1] + dl, segs[0][2], segs[0][3])
    end = np.asarray(segs[-1][-1], float)
    d, i = others_tree.query(end)
    if d < 0.7 * w:
        q = others_pts[i]
        dl = q - end
        sg = segs[-1]
        segs[-1] = ("L", q.copy()) if sg[0] == "L" else ("C", sg[1], sg[2] + dl, q.copy())
    return (kind, (start, segs))


def _local_cost(shapes_before, shapes_after, ref, size, w, z):
    from .tracer import _render_shapes, count_anchors
    a = _render_shapes(shapes_before, size, z, w) > 0
    b = _render_shapes(shapes_after, size, z, w) > 0
    zone = a ^ b
    if not zone.any():
        return 0.0
    ea = ((a ^ ref) & zone).sum()
    eb = ((b ^ ref) & zone).sum()
    return (eb - ea) / (w * z) ** 2 + 0.03 * (count_anchors(shapes_after) - count_anchors(shapes_before))


def remove_overlaps(shapes, work, w):
    import cv2
    H, W = work.shape
    z = 3
    ref = cv2.resize(work, (W * z, H * z), interpolation=cv2.INTER_CUBIC) > 0.5
    step = 0.25 * w
    shapes = list(shapes)
    order = sorted(range(len(shapes)), key=lambda i: _length(shapes[i]) if shapes[i][0] != "circle" else 1e9)
    alive = {i: [shapes[i]] for i in range(len(shapes))}
    for i in order:
        cur = alive[i]
        if len(cur) != 1:
            continue
        sh = cur[0]
        if sh[0] == "circle":
            # a circle that another shape hides in part (a head behind a
            # head) is cut like any path
            from .geometry import ellipse_shape
            cx, cy, r = sh[1]
            sh = ellipse_shape(cx, cy, r, r)
        others = [x for j, lst in alive.items() if j != i for x in lst]
        opts = np.vstack([shape_points(x, 32) for x in others]) if others else np.zeros((0, 2))
        tree = cKDTree(opts) if len(opts) else None
        pieces = _pieces(sh)
        S = _samples(pieces, step)
        P = np.array([s[2] for s in S])
        on_other = tree.query(P)[0] < 0.6 * w if tree is not None else np.zeros(len(P), bool)
        ink = map_coordinates(work, [P[:, 1] - 0.5, P[:, 0] - 0.5], order=1, mode="constant") < 0.3
        closed = sh[0] == "path"
        new = _cut_shape(sh, on_other | ink, S, pieces, closed, min_mid=1.5 * w, min_end=1.2 * w, w=w)
        if new is None:
            # nothing to cut: a free end still snaps onto the line it meets
            if cur[0][0] != "circle" and sh[0] == "open" and tree is not None:
                sn = _snap_ends(sh, tree, opts, w)
                if _local_cost([sh], [sn], ref, (H, W), w, z) <= 0.05:
                    alive[i] = [sn]
            continue
        new = [_snap_ends(x, tree, opts, w) for x in new]
        if _local_cost(others + [sh], others + new, ref, (H, W), w, z) <= 0.3:
            alive[i] = new
    out = []
    for i in range(len(shapes)):
        out += alive[i]
    return out


def _closest_on(shape, p, n=64):
    """Closest point to p on shape (dense sampling + local refinement)."""
    P = shape_points(shape, n)
    d = np.linalg.norm(P - p, axis=1)
    k = int(np.argmin(d))
    # refine on the polyline segments around k
    best = (d[k], P[k])
    for j in (k - 1, k):
        if 0 <= j < len(P) - 1:
            a, b = P[j], P[j + 1]
            ab = b - a
            t = np.clip((p - a) @ ab / (ab @ ab + 1e-12), 0, 1)
            q = a + t * ab
            dq = np.linalg.norm(q - p)
            if dq < best[0]:
                best = (dq, q)
    return best


def _ray_hit(p, dirv, polys, max_t):
    """Smallest |t| (t in [-max_t, max_t]) where p + t*dirv crosses a polyline."""
    best = None
    nrm = np.array([-dirv[1], dirv[0]])
    eps = 1e-6
    for P in polys:
        s0 = (P[:-1] - p) @ nrm
        s1 = (P[1:] - p) @ nrm
        s0 = np.where(np.abs(s0) < eps, 0.0, s0)
        s1 = np.where(np.abs(s1) < eps, 0.0, s1)
        cross = np.flatnonzero(s0 * s1 <= 0)
        for j in cross:
            den = s0[j] - s1[j]
            if abs(den) < 1e-9:
                # the other line runs along the ray: its nearer end is the hit
                ts = [(P[j] - p) @ dirv, (P[j + 1] - p) @ dirv]
            else:
                q = P[j] + s0[j] / den * (P[j + 1] - P[j])
                ts = [(q - p) @ dirv]
            for t in ts:
                if abs(t) <= max_t and (best is None or abs(t) < abs(best)):
                    best = t
    return best


def weld_ends(shapes, w, reach=1.3):
    """Every free end of an open stroke that stops near another line is
    extended or shortened along its own direction until it lies exactly on
    that line's centre: the lines touch, with no gap and no overlap."""
    from .geometry import ellipse_shape
    shapes = [(k, d) for k, d in shapes]
    polys = []
    for k, d in shapes:
        if k == "circle":
            cx, cy, r = d
            polys.append(shape_points(ellipse_shape(cx, cy, r, r), 64))
        else:
            polys.append(shape_points((k, d), 32))
    for i, (kind, data) in enumerate(shapes):
        if kind != "open":
            continue
        start, segs = data
        start = np.asarray(start, float).copy()
        segs = [tuple(x if isinstance(x, str) else np.asarray(x, float).copy() for x in sg) for sg in segs]
        others = [P for j, P in enumerate(polys) if j != i]
        if not others:
            continue
        for end in (0, 1):
            if end == 0:
                p = start
                q = np.asarray(segs[0][1], float)
            else:
                p = np.asarray(segs[-1][-1], float)
                q = np.asarray(segs[-1][2], float) if segs[-1][0] == "C" else \
                    (np.asarray(segs[-2][-1], float) if len(segs) > 1 else start)
            dirv = p - q
            nd = np.linalg.norm(dirv)
            if nd < 1e-9:
                continue
            dirv /= nd
            # only ends that stop near another line (not free ends in the open)
            near = min(np.min(np.linalg.norm(P - p, axis=1)) for P in others)
            if near > reach * w:
                continue
            t = _ray_hit(p, dirv, others, reach * w)
            if t is None or abs(t) < 1e-3:
                continue
            target = p + t * dirv
            if end == 0:
                dl = target - start
                start = target.copy()
                if segs[0][0] == "C":
                    segs[0] = ("C", segs[0][1] + dl, segs[0][2], segs[0][3])
            else:
                sg = segs[-1]
                dl = target - sg[-1]
                segs[-1] = ("L", target.copy()) if sg[0] == "L" else ("C", sg[1], sg[2] + dl, target.copy())
        shapes[i] = (kind, (start, segs))
        polys[i] = shape_points(shapes[i], 32)
    return _join_facing_ends(shapes, w, reach, polys)


def _join_facing_ends(shapes, w, reach, polys):
    """Two free ends that face each other across a small gap (a line broken
    by a cut) and touch nothing else meet in the middle."""
    ends = []
    for i, (k, d) in enumerate(shapes):
        if k != "open":
            continue
        start, segs = d
        for e, p in ((0, np.asarray(start, float)), (1, np.asarray(segs[-1][-1], float))):
            on_line = any(np.min(np.linalg.norm(P - p, axis=1)) < 0.15 * w
                          for j, P in enumerate(polys) if j != i)
            if not on_line:
                ends.append((i, e, p))
    moved = {}
    for a in range(len(ends)):
        for b in range(a + 1, len(ends)):
            ia, ea, pa = ends[a]
            ib, eb, pb = ends[b]
            if ia == ib or (ia, ea) in moved or (ib, eb) in moved:
                continue
            d = np.linalg.norm(pa - pb)
            if 1e-3 < d < reach * w:
                m = (pa + pb) / 2
                moved[(ia, ea)] = m
                moved[(ib, eb)] = m
    out = list(shapes)
    for (i, e), m in moved.items():
        k, (start, segs) = out[i]
        start = np.asarray(start, float).copy()
        segs = list(segs)
        if e == 0:
            dl = m - start
            start = m.copy()
            if segs[0][0] == "C":
                segs[0] = ("C", segs[0][1] + dl, segs[0][2], segs[0][3])
        else:
            sg = segs[-1]
            dl = m - sg[-1]
            segs[-1] = ("L", m.copy()) if sg[0] == "L" else ("C", sg[1], sg[2] + dl, m.copy())
        out[i] = (k, (start, segs))
    return out
