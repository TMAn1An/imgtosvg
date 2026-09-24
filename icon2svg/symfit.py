"""Symmetric shape fitting and designer-style handle cleanup.

A shape that is mirror-symmetric about a vertical axis (a map pin, a head, a
body, a briefcase lid, ...) is fitted only on one half; the other half is its
exact mirror. That gives a perfectly symmetric result with half the fitting
noise and fewer anchor points, the way a designer draws it.
"""
import numpy as np
from scipy.spatial import cKDTree

from .bezier import unit


def _resample(P, n, closed):
    Q = np.vstack([P, P[:1]]) if closed else P
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    t = np.linspace(0, d[-1], n, endpoint=not closed)
    return np.c_[np.interp(t, d, Q[:, 0]), np.interp(t, d, Q[:, 1])]


def mirror_axis(P, s):
    """Best vertical mirror axis x=c for the closed polyline P and its rms error."""
    Q = _resample(P, 240, True)
    tree = cKDTree(Q)
    cx = Q[:, 0].mean()
    best = (np.inf, cx)
    for c in np.arange(cx - 2.5 * s, cx + 2.5 * s + 1e-9, 0.1 * s):
        M = Q.copy()
        M[:, 0] = 2 * c - M[:, 0]
        d, _ = tree.query(M)
        e = np.sqrt(np.mean(d ** 2))
        if e < best[0]:
            best = (e, c)
    return best[1], best[0]


def _crossings(Q, c):
    """Indices/points where the closed polyline crosses x=c."""
    out = []
    n = len(Q)
    for i in range(n):
        a, b = Q[i], Q[(i + 1) % n]
        da, db = a[0] - c, b[0] - c
        if da == 0:
            out.append((i, a.copy()))
        elif da * db < 0:
            t = da / (da - db)
            out.append((i, a + t * (b - a)))
    return out


def _reverse(start, segs):
    pts = [np.asarray(start, float)] + [np.asarray(sg[-1], float) for sg in segs]
    out = []
    for i in range(len(segs) - 1, -1, -1):
        sg = segs[i]
        if sg[0] == "L":
            out.append(("L", pts[i]))
        else:
            out.append(("C", np.asarray(sg[2], float), np.asarray(sg[1], float), pts[i]))
    return pts[-1], out


def _mirror_seg(sg, c):
    def m(p):
        p = np.asarray(p, float).copy()
        p[0] = 2 * c - p[0]
        return p
    return (sg[0],) + tuple(m(p) for p in sg[1:])


def fit_symmetric_closed(P, fit_open, s, max_err=None):
    """Fit a closed boundary as a mirror-symmetric path, or return None.
    `fit_open(points) -> (start, segs)` fits one open half."""
    size = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    if size < 4 * s:
        return None
    c, err = mirror_axis(P, s)
    if err > (max_err if max_err is not None else 0.35 * s + 0.015 * size):
        return None
    Q = _resample(P, 400, True)
    cr = _crossings(Q, c)
    if len(cr) != 2:
        return None
    (i0, p0), (i1, p1) = cr
    if p0[1] > p1[1]:
        (i0, p0), (i1, p1) = (i1, p1), (i0, p0)  # p0 = top crossing
    n = len(Q)

    def arc(i_from, p_from, i_to, p_to):
        idx = []
        k = (i_from + 1) % n
        while True:
            idx.append(k)
            if k == i_to:
                break
            k = (k + 1) % n
        return np.vstack([p_from, Q[idx], p_to])

    A = arc(i0, p0, i1, p1)   # top -> bottom one way
    B = arc(i1, p1, i0, p0)[::-1]  # top -> bottom the other way
    if A[:, 0].mean() > c:
        A, B = B, A  # A = left half
    B = B.copy()
    B[:, 0] = 2 * c - B[:, 0]
    m = 120
    H = (_resample(A, m, False) + _resample(B, m, False)) / 2
    H[0, 0] = H[-1, 0] = c
    if np.any(H[1:-1, 0] > c + 0.3 * s):
        return None
    H = _sharpen_tip(H, c, s)
    start, segs = fit_open(H)
    if not segs:
        return None
    # smooth at the axis -> the handle there is exactly horizontal
    for at_start in (True, False):
        pts_near = H[: max(3, m // 25)] if at_start else H[-max(3, m // 25):][::-1]
        t = unit(pts_near[-1] - pts_near[0])
        smooth = abs(t[1]) < np.sin(np.radians(30))
        if not smooth:
            continue
        if at_start and segs[0][0] == "C":
            h = np.asarray(segs[0][1], float).copy()
            h[1] = start[1]
            segs[0] = ("C", h, segs[0][2], segs[0][3])
        if not at_start and segs[-1][0] == "C":
            end = np.asarray(segs[-1][3], float)
            h = np.asarray(segs[-1][2], float).copy()
            h[1] = end[1]
            segs[-1] = ("C", segs[-1][1], h, segs[-1][3])
    start = np.asarray(start, float).copy()
    start[0] = c
    last = segs[-1]
    end = np.asarray(last[-1], float).copy()
    end[0] = c
    segs[-1] = last[:-1] + (end,)
    # right half = mirrored, reversed left half
    rs, rsegs = _reverse(start, segs)
    rsegs = [_mirror_seg(sg, c) for sg in rsegs]
    full = list(segs) + rsegs
    full = _merge_collinear_ring(start, full)
    return ("path", full)


def _sharpen_tip(H, c, s):
    """If a symmetric half ends in a flat/cupped bottom that runs sideways into
    the axis (a pin resting on its base), cut that part off and extend the
    side down to a sharp tip on the axis."""
    m = len(H)
    d = np.diff(H, axis=0)
    seglen = np.linalg.norm(d, axis=1) + 1e-12
    t = d / seglen[:, None]
    L = seglen.sum()
    k = m - 2
    run = 0.0
    while k > 0 and abs(t[k, 1]) < 0.55 and run < 0.3 * L:
        run += seglen[k]
        k -= 1
    if run < 1.0 * s or run >= 0.3 * L:
        return H
    # side direction just above the cut
    j0 = max(0, k - max(3, m // 20))
    side = H[k] - H[j0]
    if np.linalg.norm(side) < 1e-6 or side[1] <= 0 or side[0] <= 0:
        return H
    side = side / np.linalg.norm(side)
    tt = (c - H[k][0]) / side[0]
    if tt <= 0 or tt > 0.5 * L:
        return H
    tip = H[k] + tt * side
    if tip[1] < H[k][1]:
        return H
    ext = np.linspace(H[k], tip, max(3, int(tt / 0.3)))[1:]
    out = np.vstack([H[:k + 1], ext])
    out[-1, 0] = c
    return out


def _merge_collinear_ring(start, segs):
    """Drop anchors between two collinear straight segments (also across the
    start point of the closed path)."""
    pts = [np.asarray(start, float)] + [np.asarray(sg[-1], float) for sg in segs]

    def col(a, b, c):
        d1, d2 = unit(b - a), unit(c - b)
        return d1 @ d2 > 0.9995
    changed = True
    while changed and len(segs) > 2:
        changed = False
        for i in range(len(segs)):
            j = (i + 1) % len(segs)
            if segs[i][0] == "L" and segs[j][0] == "L":
                a, b, c = pts[i], pts[i + 1], pts[(j + 1)]
                if col(a, b, c):
                    if j == 0:  # across the start point: rotate first
                        segs = segs[1:] + segs[:1]
                        pts = pts[1:] + [pts[1]]
                        changed = True
                        break
                    segs = segs[:i] + [("L", c)] + segs[j + 1:]
                    pts = pts[:i + 1] + pts[i + 2:]
                    changed = True
                    break
    return (pts[0], segs)


def snap_handles(shapes, deg=8.0):
    """At smooth anchors whose tangent is almost horizontal/vertical, make both
    handles exactly horizontal/vertical (anchors at extremes, like a designer)."""
    lim = np.radians(deg)
    out = []
    for kind, data in shapes:
        if kind not in ("path", "open"):
            out.append((kind, data))
            continue
        start, segs = data
        segs = [tuple(np.asarray(p, float).copy() if k else p for k, p in enumerate(sg)) for sg in segs]
        n = len(segs)
        pts = [np.asarray(start, float)] + [sg[-1] for sg in segs]
        closed = kind == "path"
        for i in range(n):
            j = i + 1 if i + 1 < n else (0 if closed else None)
            if j is None:
                continue
            a, b = segs[i], segs[j]
            if a[0] != "C" or b[0] != "C":
                continue
            p = a[3]
            hin, hout = a[2], b[1]
            v1, v2 = p - hin, hout - p
            L1, L2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if L1 < 1e-6 or L2 < 1e-6 or unit(v1) @ unit(v2) < np.cos(np.radians(12)):
                continue
            t = unit(v1 * L2 + v2 * L1)
            ang = np.arctan2(t[1], t[0])
            for axis in (0, np.pi / 2, np.pi, -np.pi / 2, -np.pi):
                if abs(ang - axis) < lim:
                    d = np.array([np.cos(axis), np.sin(axis)])
                    a = (a[0], a[1], p - d * L1, p)
                    b = (b[0], p + d * L2, b[2], b[3])
                    segs[i], segs[j] = a, b
                    break
        out.append((kind, (start, segs)))
    return out
