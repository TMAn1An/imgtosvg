"""Curve fitting helpers: least-squares cubic Bezier fitting (Schneider style)
with optional tangent constraints, plus small geometry utilities."""
import numpy as np


def bez(ctrl, t):
    t = np.asarray(t)[:, None]
    mt = 1 - t
    return (mt ** 3) * ctrl[0] + 3 * (mt ** 2) * t * ctrl[1] + 3 * mt * (t ** 2) * ctrl[2] + (t ** 3) * ctrl[3]


def bez_d1(ctrl, t):
    t = np.asarray(t)[:, None]
    mt = 1 - t
    return 3 * (mt ** 2) * (ctrl[1] - ctrl[0]) + 6 * mt * t * (ctrl[2] - ctrl[1]) + 3 * (t ** 2) * (ctrl[3] - ctrl[2])


def bez_d2(ctrl, t):
    t = np.asarray(t)[:, None]
    return 6 * (1 - t) * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) + 6 * t * (ctrl[3] - 2 * ctrl[2] + ctrl[1])


def chord_params(pts):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if d[-1] <= 0:
        return np.linspace(0, 1, len(pts))
    return d / d[-1]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _solve(pts, u, p0, p3, t1, t2):
    """Least squares for the inner control points.
    t1/t2 are unit tangents (t1 points into the curve from p0, t2 points into
    the curve from p3) or None meaning the control point is free."""
    mt = 1 - u
    b0, b1, b2, b3 = mt ** 3, 3 * mt ** 2 * u, 3 * mt * u ** 2, u ** 3
    rhs = pts - np.outer(b0, p0) - np.outer(b3, p3)
    cols = []
    base = np.zeros_like(pts)
    if t1 is None:
        cols += [np.c_[b1, 0 * b1], np.c_[0 * b1, b1]]
    else:
        base += np.outer(b1, p0)
        cols.append(np.outer(b1, t1))
    if t2 is None:
        cols += [np.c_[b2, 0 * b2], np.c_[0 * b2, b2]]
    else:
        base += np.outer(b2, p3)
        cols.append(np.outer(b2, t2))
    rhs = (rhs - base).ravel()
    A = np.stack([c.ravel() for c in cols], axis=1)
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    chord = np.linalg.norm(p3 - p0)
    k = 0
    if t1 is None:
        p1 = sol[0:2]
        k = 2
    else:
        a = sol[0]
        k = 1
        if not (1e-3 * chord < a < 1.5 * chord + 1):
            a = chord / 3
        p1 = p0 + a * t1
    if t2 is None:
        p2 = sol[k:k + 2]
    else:
        a = sol[k]
        if not (1e-3 * chord < a < 1.5 * chord + 1):
            a = chord / 3
        p2 = p3 + a * t2
    return np.array([p0, p1, p2, p3], float)


def _reparam(ctrl, pts, u):
    d = bez(ctrl, u) - pts
    d1 = bez_d1(ctrl, u)
    d2 = bez_d2(ctrl, u)
    num = (d * d1).sum(1)
    den = (d1 * d1).sum(1) + (d * d2).sum(1)
    ok = np.abs(den) > 1e-12
    nu = u.copy()
    nu[ok] = u[ok] - num[ok] / den[ok]
    nu = np.clip(nu, 0, 1)
    nu[0], nu[-1] = 0, 1
    return np.maximum.accumulate(nu)


def fit_single(pts, p0, p3, t1, t2, iters=6):
    u = chord_params(pts)
    ctrl = _solve(pts, u, p0, p3, t1, t2)
    for _ in range(iters):
        u = _reparam(ctrl, pts, u)
        ctrl = _solve(pts, u, p0, p3, t1, t2)
    err = np.linalg.norm(bez(ctrl, u) - pts, axis=1)
    return ctrl, err


def tangent_at(pts, i, win):
    a = max(0, i - win)
    b = min(len(pts) - 1, i + win)
    return unit(pts[b] - pts[a])


def snap_dir(t, deg):
    """Snap a direction to horizontal/vertical when within `deg` degrees."""
    ang = np.degrees(np.arctan2(t[1], t[0]))
    for axis in (0, 90, 180, -90, -180):
        if abs(ang - axis) < deg:
            r = np.radians(axis)
            return np.array([np.cos(r), np.sin(r)])
    return t


def fit_curve(pts, p0, p3, t1, t2, tol, ds, depth=0):
    """Fit pts with as few cubic segments as possible within tol.
    Returns a list of 4x2 control arrays."""
    n = len(pts)
    if n < 4:
        c = np.array([p0, p0 + (p3 - p0) / 3, p0 + 2 * (p3 - p0) / 3, p3])
        return [c]
    ctrl, err = fit_single(pts, p0, p3, t1, t2)
    if err.max() <= tol or depth > 12 or n < 8:
        return [ctrl]
    # choose split: prefer a nearby axis extremum (where designers put anchors)
    imax = int(np.argmax(err))
    win = max(2, int(round(0.8 / ds)))
    lo, hi = win, n - 1 - win
    if hi <= lo:
        return [ctrl]
    tang = np.array([tangent_at(pts, i, win) for i in range(lo, hi + 1)])
    ext = []
    for comp in (0, 1):
        s = np.sign(tang[:, comp])
        idx = np.where(s[:-1] * s[1:] < 0)[0]
        ext += list(idx + lo)
    split = min(max(imax, lo), hi)
    if ext:
        cand = min(ext, key=lambda i: abs(i - imax))
        if abs(cand - imax) < 0.35 * n:
            split = cand
    tm = snap_dir(tangent_at(pts, split, win), 4.0)
    pm = pts[split]
    left = fit_curve(pts[:split + 1], p0, pm, t1, -tm, tol, ds, depth + 1)
    right = fit_curve(pts[split:], pm, p3, tm, t2, tol, ds, depth + 1)
    return left + right


def line_fit(pts):
    """Total least squares line. Returns (centroid, direction, max deviation)."""
    c = pts.mean(0)
    q = pts - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    d = vt[0]
    dev = np.abs(q @ np.array([-d[1], d[0]]))
    return c, d, dev.max()


def intersect(p, d, q, e):
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-9:
        return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return p + t * d


def _seg_ctrl(p0, sg):
    if sg[0] == "L":
        p3 = np.asarray(sg[1], float)
        return np.array([p0, p0 + (p3 - p0) / 3, p0 + 2 * (p3 - p0) / 3, p3]), True
    return np.array([p0, sg[1], sg[2], sg[3]], float), False


def _t_start(c):
    v = c[1] - c[0]
    if np.linalg.norm(v) < 1e-9:
        v = c[2] - c[0]
    if np.linalg.norm(v) < 1e-9:
        v = c[3] - c[0]
    return unit(v)


def _t_end(c):
    v = c[3] - c[2]
    if np.linalg.norm(v) < 1e-9:
        v = c[3] - c[1]
    if np.linalg.norm(v) < 1e-9:
        v = c[3] - c[0]
    return unit(v)


def simplify(start, segs, closed, tol, corner_deg=35.0, straight_tol=0.08, keep_line=3.0):
    """Greedy anchor removal (like Illustrator's Simplify): repeatedly drop the
    smooth anchor whose two neighbouring segments can be replaced by one cubic
    with the smallest error, while that error stays below tol."""
    if len(segs) < 2:
        return segs
    ctrls = []
    cur = np.asarray(start, float)
    for sg in segs:
        c, is_line = _seg_ctrl(cur, sg)
        ctrls.append([c, is_line])
        cur = c[3]
    cos_lim = np.cos(np.radians(corner_deg))
    while len(ctrls) > 1:
        best = None
        for i in range(1, len(ctrls)):
            a, b = ctrls[i - 1][0], ctrls[i][0]
            # real straight edges stay straight
            long_a = ctrls[i - 1][1] and np.linalg.norm(a[3] - a[0]) > keep_line
            long_b = ctrls[i][1] and np.linalg.norm(b[3] - b[0]) > keep_line
            if long_a or long_b:
                if not (ctrls[i - 1][1] and ctrls[i][1]) or _t_end(a) @ _t_start(b) < 0.9995:
                    continue
            if _t_end(a) @ _t_start(b) < cos_lim:
                continue
            u = np.linspace(0, 1, 14)
            pts = np.vstack([bez(a, u), bez(b, u)[1:]])
            t1, t2 = _t_start(a), -_t_end(b)
            c, err = fit_single(pts, a[0], b[3], t1, t2)
            e = err.max()
            if e <= tol and (best is None or e < best[0]):
                best = (e, i, c)
        if best is None:
            break
        _, i, c = best
        ch = c[3] - c[0]
        L = np.linalg.norm(ch)
        is_line = False
        if L > 1e-9:
            nrm = np.array([-ch[1], ch[0]]) / L
            is_line = abs((c[1] - c[0]) @ nrm) < straight_tol and abs((c[2] - c[0]) @ nrm) < straight_tol
        ctrls[i - 1:i + 1] = [[c, is_line]]
    out = []
    for c, is_line in ctrls:
        out.append(("L", c[3]) if is_line else ("C", c[1], c[2], c[3]))
    return out


def fillet(start, segs, closed, tol, keep_line, max_len, sharp_r):
    """Clean up the joint between two long straight edges: the short wiggly
    segments between them become either one tangent-continuous curve (a
    rounded corner, 2 anchors) or a crisp corner at the lines' intersection."""
    ctrls = []
    cur = np.asarray(start, float)
    for sg in segs:
        c, is_line = _seg_ctrl(cur, sg)
        ctrls.append([c, is_line])
        cur = c[3]
    n = len(ctrls)
    is_long = [c[1] and np.linalg.norm(c[0][3] - c[0][0]) > keep_line for c in ctrls]
    if sum(is_long) < (1 if closed else 2):
        return start, segs
    if closed:
        k = is_long.index(True)
        ctrls = ctrls[k:] + ctrls[:k]
        is_long = is_long[k:] + is_long[:k]
        ctrls.append(ctrls[0])  # sentinel: the first line again
        is_long.append(True)
    out = [ctrls[0]]
    i = 1
    N = len(ctrls)
    while i < N:
        if is_long[i] or out[-1] is None:
            out.append(ctrls[i])
            i += 1
            continue
        # run of short segments i..j-1 followed by a long line j
        j = i
        while j < N and not is_long[j]:
            j += 1
        prev_long = out[-1][1] and np.linalg.norm(out[-1][0][3] - out[-1][0][0]) > keep_line
        if j >= N or not prev_long:
            out.extend(ctrls[i:j])
            i = j
            continue
        run = ctrls[i:j]
        u = np.linspace(0, 1, 10)
        pts = np.vstack([bez(c[0], u) for c in run])
        L = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        A = out[-1][0]
        B = ctrls[j][0]
        dA, dB = unit(A[3] - A[0]), unit(B[3] - B[0])
        if L > max_len or dA @ dB > 0.985:
            out.extend(run)
            i = j
            continue
        X = intersect(A[0], dA, B[0], dB)
        # blur rounds acute corners more: scale the allowance by 1/sin(half interior angle)
        half = np.arccos(np.clip(-dA @ dB, -1, 1)) / 2
        lim = sharp_r / max(np.sin(half), 0.3)
        if X is not None and np.min(np.linalg.norm(pts - X, axis=1)) < lim \
                and (X - A[0]) @ dA > 0 and (B[3] - X) @ dB > 0:
            # crisp corner: extend both lines to their intersection
            out[-1] = [np.array([A[0], A[0] + (X - A[0]) / 3, A[0] + 2 * (X - A[0]) / 3, X]), True]
            B2 = np.array([X, X + (B[3] - X) / 3, X + 2 * (B[3] - X) / 3, B[3]])
            ctrls[j] = [B2, True]
            i = j
            continue
        pts[0], pts[-1] = A[3], B[0]
        c, err = fit_single(pts, A[3], B[0], dA, -dB)
        if err.max() <= tol:
            out.append([c, False])
        else:
            out.extend(run)
        i = j
    if closed:
        last = out.pop()  # sentinel (possibly modified start)
        out[0] = [np.array([last[0][0], out[0][0][1], out[0][0][2], out[0][0][3]]) if False else last[0], True]
        start = out[0][0][0]
    else:
        start = out[0][0][0]
    res = []
    for c, is_line in out:
        res.append(("L", c[3]) if is_line else ("C", c[1], c[2], c[3]))
    return start, res


def arc_to_cubics(c, r, a0, a1):
    """Circular arc from angle a0 to a1 (radians, any direction) as cubics,
    one per <= 90 degrees (the way vector editors build circles)."""
    sweep = a1 - a0
    n = max(1, int(np.ceil(abs(sweep) / (np.pi / 2) - 1e-6)))
    out = []
    for i in range(n):
        t0 = a0 + sweep * i / n
        t1 = a0 + sweep * (i + 1) / n
        k = 4 / 3 * np.tan((t1 - t0) / 4)
        p0 = c + r * np.array([np.cos(t0), np.sin(t0)])
        p3 = c + r * np.array([np.cos(t1), np.sin(t1)])
        c1 = p0 + k * r * np.array([-np.sin(t0), np.cos(t0)])
        c2 = p3 - k * r * np.array([-np.sin(t1), np.cos(t1)])
        out.append(("C", c1, c2, p3))
    return out


def snap_axis(start, segs, closed, deg=3.0):
    """Make almost horizontal/vertical straight segments exactly H/V, moving
    the shared anchors (and the handles attached to them) consistently."""
    if not segs:
        return start, segs
    pts = [np.array(start, float)] + [np.array(sg[-1], float) for sg in segs]
    hin = [None] + [np.array(sg[2], float) if sg[0] == "C" else None for sg in segs]   # handle into anchor i
    hout = [np.array(sg[1], float) if sg[0] == "C" else None for sg in segs] + [None]  # handle out of anchor i
    n = len(segs)
    lim = np.tan(np.radians(deg))
    for i, sg in enumerate(segs):
        if sg[0] != "L":
            continue
        a, b = pts[i], pts[i + 1]
        d = b - a
        if np.hypot(*d) < 1e-6:
            continue
        for ax in (0, 1):  # ax=0: vertical line (same x); ax=1: horizontal (same y)
            other = 1 - ax
            if abs(d[ax]) <= lim * abs(d[other]) and abs(d[ax]) > 1e-9:
                m = (a[ax] + b[ax]) / 2
                for j in (i, i + 1):
                    idx = [j]
                    if closed and j in (0, n):
                        idx = [0, n]
                    for k in idx:
                        delta = m - pts[k][ax]
                        pts[k][ax] = m
                        if hin[k] is not None:
                            hin[k][ax] += delta
                        if hout[k] is not None:
                            hout[k][ax] += delta
                break
    out = []
    for i, sg in enumerate(segs):
        if sg[0] == "L":
            out.append(("L", pts[i + 1]))
        else:
            out.append(("C", hout[i], hin[i + 1], pts[i + 1]))
    return pts[0], out
