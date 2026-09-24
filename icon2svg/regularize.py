"""Whole-icon clean-up, the part a designer does by eye:
  * collinear straight strokes that continue each other become one line
  * horizontal/vertical edges that are almost aligned get exactly the same
    coordinate (shared grid), so parallel walls, window rows etc. line up
"""
import numpy as np


class _Ed:
    """Editable path: anchors + in/out handles."""

    def __init__(self, kind, data):
        self.kind = kind
        start, segs = data
        self.pts = [np.array(start, float)] + [np.array(sg[-1], float) for sg in segs]
        self.types = [sg[0] for sg in segs]
        self.hout = [np.array(sg[1], float) if sg[0] == "C" else None for sg in segs] + [None]
        self.hin = [None] + [np.array(sg[2], float) if sg[0] == "C" else None for sg in segs]

    def move(self, i, ax, val):
        idx = [i]
        if self.kind == "path" and i in (0, len(self.pts) - 1) and np.allclose(self.pts[0], self.pts[-1]):
            idx = [0, len(self.pts) - 1]
        for k in idx:
            d = val - self.pts[k][ax]
            self.pts[k][ax] = val
            if self.hin[k] is not None:
                self.hin[k][ax] += d
            if self.hout[k] is not None:
                self.hout[k][ax] += d

    def data(self):
        segs = []
        for i, t in enumerate(self.types):
            if t == "L":
                segs.append(("L", self.pts[i + 1]))
            else:
                segs.append(("C", self.hout[i], self.hin[i + 1], self.pts[i + 1]))
        return (self.pts[0], segs)


def _axis_of(a, b, eps=1e-6):
    if abs(a[0] - b[0]) < eps and abs(a[1] - b[1]) > eps:
        return 0  # vertical: shared x
    if abs(a[1] - b[1]) < eps and abs(a[0] - b[0]) > eps:
        return 1  # horizontal: shared y
    return None


def _cluster(vals, weights, tol):
    """1-D clustering; returns list of (indices, weighted mean)."""
    order = np.argsort(vals)
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if vals[i] - vals[cur[0]] <= tol:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    out = []
    for g in groups:
        wv = np.array([weights[i] for i in g])
        out.append((g, float(np.sum(np.array([vals[i] for i in g]) * wv) / wv.sum())))
    return out


def merge_collinear(shapes, s, w):
    """Join open straight H/V strokes that continue each other (also across a
    stretch drawn by another shape's edge)."""
    lines, bridges, rest = [], [], []
    for kind, data in shapes:
        if kind == "open" and len(data[1]) == 1 and data[1][0][0] == "L":
            a, b = np.asarray(data[0], float), np.asarray(data[1][0][1], float)
            ax = _axis_of(a, b)
            if ax is not None:
                lines.append([ax, a, b])
                continue
        rest.append((kind, data))
        if kind in ("path", "open"):
            ed = _Ed(kind, data)
            for i, t in enumerate(ed.types):
                if t == "L":
                    ax = _axis_of(ed.pts[i], ed.pts[i + 1])
                    if ax is not None:
                        bridges.append((ax, ed.pts[i].copy(), ed.pts[i + 1].copy()))
    tol_pos = 0.8 * s
    gap = 0.7 * w
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                ax, a1, b1 = lines[i]
                ax2, a2, b2 = lines[j]
                if ax != ax2 or abs(a1[ax] - a2[ax]) > tol_pos:
                    continue
                o = 1 - ax  # coordinate along the line
                lo1, hi1 = sorted((a1[o], b1[o]))
                lo2, hi2 = sorted((a2[o], b2[o]))
                if lo2 < lo1:
                    lo1, hi1, lo2, hi2 = lo2, hi2, lo1, hi1
                g0, g1 = hi1, lo2  # the gap between them (may be negative = overlap)
                ok = g1 - g0 <= gap
                if not ok:
                    covered = [(min(p[o], q[o]), max(p[o], q[o])) for bx, p, q in bridges
                               if bx == ax and abs(p[ax] - a1[ax]) <= tol_pos]
                    covered.sort()
                    x = g0
                    for c0, c1 in covered:
                        if c0 - gap <= x:
                            x = max(x, c1)
                    ok = x >= g1 - gap
                if ok:
                    L1, L2 = abs(hi1 - lo1), abs(hi2 - lo2)
                    pos = (a1[ax] * L1 + a2[ax] * L2) / max(L1 + L2, 1e-9)
                    na, nb = np.zeros(2), np.zeros(2)
                    na[ax] = nb[ax] = pos
                    na[o], nb[o] = min(lo1, lo2), max(hi1, hi2)
                    lines[i] = [ax, na, nb]
                    lines.pop(j)
                    changed = True
                    break
            if changed:
                break
    out = rest + [("open", (a, [("L", b)])) for _, a, b in lines]
    return out


def align(shapes, s, w):
    """Snap nearly equal x (of vertical edges) and y (of horizontal edges)."""
    eds = []
    refs = {0: [], 1: []}  # ax -> [(value, weight, ed_index, anchor_index)]
    for kind, data in shapes:
        if kind not in ("path", "open"):
            eds.append(None)
            continue
        ed = _Ed(kind, data)
        k = len(eds)
        eds.append(ed)
        for i, t in enumerate(ed.types):
            if t != "L":
                continue
            a, b = ed.pts[i], ed.pts[i + 1]
            ax = _axis_of(a, b)
            if ax is None:
                continue
            L = np.linalg.norm(b - a)
            refs[ax].append((a[ax], L, k, i))
            refs[ax].append((b[ax], L, k, i + 1))
    for ax in (0, 1):
        R = refs[ax]
        if not R:
            continue
        vals = np.array([r[0] for r in R])
        wts = np.array([r[1] for r in R]) + 1e-3
        for g, m in _cluster(vals, wts, 0.9 * s):
            for idx in g:
                _, _, k, i = R[idx]
                eds[k].move(i, ax, m)
    out = []
    for (kind, data), ed in zip(shapes, eds):
        out.append((kind, ed.data()) if ed is not None else (kind, data))
    return out


def snap_all(shapes, deg=4.0):
    from .bezier import snap_axis
    out = []
    for kind, data in shapes:
        if kind in ("path", "open"):
            st, sg = snap_axis(data[0], data[1], kind == "path", deg)
            out.append((kind, (st, sg)))
        else:
            out.append((kind, data))
    return out


def regularize(shapes, s, w):
    from .symfit import snap_handles
    shapes = snap_handles(shapes)
    shapes = snap_all(shapes)
    shapes = align(shapes, s, w)
    shapes = merge_collinear(shapes, s, w)
    shapes = align(shapes, s, w)
    return shapes
