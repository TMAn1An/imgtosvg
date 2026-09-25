"""Closed shapes from the holes of the ink.

Every white area enclosed by a line (a window, one half of a compass needle)
is the inside of a closed shape.  The stroke graph can fail to close such a
loop where lines meet in a blob; the hole itself cannot: its outline, pushed
out by half a line width, is the centre line of the shape around it.  That
loop is fitted with an exact primitive (triangle, rectangle, ellipse ...) and
replaces the strokes it draws, when the result matches the ink at least as
well.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geometry import primitive_closed, shape_points

Z = 4


def _cost(shapes, ref, size, w):
    from .tracer import _render_shapes, count_anchors
    got = _render_shapes(shapes, size, Z, w) > 0
    return (got ^ ref).sum() / (w * Z) ** 2 + 0.08 * count_anchors(shapes)


def _remove_covered(shapes, added, w):
    if not added:
        return shapes
    tree = cKDTree(np.vstack([shape_points(p, 40) for p in added]))
    out = []
    for sh in shapes:
        dd, _ = tree.query(shape_points(sh, 24))
        if (dd < 0.5 * w).mean() >= 0.9:
            continue          # drawn by the new shapes
        out.append(sh)
    return out


def repair_holes(shapes, work, w, s, smooth_opt=None, max_frac=0.25):
    H, W = work.shape
    big = cv2.resize(work, (W * Z, H * Z), interpolation=cv2.INTER_CUBIC)
    ref = big > 0.5
    white = (~ref).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    holes = [k for k in range(1, n) if k not in border
             and (0.8 * w * Z) ** 2 < stats[k, cv2.CC_STAT_AREA] < max_frac * H * W * Z * Z]
    if not holes:
        return shapes
    r = max(1, int(round(0.5 * w * Z)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    prims = []
    for k in sorted(holes, key=lambda k: stats[k, cv2.CC_STAT_AREA]):
        region = cv2.dilate((lab == k).astype(np.uint8), ker)
        cs, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        loop = (max(cs, key=len)[:, 0, :].astype(float) + 0.5) / Z
        if len(loop) < 12:
            continue
        done = None
        for sh in shapes:
            if sh[0] == "open":
                continue
            d, _ = cKDTree(shape_points(sh, 24)).query(loop)
            if (d < 0.35 * w).mean() >= 0.9 and d.max() < 0.7 * w:
                done = sh         # already drawn as one closed shape
                break
        # a very small hole is usually a blot where lines meet, not a shape
        if stats[k, cv2.CC_STAT_AREA] < 1.7 * (w * Z) ** 2:
            continue
        prim = _offset_polygon(lab == k, w) or primitive_closed(loop, s, w)
        if done is not None:
            n_done = 4 if done[0] == "circle" else len(done[1][1])
            n_new = None if prim is None else (4 if prim[0] == "circle" else len(prim[1][1]))
            # only a cleaner, simpler shape replaces one that is already there
            if n_new is None or n_new > n_done or done[0] == "circle":
                continue
        if prim is None and smooth_opt is not None and stats[k, cv2.CC_STAT_AREA] < 0.06 * H * W * Z * Z:
            # a free-form hole (the inside of a swoosh arrow, one scallop):
            # one smooth closed shape around it
            from .tracer import fit_path
            try:
                prim = fit_path(loop, True, smooth_opt, s)
                if len(prim[1][1]) > 8:
                    prim = None
            except Exception:
                prim = None
        if prim is not None:
            # sit it exactly on the ink centre before judging it
            from .refine import refine_shapes
            try:
                prim = refine_shapes([prim], work, w)[0]
            except Exception:
                pass
            prims.append(prim)
    if not prims:
        return shapes
    base = _cost(shapes, ref, (H, W), w)
    # all together first (neighbouring holes share the strokes they replace)
    cand = _remove_covered(shapes, prims, w) + prims
    c_all = _cost(cand, ref, (H, W), w)
    if c_all <= base + 0.3:
        shapes, base, added = cand, c_all, list(prims)
        # drop any new shape the result is better without
        for p in list(added):
            rest = [x for x in shapes if x is not p]
            c = _cost(rest, ref, (H, W), w)
            if c < base:
                shapes, base = rest, c
        return shapes
    # else one by one
    for p in prims:
        cand = _remove_covered(shapes, [p], w) + [p]
        c = _cost(cand, ref, (H, W), w)
        # a clean designer shape is worth a tiny, invisible pixel difference
        if c <= base + 0.8:
            shapes, base = cand, c
    return shapes


def _offset_polygon(mask, w, max_vertices=8):
    """A small polygonal hole (a triangle): its polygon pushed out by half a
    line width, corners kept sharp (miter)."""
    from .bezier import intersect
    from .geometry import polygon_shape
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=len)
    P = c[:, 0, :].astype(float)

    def dev(V):
        """max distance of the hole outline from the polygon's edge segments"""
        D = []
        for j in range(len(V)):
            a, b = V[j], V[(j + 1) % len(V)]
            ab = b - a
            t = np.clip((P - a) @ ab / (ab @ ab + 1e-9), 0, 1)
            D.append(np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
        return np.min(D, axis=0).max()
    best = None
    # fewest corners whose polygon really is the hole (a triangle before a
    # triangle with one blunted corner)
    for eps in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        V = cv2.approxPolyDP(c, eps * w * Z, True)[:, 0, :].astype(float)
        if 3 <= len(V) <= max_vertices and dev(V) <= (0.4 if len(V) <= 4 else 0.25) * w * Z:
            if best is None or len(V) < len(best):
                best = V
    if best is None:
        return None
    V = best
    # edges that bulge are arcs of a round shape, not polygon edges
    from .geometry import _bowed_edges
    idx = sorted(int(np.argmin(np.linalg.norm(P - v, axis=1))) for v in V)
    if _bowed_edges(P, idx, True, 0.12 * w * Z):
        return None
    if len(V) > 4:
        # a round hole is not a polygon: compare with a circle/ellipse fit
        from .geometry import fit_circle_ellipse
        e = fit_circle_ellipse((P + 0.5) / Z, 1.0, w, robust=False, min_cover=0.9)
        if e is not None:
            return None
    V = (V + 0.5) / Z
    area = 0.5 * np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1])
    sgn = 1 if area > 0 else -1
    lines = []
    for j in range(len(V)):
        a, b = V[j], V[(j + 1) % len(V)]
        dv = (b - a) / (np.linalg.norm(b - a) + 1e-12)
        if abs(dv[0]) > np.cos(np.radians(6)):
            dv = np.array([np.sign(dv[0]), 0.0])      # level edges stay level
            a = np.array([a[0], (a[1] + b[1]) / 2])
        elif abs(dv[1]) > np.cos(np.radians(6)):
            dv = np.array([0.0, np.sign(dv[1])])
            a = np.array([(a[0] + b[0]) / 2, a[1]])
        nrm = sgn * np.array([dv[1], -dv[0]])       # outward
        lines.append((a + nrm * w / 2, dv))
    out = []
    for j in range(len(V)):
        p0, d0 = lines[j - 1]
        p1, d1 = lines[j]
        x = intersect(p0, d0, p1, d1)
        if x is None:
            return None
        out.append(np.asarray(x, float))
    return polygon_shape(out, True)


def share_edges(shapes, w, max_pair=1.6):
    """Two polygons that sit back to back (the two halves of a compass
    needle) share one edge: the facing edges are replaced by one common
    edge (their end points averaged), drawn once, so nothing overlaps."""
    polys = []
    for i, (k, d) in enumerate(shapes):
        if k == "path" and all(sg[0] == "L" for sg in d[1]) and 3 <= len(d[1]) <= 6:
            polys.append((i, [np.asarray(sg[1], float) for sg in d[1]]))
    shapes = list(shapes)
    used = set()
    for ai in range(len(polys)):
        for bi in range(ai + 1, len(polys)):
            ia, A = polys[ai]
            ib, B = polys[bi]
            if ia in used or ib in used:
                continue
            best = None
            for ea in range(len(A)):
                a0, a1 = A[ea - 1], A[ea]
                for eb in range(len(B)):
                    b0, b1 = B[eb - 1], B[eb]
                    for flip in (False, True):
                        c0, c1 = (b1, b0) if flip else (b0, b1)
                        d = max(np.linalg.norm(a0 - c0), np.linalg.norm(a1 - c1))
                        if d < max_pair * w and (best is None or d < best[0]):
                            best = (d, ea, eb, flip)
            if best is None:
                continue
            _, ea, eb, flip = best
            a0, a1 = A[ea - 1], A[ea]
            b0, b1 = B[eb - 1], B[eb]
            c0, c1 = (b1, b0) if flip else (b0, b1)
            m0, m1 = (a0 + c0) / 2, (a1 + c1) / 2
            if np.linalg.norm(m1 - m0) < 1.5 * w:
                continue
            NA = [v.copy() for v in A]
            NA[ea - 1], NA[ea] = m0.copy(), m1.copy()
            NB = [v.copy() for v in B]
            if flip:
                NB[eb - 1], NB[eb] = m1.copy(), m0.copy()
            else:
                NB[eb - 1], NB[eb] = m0.copy(), m1.copy()
            # A keeps the shared edge (closed); B is drawn without it (open)
            shapes[ia] = ("path", (NA[-1].copy(), [("L", v.copy()) for v in NA]))
            n = len(NB)
            order = [NB[(eb + k) % n] for k in range(n)]
            shapes[ib] = ("open", (order[0].copy(), [("L", v.copy()) for v in order[1:]]))
            used |= {ia, ib}
    return shapes
