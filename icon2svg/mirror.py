"""Mirror symmetry in vector form.

The raster pass (tracer.symmetrize) makes the two halves of a symmetric icon
the same pixels, but the tracer can still split and fit them differently (a
hand on the left drawn one way, the same hand on the right another way).  A
designer draws one half and mirrors it.  Here: shapes of one half are copied
onto the other half wherever the ink is mirror-symmetric, and the other half's
shapes that the copies already draw are dropped.  Both directions (left->right
and right->left) are tried and the result is kept only if it matches the ink
as well as before.
"""
import numpy as np
from scipy.spatial import cKDTree

from .geometry import shape_points


def find_axis(ink):
    from .tracer import _mirror
    h, w = ink.shape
    cols = ink.sum(0)
    if cols.sum() <= 0:
        return 0.0, 0.0
    cx = (cols * (np.arange(w) + 0.5)).sum() / cols.sum()
    best = (-1.0, cx)
    for c in np.arange(cx - 0.12 * w, cx + 0.12 * w, 0.25):
        m = _mirror(ink, c)
        sc = np.minimum(ink, m).sum() / max(np.maximum(ink, m).sum(), 1e-6)
        if sc > best[0]:
            best = (sc, c)
    for c in np.arange(best[1] - 0.25, best[1] + 0.25, 0.05):
        m = _mirror(ink, c)
        sc = np.minimum(ink, m).sum() / max(np.maximum(ink, m).sum(), 1e-6)
        if sc > best[0]:
            best = (sc, c)
    return best


def _mirror_shape(shape, c):
    kind, data = shape
    f = lambda p: np.array([2 * c - p[0], p[1]], float)
    if kind == "circle":
        cx, cy, r = data
        return (kind, (2 * c - cx, cy, r))
    start, segs = data
    return (kind, (f(start), [(sg[0],) + tuple(f(p) for p in sg[1:]) for sg in segs]))


def _on_ink(P, ink, frac=0.9):
    from scipy.ndimage import map_coordinates
    v = map_coordinates(ink, [P[:, 1] - 0.5, P[:, 0] - 0.5], order=1, mode="constant")
    return (v > 0.35).mean() >= frac


def _covered(P, tree, w):
    if tree is None:
        return False
    d, _ = tree.query(P)
    return (d < 0.6 * w).mean() >= 0.95


def _variant(shapes, pts, c, work, w, master_left):
    side = lambda P: (P[:, 0].max() < c + 0.3 * w) if master_left else (P[:, 0].min() > c - 0.3 * w)
    other = lambda P: (P[:, 0].min() > c - 0.3 * w) if master_left else (P[:, 0].max() < c + 0.3 * w)
    masters = [i for i, P in enumerate(pts) if side(P) and not other(P)]
    copies = []
    for i in masters:
        m = _mirror_shape(shapes[i], c)
        if _on_ink(shape_points(m, 20), work):
            copies.append(m)
    if not copies:
        return None
    tree = cKDTree(np.vstack([shape_points(m, 20) for m in copies]))
    out = [shapes[i] for i in range(len(shapes)) if not (other(pts[i]) and not side(pts[i]))]
    # the other half's shapes that the copies do not draw stay
    for i, P in enumerate(pts):
        if other(P) and not side(P) and not _covered(P, tree, w):
            out.append(shapes[i])
    # shapes across the axis that the copies and the rest already draw go
    keep = []
    for k, sh in enumerate(out):
        P = shape_points(sh, 20)
        if P[:, 0].min() < c - 0.3 * w < c + 0.3 * w < P[:, 0].max():
            rest = [x for j, x in enumerate(out) if j != k] + copies
            if rest and _covered(P, cKDTree(np.vstack([shape_points(x, 20) for x in rest])), w):
                continue
        keep.append(sh)
    return keep + copies


def _cost(shapes, work, ref, w, z=3):
    from .tracer import _render_shapes, count_anchors
    H, W = work.shape
    got = _render_shapes(shapes, (H, W), z, w) > 0
    return (got ^ ref).sum() / (w * z) ** 2 + 0.08 * count_anchors(shapes)


def mirror_shapes(shapes, work, w, min_score=0.85):
    import cv2
    score, c = find_axis(work)
    if score < min_score or not shapes:
        return shapes
    H, W = work.shape
    ref = cv2.resize(work, (W * 3, H * 3), interpolation=cv2.INTER_CUBIC) > 0.5
    pts = [shape_points(sh, 20) for sh in shapes]
    base = _cost(shapes, work, ref, w)
    best = (base + 0.5, shapes)   # symmetric output is worth a tiny, invisible loss
    for master_left in (True, False):
        v = _variant(shapes, pts, c, work, w, master_left)
        if v is None:
            continue
        cst = _cost(v, work, ref, w)
        if cst < best[0]:
            best = (cst, v)
    return best[1]
