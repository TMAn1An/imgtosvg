"""Repeated parts are drawn once and copied, like a designer does.

Shapes that are the same up to a translation, a rotation by a multiple of
45 degrees and/or a mirror (the pins around a map, the windows of a building,
the arrows around a compass) are grouped.  The member that reproduces the ink
best everywhere, with the fewest anchors, is copied onto all positions, so
the copies are exactly identical.  Circles of nearly the same radius get the
same radius.
"""
import numpy as np

from .geometry import shape_points


def _resample(P, n=96):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-9:
        return np.repeat(P[:1], n, 0), 0.0
    t = np.linspace(0, d[-1], n)
    return np.c_[np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])], d[-1]


def _mats():
    out = []
    for k in range(8):
        a = k * np.pi / 4
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        out.append(("r%d" % k, R))
        out.append(("m%d" % k, R @ np.array([[-1.0, 0.0], [0.0, 1.0]])))
    return out


MATS = _mats()
STATS = []  # sizes of the groups that were unified (for tests / tuning)


def _nearest(A, B):
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return d.min(axis=1)


def _match(A, B, w):
    """Best transform M (B ~ M(A - cA) + cB): returns (M, err) or None."""
    ca, cb = A.mean(0), B.mean(0)
    A0, B0 = A - ca, B - cb
    best = None
    for name, M in MATS:
        T = A0 @ M.T
        # a couple of ICP steps for the translation only
        off = np.zeros(2)
        for _ in range(2):
            d = np.linalg.norm((T + off)[:, None, :] - B0[None, :, :], axis=2)
            off += (B0[d.argmin(1)] - (T + off)).mean(0)
        e1 = _nearest(T + off, B0)
        e2 = _nearest(B0, T + off)
        err = max(e1.max(), e2.max())
        if best is None or err < best[1]:
            best = (M, err, off, name)
    if best is None or best[1] > 0.65 * w:
        return None
    return best


def _transform_shape(shape, M, ca, cb):
    kind, data = shape
    f = lambda p: (np.asarray(p, float) - ca) @ M.T + cb
    start, segs = data
    return (kind, (f(start), [(sg[0],) + tuple(f(p) for p in sg[1:]) for sg in segs]))


def _anchors(shape):
    return len(shape[1][1])


def centre_error(work, z=4):
    """ink_error(shape): mean distance (work px) of the shape from the ink's
    centre line (skeleton)."""
    import cv2
    from scipy.ndimage import map_coordinates
    from skimage.morphology import skeletonize
    H, W = work.shape
    big = cv2.resize(work, (W * z, H * z), interpolation=cv2.INTER_CUBIC) > 0.5
    sk = skeletonize(big)
    D = cv2.distanceTransform((~sk).astype(np.uint8), cv2.DIST_L2, 5) / z

    def err(shape):
        P = shape_points(shape)
        P, _ = _resample(P, 80)
        v = map_coordinates(D, [P[:, 1] * z - 0.5, P[:, 0] * z - 0.5], order=1, mode="nearest")
        return float(v.mean())
    return err


def unify_repeats(shapes, work, w, ink_error=None):
    """ink_error(shape) -> mean distance of the shape from the ink centre."""
    ink_error = ink_error or centre_error(work)
    # circles: same radius when nearly the same
    circ = [i for i, (k, _) in enumerate(shapes) if k == "circle"]
    shapes = list(shapes)
    used = set()
    for i in circ:
        if i in used:
            continue
        grp = [j for j in circ if j not in used and abs(shapes[j][1][2] - shapes[i][1][2]) < 0.25 * w]
        if len(grp) > 1:
            r = float(np.median([shapes[j][1][2] for j in grp]))
            for j in grp:
                cx, cy, _ = shapes[j][1]
                shapes[j] = ("circle", (cx, cy, r))
        used.update(grp)

    idx = [i for i, (k, _) in enumerate(shapes) if k != "circle"]
    samp, length, closed = {}, {}, {}
    for i in idx:
        P = shape_points(shapes[i])
        samp[i], length[i] = _resample(P, 64)
        closed[i] = shapes[i][0] == "path"
    groups, seen = [], set()
    for a in idx:
        if a in seen or length[a] < 3 * w:
            continue
        grp = [(a, None)]
        for b in idx:
            if b == a or b in seen or closed[b] != closed[a]:
                continue
            if abs(length[b] - length[a]) > 0.2 * max(length[a], length[b]) + 0.5 * w:
                continue
            m = _match(samp[a], samp[b], w)
            if m is not None:
                grp.append((b, m))
        if len(grp) > 1:
            groups.append(grp)
            seen.update(g for g, _ in grp)
    if not groups:
        return shapes
    import cv2
    from .tracer import _render_shapes
    H, W = work.shape
    z = 3
    ink = cv2.resize(work, (W * z, H * z), interpolation=cv2.INTER_CUBIC) > 0.5
    pad = int(np.ceil(w * z))

    def window(i):
        P = samp[i] * z
        x0, y0 = np.floor(P.min(0)).astype(int) - pad
        x1, y1 = np.ceil(P.max(0)).astype(int) + pad
        return max(y0, 0), min(y1, H * z), max(x0, 0), min(x1, W * z)

    for grp in groups:
        members = [g for g, _ in grp]
        others = [shapes[i] for i in range(len(shapes)) if i not in members]
        base = _render_shapes(others, (H, W), z, w) > 0 if others else np.zeros_like(ink)
        wins = [window(m) for m in members]

        def wrong(placed):
            got = base | (_render_shapes(placed, (H, W), z, w) > 0)
            return sum((got[a:b, c:d] ^ ink[a:b, c:d]).sum() for a, b, c, d in wins) / (w * z) ** 2

        cur = wrong([shapes[m] for m in members])
        best = None
        for c in members:
            placed = []
            for m in members:
                if m == c:
                    placed.append(shapes[c])
                    continue
                mm = _match(samp[c], samp[m], w)
                if mm is None:
                    placed = None
                    break
                M, _, off, _ = mm
                placed.append(_transform_shape(shapes[c], M, samp[c].mean(0), samp[m].mean(0) + off))
            if placed is None:
                continue
            cost = wrong(placed) + 0.08 * len(members) * _anchors(shapes[c])
            if best is None or cost < best[0]:
                best = (cost, placed, c)
        if best is None:
            continue
        cur_cost = cur + 0.08 * sum(_anchors(shapes[m]) for m in members)
        # identical copies are worth a tiny, invisible loss of pixel accuracy
        if best[0] > cur_cost + 0.1 * len(members):
            continue
        for m, p in zip(members, best[1]):
            shapes[m] = p
        STATS.append(len(members))
    return shapes
