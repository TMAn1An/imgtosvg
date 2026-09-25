"""Final anchor clean-up, the way a designer tidies a traced path.

* Two anchors closer than a line's width are one anchor too many: with round
  line joins a tiny 2-anchor rounded corner looks exactly like a sharp corner,
  so it becomes one corner anchor where the two neighbouring directions meet
  (or one anchor in the middle when they do not meet).
* Smooth anchors whose neighbouring curves can be replaced by one curve
  without visibly moving the line are removed (Illustrator "Simplify").
"""
import numpy as np

from .bezier import _seg_ctrl, intersect, simplify


def _dir_in(c):
    for h in (c[2], c[1], c[0]):
        d = c[3] - h
        if np.linalg.norm(d) > 1e-6:
            return d / np.linalg.norm(d)
    return None


def _dir_out(c):
    for h in (c[1], c[2], c[3]):
        d = h - c[0]
        if np.linalg.norm(d) > 1e-6:
            return d / np.linalg.norm(d)
    return None


def _to_ctrls(start, segs):
    out, cur = [], np.asarray(start, float)
    for sg in segs:
        c, is_line = _seg_ctrl(cur, sg)
        out.append([np.array(c, float), sg[0] == "L"])
        cur = c[3]
    return out


def _from_ctrls(ctrls):
    segs = [("L", c[3].copy()) if ln else ("C", c[1].copy(), c[2].copy(), c[3].copy()) for c, ln in ctrls]
    return ctrls[0][0][0].copy(), segs


def _set_end(item, X):
    c, ln = item
    d = X - c[3]
    c[3] = X.copy()
    c[2] = c[2] + d if not ln else c[0] + 2 * (X - c[0]) / 3
    if ln:
        c[1] = c[0] + (X - c[0]) / 3


def _set_start(item, X):
    c, ln = item
    d = X - c[0]
    c[0] = X.copy()
    c[1] = c[1] + d if not ln else X + (c[3] - X) / 3
    if ln:
        c[2] = X + 2 * (c[3] - X) / 3


def collapse_short(start, segs, closed, w, max_chord):
    """Merge anchor pairs closer than max_chord (see module doc)."""
    ctrls = _to_ctrls(start, segs)
    while True:
        n = len(ctrls)
        if n <= (3 if closed else 1):
            break
        best = None
        for j in range(n):
            c = ctrls[j][0]
            L = np.linalg.norm(c[3] - c[0])
            if L < max_chord and (best is None or L < best[0]):
                best = (L, j)
        if best is None:
            break
        j = best[1]
        if not closed and j in (0, n - 1):
            # free end of an open stroke: keep the end point (it usually meets
            # another shape) and drop the anchor just before it
            if j == n - 1:
                _set_end(ctrls[n - 2], ctrls[n - 1][0][3])
            else:
                _set_start(ctrls[1], ctrls[0][0][0])
            del ctrls[j]
            continue
        prev, nxt = ctrls[(j - 1) % n], ctrls[(j + 1) % n]
        A, B = ctrls[j][0][0], ctrls[j][0][3]
        mid = (A + B) / 2
        X = mid
        di, do = _dir_in(prev[0]), _dir_out(nxt[0])
        if di is not None and do is not None and abs(di[0] * do[1] - di[1] * do[0]) > 0.25:
            P = intersect(A, di, B, do)
            if P is not None and np.linalg.norm(P - mid) < 0.9 * w:
                X = np.asarray(P, float)
        _set_end(prev, X)
        _set_start(nxt, X)
        del ctrls[j]
    return _from_ctrls(ctrls)


def tidy(shapes, w, max_chord=1.0, simplify_tol=0.12):
    """w: line width in the shapes' units."""
    out = []
    for kind, data in shapes:
        if kind == "circle":
            out.append((kind, data))
            continue
        closed = kind == "path"
        start, segs = data
        try:
            start, segs = collapse_short(start, segs, closed, w, max_chord * w)
            segs = simplify(start, segs, closed, simplify_tol * w, keep_line=1.5 * w)
        except Exception:  # never lose a shape over a clean-up step
            start, segs = data
        out.append((kind, (start, segs)))
    return out
