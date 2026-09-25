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


def tidy(shapes, w, max_chord=1.0, simplify_tol=0.12, bent=True):
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
            # a chain of straight pieces bending a little is one gentle curve
            if bent:
                start, segs = bent_lines_to_curve(start, segs, closed, w)
        except Exception:  # never lose a shape over a clean-up step
            start, segs = data
        out.append((kind, (start, segs)))
    return out


def bent_lines_to_curve(start, segs, closed, w, max_bend=16.0, tol=0.22):
    """Runs of 2+ straight pieces that each turn only a little (a traced
    gentle curve) become one cubic curve when it stays within tol*w."""
    from .bezier import fit_single
    ctrls = _to_ctrls(start, segs)
    n = len(ctrls)
    if n < 2:
        return start, segs
    cosb = np.cos(np.radians(max_bend))
    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
    out, i = [], 0
    while i < n:
        if not ctrls[i][1]:
            out.append(ctrls[i])
            i += 1
            continue
        j = i
        short = lambda k: np.linalg.norm(ctrls[k][0][3] - ctrls[k][0][0]) < 4.0 * w
        while j + 1 < n and ctrls[j + 1][1] and short(j) and short(j + 1) and \
                unit(ctrls[j][0][3] - ctrls[j][0][0]) @ unit(ctrls[j + 1][0][3] - ctrls[j + 1][0][0]) > cosb:
            j += 1
        # a traced curve is a chain of several short pieces; long straight
        # edges are real lines and stay
        if j >= i + 2:
            run = ctrls[i:j + 1]
            pts = np.vstack([np.linspace(c[0], c[3], 8)[:-1] for c, _ in run] + [run[-1][0][3][None]])
            t1 = unit(run[0][0][3] - run[0][0][0])
            t2 = -unit(run[-1][0][3] - run[-1][0][0])
            c, err = fit_single(pts, run[0][0][0], run[-1][0][3], t1, t2)
            ch = run[-1][0][3] - run[0][0][0]
            L = np.linalg.norm(ch)
            straight = L > 1e-9 and np.abs((pts - run[0][0][0]) @ np.array([-ch[1], ch[0]]) / L).max() < 0.1 * w
            if straight:
                A, B = run[0][0][0], run[-1][0][3]
                out.append([np.array([A, A + (B - A) / 3, A + 2 * (B - A) / 3, B]), True])
                i = j + 1
                continue
            if err.max() < tol * w:
                out.append([np.array(c, float), False])
                i = j + 1
                continue
        out.append(ctrls[i])
        i += 1
    s0, sg = _from_ctrls(out)
    if closed:
        sg[-1] = sg[-1][:-1] + (s0.copy(),)
    return s0, sg

