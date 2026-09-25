"""Corner optimizer: where two straight lines meet, decide by looking at the
ink how the corner is really drawn.

Candidates for every corner (a sharp anchor, or a group of up to two short
pieces between two straight lines):
  * one sharp anchor where the two lines meet
  * a circular fillet (2 anchors) of several radii
  * what the tracer produced
Each candidate is rendered locally (a round-capped line of the stroke width)
and compared with the ink around the corner; the best match wins, with a
small bonus for fewer anchors.  A traced handle drawn as "line, two slanted
chamfers, line" thus becomes a clean quarter circle, and a rounded rectangle
drawn with sharp corners gets its rounding back.
"""
import cv2
import numpy as np

from .bezier import _seg_ctrl, bez, intersect

Z = 4            # render zoom for the local comparison
ANCHOR_COST = 0.06   # in units of (line width)^2 of wrong ink


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _render_local(polys, box, w):
    """Mask of round-capped strokes (width w) of the given polylines, in the
    window box=(x0, y0, x1, y1) at zoom Z."""
    x0, y0, x1, y1 = box
    Wp, Hp = int((x1 - x0) * Z) + 1, int((y1 - y0) * Z) + 1
    img = np.full((Hp, Wp), 255, np.uint8)
    for P in polys:
        q = np.round((np.asarray(P) - [x0, y0]) * Z * 16).astype(np.int32)
        cv2.polylines(img, [q], False, 0, 1, cv2.LINE_8, 4)
    d = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    return d <= w * Z / 2


def _ink_local(big, box):
    x0, y0, x1, y1 = box
    a = big[int(y0 * Z):int(y0 * Z) + int((y1 - y0) * Z) + 1, int(x0 * Z):int(x0 * Z) + int((x1 - x0) * Z) + 1]
    return a > 0.5


def _fillet(V, u1, u2, r):
    """Arc of radius r tangent to the rays V+t*u1 and V+t*u2 (u pointing away
    from the corner).  Returns (P1, c1, c2, P2) as a cubic, P1 on ray 1."""
    phi = np.arccos(np.clip(u1 @ u2, -1, 1))
    t = r / np.tan(phi / 2)
    P1, P2 = V + t * u1, V + t * u2
    theta = np.pi - phi
    h = 4 / 3 * np.tan(theta / 4) * r
    return P1, P1 - u1 * h, P2 - u2 * h, P2, t


def _poly_of(ctrls):
    pts = []
    for c, ln in ctrls:
        seg = np.array([c[0], c[3]]) if ln else bez(c, np.linspace(0, 1, 16))
        pts.extend(seg if not pts else seg[1:])
    return np.array(pts)


def optimize_corners(shapes, work, w, max_group=2):
    H, W = work.shape
    big = cv2.resize(work, (W * Z, H * Z), interpolation=cv2.INTER_CUBIC)
    out = []
    for kind, data in shapes:
        if kind == "circle":
            out.append((kind, data))
            continue
        try:
            out.append((kind, _optimize_path(kind, data, big, (W, H), w, max_group)))
        except Exception:  # never lose a shape over an optional step
            out.append((kind, data))
    return out


def _optimize_path(kind, data, big, size, w, max_group):
    closed = kind == "path"
    start, segs = data
    ctrls, cur = [], np.asarray(start, float)
    for sg in segs:
        c, ln = _seg_ctrl(cur, sg)
        ctrls.append([np.array(c, float), sg[0] == "L"])
        cur = c[3]
    n = len(ctrls)
    is_long = [it[1] and np.linalg.norm(it[0][3] - it[0][0]) >= 2.5 * w for it in ctrls]
    longs = [i for i in range(n) if is_long[i]]
    if len(longs) < 2 and not (closed and len(longs) == 1):
        return start, segs
    pairs = list(zip(longs, longs[1:]))
    if closed:
        pairs.append((longs[-1], longs[0] + n))
    new_start = {i: ctrls[i][0][0].copy() for i in longs}
    new_end = {i: ctrls[i][0][3].copy() for i in longs}
    replace = {}   # index of line a -> (middle indices, corner items)
    for ja, kb in pairs:
        mids = [m % n for m in range(ja + 1, kb)]
        if len(mids) > max_group or ja == kb % n:
            continue
        if any(np.linalg.norm(ctrls[m][0][3] - ctrls[m][0][0]) >= 4.0 * w for m in mids):
            continue
        k = kb % n
        res = _best_corner(ctrls[ja], [ctrls[m] for m in mids], ctrls[k], big, size, w)
        if res is None:
            continue
        end_a, corner, start_b = res
        new_end[ja] = end_a
        new_start[k] = start_b
        replace[ja] = (mids, corner)
    if not replace:
        return start, segs
    skip = set(m for mids, _ in replace.values() for m in mids)
    items = []
    for i in range(n):
        if i in skip:
            continue
        if i in new_start:
            p0, p1 = new_start[i], new_end[i]
            items.append(("L", p0, p1))
        else:
            items.append(("X", ctrls[i]))
        if i in replace:
            for it in replace[i][1]:
                items.append(("X", it))
    # rebuild as (start, segs); straight lines are re-anchored to the new points
    first = items[0]
    start = (first[1] if first[0] == "L" else first[1][0][0]).copy()
    out = []
    for it in items:
        if it[0] == "L":
            out.append(("L", it[2].copy()))
        else:
            c, ln = it[1]
            out.append(("L", c[3].copy()) if ln else ("C", c[1].copy(), c[2].copy(), c[3].copy()))
    if closed:
        # the wrap-around corner may have moved the start of the first line
        last = out[-1]
        out[-1] = (last[0],) + tuple(last[1:-1]) + (start.copy(),)
    return start, out


def _best_corner(a, mids, b, big, size, w):
    """a, b: long line items (a ends at the corner, b starts after it)."""
    A0, A1 = a[0][0], a[0][3]
    B0, B1 = b[0][0], b[0][3]
    da, db = _unit(A1 - A0), _unit(B1 - B0)
    if abs(da[0] * db[1] - da[1] * db[0]) < 0.17:   # (almost) parallel: no corner
        return None
    V = intersect(A1, da, B0, db)
    if V is None:
        return None
    V = np.asarray(V, float)
    u1, u2 = -da, db                  # rays away from the corner
    La = (V - A0) @ da                # room along each line
    Lb = (B1 - V) @ db
    if La < 0.5 * w or Lb < 0.5 * w:
        return None
    phi = np.arccos(np.clip(u1 @ u2, -1, 1))
    if phi < np.radians(20) or phi > np.radians(165):
        return None
    if np.linalg.norm(V - (A1 + B0) / 2) > 4 * w:
        return None
    # local window
    reach = min(max(La, Lb), 6 * w)
    R = reach + w
    box = (max(V[0] - R, 0), max(V[1] - R, 0), min(V[0] + R, size[0] - 1), min(V[1] + R, size[1] - 1))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    ink = _ink_local(big, box)
    ea = V + u1 * min(La, reach)      # far points of the two lines in the window
    eb = V + u2 * min(Lb, reach)

    unit_area = (w * Z) ** 2

    def score(poly):
        """minus the wrong ink area, in (line width)^2"""
        m = _render_local([poly], box, w)
        hh, ww = min(m.shape[0], ink.shape[0]), min(m.shape[1], ink.shape[1])
        return -(m[:hh, :ww] ^ ink[:hh, :ww]).sum() / unit_area

    # current geometry, clipped to the same window
    cur_items = [a] + mids + [b]
    cur_poly = _poly_of(cur_items)
    keep = np.linalg.norm(cur_poly - V, axis=1) <= reach + 1e-6
    cur_score = score(cur_poly[keep]) if keep.sum() >= 2 else -1e9
    cands = [(cur_score - ANCHOR_COST * (len(mids) + 1), "cur", None)]
    # sharp corner
    s_sharp = score(np.array([ea, V, eb]))
    cands.append((s_sharp - ANCHOR_COST, "sharp", None))
    tmax = 0.9 * min(La, Lb)
    radii = [f * w for f in (0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0)]
    if phi < np.radians(70):
        radii = []   # sharp tips (arrow heads) stay one anchor, like a designer draws them
    for r in radii:
        t = r / np.tan(phi / 2)
        if t > tmax:
            break
        P1, c1, c2, P2, _ = _fillet(V, u1, u2, r)
        arc = bez(np.array([P1, c1, c2, P2]), np.linspace(0, 1, 16))
        poly = np.vstack([[ea], arc, [eb]])
        cands.append((score(poly) - 2 * ANCHOR_COST, "fillet", (P1, c1, c2, P2)))
    best = max(cands, key=lambda c: c[0])
    if best[1] == "cur":
        return None
    if best[1] == "sharp":
        return V.copy(), [], V.copy()
    P1, c1, c2, P2 = best[2]
    return P1, [[np.array([P1, c1, c2, P2]), False]], P2


def straighten_curves(shapes, work, w, max_dev=0.6):
    """A curve whose ink is really straight becomes a straight line lying on
    the ink (a designer draws arrow heads and shafts with straight lines).
    The ink centre is measured across the curve; if those centre points lie
    on a line, the segment becomes that line and its end anchors move onto
    it."""
    from .refine import _scan
    from .tidy import _from_ctrls, _set_end, _set_start, _to_ctrls
    H, W = work.shape
    big = cv2.resize(work, (W * Z, H * Z), interpolation=cv2.INTER_CUBIC)
    out = []
    for kind, data in shapes:
        if kind == "circle":
            out.append((kind, data))
            continue
        closed = kind == "path"
        ctrls = _to_ctrls(*data)
        n = len(ctrls)
        changed = False
        for j in range(n):
            c, ln = ctrls[j]
            if ln:
                if _axis_line(ctrls, j, n, closed, work, w, big, _scan, _set_end, _set_start):
                    changed = True
                continue
            ch = c[3] - c[0]
            L = np.linalg.norm(ch)
            if L < 2.0 * w:
                continue
            nrm = np.array([-ch[1], ch[0]]) / L
            t = np.linspace(0.1, 0.9, 17)
            P = bez(c, t)
            if np.abs((P - c[0]) @ nrm).max() > max_dev * w:
                continue
            D = np.gradient(P, axis=0)
            D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-12
            N = np.c_[-D[:, 1], D[:, 0]]
            off = _scan(work, P, N, w)
            ok = np.isfinite(off)
            if ok.sum() < 9:
                continue
            Q = P[ok] + off[ok, None] * N[ok]
            m = Q.mean(0)
            u, sv, vt = np.linalg.svd(Q - m)
            d = vt[0]
            res = (Q - m) @ np.array([-d[1], d[0]])
            if np.sqrt(np.mean(res ** 2)) > 0.12 * w or np.abs(res).max() > 0.3 * w:
                continue
            proj = lambda x: m + ((x - m) @ d) * d
            A, B = proj(c[0]), proj(c[3])
            if np.linalg.norm(A - c[0]) > 0.8 * w or np.linalg.norm(B - c[3]) > 0.8 * w:
                continue
            idx = [k % n for k in (j - 1, j, j + 1) if closed or 0 <= k < n]
            before = [[it[0].copy(), it[1]] for it in (ctrls[k] for k in idx)]
            old_poly = _poly_of([ctrls[k] for k in idx])
            ctrls[j] = [np.array([A, A + (B - A) / 3, A + 2 * (B - A) / 3, B]), True]
            if j > 0 or closed:
                _set_end(ctrls[(j - 1) % n], A)
            if j < n - 1 or closed:
                _set_start(ctrls[(j + 1) % n], B)
            new_poly = _poly_of([ctrls[k] for k in idx])
            # keep it only if the drawing does not match the ink worse there
            both = np.vstack([old_poly, new_poly])
            box = (max(both.min(0)[0] - w, 0), max(both.min(0)[1] - w, 0),
                   min(both.max(0)[0] + w, W - 1), min(both.max(0)[1] + w, H - 1))
            if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                ink = _ink_local(big, box)

                def wrong(poly):
                    mk = _render_local([poly], box, w)
                    hh, ww = min(mk.shape[0], ink.shape[0]), min(mk.shape[1], ink.shape[1])
                    return (mk[:hh, :ww] ^ ink[:hh, :ww]).sum() / (w * Z) ** 2
                if wrong(new_poly) > wrong(old_poly) + 0.1:
                    for k, it in zip(idx, before):
                        ctrls[k] = it
                    continue
            changed = True
        if changed:
            start, segs = _from_ctrls(ctrls)
            if closed:
                segs[-1] = segs[-1][:-1] + (start.copy(),)
            out.append((kind, (start, segs)))
        else:
            out.append((kind, data))
    return out


def _axis_line(ctrls, j, n, closed, work, w, big, _scan, _set_end, _set_start, max_deg=12.0):
    """A straight line a few degrees off vertical/horizontal becomes exactly
    vertical/horizontal on the ink centre, if the ink agrees."""
    H, W = work.shape
    c = ctrls[j][0]
    A, B = c[0], c[3]
    d = B - A
    L = np.linalg.norm(d)
    if L < 1.5 * w:
        return False
    ang = np.degrees(np.arctan2(abs(d[1]), abs(d[0])))
    if 0.3 < ang < max_deg:
        ax = 1          # nearly horizontal: make y equal
    elif 0.3 < 90 - ang < max_deg:
        ax = 0          # nearly vertical: make x equal
    else:
        return False
    t = np.linspace(0.15, 0.85, 13)
    P = A + t[:, None] * d
    u = d / L
    N = np.tile(np.array([-u[1], u[0]]), (len(t), 1))
    off = _scan(work, P, N, w)
    ok = np.isfinite(off)
    if ok.sum() < 7:
        return False
    Q = P[ok] + off[ok, None] * N[ok]
    if np.std(Q[:, ax]) > 0.2 * w:      # the ink itself is slanted
        return False
    v = float(np.median(Q[:, ax]))
    A2, B2 = A.copy(), B.copy()
    A2[ax] = v
    B2[ax] = v
    if max(abs(A2[ax] - A[ax]), abs(B2[ax] - B[ax])) > 0.9 * w:
        return False
    idx = [k % n for k in (j - 1, j, j + 1) if closed or 0 <= k < n]
    before = [[ctrls[k][0].copy(), ctrls[k][1]] for k in idx]
    old_poly = _poly_of([ctrls[k] for k in idx])
    ctrls[j] = [np.array([A2, A2 + (B2 - A2) / 3, A2 + 2 * (B2 - A2) / 3, B2]), True]
    if j > 0 or closed:
        _set_end(ctrls[(j - 1) % n], A2)
    if j < n - 1 or closed:
        _set_start(ctrls[(j + 1) % n], B2)
    new_poly = _poly_of([ctrls[k] for k in idx])
    both = np.vstack([old_poly, new_poly])
    box = (max(both.min(0)[0] - w, 0), max(both.min(0)[1] - w, 0),
           min(both.max(0)[0] + w, W - 1), min(both.max(0)[1] + w, H - 1))
    ink = _ink_local(big, box)

    def wrong(poly):
        mk = _render_local([poly], box, w)
        hh, ww = min(mk.shape[0], ink.shape[0]), min(mk.shape[1], ink.shape[1])
        return (mk[:hh, :ww] ^ ink[:hh, :ww]).sum() / (w * Z) ** 2
    if wrong(new_poly) > wrong(old_poly) + 0.1:
        for k, it in zip(idx, before):
            ctrls[k] = it
        return False
    return True


def merge_double_lines(shapes, w, max_deg=10.0, max_gap=0.6, min_overlap=0.6):
    """Two shapes that share an edge (two faces of the icon meeting at one
    stroke) each carry a copy of that edge.  When the copies are almost on top
    of each other, the shorter one is put exactly onto the longer one, so the
    stroke is not drawn twice, thick and crooked."""
    from .tidy import _from_ctrls, _set_end, _set_start, _to_ctrls
    paths = []
    for kind, data in shapes:
        paths.append((kind, _to_ctrls(*data) if kind != "circle" else None, data))
    lines = []
    for a, (kind, ctrls, _) in enumerate(paths):
        if ctrls is None:
            continue
        for j, (c, ln) in enumerate(ctrls):
            if ln and np.linalg.norm(c[3] - c[0]) > 1.5 * w:
                lines.append((a, j))
    cos_lim = np.cos(np.radians(max_deg))
    touched = set()
    for x in range(len(lines)):
        for y in range(len(lines)):
            a, i = lines[x]
            b, j = lines[y]
            if a == b or (b, j) in touched or (a, i) in touched:
                continue
            ca, cb = paths[a][1][i][0], paths[b][1][j][0]
            La, Lb = np.linalg.norm(ca[3] - ca[0]), np.linalg.norm(cb[3] - cb[0])
            # b is the copy that moves: a horizontal/vertical copy always stays,
            # otherwise the longer one stays
            axis = lambda c: min(abs(c[3][0] - c[0][0]), abs(c[3][1] - c[0][1])) < 1e-3
            if axis(cb) and not axis(ca):
                continue
            if Lb > La and not (axis(ca) and not axis(cb)):
                continue
            u = (ca[3] - ca[0]) / La
            v = (cb[3] - cb[0]) / Lb
            if abs(u @ v) < cos_lim:
                continue
            nrm = np.array([-u[1], u[0]])
            d0, d1 = (cb[0] - ca[0]) @ nrm, (cb[3] - ca[0]) @ nrm
            if max(abs(d0), abs(d1)) > max_gap * w:
                continue
            s0, s1 = sorted([(cb[0] - ca[0]) @ u, (cb[3] - ca[0]) @ u])
            overlap = min(s1, La) - max(s0, 0)
            if overlap < min_overlap * Lb:
                continue
            P0 = cb[0] - d0 * nrm
            P1 = cb[3] - d1 * nrm
            kind_b, ctrls_b, _ = paths[b]
            n = len(ctrls_b)
            closed = kind_b == "path"
            ctrls_b[j] = [np.array([P0, P0 + (P1 - P0) / 3, P0 + 2 * (P1 - P0) / 3, P1]), True]
            if j > 0 or closed:
                _set_end(ctrls_b[(j - 1) % n], P0)
            if j < n - 1 or closed:
                _set_start(ctrls_b[(j + 1) % n], P1)
            touched.add((b, j))
    out = []
    for kind, ctrls, data in paths:
        if ctrls is None:
            out.append((kind, data))
            continue
        start, segs = _from_ctrls(ctrls)
        if kind == "path":
            segs[-1] = segs[-1][:-1] + (start.copy(),)
        out.append((kind, (start, segs)))
    return out
