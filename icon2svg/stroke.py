"""Centerline ("stroke") vectorisation for line icons.

The icon is reduced to its skeleton, the skeleton is turned into a graph,
spurs are pruned, strokes that run straight through a junction are joined,
and every stroke is fitted with lines/Beziers. The result is drawn with a
single stroke width, round caps and round joins - the way line icons are
drawn by hand in Illustrator/Figma. Parts that are much thicker than the
stroke (dots, solid areas) are kept as filled outlines.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

from .bezier import unit, arc_to_cubics
from .tracer import fit_circle
from .tracer import extract_contours, fit_path

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _upsample(ink, s, opt):
    h, w = ink.shape
    up = int(np.clip(np.ceil(1000 / max(h, w)), 2, 8))
    big = cv2.resize(ink, (w * up, h * up), interpolation=cv2.INTER_CUBIC)
    sig = opt.blur * s * up
    if sig > 0.3:
        big = cv2.GaussianBlur(big, (0, 0), sig)
    return big, up


def _graph(sk):
    """Skeleton pixels -> (nodes, edges). Edges are lists of (r, c) pixels
    that start and end on node pixels; pure loops start and end on the same pixel."""
    H, W = sk.shape
    ys, xs = np.nonzero(sk)
    pix = set(zip(ys.tolist(), xs.tolist()))

    def nbrs(p):
        r, c = p
        return [(r + dr, c + dc) for dr, dc in NB if (r + dr, c + dc) in pix]

    deg = {p: len(nbrs(p)) for p in pix}
    node_pix = {p for p, d in deg.items() if d != 2}
    # cluster adjacent node pixels into one node
    node_id = {}
    nodes = []
    for p in node_pix:
        if p in node_id:
            continue
        stack = [p]
        node_id[p] = len(nodes)
        members = []
        while stack:
            q = stack.pop()
            members.append(q)
            for n in nbrs(q):
                if n in node_pix and n not in node_id:
                    node_id[n] = len(nodes)
                    stack.append(n)
        nodes.append(np.mean(members, axis=0))
    edges = []
    visited = set()  # visited degree-2 pixels
    for p in node_pix:
        for n in nbrs(p):
            if n in node_pix:
                continue
            if n in visited:
                continue
            path = [p, n]
            visited.add(n)
            prev, cur = p, n
            while True:
                nx = [q for q in nbrs(cur) if q != prev and q not in path[-3:-1]]
                nxt_node = [q for q in nx if q in node_pix]
                nx2 = [q for q in nx if q not in node_pix and q not in visited]
                if nx2:
                    q = nx2[0]
                elif nxt_node:
                    q = nxt_node[0]
                    path.append(q)
                    break
                else:
                    break
                visited.add(q)
                path.append(q)
                prev, cur = cur, q
            if path[-1] in node_pix:
                edges.append([node_id[path[0]], node_id[path[-1]], path])
    # loops without any node pixel
    for p in pix:
        if p in node_pix or p in visited:
            continue
        path = [p]
        visited.add(p)
        prev, cur = None, p
        while True:
            nx = [q for q in nbrs(cur) if q != prev and q not in visited]
            if not nx:
                break
            prev, cur = cur, nx[0]
            visited.add(cur)
            path.append(cur)
        if len(path) > 8:
            edges.append([-1, -1, path])
    return nodes, edges


def _length(path):
    a = np.asarray(path, float)
    return np.linalg.norm(np.diff(a, axis=0), axis=1).sum()


def _prune_and_merge(nodes, edges, spur):
    """Remove short spurs ending in free ends, then merge through degree-2 nodes."""
    changed = True
    while changed:
        changed = False
        deg = {}
        for a, b, _ in edges:
            if a >= 0:
                deg[a] = deg.get(a, 0) + 1
                deg[b] = deg.get(b, 0) + 1
        keep = []
        for e in edges:
            a, b, path = e
            if a >= 0 and a != b and (deg[a] == 1) != (deg[b] == 1) and _length(path) < spur:
                changed = True
                continue
            keep.append(e)
        edges = keep
        # merge edges through degree-2 nodes
        deg = {}
        for a, b, _ in edges:
            if a >= 0:
                deg[a] = deg.get(a, 0) + 1
                deg[b] = deg.get(b, 0) + 1
        for nid, d in deg.items():
            if d != 2:
                continue
            inc = [e for e in edges if e[0] == nid or e[1] == nid]
            if len(inc) != 2 or inc[0] is inc[1]:
                continue
            e1, e2 = inc
            p1 = e1[2] if e1[1] == nid else e1[2][::-1]
            s1 = e1[0] if e1[1] == nid else e1[1]
            p2 = e2[2] if e2[0] == nid else e2[2][::-1]
            s2 = e2[1] if e2[0] == nid else e2[0]
            edges = [e for e in edges if e is not e1 and e is not e2]
            if s1 == nid and s2 == nid:
                continue
            edges.append([s1, s2, p1 + p2[1:]])
            changed = True
            break
    return edges


def _end_dir(path, at_start, n):
    a = np.asarray(path, float)
    if not at_start:
        a = a[::-1]
    k = min(len(a) - 1, n)
    return unit(a[k] - a[0])  # pointing away from the node


def _join_through(nodes, edges, win):
    """At junctions, join pairs of strokes that continue almost straight."""
    changed = True
    while changed:
        changed = False
        inc = {}
        for i, (a, b, p) in enumerate(edges):
            if a < 0 or a == b:
                continue
            inc.setdefault(a, []).append((i, True))
            inc.setdefault(b, []).append((i, False))
        for nid, lst in inc.items():
            if len(lst) < 3:
                continue
            best = None
            for x in range(len(lst)):
                for y in range(x + 1, len(lst)):
                    (i, si), (j, sj) = lst[x], lst[y]
                    if i == j:
                        continue
                    d1 = _end_dir(edges[i][2], si, win)
                    d2 = _end_dir(edges[j][2], sj, win)
                    c = d1 @ d2
                    if c < -0.94 and (best is None or c < best[0]):
                        best = (c, i, si, j, sj)
            if best is None:
                continue
            _, i, si, j, sj = best
            ei, ej = edges[i], edges[j]
            p1 = ei[2][::-1] if si else ei[2]          # ends at node
            o1 = ei[1] if si else ei[0]
            p2 = ej[2] if sj else ej[2][::-1]          # starts at node
            o2 = ej[1] if sj else ej[0]
            new = [o1, o2, p1 + p2[1:]]
            edges = [e for k, e in enumerate(edges) if k not in (i, j)] + [new]
            changed = True
            break
    return edges


def deg_of(edges, nid):
    return sum((e[0] == nid) + (e[1] == nid) for e in edges)


def trace_strokes(work, opt, s):
    """Return (shapes_stroke, shapes_fill, stroke_width) in `work` pixel units."""
    big, up = _upsample(work, s, opt)
    B = big > opt.threshold
    dt = cv2.distanceTransform(B.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(B)
    if sk.sum() == 0:
        return [], [], 0
    # stroke width = ink area / centre-line length (robust to junction blobs)
    thin_sk = sk & (dt < 1.6 * np.median(dt[sk]))
    w = float(B[thin_sk.any() and slice(None) or slice(0)].sum() / max(sk.sum(), 1)) / up
    w = float(np.clip(w, 0.6 * 2 * np.median(dt[sk]) / up, 2 * np.median(dt[sk]) / up))
    r_up = w * up / 2

    # thick (solid) regions: survive an opening wider than the stroke
    kr = int(np.ceil(r_up * 2.2))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kr + 1, 2 * kr + 1))
    thick = cv2.morphologyEx(B.astype(np.uint8), cv2.MORPH_OPEN, ker) > 0
    fills = []
    if thick.sum() > (2 * kr) ** 2:
        # grow back to the real outline of the solid part
        grow = cv2.dilate(thick.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        solid = (grow & B).astype(np.float32)
        solid = cv2.resize(solid, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_AREA)
        for c in extract_contours(solid, opt, s):
            fills.append(fit_path(c, True, opt, s))
        sk = sk & ~cv2.dilate(thick.astype(np.uint8), ker).astype(bool)

    nodes, edges = _graph(sk)
    edges = _prune_and_merge(nodes, edges, spur=1.2 * r_up + 2 * up * s)
    edges = _join_through(nodes, edges, win=int(r_up * 1.5) + 2)

    # refine the width from anti-aliased coverage: ink area / centre-line length
    total_len = sum(_length(e[2]) for e in edges) / up
    if total_len > 0:
        cov = work.copy()
        if fills:
            solid_mask = cv2.resize(cv2.dilate(thick.astype(np.uint8), ker), (work.shape[1], work.shape[0]),
                                    interpolation=cv2.INTER_NEAREST) > 0
            cov[solid_mask] = 0
        area = cov[cov > 0.05].sum()
        # round caps add ~ w^2 * (1 - pi/4) per free end; ignore (small)
        free = sum(1 for e in edges if e[0] >= 0 for nd in (e[0], e[1]) if deg_of(edges, nd) == 1)
        w2 = area / (total_len + 0.5 * w * free)
        if 0.5 * w < w2 < 1.3 * w:
            w = w2

    import dataclasses
    extra = dict(opt.extra)
    extra.setdefault("keep_line", max(2.6, 1.5 * w / s))
    extra.setdefault("sharp_r", 0.42 * w / s)
    extra.setdefault("corner_k", max(0.9, 0.55 * w / s))
    sopt = dataclasses.replace(opt, tolerance=opt.tolerance * 1.6, corner_angle=max(opt.corner_angle, 60), extra=extra)
    deg = {}
    for a, b, _ in edges:
        if a >= 0:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
    rj = 1.3 * w / 2  # junction wobble radius (work px)

    def straighten(pts, at_start):
        """Replace the wobbly part of a stroke next to a junction by a straight run."""
        q = pts if at_start else pts[::-1]
        d = np.linalg.norm(q - q[0], axis=1)
        out = np.nonzero(d > rj)[0]
        if len(out) == 0 or out[0] < 2:
            return pts
        k = out[0]
        seg = np.linspace(q[0], q[k], k + 1)
        q = np.vstack([seg[:-1], q[k:]])
        return q if at_start else q[::-1]

    items = []  # (closed, pts)
    for e in edges:
        a, b, path = e[0], e[1], e[2]
        pts = np.asarray(path, float)[:, ::-1]  # (r,c) -> (x,y)
        pts = (pts + 0.5) / up
        closed = a < 0 or (a == b and len(path) > 8 and np.linalg.norm(pts[0] - pts[-1]) < 1.5 / up + 1e-6)
        if closed:
            if a >= 0:
                pts = pts[:-1]
            if len(pts) < 8:
                continue
            items.append((True, pts))
            continue
        # snap ends to node centres so strokes connect exactly
        if a >= 0 and a < len(nodes):
            pts[0] = (np.asarray(nodes[a])[::-1] + 0.5) / up
        if b >= 0 and b < len(nodes):
            pts[-1] = (np.asarray(nodes[b])[::-1] + 0.5) / up
        L = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        if L < 0.6 * s:
            continue
        if L > 3 * rj:
            if deg.get(a, 0) >= 3:
                pts = straighten(pts, True)
            if deg.get(b, 0) >= 3:
                pts = straighten(pts, False)
        items.append((False, pts))

    shapes, items = _circles(items, s, w, opt)
    # re-attach stroke ends to circles they touched before the circle was idealised
    circ = [d for k, d in shapes if k == "circle"]
    for _, pts in items:
        for e in (0, -1):
            for cx, cy, r in circ:
                v = pts[e] - np.array([cx, cy])
                dist = np.linalg.norm(v)
                if dist > 1e-6 and abs(dist - r) < 0.9 * w:
                    pts[e] = np.array([cx, cy]) + v / dist * r
    for closed, pts in items:
        shapes.append(fit_path(pts, closed, sopt, s))
    return shapes, fills, w


def _circles(items, s, w, opt):
    """Find strokes that lie on a common circle (a ring cut by junctions) and
    redraw them as one true circle; single clean arcs become exact arcs."""
    if not opt.detect_circles:
        return [], items
    cand = []
    for idx, (closed, pts) in enumerate(items):
        L = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        if L < 3 * s or closed:
            continue
        cx, cy, r, dev = fit_circle(pts)
        if not (1.5 * s < r < 60 * s):
            continue
        res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        # ignore the straightened ends near junctions
        core = res[len(res) // 8: len(res) - len(res) // 8] if len(res) > 16 else res
        if np.sqrt(np.mean(core ** 2)) > 0.12 * s + 0.012 * r or core.max() > 0.35 * s + 0.03 * r:
            continue
        ang = np.unwrap(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
        if abs(ang[-1] - ang[0]) < np.radians(15):
            continue
        cand.append([idx, np.array([cx, cy]), r, ang])
    # group: grow a circle from each candidate by absorbing strokes lying on it
    groups, taken = [], set()
    cand.sort(key=lambda c: -abs(c[3][-1] - c[3][0]) * c[2])
    for c in cand:
        if c[0] in taken:
            continue
        cen, r = c[1], c[2]
        members = [c]
        for _ in range(3):
            allp = np.vstack([items[m[0]][1] for m in members])
            cx, cy, r, _d = fit_circle(allp)
            cen = np.array([cx, cy])
            tol = 0.45 * s + 0.03 * r
            new = []
            for d in cand:
                if d[0] in taken:
                    continue
                pts = items[d[0]][1]
                res = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
                core = res[len(res) // 8: len(res) - len(res) // 8] if len(res) > 16 else res
                if core.max() < tol:
                    new.append(d)
            if not new:
                break
            members = new
        for m in members:
            m[1], m[2] = cen, r
            m[3] = np.unwrap(np.arctan2(items[m[0]][1][:, 1] - cen[1], items[m[0]][1][:, 0] - cen[0]))
        taken |= {m[0] for m in members}
        groups.append(members)
    shapes, used = [], set()
    for g in groups:
        bins = np.zeros(72, bool)
        for c in g:
            a = np.mod(c[3], 2 * np.pi)
            bins[(a / (2 * np.pi) * 72).astype(int) % 72] = True
        cover = bins.mean()
        if cover >= 0.85 or (len(g) >= 2 and cover >= 0.6):
            cen, r = g[0][1], g[0][2]
            shapes.append(("circle", (cen[0], cen[1], r)))
            used |= {c[0] for c in g}
        else:
            for c in g:
                idx, cen, r, ang = c
                if abs(ang[-1] - ang[0]) < np.radians(40):
                    continue
                pts = items[idx][1]
                segs = arc_to_cubics(cen, r, ang[0], ang[-1])
                start = cen + r * np.array([np.cos(ang[0]), np.sin(ang[0])])
                # keep exact end points (they connect to other strokes)
                if np.linalg.norm(start - pts[0]) > 0.6 * s or np.linalg.norm(segs[-1][3] - pts[-1]) > 0.6 * s:
                    continue
                shapes.append(("open", (start, segs)))
                used.add(idx)
    return shapes, [it for k, it in enumerate(items) if k not in used]
