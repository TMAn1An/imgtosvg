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
from .geometry import primitive_closed, primitive_open, fit_circle_ellipse, _err
from .regularize import regularize
from .symfit import fit_symmetric_closed
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


def _faces(edges, r_up):
    """Faces of the planar stroke graph. Returns [(edge_ids, polyline(r,c), signed_area)]."""
    out = {}
    for i, (a, b, p) in enumerate(edges):
        if a < 0 or a == b:
            continue
        for fwd in (True, False):
            P = np.asarray(p if fwd else p[::-1], float)
            node = a if fwd else b
            d = np.linalg.norm(P - P[0], axis=1)
            k = np.nonzero(d > 2 * r_up)[0]
            k = k[0] if len(k) else len(P) - 1
            v = P[k] - P[0]
            out.setdefault(node, []).append((np.arctan2(v[0], v[1]), (i, fwd)))
    for n in out:
        out[n].sort(key=lambda t: t[0])
    pos = {he: idx for n, lst in out.items() for idx, (_, he) in enumerate(lst)}
    seen, faces = set(), []
    for n, lst in out.items():
        for _, he in lst:
            if he in seen:
                continue
            face, h = [], he
            while h not in seen:
                seen.add(h)
                face.append(h)
                i, fwd = h
                a, b, _ = edges[i]
                v = b if fwd else a
                lv = out[v]
                h = lv[(pos[(i, not fwd)] - 1) % len(lv)][1]
            # drop dangling edges (walked both ways inside the same face)
            ids = [i for i, _ in face]
            face = [h for h in face if ids.count(h[0]) == 1]
            if len(face) < 1:
                continue
            pts = []
            for i, fwd in face:
                p = edges[i][2] if fwd else edges[i][2][::-1]
                pts.extend(p if not pts else p[1:])
            P = np.asarray(pts, float)
            area = 0.5 * (np.dot(P[:, 1], np.roll(P[:, 0], -1)) - np.dot(P[:, 0], np.roll(P[:, 1], -1)))
            faces.append((sorted({i for i, _ in face}), P, area))
    return faces


def _chain_cycle(edges, ids):
    """Order edges `ids` into one closed polyline (r,c) or None."""
    ids = list(ids)
    if not ids:
        return None
    first = ids.pop(0)
    a, b, p = edges[first]
    pts = list(p)
    start, cur = a, b
    while ids:
        for k, i in enumerate(ids):
            ea, eb, ep = edges[i]
            if ea == cur:
                pts += list(ep[1:]); cur = eb; break
            if eb == cur:
                pts += list(ep[::-1][1:]); cur = ea; break
        else:
            return None
        ids.pop(k)
    if cur != start:
        return None
    return np.asarray(pts, float)


def _simple_cycles(edges, max_len=8, max_cycles=3000, max_extent=1e9):
    """Simple cycles of the stroke graph (as lists of edge ids), each once."""
    adj = {}
    box = {}
    for i, (a, b, p) in enumerate(edges):
        if a < 0 or a == b:
            continue
        adj.setdefault(a, []).append((i, b))
        adj.setdefault(b, []).append((i, a))
        P = np.asarray(p, float)
        box[i] = (P.min(0), P.max(0))
    seen, out = set(), []

    def dfs(s0, node, path, visited, lo, hi):
        if len(out) >= max_cycles:
            return
        for i, nb in adj[node]:
            if path and i == path[-1]:
                continue
            nlo, nhi = np.minimum(lo, box[i][0]), np.maximum(hi, box[i][1])
            if (nhi - nlo).max() > max_extent:
                continue
            if nb == s0:
                key = frozenset(path + [i])
                if len(key) == len(path) + 1 and key not in seen:
                    seen.add(key)
                    out.append(path + [i])
                continue
            if nb < s0 or nb in visited or len(path) + 1 >= max_len:
                continue
            dfs(s0, nb, path + [i], visited | {nb}, nlo, nhi)

    for s0 in sorted(adj):
        dfs(s0, s0, [], {s0}, np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf]))
    return out


def _cycle_primitives(edges, up, s, w, extent, work=None):
    """Whole round shapes (circle, ellipse, map pin) that other strokes cross
    or touch: found as cycles of the graph, not as the regions the crossings
    cut them into.  A shape owns only the edges that lie on it (a pin tip
    that dips into its base ellipse belongs to the pin, the ellipse just runs
    over it), and every edge has at most one owner, so nothing is drawn twice.
    Returns (shapes, consumed edge ids)."""
    from scipy.spatial import cKDTree
    from .geometry import fit_pin, shape_points
    to_pts = lambda P: (P[:, ::-1] + 0.5) / up
    epts = {i: to_pts(np.asarray(p, float)) for i, (a, b, p) in enumerate(edges)}
    cands = []
    for ids in _simple_cycles(edges, max_extent=extent * up):
        Q = _chain_cycle(edges, ids)
        if Q is None or len(Q) < 12:
            continue
        pts = to_pts(Q)
        hull = cv2.convexHull(pts.astype(np.float32))
        area = abs(cv2.contourArea(pts.astype(np.float32)))
        convex = area >= 0.85 * cv2.contourArea(hull)
        if area < (2 * w) ** 2 or area < 0.6 * cv2.contourArea(hull):
            continue     # round shapes are (almost) convex
        fits = []
        e = fit_circle_ellipse(pts, s, w, robust=True, min_cover=0.9) if convex else None
        if e is not None:
            fits.append(e)
        elif convex:
            pn = fit_pin(pts, s, w)
            if pn is not None:
                fits.append(pn)
        if not fits and len(ids) >= 3:
            # an ellipse with a deep dent: another shape's tip reaching into it
            # (a pin resting on its base).  Fit without the dent edge; the dent
            # must lie inside the ellipse.
            for k in ids:
                rest = np.vstack([epts[i] for i in ids if i != k])
                e = fit_circle_ellipse(rest, s, w, robust=False, min_cover=0.75)
                if e is None:
                    continue
                cx, cy, rx, ry = (e[0][1][0], e[0][1][1], e[0][1][2], e[0][1][2]) if e[0][0] == "circle" else \
                    (*(shape_points(e[0], 8).mean(0)), np.ptp(shape_points(e[0], 8)[:, 0]) / 2, np.ptp(shape_points(e[0], 8)[:, 1]) / 2)
                D = epts[k]
                inside = ((D[:, 0] - cx) / rx) ** 2 + ((D[:, 1] - cy) / ry) ** 2
                if inside.max() < 1.15 and inside.min() < 0.8:
                    fits.append(e)
                    break
        if not fits:
            continue
        shape, rms = min(fits, key=lambda f: f[1])
        if work is not None:
            # the whole outline must lie on ink (a false ellipse around a
            # group of fingers crosses the white gaps between them)
            from scipy.ndimage import map_coordinates
            O = shape_points(shape, 30)
            v = map_coordinates(work, [O[:, 1] - 0.5, O[:, 0] - 0.5], order=1, mode="constant")
            if (v > 0.35).mean() < 0.88:
                continue
        # a whole shape has no strokes running from its outline into it (the
        # outline of a group of fingers does: the gaps between the fingers)
        O = shape_points(shape, 30).astype(np.float32)
        inner = False
        nodes_c = {edges[i][0] for i in ids} | {edges[i][1] for i in ids}
        for j, (a2, b2, p2) in enumerate(edges):
            if j in ids or not ({a2, b2} & nodes_c):
                continue
            E = epts[j][::max(1, len(epts[j]) // 12)]
            ins = [cv2.pointPolygonTest(O, (float(x), float(y)), True) > 0.6 * w for x, y in E]
            if np.mean(ins) > 0.5:
                inner = True
                break
        if inner:
            continue
        # a rounded rectangle / polygon that fits better is not a round shape
        alt = primitive_closed(pts, s, w)
        if alt is not None and alt[0] == "path" and _err(pts, alt) < 0.8 * _err(pts, shape):
            continue
        tree = cKDTree(shape_points(shape, 40))
        own = []
        for i in ids:
            dd = tree.query(epts[i])[0]
            if (dd < 0.4 * w).mean() >= 0.9 and dd.max() < 0.7 * w:
                own.append(i)
        if len(own) >= 2 or (len(own) == 1 and len(ids) == 1):
            cands.append((own, shape, rms, area))
    cands.sort(key=lambda c: (-len(c[0]), -c[3], c[2]))
    used, shapes = set(), []
    for own, shape, rms, area in cands:
        if used & set(own):
            continue
        shapes.append(shape)
        used |= set(own)
    return shapes, used


def _face_primitives(edges, r_up, up, s, w, sopt):
    import dataclasses
    """Replace enclosed regions by exact primitives (rect, circle, ellipse,
    polygon), also trying unions of neighbouring regions (a door inside a
    wall -> the wall is still one rectangle). Remaining simple regions become
    one smooth closed path. Returns (shapes, remaining edges)."""
    shapes = []
    consumed = set()
    if sopt.extra.get("cycles", True):
        H = max(max(np.asarray(p, float).max(0)) for _, _, p in edges) / up if edges else 0
        shapes, consumed = _cycle_primitives(edges, up, s, w, extent=0.7 * H, work=sopt.extra.get("_work"))
    faces = _faces(edges, r_up)
    to_pts = lambda P: (P[:, ::-1] + 0.5) / up
    bounded = []
    if faces:
        pos_area = sum(1 for f in faces if f[2] > 0)
        sign = 1 if pos_area >= len(faces) - pos_area else -1
        bounded = [f for f in faces if f[2] * sign > 0 and abs(f[2]) >= (1.5 * up * s) ** 2]
    bounded.sort(key=lambda f: abs(f[2]))
    # slivers smaller than the stroke itself are gaps between two touching
    # shapes (a pin tip resting on its base): swallow them
    tiny = (1.1 * w * up) ** 2
    for ids, P, area in [f for f in bounded if abs(f[2]) < tiny]:
        consumed |= set(ids)
    bounded = [f for f in bounded if abs(f[2]) >= tiny]
    matched = [False] * len(bounded)
    for k, (ids, P, area) in enumerate(bounded):
        if all(i in consumed for i in ids):
            continue
        prim = primitive_closed(to_pts(P), s, w)
        if prim is not None:
            shapes.append(prim)
            consumed |= set(ids)
            matched[k] = True
    # unions of two neighbouring regions
    for k, (ids, P, area) in enumerate(bounded):
        if matched[k]:
            continue
        for j, (ids2, P2, _) in enumerate(bounded):
            if j == k or not (set(ids) & set(ids2)):
                continue
            sym = set(ids) ^ set(ids2)
            if not sym or sym <= consumed:
                continue
            Q = _chain_cycle(edges, sym)
            if Q is None:
                continue
            prim = primitive_closed(to_pts(Q), s, w)
            if prim is not None and prim[0] != "circle" and len(prim[1][1]) <= 8 \
                    and any(sg[0] == "L" for sg in prim[1][1]):
                shapes.append(prim)
                consumed |= sym
                matched[k] = True
                break
    # simple leftover regions -> one smooth closed path
    owners = {}
    for ids, _, _ in bounded:
        for i in ids:
            owners[i] = owners.get(i, 0) + 1
    for k, (ids, P, area) in enumerate(bounded):
        if matched[k] or all(i in consumed for i in ids):
            continue
        if any(owners[i] > 1 and i not in consumed for i in ids):
            continue
        ex = dict(sopt.extra)
        ex["keep_line"] = max(ex.get("keep_line", 2.6), 5.0)
        smooth = dataclasses.replace(sopt, tolerance=sopt.tolerance * 1.25, min_line=5.0,
                                     corner_angle=70, extra=ex)
        pts = to_pts(P)
        sym = fit_symmetric_closed(pts, lambda H: fit_path(H, False, smooth, s)[1], s)
        shapes.append(sym if sym is not None else fit_path(pts, True, smooth, s))
        consumed |= set(ids)
        matched[k] = True
    # closed loops without junctions
    keep = []
    for i, e in enumerate(edges):
        if i in consumed:
            continue
        if e[0] < 0 or e[0] == e[1]:
            pts = (np.asarray(e[2], float)[:, ::-1] + 0.5) / up
            if len(pts) > 8:
                prim = primitive_closed(pts, s, w)
                if prim is not None:
                    shapes.append(prim)
                    continue
        keep.append(e)
    return shapes, keep


def _contract_junctions(nodes, edges, max_len):
    """Merge junction nodes joined by a very short edge (the skeleton splits
    one crossing into several nearby branch points)."""
    parent = list(range(len(nodes)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    deg = {}
    for a, b, _ in edges:
        if a >= 0:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
    keep = []
    for e in edges:
        a, b, p = e
        if a >= 0 and a != b and deg.get(a, 0) >= 3 and deg.get(b, 0) >= 3 and _length(p) < max_len:
            parent[find(a)] = find(b)
            continue
        keep.append(e)
    groups = {}
    for i in range(len(nodes)):
        groups.setdefault(find(i), []).append(i)
    pos = list(nodes)
    for r, g in groups.items():
        m = np.mean([nodes[i] for i in g], axis=0)
        for i in g:
            pos[i] = m
    out = [[find(a) if a >= 0 else a, find(b) if b >= 0 else b, p] for a, b, p in keep]
    return pos, out


def _continuation_strokes(nodes, edges, w_up):
    """Group skeleton edges into whole drawing strokes the way a designer sees
    them: at every junction a line continues into the branch that carries on
    most smoothly (an ellipse stays one ellipse even where a pin touches it);
    two left-over branches meeting at a junction form a corner (a pin tip).
    Returns [(closed, pts (r,c))]."""
    R = 0.9 * w_up
    ends = {}  # node -> [(edge, end, direction away from node)]
    for i, (a, b, p) in enumerate(edges):
        if a < 0:
            continue
        P = np.asarray(p, float)
        for e, node in ((0, a), (1, b)):
            Q = P if e == 0 else P[::-1]
            c = np.asarray(nodes[node], float)
            d = np.linalg.norm(Q - c, axis=1)
            win = Q[(d > R) & (d < 2.4 * w_up)]
            if len(win) >= 2:
                v = win[-1] - win[0]
                if np.linalg.norm(v) < 1e-6:
                    v = win[-1] - c
            else:
                v = Q[min(len(Q) - 1, max(1, len(Q) // 2))] - c
            n = np.linalg.norm(v)
            ends.setdefault(node, []).append((i, e, v / n if n > 1e-9 else v))
    def branch_pts(i, e, node, dmax):
        P = np.asarray(edges[i][2], float)
        Q = P if e == 0 else P[::-1]
        c = np.asarray(nodes[node], float)
        d = np.linalg.norm(Q - c, axis=1)
        k = np.nonzero(d > dmax)[0]
        Q = Q[: (k[0] if len(k) else len(Q))]
        return Q[np.linalg.norm(Q - c, axis=1) > 0.5 * R]

    def smooth_cost(node, x, y):
        """How well both branches lie on one circle (or line) that also passes
        through the junction: small for the two halves of a ring or a line
        crossing, large for a line turning into a ring."""
        A = branch_pts(x[0], x[1], node, 3.2 * w_up)
        B = branch_pts(y[0], y[1], node, 3.2 * w_up)
        if len(A) < 3 or len(B) < 3:
            return None
        Q = np.vstack([A[::-1], B])
        m = Q.mean(0)
        q = Q - m
        # straight line first
        _, sv, vt = np.linalg.svd(q, full_matrices=False)
        line_res = np.sqrt(np.mean((q @ vt[1]) ** 2))
        # algebraic circle
        M = np.c_[2 * q, np.ones(len(q))]
        sol, *_ = np.linalg.lstsq(M, (q ** 2).sum(1), rcond=None)
        cx, cy = sol[0], sol[1]
        r = np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
        circ = np.abs(np.hypot(q[:, 0] - cx, q[:, 1] - cy) - r)
        circ_res = np.sqrt(np.mean(circ ** 2))
        c = np.asarray(nodes[node], float) - m
        through = abs(np.hypot(c[0] - cx, c[1] - cy) - r)
        if r < 1.2 * w_up:
            return None
        # the two branches must leave the junction on opposite sides
        da = A.mean(0) - (c + m)
        db = B.mean(0) - (c + m)
        if da @ db / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-9) > 0.2:
            return None
        return float(min(line_res, max(circ_res, 0.5 * through)))

    pair = {}
    kind = {}
    for node, lst in ends.items():
        free = list(range(len(lst)))
        cand = []
        for x in range(len(lst)):
            for y in range(x + 1, len(lst)):
                cosv = float(lst[x][2] @ lst[y][2])
                cost = smooth_cost(node, lst[x][:2], lst[y][:2])
                if cost is None:
                    cost = 0.1 * w_up if cosv < -0.9 else None
                if cost is not None and cost < 0.16 * w_up:
                    cand.append((cost, x, y))
        cand.sort()
        for c, x, y in cand:
            if x in free and y in free:
                free.remove(x); free.remove(y)
                ex, ey = lst[x][:2], lst[y][:2]
                pair[ex], pair[ey] = ey, ex
                kind[ex] = kind[ey] = ("cont", node)
        if len(free) == 2:
            x, y = free
            ex, ey = lst[x][:2], lst[y][:2]
            if ex != ey:
                pair[ex], pair[ey] = ey, ex
                kind[ex] = kind[ey] = ("corner", node)

    def oriented(i, e):
        P = np.asarray(edges[i][2], float)
        return P if e == 0 else P[::-1]

    def trim(Q, at_start, node):
        if node is None:
            return Q
        c = np.asarray(nodes[node], float)
        d = np.linalg.norm(Q - c, axis=1)
        L = len(Q)
        if at_start:
            k = 0
            while k < L - 2 and d[k] < R and k < 0.4 * L:
                k += 1
            return Q[k:]
        k = L - 1
        while k > 1 and d[k] < R and (L - 1 - k) < 0.4 * L:
            k -= 1
        return Q[:k + 1]

    used = set()
    strokes = []

    def walk(start):
        pieces = []  # (points, how the piece joins the next)
        cur = start
        closed = False
        while True:
            i, e = cur
            used.add(i)
            a, b, _ = edges[i]
            n0 = a if e == 0 else b
            n1 = b if e == 0 else a
            Q = oriented(i, e)
            Q = trim(Q, True, n0 if n0 >= 0 else None)
            Q = trim(Q, False, n1 if n1 >= 0 else None)
            nxt = pair.get((i, 1 - e))
            pieces.append((Q, kind.get((i, 1 - e)), n1))
            if nxt is None:
                break
            if nxt[0] in used:
                closed = nxt == start
                break
            cur = nxt
        pts = []
        first_node = edges[start[0]][0] if start[1] == 0 else edges[start[0]][1]
        if not closed and first_node >= 0 and start not in pair:
            pts.append(np.asarray(nodes[first_node], float))  # open end on a junction
        for k, (Q, how, node) in enumerate(pieces):
            pts.extend(list(Q))
            last = k == len(pieces) - 1
            if how is not None and how[0] == "corner" and (not last or closed):
                pts.append(np.asarray(nodes[how[1]], float))
            elif last and not closed and node is not None and node >= 0:
                pts.append(np.asarray(nodes[node], float))
        return closed, np.asarray(pts, float)

    # open strokes first (they start at a free end), then loops
    order = []
    for i, (a, b, p) in enumerate(edges):
        if a < 0:
            used.add(i)
            strokes.append((True, np.asarray(p, float)))
            continue
        for e in (0, 1):
            if (i, e) not in pair:
                order.append((i, e))
    for st in order:
        if st[0] in used:
            continue
        strokes.append(walk(st))
    for i in range(len(edges)):
        if i not in used:
            strokes.append(walk((i, 0)))
    return [(bool(c), p) for c, p in strokes]


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
    all_edges = list(edges)

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
    extra["_work"] = work
    sopt = dataclasses.replace(opt, tolerance=opt.tolerance * 1.6, corner_angle=max(opt.corner_angle, 60), extra=extra)

    ex_s = dict(sopt.extra)
    ex_s["keep_line"] = max(ex_s.get("keep_line", 2.6), 5.0)
    smooth = dataclasses.replace(sopt, tolerance=sopt.tolerance * 1.25, min_line=5.0,
                                 corner_angle=70, extra=ex_s)
    use_prims = opt.extra.get("primitives", True)
    prim_shapes, items = [], []
    if opt.extra.get("engine", "faces") == "continuation":
        nodes2, edges2 = _contract_junctions(nodes, edges, 1.3 * w * up)
        for closed, P in _continuation_strokes(nodes2, edges2, w * up):
            if len(P) < 3:
                continue
            pts = (P[:, ::-1] + 0.5) / up
            L = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
            if L < 0.6 * s:
                continue
            if closed:
                if np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
                    pts = pts[:-1]
                prim = primitive_closed(pts, s, w) if use_prims else None
            else:
                prim = primitive_open(pts, s, w) if use_prims else None
                if prim is None and use_prims:
                    # an almost closed arc (a head resting on shoulders) is a
                    # whole ellipse partly hidden behind the other shape
                    diag = np.hypot(*np.ptp(pts, 0))
                    if np.linalg.norm(pts[0] - pts[-1]) < 0.45 * diag and L > 2.2 * diag:
                        e = fit_circle_ellipse(pts, s, w, robust=False, min_cover=0.7)
                        if e is not None:
                            prim = e[0]
            if prim is not None:
                prim_shapes.append(prim)
            else:
                items.append((closed, pts))
    else:
        # ---- designer-style primitives on enclosed regions ----
        prim_shapes = []
        if opt.extra.get("primitives", True):
            prim_shapes, edges = _face_primitives(edges, r_up, up, s, w, sopt)
        edges = _join_through(nodes, edges, win=int(r_up * 1.5) + 2)

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
            if opt.extra.get("primitives", True):
                prim = primitive_open(pts, s, w)
                if prim is not None:
                    prim_shapes.append(prim)
                    continue
            if L > 3 * rj:
                if deg.get(a, 0) >= 3:
                    pts = straighten(pts, True)
                if deg.get(b, 0) >= 3:
                    pts = straighten(pts, False)
            items.append((False, pts))


    shapes, items = _circles(items, s, w, opt)
    shapes = prim_shapes + shapes
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
        shapes.append(fit_path(pts, closed, smooth if closed else sopt, s))
    if opt.extra.get("primitives", True):
        shapes = regularize(shapes, s, w)
    if opt.extra.get("refine", True):
        # render -> compare -> fix: sit every line exactly on the ink centre
        from .refine import refine_shapes
        shapes = refine_shapes(shapes, work, w)
        if opt.extra.get("primitives", True):
            shapes = regularize(shapes, s, w)
    if opt.extra.get("tidy", True):
        # one anchor where two sit almost on top of each other, fewer smooth anchors
        from .tidy import tidy
        shapes = tidy(shapes, w, opt.extra.get("tidy_chord", 1.0), opt.extra.get("tidy_tol", 0.12),
                      opt.extra.get("bent", True))
        if opt.extra.get("corners", True):
            # sharp or rounded, and how round: decided by comparing with the ink
            from .corners import merge_double_lines, optimize_corners, straighten_curves
            shapes = straighten_curves(shapes, work, w)
            shapes = optimize_corners(shapes, work, w)
            shapes = merge_double_lines(shapes, w)
            if opt.extra.get("bent", True):
                # chains of short straight pieces that bend a little: one curve
                from .tidy import bent_lines_to_curve
                shapes = [(k, bent_lines_to_curve(d[0], d[1], k == "path", w)) if k != "circle" else (k, d)
                          for k, d in shapes]
        if opt.extra.get("refine", True):
            from .refine import refine_shapes
            shapes = refine_shapes(shapes, work, w)
        if opt.extra.get("primitives", True):
            shapes = regularize(shapes, s, w)
    if opt.extra.get("repeat", True) and opt.extra.get("primitives", True):
        # identical parts drawn once and copied
        from .repeat import unify_repeats
        shapes = regularize(unify_repeats(shapes, work, w), s, w)
    if opt.extra.get("holes", True) and opt.extra.get("primitives", True):
        # every enclosed white area is the inside of one closed shape
        from .holes import repair_holes
        from .holes import share_edges
        shapes = share_edges(regularize(repair_holes(shapes, work, w, s, smooth), s, w), w)
    if opt.extra.get("mirror", True) and opt.symmetry:
        # a symmetric icon: one half drawn, the other half its mirror image
        from .mirror import mirror_shapes
        shapes = mirror_shapes(shapes, work, w)
    if opt.extra.get("overlap", True):
        # never a line on top of a line: shared pieces kept once, hidden parts cut
        from .overlap import remove_overlaps
        shapes = remove_overlaps(shapes, work, w)
    if opt.extra.get("fill_solid", True) and opt.mode in ("stroke", "designer"):
        fills = fills + _solid_patches(shapes, work, w, opt, s)
    return shapes, fills, w


def _solid_patches(shapes, work, w, opt, s, z=4):
    """Render -> compare -> fix for solid parts: ink the strokes do not cover
    and that is a real blob (a solid tie, a compass needle), not a thin fringe
    along a line, becomes a filled shape."""
    from .tracer import _render_shapes
    H, W = work.shape
    ref = cv2.resize(work, (W * z, H * z), interpolation=cv2.INTER_CUBIC) > 0.5
    got = _render_shapes(shapes, (H, W), z, w) > 0
    got = cv2.dilate(got.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    miss = (ref & ~got).astype(np.uint8)
    r = max(1, int(round(0.25 * w * z)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core = cv2.morphologyEx(miss, cv2.MORPH_OPEN, ker)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 0.8 * (w * z) ** 2:
            continue
        # grow the blob back to the ink edge (under the strokes around it)
        g = max(1, int(round(0.6 * w * z)))
        gk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
        region = (cv2.dilate((lab == k).astype(np.uint8), gk) > 0) & ref
        small = cv2.resize(region.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        for c in extract_contours(small, opt, s):
            try:
                out.append(fit_path(c, True, opt, s))
            except Exception:
                continue
    return out


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
