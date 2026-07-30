"""numba-native contact narrow-phase kernels — the production path; the numpy versions in
``contact3d.py`` are the bit-for-bit ORACLE these are gated against.

These are @njit ports (explicit scalar arithmetic — no einsum/column_stack/outer helpers, which numba
either rejects or compiles poorly) of the per-pair vertex-face cubic barrier + ppf smoothed friction.
The per-pair loop is the dominant contact cost (~80 µs/pair in numpy is almost all interpreter/dispatch
overhead); @njit removes it. Semantics MUST match ``contact3d.tri_barrier_eval`` to ~machine precision
(gate: tests compare bit-for-bit, rtol 1e-12). Written numba-native from the start (no Python prototype
to re-port). The LBVH broad-phase is the sibling port (ppf lbvh template) — added alongside.

This is a modified numba translation of the corresponding CoupFE NumPy
adaptation. Upstream provenance and licensing are recorded in ``NOTICE``.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _pe_coeff(p, e0, e1):
    """Closest-point weights (w0, w1) of p on segment e0-e1 (UNCLAMPED t) — matches point_edge_coeff."""
    rx = e1[0] - e0[0]; ry = e1[1] - e0[1]; rz = e1[2] - e0[2]
    d = rx * rx + ry * ry + rz * rz
    if d > 0.0:
        t = (rx * (p[0] - e0[0]) + ry * (p[1] - e0[1]) + rz * (p[2] - e0[2])) / d
        return 1.0 - t, t
    return 0.5, 0.5


@njit(cache=True)
def _pt_coeff(p, t0, t1, t2):
    """Barycentric weights of p's closest point on the triangle plane; degenerate → longest edge.
    Matches point_triangle_coeff (incl. the _solve2 adj/det convention)."""
    r0x = t1[0] - t0[0]; r0y = t1[1] - t0[1]; r0z = t1[2] - t0[2]
    r1x = t2[0] - t0[0]; r1y = t2[1] - t0[1]; r1z = t2[2] - t0[2]
    px = p[0] - t0[0]; py = p[1] - t0[1]; pz = p[2] - t0[2]
    a00 = r0x * r0x + r0y * r0y + r0z * r0z
    a01 = r0x * r1x + r0y * r1y + r0z * r1z
    a11 = r1x * r1x + r1y * r1y + r1z * r1z
    b0 = r0x * px + r0y * py + r0z * pz
    b1 = r1x * px + r1y * py + r1z * pz
    det = a00 * a11 - a01 * a01
    if det != 0.0:
        u = (a11 * b0 - a01 * b1) / det
        v = (a00 * b1 - a01 * b0) / det
        return 1.0 - u - v, u, v
    # degenerate triangle → longest edge
    e0l = (t1[0]-t0[0])**2 + (t1[1]-t0[1])**2 + (t1[2]-t0[2])**2
    e1l = (t2[0]-t1[0])**2 + (t2[1]-t1[1])**2 + (t2[2]-t1[2])**2
    e2l = (t0[0]-t2[0])**2 + (t0[1]-t2[1])**2 + (t0[2]-t2[2])**2
    if e0l >= e1l and e0l >= e2l:
        w0, w1 = _pe_coeff(p, t0, t1); return w0, w1, 0.0
    if e1l >= e2l:
        w0, w1 = _pe_coeff(p, t1, t2); return 0.0, w0, w1
    w0, w1 = _pe_coeff(p, t2, t0); return w1, 0.0, w0


@njit(cache=True)
def _pt_coeff_unclassified(p, t0, t1, t2):
    """Robust barycentric weights — foot outside triangle → closest edge/vertex. Matches
    point_triangle_coeff_unclassified."""
    c0, c1, c2 = _pt_coeff(p, t0, t1, t2)
    cmin = min(c0, min(c1, c2)); cmax = max(c0, max(c1, c2))
    if cmin >= 0.0 and cmax <= 1.0:
        return c0, c1, c2
    if c0 < 0.0:
        w0, w1 = _pe_coeff(p, t1, t2)
        if 0.0 <= w0 <= 1.0:
            return 0.0, w0, w1
        return (0.0, 1.0, 0.0) if w0 > 1.0 else (0.0, 0.0, 1.0)
    if c1 < 0.0:
        w0, w1 = _pe_coeff(p, t0, t2)
        if 0.0 <= w0 <= 1.0:
            return w0, 0.0, w1
        return (1.0, 0.0, 0.0) if w0 > 1.0 else (0.0, 0.0, 1.0)
    w0, w1 = _pe_coeff(p, t0, t1)
    if 0.0 <= w0 <= 1.0:
        return w0, w1, 0.0
    return (1.0, 0.0, 0.0) if w0 > 1.0 else (0.0, 1.0, 0.0)


@njit(cache=True)
def tri_barrier_eval_nb(P, A, B, C, dhat, kappa, mass, mu, eps, X0, has_mass, has_fric):
    """numba port of contact3d.tri_barrier_eval (vertex-face cubic barrier + ppf smoothed friction).
    ``mass`` (n_pair,) and ``X0`` (n_pair,4,3) are always passed (dummy when off); ``has_mass``/
    ``has_fric`` gate them. Returns R (n_pair,12), K (n_pair,12,12), active (n_pair,)."""
    n_pair = P.shape[0]
    R = np.zeros((n_pair, 12))
    K = np.zeros((n_pair, 12, 12))
    active = np.zeros(n_pair, dtype=np.bool_)
    grad = np.empty(12)
    cc = np.empty(4)
    nvec = np.empty(3)
    for i in range(n_pair):
        w0, w1, w2 = _pt_coeff_unclassified(P[i], A[i], B[i], C[i])
        clx = w0 * A[i, 0] + w1 * B[i, 0] + w2 * C[i, 0]
        cly = w0 * A[i, 1] + w1 * B[i, 1] + w2 * C[i, 1]
        clz = w0 * A[i, 2] + w1 * B[i, 2] + w2 * C[i, 2]
        gx = P[i, 0] - clx; gy = P[i, 1] - cly; gz = P[i, 2] - clz
        gap = np.sqrt(gx * gx + gy * gy + gz * gz)
        if gap >= dhat or gap == 0.0:
            continue
        active[i] = True
        nvec[0] = gx / gap; nvec[1] = gy / gap; nvec[2] = gz / gap
        g = dhat - gap
        s = kappa + (mass[i] / (gap * gap) if has_mass else 0.0)
        cc[0] = 1.0; cc[1] = -w0; cc[2] = -w1; cc[3] = -w2
        for b in range(4):
            for a in range(3):
                grad[3 * b + a] = cc[b] * nvec[a]
        sgg = s * g * g
        tsg = 2.0 * s * g
        for r in range(12):
            R[i, r] = -sgg * grad[r]
            for cidx in range(12):
                K[i, r, cidx] = tsg * grad[r] * grad[cidx]
        if has_fric:
            # relative tangential slip dxr = sum_b cc[b]*(cur_b - X0_b); cur = [P,A,B,C]
            dx0 = cc[0]*(P[i,0]-X0[i,0,0]) + cc[1]*(A[i,0]-X0[i,1,0]) + cc[2]*(B[i,0]-X0[i,2,0]) + cc[3]*(C[i,0]-X0[i,3,0])
            dx1 = cc[0]*(P[i,1]-X0[i,0,1]) + cc[1]*(A[i,1]-X0[i,1,1]) + cc[2]*(B[i,1]-X0[i,2,1]) + cc[3]*(C[i,1]-X0[i,3,1])
            dx2 = cc[0]*(P[i,2]-X0[i,0,2]) + cc[1]*(A[i,2]-X0[i,1,2]) + cc[2]*(B[i,2]-X0[i,2,2]) + cc[3]*(C[i,2]-X0[i,3,2])
            dn = dx0 * nvec[0] + dx1 * nvec[1] + dx2 * nvec[2]
            p0 = dx0 - dn * nvec[0]; p1 = dx1 - dn * nvec[1]; p2 = dx2 - dn * nvec[2]
            ut = np.sqrt(p0 * p0 + p1 * p1 + p2 * p2)
            denom = eps if eps > ut else ut
            lam = mu * sgg / denom
            pdx = (p0, p1, p2)
            # Pp = I - n⊗n ; Rf[3b+a] = cc[b]*lam*pdx[a] ; Kf block (b,d) = lam*cc[b]*cc[d]*Pp
            for b in range(4):
                R[i, 3 * b + 0] += cc[b] * lam * p0
                R[i, 3 * b + 1] += cc[b] * lam * p1
                R[i, 3 * b + 2] += cc[b] * lam * p2
            for b in range(4):
                for d in range(4):
                    lcd = lam * cc[b] * cc[d]
                    for a in range(3):
                        for e in range(3):
                            pp = (1.0 if a == e else 0.0) - nvec[a] * nvec[e]
                            K[i, 3 * b + a, 3 * d + e] += lcd * pp
    return R, K, active


@njit(cache=True)
def ee_pairs_nb(cand, ptr, edges, emin, emax, dhat, exclude_shared):
    """Edge-edge candidate pairs ``(i<j)`` from the LBVH CSR query: shared-vertex exclusion + AABB-to-AABB
    distance prune — the numba form of ``edge_edge_candidates._consider`` (the per-pair Python loop is the
    contact bottleneck; bit-identical superset to the numpy oracle). Two passes: count, then fill."""
    d2 = dhat * dhat
    n = ptr.shape[0] - 1
    cnt = 0
    for _pass in range(2):
        if _pass == 1:
            oi = np.empty(cnt, dtype=np.int64)
            oj = np.empty(cnt, dtype=np.int64)
            cnt = 0
        for i in range(n):
            for k in range(ptr[i], ptr[i + 1]):
                j = cand[k]
                if j <= i:
                    continue
                if exclude_shared:
                    a0 = edges[i, 0]; a1 = edges[i, 1]; b0 = edges[j, 0]; b1 = edges[j, 1]
                    if a0 == b0 or a0 == b1 or a1 == b0 or a1 == b1:
                        continue
                s2 = 0.0
                for ax in range(3):
                    g = emin[i, ax] - emax[j, ax]
                    g2 = emin[j, ax] - emax[i, ax]
                    if g2 > g:
                        g = g2
                    if g > 0.0:
                        s2 += g * g
                if s2 < d2:
                    if _pass == 1:
                        oi[cnt] = i; oj[cnt] = j
                    cnt += 1
    return oi, oj


@njit(cache=True)
def vf_drop_incident_nb(vert_ids, cand_faces, faces):
    """Self-contact incident-exclusion (numba). ``keep[m] = False`` where vertex ``vert_ids[m]`` is a
    node of candidate face ``cand_faces[m]`` — a surface vertex must not contact the faces incident to
    it (gap ≡ 0). Applied to the broad-phase pairs so BOTH the barrier closest-face and the CCD
    ``max_step`` skip them (an incident pair would otherwise pin the time-of-impact at 0)."""
    M = cand_faces.shape[0]
    keep = np.ones(M, dtype=np.bool_)
    for m in range(M):
        v = vert_ids[m]
        f = cand_faces[m]
        if v == faces[f, 0] or v == faces[f, 1] or v == faces[f, 2]:
            keep[m] = False
    return keep


@njit(cache=True)
def closest_faces_nb(pos, verts, faces, cand_flat, cand_ptr, dhat):
    """Per owned vertex, the closest candidate face (min closest-point gap) and that gap. Matches
    DeformableBarrierContact3D._closest_face over the broad-phase candidates (CSR: cand_flat / cand_ptr).
    Returns best_face (n_v,) [-1 if no candidate] and best_gap (n_v,)."""
    nv = verts.shape[0]
    best_f = np.full(nv, -1, dtype=np.int64)
    best_g = np.full(nv, 1.0e30)
    for vi in range(nv):
        v = verts[vi]
        for k in range(cand_ptr[vi], cand_ptr[vi + 1]):
            f = cand_flat[k]
            f0 = faces[f, 0]; f1 = faces[f, 1]; f2 = faces[f, 2]
            w0, w1, w2 = _pt_coeff_unclassified(pos[v], pos[f0], pos[f1], pos[f2])
            clx = w0 * pos[f0, 0] + w1 * pos[f1, 0] + w2 * pos[f2, 0]
            cly = w0 * pos[f0, 1] + w1 * pos[f1, 1] + w2 * pos[f2, 1]
            clz = w0 * pos[f0, 2] + w1 * pos[f1, 2] + w2 * pos[f2, 2]
            gx = pos[v, 0] - clx; gy = pos[v, 1] - cly; gz = pos[v, 2] - clz
            gap = np.sqrt(gx * gx + gy * gy + gz * gz)
            if gap < best_g[vi]:
                best_g[vi] = gap
                best_f[vi] = f
    return best_f, best_g


# ----------------------------------------------------------------- edge-edge barrier + ACCD (numba)
@njit(cache=True)
def _pe_coeff_s(px, py, pz, e0x, e0y, e0z, e1x, e1y, e1z):
    """Scalar point-edge weight (w0, w1) — UNCLAMPED. Matches contact3d.point_edge_coeff."""
    rx = e1x - e0x; ry = e1y - e0y; rz = e1z - e0z
    d = rx * rx + ry * ry + rz * rz
    if d > 0.0:
        t = (rx * (px - e0x) + ry * (py - e0y) + rz * (pz - e0z)) / d
        return 1.0 - t, t
    return 0.5, 0.5


@njit(cache=True)
def _pe_clamp_s(px, py, pz, e0x, e0y, e0z, e1x, e1y, e1z):
    w0, w1 = _pe_coeff_s(px, py, pz, e0x, e0y, e0z, e1x, e1y, e1z)
    if w0 < 0.0:
        return 0.0, 1.0
    if w0 > 1.0:
        return 1.0, 0.0
    return w0, w1


@njit(cache=True)
def _ee_coeff(ea0, ea1, eb0, eb1):
    """Edge-edge closest-point weights (a0,a1,b0,b1). Matches contact3d.edge_edge_coeff (direct 2×2
    solve + ea0-vs-edge-b fallback + 4-step alternating projection)."""
    r0x = ea1[0]-ea0[0]; r0y = ea1[1]-ea0[1]; r0z = ea1[2]-ea0[2]
    r1x = eb1[0]-eb0[0]; r1y = eb1[1]-eb0[1]; r1z = eb1[2]-eb0[2]
    a00 = r0x*r0x + r0y*r0y + r0z*r0z
    a01 = -(r0x*r1x + r0y*r1y + r0z*r1z)
    a11 = r1x*r1x + r1y*r1y + r1z*r1z
    dx0 = eb0[0]-ea0[0]; dy0 = eb0[1]-ea0[1]; dz0 = eb0[2]-ea0[2]
    bb0 = r0x*dx0 + r0y*dy0 + r0z*dz0
    bb1 = -(r1x*dx0 + r1y*dy0 + r1z*dz0)
    det = a00*a11 - a01*a01
    c0, c1 = _pe_coeff_s(ea0[0], ea0[1], ea0[2], eb0[0], eb0[1], eb0[2], eb1[0], eb1[1], eb1[2])
    res0 = 1.0; res1 = 0.0; res2 = c0; res3 = c1
    pbx = c0*eb0[0] + c1*eb1[0]; pby = c0*eb0[1] + c1*eb1[1]; pbz = c0*eb0[2] + c1*eb1[2]
    mdx = ea0[0]-pbx; mdy = ea0[1]-pby; mdz = ea0[2]-pbz
    min_dist = mdx*mdx + mdy*mdy + mdz*mdz
    if det != 0.0:
        xx0 = (a11*bb0 - a01*bb1) / det
        xx1 = (-a01*bb0 + a00*bb1) / det
        d0 = 1.0-xx0; d1 = xx0; d2 = 1.0-xx1; d3 = xx1
        padx = d0*ea0[0]+d1*ea1[0]; pady = d0*ea0[1]+d1*ea1[1]; padz = d0*ea0[2]+d1*ea1[2]
        pbdx = d2*eb0[0]+d3*eb1[0]; pbdy = d2*eb0[1]+d3*eb1[1]; pbdz = d2*eb0[2]+d3*eb1[2]
        ddx = padx-pbdx; ddy = pady-pbdy; ddz = padz-pbdz
        if ddx*ddx + ddy*ddy + ddz*ddz < min_dist:
            res0 = d0; res1 = d1; res2 = d2; res3 = d3
    xa = res1; xb = res3
    q0x = eb0[0]-ea0[0]; q0y = eb0[1]-ea0[1]; q0z = eb0[2]-ea0[2]
    q1x = eb1[0]-ea0[0]; q1y = eb1[1]-ea0[1]; q1z = eb1[2]-ea0[2]
    p0x = ea0[0]-eb0[0]; p0y = ea0[1]-eb0[1]; p0z = ea0[2]-eb0[2]
    p1x = ea1[0]-eb0[0]; p1y = ea1[1]-eb0[1]; p1z = ea1[2]-eb0[2]
    for _ in range(4):
        _, xa = _pe_coeff_s(xb*r1x, xb*r1y, xb*r1z, p0x, p0y, p0z, p1x, p1y, p1z)
        _, xb = _pe_coeff_s(xa*r0x, xa*r0y, xa*r0z, q0x, q0y, q0z, q1x, q1y, q1z)
    return 1.0-xa, xa, 1.0-xb, xb


@njit(cache=True)
def _ee_coeff_unclassified(ea0, ea1, eb0, eb1):
    """Matches contact3d.edge_edge_coeff_unclassified (4-endpoint COG-centered fallback)."""
    a0, a1, b0, b1 = _ee_coeff(ea0, ea1, eb0, eb1)
    lo = min(a0, min(a1, min(b0, b1))); hi = max(a0, max(a1, max(b0, b1)))
    if lo >= 0.0 and hi <= 1.0:
        return a0, a1, b0, b1
    c1a, c1b = _pe_clamp_s(ea0[0], ea0[1], ea0[2], eb0[0], eb0[1], eb0[2], eb1[0], eb1[1], eb1[2])
    c2a, c2b = _pe_clamp_s(ea1[0], ea1[1], ea1[2], eb0[0], eb0[1], eb0[2], eb1[0], eb1[1], eb1[2])
    c3a, c3b = _pe_clamp_s(eb0[0], eb0[1], eb0[2], ea0[0], ea0[1], ea0[2], ea1[0], ea1[1], ea1[2])
    c4a, c4b = _pe_clamp_s(eb1[0], eb1[1], eb1[2], ea0[0], ea0[1], ea0[2], ea1[0], ea1[1], ea1[2])
    types = np.empty((4, 4))
    types[0, 0] = 1.0; types[0, 1] = 0.0; types[0, 2] = c1a; types[0, 3] = c1b
    types[1, 0] = 0.0; types[1, 1] = 1.0; types[1, 2] = c2a; types[1, 3] = c2b
    types[2, 0] = c3a; types[2, 1] = c3b; types[2, 2] = 1.0; types[2, 3] = 0.0
    types[3, 0] = c4a; types[3, 1] = c4b; types[3, 2] = 0.0; types[3, 3] = 1.0
    cogx = 0.25*(ea0[0]+ea1[0]+eb0[0]+eb1[0]); cogy = 0.25*(ea0[1]+ea1[1]+eb0[1]+eb1[1]); cogz = 0.25*(ea0[2]+ea1[2]+eb0[2]+eb1[2])
    a0x = ea0[0]-cogx; a0y = ea0[1]-cogy; a0z = ea0[2]-cogz
    a1x = ea1[0]-cogx; a1y = ea1[1]-cogy; a1z = ea1[2]-cogz
    b0x = eb0[0]-cogx; b0y = eb0[1]-cogy; b0z = eb0[2]-cogz
    b1x = eb1[0]-cogx; b1y = eb1[1]-cogy; b1z = eb1[2]-cogz
    bd = 1e300; bi = 0
    for ti in range(4):
        ta0 = types[ti, 0]; ta1 = types[ti, 1]; tb0 = types[ti, 2]; tb1 = types[ti, 3]
        xax = ta0*a0x + ta1*a1x; xay = ta0*a0y + ta1*a1y; xaz = ta0*a0z + ta1*a1z
        xbx = tb0*b0x + tb1*b1x; xby = tb0*b0y + tb1*b1y; xbz = tb0*b0z + tb1*b1z
        dx = xbx-xax; dy = xby-xay; dz = xbz-xaz
        d = dx*dx + dy*dy + dz*dz
        if d < bd:
            bd = d; bi = ti
    return types[bi, 0], types[bi, 1], types[bi, 2], types[bi, 3]


@njit(cache=True)
def edge_edge_barrier_eval_nb(P0, P1, Q0, Q1, dhat, kappa, mass, mu, eps, X0, has_mass, has_fric):
    """numba port of contact3d.edge_edge_barrier_eval (cubic barrier on the unclassified closest
    distance + ppf smoothed friction). Weights cc=[a0,a1,-b0,-b1] (Σcc=0 → action-reaction)."""
    n_pair = P0.shape[0]
    R = np.zeros((n_pair, 12)); K = np.zeros((n_pair, 12, 12)); active = np.zeros(n_pair, dtype=np.bool_)
    cc = np.empty(4); nvec = np.empty(3); grad = np.empty(12)
    for i in range(n_pair):
        a0, a1, b0, b1 = _ee_coeff_unclassified(P0[i], P1[i], Q0[i], Q1[i])
        gx = (a0*P0[i, 0] + a1*P1[i, 0]) - (b0*Q0[i, 0] + b1*Q1[i, 0])
        gy = (a0*P0[i, 1] + a1*P1[i, 1]) - (b0*Q0[i, 1] + b1*Q1[i, 1])
        gz = (a0*P0[i, 2] + a1*P1[i, 2]) - (b0*Q0[i, 2] + b1*Q1[i, 2])
        gap = np.sqrt(gx*gx + gy*gy + gz*gz)
        if gap >= dhat or gap == 0.0:
            continue
        active[i] = True
        nvec[0] = gx/gap; nvec[1] = gy/gap; nvec[2] = gz/gap
        g = dhat - gap
        s = kappa + (mass[i]/(gap*gap) if has_mass else 0.0)
        cc[0] = a0; cc[1] = a1; cc[2] = -b0; cc[3] = -b1
        for b in range(4):
            for a in range(3):
                grad[3*b + a] = cc[b]*nvec[a]
        sgg = s*g*g; tsg = 2.0*s*g
        for r in range(12):
            R[i, r] = -sgg*grad[r]
            for cidx in range(12):
                K[i, r, cidx] = tsg*grad[r]*grad[cidx]
        if has_fric:
            dx0 = cc[0]*(P0[i,0]-X0[i,0,0]) + cc[1]*(P1[i,0]-X0[i,1,0]) + cc[2]*(Q0[i,0]-X0[i,2,0]) + cc[3]*(Q1[i,0]-X0[i,3,0])
            dx1 = cc[0]*(P0[i,1]-X0[i,0,1]) + cc[1]*(P1[i,1]-X0[i,1,1]) + cc[2]*(Q0[i,1]-X0[i,2,1]) + cc[3]*(Q1[i,1]-X0[i,3,1])
            dx2 = cc[0]*(P0[i,2]-X0[i,0,2]) + cc[1]*(P1[i,2]-X0[i,1,2]) + cc[2]*(Q0[i,2]-X0[i,2,2]) + cc[3]*(Q1[i,2]-X0[i,3,2])
            dn = dx0*nvec[0] + dx1*nvec[1] + dx2*nvec[2]
            p0 = dx0 - dn*nvec[0]; p1 = dx1 - dn*nvec[1]; p2 = dx2 - dn*nvec[2]
            ut = np.sqrt(p0*p0 + p1*p1 + p2*p2)
            denom = eps if eps > ut else ut
            lam = mu*sgg/denom
            for b in range(4):
                R[i, 3*b+0] += cc[b]*lam*p0; R[i, 3*b+1] += cc[b]*lam*p1; R[i, 3*b+2] += cc[b]*lam*p2
            for b in range(4):
                for d in range(4):
                    lcd = lam*cc[b]*cc[d]
                    for a in range(3):
                        for e in range(3):
                            pp = (1.0 if a == e else 0.0) - nvec[a]*nvec[e]
                            K[i, 3*b+a, 3*d+e] += lcd*pp
    return R, K, active


@njit(cache=True)
def _pt_sq_nb(x):                                          # x (4,3) = [P, t0, t1, t2]
    w0, w1, w2 = _pt_coeff_unclassified(x[0], x[1], x[2], x[3])
    yx = w0*(x[1,0]-x[0,0]) + w1*(x[2,0]-x[0,0]) + w2*(x[3,0]-x[0,0])
    yy = w0*(x[1,1]-x[0,1]) + w1*(x[2,1]-x[0,1]) + w2*(x[3,1]-x[0,1])
    yz = w0*(x[1,2]-x[0,2]) + w1*(x[2,2]-x[0,2]) + w2*(x[3,2]-x[0,2])
    return yx*yx + yy*yy + yz*yz


@njit(cache=True)
def _ee_sq_nb(x):                                          # x (4,3) = [p0, p1, q0, q1]
    a0, a1, b0, b1 = _ee_coeff_unclassified(x[0], x[1], x[2], x[3])
    dx = (a0*x[0,0] + a1*x[1,0]) - (b0*x[2,0] + b1*x[3,0])
    dy = (a0*x[0,1] + a1*x[1,1]) - (b0*x[2,1] + b1*x[3,1])
    dz = (a0*x[0,2] + a1*x[1,2]) - (b0*x[2,2] + b1*x[3,2])
    return dx*dx + dy*dy + dz*dz


@njit(cache=True)
def _accd_toi_nb(x0, dx, kind, offset, max_t, reduction, max_iter):
    """Additive-CCD time-of-impact. kind 0=point-triangle, 1=edge-edge. Matches contact3d.accd_toi
    (ppf accd::ccd_helper conservative advancement)."""
    C = x0.shape[0]
    u_max = 0.0
    for i in range(C):
        for j in range(i + 1, C):
            ddx = dx[i,0]-dx[j,0]; ddy = dx[i,1]-dx[j,1]; ddz = dx[i,2]-dx[j,2]
            s = ddx*ddx + ddy*ddy + ddz*ddz
            if s > u_max:
                u_max = s
    u_max = np.sqrt(u_max)
    if u_max == 0.0:
        return max_t
    d0 = _pt_sq_nb(x0) if kind == 0 else _ee_sq_nb(x0)
    toi = 0.0
    eps = reduction*(np.sqrt(d0) - offset)
    target = eps + offset
    eps_sqr = eps*eps
    inv = 1.0/u_max
    xt = np.empty((C, 3))
    for _ in range(max_iter):
        for c in range(C):
            xt[c, 0] = x0[c, 0] + toi*dx[c, 0]
            xt[c, 1] = x0[c, 1] + toi*dx[c, 1]
            xt[c, 2] = x0[c, 2] + toi*dx[c, 2]
        d2 = _pt_sq_nb(xt) if kind == 0 else _ee_sq_nb(xt)
        dmt = (d2 - target*target)/(np.sqrt(d2) + target)
        if (max_t - toi)*u_max < dmt - eps:
            return max_t
        if toi > 0.0 and dmt*dmt < eps_sqr:
            break
        toi_next = toi + dmt*inv
        if toi_next == toi:
            break
        toi = toi_next
        if toi > max_t:
            return max_t
    return toi


@njit(cache=True)
def _centerize4(a, b, c, d):                               # 4 points (each 3,) → centered (4,3)
    out = np.empty((4, 3))
    for j in range(3):
        m = 0.25*(a[j] + b[j] + c[j] + d[j])
        out[0, j] = a[j] - m; out[1, j] = b[j] - m; out[2, j] = c[j] - m; out[3, j] = d[j] - m
    return out


@njit(cache=True)
def point_triangle_toi_nb(p0, p1, t00, t01, t02, t10, t11, t12, offset, max_t, reduction, max_iter):
    x0 = _centerize4(p0, t00, t01, t02)
    d0 = np.empty((4, 3))
    for j in range(3):
        dv0 = p1[j]-p0[j]; dv1 = t10[j]-t00[j]; dv2 = t11[j]-t01[j]; dv3 = t12[j]-t02[j]
        m = 0.25*(dv0 + dv1 + dv2 + dv3)
        d0[0, j] = dv0 - m; d0[1, j] = dv1 - m; d0[2, j] = dv2 - m; d0[3, j] = dv3 - m
    return _accd_toi_nb(x0, d0, 0, offset, max_t, reduction, max_iter)


@njit(cache=True)
def edge_edge_toi_nb(a00, a01, b00, b01, a10, a11, b10, b11, offset, max_t, reduction, max_iter):
    x0 = _centerize4(a00, a01, b00, b01)
    d0 = np.empty((4, 3))
    for j in range(3):
        dv0 = a10[j]-a00[j]; dv1 = a11[j]-a01[j]; dv2 = b10[j]-b00[j]; dv3 = b11[j]-b01[j]
        m = 0.25*(dv0 + dv1 + dv2 + dv3)
        d0[0, j] = dv0 - m; d0[1, j] = dv1 - m; d0[2, j] = dv2 - m; d0[3, j] = dv3 - m
    return _accd_toi_nb(x0, d0, 1, offset, max_t, reduction, max_iter)


@njit(cache=True)
def min_toi_vf_nb(Pc, Pn, T0c, T0n, T1c, T1n, T2c, T2n, offset, max_t, reduction, max_iter):
    """Min point-triangle ACCD toi over a batch of (vertex, face) candidate pairs (current `*c` and
    predicted `*n` positions, each (n_pair,3))."""
    m = max_t
    for i in range(Pc.shape[0]):
        t = point_triangle_toi_nb(Pc[i], Pn[i], T0c[i], T1c[i], T2c[i], T0n[i], T1n[i], T2n[i],
                                  offset, max_t, reduction, max_iter)
        if t < m:
            m = t
    return m


@njit(cache=True)
def min_toi_ee_nb(A0c, A0n, A1c, A1n, B0c, B0n, B1c, B1n, offset, max_t, reduction, max_iter):
    """Min edge-edge ACCD toi over a batch of (edge i, edge j) candidate pairs."""
    m = max_t
    for i in range(A0c.shape[0]):
        t = edge_edge_toi_nb(A0c[i], A1c[i], B0c[i], B1c[i], A0n[i], A1n[i], B0n[i], B1n[i],
                             offset, max_t, reduction, max_iter)
        if t < m:
            m = t
    return m
