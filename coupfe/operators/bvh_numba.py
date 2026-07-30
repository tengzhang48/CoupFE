"""numba-native bounding-volume hierarchy for contact broad-phase.

Ports the *structure* of ppf-contact-solver's LBVH (`contact/aabb.hpp` + `lbvh/`): flat AABB nodes +
a stack-based box-overlap query (`aabb::query` / `aabb::overlap`). The BUILD is a **Morton-sorted
midpoint split** — a simple, correct, numba-native simplification of ppf's Karras radix-LCP build
(the Morton sort gives spatial locality; midpoint splits a balanced tree; the GPU Karras build is the
escalation, and the node layout + query here are deliberately ppf-compatible so that port is direct).

Why BVH over the uniform grid: a grid degrades on **non-uniform** meshes (varying element size /
clustered geometry); a BVH adapts to the actual distribution. Returns a conservative **superset** of
AABB-overlapping primitives (caller applies the exact distance prune + narrow phase) — gated for
**0 misses** vs brute force (uniform / non-uniform / duplicate-code / degenerate).

This implementation is a modified NumPy/numba adaptation. Upstream provenance
and licensing are recorded in the repository ``NOTICE`` file.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _expand_bits(v):
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v


@njit(cache=True)
def _morton(x, y, z):                                    # x,y,z in [0,1] → 30-bit Morton code
    xi = min(max(int(x * 1024.0), 0), 1023)
    yi = min(max(int(y * 1024.0), 0), 1023)
    zi = min(max(int(z * 1024.0), 0), 1023)
    return _expand_bits(xi) * 4 + _expand_bits(yi) * 2 + _expand_bits(zi)


@njit(cache=True)
def build_bvh(plo, phi, leaf_size):
    """Build over primitive AABBs ``(plo, phi)`` each (n,3). Returns ``(order, nlo, nhi, nleft, nright,
    nstart, nend)`` — flat node arrays; ``nleft==-1`` marks a leaf holding ``order[nstart:nend]``.

    ITERATIVE (work-stack) build, not recursive — avoids numba's recursive-function caching fragility
    (a recursive @njit + cache=True can stale-segfault). Phase 1 lays out the tree top-down (children
    get HIGHER node indices than their parent); phase 2 fills node AABBs bottom-up by a single reverse-
    index sweep (a child's index > its parent's, so children are done first)."""
    n = plo.shape[0]
    order = np.arange(n)
    if n > 1:
        cx = 0.5 * (plo[:, 0] + phi[:, 0]); cy = 0.5 * (plo[:, 1] + phi[:, 1]); cz = 0.5 * (plo[:, 2] + phi[:, 2])
        mnx = cx.min(); mxx = cx.max(); mny = cy.min(); mxy = cy.max(); mnz = cz.min(); mxz = cz.max()
        rx = mxx - mnx if mxx > mnx else 1.0
        ry = mxy - mny if mxy > mny else 1.0
        rz = mxz - mnz if mxz > mnz else 1.0
        codes = np.empty(n, dtype=np.int64)
        for i in range(n):
            codes[i] = _morton((cx[i] - mnx) / rx, (cy[i] - mny) / ry, (cz[i] - mnz) / rz)
        order = np.argsort(codes)
    maxn = 2 * n if n > 0 else 1
    nlo = np.empty((maxn, 3)); nhi = np.empty((maxn, 3))
    nleft = np.full(maxn, -1, np.int64); nright = np.full(maxn, -1, np.int64)
    nstart = np.full(maxn, -1, np.int64); nend = np.full(maxn, -1, np.int64)
    nn = 0
    if n > 0:
        # phase 1: tree structure (top-down, work-stack of node ranges into `order`)
        snode = np.empty(maxn, np.int64); slo = np.empty(maxn, np.int64); shi = np.empty(maxn, np.int64)
        snode[0] = 0; slo[0] = 0; shi[0] = n
        sp = 1; ctr = 1
        while sp > 0:
            sp -= 1; ni = snode[sp]; lo_i = slo[sp]; hi_i = shi[sp]
            if hi_i - lo_i <= leaf_size:
                nstart[ni] = lo_i; nend[ni] = hi_i; nleft[ni] = -1
            else:
                mid = (lo_i + hi_i) // 2
                l = ctr; r = ctr + 1; ctr += 2
                nleft[ni] = l; nright[ni] = r; nstart[ni] = -1
                snode[sp] = l; slo[sp] = lo_i; shi[sp] = mid; sp += 1
                snode[sp] = r; slo[sp] = mid; shi[sp] = hi_i; sp += 1
        nn = ctr
        # phase 2: node AABBs bottom-up (reverse index → children before parents)
        for ni in range(nn - 1, -1, -1):
            if nleft[ni] == -1:
                mnx = 1e300; mny = 1e300; mnz = 1e300; mxx = -1e300; mxy = -1e300; mxz = -1e300
                for k in range(nstart[ni], nend[ni]):
                    pr = order[k]
                    if plo[pr, 0] < mnx: mnx = plo[pr, 0]
                    if plo[pr, 1] < mny: mny = plo[pr, 1]
                    if plo[pr, 2] < mnz: mnz = plo[pr, 2]
                    if phi[pr, 0] > mxx: mxx = phi[pr, 0]
                    if phi[pr, 1] > mxy: mxy = phi[pr, 1]
                    if phi[pr, 2] > mxz: mxz = phi[pr, 2]
                nlo[ni, 0] = mnx; nlo[ni, 1] = mny; nlo[ni, 2] = mnz
                nhi[ni, 0] = mxx; nhi[ni, 1] = mxy; nhi[ni, 2] = mxz
            else:
                l = nleft[ni]; r = nright[ni]
                nlo[ni, 0] = min(nlo[l, 0], nlo[r, 0]); nlo[ni, 1] = min(nlo[l, 1], nlo[r, 1]); nlo[ni, 2] = min(nlo[l, 2], nlo[r, 2])
                nhi[ni, 0] = max(nhi[l, 0], nhi[r, 0]); nhi[ni, 1] = max(nhi[l, 1], nhi[r, 1]); nhi[ni, 2] = max(nhi[l, 2], nhi[r, 2])
    return order, nlo[:nn].copy(), nhi[:nn].copy(), nleft[:nn].copy(), nright[:nn].copy(), nstart[:nn].copy(), nend[:nn].copy()


@njit(cache=True)
def query_csr(qlo, qhi, plo, phi, order, nlo, nhi, nleft, nright, nstart, nend):
    """For each query box ``[qlo[i], qhi[i]]`` (both (nq,3)), the prim indices whose AABB ``(plo,phi)``
    overlaps it — EXACT (per-prim AABB test at leaves, so it equals brute-force overlap, not just the
    overlapping-leaf grouping). Two-pass (count, fill) → CSR. Stack depth 64 ⟂ tree depth (log2 n ≪ 64)."""
    nq = qlo.shape[0]
    cand_ptr = np.zeros(nq + 1, np.int64)
    stack = np.empty(64, np.int64)
    nnodes = nlo.shape[0]
    for q in range(nq):
        cnt = 0
        if nnodes > 0:
            sp = 1; stack[0] = 0
            while sp > 0:
                sp -= 1; ni = stack[sp]
                if (qlo[q, 0] <= nhi[ni, 0] and qhi[q, 0] >= nlo[ni, 0] and
                        qlo[q, 1] <= nhi[ni, 1] and qhi[q, 1] >= nlo[ni, 1] and
                        qlo[q, 2] <= nhi[ni, 2] and qhi[q, 2] >= nlo[ni, 2]):
                    if nleft[ni] == -1:
                        for k in range(nstart[ni], nend[ni]):
                            pr = order[k]
                            if (qlo[q, 0] <= phi[pr, 0] and qhi[q, 0] >= plo[pr, 0] and
                                    qlo[q, 1] <= phi[pr, 1] and qhi[q, 1] >= plo[pr, 1] and
                                    qlo[q, 2] <= phi[pr, 2] and qhi[q, 2] >= plo[pr, 2]):
                                cnt += 1
                    else:
                        stack[sp] = nleft[ni]; sp += 1; stack[sp] = nright[ni]; sp += 1
        cand_ptr[q + 1] = cand_ptr[q] + cnt
    cand = np.empty(cand_ptr[nq], np.int64)
    for q in range(nq):
        w = cand_ptr[q]
        if nnodes > 0:
            sp = 1; stack[0] = 0
            while sp > 0:
                sp -= 1; ni = stack[sp]
                if (qlo[q, 0] <= nhi[ni, 0] and qhi[q, 0] >= nlo[ni, 0] and
                        qlo[q, 1] <= nhi[ni, 1] and qhi[q, 1] >= nlo[ni, 1] and
                        qlo[q, 2] <= nhi[ni, 2] and qhi[q, 2] >= nlo[ni, 2]):
                    if nleft[ni] == -1:
                        for k in range(nstart[ni], nend[ni]):
                            pr = order[k]
                            if (qlo[q, 0] <= phi[pr, 0] and qhi[q, 0] >= plo[pr, 0] and
                                    qlo[q, 1] <= phi[pr, 1] and qhi[q, 1] >= plo[pr, 1] and
                                    qlo[q, 2] <= phi[pr, 2] and qhi[q, 2] >= plo[pr, 2]):
                                cand[w] = pr; w += 1
                    else:
                        stack[sp] = nleft[ni]; sp += 1; stack[sp] = nright[ni]; sp += 1
    return cand, cand_ptr
