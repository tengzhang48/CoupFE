"""Contact broad-phase search — uniform spatial hash (Stage S1).

Replaces the brute-force O(N²) "every vertex × every edge" candidate scan in the contact operators
with an O(N) spatial hash for roughly-uniform feature sizes (FE surfaces). Self-contained — geometry
in → candidate ``(vertex, edge)`` pairs out, no PETSc/MPI state — so it drops into the serial
operators now and becomes the local kernel of the distributed spatial layer later. BVH is the
escalation for highly non-uniform meshes; numba is the acceleration for the hot loop
(`docs/dev/contact.md` → "Broad-phase search").

**Conservative by construction:** each edge is binned into every cell its ``dhat``-expanded AABB
overlaps, so a vertex within ``dhat`` of an edge always lands in a cell holding that edge — the
result is a **superset** of the true within-``dhat`` set (no contact is ever missed; the brute-force
result is the correctness oracle). The narrow phase (closest point, active set, force) is unchanged.
"""

from __future__ import annotations

import numpy as np


def candidate_pairs(positions, vertices, edges, dhat, *, exclude_incident=True, cell=None):
    """Uniform spatial-hash broad phase.

    Args:
        positions: ``(n_node, dim)`` current nodal positions.
        vertices: ``(n_v,)`` surface-vertex node ids to query.
        edges: ``(n_e, 2)`` surface-edge node-id pairs.
        dhat: contact band — candidates are guaranteed to include every vertex within ``dhat``
            of an edge (Euclidean point-segment distance).
        exclude_incident: drop edges incident to the query vertex (self-contact: a vertex never
            contacts its own edges).
        cell: grid cell size; default ``max(max_edge_length, dhat)`` (an edge then spans O(1) cells).

    Returns:
        ``dict {vertex_id: np.ndarray of edge indices into ``edges``}`` — the candidate edges per
        vertex (a **superset** of the true within-``dhat`` set). Vertices with no candidate are
        absent from the dict.
    """
    X = np.asarray(positions, dtype=float)
    vertices = np.asarray(vertices, dtype=int)
    edges = np.asarray(edges, dtype=int)
    if len(edges) == 0 or len(vertices) == 0:
        return {}
    xe = X[edges]                                          # (n_e, 2, dim)

    if cell is None:
        elen = np.sqrt(((xe[:, 1] - xe[:, 0]) ** 2).sum(-1))
        cell = float(max(elen.max(), dhat))
    cell = max(cell, 1e-30)
    inv = 1.0 / cell

    # bin each edge into all cells its (AABB + dhat) overlaps (so no within-dhat vertex is missed)
    lo = xe.min(axis=1) - dhat                             # (n_e, dim)
    hi = xe.max(axis=1) + dhat
    cl = np.floor(lo * inv).astype(np.int64)              # (n_e, dim) lower cell index
    ch = np.floor(hi * inv).astype(np.int64)              # upper cell index
    grid: dict[tuple, list] = {}
    for e in range(len(edges)):
        for i in range(int(cl[e, 0]), int(ch[e, 0]) + 1):
            for j in range(int(cl[e, 1]), int(ch[e, 1]) + 1):
                grid.setdefault((i, j), []).append(e)

    # query each vertex's own cell (the edge's expanded AABB already covers the dhat reach)
    vcell = np.floor(X[vertices] * inv).astype(np.int64)
    out: dict[int, np.ndarray] = {}
    for k in range(len(vertices)):
        v = int(vertices[k])
        cands = grid.get((int(vcell[k, 0]), int(vcell[k, 1])))
        if not cands:
            continue
        c = np.asarray(cands, dtype=int)
        if exclude_incident:
            c = c[(edges[c, 0] != v) & (edges[c, 1] != v)]
        if len(c):
            out[v] = c
    return out
