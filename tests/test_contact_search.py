"""Broad-phase contact search (S1) — uniform spatial hash.

The correctness oracle is the brute-force within-`dhat` set: the hash result must be a **superset**
(no contact missed), exclude incident edges, and be O(N) not O(N²) in candidate work.
"""

import numpy as np

from coupfe.operators.contact_search import candidate_pairs


def _point_segment_dist(p, a, b):
    e = b - a
    L2 = float(e @ e)
    t = 0.0 if L2 == 0.0 else min(1.0, max(0.0, float((p - a) @ e) / L2))
    return float(np.linalg.norm(p - (a + t * e)))


def _brute_within(positions, vertices, edges, dhat, exclude_incident=True):
    """Ground truth: {v: set(edge_idx)} of edges within dhat of vertex v (point-segment)."""
    truth = {}
    for v in vertices:
        for ei, (a, b) in enumerate(edges):
            if exclude_incident and (v == a or v == b):
                continue
            if _point_segment_dist(positions[v], positions[a], positions[b]) <= dhat:
                truth.setdefault(int(v), set()).add(ei)
    return truth


def _grid_mesh(nx, ny, h=1.0):
    """A grid of nodes + the horizontal/vertical edges between neighbors (a surface-ish soup)."""
    xs = np.arange(nx) * h
    ys = np.arange(ny) * h
    nodes = np.array([(x, y) for y in ys for x in xs], dtype=float)
    edges = []
    idx = lambda i, j: j * nx + i
    for j in range(ny):
        for i in range(nx):
            if i + 1 < nx:
                edges.append((idx(i, j), idx(i + 1, j)))
            if j + 1 < ny:
                edges.append((idx(i, j), idx(i, j + 1)))
    return nodes, np.array(edges, dtype=int)


def _assert_superset(positions, vertices, edges, dhat):
    cand = candidate_pairs(positions, vertices, edges, dhat)
    truth = _brute_within(positions, vertices, edges, dhat)
    for v, true_edges in truth.items():
        got = set(cand.get(v, np.array([], dtype=int)).tolist())
        missing = true_edges - got
        assert not missing, f"vertex {v}: broad-phase MISSED within-dhat edges {missing}"
    return cand, truth


def test_superset_of_bruteforce_uniform():
    nodes, edges = _grid_mesh(6, 6, h=1.0)
    verts = np.arange(len(nodes))
    for dhat in (0.1, 0.5, 1.2):
        _assert_superset(nodes, verts, edges, dhat)


def test_superset_under_perturbation():
    """Random jitter (non-grid-aligned) — the superset guarantee must still hold."""
    rng = np.random.default_rng(0)
    nodes, edges = _grid_mesh(5, 5, h=1.0)
    nodes = nodes + rng.normal(scale=0.15, size=nodes.shape)
    verts = np.arange(len(nodes))
    for dhat in (0.2, 0.6):
        _assert_superset(nodes, verts, edges, dhat)


def test_incident_edges_excluded():
    nodes, edges = _grid_mesh(4, 4, h=1.0)
    cand = candidate_pairs(nodes, np.arange(len(nodes)), edges, dhat=2.0)
    for v, ce in cand.items():
        assert np.all((edges[ce, 0] != v) & (edges[ce, 1] != v))


def test_two_separated_bodies_no_cross_candidates_when_far():
    """Two grids far apart (gap ≫ dhat): no candidate pair bridges them."""
    a_nodes, a_edges = _grid_mesh(3, 3, h=1.0)
    b_nodes, b_edges = _grid_mesh(3, 3, h=1.0)
    b_nodes = b_nodes + np.array([10.0, 0.0])             # far away
    nodes = np.vstack([a_nodes, b_nodes])
    nA = len(a_nodes)
    edges = np.vstack([a_edges, b_edges + nA])
    cand = candidate_pairs(nodes, np.arange(len(nodes)), edges, dhat=0.5)
    for v, ce in cand.items():
        body_v = 0 if v < nA else 1
        for e in ce:
            body_e = 0 if edges[e, 0] < nA else 1
            assert body_v == body_e, "broad-phase bridged two far-apart bodies"


def test_scales_linearly_not_quadratically():
    """Total candidate work grows ~O(N), not O(N²) — the whole point of the broad phase."""
    counts = []
    for n in (8, 16, 32):
        nodes, edges = _grid_mesh(n, n, h=1.0)
        cand = candidate_pairs(nodes, np.arange(len(nodes)), edges, dhat=0.5)
        total = int(sum(len(c) for c in cand.values()))   # candidate (vertex,edge) checks
        brute = len(nodes) * len(edges)                    # what O(N²) would scan
        counts.append((len(nodes), total, brute))
    # candidate work per node is bounded (local), so total ~ c·N while brute ~ N²·const
    (n0, t0, b0), (n2, t2, b2) = counts[0], counts[-1]
    assert t2 / t0 < 3.0 * (n2 / n0)                       # ~linear, not quadratic
    assert t2 < 0.1 * b2                                   # far below the brute-force scan


def test_penalty_operators_broadphase_equals_bruteforce():
    """The broad-phase penalty operators give the SAME residual as a brute-force band (search_band
    huge) — the spatial hash only prunes, it doesn't change the active set/forces."""
    from coupfe.operators.contact import DeformableContact2D, SurfaceContact2D

    # DeformableContact2D: secondary node penetrating a primary edge (g<0)
    X = np.array([[0.5, -0.02], [0.0, 0.0], [1.0, 0.0]])
    bp = DeformableContact2D(X, [0], [[1, 2]], dof_per_node=2, k=1.0e3)            # default band
    bf = DeformableContact2D(X, [0], [[1, 2]], dof_per_node=2, k=1.0e3, search_band=1e9)
    U = np.zeros(6)
    rbp, rbf = bp.residual(U, None, 0, 0), bf.residual(U, None, 0, 0)
    assert np.array_equal(rbp.gdofs, rbf.gdofs) and np.allclose(rbp.values, rbf.values)
    assert rbp.values.size > 0                                                     # contact active

    # SurfaceContact2D: a vertex penetrating a non-adjacent edge
    Xs = np.array([[0.5, -0.02], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    edges = [[1, 2], [1, 3]]
    sbp = SurfaceContact2D(Xs, [0, 1, 2, 3], edges, dof_per_node=2, k=1.0e3)
    sbf = SurfaceContact2D(Xs, [0, 1, 2, 3], edges, dof_per_node=2, k=1.0e3, search_band=1e9)
    U2 = np.zeros(8)
    a_bp = sorted(t[1:3] for t in sbp._active(U2))
    a_bf = sorted(t[1:3] for t in sbf._active(U2))
    assert a_bp == a_bf                                                            # same active set
