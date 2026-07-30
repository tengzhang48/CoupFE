"""2D all-primitive cubic barrier (the 2D ppf port) — `edge_barrier_eval` + the operator's
`all_primitive` mode.

The all-primitive barrier sums a smooth cubic over EVERY nearby (vertex, edge) pair via the ported
`point_edge_coeff_unclassified` (which clamps to an endpoint, covering point-edge AND point-point) —
the faithful 2D analogue of `contact3d.tri_barrier_eval`. Unlike node-to-segment it makes no
single-closest-edge choice, so there is nothing to flip at a vertex (no freeze needed). Gates:
(1) one-edge case agrees with node-to-segment; (2) at a corner a node sees BOTH edges.
"""
from __future__ import annotations

import numpy as np

from coupfe.operators.contact import DeformableBarrierContact2D, edge_barrier_eval


def _force(op, nodes):
    R = op.residual(np.zeros(len(nodes) * 2), None, 1.0, 1.0)
    r = np.zeros(len(nodes) * 2)
    np.add.at(r, R.gdofs, R.values)
    return r


def test_edge_barrier_interior():
    """Point above an edge interior (gap < d̂): repulsive normal force; the two edge nodes share the
    reaction equally (foot at the midpoint, w₀=w₁=½); Σforce=0."""
    P = np.array([[0.5, 0.3]]); A = np.array([[0.0, 0.0]]); B = np.array([[1.0, 0.0]])
    R, K, active = edge_barrier_eval(P, A, B, dhat=0.6, kappa=100.0)
    assert active[0]
    assert R[0, 1] < 0                                  # residual = -force; force_y on P > 0 (up)
    assert abs(R[0, 3] - R[0, 5]) < 1e-12               # A,B share reaction (w₀=w₁)
    assert abs(R.reshape(3, 2).sum(0)).max() < 1e-12    # Σforce = 0 (action-reaction)
    assert np.allclose(K[0], K[0].T) and np.linalg.eigvalsh(K[0]).min() > -1e-9   # PSD symmetric


def test_all_primitive_agrees_with_nts_single_edge():
    """One edge in the band → all-primitive == node-to-segment (same perpendicular cubic)."""
    nodes = np.array([[0., 0.], [1., 0.], [10., 0.], [11., 0.], [0.5, 0.3]])
    edges = np.array([[0, 1], [2, 3]])                  # 2nd edge far away
    sec = np.array([4])
    kw = dict(dof_per_node=2, comps=(0, 1), dhat=0.6, kappa=100.0)
    fa = _force(DeformableBarrierContact2D(nodes, sec, edges, all_primitive=True, **kw), nodes)
    fn = _force(DeformableBarrierContact2D(nodes, sec, edges, **kw), nodes)
    assert np.allclose(fa, fn, atol=1e-10), "AP must match node-to-segment when one edge is in range"


def test_point_edge_accd_prevents_crossing():
    """2D ACCD (point_edge_toi): a vertex driven straight through an edge in one step is stopped
    short — the gap→0 feasibility guarantee (so the M/d² barrier never overflows)."""
    from coupfe.operators.contact3d import point_edge_toi
    a = np.array([0., 0.]); b = np.array([1., 0.])          # static edge on y=0
    toi = point_edge_toi(np.array([0.5, 0.5]), np.array([0.5, -0.5]),  # P: y 0.5 → -0.5 (crosses)
                         a, a, b, b)
    assert 0.0 < toi < 1.0, "ACCD must clip the step that would cross the edge"
    assert 0.5 + toi * (-1.0) > 0.0, "after the clipped step the vertex stays above the edge (gap>0)"


def test_all_primitive_vertex_sees_both_edges():
    """At a right-angle corner (edges 0-1 ⟂ 1-2 meeting at vertex 1), a node near the corner pairs
    with BOTH edges under all-primitive, but only ONE under node-to-segment — the source of the
    closest-edge flip the freeze patches."""
    nodes = np.array([[-1., 0.], [0., 0.], [0., 1.], [0.2, 0.2]])
    edges = np.array([[0, 1], [1, 2]])
    sec = np.array([3])
    kw = dict(dof_per_node=2, comps=(0, 1), dhat=0.6, kappa=100.0)
    ap = DeformableBarrierContact2D(nodes, sec, edges, all_primitive=True, **kw)
    nts = DeformableBarrierContact2D(nodes, sec, edges, **kw)
    assert len(ap._pair_topology(np.zeros(8))) == 2, "all-primitive: node pairs with BOTH corner edges"
    assert len(nts._pair_topology(np.zeros(8))) == 1, "node-to-segment: a single closest edge"
    # the all-primitive force is finite and pushes the node away from the corner (+x,+y).
    # _force returns the RESIDUAL (= −force), so the force is its negation.
    force = -_force(ap, nodes).reshape(4, 2)[3]
    assert np.all(np.isfinite(force)) and force[0] > 0 and force[1] > 0
