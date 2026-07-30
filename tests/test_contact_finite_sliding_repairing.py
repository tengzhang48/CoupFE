"""Finite-sliding RE-PAIRING in the real deformable contact operator (Build b foundation).

For large sliding the contact point migrates across primary edges, so the operator must re-pair (pick the
new closest edge) as the secondary slides — the kinematic foundation the persistent friction-state transfer
plugs into. This gates that DeformableBarrierContact2D re-pairs across a multi-edge slide and stays active
throughout. (The remaining piece of the unified finite-sliding exact-stick solver is carrying the friction
STATE — the committed tangential force / accumulated slip, the ε_p analog — across the re-pairing with the
contact frame, per docs/lessons_learned.md 2026-06-24.)"""
from __future__ import annotations

import numpy as np

from coupfe.operators.contact import DeformableBarrierContact2D


def test_operator_repairs_across_a_multi_edge_slide():
    M = 5                                                  # a 5-edge primary polyline along x at y=0
    nodes = np.array([[i, 0.0] for i in range(M + 1)] + [[0.0, 0.03]], dtype=float)
    edges = np.array([[i, i + 1] for i in range(M)], dtype=int)
    sec = np.array([M + 1])                                # the secondary node, in the contact band above
    op = DeformableBarrierContact2D(nodes, sec, edges, dof_per_node=2, dhat=0.05,
                                    kappa=1.0e3, mu=0.3, friction_kt=1.0e3)
    active_edge = []
    for xs in np.linspace(0.3, M - 0.3, 40):              # slide the secondary across all edges
        U = np.zeros(nodes.size); U[sec[0] * 2] = xs - nodes[sec[0], 0]
        p = op._pairs(U)
        active_edge.append(-1 if p is None else int(p[0][0][2] // 2))   # primary edge start-node
    active_edge = np.array(active_edge)
    live = active_edge[active_edge >= 0]
    assert np.all(active_edge >= 0)                        # contact stays active across the whole slide
    assert int(np.sum(np.diff(live) != 0)) >= 3           # the active edge migrates (re-pairing)
    assert live[0] < live[-1]                              # it walks monotonically across the edges
