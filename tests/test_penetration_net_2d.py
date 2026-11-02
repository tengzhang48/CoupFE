"""2D penetration safety net (`DeformableBarrierContact2D._any_penetration`).

It used to check only the closest edge's line gap with the foot on the edge, so a node inside a
concave corner, whose closest point is the corner node, was missed.
"""
import numpy as np

from coupfe.operators.contact import DeformableBarrierContact2D
from coupfe.operators.contact_search import candidate_pairs

DHAT = 0.1


def _op(nodes, edges, sec):
    return DeformableBarrierContact2D(nodes, np.array(sec), np.array(edges), dof_per_node=2,
                                      dhat=DHAT, kappa=1.0e3)


def _net(nodes, edges, sec):
    cands = candidate_pairs(nodes, np.array(sec), np.array(edges), DHAT)
    return _op(nodes, edges, sec)._any_penetration(np.zeros(nodes.size), cands)


def test_node_inside_a_concave_corner_is_caught():
    # an L-shaped solid (clockwise, outward normals at +90 deg); node 6 is inside it,
    # nearest to the inner corner (1, 1)
    X = np.array([[0, 0], [0, 4], [1, 4], [1, 1], [4, 1], [4, 0], [0.9, 0.8]], float)
    E = [[i, (i + 1) % 6] for i in range(6)]
    assert _net(X, E, [6])                                 # old code: False


def test_open_surface_end_and_outside_nodes_are_not_flagged():
    # an open top surface y = 0 (outward normal +y), three edges
    top = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    E = [[0, 1], [1, 2], [2, 3]]
    beside_end = np.array(top + [[3.05, -0.05]])           # beside the free end, below its line
    assert not _net(beside_end, E, [4])
    above = np.array(top + [[1.5, 0.05]])
    assert not _net(above, E, [4])
    above_node = np.array(top + [[1.0, 0.05]])
    assert not _net(above_node, E, [4])
    below = np.array(top + [[1.5, -0.05]])                 # behind an edge, foot on it
    assert _net(below, E, [4])


def test_outside_a_convex_tip_and_a_junction_are_not_flagged():
    # closed clockwise triangle with a sharp tip at (0, 0): a node just outside the tip has the
    # tip node as its closest point (the two-edge node-normal branch)
    tri = [[-0.01, -1.0], [0.0, 0.0], [0.01, -1.0]]
    E = [[0, 1], [1, 2], [2, 0]]
    X = np.array(tri + [[0.05, 0.001]])
    assert not _net(X, E, [3])
    # zero motion: the public step bound must stay 1 (a false alarm halves it to ~1e-9)
    assert _op(X, E, [3]).max_step(np.zeros(X.size), np.zeros(X.size)) == 1.0
    # a third edge into the tip: a junction, not a two-edge corner, so not flagged
    Xj = np.array(tri + [[0.005, -1.0], [0.05, 0.001]])
    assert not _net(Xj, E + [[3, 1]], [4])

