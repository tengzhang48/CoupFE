"""2D pairing freeze: freeze ON ENTRY, per node.

``DeformableBarrierContact2D(freeze_pairing=True)`` used to cache the whole topology of the
step's FIRST evaluation, so a node that entered the barrier band later in the step had no
pair, and no force, until the next step.

Two flaws of the old cache, one test each (both fail on the old code):
- a node outside the broad-phase candidates at the first evaluation got no pair at all;
- a node inside the candidates but outside the band was frozen to its step-start closest
  edge, not to the edge it was closest to when it entered the band.
"""
import numpy as np

from coupfe.operators.contact import DeformableBarrierContact2D

DHAT, KAPPA = 0.1, 1.0e3
# primary: a flat surface y = 0 of two edges; secondaries A, B, C (see below)
NODES = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],      # P0, P1, P2
                  [0.5, 0.05],                             # A (node 3): in the band
                  [1.5, 0.50],                             # B (node 4): a broad-phase candidate, outside the band
                  [1.5, 4.00]])                            # C (node 5): not even a candidate
EDGES = np.array([[0, 1], [1, 2]])
SEC = np.array([3, 4, 5])
NDOF = 2 * len(NODES)


def _op(freeze, **kw):
    return DeformableBarrierContact2D(NODES, SEC, EDGES, dof_per_node=2, dhat=DHAT,
                                      kappa=KAPPA, freeze_pairing=freeze, **kw)


def _dense(op, U):
    r = op.residual(U, None, 0.0, 1.0)
    out = np.zeros(NDOF)
    np.add.at(out, np.asarray(r.gdofs, dtype=int), np.asarray(r.values, dtype=float))
    return out


def _moved(node, dx, dy, U=None):
    U = np.zeros(NDOF) if U is None else U.copy()
    U[2 * node] += dx
    U[2 * node + 1] += dy
    return U


def test_a_node_from_outside_the_candidates_is_paired_on_entry():
    frozen = _op(True)
    _dense(frozen, np.zeros(NDOF))                         # first evaluation: C has no candidate
    U1 = _moved(5, 0.0, -3.95)                             # C drops into the band (gap 0.05)
    c = slice(10, 12)
    R_fresh = _dense(_op(False), U1)
    assert np.linalg.norm(R_fresh[c]) > 0.0
    # old code: C had no pair for the rest of the step, R_frozen[c] == 0
    np.testing.assert_allclose(_dense(frozen, U1)[c], R_fresh[c], rtol=0, atol=1e-14)


def test_a_node_is_frozen_to_the_edge_it_enters_the_band_at():
    frozen = _op(True)
    _dense(frozen, np.zeros(NDOF))                         # B: closest edge P1-P2, outside the band
    U1 = _moved(4, -0.55, -0.45)                           # B enters the band above P0-P1 (x 0.95)
    b = slice(8, 10)
    R_fresh = _dense(_op(False), U1)
    assert np.linalg.norm(R_fresh[b]) > 0.0
    # old code: B stayed frozen to P1-P2, where it is past the edge end (xi < 0), so the
    # gate gave it no force; now it is paired to P0-P1 when it enters the band
    np.testing.assert_allclose(_dense(frozen, U1), R_fresh, rtol=0, atol=1e-14)


def test_a_paired_node_keeps_its_edge_until_commit():
    frozen = _op(True)
    _dense(frozen, np.zeros(NDOF))                         # A frozen to edge P0-P1
    U2 = _moved(3, 0.55, 0.0)                              # A at x = 1.05: closest edge is now P1-P2
    a = slice(6, 8)
    R_fresh = _dense(_op(False), U2)
    assert np.linalg.norm(R_fresh[a]) > 0.0
    # frozen to P0-P1, A is past its end (xi > 1): the gate switches its force off (the
    # edge-end jump recorded in the root-cause note, item (a) results)
    assert np.linalg.norm(_dense(frozen, U2)[a]) == 0.0
    frozen.commit(U2, None, 0.0, 1.0)                      # next step: re-paired
    np.testing.assert_allclose(_dense(frozen, U2), R_fresh, rtol=0, atol=1e-14)


def test_freeze_off_is_unchanged():
    op = _op(False)
    U1 = _moved(4, 0.0, -0.45)
    R1 = _dense(op, U1)
    R0 = _dense(op, np.zeros(NDOF))
    np.testing.assert_allclose(_dense(op, U1), R1, rtol=0, atol=0)
    assert np.linalg.norm(R0[8:10]) == 0.0 and np.linalg.norm(R1[8:10]) > 0.0


def test_a_paired_node_keeps_its_edge_when_it_leaves_and_reenters():
    frozen = _op(True)
    _dense(frozen, np.zeros(NDOF))                         # A frozen to P0-P1
    _dense(frozen, _moved(3, 0.0, 0.45))                   # A leaves the band
    U3 = _moved(3, 0.55, 0.0)                              # back in the band, above P1-P2
    assert np.linalg.norm(_dense(frozen, U3)[6:8]) == 0.0  # still P0-P1 (past its end: off)


def test_a_deeply_penetrated_node_is_paired_and_pushed_out():
    # A below P0-P1 by 0.15 > d̂: clamped distance 0.15 is outside the band, but the signed
    # gap is negative with the foot on the edge, so the pair is active (recovery force)
    U = _moved(3, 0.0, -0.20)
    frozen, fresh = _op(True), _op(False)
    R_fresh = _dense(fresh, U)
    assert R_fresh[7] < 0.0                                # residual -f: the force pushes A up
    np.testing.assert_allclose(_dense(frozen, U), R_fresh, rtol=0, atol=1e-14)


def test_all_primitive_ignores_the_freeze():
    # old code froze the all-primitive candidate list, missing edges reached later
    ap_frozen = _op(True, all_primitive=True)
    _dense(ap_frozen, np.zeros(NDOF))
    U1 = _moved(5, 0.0, -3.95)                             # C enters the band
    np.testing.assert_allclose(_dense(ap_frozen, U1), _dense(_op(False, all_primitive=True), U1),
                               rtol=0, atol=1e-14)

