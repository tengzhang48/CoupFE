"""2D node-to-segment barrier at edge ends.

A pair whose foot falls off its edge used to be switched off (the 0 <= xi <= 1 gate), with
three results:
no force in the wedge outside a convex corner, a force that switched off when a frozen node
passed its edge end, and a ~1e-8 zero-force interval just past a flat junction (a rounding
tie keeps the old edge, whose gate is off). Past an end the gap is now the distance to that
end node, which equals the line gap there for a separated node.
"""
import numpy as np
import pytest

from coupfe.operators.contact import DeformableBarrierContact2D, deformable_barrier_eval

DHAT, KAPPA = 0.1, 1.0e3


def _residual(op, U, ndof):
    r = op.residual(U, None, 0.0, 1.0)
    out = np.zeros(ndof)
    np.add.at(out, np.asarray(r.gdofs, dtype=int), np.asarray(r.values, dtype=float))
    return out


def test_no_dead_zone_above_a_convex_corner():
    # primary: a roof, slopes of 15 degrees down from V = (0, 0); the node right above V
    t = np.tan(np.radians(15.0))
    nodes = np.array([[-1.0, -t], [0.0, 0.0], [1.0, -t], [0.0, 0.05]])
    op = DeformableBarrierContact2D(nodes, np.array([3]), np.array([[0, 1], [1, 2]]),
                                    dof_per_node=2, dhat=DHAT, kappa=KAPPA)
    R = _residual(op, np.zeros(8), 8)
    # before the fix: R = 0 (both feet off their edges); now pushed straight up from V
    np.testing.assert_allclose(R[6:8], [0.0, -KAPPA * (DHAT - 0.05) ** 2], rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(R.reshape(-1, 2).sum(0), 0.0, atol=1e-15)   # action-reaction


@pytest.mark.parametrize("reverse", [False, True])
def test_no_zero_force_interval_past_a_flat_junction(reverse):
    edges = np.array([[0, 1], [1, 2]])
    if reverse:
        edges = edges[::-1].copy()
    h = 0.05
    # the interval is about sqrt(eps)*h ~ 7e-10 wide here: sample inside and outside it
    for s in (-2e-9, -5e-10, -1e-10, -1e-12, 0.0, 1e-12, 1e-10, 5e-10, 2e-9):
        nodes = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [s, h]])
        op = DeformableBarrierContact2D(nodes, np.array([3]), edges, dof_per_node=2,
                                        dhat=DHAT, kappa=KAPPA)
        Ry = _residual(op, np.zeros(8), 8)[7]
        # before the fix: 0 on one side of V for |s| below ~7e-10 (the side depends on the order)
        assert Ry == pytest.approx(-KAPPA * (DHAT - h) ** 2, rel=1e-12), s


def test_force_is_continuous_across_an_edge_end():
    xa, xb = np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])
    h, delta = 0.04, 1e-7
    kw = dict(dhat=DHAT, kappa=KAPPA, mass=np.array([1e-3]))
    R_in, _, a_in = deformable_barrier_eval(np.array([[1.0 - delta, h]]), xa, xb, **kw)
    R_out, _, a_out = deformable_barrier_eval(np.array([[1.0 + delta, h]]), xa, xb, **kw)
    assert a_in[0] and a_out[0]                            # before the fix: a_out[0] was False
    assert np.max(np.abs(R_out - R_in)) < 1e-5 * np.max(np.abs(R_in))


@pytest.mark.parametrize("x", [(-0.03, 0.02), (1.05, 0.03), (1.02, -0.01)])
def test_end_node_branch_tangent_matches_finite_differences(x):
    xa, xb = np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])
    kw = dict(dhat=DHAT, kappa=KAPPA, mass=np.array([1e-3]))
    X = np.concatenate([np.array([x]), xa, xb], axis=1)[0]

    def R_of(Xv):
        return deformable_barrier_eval(Xv[None, 0:2], Xv[None, 2:4], Xv[None, 4:6], **kw)[0][0]

    _, K, act = deformable_barrier_eval(X[None, 0:2], xa, xb, **kw)
    assert act[0]
    h = 1e-7                                               # small enough to stay on the branch
    FD = np.column_stack([(R_of(X + h * e) - R_of(X - h * e)) / (2 * h) for e in np.eye(6)])
    np.testing.assert_allclose(K[0], FD, rtol=1e-6, atol=1e-6 * np.max(np.abs(FD)))


def test_end_node_branch_reaction_and_friction():
    """Point-point branch: the force acts along (xs - end node), all reaction on that node
    (none on the other edge node, no torque); smoothed friction is perpendicular to it."""
    xa, xb = np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])
    xs = np.array([[1.03, 0.04]])                          # past xb, 0.05 from it
    R, _, act = deformable_barrier_eval(xs, xa, xb, dhat=DHAT, kappa=KAPPA)
    assert act[0]
    u = (xs[0] - xb[0]) / np.linalg.norm(xs[0] - xb[0])
    f = -R[0].reshape(3, 2)                                # physical forces on xs, xa, xb
    np.testing.assert_allclose(f[0], KAPPA * (DHAT - 0.05) ** 2 * u, rtol=1e-12)
    np.testing.assert_allclose(f[1], 0.0, atol=1e-15)
    np.testing.assert_allclose(f[2], -f[0], rtol=1e-12)
    # smoothed friction: slip along the edge since the step start
    Rf, _, _ = deformable_barrier_eval(xs, xa, xb, dhat=DHAT, kappa=KAPPA, mu=0.4,
                                       Xs0=xs - [[0.01, 0.0]], Xa0=xa, Xb0=xb)
    ff = -(Rf[0] - R[0]).reshape(3, 2)                     # friction part only
    assert abs(ff[0] @ u) < 1e-12 and np.linalg.norm(ff[0]) > 0.0
    np.testing.assert_allclose(ff[1], 0.0, atol=1e-15)
    np.testing.assert_allclose(ff[2], -ff[0], rtol=1e-12)


def test_node_exactly_on_the_end_node_is_finite():
    """A node on an end node has xi = 1 exactly: the on-edge branch (line gap 0, capacity
    floored by F2), finite with no warning. The end-node branch never sees a zero distance."""
    import warnings
    xa, xb = np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        R, K, act = deformable_barrier_eval(np.array([[1.0, 0.0]]), xa, xb, dhat=DHAT,
                                            kappa=KAPPA, mass=np.array([1e-3]))
    assert act[0] and np.all(np.isfinite(R)) and np.all(np.isfinite(K))
