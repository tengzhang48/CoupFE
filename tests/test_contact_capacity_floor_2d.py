"""The 2D barrier's adaptive capacity stays finite on the edge line."""

import numpy as np
import pytest


def test_adaptive_capacity_is_finite_on_the_edge_line():
    """With mass, s = κ + M/d² gave NaN in R and K for a node on the
    edge line (d = 0, e.g. parts that start touching). The squared gap is floored at
    1e-24, as in rigid_barrier_eval; above the floor the result is unchanged."""
    import warnings
    from coupfe.operators.contact import deformable_barrier_eval
    Xa, Xb = np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])
    kw = dict(dhat=0.1, kappa=1.0e3, mass=np.array([1.0e-2]))
    for d in (0.0, 1.0e-170, -1.0e-14):
        with warnings.catch_warnings():
            warnings.simplefilter("error")                 # no division-by-zero warning either
            R, K, act = deformable_barrier_eval(np.array([[0.5, d]]), Xa, Xb, **kw)
            Rf, Kf, _ = deformable_barrier_eval(          # friction path, same capacity
                np.array([[0.5, d]]), Xa, Xb, mu=0.3, Xs0=np.array([[0.49, d]]),
                Xa0=Xa, Xb0=Xb, **kw)
        assert act[0] and np.all(np.isfinite(R)) and np.all(np.isfinite(K))
        assert np.all(np.isfinite(Rf)) and np.all(np.isfinite(Kf))
        # below the floor s is constant: R_y = -s g², pushing the node out; K_yy = 2 s g
        s_floor, g = 1.0e3 + 1.0e-2 / 1.0e-24, 0.1 - d
        assert R[0, 1] == pytest.approx(-s_floor * g * g, rel=1e-12)
        assert K[0, 1, 1] == pytest.approx(2.0 * s_floor * g, rel=1e-12)
    # above the floor: the unfloored formula, and the complex-step tangent still carries s′
    d = 1.0e-3
    R, K, _ = deformable_barrier_eval(np.array([[0.5, d]]), Xa, Xb, **kw)
    s = 1.0e3 + 1.0e-2 / d ** 2
    assert R[0, 1] == pytest.approx(-s * (0.1 - d) ** 2, rel=1e-14)
    ds = -2.0 * 1.0e-2 / d ** 3                            # s′(d)
    dRy = -(ds * (0.1 - d) ** 2 - 2.0 * s * (0.1 - d))      # ∂R_y/∂y_s
    assert K[0, 1, 1] == pytest.approx(dRy, rel=1e-10)
