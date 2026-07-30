"""Return-map (radial-return) EXACT-STICK friction mode on DeformableBarrierContact2D (friction_kt=).

The default friction is ppf/IPC *smoothed* (a soft spring that creeps); this mode is the exact-cone
return-map: a stiff stick spring k_t projected onto the Coulomb cone. Gates the two properties the
smoothed mode cannot deliver:
  * the friction force is EXACT Coulomb — f = k_t·slip while inside the cone, locked at μλ_n in slip;
  * the slip-regime *consistent* tangent has ZERO stiffness in the saturated slip direction (it keeps
    the −t̂⊗t̂ term the smoothed Gauss-Newton drops), i.e. no plateau floor — whereas the smoothed mode
    has a nonzero floor μλ_n/u_t there (the broken-control contrast).
Plus: friction_kt is a no-op when μ=0.
"""
from __future__ import annotations

import numpy as np

from coupfe.operators.contact import deformable_barrier_eval

XA = np.array([[0.0, 0.0]])
XB = np.array([[1.0, 0.0]])
XS = np.array([[0.5, 0.02]])                       # secondary 0.02 above a horizontal edge
DHAT, KAPPA, MU = 0.04, 1.0e3, 0.4
_D = 0.02                                          # signed gap (e×r)/L for this geometry
LAM_N = KAPPA * (DHAT - _D) ** 2                   # barrier normal force (frozen): 0.4
CAP = MU * LAM_N                                   # Coulomb cap μλ_n: 0.16


def _eval(friction_kt, slip, mu=MU):
    Xs0 = XS - np.array([[slip, 0.0]])             # secondary slipped +x by `slip` since step start
    return deformable_barrier_eval(XS, XA, XB, dhat=DHAT, kappa=KAPPA, mu=mu,
                                   Xs0=Xs0, Xa0=XA, Xb0=XB, friction_kt=friction_kt)


def _friction_force(friction_kt, slip):
    """Friction force on the secondary = (with-friction − barrier-only) residual at its dofs."""
    R, _, _ = _eval(friction_kt, slip)
    R0, _, _ = _eval(friction_kt, slip, mu=0.0)    # μ=0 ⇒ barrier only
    return float(np.linalg.norm((R - R0)[0, :2]))


def test_return_map_exact_coulomb_force():
    kt = 1.0e3
    # stick: k_t·slip < cap  ⇒  |f| = k_t·slip (a stiff spring inside the cone)
    slip = 0.5 * CAP / kt
    assert abs(_friction_force(kt, slip) - kt * slip) < 1e-12
    # slip: k_t·slip > cap  ⇒  |f| = cap = μλ_n exactly (locked on the cone)
    slip = 3.0 * CAP / kt
    assert abs(_friction_force(kt, slip) - CAP) < 1e-12


def test_slip_tangent_has_no_plateau_floor_unlike_smoothed():
    kt = 1.0e3
    slip = 3.0 * CAP / kt                          # slip regime for both modes
    _, K_rm, _ = _eval(kt, slip)
    _, K_b, _ = _eval(kt, slip, mu=0.0)            # barrier-only tangent
    _, K_sm, _ = _eval(None, slip)                 # smoothed mode, same slip
    rm_xx = (K_rm - K_b)[0][0, 0]                  # return-map friction tangent, saturated x-x
    sm_xx = (K_sm - K_b)[0][0, 0]                  # smoothed friction tangent, same entry
    assert abs(rm_xx) < 1e-9                        # return-map: exactly zero (P − t̂⊗t̂ = 0 here)
    assert sm_xx > 0.5 * CAP / slip                 # smoothed: the nonzero plateau floor μλ_n/u_t


def test_friction_kt_is_noop_without_friction():
    R_k, K_k, _ = _eval(1.0e3, 0.05, mu=0.0)
    R_n, K_n, _ = _eval(None, 0.05, mu=0.0)
    assert np.array_equal(R_k, R_n) and np.array_equal(K_k, K_n)
