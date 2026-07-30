"""3D port of the exact-stick return-map + persistent finite-sliding friction (DeformableBarrierContact3D).

The 3D contact gains the return-map (exact-cone) friction (``friction_kt``) and the persistent state
(``friction_persistent`` — the committed tangential force per surface vertex, the ε_p analog, carried
across steps and re-framed onto the current tangent plane at re-pairing), mirroring the 2D operator.
Smoothed friction (``friction_kt=None``) is unchanged and stays on the numba path."""
from __future__ import annotations

import numpy as np

from coupfe.operators.contact3d import DeformableBarrierContact3D


def _held_force_3d(persistent, hold_scale, n=5):
    """Press vertex 3 onto face (0,1,2), hold it at ``hold_scale·(cap/k_t)`` tangential, n steps; return
    |friction force| per step (the vertex-3 x reaction)."""
    dhat = 0.04
    nodes = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0.3, 0.3, 0.5 * dhat]], dtype=float)
    op = DeformableBarrierContact3D(nodes, np.array([3]), np.array([[0, 1, 2]]), np.zeros((0, 2), int),
                                    dof_per_node=3, dhat=dhat, kappa=1.0e3, mu=0.5,
                                    friction_kt=1.0e3, friction_persistent=persistent)
    cap = 0.5 * 1.0e3 * (0.5 * dhat) ** 2                  # μ·λ_n
    U = np.zeros(12); U[3 * 3 + 0] = hold_scale * cap / 1.0e3   # hold vertex-3 in +x
    f = []
    for _ in range(n):
        R = op.residual(U, None, 0.0, 0.0)
        fx = float(np.sum(R.values[R.gdofs == 3 * 3 + 0])) if len(R.gdofs) else 0.0
        f.append(abs(fx))
        op.commit(U, None, 0.0, 0.0)
    return np.array(f), cap


def test_persistent_holds_while_per_step_forgets_3d():
    fp, _ = _held_force_3d(persistent=True, hold_scale=0.5)       # sub-cap → sticks
    fs, _ = _held_force_3d(persistent=False, hold_scale=0.5)
    assert fp[0] > 1e-3
    assert np.all(fp > 0.95 * fp[0])                              # PERSISTENT: holds across every step
    assert fs[0] > 1e-3 and fs[-1] < 0.05 * fs[0]                 # per-step: collapses once motion stops


def test_returnmap_respects_the_cone_3d():
    """A super-cap held displacement saturates the friction force at the exact cone μλ_n (slip)."""
    fp, cap = _held_force_3d(persistent=True, hold_scale=3.0)     # super-cap → slips, force = μλ_n
    assert abs(fp[0] - cap) < 1e-9
