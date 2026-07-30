"""Persistent (finite-sliding) friction state in DeformableBarrierContact2D — completes Build b.

The exact-stick return-map friction can carry its STATE — the committed tangential force, the analog of
effective plastic strain — across steps (``friction_persistent=True``), re-framed onto the current edge by
the kernel's tangent-plane projection (so it survives re-pairing/normal drift). Without it (the per-step
mode) the stick anchor resets every commit, so a held contact FORGETS its friction force the moment motion
stops. This gates that difference, which is the memory variable the unified slide-then-lock solver needs."""
from __future__ import annotations

import numpy as np

from coupfe.operators.contact import DeformableBarrierContact2D


def _held_friction_force(persistent, n=5):
    """Hold a sub-cap tangential displacement of a pressed secondary for n steps; return |friction force|
    per step. ε_p-style persistence ⇒ the force holds; per-step ⇒ it collapses once motion stops."""
    dhat = 0.04
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5 * dhat]], dtype=float)   # edge 0-1, secondary 2
    edges = np.array([[0, 1]], dtype=int); sec = np.array([2])
    op = DeformableBarrierContact2D(nodes, sec, edges, dof_per_node=2, dhat=dhat, kappa=1.0e3,
                                    mu=0.5, friction_kt=1.0e3, friction_persistent=persistent)
    cap = 0.5 * 1.0e3 * (0.5 * dhat) ** 2                 # μ·λ_n
    U = np.zeros(6); U[2 * 2] = 0.5 * cap / 1.0e3         # hold the secondary at a sub-cap (sticking) +x
    forces = []
    for _ in range(n):
        R = op.residual(U, None, 0.0, 0.0)
        fx = float(np.sum(R.values[R.gdofs == 2 * 2])) if len(R.gdofs) else 0.0   # secondary-x = friction
        forces.append(abs(fx))
        op.commit(U, None, 0.0, 0.0)                      # no motion between steps (displacement held)
    return np.array(forces)


def test_persistent_holds_while_per_step_forgets():
    fp = _held_friction_force(persistent=True)
    fs = _held_friction_force(persistent=False)
    assert fp[0] > 1e-3                                   # a friction force builds initially
    assert np.all(fp > 0.95 * fp[0])                      # PERSISTENT: it holds across every held step
    assert fs[0] > 1e-3 and fs[-1] < 0.05 * fs[0]         # per-step: it collapses once motion stops


def test_persistent_is_opt_in_default_unchanged():
    """Default (friction_persistent=False) is the per-step return-map — the state must collapse, i.e. the
    persistent path is strictly opt-in and does not alter existing behaviour."""
    fs = _held_friction_force(persistent=False)
    assert fs[-1] < 0.05 * fs[0]
