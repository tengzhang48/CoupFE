"""Implicit (backward-Euler) dynamics driver — the substrate for robust contact.

Two independent checks: (1) free fall reproduces the exact velocity ``v = −g·t`` (and ≈ −½g t²);
(2) with a load held, the dissipative integrator settles to the **quasistatic** equilibrium
``u = F/k`` — i.e. dynamics doubles as dynamic relaxation. No f2py needed (simple operators).
"""

import numpy as np

from coupfe import InertiaOperator, solve_dynamics
from coupfe.operators.base import Residual, Tangent


class LinearSpring:
    """Internal force ``R = k·u`` (restoring), constant tangent ``k``."""

    def __init__(self, k, gd):
        self.k = float(k)
        self.gd = np.asarray(gd, dtype=int)

    def residual(self, U, state, t, dt):
        return Residual(self.gd, self.k * np.asarray(U, dtype=float)[self.gd])

    def tangent(self, U, state, t, dt):
        return Tangent(self.gd, self.gd, np.full(len(self.gd), self.k))

    def commit(self, U, state, t, dt):
        return state


class ConstForce:
    """Constant external force ``f`` on dofs ``gd`` (residual contribution ``−f``)."""

    def __init__(self, gd, f):
        self.gd = np.asarray(gd, dtype=int)
        self.f = np.asarray(f, dtype=float)

    def residual(self, U, state, t, dt):
        return Residual(self.gd, -self.f)

    def tangent(self, U, state, t, dt):
        return Tangent(np.array([], dtype=int), np.array([], dtype=int), np.array([]))

    def commit(self, U, state, t, dt):
        return state


def test_free_fall_velocity_is_exact():
    m, g, dt, n = 2.0, 9.81, 1e-3, 1000          # t = 1.0 s
    inertia = InertiaOperator([m], 1)
    gravity = ConstForce([0], [-m * g])
    U, info = solve_dynamics([inertia, gravity], np.zeros(1), 1, lambda t: {},
                             dt=dt, n_steps=n)
    t = n * dt
    assert abs(inertia.v_prev[0] - (-g * t)) < 1e-9      # backward-Euler: v = -g t EXACT
    assert abs(U[0] - (-0.5 * g * t ** 2)) < 1e-2        # u ≈ -½ g t² (O(dt))
    assert info["t"] == t


def test_dynamic_relaxation_reaches_quasistatic_equilibrium():
    # mass-spring under a constant load: the dissipative integrator settles to u = F/k
    m, k, F = 1.0, 100.0, 10.0
    u_eq = F / k                                          # 0.1
    inertia = InertiaOperator([m], 1)
    spring = LinearSpring(k, [0])
    load = ConstForce([0], [F])
    U, _ = solve_dynamics([inertia, spring, load], np.zeros(1), 1, {}, dt=0.02, n_steps=500)
    assert abs(U[0] - u_eq) < 1e-3                        # settled to the static solution
    assert abs(inertia.v_prev[0]) < 1e-2                  # ...and at rest


def test_inertia_predictor_and_zero_mass_dofs():
    # zero-mass dofs contribute nothing; predictor = u_prev + dt v_prev
    op = InertiaOperator([0.0, 3.0], 2, u0=[1.0, 2.0], v0=[0.0, 5.0])
    assert op._idx.tolist() == [1]                        # only the inertial dof enters
    assert np.allclose(op.predictor(0.1), [1.0, 2.5])
    r = op.residual(np.array([1.0, 2.0]), None, 1.0, 0.1)
    assert r.gdofs.tolist() == [1]                        # no contribution from dof 0


def test_adaptive_timestep_matches_fixed_and_saves_steps():
    """Adaptive dynamics must agree with the fixed-step oracle on a smooth problem,
    while growing the step when convergence is easy."""
    from coupfe import solve_dynamics_adaptive

    m, g = 2.0, 9.81
    t_end = 1.0
    inertia = InertiaOperator([m], 1)
    gravity = ConstForce([0], [-m * g])

    # fixed oracle with the same dt the adaptive run starts from and is capped at
    dt_oracle = 0.01
    U_fixed, info_fixed = solve_dynamics([inertia, gravity], np.zeros(1), 1,
                                          lambda t: {}, dt=dt_oracle,
                                          n_steps=int(round(t_end / dt_oracle)))
    # adaptive run with identical max dt: it should reproduce the fixed solution
    inertia_adapt = InertiaOperator([m], 1)
    U_adapt, info_adapt = solve_dynamics_adaptive(
        [inertia_adapt, gravity], np.zeros(1), 1, lambda t: {},
        t_end=t_end, dt_init=dt_oracle, dt_max=dt_oracle, growth=1.5,
        cut=0.5, maxit=20, rtol=1e-9
    )
    assert info_adapt["t"] == t_end
    assert abs(U_adapt[0] - U_fixed[0]) < 1e-9
    assert abs(inertia_adapt.v_prev[0] - inertia.v_prev[0]) < 1e-9
    assert info_adapt["steps"] == info_fixed["steps"]

    # a smaller initial dt must be grown up to dt_max, producing fewer steps than
    # a fixed run at the initial dt while still matching the physical result
    inertia_adapt2 = InertiaOperator([m], 1)
    U_adapt2, info_adapt2 = solve_dynamics_adaptive(
        [inertia_adapt2, gravity], np.zeros(1), 1, lambda t: {},
        t_end=t_end, dt_init=dt_oracle / 4, dt_max=dt_oracle, growth=1.5,
        cut=0.5, maxit=20, rtol=1e-9
    )
    assert info_adapt2["t"] == t_end
    assert abs(U_adapt2[0] - U_fixed[0]) < 1e-2           # O(dt_max) agreement
    assert abs(inertia_adapt2.v_prev[0] - inertia.v_prev[0]) < 1e-6
    assert info_adapt2["steps"] < int(round(t_end / (dt_oracle / 4)))
    assert info_adapt2["n_grows"] > 0


def test_adaptive_dynamics_handles_contact_impact():
    """Adaptive timestepping must cut dt when a contact operator cannot handle
    a large step, and still keep the solution stable / penetration-free."""
    from coupfe import solve_dynamics_adaptive

    class DtSensitiveFloor:
        """1-D penalty floor that raises if the timestep is too large.

        This is a stand-in for a hard contact/impact where the CCD bound or
        the nonlinear solve only becomes reliable below a critical dt.  The
        adaptive driver must cut dt until the operator is happy.
        """

        def __init__(self, k=1.0e4, dt_max=0.01):
            self.k = float(k)
            self.dt_max = float(dt_max)

        def residual(self, U, state, t, dt):
            if dt > self.dt_max:
                raise RuntimeError("timestep too large for contact operator")
            x = float(U[0])
            if x >= 0.0:
                return Residual(np.array([0], dtype=int), np.array([0.0]))
            return Residual(np.array([0], dtype=int), np.array([-self.k * x]))

        def tangent(self, U, state, t, dt):
            x = float(U[0])
            return Tangent(np.array([0], dtype=int), np.array([0], dtype=int),
                           np.array([0.0 if x >= 0.0 else -self.k]))

        def commit(self, U, state, t, dt):
            return state

        def max_step(self, U, dU):
            if dU[0] >= 0.0:
                return 1.0
            return max(0.0, min(1.0, -0.99 * U[0] / dU[0]))

    m = 1.0
    inertia = InertiaOperator([m], 1, u0=[1.0], v0=[-10.0])   # drop toward floor
    floor = DtSensitiveFloor(k=1.0e4, dt_max=0.01)

    U, info = solve_dynamics_adaptive(
        [inertia, floor], np.array([1.0]), 1, {},
        t_end=0.5, dt_init=0.05, dt_min=1e-6, dt_max=0.05,
        growth=1.2, cut=0.25, maxit=15, rtol=1e-8
    )
    assert abs(info["t"] - 0.5) < 1.0e-12
    assert np.isfinite(U).all()
    assert U[0] >= -1e-6                                      # no penetration
    assert info["n_cuts"] > 0                                 # large steps forced cuts
