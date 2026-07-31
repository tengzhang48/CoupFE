"""Lumped-mass inertia operator for implicit (backward-Euler) dynamics.

Implicit dynamics can regularize difficult stick/slip and active-set
transitions. With a staged ramp, hold, and explicit residual/kinetic-energy
checks, backward-Euler damping can also support a scoped dynamic-relaxation
study. This operator supplies the inertia term through the same
`(residual, tangent, commit)` contract; `solve_dynamics` provides the driver.
"""

from __future__ import annotations

import numpy as np

from coupfe.operators.base import Residual, Tangent


def lumped_mass(coords, elems, density, dof_per_node, comps=None):
    """Lumped (row-summed) mass per global DOF for 2D Quad4: ``density × element area``
    distributed equally to the element's nodes, on the spatial ``comps``. Returns ``(ndof,)``
    (0 on non-``comps`` DOFs). Setup-time helper for :class:`InertiaOperator`."""
    coords = np.asarray(coords, dtype=float)
    elems = np.asarray(elems, dtype=int)
    n_node, nne = coords.shape[0], elems.shape[1]
    comps = (np.arange(dof_per_node) if comps is None else np.asarray(comps, dtype=int))
    xe = coords[elems]                                # (ne, nne, 2)
    xs, ys = xe[..., 0], xe[..., 1]
    area = 0.5 * np.abs(np.sum(xs * np.roll(ys, -1, axis=1)
                               - np.roll(xs, -1, axis=1) * ys, axis=1))   # shoelace (ne,)
    nodal = np.zeros(n_node)
    np.add.at(nodal, elems.ravel(), np.repeat(density * area / nne, nne))
    M = np.zeros(n_node * dof_per_node)
    for c in comps:
        M[int(c)::dof_per_node] = nodal
    return M


class InertiaOperator:
    """Backward-Euler inertia: residual ``M/dt²·(u − û)`` and tangent ``M/dt²`` (diagonal),
    where ``û = u_prev + dt·v_prev`` is the inertial predictor. Holds the dynamic state
    (``u_prev``, ``v_prev``); ``commit`` advances ``v = (u − u_prev)/dt`` then ``u_prev = u``.

    ``mass`` is the **lumped** mass per global DOF (0 for non-inertial DOFs — e.g. coupled
    scalar fields — which then contribute nothing).
    """

    def __init__(self, mass, ndof, *, u0=None, v0=None, damping=0.0):
        self.M = np.asarray(mass, dtype=float)
        self.ndof = int(ndof)
        self.u_prev = (np.zeros(ndof) if u0 is None else np.asarray(u0, dtype=float).copy())
        self.v_prev = (np.zeros(ndof) if v0 is None else np.asarray(v0, dtype=float).copy())
        self.damping = float(damping)                 # mass-proportional (Rayleigh α): C = αM
        self._idx = np.nonzero(self.M != 0.0)[0]      # only inertial DOFs enter the system
        self._m = self.M[self._idx]

    def predictor(self, dt):
        """The inertial predictor ``û = u_prev + dt·v_prev`` (a good Newton warm start)."""
        return self.u_prev + dt * self.v_prev

    def residual(self, U, state, t, dt) -> Residual:
        U = np.asarray(U, dtype=float)
        uhat = self.u_prev + dt * self.v_prev
        r = (self._m / dt ** 2) * (U[self._idx] - uhat[self._idx])
        if self.damping:                              # + αM·v (v = (u - u_prev)/dt)
            r = r + self.damping * self._m * (U[self._idx] - self.u_prev[self._idx]) / dt
        return Residual(self._idx, r)

    def tangent(self, U, state, t, dt) -> Tangent:
        d = self._m / dt ** 2
        if self.damping:
            d = d + self.damping * self._m / dt
        return Tangent(self._idx, self._idx, d)

    def commit(self, U, state, t, dt):
        U = np.asarray(U, dtype=float)
        self.v_prev = (U - self.u_prev) / dt          # backward-Euler velocity update
        self.u_prev = U.copy()
        return state
