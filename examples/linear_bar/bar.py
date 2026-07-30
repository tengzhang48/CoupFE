"""A chain of nonlinear 1D bar elements — the smallest real CoupFE operator.

It demonstrates the package pattern end to end:

* **one residual is the source of truth** — ``_elem_residual`` is the only place
  the physics lives;
* the **tangent is derived from it by complex step** (no hand-coded stiffness);
* an **analytic tangent is provided only as a test oracle**, so the harness-style
  cross-check ``complex-step == analytic`` validates the pattern.

Energy density ``w(eps) = 1/2 E eps^2 + 1/3 H eps^3`` → axial stress
``sigma = E eps + H eps^2`` (a deliberately nonlinear spring).  A tip load is
ramped by the pseudo-time ``t``.
"""

from __future__ import annotations

import numpy as np

from coupfe.operators.base import Residual, Tangent, complex_step_tangent


class Bar1D:
    """Nonlinear axial bar chain on ``n_elem`` equal/uneven elements."""

    def __init__(self, n_elem: int, L: float, E: float, H: float, f_tip: float):
        self.n = n_elem
        self.x = np.linspace(0.0, L, n_elem + 1)
        self.le = np.diff(self.x)
        self.E, self.H, self.f_tip = E, H, f_tip
        self.ndof = n_elem + 1

    # --- the single source of truth: one element's internal nodal force ---
    def _elem_residual(self, ue, le):
        eps = (ue[1] - ue[0]) / le          # axial strain
        sig = self.E * eps + self.H * eps**2  # nonlinear stress
        return sig * np.array([-1.0, 1.0])  # nodal internal force (area = 1)

    # --- Operator contract ---
    def residual(self, U, state, t, dt):
        R = np.zeros(self.ndof, dtype=U.dtype)
        for e in range(self.n):
            R[e:e + 2] += self._elem_residual(U[e:e + 2], self.le[e])
        R[-1] -= self.f_tip * t             # external tip load, ramped by t
        return Residual(gdofs=np.arange(self.ndof), values=R)

    def tangent(self, U, state, t, dt):
        rows, cols, vals = [], [], []
        for e in range(self.n):
            le = self.le[e]
            Ke = complex_step_tangent(lambda ue: self._elem_residual(ue, le),
                                      U[e:e + 2])
            for i in range(2):
                for j in range(2):
                    rows.append(e + i)
                    cols.append(e + j)
                    vals.append(Ke[i, j])
        return Tangent(rows=np.array(rows), cols=np.array(cols),
                       values=np.array(vals))

    def commit(self, U, state, t, dt):
        return state                        # stateless element

    # --- test oracle only (NOT used by the solver) ---
    def analytic_tangent(self, U):
        K = np.zeros((self.ndof, self.ndof))
        for e in range(self.n):
            le = self.le[e]
            eps = (U[e + 1] - U[e]) / le
            k = (self.E + 2.0 * self.H * eps) / le
            K[e:e + 2, e:e + 2] += k * np.array([[1.0, -1.0], [-1.0, 1.0]])
        return K
