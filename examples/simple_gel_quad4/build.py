"""Simple two-field (u, mu) gel Quad4 example.

A coupled displacement-chemical-potential Quad4 element.  Both fields are
degree-1 on corner nodes (3 DOFs/node, 12 NDOFEL).  This exercises low-order
coupled field handling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import coupfe.codegen as au
from coupfe.codegen.core.tensor import eye
from coupfe.codegen.generators.uel_gen import generate_uel
from coupfe.runtime.compiled_element import build_element_kernel


class SimpleGelMaterial(au.Material):
    """Simplified gel: linear elasticity + chemical expansion + diffusion."""

    props = dict(G=1.0, D=0.5)

    def stress_PK1(self, F, mu):
        """PK1 stress: mechanical (G*F) + chemical swelling (mu*I)."""
        return self.G * (F - eye(3)) + mu * eye(3)

    def solvent_flux(self, F, mu, grad_mu):
        """Fickian diffusion: j_R = -D * Grad(mu)."""
        return -self.D * grad_mu

    def solvent_storage(self, F, F_old, mu, dt):
        """Linear storage: c_dot = mu (simplified)."""
        return mu


class SimpleGelQuad4(au.WeakForm):
    """Two-field (u, mu) weak form for Quad4 gel element."""

    material = SimpleGelMaterial
    ndim = 2

    def define_fields(self):
        self.u = au.VectorField('u', degree=1)
        self.mu = au.ScalarField('mu', degree=1)

    def momentum_equation(self, v, F, mu):
        return self.material.stress_PK1(F, mu)

    def transport_equation(self, w, F, mu, grad_mu, F_old, dt):
        c_dot = self.material.solvent_storage(F, F_old, mu, dt)
        j_R = self.material.solvent_flux(F, mu, grad_mu)
        return c_dot, j_R


DEFAULT_PROPS = (1.0, 0.5)


def verification_state():
    """State for material-point verification."""
    F = np.array([[1.05, 0.02, 0.0],
                  [0.01, 1.03, 0.0],
                  [0.0, 0.0, 1.0]])
    return dict(
        F=F,
        F_old=0.98 * F,
        mu=1.2,
        grad_mu=np.array([2.0, -1.0, 0.0]),
        dt=0.1,
    )


def build_kernel(
    tmp_path: Optional[str] = None,
    compile_kernel: bool = False,
) -> Tuple[Path, Optional[object]]:
    """Generate (and optionally compile) the coupled Quad4 gel kernel."""
    problem = SimpleGelQuad4()
    problem.verify(state=verification_state(), verbose=False)

    here = Path(__file__).resolve().parent
    out = Path(tmp_path) if tmp_path else here
    out.mkdir(parents=True, exist_ok=True)
    for_path = out / "simple_gel_quad4_uel.for"

    generate_uel(problem, str(for_path), element='Quad4', formulation='standard')

    if compile_kernel:
        mod = build_element_kernel(
            str(for_path), "simple_gel_q4_module", workdir=str(out))
        return for_path, mod
    return for_path, None


if __name__ == '__main__':
    path, _ = build_kernel()
    print(f'Generated {path}')
