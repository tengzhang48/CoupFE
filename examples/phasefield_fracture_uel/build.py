"""Minimal AT2-style phase-field fracture Quad4 RESEARCH/codegen demo.

A coupled displacement-damage (u+d) Quad4 exercise. The phase equation has
no history/irreversibility: the driving force H0 is a prescribed constant.
It demonstrates coupled fields without stored state, not a qualified fracture
benchmark.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import coupfe.codegen as au
from coupfe.codegen.core.tensor import det, inv, log
from coupfe.codegen.generators.uel_gen import generate_uel
from coupfe.runtime.compiled_element import build_element_kernel


class PhaseFieldFractureMaterial(au.Material):
    """Degraded neo-Hookean mechanics + AT2 phase-field damage."""

    props = dict(G=1.0, K=10.0, Gc=1.0, ell=0.1, kappa=1e-6, H0=0.01)

    def stress_PK1(self, F, d):
        J = det(F)
        finv_t = inv(F).T
        lame_lambda = self.K - 2.0 * self.G / 3.0
        P0 = self.G * (F - finv_t) + lame_lambda * log(J) * finv_t
        g_d = (1.0 - d) ** 2 + self.kappa
        return g_d * P0

    def phase_storage(self, d):
        return (self.Gc / self.ell) * d - 2.0 * (1.0 - d) * self.H0

    def phase_flux(self, grad_d):
        # The UEL tuple convention is storage*eta - flux.grad(eta).
        # The physical AT2 gradient term is +Gc*ell*grad(d).grad(eta),
        # so return the negative coefficient.
        return -self.Gc * self.ell * grad_d


class PhaseFieldFractureQuad4(au.WeakForm):
    """2D Quad4 u+d phase-field fracture problem."""

    material = PhaseFieldFractureMaterial
    ndim = 2

    def define_fields(self):
        self.u = au.VectorField("u", degree=1)
        self.d = au.ScalarField("d", degree=1, test="eta")

    def momentum_equation(self, v, F, d):
        return self.material.stress_PK1(F, d)

    def phase_equation(self, eta, d, grad_d):
        return (
            self.material.phase_storage(d),
            self.material.phase_flux(grad_d),
        )


DEFAULT_PROPS = tuple(PhaseFieldFractureMaterial.props.values())


def verification_state():
    """State for tangent verification with nonzero d and grad_d."""
    F = np.array([[1.02, 0.01, 0.0],
                  [0.005, 1.01, 0.0],
                  [0.0, 0.0, 1.0]])
    return dict(
        F=F,
        d=0.2,
        grad_d=np.array([0.1, -0.05, 0.0]),
    )


def build_kernel(
    tmp_path: Optional[str] = None,
    compile_kernel: bool = False,
) -> Tuple[Path, Optional[object]]:
    """Generate (and optionally compile) the phase-field Quad4 kernel."""
    problem = PhaseFieldFractureQuad4()
    problem.verify(state=verification_state(), verbose=False)

    here = Path(__file__).resolve().parent
    out = Path(tmp_path) if tmp_path else here
    out.mkdir(parents=True, exist_ok=True)
    for_path = out / "phasefield_fracture_uel.for"

    generate_uel(problem, str(for_path), element="Quad4", formulation="standard")

    if compile_kernel:
        mod = build_element_kernel(
            str(for_path), "phasefield_fracture_q4_module", workdir=str(out))
        return for_path, mod
    return for_path, None


if __name__ == "__main__":
    path, _ = build_kernel()
    print(f"Generated {path}")
