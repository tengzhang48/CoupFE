"""Coupled thermo-mechanical Quad4 example with scalar heat diffusion.

This uses ``coupfe.codegen`` to define a Quad4 element with displacement DOFs
plus one scalar (temperature) DOF per
node, neo-Hookean mechanics with thermal expansion, and Fourier heat conduction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import coupfe.codegen as au
from coupfe.codegen.core.tensor import det, inv, log
from coupfe.codegen.generators.uel_gen import generate_uel
from coupfe.runtime.compiled_element import build_element_kernel


class HeatDiffusionMaterial(au.Material):
    props = dict(G=1.0, K=10.0, alpha=1e-3, k=0.5, rho_cp=1.0)

    def stress_PK1(self, F, T):
        J = det(F)
        finv_t = inv(F).T
        lame_lambda = self.K - 2.0 * self.G / 3.0
        P_mech = self.G * (F - finv_t) + lame_lambda * log(J) * finv_t
        # Thermal pressure is parameterized by physical bulk modulus K.
        P_thermal = -self.K * self.alpha * T * finv_t
        return P_mech + P_thermal

    def solvent_storage(self, F, F_old, T, T_old, dt):
        return self.rho_cp * (T - T_old) / dt

    def solvent_flux(self, F, T, grad_T):
        return -self.k * grad_T


class HeatDiffusionProblem(au.WeakForm):
    material = HeatDiffusionMaterial
    ndim = 2

    def define_fields(self):
        self.u = au.VectorField("u", degree=1)
        self.T = au.ScalarField("T", degree=1, test="theta")

    def momentum_equation(self, v, F, T):
        return self.material.stress_PK1(F, T)

    def transport_equation(self, theta, F, T, grad_T, F_old, T_old, dt):
        storage = self.material.solvent_storage(F, F_old, T, T_old, dt)
        flux = self.material.solvent_flux(F, T, grad_T)
        return storage, flux


DEFAULT_PROPS = (1.0, 10.0, 1e-3, 0.5, 1.0)


def verification_state():
    F = np.array([[1.05, 0.02, 0.0],
                  [0.01, 1.03, 0.0],
                  [0.0, 0.0, 1.0]])
    return dict(
        F=F,
        F_old=0.98 * F,
        T=10.0,
        T_old=5.0,
        grad_T=np.array([2.0, -1.0, 0.5]),
        dt=0.1,
    )


def build_kernel(
    tmp_path: Optional[str] = None,
    compile_kernel: bool = False,
) -> Tuple[Path, Optional[object]]:
    """Generate (and optionally compile) the coupled Quad4 kernel.

    Returns ``(for_path, module_or_None)``.
    """
    problem = HeatDiffusionProblem()
    problem.verify(state=verification_state(), verbose=False)

    here = Path(__file__).resolve().parent
    out = Path(tmp_path) if tmp_path else here
    out.mkdir(parents=True, exist_ok=True)
    for_path = out / "scalar_diffusion_uel.for"

    generate_uel(
        problem,
        str(for_path),
        element="Quad4",
        formulation="standard",
    )

    if compile_kernel:
        mod = build_element_kernel(str(for_path), "heat_diff_q4_module", workdir=str(out))
        return for_path, mod
    return for_path, None


if __name__ == "__main__":
    path, _ = build_kernel()
    print(f"Generated {path}")
