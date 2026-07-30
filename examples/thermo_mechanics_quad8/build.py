"""Thermo-mechanical Quad8 example.

Coupled displacement-temperature problem on a Quad8 element (2D).  The scalar
field is intentionally named ``T`` (not ``mu``) to exercise generic scalar-field
handling in the codegen.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import coupfe.codegen as au
from coupfe.codegen.core.tensor import det, inv, log
from coupfe.codegen.generators.uel_gen import generate_uel
from coupfe.runtime.compiled_element import build_element_kernel


class ThermoMechanicalMaterial(au.Material):
    """Compressible neo-Hookean mechanics with isotropic thermal pressure."""

    props = dict(G=1.0, K=100.0, alpha=1.0e-3, kappa=0.25, cT=1.0)

    def stress_PK1(self, F, T):
        finv_t = inv(F).T
        J = det(F)
        P_mech = self.G * (F - finv_t) + self.K * log(J) * finv_t
        P_thermal = -self.K * self.alpha * T * finv_t
        return P_mech + P_thermal

    def solvent_flux(self, F, T, grad_T):
        """Referential Fourier/Fick-style flux for the scalar field."""
        Cinv = inv(F.T @ F)
        return -self.kappa * (Cinv @ grad_T)

    def solvent_storage(self, F, F_old, T, T_old, dt):
        """Backward-Euler scalar storage."""
        return self.cT * (T - T_old) / dt


class ThermoMechanicalQuad8(au.WeakForm):
    """Coupled displacement-temperature problem using generic scalar names."""

    material = ThermoMechanicalMaterial
    ndim = 2

    def define_fields(self):
        self.u = au.VectorField("u", degree=2)
        self.T = au.ScalarField("T", degree=1, test="theta")

    def momentum_equation(self, v, F, T):
        return self.material.stress_PK1(F, T)

    def transport_equation(self, theta, F, T, grad_T, F_old, T_old, dt):
        storage = self.material.solvent_storage(F, F_old, T, T_old, dt)
        flux = self.material.solvent_flux(F, T, grad_T)
        return storage, flux


DEFAULT_PROPS = (1.0, 100.0, 1.0e-3, 0.25, 1.0)


def verification_state():
    """State with all generic scalar arguments required by verify()."""
    F = np.array([[1.08, 0.04, 0.0],
                  [0.02, 1.05, 0.0],
                  [0.0, 0.0, 1.0]])
    return dict(
        F=F,
        F_old=0.98 * F,
        T=15.0,
        T_old=10.0,
        grad_T=np.array([2.0, -1.0, 0.5]),
        dt=0.1,
    )


def build_kernel(
    tmp_path: Optional[str] = None,
    compile_kernel: bool = False,
) -> Tuple[Path, Optional[object]]:
    """Generate (and optionally compile) the coupled Quad8 kernel.

    Returns ``(for_path, module_or_None)``.
    """
    problem = ThermoMechanicalQuad8()
    problem.verify(state=verification_state(), verbose=False)

    here = Path(__file__).resolve().parent
    out = Path(tmp_path) if tmp_path else here
    out.mkdir(parents=True, exist_ok=True)
    for_path = out / "thermo_mechanics_quad8_uel.for"

    generate_uel(
        problem,
        str(for_path),
        element="Quad8",
        formulation="standard",
    )

    if compile_kernel:
        mod = build_element_kernel(
            str(for_path), "thermo_q8_module", workdir=str(out))
        return for_path, mod
    return for_path, None


if __name__ == "__main__":
    path, _ = build_kernel()
    print(f"Generated {path}")
