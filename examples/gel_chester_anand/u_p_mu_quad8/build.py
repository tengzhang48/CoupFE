# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Teng Zhang
"""Three-field u-p-mu Chester-Anand gel Quad8 example.

Plane-strain mixed Quad8 gel: quadratic displacement ``u`` and chemical
potential ``mu`` on all 8 nodes, bilinear pressure ``p`` on the 4 corner nodes.
This exercises a non-trivial three-field coupled formulation on Quad8.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import coupfe.codegen as au
from coupfe.codegen.core.tensor import det, exp, inv, log
from coupfe.codegen.generators.uel_gen import generate_uel
from coupfe.runtime.compiled_element import build_element_kernel


class ChesterAnandGelMaterial(au.Material):
    """Neo-Hookean + Flory-Huggins gel material for the mixed u-p-mu UEL."""

    props = dict(
        G=1.0,
        K=100.0,
        chi=0.1,
        D=5.0e-9,
        mu0=0.0,
        Omega=1.0e-4,
        Rgas=8.314,
        theta=298.0,
        phi0=0.5,
    )

    def stress_PK1(self, F, p, mu):
        return self.G * (F - inv(F).T) + p * inv(F).T

    def pressure_resid(self, F, p, mu):
        J = det(F)
        Je = exp(p / self.K)
        phi = self.phi0 * Je / J
        RT = self.Rgas * self.theta
        return (
            mu
            - self.mu0
            - RT * (log(1.0 - phi) + phi + self.chi * phi**2)
            + (phi / self.phi0) * self.Omega * p
        )

    def solvent_flux(self, F, p, mu, grad_mu):
        J = det(F)
        Je = exp(p / self.K)
        phi = self.phi0 * Je / J
        C = F.T @ F
        Cinv = inv(C)
        cR0 = (1.0 - self.phi0) / self.Omega
        cR = cR0 + (self.phi0 - phi) / (self.Omega * phi)
        M = self.D * cR / (self.Rgas * self.theta)
        return -M * (Cinv @ grad_mu)

    def solvent_storage(self, F, F_old, p, p_old, dt):
        J = det(F)
        J_old = det(F_old)
        Je = exp(p / self.K)
        Je_old = exp(p_old / self.K)
        return (J / Je - J_old / Je_old) / (self.Omega * dt)


class ChesterAnandUPMuQuad8(au.WeakForm):
    """Plane-strain mixed Quad8 gel: quadratic u/mu and corner pressure."""

    material = ChesterAnandGelMaterial
    ndim = 2

    def define_fields(self):
        self.u = au.VectorField("u", degree=2)
        self.p = au.ScalarField("p", degree=1)
        self.mu = au.ScalarField("mu", degree=2)

    def momentum_equation(self, v, F, p, mu):
        return self.material.stress_PK1(F, p, mu)

    def pressure_equation(self, q, F, p, mu):
        return self.material.pressure_resid(F, p, mu)

    def transport_equation(self, w, F, p, mu, grad_mu, F_old, p_old, dt):
        c_dot = self.material.solvent_storage(F, F_old, p, p_old, dt)
        j_R = self.material.solvent_flux(F, p, mu, grad_mu)
        return c_dot, j_R


DEFAULT_PROPS = tuple(ChesterAnandGelMaterial.props.values())


def verification_state():
    """State for material-point verification."""
    F = np.array([[1.08, 0.04, 0.0],
                  [0.02, 1.05, 0.0],
                  [0.0, 0.0, 1.0]])
    return dict(
        F=F,
        F_old=0.98 * F,
        p=2.0,
        p_old=1.8,
        mu=1.0,
        grad_mu=np.array([1.0, -0.5, 0.0]),
        dt=0.1,
    )


def build_kernel(
    tmp_path: Optional[str] = None,
    compile_kernel: bool = False,
) -> Tuple[Path, Optional[object]]:
    """Generate (and optionally compile) the mixed Quad8 gel kernel."""
    problem = ChesterAnandUPMuQuad8()
    problem.verify(state=verification_state(), verbose=False)

    here = Path(__file__).resolve().parent
    out = Path(tmp_path) if tmp_path else here
    out.mkdir(parents=True, exist_ok=True)
    for_path = out / "chester_anand_upmu_quad8_uel.for"

    generate_uel(
        problem,
        str(for_path),
        element="Quad8",
        formulation="standard",
        mat_prefix="chesteranandupmu",
    )

    if compile_kernel:
        mod = build_element_kernel(
            str(for_path), "chester_upmu_q8_module", workdir=str(out))
        return for_path, mod
    return for_path, None


if __name__ == "__main__":
    path, _ = build_kernel()
    print(f"Generated {path}")
