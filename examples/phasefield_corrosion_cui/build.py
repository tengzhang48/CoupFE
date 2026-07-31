# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Teng Zhang
"""Cui-style uniform-Q8R phase-field corrosion UEL.

This implements the project-authored CoupFE version without the Abaqus-only
visualization bridge (UEXTERNALDB/UVARM/Mutex). Provenance and formulation
references are recorded in ``NOTICE`` and ``examples/REFERENCES.md``. It is the
u-phi-c corrosion element with small-strain J2 plasticity and the
fatigue/repassivation state variables used in Cui's PhaseFieldSCC benchmark.

Fields per node (node-major): [ux, uy, phi, c]  -> 32 NDOFEL.
Element: Quad8R, 2x2 reduced integration, 4 Gauss points.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import coupfe.codegen as au
from coupfe.codegen.core.small_strain_plasticity import flow_direction, q_mises
from coupfe.codegen.core.tensor import exp, eye, trace
from coupfe.codegen.generators.uel_gen import generate_uel
from coupfe.runtime.compiled_element import build_element_kernel


class CuiJ2CorrosionMaterial(au.Material):
    """Cui-style corrosion model with small-strain J2 plasticity."""

    props = dict(
        E=190000.0,
        nu=0.3,
        sigma_y=520.0,
        hardening_n=0.067,
        D=8.5e-4,
        L0=1.0e-3,
        kappa=5.1e-5,
        omega=35.3,
        Achem=53.5,
        k_repassivation=5.0e-4,
        eps_f=3.0e-3,
        t0=10.0,
        c_solid=1.0,
        c_liquid=0.036,
    )
    state_vars = dict(
        ep=0.0,
        epsp=np.zeros((3, 3)),
        deqpl=0.0,
        hydro=0.0,
        xL=1.0e-3,
        ti=0.0,
        ei=0.0,
    )
    state_var_props = dict(xL="L0")

    def stress_PK1(self, F, phi, ep_old, epsp_old, deqpl_old, hydro_old,
                   xL_old, ti_old, ei_old, dt):
        """Small-strain stress returned in the PK1 slot."""
        eps = 0.5 * (F + F.T) - eye(3)
        mu = self.E / (2.0 * (1.0 + self.nu))
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        eps_e_trial = eps - epsp_old
        sigma_trial = lam * trace(eps_e_trial) * eye(3) + 2.0 * mu * eps_e_trial
        seq = q_mises(sigma_trial)
        yield_old = self.sigma_y * (
            1.0 + self.E * ep_old / self.sigma_y
        ) ** self.hardening_n
        f_trial = seq - yield_old

        if f_trial.real > 0.0:
            deqpl = 0.0 * f_trial
            yield_new = yield_old
            tangent_hard = self.E * self.hardening_n * (
                1.0 + self.E * ep_old / self.sigma_y
            ) ** (self.hardening_n - 1.0)
            for iteration in range(20):
                resid = seq - 3.0 * mu * deqpl - yield_new
                deqpl = deqpl + resid / (3.0 * mu + tangent_hard)
                yield_new = self.sigma_y * (
                    1.0 + self.E * (ep_old + deqpl) / self.sigma_y
                ) ** self.hardening_n
                tangent_hard = self.E * self.hardening_n * (
                    1.0 + self.E * (ep_old + deqpl) / self.sigma_y
                ) ** (self.hardening_n - 1.0)
            n = flow_direction(sigma_trial, seq)
            sigma0 = sigma_trial - 2.0 * mu * deqpl * n
            ep_new = ep_old + deqpl
            epsp_new = epsp_old + deqpl * n
        else:
            sigma0 = sigma_trial
            ep_new = ep_old
            epsp_new = epsp_old
            deqpl = 0.0 * f_trial

        degr = phi * phi * (3.0 - 2.0 * phi)
        xkap = 1.0e-3
        sigma_damaged = (degr + xkap) * sigma0
        hydro_new = trace(sigma_damaged) / 3.0
        gas_const = 8314.0
        temp_abs = 300.0
        mech_factor = exp(hydro_new * 7.12e3 / (gas_const * temp_abs)) * (
            1.0 + ep_new / (self.sigma_y / self.E)
        )

        ei_trial = ei_old + deqpl
        if ei_trial.real > self.eps_f.real:
            ti_cycle = 0.0 * ti_old
            ei_cycle = 0.0 * ei_old
        else:
            ti_cycle = ti_old + dt
            ei_cycle = ei_trial

        if ti_cycle.real < self.t0.real:
            repassivation = 1.0 + 0.0 * ti_cycle
        else:
            repassivation = exp(
                -self.k_repassivation * (ti_cycle - self.t0))
        xL_new = self.L0 * mech_factor * repassivation
        return sigma_damaged, {
            'ep': ep_new,
            'epsp': epsp_new,
            'deqpl': deqpl,
            'hydro': hydro_new,
            'xL': xL_new,
            'ti': ti_cycle,
            'ei': ei_cycle,
        }

    def phase_storage(self, F, phi, c, phi_old, xL_old, dt):
        h = phi * phi * (3.0 - 2.0 * phi)
        dh = 6.0 * phi * (1.0 - phi)
        dc_eq = self.c_solid - self.c_liquid
        c_mix = self.c_liquid + h * (self.c_solid - self.c_liquid)
        chem_arg = c - c_mix
        dpsi_chem = -2.0 * self.Achem * chem_arg * dc_eq * dh
        dpsi_dw = (
            2.0 * self.omega * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
        )
        phidot = (phi - phi_old) / dt
        return -phidot / xL_old - dpsi_chem - dpsi_dw

    def phase_flux(self, F, phi, grad_phi):
        return self.kappa * grad_phi

    def species_storage(self, F, c, c_old, dt):
        return (c - c_old) / dt

    def species_flux(self, F, phi, grad_phi, grad_c):
        dh = 6.0 * phi * (1.0 - phi)
        dc_eq = self.c_solid - self.c_liquid
        return -self.D * (grad_c - dc_eq * dh * grad_phi)


class CuiJ2Corrosion(au.WeakForm):
    """Uniform-Q8R u-phi-c corrosion problem (production staggered tangent)."""

    material = CuiJ2CorrosionMaterial
    ndim = 2

    # Drop K_uphi from the Jacobian; residual stays fully coupled.
    drop_tangent_coupling = [('momentum_equation', 'phi')]

    def define_fields(self):
        self.u = au.VectorField("u", degree=2)
        self.phi = au.ScalarField("phi", degree=2, test="eta")
        self.c = au.ScalarField("c", degree=2, test="zeta")

    def momentum_equation(self, v, F, phi):
        return self.material.stress_PK1(F, phi)

    def phase_equation(self, eta, F, phi, c, grad_phi, phi_old, xL_old, dt):
        return (
            self.material.phase_storage(F, phi, c, phi_old, xL_old, dt),
            self.material.phase_flux(F, phi, grad_phi),
        )

    def species_transport_equation(self, zeta, F, phi, c, grad_phi,
                                   grad_c, c_old, dt):
        return (
            self.material.species_storage(F, c, c_old, dt),
            self.material.species_flux(F, phi, grad_phi, grad_c),
        )


class CuiJ2CorrosionFull(CuiJ2Corrosion):
    """Consistent-tangent variant used for FD-tangent and reference-assembly gates."""

    drop_tangent_coupling = []


DEFAULT_PROPS = tuple(CuiJ2CorrosionMaterial.props.values())


def verification_state():
    F = np.array([[1.0015, 0.0002, 0.0],
                  [0.0001, 1.0005, 0.0],
                  [0.0, 0.0, 1.0]])
    return dict(
        F=F,
        phi=0.72,
        phi_old=0.725,
        c=0.68,
        c_old=0.69,
        grad_phi=np.array([0.12, -0.05, 0.0]),
        grad_c=np.array([-0.03, 0.02, 0.0]),
        ep_old=0.0,
        epsp_old=np.zeros((3, 3)),
        deqpl_old=0.0,
        hydro_old=0.0,
        xL_old=1.0e-3,
        ti_old=0.2,
        ei_old=0.0,
        dt=0.01,
    )


def build_kernel(
    variant: str = "full",
    tmp_path: Optional[str] = None,
    compile_kernel: bool = False,
) -> Tuple[Path, Optional[object]]:
    """Generate (and optionally compile) the Cui Quad8R kernel.

    variant: "full" for the consistent tangent used in verification,
             "production" for the staggered K_uphi-dropped tangent.
    """
    if variant == "full":
        problem = CuiJ2CorrosionFull()
    elif variant == "production":
        problem = CuiJ2Corrosion()
    else:
        raise ValueError(f"unknown variant: {variant}")

    problem.verify(state=verification_state(), verbose=False)

    here = Path(__file__).resolve().parent
    out = Path(tmp_path) if tmp_path else here
    out.mkdir(parents=True, exist_ok=True)
    for_path = out / f"phasefield_corrosion_cui_{variant}_uel.for"

    generate_uel(problem, str(for_path), element='Quad8R',
                 formulation='standard')

    if compile_kernel:
        mod = build_element_kernel(
            str(for_path), f"cui_q8r_{variant}_module", workdir=str(out))
        return for_path, mod
    return for_path, None


if __name__ == '__main__':
    path, _ = build_kernel(variant="full")
    print(f'Generated {path}')
