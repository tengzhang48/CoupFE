# SPDX-FileCopyrightText: 2026 Teng Zhang
# SPDX-License-Identifier: MIT
"""Research port of the stabilized mixed u-theta Tet4 block example.

The element follows the Section 6.3 / Figure 14 block benchmark in Scovazzi,
Zorrilla, and Rossi (2023).  The project-authored implementation was ported
from ``abaqus_ufl`` to the public ``coupfe.codegen`` namespace without adding
or changing any CoupFE core API.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

import coupfe.codegen as au
from coupfe.codegen.core.tensor import det, eye, inv, log
from coupfe.codegen.generators.uel_gen import generate_uel


class ScovazziBlockMaterial(au.Material):
    """Neo-Hookean material and VMS terms for the block benchmark.

    ``PROPS`` order and units are:

    1. ``mu`` [N/mm2]
    2. ``lam`` [N/mm2]
    3. ``c_tau_u`` [-]
    4. ``c_tau_theta`` [-]
    5. ``h_elem`` [mm]
    """

    props = dict(
        mu=80.194,
        lam=400889.806,
        c_tau_u=2.0,
        c_tau_theta=0.1,
        h_elem=0.0625,
    )

    # P(F, theta) is part of a mixed stabilized residual; dP/dF at fixed
    # theta is not expected to have hyperelastic major symmetry.
    symmetric_tangent = False

    def _S_enriched(self, F, thetat):
        """Second Piola stress evaluated at the theta-enriched strain."""
        I = eye(3)
        theta = 1.0 + thetat
        J = det(F)
        C = F.T @ F
        Cbar = theta ** (2.0 / 3.0) * J ** (-2.0 / 3.0) * C
        Cbar_inv = inv(Cbar)
        coefficient = self.lam * log(theta) - self.mu
        S = self.mu * I + coefficient * Cbar_inv
        return S, Cbar_inv, theta, J

    def _Dbar(self, Cbar_inv, theta):
        """Return ``(dS/dE):Cbar`` for the benchmark energy."""
        factor = 3.0 * self.lam + 2.0 * self.mu
        factor = factor - 2.0 * self.lam * log(theta)
        return factor * Cbar_inv

    def stress_PK1(self, F, thetat):
        S, Cbar_inv, theta, J = self._S_enriched(F, thetat)
        dS_dtheta = self._Dbar(Cbar_inv, theta) / (3.0 * theta)
        bulk = self.lam + 2.0 * self.mu / 3.0
        tau_theta = self.c_tau_theta * self.mu / (self.mu + bulk)
        return F @ (S - tau_theta * (theta - J) * dS_dtheta)

    def phase_storage(self, F, thetat):
        theta = 1.0 + thetat
        J = det(F)
        bulk = self.lam + 2.0 * self.mu / 3.0
        tau_theta = self.c_tau_theta * self.mu / (self.mu + bulk)
        return (1.0 - tau_theta) * (theta - J)

    def phase_flux(self, F, thetat, grad_thetat):
        S_unused, Cbar_inv, theta, J = self._S_enriched(F, thetat)
        Dbar = self._Dbar(Cbar_inv, theta)
        tau_u = self.c_tau_u * self.h_elem ** 2 / (2.0 * self.mu)
        gradient_term = (tau_u / 3.0) * (J / theta)
        gradient_term = gradient_term * (Dbar @ grad_thetat)
        # CoupFE assembles storage*N - flux.Grad(N); the paper's weak
        # gradient term has a positive sign.
        return -1.0 * gradient_term


class ScovazziBlockTet4(au.WeakForm):
    """Equal-order displacement/Jacobian-discrepancy Tet4."""

    material = ScovazziBlockMaterial
    ndim = 3

    def define_fields(self):
        self.u = au.VectorField("u", degree=1)
        self.thetat = au.ScalarField("thetat", degree=1, test="q")

    def momentum_equation(self, v, F, thetat):
        return self.material.stress_PK1(F, thetat)

    def phase_equation(self, q, F, thetat, grad_thetat):
        return (
            self.material.phase_storage(F, thetat),
            self.material.phase_flux(F, thetat, grad_thetat),
        )


DEFAULT_PROPS = tuple(ScovazziBlockMaterial.props.values())


def verification_states() -> Iterable[Dict[str, object]]:
    """Non-benign states exercising every stabilized tangent block."""
    deformations = (
        np.array([
            [1.05, 0.02, 0.01],
            [0.015, 0.98, 0.005],
            [0.01, 0.0, 1.03],
        ]),
        np.array([
            [0.94, -0.03, 0.0],
            [0.02, 0.97, 0.01],
            [0.0, 0.015, 0.96],
        ]),
        np.eye(3),
    )
    thetat_values = (0.04, -0.05, 0.02)
    for F, thetat in zip(deformations, thetat_values):
        yield {
            "F": F,
            "thetat": thetat,
            "grad_thetat": np.array([0.03, -0.02, 0.01]),
            "dt": 1.0,
        }


def _independent_second_pk(F, thetat, material=None):
    """NumPy statement of the enriched constitutive law for test oracles."""
    mat = ScovazziBlockMaterial() if material is None else material
    F = np.asarray(F, dtype=complex)
    theta = 1.0 + thetat
    J = np.linalg.det(F)
    C = F.T @ F
    Cbar = theta ** (2.0 / 3.0) * J ** (-2.0 / 3.0) * C
    Cbar_inv = np.linalg.inv(Cbar)
    return mat.mu * np.eye(3) + (
        mat.lam * np.log(theta) - mat.mu
    ) * Cbar_inv


def derivative_oracle_relative_error() -> float:
    """Compare the analytic ``dS/dtheta`` to an independent complex step."""
    mat = ScovazziBlockMaterial()
    state = next(iter(verification_states()))
    F = np.asarray(state["F"], dtype=complex)
    thetat = float(state["thetat"])
    step = 1.0e-25
    derivative_cs = np.imag(
        _independent_second_pk(F, thetat + 1j * step, mat)
    ) / step

    _, Cbar_inv, theta, _ = mat._S_enriched(F, thetat)
    derivative = np.asarray(mat._Dbar(Cbar_inv, theta) / (3.0 * theta))
    scale = max(float(np.max(np.abs(derivative_cs))), 1.0e-30)
    return float(np.max(np.abs(derivative - derivative_cs)) / scale)


def homogeneous_oracle_relative_error() -> float:
    """At ``theta=J``, recover the original law and zero scalar residual."""
    mat = ScovazziBlockMaterial()
    state = next(iter(verification_states()))
    F = np.asarray(state["F"], dtype=complex)
    J = np.linalg.det(F)
    C_inv = np.linalg.inv(F.T @ F)
    expected_S = (
        mat.mu * np.eye(3)
        + (mat.lam * np.log(J) - mat.mu) * C_inv
    )
    expected = F @ expected_S
    actual = np.asarray(mat.stress_PK1(F, J - 1.0))
    scale = max(float(np.max(np.abs(expected))), 1.0e-30)
    stress_error = float(np.max(np.abs(actual - expected)) / scale)
    storage_error = float(abs(complex(mat.phase_storage(F, J - 1.0))))
    flux_error = float(np.max(np.abs(
        mat.phase_flux(F, J - 1.0, np.zeros(3))
    )))
    return max(stress_error, storage_error, flux_error)


def run_reference_checks() -> Dict[str, float]:
    """Run the two closed-form/independent Python checks."""
    errors = {
        "dS_dtheta": derivative_oracle_relative_error(),
        "homogeneous": homogeneous_oracle_relative_error(),
    }
    if errors["dS_dtheta"] >= 1.0e-9:
        raise RuntimeError("dS/dtheta oracle failed: {}".format(errors))
    if errors["homogeneous"] >= 1.0e-12:
        raise RuntimeError("homogeneous oracle failed: {}".format(errors))
    return errors


def build(output_dir=None, *, verify=True) -> Path:
    """Verify and generate the standard four-point Tet4 Abaqus UEL."""
    if verify:
        run_reference_checks()

    problem = ScovazziBlockTet4()
    if verify:
        for index, state in enumerate(verification_states()):
            if not problem.verify(state=state, tol=5.0e-5, verbose=False):
                raise RuntimeError(
                    "framework verification failed at state {}".format(index)
                )

    target_dir = (
        Path(__file__).resolve().parent
        if output_dir is None
        else Path(output_dir)
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / "scovazzi_block_tet4.for"
    generate_uel(problem, str(output), element="tet4", formulation="standard")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()

    errors = run_reference_checks()
    print("dS/dtheta oracle: {:.3e}".format(errors["dS_dtheta"]))
    print("homogeneous oracle: {:.3e}".format(errors["homogeneous"]))
    if not args.self_check_only:
        print("Generated {}".format(build(args.output_dir)))


if __name__ == "__main__":
    main()
