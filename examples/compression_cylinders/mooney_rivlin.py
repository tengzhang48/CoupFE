"""Mooney-Rivlin (N=1) hyperelastic — the faithful rubber for the cylinder pack.

Strain energy (compressible, deviatoric/volumetric split):

    W = C10 (Ī1 − 3) + C01 (Ī2 − 3) + (1/D1)(J − 1)²,
    Ī1 = J^(−2/3) I1,  Ī2 = J^(−4/3) I2,  I1 = tr C,  I2 = ½(I1² − tr C²),  C = FᵀF.

First Piola-Kirchhoff stress P = ∂W/∂F:

    P = 2 C10 J^(−2/3) (F − ⅓ I1 F⁻ᵀ)
      + 2 C01 J^(−4/3) (I1 F − F C − ⅔ I2 F⁻ᵀ)
      + (2/D1)(J − 1) J F⁻ᵀ.

Generated as an **F-bar Quad4** (the rubber is near-incompressible, ν≈0.40 from
C10/D1 → would volumetrically lock with full integration).  The Abaqus deck's
constants: C10 = C01 = 4.48632, D1 = 0.02229.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import coupfe.codegen as au
from coupfe.codegen.core.tensor import det, exp, inv, log, trace, transpose
from coupfe.codegen.generators.uel_gen import generate_element
from coupfe.runtime.compiled_element import build_element_kernel


class MooneyRivlin(au.Material):
    props = dict(C10=4.48632, C01=4.48632, D1=0.02229)

    def stress_PK1(self, F):
        J = det(F)
        FinvT = transpose(inv(F))
        C = transpose(F) @ F
        I1 = trace(C)
        I2 = 0.5 * (I1 * I1 - trace(C @ C))
        Jm23 = exp(log(J) * (-2.0 / 3.0))            # J^(-2/3), CS-safe (J>0)
        Jm43 = Jm23 * Jm23
        FC = F @ C
        P1 = 2.0 * self.C10 * Jm23 * (F - (1.0 / 3.0) * I1 * FinvT)
        P2 = 2.0 * self.C01 * Jm43 * (I1 * F - FC - (2.0 / 3.0) * I2 * FinvT)
        Pvol = (2.0 / self.D1) * (J - 1.0) * J * FinvT
        return P1 + P2 + Pvol


class MRQuad4(au.WeakForm):
    material = MooneyRivlin
    ndim = 2

    def define_fields(self):
        self.u = au.VectorField("u", degree=1)

    def momentum_equation(self, v, F):
        return self.material.stress_PK1(F)


# --- independent numpy oracle (different code path than the codegen material) ---
def mr_pk1(F, C10=4.48632, C01=4.48632, D1=0.02229):
    F = np.asarray(F, dtype=complex)
    J = np.linalg.det(F)
    FinvT = np.linalg.inv(F).T
    C = F.T @ F
    I1 = np.trace(C)
    I2 = 0.5 * (I1 * I1 - np.trace(C @ C))
    Jm23 = J ** (-2.0 / 3.0)
    P = (2.0 * C10 * Jm23 * (F - (1.0 / 3.0) * I1 * FinvT)
         + 2.0 * C01 * Jm23 * Jm23 * (I1 * F - F @ C - (2.0 / 3.0) * I2 * FinvT)
         + (2.0 / D1) * (J - 1.0) * J * FinvT)
    return P


def verify_material(eps=1e-6, tol=1e-6, verbose=True):
    """Stress-free at F=I + complex-step tangent == FD of the residual + the
    codegen material == the independent numpy oracle."""
    m = MooneyRivlin()
    F0 = np.array([[1.08, 0.05, 0.0], [0.02, 0.94, 0.0], [0.0, 0.0, 1.0]])

    # stress-free
    P_I = np.asarray(m.stress_PK1(np.eye(3) + 0j)).real
    free = float(np.max(np.abs(P_I)))

    # codegen material vs independent oracle
    P_cg = np.asarray(m.stress_PK1(F0 + 0j)).real
    P_or = mr_pk1(F0).real
    oracle_err = float(np.max(np.abs(P_cg - P_or)) / np.max(np.abs(P_or)))

    # CS tangent vs FD tangent (dP/dF)
    def P_of(Fc):
        return np.asarray(m.stress_PK1(Fc))
    cs = np.zeros((3, 3, 3, 3)); fd = np.zeros((3, 3, 3, 3))
    for a in range(3):
        for b in range(3):
            Fc = F0.astype(complex).copy(); Fc[a, b] += 1j * 1e-20
            cs[:, :, a, b] = P_of(Fc).imag / 1e-20
            Fp = F0.copy(); Fp[a, b] += eps; Fm = F0.copy(); Fm[a, b] -= eps
            fd[:, :, a, b] = (P_of(Fp + 0j).real - P_of(Fm + 0j).real) / (2 * eps)
    rel = float(np.max(np.abs(cs - fd)) / np.max(np.abs(cs)))
    ok = free < 1e-9 and oracle_err < 1e-10 and rel < tol
    if verbose:
        print(f"  stress-free P(F=I): {free:.2e}")
        print(f"  codegen vs numpy oracle: rel={oracle_err:.2e}")
        print(f"  dP/dF CS-vs-FD: rel={rel:.2e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def build_kernel(tmpdir=None, module_name="mr_fbar_q4"):
    problem = MRQuad4()
    if not verify_material(verbose=False):
        raise RuntimeError("MooneyRivlin material verification failed")
    here = Path(__file__).resolve().parent
    out = Path(tmpdir) if tmpdir else here
    for_path = out / "mr_fbar_q4.for"
    generate_element(problem, str(for_path), element="Quad4",
                     formulation="fbar_mechanics", backend="native")
    return for_path, build_element_kernel(str(for_path), module_name, workdir=str(out))


if __name__ == "__main__":
    print("Mooney-Rivlin material verification:")
    verify_material()
