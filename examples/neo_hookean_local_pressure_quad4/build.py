"""Neo-Hookean with element-local condensed pressure (mixed u-p Quad4).

This is a **variationally-consistent "average"** near-incompressible element and
an alternative to F-bar. The pressure ``p`` is one constant per element
(``LocalScalar``, stored in SVARS, statically condensed), and the element
pressure equation

    ∫ q (p − K·lnJ) dV = 0   with constant q,p   ⇒   p = K · (1/V)∫ lnJ dV

makes ``p`` the **volume-AVERAGE** of the volumetric response (the L2
projection), whereas F-bar uses the **centroid** value. It condenses to a
displacement-only Quad4. This ships as a RESEARCH example; it does not
establish a general inf-sup, locking, or inversion-robustness result. Promotion
to READY requires a formulation citation and provenance record.

Total stress: P = G(F − F⁻ᵀ) [deviatoric, full Q1] + p·F⁻ᵀ [volumetric, via the
element-constant pressure].  Props match the cylinder rubber's neo-Hookean
equivalent (G=17.95, K=89.7).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import coupfe.codegen as au
from coupfe.codegen.core.tensor import det, inv, log
from coupfe.codegen.generators.element_config import ELEMENT_CONFIGS
from coupfe.codegen.generators.uel_local_pressure import generate_uel_local_pressure
from coupfe.runtime.compiled_element import build_element_kernel


class NeoHookeanUP(au.Material):
    """Near-incompressible neo-Hookean, pressure as a local (condensed) variable."""

    props = dict(G=17.95, K=89.7)
    state_vars = dict(p=0.0)

    def stress_PK1(self, F, p):
        return self.G * (F - inv(F).T) + p * inv(F).T

    def pressure_resid(self, F, p):
        # p is the (element-constant) volumetric stress; condensation enforces
        # the volume-average  p = K·avg(lnJ).
        return p - self.K * log(det(F))


class NeoHookeanUPQuad4(au.WeakForm):
    material = NeoHookeanUP
    ndim = 2

    def define_fields(self):
        self.u = au.VectorField("u", degree=1)
        self.p = au.LocalScalar("p", storage="SVARS", condensed=True, initial=0.0)

    def momentum_equation(self, v, F, p):
        return self.material.stress_PK1(F, p)

    def pressure_equation(self, q, F, p):
        return self.material.pressure_resid(F, p)


def verification_state():
    F = np.array([[1.05, 0.03, 0.0], [0.01, 0.97, 0.0], [0.0, 0.0, 1.0]])
    return dict(F=F, p=2.5)


def build_kernel(tmp_path: Optional[str] = None,
                 module_name: str = "neohookean_up_q4") -> Tuple[Path, object]:
    problem = NeoHookeanUPQuad4()
    problem.verify(state=verification_state(), verbose=False)
    here = Path(__file__).resolve().parent
    out = Path(tmp_path) if tmp_path else here
    out.mkdir(parents=True, exist_ok=True)
    for_path = out / "neohookean_up_q4_uel.for"
    generate_uel_local_pressure(problem, str(for_path),
                                element_config=ELEMENT_CONFIGS["quad4"],
                                mat_prefix="neohookeanup")
    workdir = str(tmp_path) if tmp_path else tempfile.mkdtemp(prefix="nh_up_build_")
    mod = build_element_kernel(str(for_path), module_name, workdir=workdir)
    return for_path, mod


if __name__ == "__main__":
    p, _ = build_kernel()
    print(f"Generated + compiled {p.name}")
