"""Near-incompressible neo-Hookean with element-local condensed pressure — **mixed u-p Hex8**.

The 3D analogue of `examples/neo_hookean_local_pressure_quad4`. A research **non-F-bar**
near-incompressible element used by the 3D rubber-contact studies (see
`examples/REFERENCES.md`). It avoids relying on the F-bar `(J̄/J)^{1/d}`
rescaling when centroid `J̄≤0`; instead it
condenses an element-constant pressure `p` enforcing the **volume-average** constraint

    ∫ q (p − K·lnJ) dV = 0,  q,p constant  ⇒  p = K · (1/V)∫ lnJ dV   (mean dilatation / L2 projection),

It condenses to a displacement-only Hex8. This ships as a RESEARCH example;
the present checks do not establish a general inf-sup, locking, or
inversion-robustness claim. Promotion to READY requires a formulation citation
and provenance record. Total stress  P = G(F − F⁻ᵀ) [deviatoric, full Q1] +
p·F⁻ᵀ [volumetric].

The material (`stress_PK1`/`pressure_resid`) is dimension-agnostic (3×3 `F`); only the weak form's
`ndim` and the element config change vs the Quad4 version.  Props are runtime arguments to the compiled
kernel, so one generated kernel serves any `(G, K)` — including near-incompressible `K/G ≫ 1` for rubber.
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

    props = dict(G=1.0, K=1.0e3)            # rubber: near-incompressible (K/G = 1000)
    state_vars = dict(p=0.0)

    def stress_PK1(self, F, p):
        return self.G * (F - inv(F).T) + p * inv(F).T

    def pressure_resid(self, F, p):
        # p = element-constant volumetric stress; condensation enforces p = K·avg(lnJ).
        return p - self.K * log(det(F))


class NeoHookeanUPHex8(au.WeakForm):
    material = NeoHookeanUP
    ndim = 3

    def define_fields(self):
        self.u = au.VectorField("u", degree=1)
        self.p = au.LocalScalar("p", storage="SVARS", condensed=True, initial=0.0)

    def momentum_equation(self, v, F, p):
        return self.material.stress_PK1(F, p)

    def pressure_equation(self, q, F, p):
        return self.material.pressure_resid(F, p)


def verification_state():
    # A genuine 3D deformation gradient (not a 2D state with an identity z-row).
    F = np.array([[1.08, 0.04, 0.02],
                  [0.01, 0.96, 0.03],
                  [0.02, 0.01, 1.05]])
    return dict(F=F, p=2.5)


def build_kernel(tmp_path: Optional[str] = None,
                 module_name: str = "neohookean_up_hex8") -> Tuple[Path, object]:
    problem = NeoHookeanUPHex8()
    problem.verify(state=verification_state(), verbose=False)   # codegen CS-vs-FD self-check
    here = Path(__file__).resolve().parent
    out = Path(tmp_path) if tmp_path else here
    out.mkdir(parents=True, exist_ok=True)
    for_path = out / "neohookean_up_hex8_uel.for"
    generate_uel_local_pressure(problem, str(for_path),
                                element_config=ELEMENT_CONFIGS["hex8"],
                                mat_prefix="neohookeanup")
    workdir = str(tmp_path) if tmp_path else tempfile.mkdtemp(prefix="nh_up_hex8_build_")
    mod = build_element_kernel(str(for_path), module_name, workdir=workdir)
    return for_path, mod


if __name__ == "__main__":
    p, _ = build_kernel()
    print(f"Generated + compiled {p.name}")
