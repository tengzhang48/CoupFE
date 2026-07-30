"""Neo-Hookean with isotropic inelastic volume and local condensed pressure.

The inelastic volume change is stress-free.  The material exposes
``inelastic_jacobian(F)`` so the generator forms the elastic Jacobian
``J_e = J / J_inel`` and the pressure equation becomes

    p = K * avg(ln J_e).

The elastic deformation gradient is ``F_e = F / J_inel**(1/2)`` in 2D plane
strain (out-of-plane stretch = 1).  ``stress_PK1`` returns the reference PK1
stress, which for isotropic inelastic expansion requires the mapping factor
``P = sqrt(J_inel) * P_e`` (equivalently ``P = J_inel * P_e * F_inel^{-T}``).

With ``F = J_inel**(1/2) * I`` the elastic part is the identity, so a free
inelastic expansion gives ``p = 0`` and zero stress.  Setting ``J_inel = 1``
recovers the pure-mechanical local-pressure Quad4.
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


class NeoHookeanInelasticUP(au.Material):
    """Near-incompressible neo-Hookean with isotropic inelastic volume."""

    props = dict(G=17.95, K=89.7, J_inel=1.0)
    state_vars = dict(p=0.0)

    def inelastic_jacobian(self, F):
        return self.J_inel

    def stress_PK1(self, F, p, J_inel):
        # 2D plane strain: isotropic inelastic stretch = J_inel**(1/2)
        s = J_inel ** 0.5
        F_e = F / s
        P_e = self.G * (F_e - inv(F_e).T) + p * inv(F_e).T
        # Map PK1 from the intermediate config to the reference config.
        # For isotropic F_inel = s*I this is P = s * P_e.
        return s * P_e

    def pressure_resid(self, F, p, J_inel):
        return p - self.K * log(det(F) / J_inel)


class NeoHookeanInelasticUPQuad4(au.WeakForm):
    material = NeoHookeanInelasticUP
    ndim = 2

    def define_fields(self):
        self.u = au.VectorField("u", degree=1)
        self.p = au.LocalScalar("p", storage="SVARS", condensed=True, initial=0.0)

    def momentum_equation(self, v, F, p, J_inel):
        return self.material.stress_PK1(F, p, J_inel)

    def pressure_equation(self, q, F, p, J_inel):
        return self.material.pressure_resid(F, p, J_inel)


def verification_state():
    F = np.array([[1.05, 0.03, 0.0], [0.01, 0.97, 0.0], [0.0, 0.0, 1.0]])
    return dict(F=F, p=2.5, J_inel=1.5)


def build_kernel(tmp_path: Optional[str] = None,
                 module_name: str = "neohookean_inel_up_q4") -> Tuple[Path, object]:
    problem = NeoHookeanInelasticUPQuad4()
    problem.verify(state=verification_state(), verbose=False)
    here = Path(__file__).resolve().parent
    out = Path(tmp_path) if tmp_path else here
    out.mkdir(parents=True, exist_ok=True)
    for_path = out / "neohookean_inel_up_q4_uel.for"
    generate_uel_local_pressure(problem, str(for_path),
                                element_config=ELEMENT_CONFIGS["quad4"],
                                mat_prefix="neohookeaninelup")
    workdir = str(tmp_path) if tmp_path else tempfile.mkdtemp(prefix="nh_inel_up_build_")
    mod = build_element_kernel(str(for_path), module_name, workdir=workdir)
    return for_path, mod


if __name__ == "__main__":
    p, _ = build_kernel()
    print(f"Generated + compiled {p.name}")
