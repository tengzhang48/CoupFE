"""Mixed u-p Quad8 neo-Hookean example.

A Quad8 element with displacement DOFs at all 8 nodes and pressure DOFs at the
4 corner nodes.  The pressure field is treated as a scalar field with test
function ``q`` via ``pressure_equation``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import coupfe.codegen as au
from coupfe.codegen.core.tensor import det, inv
from coupfe.codegen.generators.uel_gen import generate_uel
from coupfe.runtime.compiled_element import build_element_kernel


class NeoHookeanMixed(au.Material):
    """Compressible neo-Hookean with physical bulk modulus ``K``.

    Eliminating pressure gives the volumetric coefficient
    ``lambda = K - 2G/3`` required by the non-isochoric shear term.
    """

    props = dict(G=0.5, K=50.0)

    def stress_PK1(self, F, p):
        finv_t = inv(F).T
        j = det(F)
        return self.G * (F - finv_t) + p * j * finv_t

    def pressure_resid(self, F, p):
        lame_lambda = self.K - 2.0 * self.G / 3.0
        return det(F) - 1.0 - p / lame_lambda


class Quad8MixedPatch(au.WeakForm):
    material = NeoHookeanMixed
    ndim = 2

    def define_fields(self):
        self.u = au.VectorField('u', degree=2)
        self.p = au.ScalarField('p', degree=1)

    def momentum_equation(self, v, F, p):
        return self.material.stress_PK1(F, p)

    def pressure_equation(self, q, F, p):
        return self.material.pressure_resid(F, p)


DEFAULT_PROPS = (0.5, 50.0)


def verification_state():
    """State for material-point verification."""
    F = np.array([[1.08, 0.04, 0.0],
                  [0.02, 1.05, 0.0],
                  [0.0, 0.0, 1.0]])
    return dict(F=F, p=2.0)


def build_kernel(
    tmp_path: Optional[str] = None,
    compile_kernel: bool = False,
) -> Tuple[Path, Optional[object]]:
    """Generate (and optionally compile) the mixed Quad8 kernel."""
    problem = Quad8MixedPatch()
    problem.verify(state=verification_state(), verbose=False)

    here = Path(__file__).resolve().parent
    out = Path(tmp_path) if tmp_path else here
    out.mkdir(parents=True, exist_ok=True)
    for_path = out / "neo_hookean_mixed_uel.for"

    generate_uel(problem, str(for_path), element='Quad8', formulation='standard')

    if compile_kernel:
        mod = build_element_kernel(
            str(for_path), "neo_mixed_q8_module", workdir=str(out))
        return for_path, mod
    return for_path, None


if __name__ == '__main__':
    path, _ = build_kernel()
    print(f'Generated {path}')
