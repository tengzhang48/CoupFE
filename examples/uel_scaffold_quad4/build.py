"""Minimal Quad4 UEL scaffold example.

A single-field compressible neo-Hookean mechanical element on a Quad4 patch,
generated with the F-bar formulation. It keeps only the codegen/build glue
needed to demonstrate the scaffold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import coupfe.codegen as au
from coupfe.codegen.core.tensor import det, inv, log
from coupfe.codegen.generators.uel_gen import generate_uel
from coupfe.runtime.compiled_element import build_element_kernel


class NeoHookean(au.Material):
    props = dict(G=0.5, K=50.0)

    def stress_PK1(self, F):
        J = det(F)
        FinvT = inv(F).T
        return self.G * (F - FinvT) + self.K * log(J) * FinvT


class NeoQuad4(au.WeakForm):
    material = NeoHookean
    ndim = 2

    def define_fields(self):
        self.u = au.VectorField("u", degree=1)

    def momentum_equation(self, v, F):
        return self.material.stress_PK1(F)


DEFAULT_PROPS = (0.5, 50.0)


def build_kernel(
    tmp_path: Optional[str] = None,
    compile_kernel: bool = False,
) -> Tuple[Path, Optional[object]]:
    """Generate (and optionally compile) the Quad4 kernel.

    Returns ``(for_path, module_or_None)``.
    """
    problem = NeoQuad4()
    problem.verify(verbose=False)

    here = Path(__file__).resolve().parent
    out = Path(tmp_path) if tmp_path else here
    out.mkdir(parents=True, exist_ok=True)
    for_path = out / "neo_quad4_uel.for"

    generate_uel(
        problem,
        str(for_path),
        element="Quad4",
        formulation="fbar_mechanics",
    )

    if compile_kernel:
        mod = build_element_kernel(str(for_path), "neo_q4_module", workdir=str(out))
        return for_path, mod
    return for_path, None


if __name__ == "__main__":
    path, _ = build_kernel()
    print(f"Generated {path}")
