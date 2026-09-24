"""Generated axisymmetric Yeoh kernels: Python declaration -> native Fortran.

The material and weak forms below are ``coupfe.codegen`` declarations. Each
element/formulation pair is generated for the native CoupFE ABI with an
``*_axi`` element configuration, compiled with f2py, cached by source hash and
run through :class:`coupfe.ElementGroup`.

Formulations:

* ``standard`` -- displacement element with full integration;
* ``fbar`` -- F-bar (Quad4), ``Fbar = (Jbar/J)**(1/3) F`` with the
  ring-weighted element-average ``Jbar``;
* ``mixed`` -- u-p with a continuous corner pressure (Quad8/Quad4 or
  Tri6/Tri3, the Taylor-Hood pairs): ``int [W_iso + p (J-1) - p**2/(2K)] dV``.

The material is ``W = C10 (I1b-3) + C20 (I1b-3)**2 + C30 (I1b-3)**3
+ K/2 (J-1)**2`` with ``I1b = J**(-2/3) tr(F^T F)``; properties are passed in the
order ``(C10, C20, C30, K)``.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

import coupfe.codegen as au
from coupfe import ElementGroup
from coupfe.codegen.core.tensor import det, exp, inv, log, trace
from coupfe.codegen.generators.uel_gen import generate_element
from coupfe.operators.element_group import mixed_dof_map
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

CONFIG = {"quad4": "quad4_axi", "quad8": "quad8_axi", "quad8r": "quad8r_axi",
          "tri3": "tri3_axi", "tri6": "tri6_axi"}
CORNERS = {"quad4": 4, "quad8": 4, "quad8r": 4, "tri3": 3, "tri6": 3}
QUADRATIC = {"quad8", "quad8r", "tri6"}


class Yeoh(au.Material):
    """Isochoric Yeoh law with the volumetric energy ``K/2 (J-1)**2``."""

    props = dict(C10=0.5, C20=0.0, C30=0.0, K=100.0)

    def stress_PK1(self, F):
        J = det(F)
        FiT = inv(F).T
        I1 = trace(F.T @ F)
        Jm23 = exp(-2.0 / 3.0 * log(J))
        x = Jm23 * I1 - 3.0
        c = 2.0 * (self.C10 + 2.0 * self.C20 * x + 3.0 * self.C30 * x * x) * Jm23
        return c * F - (c * I1 / 3.0) * FiT + self.K * (J - 1.0) * J * FiT


class YeohMixed(au.Material):
    """Isochoric Yeoh stress plus an independent pressure ``p = K (J - 1)``."""

    props = dict(C10=0.5, C20=0.0, C30=0.0, K=100.0)

    def stress_PK1(self, F, p):
        J = det(F)
        FiT = inv(F).T
        I1 = trace(F.T @ F)
        Jm23 = exp(-2.0 / 3.0 * log(J))
        x = Jm23 * I1 - 3.0
        c = 2.0 * (self.C10 + 2.0 * self.C20 * x + 3.0 * self.C30 * x * x) * Jm23
        return c * F - (c * I1 / 3.0) * FiT + p * J * FiT

    def pressure_resid(self, F, p):
        return det(F) - 1.0 - p / self.K


def weak_form(element, formulation):
    """The codegen declaration for one element/formulation pair."""
    degree = 2 if element in QUADRATIC else 1
    if formulation == "mixed":
        if degree != 2:
            raise ValueError("the mixed u-p form needs a quadratic element (quad8 or tri6)")

        class MixedForm(au.WeakForm):
            material = YeohMixed

            def define_fields(self):
                self.u = au.VectorField("u", degree=2)
                self.p = au.ScalarField("p", degree=1)

            def momentum_equation(self, v, F, p):
                return self.material.stress_PK1(F, p)

            def pressure_equation(self, q, F, p):
                return self.material.pressure_resid(F, p)

        return MixedForm()
    if formulation == "fbar" and element != "quad4":
        raise ValueError("the F-bar route is generated for quad4")

    class DisplacementForm(au.WeakForm):
        material = Yeoh

        def define_fields(self):
            self.u = au.VectorField("u", degree=degree)

        def momentum_equation(self, v, F):
            return self.material.stress_PK1(F)

    return DisplacementForm()


def cache_dir():
    root = os.environ.get("COUPFE_AXI_KERNEL_CACHE")
    return Path(root) if root else Path(tempfile.gettempdir()) / "coupfe-axi-kernels"


def kernel(element, formulation, directory=None):
    """Generate (once) and import the compiled kernel module."""
    form = weak_form(element, formulation)
    directory = Path(directory) if directory else cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "kernel.for"
        with contextlib.redirect_stdout(io.StringIO()):
            generate_element(form, str(source), element=CONFIG[element],
                             formulation="fbar_mechanics" if formulation == "fbar" else "standard",
                             backend="native")
        text = source.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    name = f"axi_{element}_{formulation}_{digest}"
    workdir = directory / name
    if name in sys.modules:
        return sys.modules[name]
    if workdir.is_dir() and any(p.name.startswith(name) and p.suffix in (".so", ".pyd")
                                for p in workdir.iterdir()):
        if str(workdir) not in sys.path:
            sys.path.insert(0, str(workdir))
        return importlib.import_module(name)
    workdir.mkdir(parents=True, exist_ok=True)
    for_path = workdir / f"{name}.for"
    for_path.write_text(text)
    return build_element_kernel(str(for_path), name, workdir=str(workdir))


def layout(element, formulation):
    """Global DOFs per node: ``(u_r, u_z)`` or ``(u_r, u_z, p)``."""
    return 3 if formulation == "mixed" else 2


def element_group(nodes, cells, element, formulation, props, *, dof_per_node=None, directory=None):
    """An :class:`ElementGroup` of generated axisymmetric Yeoh elements.

    ``props`` is ``(C10, C20, C30, K)``. Mixed elements write ``p`` as global
    component 2 of their corner nodes; the caller prescribes the unused
    pressure slot of mid-side nodes.
    """
    dpn = dof_per_node or layout(element, formulation)
    module = kernel(element, formulation, directory)
    cells = np.asarray(cells, dtype=int)
    compiled = CompiledElement(module, props=np.asarray(props, float), dof_per_node=dpn,
                               mcrd=2, n_elem=len(cells))
    if formulation == "mixed":
        dof_map = mixed_dof_map(cells, dpn, CORNERS[element], (0, 1, 2), (0, 1))
        return ElementGroup(compiled, nodes, cells, dpn, dof_map=dof_map)
    return ElementGroup(compiled, nodes, cells, dpn, comps=(0, 1))


def unused_pressure_dofs(cells, element, formulation, n_nodes):
    """Pressure slots that no mixed element writes.

    ``cells`` must hold *every* mixed element sharing the global layout
    (stack the blocks): a slot is unused only when its node is a corner of no
    mixed element. Passing one block alone would pin the pressures of the
    other blocks.
    """
    if formulation != "mixed":
        return np.array([], dtype=int)
    corner = np.unique(np.asarray(cells)[:, :CORNERS[element]])
    return 3 * np.setdiff1d(np.arange(n_nodes), corner) + 2
