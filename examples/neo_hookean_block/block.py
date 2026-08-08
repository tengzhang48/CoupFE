"""A 2D compressible neo-Hookean block — one real compiled element, end to end.

This is the smallest *compiled-kernel* CoupFE operator (the bar example is the
smallest pure-Python one).  It wires the vendored native neo-Hookean Quad4
kernel (``coupfe/runtime/elements/neo_hookean_q4_native.for``,
``P = G(F - F^-T) + lambda ln(J) F^-T``) into an :class:`ElementGroup` and solves a
small block/patch through ``newton_solve`` — the same contract as every other
operator.

Meshing and boundary-condition bookkeeping live here in the example layer on
purpose; ``docs/DESIGN.md`` treats them as important application-specific code
rather than Core policy. The kernel is built
once and cached so importing this module is cheap to reuse across the run and tests.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from coupfe.operators.element_group import ElementGroup, GroupState
from coupfe.materials import neo_hookean_kernel_props
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

# Vendored native Neo-Hookean Quad4 kernel shipped with CoupFE.
_FOR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "coupfe", "runtime", "elements", "neo_hookean_q4_native.for")

# Public example inputs: shear modulus G and physical small-strain bulk modulus K.
# ``make_group`` converts these to the retained raw kernel ABI ``(G, lambda)``.
DEFAULT_PROPS = (1.0, 10.0)


def structured_quad_mesh(nx, ny, Lx=1.0, Ly=1.0):
    """``nx`` x ``ny`` Quad4 mesh on ``[0,Lx] x [0,Ly]``.

    Returns ``(nodes, elems)``: ``nodes`` (Nnode, 2), ``elems`` (Nelem, 4) in CCW
    corner order (bl, br, tr, tl → positive Jacobian), matching the kernel.
    """
    nx, ny = int(nx), int(ny)
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    nodes = np.array([(x, y) for y in ys for x in xs], dtype=float)
    elems = np.empty((nx * ny, 4), dtype=int)
    e = 0
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elems[e] = (n0, n0 + 1, n0 + nx + 2, n0 + nx + 1)
            e += 1
    return nodes, elems


@lru_cache(maxsize=4)
def _kernel(props=DEFAULT_PROPS, module_name="coupfe_neo_q4"):
    """Build (once) and cache the compiled neo-Hookean Quad4 module."""
    return build_element_kernel(_FOR, module_name)


def make_group(nodes, elems, props=DEFAULT_PROPS, dof_per_node=2, comps=(0, 1)):
    """An :class:`ElementGroup` over ``elems`` driving the neo-Hookean kernel.

    The kernel is stateless (``n_svars=0``); ``dof_per_node``/``comps`` default to a
    pure-u (2 DOF/node) single-material layout but accept the multi-material pattern.
    """
    mod = _kernel(tuple(props))
    raw_props = neo_hookean_kernel_props(*props)
    elem = CompiledElement(mod, props=raw_props, dof_per_node=2, n_svars=0,
                           mcrd=2, n_elem=len(elems))
    return ElementGroup(elem, nodes, elems, dof_per_node=dof_per_node, comps=comps)


def uniaxial_problem(nx=4, ny=4, Lx=1.0, Ly=1.0, stretch=0.20, props=DEFAULT_PROPS):
    """A displacement-controlled uniaxial-stretch block.

    Left edge fixed in x, bottom-left corner pinned in y (statically determinate,
    no rigid-body modes), right edge pulled to ``x*(1+stretch)``.  The top/bottom
    are traction-free, so the block contracts laterally (Poisson-like) — a genuine
    finite-strain solve, not a prescribed homogeneous field.

    Returns ``(nodes, elems, group, dirichlet)`` ready for ``newton_solve``.
    """
    nodes, elems = structured_quad_mesh(nx, ny, Lx, Ly)
    group = make_group(nodes, elems, props=props)
    dpn = 2
    tol = 1e-9
    left = np.nonzero(np.abs(nodes[:, 0] - 0.0) < tol)[0]
    right = np.nonzero(np.abs(nodes[:, 0] - Lx) < tol)[0]
    corner = np.nonzero((np.abs(nodes[:, 0]) < tol) & (np.abs(nodes[:, 1]) < tol))[0]
    dirichlet = {}
    for n in left:
        dirichlet[n * dpn + 0] = 0.0                  # left edge: u_x = 0
    for n in corner:
        dirichlet[n * dpn + 1] = 0.0                  # one corner: u_y = 0 (kill RBM)
    for n in right:
        dirichlet[n * dpn + 0] = Lx * stretch         # right edge: prescribed stretch
    return nodes, elems, group, dirichlet
