"""Curved-boundary convergence on a thick-annulus sector — distributed-mesh M2.

The independent oracle is the exact axisymmetric linear-elastic field
``u_r(r) = a·r + b/r`` (the homogeneous solution of radial equilibrium — no body
force, material-independent). Prescribed on the *whole* boundary, it is the exact
interior solution; the small load keeps the neo-Hookean kernel in its linear regime.
Refining the curved mesh (with geometry re-embedding) must drive the interior error to
zero at the Quad4 rate (~h²). Without re-embedding the boundary is faceted, not curved,
and the error is larger — the broken control.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# reuse the neo-Hookean kernel builder (cached) from the block example.
# abspath so block.py's own dirname-based path to the .for resolves correctly.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "neo_hookean_block")))
from block import DEFAULT_PROPS, _kernel  # noqa: E402

from coupfe import solve_increments  # noqa: E402
from coupfe.mesh import Circle, KernelMeshView, uniform_refine_quad  # noqa: E402
from coupfe.operators.element_group import ElementGroup  # noqa: E402
from coupfe.runtime.compiled_element import CompiledElement  # noqa: E402

R1, R2, THETA = 0.5, 2.0, np.pi / 4.0      # wide radius ratio → strong 1/r curvature
A_LAME, B_LAME = 0.0, 1.0e-6               # pure 1/r field; small → linear regime


def u_exact(x):
    """Exact field ``u_r(r) r̂`` at points ``x`` (..., 2)."""
    r = np.linalg.norm(x, axis=-1, keepdims=True)
    ur = A_LAME * r + B_LAME / r
    return ur * (x / r)


def coarse_view(nr=2, nt=4):
    """Annulus-sector Quad4 mesh; arcs classified to circles, radial edges flat."""
    rs = np.linspace(R1, R2, nr + 1)
    ths = np.linspace(0.0, THETA, nt + 1)
    nodes = np.array([[r * np.cos(t), r * np.sin(t)] for r in rs for t in ths])

    def nid(ir, it):
        return ir * (nt + 1) + it

    elems = [[nid(ir, it), nid(ir + 1, it), nid(ir + 1, it + 1), nid(ir, it + 1)]
             for ir in range(nr) for it in range(nt)]
    inner = [nid(0, it) for it in range(nt + 1)]
    outer = [nid(nr, it) for it in range(nt + 1)]
    th0 = [nid(ir, 0) for ir in range(nr + 1)]
    thM = [nid(ir, nt) for ir in range(nr + 1)]
    node_geometry = {**{n: "inner" for n in inner}, **{n: "outer" for n in outer}}
    return KernelMeshView(
        nodes, np.array(elems, dtype=int), dof_per_node=2,
        node_sets={"inner": np.array(inner), "outer": np.array(outer),
                   "th0": np.array(th0), "thM": np.array(thM)},
        node_geometry=node_geometry,
        geometries={"inner": Circle(R1), "outer": Circle(R2)})


def solve_level(level, reembed=True):
    """Refine ``level`` times, solve the Dirichlet problem, return (h, interior error)."""
    v = coarse_view()
    for _ in range(level):
        v = uniform_refine_quad(v, reembed=reembed)

    bnd = np.unique(np.concatenate(
        [v.node_sets[k] for k in ("inner", "outer", "th0", "thM")]))
    is_bnd = np.zeros(v.n_node, dtype=bool)
    is_bnd[bnd] = True

    elem = CompiledElement(_kernel(DEFAULT_PROPS), props=DEFAULT_PROPS,
                           dof_per_node=2, n_svars=0, mcrd=2, n_elem=v.n_elem)
    group = ElementGroup.from_view(v, elem, comps=(0, 1))

    ue = u_exact(v.nodes)
    d = {}
    for n in bnd:
        d[n * 2 + 0] = float(ue[n, 0])
        d[n * 2 + 1] = float(ue[n, 1])
    U, _ = solve_increments([group], np.zeros(v.ndof), v.ndof, d, n_steps=2)

    Uf = U.reshape(-1, 2)
    diff, ref = (Uf - ue)[~is_bnd], ue[~is_bnd]
    interior_err = float(np.linalg.norm(diff) / np.linalg.norm(ref))   # relative L2
    h = (THETA * R2) / (4 * 2 ** level)    # nominal element size
    return h, interior_err, v.n_elem
