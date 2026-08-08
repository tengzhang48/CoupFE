"""Serial integration gate: mesh → refine → element → contact → solve.

The test builds a structured block as a ``KernelMeshView``, refines it while
propagating labels, wraps a compiled neo-Hookean element, adds rigid contact,
and solves with load-stepped Newton. Distributed paths have separate scoped
tests and examples.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "neo_hookean_block"))
try:
    from block import DEFAULT_PROPS, _kernel, structured_quad_mesh
    _kernel(DEFAULT_PROPS)                              # build the f2py kernel once
    _HAVE = True
except Exception as exc:                               # pragma: no cover
    _HAVE = False
    _WHY = str(exc)

pytestmark = pytest.mark.skipif(
    not _HAVE, reason="neo-Hookean kernel unavailable"
    + (f": {_WHY}" if not _HAVE else ""))

if _HAVE:
    from coupfe import (
        assemble_residual,
        neo_hookean_kernel_props,
        solve_increments,
    )
    from coupfe.mesh import KernelMeshView, check_positive_jacobian, uniform_refine_quad
    from coupfe.operators.contact import HalfSpace, RigidContact
    from coupfe.operators.element_group import ElementGroup
    from coupfe.runtime.compiled_element import CompiledElement


def test_serial_pipeline_refined_block_on_plane():
    # 1. mesh as a KernelMeshView, with labelled boundary node sets
    nodes, elems = structured_quad_mesh(4, 4, 1.0, 1.0)
    tol = 1e-9
    sets = {
        "top": np.nonzero(np.abs(nodes[:, 1] - 1.0) < tol)[0],
        "bottom": np.nonzero(np.abs(nodes[:, 1]) < tol)[0],
        "left": np.nonzero(np.abs(nodes[:, 0]) < tol)[0],
    }
    view = KernelMeshView(nodes, elems, dof_per_node=2, node_sets=sets)

    # 2. refine (labels propagate); the mesh stays valid
    view = uniform_refine_quad(view)
    assert check_positive_jacobian(view) == []
    assert view.n_elem == 64                            # 16 → 64 under one refinement

    # 3. compiled element over the refined view
    elem = CompiledElement(_kernel(DEFAULT_PROPS),
                           props=neo_hookean_kernel_props(*DEFAULT_PROPS), dof_per_node=2,
                           n_svars=0, mcrd=2, n_elem=view.n_elem)
    group = ElementGroup.from_view(view, elem, comps=(0, 1))

    # 4. contact: the bottom edge onto a rigid plane just below it
    yplane = -0.01
    contact = RigidContact(view.nodes, view.node_sets["bottom"],
                           HalfSpace([0.0, yplane], [0.0, 1.0]),
                           dof_per_node=2, comps=(0, 1), k=1.0e4)

    # 5. BCs + load-stepped solve: fix left-x, push the top down onto the plane
    d = {int(n) * 2 + 0: 0.0 for n in view.node_sets["left"]}
    d.update({int(n) * 2 + 1: -0.05 for n in view.node_sets["top"]})
    U, nit = solve_increments([group, contact], np.zeros(view.ndof), view.ndof, d,
                              n_steps=4)

    # The composed serial path reaches its stated residual/contact checks.
    R, _ = assemble_residual([group, contact], U, None, 1.0, 1.0, view.ndof)
    free = np.ones(view.ndof, dtype=bool)
    free[np.array(sorted(d))] = False
    assert np.linalg.norm(R[free]) < 1e-5               # converged at the free DOFs
    ybot = view.nodes[view.node_sets["bottom"], 1] + U.reshape(-1, 2)[view.node_sets["bottom"], 1]
    assert ybot.min() > yplane - 2e-3                   # contact held — no penetration
    assert np.abs(U).max() > 0.01                       # the block actually deformed
    assert nit < 60                                     # load-stepped Newton, sane count
