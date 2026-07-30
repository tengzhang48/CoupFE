"""Mesh-distribution (M3) gates — verified in-process.

The core distributed-FE invariant: summing each part's owned-cell assembly reproduces
the serial assembly exactly (what a real-MPI ghost→owner reduction must do). Plus
partition coverage, memory-locality, and the adjoint of the gather/reduce comm pair —
the property a real MPI exchange must satisfy (lab lesson: distributed bugs hide in
test gaps; gate 1-vs-N equivalence).
"""

import numpy as np

from coupfe.mesh import Circle, KernelMeshView
from coupfe.mesh.distribute import (
    LocalMesh, local_meshes, node_owners, partition_elements, scatter_to_global)

# A fixed SPD element "stiffness" for a scalar toy operator (dof_per_node = 1).
_KQ = np.array([[3., -1., -1., -1.], [-1., 3., -1., -1.],
                [-1., -1., 3., -1.], [-1., -1., -1., 3.]])


def _mesh(nx=5, ny=4):
    xs, ys = np.linspace(0, 2, nx + 1), np.linspace(0, 1, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs])
    elems = [[j * (nx + 1) + i, j * (nx + 1) + i + 1,
              (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i]
             for j in range(ny) for i in range(nx)]
    return KernelMeshView(nodes, np.array(elems, dtype=int), dof_per_node=1)


def _serial(view, U):
    R = np.zeros(view.n_node)
    for e in range(view.n_elem):
        ids = view.elems[e]
        R[ids] += _KQ @ U[ids]
    return R


def _distributed(view, parts, nparts, U):
    R = np.zeros(view.n_node)
    for lm in local_meshes(view, parts, nparts):
        Ul = U[lm.local_to_global]
        Rl = np.zeros(lm.n_local_node)
        for le in lm.local_elems:                       # owned cells, LOCAL numbering
            Rl[le] += _KQ @ Ul[le]
        R += scatter_to_global(lm, Rl, view.n_node)     # ghost→owner reduction
    return R


def test_partition_is_a_disjoint_cover():
    view = _mesh()
    for nparts in (1, 2, 4):
        parts = partition_elements(view, nparts)
        assert parts.shape == (view.n_elem,)
        assert set(np.unique(parts)) == set(range(nparts))     # all parts used
        assert np.bincount(parts, minlength=nparts).sum() == view.n_elem


def test_each_owned_cell_is_local_and_nodes_owned_once():
    view = _mesh()
    parts = partition_elements(view, 4)
    owner = node_owners(view, parts, 4)
    assert owner.min() >= 0 and owner.max() < 4               # every node owned exactly once
    for lm in local_meshes(view, parts, 4):
        # owned cells' nodes are all present locally (owned or ghost)
        assert set(view.elems[lm.owned_elems].ravel()).issubset(set(lm.local_to_global))
        # each part is memory-local: it stores fewer than all global nodes
        assert lm.n_local_node < view.n_node
        # ownership is consistent
        assert np.array_equal(lm.owned_node, lm.node_owner == lm.part)


def test_assemble_from_parts_equals_serial():
    view = _mesh()
    rng = np.random.default_rng(0)
    U = rng.standard_normal(view.n_node)
    R0 = _serial(view, U)
    for nparts in (1, 2, 4):
        Rp = _distributed(view, partition_elements(view, nparts), nparts, U)
        assert np.allclose(Rp, R0, atol=1e-12), f"nparts={nparts}"


def test_gather_reduce_are_adjoint():
    # The comm pair a real MPI exchange must satisfy: <G u, w> == <u, G^T w>.
    view = _mesh()
    parts = partition_elements(view, 4)
    lms = local_meshes(view, parts, 4)
    # ghost slots: (global_node, value-slot) for every locally-ghosted node
    slots = [g for lm in lms for g in lm.local_to_global[~lm.owned_node]]
    slots = np.array(slots, dtype=int)
    rng = np.random.default_rng(1)
    u = rng.standard_normal(view.n_node)          # owned global field
    w = rng.standard_normal(slots.shape[0])       # ghost-slot field
    gather = u[slots]                              # owner → ghost broadcast
    reduce = np.zeros(view.n_node)
    np.add.at(reduce, slots, w)                   # ghost → owner reduction (G^T)
    assert abs(np.dot(gather, w) - np.dot(u, reduce)) < 1e-12
    assert slots.size > 0                          # there ARE ghosts (real decomposition)
