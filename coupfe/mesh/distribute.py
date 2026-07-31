"""Mesh distribution — partition into owned/ghost local views.

Splits a :class:`KernelMeshView` into ``nparts`` element partitions, each with a
**memory-local** view: only its owned cells and the nodes they touch (owned + a
one-layer ghost halo). A node is **owned** by the lowest-id part that touches it; the
other parts see it as a **ghost**.

The correctness invariant (gated in ``tests/test_distribute.py``): summing every part's
owned-cell contributions reproduces the serial assembly exactly — which is precisely
what a real-MPI ghost→owner reduction must do. This module is the in-process
decomposition; ``coupfe.assembly.distributed`` supplies the petsc4py execution
path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coupfe.mesh.view import KernelMeshView


def partition_elements(view: KernelMeshView, nparts: int) -> np.ndarray:
    """Coordinate sort-and-chunk along the longest centroid axis.

    Contiguous, balanced, deterministic — good locality (small ghost halos). Returns a
    ``(n_elem,)`` array of part ids in ``[0, nparts)``.
    """
    cent = view.nodes[view.elems].mean(axis=1)          # (n_elem, dim)
    axis = int(np.argmax(cent.max(0) - cent.min(0)))
    order = np.argsort(cent[:, axis], kind="stable")
    parts = np.empty(view.n_elem, dtype=int)
    for p, chunk in enumerate(np.array_split(order, nparts)):
        parts[chunk] = p
    return parts


@dataclass
class LocalMesh:
    """One part's memory-local view (only its owned cells + touched nodes)."""

    part: int
    owned_elems: np.ndarray        # global element ids this part owns
    local_to_global: np.ndarray    # (n_local_node,) global node id per local node
    local_elems: np.ndarray        # (n_owned_elem, nne) in LOCAL node numbering
    owned_node: np.ndarray         # (n_local_node,) bool — this part owns the node
    node_owner: np.ndarray         # (n_local_node,) owning part id per local node

    @property
    def n_local_node(self) -> int:
        return int(self.local_to_global.shape[0])


def node_owners(view: KernelMeshView, parts: np.ndarray, nparts: int) -> np.ndarray:
    """Owning part of each global node (the lowest part id touching it)."""
    owner = np.full(view.n_node, nparts, dtype=int)
    flat_nodes = view.elems.ravel()
    flat_parts = np.repeat(parts, view.elems.shape[1])
    np.minimum.at(owner, flat_nodes, flat_parts)
    return owner


def local_meshes(view: KernelMeshView, parts: np.ndarray, nparts: int):
    """Build the per-part owned/ghost local views from an element partition."""
    owner = node_owners(view, parts, nparts)
    out = []
    for p in range(nparts):
        oe = np.where(parts == p)[0]
        l2g = np.unique(view.elems[oe])                 # touched nodes (owned + ghost)
        local_elems = np.searchsorted(l2g, view.elems[oe])
        out.append(LocalMesh(p, oe, l2g, local_elems,
                             owner[l2g] == p, owner[l2g]))
    return out


def scatter_to_global(local: LocalMesh, local_vec: np.ndarray, n_global_node: int,
                      dof_per_node: int = 1) -> np.ndarray:
    """Scatter a part's local (node-block) vector to a global vector (summed).

    Summing this over all parts equals the serial assembly — the M3 invariant. Under
    real MPI this is the ghost→owner reduction.
    """
    g = np.zeros(n_global_node * dof_per_node)
    gdofs = (local.local_to_global[:, None] * dof_per_node
             + np.arange(dof_per_node)[None, :]).ravel()
    np.add.at(g, gdofs, np.asarray(local_vec).ravel())
    return g
