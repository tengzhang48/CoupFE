"""``KernelMeshView`` — the compact, rank-local mesh the operators consume.

This is the narrow contract between meshing and physics. Operators (``ElementGroup``,
contact, …) see only this view — never a DMPlex/forest object, never cones/closures/
sections. That keeps the kernels backend-agnostic: a serial mesh, a DMPlex partition,
or a native forest can all be translated to the *same* view. Format/backend
adapters remain application-owned; core consumes this contract only.

Boundary *classification* travels with the mesh: ``node_geometry`` maps a node to a
geometry key in ``geometries`` (which curve/surface it sits on), so refinement can
re-embed new boundary nodes on the true geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class KernelMeshView:
    """Compact rank-local mesh + labels + boundary geometry classification.

    Parameters
    ----------
    nodes : (Nnode, dim) float array — reference coordinates.
    elems : (Nelem, nne) int array — element connectivity.
    dof_per_node : int — uniform global DOFs per node.
    node_sets : dict[str, int array] — named node groups (Dirichlet sets, labels).
    elem_sets : dict[str, int array] — named element groups (material regions).
    node_geometry : dict[int, str] — node index → geometry key it is classified onto.
    geometries : dict[str, GeometryBackend] — the geometry sources.
    """

    nodes: np.ndarray
    elems: np.ndarray
    dof_per_node: int = 2
    node_sets: Dict[str, np.ndarray] = field(default_factory=dict)
    elem_sets: Dict[str, np.ndarray] = field(default_factory=dict)
    node_geometry: Dict[int, str] = field(default_factory=dict)
    geometries: Dict[str, object] = field(default_factory=dict)

    @property
    def n_node(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elem(self) -> int:
        return self.elems.shape[0]

    @property
    def ndof(self) -> int:
        return self.n_node * self.dof_per_node

    def boundary_error(self) -> float:
        """Max distance of every geometry-classified node from its geometry.

        The mesh-side validation oracle: after refinement + re-embedding this is ~0
        (every classified node lies on its curve/surface). Without re-embedding it is
        the chord sagitta (> 0) — the broken control.
        """
        worst = 0.0
        for n, key in self.node_geometry.items():
            x = self.nodes[n]
            worst = max(worst, float(np.linalg.norm(x - self.geometries[key].project(x))))
        return worst
