"""CoupFE meshing — the geometry + kernel-view contracts (serial; distributed later).

Operators consume a :class:`KernelMeshView` only; geometry (curvature) is a separate
:class:`GeometryBackend`. Uniform refinement re-embeds curved boundaries onto the
geometry. See ``docs/dev/distributed_mesh.md``.
"""

from coupfe.mesh.geometry import Circle, GeometryBackend, Plane, Sphere
from coupfe.mesh.view import KernelMeshView
from coupfe.mesh.refine import check_positive_jacobian, uniform_refine_quad
from coupfe.mesh.distribute import (
    LocalMesh, local_meshes, node_owners, partition_elements, scatter_to_global)

__all__ = [
    "GeometryBackend",
    "Circle",
    "Sphere",
    "Plane",
    "KernelMeshView",
    "uniform_refine_quad",
    "check_positive_jacobian",
    "partition_elements",
    "local_meshes",
    "node_owners",
    "scatter_to_global",
    "LocalMesh",
]
