"""Uniform Quad4 refinement with geometry re-embedding + a Jacobian invariant.

Each quad → 4 sub-quads (edge midpoints + a center node, shared edge nodes deduped).
A new edge-midpoint **inherits the geometry classification** of its parent edge when
both endpoints lie on the *same* geometry, and is then **projected** onto it — so a
refined curved boundary lands on the true curve, not on the chord. Labels (node/elem
sets) propagate. ``check_positive_jacobian`` is the mesh-side correctness gate.
"""

from __future__ import annotations

import numpy as np

from coupfe.mesh.view import KernelMeshView


def _quad4_dN(xi, eta):
    """Bilinear shape-function derivatives d(N)/d(xi,eta), node order bl,br,tr,tl."""
    dNdxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4.0
    dNdeta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4.0
    return np.column_stack([dNdxi, dNdeta])


def check_positive_jacobian(view: KernelMeshView):
    """Return the indices of elements with a non-positive Jacobian at any corner.

    Empty ⇒ every element is valid. Boundary projection can invert a cell, so this
    must pass before a refined level is used (plan §26.8)."""
    bad = []
    for e in range(view.n_elem):
        X = view.nodes[view.elems[e]]
        if X.shape[1] != 2:
            continue
        ok = all(np.linalg.det(_quad4_dN(xi, eta).T @ X) > 0.0
                 for xi in (-1.0, 1.0) for eta in (-1.0, 1.0))
        if not ok:
            bad.append(e)
    return bad


def uniform_refine_quad(view: KernelMeshView, reembed: bool = True) -> KernelMeshView:
    """One level of uniform Quad4 refinement of ``view``.

    With ``reembed`` (default) new boundary midpoints are projected onto their
    classified geometry. ``reembed=False`` leaves them on the chord — the broken
    control that shows re-embedding matters (a faceted, not curved, boundary)."""
    nodes = [view.nodes[i].copy() for i in range(view.n_node)]
    new_geom = dict(view.node_geometry)
    edge_mid = {}

    def midpoint(a, b):
        key = (a, b) if a < b else (b, a)
        if key in edge_mid:
            return edge_mid[key]
        idx = len(nodes)
        nodes.append(0.5 * (view.nodes[a] + view.nodes[b]))
        ga, gb = view.node_geometry.get(a), view.node_geometry.get(b)
        if ga is not None and ga == gb:                # both on the same curve/surface
            new_geom[idx] = ga
        edge_mid[key] = idx
        return idx

    new_elems, parent = [], []
    for e in range(view.n_elem):
        n0, n1, n2, n3 = (int(v) for v in view.elems[e])
        m01, m12 = midpoint(n0, n1), midpoint(n1, n2)
        m23, m30 = midpoint(n2, n3), midpoint(n3, n0)
        c = len(nodes)
        nodes.append(0.25 * sum(view.nodes[k] for k in (n0, n1, n2, n3)))
        new_elems += [[n0, m01, c, m30], [m01, n1, m12, c],
                      [c, m12, n2, m23], [m30, c, m23, n3]]
        parent += [e, e, e, e]

    nodes = np.array(nodes)
    if reembed:
        for idx in range(view.n_node, len(nodes)):     # re-embed new boundary nodes
            if idx in new_geom:
                nodes[idx] = view.geometries[new_geom[idx]].project(nodes[idx])

    new_node_sets = {}
    for name, ids in view.node_sets.items():
        s = {int(i) for i in ids}
        out = set(s)
        out.update(idx for (a, b), idx in edge_mid.items() if a in s and b in s)
        new_node_sets[name] = np.array(sorted(out), dtype=int)

    parent = np.array(parent)
    new_elem_sets = {}
    for name, ids in view.elem_sets.items():
        s = {int(i) for i in ids}
        new_elem_sets[name] = np.array(
            [ci for ci in range(len(new_elems)) if int(parent[ci]) in s], dtype=int)

    return KernelMeshView(nodes, np.array(new_elems, dtype=int), view.dof_per_node,
                          new_node_sets, new_elem_sets, new_geom, view.geometries)
