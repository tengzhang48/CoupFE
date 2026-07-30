"""Mesh M1 gates: geometry re-embedding under refinement + the Jacobian invariant.

Independent oracle: after uniform refinement with geometry re-embedding, every
boundary-classified node lies *exactly* on its curve (``boundary_error → 0``). Broken
controls: (1) the un-projected chord midpoint is *off* the circle (so re-embedding does
real work); (2) an inverted element is caught by the Jacobian gate.
"""

import numpy as np

from coupfe.mesh import (
    Circle, KernelMeshView, check_positive_jacobian, uniform_refine_quad)
from coupfe.operators.element_group import ElementGroup

R1, R2, THETA = 1.0, 1.5, np.pi / 3


def _arc_mesh(nr=2, nt=3):
    """Annulus-sector Quad4 mesh; inner/outer rings classified to circles."""
    rs = np.linspace(R1, R2, nr + 1)
    ths = np.linspace(0.0, THETA, nt + 1)
    nodes = np.array([[r * np.cos(t), r * np.sin(t)] for r in rs for t in ths])

    def nid(ir, it):
        return ir * (nt + 1) + it

    elems = [[nid(ir, it), nid(ir + 1, it), nid(ir + 1, it + 1), nid(ir, it + 1)]
             for ir in range(nr) for it in range(nt)]
    inner = [nid(0, it) for it in range(nt + 1)]
    outer = [nid(nr, it) for it in range(nt + 1)]
    node_geometry = {**{n: "inner" for n in inner}, **{n: "outer" for n in outer}}
    return KernelMeshView(
        nodes, np.array(elems, dtype=int), dof_per_node=2,
        node_sets={"inner": np.array(inner), "outer": np.array(outer)},
        node_geometry=node_geometry,
        geometries={"inner": Circle(R1), "outer": Circle(R2)})


def test_refine_reembeds_curved_boundary():
    v = _arc_mesh()
    assert v.boundary_error() < 1e-12          # coarse nodes exactly on the circles
    assert check_positive_jacobian(v) == []

    v1 = uniform_refine_quad(v)
    assert check_positive_jacobian(v1) == []
    assert v1.boundary_error() < 1e-12         # refined boundary RE-EMBEDDED on circles
    assert v1.n_elem == 4 * v.n_elem

    v2 = uniform_refine_quad(v1)
    assert v2.boundary_error() < 1e-12
    assert v2.n_elem == 16 * v.n_elem

    # labels preserved and grown: an arc of n nodes / (n-1) edges gains (n-1) midpoints
    out0, out1 = v.node_sets["outer"], v1.node_sets["outer"]
    assert set(out0).issubset(set(out1))
    assert len(out1) == 2 * len(out0) - 1


def test_broken_control_without_reembedding_is_off_the_circle():
    v = _arc_mesh()
    outer = v.node_sets["outer"]
    chord_mid = 0.5 * (v.nodes[outer[0]] + v.nodes[outer[1]])
    # un-projected midpoint sits inside the circle (the sagitta) — off the geometry
    assert np.linalg.norm(chord_mid) < R2 - 1e-4

    # re-embedding fixes exactly this: the corresponding refined node is ON the circle
    v1 = uniform_refine_quad(v)
    o1 = v1.node_sets["outer"]
    near = o1[np.argmin(np.linalg.norm(v1.nodes[o1] - chord_mid, axis=1))]
    assert abs(np.linalg.norm(v1.nodes[near]) - R2) < 1e-12


def test_jacobian_gate_catches_inverted_element():
    v = _arc_mesh()
    assert check_positive_jacobian(v) == []
    bad = KernelMeshView(v.nodes.copy(), v.elems.copy(), 2)
    bad.elems[0] = bad.elems[0][[0, 3, 2, 1]]   # reverse orientation of element 0
    assert 0 in check_positive_jacobian(bad)


def test_element_group_from_view_maps_dofs():
    # No kernel needed: ElementGroup.__init__ only builds the DOF map from the view.
    v = _arc_mesh()
    g = ElementGroup.from_view(v, element=object(), comps=(0, 1))
    assert g.gm.shape == (v.n_elem, 4 * 2)      # ndofel = nne * len(comps)
    assert np.array_equal(g.elems, v.elems)
