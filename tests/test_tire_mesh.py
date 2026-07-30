"""Gate for the torus ("tire") Hex8 mesh — exact dimensions + element validity (Phase 2a)."""
from __future__ import annotations

import numpy as np

from coupfe.mesh import KernelMeshView, check_positive_jacobian
from examples.tire_contact.mesh import torus_hex_mesh, tread_surface_nodes


def test_torus_mesh_dimensions_and_validity():
    R, r_in, r_out = 1.0, 0.30, 0.45
    n_phi, n_theta, n_rho = 32, 12, 2
    nodes, elems = torus_hex_mesh(R, r_in, r_out, n_phi=n_phi, n_theta=n_theta, n_rho=n_rho)

    assert elems.shape == (n_phi * n_theta * n_rho, 8)
    rad_xz = np.hypot(nodes[:, 0], nodes[:, 2])
    assert np.isclose(rad_xz.max(), R + r_out, atol=1e-9)            # outer tread radius
    assert np.isclose(rad_xz.min(), R - r_out, atol=1e-9)            # inner radius
    assert np.isclose(nodes[:, 1].max() - nodes[:, 1].min(), 2 * r_out, atol=1e-9)  # width along the axle (y)
    assert np.isclose(nodes[:, 2].min(), -(R + r_out), atol=1e-9)    # lowest point = ground plane

    view = KernelMeshView(nodes, elems, dof_per_node=3)
    assert len(check_positive_jacobian(view)) == 0, "non-positive Jacobian elements present"

    assert len(tread_surface_nodes(nodes, R, r_out)) == n_phi * n_theta   # outer surface ring
