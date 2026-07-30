"""Structured Hex8 torus ("tire") mesh for a RESEARCH contact workflow.

A **hollow thick-walled torus** standing in the xz-plane (axle along y), resting tread-down on a
rigid ground at the bottom. It is inspired by the general GetFEM tire problem,
but no exact upstream source, parameters, or geometry record is retained, so it
is not a GetFEM reproduction. This implementation uses linear Hex8 with a few
layers through the wall.

Parametrisation — main circle (the wheel) in the xz-plane, tube cross-section in the (radial, axle)
plane:

    node(φ, θ, ρ) = ( (R + ρ cosθ) cosφ ,  ρ sinθ ,  (R + ρ cosθ) sinφ )

with φ around the wheel (periodic), θ around the tube cross-section (periodic), ρ ∈ [r_in, r_out]
through the wall (not periodic). The tire's lowest point is z = −(R + r_out); the ground sits there.

Connectivity is structured; the reviewed public mesh gate checks dimensions
and positive corner Jacobians.
"""
from __future__ import annotations

import numpy as np


def torus_hex_mesh(R=1.0, r_in=0.30, r_out=0.45, n_phi=64, n_theta=20, n_rho=2):
    """Hollow-torus Hex8 mesh. Returns (nodes (N,3), elems (M,8))."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)          # periodic
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)      # periodic
    rho = np.linspace(r_in, r_out, n_rho + 1)                           # through-wall

    # node id: (i_phi, j_theta, k_rho) -> contiguous, k fastest
    def nid(i, j, k):
        return (i % n_phi) * (n_theta * (n_rho + 1)) + (j % n_theta) * (n_rho + 1) + k

    nodes = np.empty((n_phi * n_theta * (n_rho + 1), 3), dtype=float)
    for i, f in enumerate(phi):
        cf, sf = np.cos(f), np.sin(f)
        for j, t in enumerate(theta):
            ct, st = np.cos(t), np.sin(t)
            for k, rr in enumerate(rho):
                rad = R + rr * ct
                nodes[nid(i, j, k)] = (rad * cf, rr * st, rad * sf)

    elems = []
    for i in range(n_phi):
        for j in range(n_theta):
            for k in range(n_rho):
                # bottom face (ρ=k) CCW, then top face (ρ=k+1) — see _orient check below
                elems.append([
                    nid(i, j, k),     nid(i + 1, j, k),     nid(i + 1, j + 1, k),     nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1),
                ])
    elems = np.array(elems, dtype=int)
    return _orient_positive(nodes, elems)


def _hex_jacobian_sign(nodes, e):
    """Sign of the Hex8 Jacobian at the element centre (ξ=η=ζ=0)."""
    X = nodes[e]
    # dN/dξ at centre for the standard node order (±1 corners, bottom then top).
    s = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    dNdxi = 0.125 * s
    J = dNdxi.T @ X
    return np.sign(np.linalg.det(J))


def _orient_positive(nodes, elems):
    """Ensure every hex has a positive Jacobian; flip top/bottom if not (consistent winding)."""
    if _hex_jacobian_sign(nodes, elems[0]) < 0:
        elems = elems[:, [4, 5, 6, 7, 0, 1, 2, 3]]   # swap bottom<->top faces
    bad = sum(_hex_jacobian_sign(nodes, e) <= 0 for e in elems)
    if bad:
        raise ValueError(f"{bad} elements still have non-positive Jacobian")
    return nodes, elems


def _cross_section_rho(nodes, R):
    """Tube cross-section radial coordinate ρ of each node (distance from the tube centre circle)."""
    rad_xz = np.hypot(nodes[:, 0], nodes[:, 2])          # distance from the y-axis (axle)
    return np.hypot(rad_xz - R, nodes[:, 1])             # ρ in the (radial, axle) plane


def tread_surface_nodes(nodes, R, r_out, tol=1e-9):
    """Outer-tread surface node indices (ρ ≈ r_out — the part that contacts the ground)."""
    return np.nonzero(_cross_section_rho(nodes, R) > r_out - tol)[0]


def rim_surface_nodes(nodes, R, r_in, tol=1e-9):
    """Inner-rim surface node indices (ρ ≈ r_in — the wheel mount, held fixed)."""
    return np.nonzero(_cross_section_rho(nodes, R) < r_in + tol)[0]


if __name__ == "__main__":
    from coupfe.mesh import check_positive_jacobian, KernelMeshView

    R, r_in, r_out = 1.0, 0.30, 0.45
    nodes, elems = torus_hex_mesh(R, r_in, r_out)
    view = KernelMeshView(nodes, elems, dof_per_node=3)
    bad = check_positive_jacobian(view)                 # returns indices of bad elements (corner-wise)
    ok = (len(bad) == 0)
    z_lo = nodes[:, 2].min()
    print(f"torus: {len(nodes)} nodes, {len(elems)} Hex8")
    print(f"  outer radius (xz) = {np.hypot(nodes[:,0], nodes[:,2]).max():.3f} (expect {R+r_out:.3f})")
    print(f"  width along y     = {nodes[:,1].max()-nodes[:,1].min():.3f} (expect {2*r_out:.3f})")
    print(f"  lowest z          = {z_lo:.3f} (expect {-(R+r_out):.3f})  -> ground here")
    print(f"  tread nodes       = {len(tread_surface_nodes(nodes, R, r_out))}")
    print(f"  positive Jacobian = {ok}")
