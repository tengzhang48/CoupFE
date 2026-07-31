"""Serial 3D deformable contact: two soft Hex8 blocks collide. Run with plain ``python run.py``.

The approachable, single-process companion to the distributed demos in ``examples/mpi_smoke``. It shows
the whole contact stack composing through the operator contract:

    solve_dynamics([ bulk element group, inertia, deformable barrier contact ], ...)

Two compressible neo-Hookean **F-bar Hex8** blocks; the top
one is given a downward initial velocity and collides with the bottom one. The **cubic-barrier**
deformable contact across the interface keeps them **penetration-free** (the CCD ``max_step`` bounds
every step so no node ever crosses), both blocks deform on impact, and backward-Euler damping settles
them. Vertex-face only (two flat parallel surfaces → edge-edge would be slow + degenerate; vertex-face
is correct and sufficient here). Contact runs on the **numba** narrow-phase + LBVH broad-phase
(``contact3d_numba``/``bvh_numba``), with a NumPy fallback. This example does
not make a general no-locking or acceleration-parity claim.

Self-check: the blocks **collided** (the minimum interface gap over the trajectory dipped below ``d̂``)
AND stayed **penetration-free** (it never reached 0), with bounded deformation. Prints ``OK`` / ``FAIL``.
"""
from __future__ import annotations

import numpy as np

from coupfe import InertiaOperator, solve_dynamics
from coupfe.mesh import KernelMeshView
from coupfe.operators.contact3d import DeformableBarrierContact3D, point_triangle_coeff_unclassified
from coupfe.operators.element_group import ElementGroup
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

_HEX8_FOR = "coupfe/runtime/elements/neo_hookean_hex8_fbar.for"   # scoped F-bar formulation
NE = 2                                  # NE×NE×NE hexes per block
G, K_BULK, DENSITY = 1.0, 10.0, 1.0     # K/G = 10 (moderate compressibility)
DHAT, KAPPA = 0.04, 1.0e2               # κ ~ K_bulk (matched); CCD owns non-penetration
GAP0 = 1.2 * DHAT                       # top block starts just above the barrier band (separated)
V0 = 0.15                               # gentle downward impact velocity
DT, N_STEPS, DAMP = 0.02, 25, 2.0


def _hex8_block(ne, z0):
    xs = np.linspace(0.0, 1.0, ne + 1); zs = np.linspace(z0, z0 + 1.0, ne + 1)
    nodes = np.array([[x, y, z] for z in zs for y in xs for x in xs], dtype=float)
    nn = ne + 1

    def nid(i, j, k):
        return k * nn * nn + j * nn + i

    elems = [[nid(i, j, k), nid(i+1, j, k), nid(i+1, j+1, k), nid(i, j+1, k),
              nid(i, j, k+1), nid(i+1, j, k+1), nid(i+1, j+1, k+1), nid(i, j+1, k+1)]
             for k in range(ne) for j in range(ne) for i in range(ne)]
    return nodes, np.array(elems, dtype=int)


def _surface_grid(ne, offset, z_pick):
    nn = ne + 1; k = 0 if z_pick == "min" else ne
    return np.array([[offset + k*nn*nn + j*nn + i for i in range(nn)] for j in range(nn)])


def _triangulate(grid):
    nn = grid.shape[0]; tris = []
    for j in range(nn - 1):
        for i in range(nn - 1):
            a, b, c, d = grid[j, i], grid[j, i+1], grid[j+1, i+1], grid[j+1, i]
            tris += [[a, b, d], [b, c, d]]
    return np.array(tris, dtype=int)


def _lumped_mass(nodes, elems, density, ndof):
    M = np.zeros(ndof); nodal = np.zeros(len(nodes))
    for e in elems:
        vol = float(np.prod(nodes[e].max(0) - nodes[e].min(0)))
        np.add.at(nodal, e, density * vol / 8.0)
    for c in range(3):
        M[c::3] = nodal
    return M


def _min_interface_gap(U, nodes, secondary, faces):
    pos = nodes + U.reshape(len(nodes), 3); g = np.inf
    for v in secondary:
        v = int(v)
        for f in faces:
            w = point_triangle_coeff_unclassified(pos[v], pos[f[0]], pos[f[1]], pos[f[2]])
            g = min(g, float(np.linalg.norm(pos[v] - (w[0]*pos[f[0]] + w[1]*pos[f[1]] + w[2]*pos[f[2]]))))
    return g


def main():
    nb, eb = _hex8_block(NE, 0.0)                        # bottom block z∈[0,1]
    Nb = len(nb)
    nt, et = _hex8_block(NE, 1.0 + GAP0)                 # top block, just above
    nodes = np.vstack([nb, nt]); elems = np.vstack([eb, et + Nb])
    faces = _triangulate(_surface_grid(NE, 0, "max"))    # bottom block's top face (primary)
    secondary = _surface_grid(NE, Nb, "min").ravel()     # top block's bottom face (secondary)
    base = np.arange((NE + 1) ** 2)                      # bottom block z=0 face, fixed
    top_block = np.arange(Nb, len(nodes))                # all top-block nodes (carry v0)

    view = KernelMeshView(nodes, elems, dof_per_node=3)
    ndof = view.ndof
    elem = CompiledElement(build_element_kernel(_HEX8_FOR, "neo_hex8_serial"), props=(G, K_BULK),
                           dof_per_node=3, n_svars=0, mcrd=3, n_elem=len(elems))
    grp = ElementGroup.from_view(view, elem, comps=(0, 1, 2))
    M = _lumped_mass(nodes, elems, DENSITY, ndof)
    v0 = np.zeros(ndof); v0[top_block * 3 + 2] = -V0      # downward impact on the top block
    inertia = InertiaOperator(M, ndof, v0=v0, damping=DAMP)
    contact = DeformableBarrierContact3D(nodes, secondary, faces, np.zeros((0, 2), int),
                                         dof_per_node=3, dhat=DHAT, kappa=KAPPA)   # vertex-face only
    dirich = {int(n) * 3 + c: 0.0 for n in base for c in (0, 1, 2)}
    ops = [grp, inertia, contact]

    U = np.zeros(ndof); min_gap = np.inf
    for _ in range(N_STEPS):                              # step-by-step → track the min gap over time
        U, _info = solve_dynamics(ops, U, ndof, dirich, dt=DT, n_steps=1)
        min_gap = min(min_gap, _min_interface_gap(U, nodes, secondary, faces))

    max_u = float(np.max(np.abs(U)))
    collided = min_gap < DHAT
    penetration_free = min_gap > 0.0
    ok = collided and penetration_free and max_u < 0.2
    print(f"nodes={len(nodes)} hexes={len(elems)} ndof={ndof}  max|U|={max_u:.4f}")
    print(f"min interface gap over trajectory = {min_gap:.3e}  (d̂={DHAT})  "
          f"collided={collided} penetration_free={penetration_free}  -> {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
