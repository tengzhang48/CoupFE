"""3D distributed dynamics, deformable barrier, and friction smoke. Run under mpirun.

The 3D analog of `distributed_dynamics_friction.py` (2D) and the frictional sibling of the 3D
collision capstone `distributed_dynamics_3d_blocks.py`. Two F-bar Hex8 blocks are pressed into light
contact and the top block is sheared sideways (`+x`); ppf smoothed friction on the 3D tangent plane
(`P = I − n⊗n`) at the cross-rank interface resists the relative sliding of the top block's bottom
over the bottom block's top. Friction rides the same machinery as 2D — `_DistDeformableContact3D`
threads `mu` into each rank's owned `DeformableBarrierContact3D` (vertex-face friction on the owned
secondary vertices) and advances the step-start reference `_x0` in `commit`.
Because that state is path-dependent, saved results should be compared across
rank counts rather than assuming distribution leaves the trajectory unchanged.

Modest gravity supplies the sustained normal load needed for friction. A held
overlap under a purely repulsive barrier relaxes toward the activation boundary
and does not provide the same load. Parameters keep the dimensionless
gravitational strain moderate. This flat, parallel interface uses vertex-face
contact only, avoiding redundant coplanar edge-edge candidates. See
`docs/theory/contact_dynamics.md`.

The current run checks that friction reduces slip, the solver converges, and the
reported gap stays positive. An optional output path supports an external
same-revision rank comparison; no retained multi-rank record ships here.

    OMP_NUM_THREADS=1 mpirun -n 4 python examples/mpi_smoke/distributed_dynamics_3d_friction.py
"""

from __future__ import annotations

import sys

import numpy as np
from petsc4py import PETSc

from coupfe.assembly.distributed import element_partition, solve_dynamics_distributed
from coupfe.mesh import KernelMeshView
from coupfe.operators.contact3d import point_triangle_coeff_unclassified
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

_HEX8_FOR = "coupfe/runtime/elements/neo_hookean_hex8_fbar.for"   # scoped F-bar formulation
NE = 2                                  # NE×NE×NE hexes per block
G, K_BULK, DENSITY = 1.0, 10.0, 1.0     # K/G = 10 (moderate compressibility)
DENSITY, GRAV = 1.0, 0.4                # modest gravity: εg = ρgL/G = 0.4 (seats + sustains λ_n, converges)
DHAT, KAPPA = 0.04, 2.0e3              # stiff barrier (inertia regularizes under dynamics)
GAP0 = 0.5 * DHAT                       # top block starts INSIDE the band → gravity seats it immediately
MU, SHEAR = 0.8, 0.05                   # friction coefficient; total +x drive (gentle → secondaries stay on the face)
FRICTION_EPS = 2.0e-3                   # smoothing length for this near-stick study
DT, N_STEPS, DAMP = 0.02, 50, 3.0      # example time-step and damping choices


def hex8_block(ne, z0):
    """ne³ Hex8 mesh of [0,1]²×[z0,z0+1], C3D8 corner ordering (bottom z CCW, then top z CCW)."""
    xs = np.linspace(0.0, 1.0, ne + 1)
    zs = np.linspace(z0, z0 + 1.0, ne + 1)
    nodes = np.array([[x, y, z] for z in zs for y in xs for x in xs], dtype=float)
    nn = ne + 1

    def nid(i, j, k):
        return k * nn * nn + j * nn + i

    elems = []
    for k in range(ne):
        for j in range(ne):
            for i in range(ne):
                elems.append([nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                              nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1),
                              nid(i, j + 1, k + 1)])
    return nodes, np.array(elems, dtype=int)


def _surface_grid(ne, offset, z_pick):
    nn = ne + 1
    k = 0 if z_pick == "min" else ne
    return np.array([[offset + k * nn * nn + j * nn + i for i in range(nn)] for j in range(nn)])


def _triangulate(grid):
    nn = grid.shape[0]
    tris = []
    for j in range(nn - 1):
        for i in range(nn - 1):
            a, b, c, d = grid[j, i], grid[j, i + 1], grid[j + 1, i + 1], grid[j + 1, i]
            tris += [[a, b, d], [b, c, d]]
    return np.array(tris, dtype=int)


def build():
    nb, eb = hex8_block(NE, 0.0)                          # bottom block: z in [0,1]
    Nb = len(nb)
    nt, et = hex8_block(NE, 1.0 + GAP0)                   # top block: z in [1+GAP0, 2+GAP0]
    nodes = np.vstack([nb, nt])
    elems = np.vstack([eb, et + Nb])
    faces = _triangulate(_surface_grid(NE, 0, "max"))    # bottom block's top face (primary)
    secondary = _surface_grid(NE, Nb, "min").ravel()     # top block's bottom face (secondary verts)
    base = np.arange((NE + 1) ** 2)                      # bottom block's z=0 face, fixed
    top_face = _surface_grid(NE, Nb, "max").ravel()      # top block's TOP face (driven in +x)
    top_block = np.arange(Nb, len(nodes))                # all top-block nodes (carry gravity)
    return nodes, elems, secondary, faces, base, top_face, top_block


def lumped_mass_hex8(nodes, elems, density, ndof):
    M = np.zeros(ndof)
    nodal = np.zeros(len(nodes))
    for e in elems:
        vol = float(np.prod(nodes[e].max(0) - nodes[e].min(0)))   # axis-aligned box volume
        np.add.at(nodal, e, density * vol / 8.0)
    for c in range(3):
        M[c::3] = nodal
    return M


def _min_interface_gap(U, nodes, secondary, faces):
    pos = nodes + U.reshape(len(nodes), 3)
    g = np.inf
    for v in secondary:
        v = int(v)
        for f in faces:
            w = point_triangle_coeff_unclassified(pos[v], pos[f[0]], pos[f[1]], pos[f[2]])
            g = min(g, float(np.linalg.norm(pos[v] - (w[0]*pos[f[0]] + w[1]*pos[f[1]] + w[2]*pos[f[2]]))))
    return g


def main():
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    nodes, elems, secondary, faces, base, top_face, top_block = build()
    view = KernelMeshView(nodes, elems, dof_per_node=3)
    ndof = view.ndof
    M = lumped_mass_hex8(nodes, elems, DENSITY, ndof)
    force = np.zeros(ndof)
    force[top_block * 3 + 2] = -GRAV * M[top_block * 3 + 2]    # gravity (−z) seats the top block

    my_gm, my_coords, _ = element_partition(view, rank, size)
    elem = CompiledElement(build_element_kernel(_HEX8_FOR, "neo_hex8_dist_fr"),
                           props=(G, K_BULK), dof_per_node=3, n_svars=0, mcrd=3,
                           n_elem=max(len(my_gm), 1))

    # bottom block fixed; top face dragged +x (ramped) with z,y FREE → gravity seats the block onto
    # contact (sustained λ_n) while the shear slides the interface; friction resists.
    def dirichlet(t):
        frac = t / (N_STEPS * DT)
        d = {int(n) * 3 + c: 0.0 for n in base for c in (0, 1, 2)}
        for n in top_face:
            d[int(n) * 3 + 0] = SHEAR * frac        # x: ramped shear (z, y free → gravity seats it)
        return d

    def run(mu):
        # vertex-face ONLY (flat interface; edge-edge would be slow + degenerate here)
        dc = {"kind": "barrier", "nodes_ref": nodes, "vertices": secondary, "faces": faces,
              "edges": np.zeros((0, 2), int), "dhat": DHAT, "kappa": KAPPA, "mu": mu,
              "friction_eps": FRICTION_EPS}
        U, info = solve_dynamics_distributed(
            ndof, my_gm, my_coords, 3, elem.element_rk_batch, dirichlet, M,
            dt=DT, n_steps=N_STEPS, damping=DAMP, force=force, tol=1e-7,
            pc="lu", solver="superlu_dist", deformable_contact=dc)
        return U, info

    U_fric, info = run(MU)
    U_free, _ = run(0.0)

    if rank == 0:
        # interface slip = mean +x displacement of the secondary (top-block bottom) nodes
        slip_fric = float(np.abs(np.mean(U_fric[secondary * 3 + 0])))
        slip_free = float(np.abs(np.mean(U_free[secondary * 3 + 0])))
        min_gap = _min_interface_gap(U_fric, nodes, secondary, faces)
        converged = info["rnorm"] < 1e-6 and not info["ksp_diverged"]
        held = slip_fric < 0.7 * slip_free          # friction reduces interface sliding
        penetration_free = min_gap > 0.0
        ok = converged and penetration_free and held and slip_free > 1e-3
        if len(sys.argv) > 1:
            np.save(sys.argv[1], U_fric)
        print(f"[size={size}] my_ne={info['my_ne']} newton(last)={info['n_newton']} "
              f"|R|={info['rnorm']:.2e} converged={converged} min_gap={min_gap:.3e} (d̂={DHAT})")
        print(f"[size={size}] interface slip: μ={MU} -> {slip_fric:.4e}, μ=0 -> {slip_free:.4e} "
              f"(ratio {slip_fric / max(slip_free, 1e-30):.2f}) held={held} pen_free={penetration_free} "
              f"-> {'OK' if ok else 'FAIL'}", flush=True)
    comm.barrier()


if __name__ == "__main__":
    main()
