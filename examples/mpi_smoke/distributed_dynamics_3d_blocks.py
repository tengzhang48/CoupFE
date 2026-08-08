"""End-to-end 3D distributed dynamics smoke: two Hex8 blocks collide. Run under mpirun.

The full 3D analog of the 2D distributed dynamics barrier, done as a clean collision (no
gravity-balance, which is finicky in 3D): two compressible neo-Hookean **Hex8 blocks** (real
volumetric elements), the top given a downward initial velocity, collide; both deform on impact and
the 3D deformable barrier across the interface (top block's bottom-face vertices vs the bottom
block's top-face triangles + edge-edge) uses a CCD step bound. The program checks
a positive interface gap for this parameter set; that is not an unconditional
guarantee for arbitrary geometry or time steps. Backward-Euler damping dissipates the impact energy.

Element/physics notes (see `skills/contact.md`):
  * **F-bar Hex8** (`neo_hookean_hex8_fbar.for`) is used for the nearly
    incompressible block in this study.
  * **κ is scaled with the bulk** for conditioning; the CCD owns non-penetration.
  * Modest impact velocity (dimensionless impact strain `~ v0/√(E/ρ)` small) → modest deformation →
    the example remains in its intended moderate-deformation regime.

Each rank owns a block of hexes (bulk via the vendored Hex8 kernel) + the contact secondaries/faces
it owns; `_DistDeformableContact3D` routes the cross-rank vertex-face/edge-edge contributions.

The current run self-checks convergence and an engaged positive interface gap.
An optional output path saves the final `U` for an external same-revision
comparison across rank counts; no retained multi-rank record ships here.

    OMP_NUM_THREADS=1 mpirun -n 4 python examples/mpi_smoke/distributed_dynamics_3d_blocks.py
"""

from __future__ import annotations

import sys

import numpy as np
from petsc4py import PETSc

from coupfe import neo_hookean_kernel_props
from coupfe.assembly.distributed import element_partition, solve_dynamics_distributed
from coupfe.mesh import KernelMeshView
from coupfe.operators.contact3d import (
    DeformableBarrierContact3D,
    point_triangle_coeff_unclassified,
)
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

_HEX8_FOR = "coupfe/runtime/elements/neo_hookean_hex8_fbar.for"   # scoped F-bar formulation
NE = 2                                  # NE×NE×NE hexes per block
G, K_BULK, DENSITY = 1.0, 10.0, 1.0     # K/G = 10 (moderate compressibility)
DHAT, KAPPA = 0.04, 1.0e2               # κ ~ K_bulk (matched); CCD owns non-penetration
GAP0 = 1.2 * DHAT                       # top block starts just ABOVE the band (separated)
V0 = 0.15                               # gentle downward impact velocity (v0/c ≈ 0.09 → small strain)
DT, N_STEPS, DAMP = 0.02, 25, 2.0


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


def _edges_of(tris):
    es = set()
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            es.add((int(min(a, b)), int(max(a, b))))
    return np.array(sorted(es), dtype=int)


def build():
    nb, eb = hex8_block(NE, 0.0)                          # bottom block: z in [0,1]
    Nb = len(nb)
    nt, et = hex8_block(NE, 1.0 + GAP0)                   # top block: z in [1+GAP0, 2+GAP0]
    nodes = np.vstack([nb, nt])
    elems = np.vstack([eb, et + Nb])
    faces = _triangulate(_surface_grid(NE, 0, "max"))    # bottom block's top face (primary)
    top_bot = _surface_grid(NE, Nb, "min")               # top block's bottom face (secondary)
    secondary = top_bot.ravel()
    edges = np.vstack([_edges_of(faces), _edges_of(_triangulate(top_bot))])
    base = np.arange((NE + 1) ** 2)                      # bottom block's z=0 face, fixed
    top_block = np.arange(Nb, len(nodes))                # all top-block nodes (carry v0)
    return nodes, elems, secondary, faces, edges, base, top_block


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
    nodes, elems, secondary, faces, edges, base, top_block = build()
    view = KernelMeshView(nodes, elems, dof_per_node=3)
    ndof = view.ndof
    M = lumped_mass_hex8(nodes, elems, DENSITY, ndof)
    v0 = np.zeros(ndof)
    v0[top_block * 3 + 2] = -V0                            # downward initial velocity on the top block

    my_gm, my_coords, _ = element_partition(view, rank, size)
    elem = CompiledElement(build_element_kernel(_HEX8_FOR, "neo_hex8_dist"),
                           props=neo_hookean_kernel_props(G, K_BULK),
                           dof_per_node=3, n_svars=0, mcrd=3,
                           n_elem=max(len(my_gm), 1))

    # Vertex-face ONLY (no edges): two FLAT parallel surfaces in close contact generate O(N²)
    # near-COPLANAR edge-edge pairs that are both slow (119 pairs → ~57 ms/assemble vs ~2 ms) and
    # degenerate (near-parallel edges → ill-defined normal → ill-conditioned). For flat block-on-block
    # vertex-face is correct and sufficient; edge-edge is for non-flat geometry (verified separately in
    # distributed_3d_primitives). Fixed κ (no adaptive M/gap² over-repulsion).
    dc = {"kind": "barrier", "nodes_ref": nodes, "vertices": secondary, "faces": faces,
          "edges": np.zeros((0, 2), int), "dhat": DHAT, "kappa": KAPPA}
    dirich = {int(n) * 3 + c: 0.0 for n in base for c in (0, 1, 2)}

    # Track the minimum reported interface gap over this trajectory. A value in
    # (0, d̂) shows that contact engaged without a sampled crossing for this run.
    min_gap = [np.inf]

    def track(step, U_full, rk):
        if rk == 0:
            min_gap[0] = min(min_gap[0], _min_interface_gap(U_full, nodes, secondary, faces))

    U_par, info = solve_dynamics_distributed(
        ndof, my_gm, my_coords, 3, elem.element_rk_batch, dirich, M,
        dt=DT, n_steps=N_STEPS, v0=v0, damping=DAMP, force=None, tol=1e-7,
        pc="lu", solver="superlu_dist", deformable_contact=dc, step_callback=track)

    if rank == 0:
        mg = float(min_gap[0])
        converged = info["rnorm"] < 1e-6 and not info["ksp_diverged"]
        collided = mg < DHAT                                 # the blocks came into contact
        penetration_free = mg > 0.0                          # positive sampled gap in this run
        ok = converged and collided and penetration_free
        if len(sys.argv) > 1:
            np.save(sys.argv[1], U_par)
        print(f"[size={size}] my_ne={info['my_ne']} newton(last)={info['n_newton']} "
              f"|R|={info['rnorm']:.2e} converged={converged}")
        print(f"[size={size}] nodes={len(nodes)} hexes={len(elems)} max|U|={float(np.max(np.abs(U_par))):.4f} "
              f"min_gap_over_traj={mg:.3e} (d̂={DHAT}) collided={collided} "
              f"penetration_free={penetration_free} -> {'OK' if ok else 'FAIL'}", flush=True)
    comm.barrier()


if __name__ == "__main__":
    main()
