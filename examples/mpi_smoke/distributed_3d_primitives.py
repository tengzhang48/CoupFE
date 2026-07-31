"""Distributed 3D deformable-contact primitives — cross-rank assembly + global CCD. Run under mpirun.

Verifies the genuinely-new distributed pieces of 3D deformable contact, in isolation from the solver
(robust — no dynamics, no degenerate no-bulk system): the real shared helper
``_DistDeformableContact3D`` assembles BOTH 3D primitives cross-rank — **vertex-face** (partitioned
by the secondary vertex) and **edge-edge** (partitioned by the first node of each pair's first edge,
via ``owns_edge_pair``) — with global dof indices, PETSc off-process ``ADD_VALUES`` routing the
off-rank stencil nodes, and a **global CCD** step bound (``Vec.min``).

At a fixed configuration with active vertex-face and edge-edge pairs, the
program compares the gathered residual ``R``, tangent mat-vec ``K·v``, and
global CCD bound with the serial full-surface implementation at the invoked
rank count. This is the 3D analog of ``distributed_deformable_barrier`` (2D);
no retained rank sweep ships here.

    OMP_NUM_THREADS=1 mpirun -n 4 python examples/mpi_smoke/distributed_3d_primitives.py
"""

from __future__ import annotations

import numpy as np

from coupfe.operators.contact3d import DeformableBarrierContact3D

DHAT, KAPPA = 0.05, 1.0e3


def _config():
    """A triangle + a vertex above it (vertex-face active), and two crossing edges within d̂
    (edge-edge active). Node ids span 0..7 so they partition across ranks. dU drives the pairs
    together so the CCD bound is < 1."""
    nodes = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],   # 0,1,2 triangle (z=0)
        [0.25, 0.25, 0.5 * DHAT],                            # 3 vertex above the triangle
        [0.0, 0.0, 0.30], [1.0, 0.0, 0.30],                  # 4,5 edge A (x-dir, z=0.30)
        [0.5, -0.5, 0.30 + 0.5 * DHAT], [0.5, 0.5, 0.30 + 0.5 * DHAT],  # 6,7 edge B (y-dir, above A)
    ], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    vertices = np.array([3], dtype=int)
    edges = np.array([[4, 5], [6, 7]], dtype=int)
    ndof = len(nodes) * 3
    U0 = np.zeros(ndof)
    dU = np.zeros(ndof)
    dU[3 * 3 + 2] = -0.1                                      # push the vertex down toward the face
    dU[6 * 3 + 2] = -0.1                                      # push edge B down toward edge A
    dU[7 * 3 + 2] = -0.1
    return nodes, faces, vertices, edges, U0, dU


def main():
    from petsc4py import PETSc                                # petsc4py only — never mpi4py
    from coupfe.assembly.distributed import _DistDeformableContact3D

    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    INS = PETSc.InsertMode.INSERT_VALUES
    FWD = PETSc.ScatterMode.FORWARD
    nodes, faces, vertices, edges, U0, dU = _config()
    ndof = len(nodes) * 3

    # ---- serial truth (full operator) ----
    op = DeformableBarrierContact3D(nodes, vertices, faces, edges, dof_per_node=3,
                                    dhat=DHAT, kappa=KAPPA)
    Rs = op.residual(U0, None, 0, 0)
    R_serial = np.zeros(ndof); np.add.at(R_serial, Rs.gdofs, Rs.values)
    v = np.cos(np.arange(ndof) * 0.7)
    Ts = op.tangent(U0, None, 0, 0)
    Kv_serial = np.zeros(ndof); np.add.at(Kv_serial, Ts.rows, Ts.values * v[Ts.cols])
    ccd_serial = float(op.max_step(U0, dU))
    n_active = Rs.gdofs.size // 3                             # active stencil-dofs / 3

    # ---- distributed via the shared helper ----
    U = PETSc.Vec().createMPI(ndof, comm=comm)
    rs, re = U.getOwnershipRange()
    for g in range(rs, re):
        U.setValue(g, U0[g], INS)
    U.assemble()
    dUv = U.duplicate()
    for g in range(rs, re):
        dUv.setValue(g, dU[g], INS)
    dUv.assemble()

    dc = _DistDeformableContact3D({"nodes_ref": nodes, "vertices": vertices, "faces": faces,
                                   "edges": edges, "dhat": DHAT, "kappa": KAPPA}, 3, ndof, U, comm)
    A = PETSc.Mat().createAIJ((ndof, ndof), comm=comm)
    A.setPreallocationNNZ(12 * 3)
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    R = PETSc.Vec().createMPI(ndof, comm=comm); R.set(0.0)
    dc.assemble_into(A, R, U)                                 # collective
    A.assemble(); R.assemble()
    ccd_dist = dc.ccd_alpha(U, dUv)                           # collective global min

    def gather(vec):
        sc, seq = PETSc.Scatter.toAll(vec)
        sc.scatter(vec, seq, addv=INS, mode=FWD)
        out = np.asarray(seq.getArray()).copy(); sc.destroy(); seq.destroy()
        return out

    R_dist = gather(R)
    vv = PETSc.Vec().createMPI(ndof, comm=comm)
    for g in range(rs, re):
        vv.setValue(g, v[g], INS)
    vv.assemble()
    Av = vv.duplicate(); A.mult(vv, Av); Kv_dist = gather(Av)

    err_R = float(np.max(np.abs(R_dist - R_serial)))
    err_Kv = float(np.max(np.abs(Kv_dist - Kv_serial)))
    err_ccd = abs(ccd_dist - ccd_serial)
    if rank == 0:
        ok = (err_R < 1e-10 and err_Kv < 1e-9 and err_ccd < 1e-12
              and ccd_serial < 1.0 and n_active >= 2)        # both vertex-face + edge-edge active
        print(f"[size={size}] 3D assembly: max|R_dist-R_serial|={err_R:.2e} "
              f"max|Kv_dist-Kv_serial|={err_Kv:.2e} (n_active_pairs≈{n_active})", flush=True)
        print(f"[size={size}] global CCD: dist={ccd_dist:.6f} serial={ccd_serial:.6f} "
              f"|diff|={err_ccd:.2e} (bit={ccd_serial < 1.0}) -> {'OK' if ok else 'FAIL'}",
              flush=True)


if __name__ == "__main__":
    main()
