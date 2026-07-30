"""Distributed deformable contact — cross-rank ASSEMBLY, rank-independent. Run under mpirun.

The genuinely-new piece of distributed deformable contact: a contact pair couples a secondary
*vertex* (one body) with a primary *edge* (another body), and after partitioning those can live on
different ranks. With the contact **surface replicated** to every rank, each rank computes the
contact residual/tangent COO for the **vertices it owns** (by DOF ownership — so each pair is
computed exactly once) with **global** DOF indices, and PETSc's off-process ``ADD_VALUES`` routes
the edge-node contributions to whatever rank owns them. The serial ``DeformableContact2D`` is reused
verbatim per rank.

Gate (1-vs-N invariant): the gathered distributed residual ``R`` and a tangent mat-vec ``K·v`` equal
the serial assembly to machine precision, for any rank count. (Penalty contact — no CCD/dynamics; the
barrier + global CCD layer onto this next.)

    OMP_NUM_THREADS=1 mpirun -n 4 python examples/mpi_smoke/distributed_deformable_residual.py
"""

from __future__ import annotations

import numpy as np

from coupfe.operators.contact import DeformableContact2D


def _config():
    """Two 2D bodies in penalty contact: body B = a row of edges on y=0; body A = vertices just
    below (penetrating, g<0). A's vertices (high node ids) vs B's edges (low ids) → cross-rank."""
    nx = 6
    xb = np.linspace(0.0, 1.0, nx)
    B = np.stack([xb, np.zeros(nx)], 1)                    # primary nodes 0..nx-1
    xa = 0.5 * (xb[:-1] + xb[1:])                          # midpoints
    A = np.stack([xa, -0.02 * np.ones(len(xa))], 1)       # secondary nodes, penetrating below y=0
    X = np.vstack([B, A])
    edges = np.array([(i, i + 1) for i in range(nx - 1)], dtype=int)   # B edges
    secondary = np.arange(nx, len(X))                     # A nodes
    return X, edges, secondary


def main():
    from petsc4py import PETSc                            # petsc4py only — never mpi4py

    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    ADD = PETSc.InsertMode.ADD_VALUES
    INS = PETSc.InsertMode.INSERT_VALUES
    FWD = PETSc.ScatterMode.FORWARD
    dpn, k = 2, 1.0e3
    X, edges, secondary = _config()
    N = len(X); ndof = N * dpn
    U0 = np.zeros(ndof)                                   # reference config (already penetrating)

    # ---- serial truth (full surface) ----
    op_full = DeformableContact2D(X, secondary, edges, dof_per_node=dpn, k=k)
    Rs = op_full.residual(U0, None, 0, 0)
    R_serial = np.zeros(ndof)
    np.add.at(R_serial, Rs.gdofs, Rs.values)
    v = np.cos(np.arange(ndof) * 0.7)                    # arbitrary vector for the K·v check
    Ts = op_full.tangent(U0, None, 0, 0)
    Kv_serial = np.zeros(ndof)
    np.add.at(Kv_serial, Ts.rows, Ts.values * v[Ts.cols])  # (serial K)·v

    # ---- distributed assembly ----
    U = PETSc.Vec().createMPI(ndof, comm=comm)
    rs, re = U.getOwnershipRange()
    for g in range(rs, re):
        U.setValue(g, U0[g], INS)
    U.assemble()
    # replicate the surface (here the whole mesh is the contact surface): full U on every rank
    sc, seq = PETSc.Scatter.toAll(U)
    sc.scatter(U, seq, addv=INS, mode=FWD)
    U_full = np.asarray(seq.getArray()).copy()
    # each rank owns the secondary vertices whose first DOF it owns (partition → no double count)
    owned = np.array([s for s in secondary if rs <= s * dpn < re], dtype=int)
    op = DeformableContact2D(X, owned, edges, dof_per_node=dpn, k=k)   # serial op, owned verts only

    Rv = PETSc.Vec().createMPI(ndof, comm=comm); Rv.set(0.0)
    A = PETSc.Mat().createAIJ((ndof, ndof), comm=comm)
    A.setPreallocationNNZ(6 * dpn)
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)    # contact sparsity is dynamic
    R = op.residual(U_full, None, 0, 0)
    if R.gdofs.size:
        Rv.setValues(np.asarray(R.gdofs, PETSc.IntType), np.asarray(R.values, float), addv=ADD)
    T = op.tangent(U_full, None, 0, 0)
    for r_, c_, val in zip(T.rows, T.cols, T.values):       # off-process entries route automatically
        A.setValue(int(r_), int(c_), float(val), addv=ADD)
    Rv.assemble(); A.assemble()

    # gather distributed R and compute A·v, compare to serial
    def gather(vec):
        s2, q = PETSc.Scatter.toAll(vec)
        s2.scatter(vec, q, addv=INS, mode=FWD)
        out = np.asarray(q.getArray()).copy()
        s2.destroy(); q.destroy()
        return out

    R_dist = gather(Rv)
    vv = PETSc.Vec().createMPI(ndof, comm=comm)
    for g in range(rs, re):
        vv.setValue(g, v[g], INS)
    vv.assemble()
    Av = vv.duplicate()
    A.mult(vv, Av)
    Kv_dist = gather(Av)

    err_R = float(np.max(np.abs(R_dist - R_serial)))
    err_Kv = float(np.max(np.abs(Kv_dist - Kv_serial)))
    if rank == 0:
        ok = err_R < 1e-10 and err_Kv < 1e-9
        print(f"[size={size}] cross-rank deformable contact: max|R_dist-R_serial|={err_R:.2e} "
              f"max|Kv_dist-Kv_serial|={err_Kv:.2e} (nnz_contact={Ts.rows.size}) "
              f"-> {'OK' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
