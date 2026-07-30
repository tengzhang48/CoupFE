"""Distributed deformable-BARRIER primitives — cross-rank assembly + GLOBAL CCD. Run under mpirun.

The penetration-free (cubic-barrier) deformable contact adds one genuinely-new distributed
primitive over the penalty version: the Newton step bound is a point-edge CCD ``max_step`` that is
LOCAL to each rank's owned secondaries, but the limiting pair can live on ANY rank — so the
line-search step bound must be the GLOBAL MINIMUM across ranks (a collective reduction). This
verifies BOTH barrier primitives, at a fixed approaching config, exactly as ``solve_distributed``
computes them — decoupled from the (orthogonal, serial) barrier merit-function convergence:

  1. **cross-rank barrier assembly** — gathered R and a tangent mat-vec K·v equal the serial
     full-surface barrier assembly to machine precision (each rank assembles its OWNED secondaries
     with global dof indices; PETSc off-process ADD_VALUES routes the edge-node contributions).
  2. **global CCD bound** — the per-rank ``max_step`` reduced to the global minimum (petsc4py-only
     via a 1-entry-per-rank ``Vec.min()``, never mpi4py) equals the serial full-surface ``max_step``,
     and is < 1 (the CCD actually constrains this step).

Both are rank-independent (identical at 1/2/4 ranks). This is the distributed infrastructure the
production penetration-free solve rides on; full barrier *convergence* needs an energy-merit line
search (a serial contact-solver follow-up — the residual-norm line search stalls at the
node-to-segment projection flip even at one rank). See docs/dev/contact.md.

    OMP_NUM_THREADS=1 mpirun -n 4 python examples/mpi_smoke/distributed_deformable_barrier.py
"""

from __future__ import annotations

import numpy as np

from coupfe.operators.contact import DeformableBarrierContact2D

DHAT, KAPPA = 0.05, 1.0e2


def _config():
    """Body B = a row of edges on y=0 (nodes 0..nx-1); body A = vertices at y=+0.5·d̂ (gap inside
    the band, NOT penetrating). The CCD probe direction dU drives A down and B up so the secondaries
    would cross the edges within this step → the step bound is < 1."""
    nx = 6
    xb = np.linspace(0.0, 1.0, nx)
    B = np.stack([xb, np.zeros(nx)], 1)
    xa = 0.5 * (xb[:-1] + xb[1:])
    A = np.stack([xa, 0.5 * DHAT * np.ones(len(xa))], 1)        # gap = 0.5·d̂ > 0 (penetration-free)
    X = np.vstack([B, A])
    edges = np.array([(i, i + 1) for i in range(nx - 1)], dtype=int)
    secondary = np.arange(nx, len(X))
    U0 = np.zeros(X.size)
    dU = np.zeros(X.size)
    dU[secondary * 2 + 1] = -0.1                                # push A down (toward the edges)
    for n in range(nx):
        dU[n * 2 + 1] = +0.02                                   # push B up (relative approach)
    return X, edges, secondary, U0, dU


def main():
    from petsc4py import PETSc                                  # petsc4py only — never mpi4py

    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    ADD = PETSc.InsertMode.ADD_VALUES
    INS = PETSc.InsertMode.INSERT_VALUES
    FWD = PETSc.ScatterMode.FORWARD
    dpn = 2
    X, edges, secondary, U0, dU = _config()
    N = len(X); ndof = N * dpn

    def make(sec):
        return DeformableBarrierContact2D(X, sec, edges, dof_per_node=dpn, dhat=DHAT, kappa=KAPPA)

    # ---- serial truth (full surface) ----
    op_full = make(secondary)
    Rs = op_full.residual(U0, None, 0, 0)
    R_serial = np.zeros(ndof); np.add.at(R_serial, Rs.gdofs, Rs.values)
    v = np.cos(np.arange(ndof) * 0.7)
    Ts = op_full.tangent(U0, None, 0, 0)
    Kv_serial = np.zeros(ndof); np.add.at(Kv_serial, Ts.rows, Ts.values * v[Ts.cols])
    ccd_serial = float(op_full.max_step(U0, dU))

    # ---- distributed (each rank: its OWNED secondaries) ----
    U = PETSc.Vec().createMPI(ndof, comm=comm)
    rs, re = U.getOwnershipRange()
    for g in range(rs, re):
        U.setValue(g, U0[g], INS)
    U.assemble()
    owned = np.array([s for s in secondary if rs <= s * dpn < re], dtype=int)
    op = make(owned)

    Rv = PETSc.Vec().createMPI(ndof, comm=comm); Rv.set(0.0)
    A = PETSc.Mat().createAIJ((ndof, ndof), comm=comm)
    A.setPreallocationNNZ(6 * dpn)
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    R = op.residual(U0, None, 0, 0)
    if R.gdofs.size:
        Rv.setValues(np.asarray(R.gdofs, PETSc.IntType), np.asarray(R.values, float), addv=ADD)
    T = op.tangent(U0, None, 0, 0)
    for r_, c_, val in zip(T.rows, T.cols, T.values):
        A.setValue(int(r_), int(c_), float(val), addv=ADD)
    Rv.assemble(); A.assemble()

    # global CCD: this rank's max_step over its owned secondaries, reduced to the global min
    a_loc = float(op.max_step(U0, dU))                         # 1.0 if this rank owns none active
    red = PETSc.Vec().createMPI((1, size), comm=comm)
    red.setValue(rank, a_loc); red.assemble()
    _, ccd_dist = red.min(); red.destroy()

    def gather(vec):
        s2, q = PETSc.Scatter.toAll(vec)
        s2.scatter(vec, q, addv=INS, mode=FWD)
        out = np.asarray(q.getArray()).copy(); s2.destroy(); q.destroy()
        return out

    R_dist = gather(Rv)
    vv = PETSc.Vec().createMPI(ndof, comm=comm)
    for g in range(rs, re):
        vv.setValue(g, v[g], INS)
    vv.assemble()
    Av = vv.duplicate(); A.mult(vv, Av); Kv_dist = gather(Av)

    err_R = float(np.max(np.abs(R_dist - R_serial)))
    err_Kv = float(np.max(np.abs(Kv_dist - Kv_serial)))
    err_ccd = abs(ccd_dist - ccd_serial)
    if rank == 0:
        ok = err_R < 1e-10 and err_Kv < 1e-9 and err_ccd < 1e-12 and ccd_serial < 1.0
        print(f"[size={size}] barrier assembly: max|R_dist-R_serial|={err_R:.2e} "
              f"max|Kv_dist-Kv_serial|={err_Kv:.2e}", flush=True)
        print(f"[size={size}] global CCD: dist={ccd_dist:.6f} serial={ccd_serial:.6f} "
              f"|diff|={err_ccd:.2e} (bit={ccd_serial < 1.0}) -> {'OK' if ok else 'FAIL'}",
              flush=True)


if __name__ == "__main__":
    main()
