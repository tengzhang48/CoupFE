"""Memory-local distributed SOLVE (M3b core): serial == N-rank. Run under mpirun.

Each rank generates ONLY its block of cell-rows (no global mesh is ever built),
assembles them into a distributed PETSc matrix, and a PETSc KSP solves the global
system. Gathered to rank 0 the solution must equal a serial reference — the 1-vs-N
invariant. This is the memory-local distributed solve the M3 decomposition was built
for; petsc4py only.

    mpirun -n 4 python examples/mpi_smoke/distributed_solve.py

Scope: a structured Q1 Laplace problem (the regular-mesh target). Wiring the operator
contract + the f2py element + nonlinear Newton onto this same pattern is the remaining
step, and reuses the lab's proven solve_steps_mpi_local.
"""

from __future__ import annotations

import numpy as np
from petsc4py import PETSc

NX, NY = 16, 12                                    # cells; (NX+1)*(NY+1) nodes
# Unit-square bilinear Q1 Laplacian element stiffness (CCW node order).
KE = (1.0 / 6.0) * np.array([[4., -1., -2., -1.], [-1., 4., -1., -2.],
                             [-2., -1., 4., -1.], [-1., -2., -1., 4.]])


def nid(i, j):
    return j * (NX + 1) + i


def _serial_reference(bc):
    """Reduced-system serial solve (the independent oracle)."""
    nn = (NX + 1) * (NY + 1)
    A = np.zeros((nn, nn))
    for j in range(NY):
        for i in range(NX):
            g = [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
            A[np.ix_(g, g)] += KE
    con = np.array(sorted(bc))
    uc = np.array([bc[c] for c in con])
    free = np.array([n for n in range(nn) if n not in bc])
    uf = np.linalg.solve(A[np.ix_(free, free)], -A[np.ix_(free, con)] @ uc)
    u = np.zeros(nn)
    u[free], u[con] = uf, uc
    return u


def main():
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    nn = (NX + 1) * (NY + 1)
    bc = {nid(0, j): 0.0 for j in range(NY + 1)}    # left edge u=0
    bc.update({nid(NX, j): 1.0 for j in range(NY + 1)})  # right edge u=1

    A = PETSc.Mat().createAIJ((nn, nn), comm=comm)
    A.setUp()
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    my_rows = np.array_split(np.arange(NY), size)[rank]   # owned cell-rows only
    for j in my_rows:                                     # MEMORY-LOCAL: only these cells
        for i in range(NX):
            g = np.array([nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)],
                         dtype=PETSc.IntType)
            A.setValues(g, g, KE, addv=PETSc.InsertMode.ADD_VALUES)
    A.assemble()

    x = A.createVecRight()
    x.set(0.0)
    con = np.array(sorted(bc), dtype=PETSc.IntType)
    x.setValues(con, np.array([bc[int(c)] for c in con]),
                addv=PETSc.InsertMode.INSERT_VALUES)
    x.assemble()
    b = A.createVecRight()
    b.set(0.0)
    A.zeroRowsColumns(con, 1.0, x, b)                    # symmetric Dirichlet

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("jacobi")
    ksp.setTolerances(rtol=1e-12)
    u = A.createVecRight()
    ksp.solve(b, u)

    scat, useq = PETSc.Scatter.toZero(u)
    scat.scatter(u, useq, mode=PETSc.ScatterMode.FORWARD)
    if rank == 0:
        u_par = useq.getArray().copy()
        u_ser = _serial_reference(bc)
        err = float(np.max(np.abs(u_par - u_ser)))
        print(f"[size={size}] {ksp.getIterationNumber()} CG its, "
              f"max|u_parallel - u_serial| = {err:.2e}  "
              f"-> {'OK' if err < 1e-7 else 'FAIL'}")
    comm.barrier()


if __name__ == "__main__":
    main()
