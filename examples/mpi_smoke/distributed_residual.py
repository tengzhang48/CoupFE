"""Real-MPI distributed-residual example with a serial reference.

Each rank assembles ONLY its owned elements (from the same deterministic partition)
into a distributed PETSc vector; PETSc sums the off-process contributions on assembly.
At the invoked rank count, rank 0 compares the gathered vector with the serial
assembly. No retained multi-rank qualification record ships with the release.

    mpirun -n 4 python examples/mpi_smoke/distributed_residual.py

This is real distributed *assembly* (mesh still replicated per rank; the memory-local
owned/ghost storage is M3b). petsc4py only — no mpi4py import.
"""

from __future__ import annotations

import numpy as np
from petsc4py import PETSc

from coupfe.mesh import KernelMeshView
from coupfe.mesh.distribute import partition_elements

_KQ = np.array([[3., -1., -1., -1.], [-1., 3., -1., -1.],
                [-1., -1., 3., -1.], [-1., -1., -1., 3.]])


def _mesh(nx=8, ny=6):
    xs, ys = np.linspace(0, 2, nx + 1), np.linspace(0, 1, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs])
    elems = [[j * (nx + 1) + i, j * (nx + 1) + i + 1,
              (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i]
             for j in range(ny) for i in range(nx)]
    return KernelMeshView(nodes, np.array(elems, dtype=int), dof_per_node=1)


def main():
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()

    view = _mesh()
    U = np.random.default_rng(0).standard_normal(view.n_node)   # identical on all ranks
    parts = partition_elements(view, size)

    vec = PETSc.Vec().createMPI(view.n_node, comm=comm)
    vec.set(0.0)
    for e in np.where(parts == rank)[0]:                        # owned elements only
        ids = view.elems[e].astype(PETSc.IntType)
        vec.setValues(ids, _KQ @ U[view.elems[e]],
                      addv=PETSc.InsertMode.ADD_VALUES)
    vec.assemble()                                             # PETSc sums off-process

    scat, seq = PETSc.Scatter.toZero(vec)
    scat.scatter(vec, seq, mode=PETSc.ScatterMode.FORWARD)
    if rank == 0:
        R_par = seq.getArray().copy()
        R_ser = np.zeros(view.n_node)
        for e in range(view.n_elem):
            ids = view.elems[e]
            R_ser[ids] += _KQ @ U[ids]
        err = float(np.max(np.abs(R_par - R_ser)))
        owned = int(np.sum(parts == rank))
        print(f"[size={size}] rank0 owns {owned}/{view.n_elem} elems, "
              f"max|R_parallel - R_serial| = {err:.2e}  "
              f"-> {'OK' if err < 1e-10 else 'FAIL'}")
    comm.barrier()


if __name__ == "__main__":
    main()
