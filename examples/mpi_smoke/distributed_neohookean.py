"""Distributed neo-Hookean smoke program with a serial comparison.

Each rank owns a block of elements and assembles a distributed PETSc system
from the compiled f2py neo-Hookean batch kernel. A load-stepped Newton solve is
then compared with the serial ``Model`` result. The comparison checks this
implementation path in the active PETSc/MPI environment; it is not a retained
release qualification or a physical benchmark.

    OMP_NUM_THREADS=1 mpirun -n 4 python examples/mpi_smoke/distributed_neohookean.py

Uniaxial stretch of a unit block: left edge clamped, right edge pulled +10% in x.
"""

from __future__ import annotations

import numpy as np
from petsc4py import PETSc

from coupfe import Model, NeoHookean
from coupfe.assembly.distributed import element_partition, solve_distributed
from coupfe.materials import _build
from coupfe.mesh import KernelMeshView
from coupfe.model import _structured_quad_mesh
from coupfe.runtime.compiled_element import CompiledElement

NX, NY = 8, 8
G, K, STRETCH, N_STEPS = 1.0, 10.0, 0.10, 4


def build_view():
    nodes, elems = _structured_quad_mesh(NX, NY, 1.0, 1.0)
    return KernelMeshView(nodes, elems, dof_per_node=2)


def serial_reference(view):
    """Independent oracle: the serial scipy-direct `Model` solve."""
    m = Model.from_view(view)
    m.material("block", NeoHookean(G, K))
    m.fix("left", x=0.0, y=0.0)
    m.prescribe("right", x=STRETCH)
    return m.solve(steps=N_STEPS).U


def main():
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    view = build_view()
    nodes = view.nodes
    tol = 1e-9
    left = np.nonzero(np.abs(nodes[:, 0] - nodes[:, 0].min()) < tol)[0]
    right = np.nonzero(np.abs(nodes[:, 0] - nodes[:, 0].max()) < tol)[0]

    def dirichlet_fn(frac):
        d = {}
        for n in left:
            d[int(n) * 2 + 0] = 0.0
            d[int(n) * 2 + 1] = 0.0
        for n in right:
            d[int(n) * 2 + 0] = STRETCH * frac
        return d

    my_gm, my_coords, ndof = element_partition(view, rank, size)
    mat = NeoHookean(G, K)
    elem = CompiledElement(_build(mat._for, mat._module), props=mat.props,
                           dof_per_node=2, n_svars=0, mcrd=2, n_elem=len(my_gm))

    U_par, info = solve_distributed(ndof, my_gm, my_coords, 2, elem.element_rk_batch,
                                    dirichlet_fn, n_steps=N_STEPS,
                                    pc="lu", solver="superlu_dist", tol=1e-9)

    if rank == 0:
        U_ser = serial_reference(view)
        err = float(np.max(np.abs(U_par - U_ser)))
        umax = float(np.max(np.abs(U_ser)))
        print(f"[size={size}] my_ne={info['my_ne']} owned={info['n_owned']} "
              f"ghost={info['n_ghost']}")
        print(f"[size={size}] max|U| = {umax:.4f}, "
              f"max|U_parallel - U_serial| = {err:.2e}  "
              f"-> {'OK' if err < 1e-7 else 'FAIL'}")
    comm.barrier()


if __name__ == "__main__":
    main()
