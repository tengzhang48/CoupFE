"""Distributed rigid contact WITH friction — rank-independent. Run under mpirun.

A neo-Hookean block pressed onto a frictional rigid plane and dragged sideways, solved
across ranks: the bulk via the compiled element batch, the contact as a **node-local**
per-rank contribution (each contact node handled by the rank owning it; the Coulomb stick
state lives on that rank). The gate is **rank independence** — the gathered solution is
identical at any rank count.

    OMP_NUM_THREADS=1 mpirun -n 4 python examples/mpi_smoke/distributed_friction.py [out.npy]

With an output path, rank 0 saves the gathered U (the test compares 1-rank vs N-rank).

Default regime is a real mixed **stick/slip** drag, solved with the **reproducible**
``superlu_dist`` direct solver → rank-independent AND run-to-run repeatable to machine
precision (1e-16). IMPORTANT: switching the solver to MUMPS (``CF_SOLVER=mumps``) makes it
**non-reproducible** at >1 rank — MUMPS's parallel pivoting varies run-to-run, and the
ill-conditioned slip mode amplifies that ~1e-12 noise to ~1e-3. That is a *solver* property,
not a contact-physics one (MUMPS handles the asymmetric tangent correctly; superlu_dist on
the same problem is exact). See docs/lessons_learned.md. Override with CF_MU / CF_DRAG /
CF_PRESS / CF_NSTEPS / CF_SOLVER.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from petsc4py import PETSc

from coupfe import NeoHookean
from coupfe.assembly.distributed import element_partition, solve_distributed
from coupfe.materials import _build
from coupfe.mesh import KernelMeshView
from coupfe.model import _structured_quad_mesh
from coupfe.operators.contact import HalfSpace
from coupfe.runtime.compiled_element import CompiledElement

NX, NY = 8, 8
G, K = 1.0, 10.0
PRESS = float(os.environ.get("CF_PRESS", -0.03))
DRAG = float(os.environ.get("CF_DRAG", 0.05))       # real mixed stick/slip drag
MU = float(os.environ.get("CF_MU", 0.5))
N_STEPS = int(os.environ.get("CF_NSTEPS", 5))
SOLVER = os.environ.get("CF_SOLVER", "superlu_dist")   # reproducible parallel direct solver
PLANE = ([0.0, -0.005], [0.0, 1.0])


def main():
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    nodes, elems = _structured_quad_mesh(NX, NY, 1.0, 1.0)
    view = KernelMeshView(nodes, elems, dof_per_node=2)
    tol = 1e-9
    bottom = np.nonzero(np.abs(nodes[:, 1] - nodes[:, 1].min()) < tol)[0]
    top = np.nonzero(np.abs(nodes[:, 1] - nodes[:, 1].max()) < tol)[0]

    def dirichlet_fn(frac):
        d = {}
        for n in top:
            d[int(n) * 2 + 0] = DRAG * frac
            d[int(n) * 2 + 1] = PRESS * frac
        return d

    my_gm, my_coords, ndof = element_partition(view, rank, size)
    mat = NeoHookean(G, K)
    elem = CompiledElement(_build(mat._for, mat._module), props=mat.props,
                           dof_per_node=2, n_svars=0, mcrd=2, n_elem=len(my_gm))
    contact = {"nodes": bottom, "coords": nodes[bottom], "obstacle": HalfSpace(*PLANE),
               "k": 1.0e4, "mu": MU, "comps": (0, 1)}

    U, info = solve_distributed(ndof, my_gm, my_coords, 2, elem.element_rk_batch,
                                dirichlet_fn, n_steps=N_STEPS, pc="lu", solver=SOLVER,
                                tol=1e-10, contact=contact)

    if rank == 0:
        bx = U.reshape(-1, 2)[bottom, 0].mean()
        ok = info["rnorm"] < 1e-7 and bx < DRAG          # converged + friction holds it
        print(f"[size={size}] last-step newton={info['n_newton']} |R|={info['rnorm']:.2e}")
        print(f"[size={size}] mean bottom u_x = {bx:.4f} < drag {DRAG} (friction holds) "
              f"-> {'OK' if ok else 'FAIL'}")
        if len(sys.argv) > 1:
            np.save(sys.argv[1], U)
    comm.barrier()


if __name__ == "__main__":
    main()
