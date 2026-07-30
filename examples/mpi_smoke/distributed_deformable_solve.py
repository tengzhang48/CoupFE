"""Distributed deformable-CONTACT solve: two elastic blocks, serial == N-rank. Run under mpirun.

A full finite-strain Newton solve across ranks WITH deformable-deformable contact: a top
neo-Hookean block is pushed down onto a bottom neo-Hookean block; the only thing transmitting
the load between them is node-to-segment penalty contact across the interface. Each rank owns a
block of elements (bulk, via the compiled f2py kernel) AND the contact secondaries whose DOFs it
owns; the contact surface is replicated each iteration and PETSc routes the cross-rank edge-node
contributions (``solve_distributed(..., deformable_contact=...)``).

Two gates:
  * **1-vs-N invariant** — gathered distributed U equals the serial solve for any rank count.
  * **independent serial oracle** — the serial truth is assembled by a *different* path: the
    operator-level ``solve_increments([ElementGroup, DeformableContact2D])`` (dense COO → scipy),
    not ``solve_distributed`` at one rank. So the test is not the distributed code grading itself.
  * **contact actually engaged** — the secondary nodes penetrate (g<0 absent contact) and the
    resolved penetration is small (held out by the penalty), i.e. the run is non-trivial.

    OMP_NUM_THREADS=1 mpirun -n 4 python examples/mpi_smoke/distributed_deformable_solve.py
"""

from __future__ import annotations

import numpy as np
from petsc4py import PETSc

from coupfe.assembly.assemble import solve_increments
from coupfe.assembly.distributed import element_partition, solve_distributed
from coupfe.materials import _build
from coupfe.materials import NeoHookean
from coupfe.mesh import KernelMeshView
from coupfe.model import _structured_quad_mesh
from coupfe.operators.contact import DeformableContact2D
from coupfe.operators.element_group import ElementGroup
from coupfe.runtime.compiled_element import CompiledElement

NX, NY = 6, 4
G, K_BULK = 1.0, 10.0
PUSH, KC, N_STEPS = 0.06, 1.0e4, 4         # top pushed down PUSH; penalty stiffness KC


def build():
    """Two stacked unit blocks with SEPARATE interface nodes (contact, not bonded).

    bottom = [0,1]×[0,1], top = [0,1]×[1,2]. Returns the combined mesh, the contact spec
    (secondary = top's bottom-row nodes, edges = bottom's top-row edges, left→right so the
    edge normal points +y and a secondary pushed below y=1 penetrates), and the BC node sets.
    """
    nb, eb = _structured_quad_mesh(NX, NY, 1.0, 1.0)
    Nb = len(nb)
    nt = nb + np.array([0.0, 1.0])              # shift up by 1
    nodes = np.vstack([nb, nt])
    elems = np.vstack([eb, eb + Nb])
    top_row_b = np.array([NY * (NX + 1) + i for i in range(NX + 1)], dtype=int)   # y≈1, bottom blk
    bot_row_t = np.array([Nb + i for i in range(NX + 1)], dtype=int)              # y≈1, top blk
    edges = np.array([(top_row_b[i], top_row_b[i + 1]) for i in range(NX)], dtype=int)
    secondary = bot_row_t
    base = np.arange(NX + 1)                                                      # y=0, fixed
    top_top = np.array([Nb + NY * (NX + 1) + i for i in range(NX + 1)], dtype=int)  # y=2, driven
    return nodes, elems, secondary, edges, base, top_top


def dirichlet_full(base, top_top):
    """Full-load Dirichlet (ramped by the drivers): base clamped, top edge pushed down PUSH,
    top edge x pinned to kill the top block's free lateral rigid mode (contact carries no x)."""
    d = {}
    for n in base:
        d[int(n) * 2 + 0] = 0.0
        d[int(n) * 2 + 1] = 0.0
    for n in top_top:
        d[int(n) * 2 + 0] = 0.0
        d[int(n) * 2 + 1] = -PUSH
    return d


def serial_reference(nodes, elems, secondary, edges, base, top_top, ndof):
    """Independent oracle: operator-level serial Newton (ElementGroup + DeformableContact2D)."""
    mat = NeoHookean(G, K_BULK)
    elem = CompiledElement(_build(mat._for, mat._module), props=mat.props,
                           dof_per_node=2, n_svars=0, mcrd=2, n_elem=len(elems))
    group = ElementGroup(elem, nodes, elems, dof_per_node=2)
    contact = DeformableContact2D(nodes, secondary, edges, dof_per_node=2, k=KC)
    U, _ = solve_increments([group, contact], np.zeros(ndof), ndof,
                            dirichlet_full(base, top_top), n_steps=N_STEPS)
    return U


def main():
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    nodes, elems, secondary, edges, base, top_top = build()
    view = KernelMeshView(nodes, elems, dof_per_node=2)
    ndof = view.ndof

    my_gm, my_coords, _ = element_partition(view, rank, size)
    mat = NeoHookean(G, K_BULK)
    elem = CompiledElement(_build(mat._for, mat._module), props=mat.props,
                           dof_per_node=2, n_svars=0, mcrd=2, n_elem=max(len(my_gm), 1))

    dc = {"nodes_ref": nodes, "secondary": secondary, "edges": edges, "k": KC}
    bc_full = dirichlet_full(base, top_top)

    def dirichlet_fn(frac):
        return {g: v * frac for g, v in bc_full.items()}

    # tol 1e-8 (not 1e-9): with KC≫bulk the penalty residual plateaus near the conditioning
    # floor (active-set re-freezing each iter chatters at the noise level); 1e-8 is converged
    # for penalty contact and the gathered U still matches the serial oracle to ~1e-10.
    U_par, info = solve_distributed(ndof, my_gm, my_coords, 2, elem.element_rk_batch,
                                    dirichlet_fn, n_steps=N_STEPS, pc="lu",
                                    solver="superlu_dist", tol=1e-8,
                                    deformable_contact=dc)

    if rank == 0:
        U_ser = serial_reference(nodes, elems, secondary, edges, base, top_top, ndof)
        err = float(np.max(np.abs(U_par - U_ser)))
        umax = float(np.max(np.abs(U_ser)))
        # contact engaged? the TRUE deformed gap (secondary vs its closest DEFORMED primary edge),
        # not absolute settling — both surfaces translate down together as the bottom block
        # compresses. The penalty admits a small interpenetration (g<0); KC holds it shallow.
        probe = DeformableContact2D(nodes, secondary, edges, dof_per_node=2, k=KC)
        active = probe._active(U_par)                       # (s, a, b, xi, g, nrm), g<0 = penetration
        max_pen = min((g for *_, g, _ in active), default=0.0)
        engaged = len(active) > 0 and max_pen > -1e-3       # in contact, penetration held shallow
        ok = err < 1e-7 and engaged
        print(f"[size={size}] my_ne={info['my_ne']} owned={info['n_owned']} "
              f"ghost={info['n_ghost']} newton={info['n_newton']} |R|={info['rnorm']:.2e}")
        print(f"[size={size}] max|U|={umax:.4f} max|U_par-U_ser|={err:.2e} "
              f"n_active={len(active)} max_pen={max_pen:.2e} engaged={engaged} "
              f"-> {'OK' if ok else 'FAIL'}", flush=True)
    comm.barrier()


if __name__ == "__main__":
    main()
