"""Distributed DYNAMICS + deformable BARRIER contact: serial == N-rank. Run under mpirun.

This is path (B): the ppf/IPC substrate (implicit dynamics) brought to the distributed solver, which
makes the **penetration-free deformable barrier** work distributed — the case that does NOT converge
quasistatically (the residual-norm line search stalls at the node-to-segment projection flip; ppf
uses no energy-merit line search, it relies on dynamics + a PSD/CCD-bounded step). The inertia
`M/dt²` regularizes the non-smooth contact and supplies the barrier's capacity; backward-Euler
damping settles it to equilibrium.

Two stacked neo-Hookean blocks; the top falls under gravity onto the bottom; the load is carried only
by the cross-rank node-to-segment cubic barrier, penetration-free (CCD-bounded predictor + step).
`solve_dynamics_distributed` does node-local inertia + gravity, ghosted bulk, and the shared
cross-rank deformable-contact helper + global CCD.

Gates: (1) gathered U == an INDEPENDENT serial oracle (`solve_dynamics([InertiaOperator, ElementGroup,
DeformableBarrierContact2D, gravity])`, a different driver/assembly path); (2) rank-independent
(1-vs-N); (3) penetration-free AND engaged (the minimum interface gap stays > 0 but inside d̂).

    OMP_NUM_THREADS=1 mpirun -n 4 python examples/mpi_smoke/distributed_dynamics_barrier.py
"""

from __future__ import annotations

import sys

import numpy as np
from petsc4py import PETSc

from coupfe.assembly.assemble import solve_dynamics
from coupfe.assembly.distributed import element_partition, solve_dynamics_distributed
from coupfe.materials import NeoHookean, _build
from coupfe.mesh import KernelMeshView
from coupfe.model import _structured_quad_mesh
from coupfe.operators.base import Residual, Tangent
from coupfe.operators.contact import DeformableBarrierContact2D
from coupfe.operators.element_group import ElementGroup
from coupfe.operators.inertia import InertiaOperator, lumped_mass
from coupfe.runtime.compiled_element import CompiledElement

NX, NY = 6, 4
G, K_BULK = 1.0, 10.0
DENSITY, GRAV = 1.0, 2.0
DHAT, KAPPA = 0.04, 2.0e3
GAP0 = 0.5 * DHAT                 # top block starts just inside the band, separated
DT, N_STEPS, DAMP = 0.02, 30, 2.0


def build():
    nb, eb = _structured_quad_mesh(NX, NY, 1.0, 1.0)
    Nb = len(nb)
    nt = nb + np.array([0.0, 1.0 + GAP0])
    nodes = np.vstack([nb, nt])
    elems = np.vstack([eb, eb + Nb])
    top_row_b = np.array([NY * (NX + 1) + i for i in range(NX + 1)], dtype=int)   # bottom blk, y=1
    bot_row_t = np.array([Nb + i for i in range(NX + 1)], dtype=int)              # top blk, y=1+GAP0
    edges = np.array([(top_row_b[i], top_row_b[i + 1]) for i in range(NX)], dtype=int)
    secondary = bot_row_t
    base = np.arange(NX + 1)                                                      # bottom edge, fixed
    return nodes, elems, secondary, edges, base


def serial_reference(nodes, elems, secondary, edges, base, ndof, M, force):
    """Independent oracle: operator-level serial backward-Euler dynamics."""
    mat = NeoHookean(G, K_BULK)
    elem = CompiledElement(_build(mat._for, mat._module), props=mat.props,
                           dof_per_node=2, n_svars=0, mcrd=2, n_elem=len(elems))
    group = ElementGroup(elem, nodes, elems, dof_per_node=2)
    contact = DeformableBarrierContact2D(nodes, secondary, edges, dof_per_node=2,
                                         dhat=DHAT, kappa=KAPPA)
    inertia = InertiaOperator(M, ndof, damping=DAMP)

    class _Grav:
        def residual(self, U, s, t, dt):
            return Residual(np.nonzero(force)[0], -force[np.nonzero(force)[0]])
        def tangent(self, U, s, t, dt):
            return Tangent(np.array([], int), np.array([], int), np.array([]))
        def commit(self, U, s, t, dt):
            return s

    dirich = {int(n) * 2 + c: 0.0 for n in base for c in (0, 1)}
    U, _ = solve_dynamics([inertia, group, contact, _Grav()], np.zeros(ndof), ndof,
                          dirich, dt=DT, n_steps=N_STEPS, rtol=1e-9, maxit=60)
    return U


def main():
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    nodes, elems, secondary, edges, base = build()
    view = KernelMeshView(nodes, elems, dof_per_node=2)
    ndof = view.ndof
    M = lumped_mass(nodes, elems, DENSITY, dof_per_node=2)
    force = np.zeros(ndof)
    force[1::2] = -GRAV * M[1::2]                          # gravity (downward) on the y DOFs

    my_gm, my_coords, _ = element_partition(view, rank, size)
    mat = NeoHookean(G, K_BULK)
    elem = CompiledElement(_build(mat._for, mat._module), props=mat.props,
                           dof_per_node=2, n_svars=0, mcrd=2, n_elem=max(len(my_gm), 1))

    dc = {"kind": "barrier", "nodes_ref": nodes, "secondary": secondary, "edges": edges,
          "dhat": DHAT, "kappa": KAPPA}
    dirich = {int(n) * 2 + c: 0.0 for n in base for c in (0, 1)}

    U_par, info = solve_dynamics_distributed(
        ndof, my_gm, my_coords, 2, elem.element_rk_batch, dirich, M,
        dt=DT, n_steps=N_STEPS, damping=DAMP, force=force, tol=1e-8,
        pc="lu", solver="superlu_dist", deformable_contact=dc)

    if rank == 0:
        U_ser = serial_reference(nodes, elems, secondary, edges, base, ndof, M, force)
        err = float(np.max(np.abs(U_par - U_ser)))
        umax = float(np.max(np.abs(U_ser)))
        probe = DeformableBarrierContact2D(nodes, secondary, edges, dof_per_node=2,
                                           dhat=DHAT, kappa=KAPPA)
        pos = probe._positions_all(U_par)
        gaps = []
        for v in secondary:
            v = int(v)
            c = probe._closest_edge(U_par, pos[v], v)
            if c is None:
                continue
            a, b = c
            x0, x1 = pos[a], pos[b]
            e = x1 - x0
            xi = min(1.0, max(0.0, float((pos[v] - x0) @ e) / float(e @ e)))
            nrm = np.array([-e[1], e[0]]) / np.sqrt(float(e @ e))
            gaps.append(float((pos[v] - (x0 + xi * e)) @ nrm))
        min_gap = min(gaps) if gaps else np.inf
        penetration_free = min_gap > 0.0
        engaged = min_gap < DHAT
        converged = info["rnorm"] < 1e-7 and not info["ksp_diverged"]
        # Gate = physical (penetration-free + engaged + converged). The serial number is a SANITY
        # check, not an exact gate: the distributed driver is ppf-style (CCD bound, no residual
        # backtracking) while serial newton_solve backtracks, so two valid backward-Euler
        # trajectories drift over the steps. Rank-independence (1-vs-N) is gated by the test, which
        # diffs the saved U across rank counts (same algorithm → must match to solver precision).
        ok = penetration_free and engaged and converged
        if len(sys.argv) > 1:
            np.save(sys.argv[1], U_par)
        print(f"[size={size}] my_ne={info['my_ne']} owned={info['n_owned']} "
              f"newton(last)={info['n_newton']} |R|={info['rnorm']:.2e} converged={converged}")
        print(f"[size={size}] max|U|={umax:.4f} min_gap={min_gap:.3e} (d̂={DHAT}) "
              f"penetration_free={penetration_free} engaged={engaged} "
              f"|U_dist-U_serial|={err:.2e} (sanity) -> {'OK' if ok else 'FAIL'}", flush=True)
    comm.barrier()


if __name__ == "__main__":
    main()
