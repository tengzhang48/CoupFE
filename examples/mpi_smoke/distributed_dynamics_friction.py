"""Distributed dynamics, deformable barrier, and smoothed-friction smoke. Run under mpirun.

The last cross-rank contact piece: ppf smoothed friction on the distributed deformable barrier, under
dynamics. Friction rides the same machinery — `_DistDeformableContact` threads `mu` into the
per-rank `DeformableBarrierContact2D` (so each rank's owned secondaries get the friction
residual/tangent) and advances the step-start friction reference `_x0` in
`commit`. Because that state is path-dependent, saved results should be
compared across rank counts rather than assuming an identical trajectory.

Setup: two neo-Hookean blocks held in contact by gravity; the top block's top edge is dragged
sideways (`+x`) while its `y` is free. Friction at the interface resists the relative sliding of the
top block's bottom over the bottom block's top: with `μ>0` the block shears and the interface (its
bottom row) slides LESS than the frictionless block. (A horizontal *force* would slide the same-width
blocks off each other's overlap → loss of contact; a bounded prescribed shear keeps them overlapped
and penetration-free.)

The current run checks that friction reduces slip, the solver converges, and the
reported gap stays positive. An optional output path supports an external
same-revision rank comparison; no retained multi-rank record ships here.

    OMP_NUM_THREADS=1 mpirun -n 4 python examples/mpi_smoke/distributed_dynamics_friction.py
"""

from __future__ import annotations

import sys

import numpy as np
from petsc4py import PETSc

from coupfe.assembly.distributed import element_partition, solve_dynamics_distributed
from coupfe.materials import NeoHookean, _build
from coupfe.mesh import KernelMeshView
from coupfe.model import _structured_quad_mesh
from coupfe.operators.contact import DeformableBarrierContact2D
from coupfe.operators.inertia import lumped_mass
from coupfe.runtime.compiled_element import CompiledElement

NX, NY = 6, 4
G, K_BULK = 1.0, 10.0
DENSITY, GRAV = 1.0, 2.0
DHAT, KAPPA = 0.04, 2.0e3
GAP0 = 0.5 * DHAT
DT, N_STEPS, DAMP = 0.02, 40, 2.0
MU, SHEAR = 0.4, 0.08          # friction coefficient; total +x drive on the top edge


def build():
    nb, eb = _structured_quad_mesh(NX, NY, 1.0, 1.0)
    Nb = len(nb)
    nt = nb + np.array([0.0, 1.0 + GAP0])
    nodes = np.vstack([nb, nt])
    elems = np.vstack([eb, eb + Nb])
    top_row_b = np.array([NY * (NX + 1) + i for i in range(NX + 1)], dtype=int)
    bot_row_t = np.array([Nb + i for i in range(NX + 1)], dtype=int)
    edges = np.array([(top_row_b[i], top_row_b[i + 1]) for i in range(NX)], dtype=int)
    secondary = bot_row_t
    base = np.arange(NX + 1)                                                      # bottom edge, fixed
    top_top = np.array([Nb + NY * (NX + 1) + i for i in range(NX + 1)], dtype=int)  # driven edge
    return nodes, elems, secondary, edges, base, top_top


def main():
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    nodes, elems, secondary, edges, base, top_top = build()
    view = KernelMeshView(nodes, elems, dof_per_node=2)
    ndof = view.ndof
    M = lumped_mass(nodes, elems, DENSITY, dof_per_node=2)
    force = np.zeros(ndof)
    force[1::2] = -GRAV * M[1::2]                          # gravity on the y DOFs

    my_gm, my_coords, _ = element_partition(view, rank, size)
    mat = NeoHookean(G, K_BULK)
    elem = CompiledElement(_build(mat._for, mat._module), props=mat.props,
                           dof_per_node=2, n_svars=0, mcrd=2, n_elem=max(len(my_gm), 1))

    # base fixed (x,y); top edge dragged +x (ramped), y FREE → it settles under gravity onto contact
    def dirichlet(t):
        frac = t / (N_STEPS * DT)
        d = {int(n) * 2 + c: 0.0 for n in base for c in (0, 1)}
        for n in top_top:
            d[int(n) * 2 + 0] = SHEAR * frac
        return d

    def run(mu):
        dc = {"kind": "barrier", "nodes_ref": nodes, "secondary": secondary, "edges": edges,
              "dhat": DHAT, "kappa": KAPPA, "mu": mu}
        U, info = solve_dynamics_distributed(
            ndof, my_gm, my_coords, 2, elem.element_rk_batch, dirichlet, M,
            dt=DT, n_steps=N_STEPS, damping=DAMP, force=force, tol=1e-8,
            pc="lu", solver="superlu_dist", deformable_contact=dc)
        return U, info

    U_fric, info = run(MU)
    U_free, _ = run(0.0)

    if rank == 0:
        # interface slip = mean +x displacement magnitude of the secondary (top-block bottom) nodes
        slip_fric = float(np.abs(np.mean(U_fric[secondary * 2 + 0])))
        slip_free = float(np.abs(np.mean(U_free[secondary * 2 + 0])))
        # penetration-free (μ>0 run)
        probe = DeformableBarrierContact2D(nodes, secondary, edges, dof_per_node=2,
                                           dhat=DHAT, kappa=KAPPA)
        pos = probe._positions_all(U_fric)
        gaps = []
        for v in secondary:
            v = int(v)
            c = probe._closest_edge(U_fric, pos[v], v)
            if c is None:
                continue
            a, b = c
            x0, x1 = pos[a], pos[b]
            e = x1 - x0
            xi = min(1.0, max(0.0, float((pos[v] - x0) @ e) / float(e @ e)))
            nrm = np.array([-e[1], e[0]]) / np.sqrt(float(e @ e))
            gaps.append(float((pos[v] - (x0 + xi * e)) @ nrm))
        min_gap = min(gaps) if gaps else np.inf
        converged = info["rnorm"] < 1e-7 and not info["ksp_diverged"]
        held = slip_fric < 0.7 * slip_free        # friction reduces interface sliding
        penetration_free = min_gap > 0.0
        ok = converged and penetration_free and held and slip_free > 1e-3
        if len(sys.argv) > 1:
            np.save(sys.argv[1], U_fric)
        print(f"[size={size}] newton(last)={info['n_newton']} |R|={info['rnorm']:.2e} "
              f"converged={converged} min_gap={min_gap:.3e}")
        print(f"[size={size}] interface slip: μ={MU} -> {slip_fric:.4e}, μ=0 -> {slip_free:.4e} "
              f"(ratio {slip_fric/slip_free:.2f}) held={held} pen_free={penetration_free} "
              f"-> {'OK' if ok else 'FAIL'}", flush=True)
    comm.barrier()


if __name__ == "__main__":
    main()
