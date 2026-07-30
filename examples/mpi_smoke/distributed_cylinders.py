"""Distributed MANY-BODY contact: the cylinder pack under the cross-rank barrier.

The scale question for `examples/compression_cylinders` — does the distributed
deformable barrier handle MANY mutually-contacting bodies (not just two blocks)?
Here a subset of the Abaqus pack's disks (regenerated as F-bar Quad4 disks) is
given an inward velocity so they collide and compact; the load is carried only by
the cross-rank node-to-segment cubic barrier over the UNION of all disk
boundaries (broad-phase + 1-ring incident exclusion ⇒ convex-disk mutual
contact).  No walls/lid (those are not in the distributed path yet — see the
example README).  All-rubber (one material group) so `element_partition` is clean.

Gate: penetration-free (cross-disk gap > 0) + converged; the TEST adds the key
distributed check—rank independence (1-vs-N difference of saved `U`). The
historical pack results and timings are not retained release evidence. Re-run
with exact revisions, environment, raw logs, and a rank-1 reference before
quoting tolerances or time. `build_model.boundary_edges` supplies the
outside-on-left winding needed to avoid spurious rest-state penetration.

`P_TARGET` supplies sustained radial confinement so the barrier can become
engaged. A radial trap only compacts a roughly radial cluster; the lid/walls
script provides a different confined arrangement. Neither is an Abaqus
reproduction without the missing source/result provenance.

Note on scaling: the distributed design **partitions the bulk but replicates the
contact surface** to every rank, so strong-scaling only speeds up the bulk
fraction (Amdahl)—best at bulk-dominated sizes. Treat both scripts as rerunnable
harnesses until a converged, rank-equal, retained benchmark record exists.

    OMP_NUM_THREADS=1 mpirun -n 2 python examples/mpi_smoke/distributed_cylinders.py [out.npy] [n_cyl]
"""
from __future__ import annotations

import os
import sys

import numpy as np
from petsc4py import PETSc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "compression_cylinders"))
from build_model import boundary_edges, disk_mesh, extract_geometry  # noqa: E402

from coupfe.assembly.distributed import element_partition, solve_dynamics_distributed  # noqa: E402
from coupfe.mesh import KernelMeshView  # noqa: E402
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel  # noqa: E402

_FOR = os.path.join(os.path.dirname(__file__), "..", "compression_cylinders", "neo_fbar_q4.for")
G, K_BULK = 17.95, 89.7
DHAT, KAPPA = 0.6, 2.0e2
DT, N_STEPS, DAMP = 0.02, 80, 3.0
# Confinement is specified as the inward force on the OUTERMOST node, P_TARGET, and
# applied as f = -P_TARGET*(x-C)/R_pack (harmonic trap NORMALIZED by the pack radius
# R_pack=max|x-C|). Normalizing makes the load EXTENT-INDEPENDENT: an 8-disk and a
# 58-disk pack feel the same outer force, so the engaged equilibrium is consistent
# across sizes (essential for a fair strong-scaling sweep). An un-normalized trap
# f=-K*(x-C) over-crushes big packs (outer force grows with extent).
P_TARGET = float(os.environ.get("P_TARGET", "9.0"))


def build(n_cyl, shrink=0.92, n_ref=3):
    cyls_all, _, _ = extract_geometry()
    cyls = sorted(cyls_all, key=lambda c: c["center"][1])[:n_cyl]
    all_nodes, elems, bedges, bnodes, disk_nodes = [], [], [], [], []
    off = 0
    for c in cyls:
        nd, q = disk_mesh(c["center"], c["R"] * shrink, n=n_ref)
        be = boundary_edges(q, nd) + off
        all_nodes.append(nd); elems.extend((q + off).tolist())
        bedges.extend(be.tolist()); bnodes.extend(sorted(set(be.ravel().tolist())))
        disk_nodes.append(np.arange(off, off + len(nd)))
        off += len(nd)
    nodes = np.vstack(all_nodes)
    return (nodes, np.array(elems, int), np.array(sorted(set(bnodes)), int),
            np.array(bedges, int), disk_nodes)


def min_cross_disk_gap(pos, disk_nodes):
    """Smallest surface gap between DIFFERENT disks (circle approx of the deformed
    disks): for each disk a deformed centroid C and radius R; gap = |C_i-C_j|-R_i-R_j.
    <0 ⇒ real inter-disk penetration (excludes same-disk probe artifacts)."""
    C = np.array([pos[dn].mean(0) for dn in disk_nodes])
    R = np.array([np.linalg.norm(pos[dn] - pos[dn].mean(0), axis=1).max() for dn in disk_nodes])
    g = np.inf
    for i in range(len(disk_nodes)):
        for j in range(i + 1, len(disk_nodes)):
            g = min(g, float(np.linalg.norm(C[i] - C[j]) - R[i] - R[j]))
    return g


def main():
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    n_cyl = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    n_ref = int(os.environ.get("N_REF", "3"))
    nodes, elems, bnodes, bedges, disk_nodes = build(n_cyl, n_ref=n_ref)
    view = KernelMeshView(nodes, elems, dof_per_node=2)
    ndof = view.ndof
    M = np.full(ndof, 1.0)

    # ENGAGED compaction via a sustained radial CONFINEMENT force (a soft trap),
    # f = -k_trap*(node-centroid), evaluated in the reference config (constant, so
    # it is a clean external force the distributed driver applies every step).
    # Starting from rest, damped dynamic relaxation packs the disks INWARD until
    # the cubic barrier balances the trap -> an engaged equilibrium (gap < d̂),
    # not just a transient drift. (An initial-velocity-only push with damping
    # decays after ~v0/damping and stops SHORT of contact -- the disks never
    # engage; that is why the earlier pack converged but stayed separated.)
    # A LINEAR field is used (0 at the centroid): a *normalized* -speed*(d/|d|)
    # field is singular at the centroid -> opposing nodal velocities -> J<0 ->
    # divergence; THAT, not F-bar, was the original non-convergence.
    ctr = nodes.mean(0)
    d = nodes - ctr
    R_pack = float(np.linalg.norm(d, axis=1).max())
    # P_TARGET is the inward trap force on the outermost node; it is tuned PER CONFIG
    # (disk count + refinement). Engagement is mesh-dependent through the contact
    # discretization (a finer boundary spreads the barrier over more node-edge pairs
    # -> stiffer contact -> less compression at the same force), so there is no clean
    # body-force normalization that transfers P_TARGET across refinements -- tune it
    # for the specific config you run.
    k_trap = P_TARGET / R_pack                        # outer node feels exactly P_TARGET
    fext = np.zeros(ndof)
    fext[0::2] = -k_trap * d[:, 0]
    fext[1::2] = -k_trap * d[:, 1]
    v0 = None

    my_gm, my_coords, _ = element_partition(view, rank, size)
    elem = CompiledElement(build_element_kernel(_FOR, "neo_fbar_q4_dist"),
                           props=(G, K_BULK), dof_per_node=2, n_svars=0, mcrd=2,
                           n_elem=max(len(my_gm), 1))
    dc = {"kind": "barrier", "nodes_ref": nodes, "secondary": bnodes, "edges": bedges,
          "dhat": DHAT, "kappa": KAPPA, "mass": M[0::2]}   # ppf adaptive s=κ+M/d² capacity

    U_par, info = solve_dynamics_distributed(
        ndof, my_gm, my_coords, 2, elem.element_rk_batch, {}, M,
        dt=DT, n_steps=N_STEPS, v0=v0, force=fext, damping=DAMP, tol=1e-7,
        pc="lu", solver="superlu_dist", deformable_contact=dc)

    if rank == 0:
        pos = nodes + U_par.reshape(len(nodes), 2)
        min_gap = min_cross_disk_gap(pos, disk_nodes)
        penetration_free = min_gap > -1e-3
        engaged = min_gap < DHAT                         # informational: a disk pair in contact
        converged = info["rnorm"] < 1e-6 and not info["ksp_diverged"]
        ok = penetration_free and converged              # the gate; the TEST adds 1-vs-N
        if len(sys.argv) > 1:
            np.save(sys.argv[1], U_par)
        print(f"[size={size}] {n_cyl} disks, {len(nodes)} nodes, my_ne={info['my_ne']} "
              f"owned={info['n_owned']} |R|={info['rnorm']:.2e} converged={converged}")
        print(f"[size={size}] min cross-disk gap={min_gap:.3e} (d̂={DHAT}) "
              f"penetration_free={penetration_free} engaged={engaged} "
              f"-> {'OK' if ok else 'FAIL'}", flush=True)
    comm.barrier()


if __name__ == "__main__":
    main()
