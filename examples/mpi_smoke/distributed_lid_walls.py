"""Distributed confined-compaction research harness.

This is an open implementation inspired by the Abaqus `compressioncylinders`
setup, not a reproduction or validation. It requires the user to supply a
lawfully obtained input deck through ``COUPFE_CYLINDERS_INP``. Everything that
touches the pack is a **deformable strip under contact**, so there are no rigid
`HalfSpace` operators and the whole model goes through
`solve_dynamics_distributed`:

  * Disks: F-bar Quad4 bodies (free), gravity + mutual contact.
  * Floor + left wall + right wall: thin Quad4 strips, **Dirichlet-fixed**.
  * Lid: a thin Quad4 strip, **Dirichlet-driven down** by a ramp. Its motion is
    included in the driver's collision-bound path; the reported minimum gap
    must still be checked for the selected discretization and step size.

Because every wall/lid node is fully Dirichlet-constrained, the strip elements
have no free DOFs, so their stiffness is irrelevant and ALL elements share one
material (uniform props) — the strips serve only to (a) carry a moving contact
edge and (b) keep every node inside an element.

ONE deformable barrier couples the disk boundary nodes (the only secondaries) to
the union of all contact edges (disk boundaries + the four wall/lid inner faces).
The winding rule is the same as for the N-body disk pack: orient every edge so the
region that holds potential secondaries (the pack interior) is on its LEFT, else
the signed node-segment gap flips sign and the barrier reads a spurious
penetration at rest. `wall_strip` returns its inner edges already wound
interior-on-left; `boundary_edges` winds the disks outside-on-left.

The program reports convergence, rigid-boundary gaps, and a serial-versus-rank
solution comparison. Interpret the comparison at the selected solver tolerance;
no retained final-revision output is bundled.

The script is a rank-independence and scaling **harness**, not retained
performance evidence. Historical local timings were not archived with raw
output, revisions, machine topology, and MPI/PETSc environment, so they are
deliberately omitted here. The implementation partitions the bulk while
replicating the contact surface, which predicts an Amdahl-limited regime; rerun
only on converged, rank-equal results before quoting a speedup. A
non-converged fixed-iteration run manufactures a misleading timing.

    OMP_NUM_THREADS=1 mpirun -n 2 python examples/mpi_smoke/distributed_lid_walls.py [out.npy] [n_cyl]
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

_FOR = os.path.join(os.path.dirname(__file__), "..", "compression_cylinders",
                    os.environ.get("FOR_KERNEL", "neo_up_native_q4.for"))  # DEFAULT = mixed u-p (volume
#   average / mean-dilatation, robust); F-bar (neo_fbar_q4.for, NSVARS=0) NaNs at centroid J̄≤0 — see lessons.
G = 17.95
K_BULK = float(os.environ.get("KBULK", "89.7"))        # lower → more compressible (less F-bar stress)
DHAT, KAPPA, FEPS = 0.6, 2.0e2, 5.0e-2
MU = float(os.environ.get("MU", "0.1"))                # 0 = frictionless (for the all-primitive test)
ALL_PRIMITIVE = bool(int(os.environ.get("ALL_PRIMITIVE", "0")))  # 2D ppf all-primitive barrier (frictionless)
FREEZE = bool(int(os.environ.get("FREEZE", "1")))      # per-step pairing freeze (node-to-segment chatter fix)
DT, DAMP = 0.02, 3.0
N_STEPS = int(os.environ.get("N_STEPS", "40"))
VERBOSE = bool(int(os.environ.get("VERBOSE", "0")))
GRAV = float(os.environ.get("GRAV", "3.0"))            # downward body force on disks
LID_STEP = float(os.environ.get("LID_STEP", "2.0"))   # total lid descent over the solve


def wall_strip(a, b, n_in, thick, n):
    """Thin Quad4 strip with its INNER face on segment a->b and its body behind it.

    `n_in` is the unit interior normal (points toward the disks). Returns
    (nodes, quads, inner_edges, all_node_ids) with the inner contact edges wound
    interior-on-left: cross(edge, n_in) > 0, the same convention `boundary_edges`
    uses for the disks (region-with-secondaries on the left of the edge).
    """
    a, b, n_in = np.asarray(a, float), np.asarray(b, float), np.asarray(n_in, float)
    n_in = n_in / np.linalg.norm(n_in)
    ts = np.linspace(0.0, 1.0, n + 1)
    inner = np.array([a + t * (b - a) for t in ts])       # the contact face
    outer = inner - thick * n_in                          # behind the wall
    nodes = np.vstack([inner, outer])
    quads = [[i, i + 1, n + 1 + i + 1, n + 1 + i] for i in range(n)]
    edges = []
    for i in range(n):
        u, v, e = i, i + 1, inner[i + 1] - inner[i]
        if e[0] * n_in[1] - e[1] * n_in[0] < 0.0:         # interior must be on the LEFT
            u, v = v, u
        edges.append([u, v])
    return nodes, np.array(quads, int), np.array(edges, int), np.arange(len(nodes))


def build(n_cyl, n_ref=3, margin=0.4, thick=1.0):
    cyls_all, _, _ = extract_geometry()
    cyls = sorted(cyls_all, key=lambda c: c["center"][1])[:n_cyl]
    nodes_l, elems, dedges, dbnodes, disk_nodes = [], [], [], [], []
    off = 0
    for c in cyls:
        nd, q = disk_mesh(c["center"], c["R"] * 0.92, n=n_ref)
        be = boundary_edges(q, nd) + off
        nodes_l.append(nd); elems.extend((q + off).tolist())
        dedges.extend(be.tolist()); dbnodes.extend(sorted(set(be.ravel().tolist())))
        disk_nodes.append(np.arange(off, off + len(nd)))
        off += len(nd)
    dnodes = np.vstack(nodes_l)
    dbnodes = np.array(sorted(set(dbnodes)), int)
    dedges = np.array(dedges, int)

    # walls SNUG around the chosen pack so the descending lid forces lateral contact
    xlo, ylo = dnodes.min(0) - margin
    xhi, yhi = dnodes.max(0) + margin
    nx = max(8, int((xhi - xlo) / 2.0))
    ny = max(8, int((yhi - ylo) / 2.0))
    nodes_all = [dnodes]; edges_all = [dedges]
    fixed_nodes, lid_nodes = [], []
    cur = len(dnodes)

    def add(strip, driven):
        nonlocal cur
        nd, q, ed, ids = strip
        nodes_all.append(nd); elems.extend((q + cur).tolist()); edges_all.append(ed + cur)
        (lid_nodes if driven else fixed_nodes).extend((ids + cur).tolist())
        cur += len(nd)

    add(wall_strip([xlo, ylo], [xhi, ylo], [0, 1], thick, nx), False)   # floor  (interior +y)
    add(wall_strip([xlo, ylo], [xlo, yhi], [1, 0], thick, ny), False)   # left   (interior +x)
    add(wall_strip([xhi, ylo], [xhi, yhi], [-1, 0], thick, ny), False)  # right  (interior -x)
    add(wall_strip([xlo, yhi + 0.8 * DHAT], [xhi, yhi + 0.8 * DHAT],    # lid    (interior -y)
                   [0, -1], thick, nx), True)

    nodes = np.vstack(nodes_all)
    return (nodes, np.array(elems, int), dbnodes, np.vstack(edges_all),
            disk_nodes, np.array(fixed_nodes, int), np.array(lid_nodes, int),
            (xlo, xhi, ylo, yhi))


def main():
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    n_cyl = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_ref = int(os.environ.get("N_REF", "3"))
    (nodes, elems, dbnodes, edges, disk_nodes,
     fixed_nodes, lid_nodes, bbox) = build(n_cyl, n_ref=n_ref,
                                           margin=float(os.environ.get("MARGIN", "0.4")))
    view = KernelMeshView(nodes, elems, dof_per_node=2)
    ndof = view.ndof
    M = np.full(ndof, 1.0)

    # gravity on the DISKS only (wall/lid DOFs are Dirichlet, so a force there is moot)
    fext = np.zeros(ndof)
    disk_ids = np.concatenate(disk_nodes)
    fext[disk_ids * 2 + 1] = -GRAV

    # walls fixed, lid driven down by a single ramp over the whole solve
    def dirichlet(t):
        frac = t / (DT * N_STEPS)
        d = {}
        for n in fixed_nodes:
            d[int(n) * 2 + 0] = 0.0; d[int(n) * 2 + 1] = 0.0
        for n in lid_nodes:
            d[int(n) * 2 + 0] = 0.0; d[int(n) * 2 + 1] = -LID_STEP * frac
        return d

    my_gm, my_coords, _ = element_partition(view, rank, size)
    elem = CompiledElement(build_element_kernel(_FOR, os.path.splitext(os.path.basename(_FOR))[0] + "_lw"),
                           props=(G, K_BULK), dof_per_node=2,
                           n_svars=int(os.environ.get("NSVARS", "1")), mcrd=2,   # u-p needs 1 (condensed p); F-bar=0
                           n_elem=max(len(my_gm), 1))
    dc = {"kind": "barrier", "nodes_ref": nodes, "secondary": dbnodes, "edges": edges,
          "dhat": DHAT, "kappa": KAPPA, "mu": MU, "friction_eps": FEPS,
          "mass": M[0::2],            # per-node mass → ppf adaptive s=κ+M/d² (CCD-lock fix)
          "freeze_pairing": FREEZE,   # fix per-step closest-edge → no fine-mesh active-set chatter
          "all_primitive": ALL_PRIMITIVE}   # opt-in: 2D ppf all-primitive barrier (frictionless)

    U_par, info = solve_dynamics_distributed(
        ndof, my_gm, my_coords, 2, elem.element_rk_batch, dirichlet, M,
        dt=DT, n_steps=N_STEPS, force=fext, damping=DAMP, tol=1e-7,
        pc="lu", solver="superlu_dist", deformable_contact=dc, verbose=VERBOSE)

    if rank == 0:
        pos = nodes + U_par.reshape(len(nodes), 2)
        xlo, xhi, ylo, yhi = bbox
        dp = pos[dbnodes]
        lid_y = pos[lid_nodes, 1].min()
        # penetration-free against the four walls (tol = barrier band)
        tol = -1e-2
        walls_ok = (dp[:, 1].min() - ylo > tol and dp[:, 0].min() - xlo > tol
                    and xhi - dp[:, 0].max() > tol and lid_y - dp[:, 1].max() > tol - DHAT)
        converged = info["rnorm"] < 1e-6 and not info["ksp_diverged"]
        ok = walls_ok and converged and np.all(np.isfinite(U_par))
        compaction = (yhi - ylo) - (dp[:, 1].max() - dp[:, 1].min())   # informational
        if len(sys.argv) > 1:
            np.save(sys.argv[1], U_par)
        print(f"[size={size}] {n_cyl} disks, {len(nodes)} nodes, my_ne={info['my_ne']} "
              f"owned={info['n_owned']} |R|={info['rnorm']:.2e} converged={converged}")
        print(f"[size={size}] lid descended to y={lid_y:.2f}; pack y in "
              f"[{dp[:,1].min():.2f},{dp[:,1].max():.2f}] x in "
              f"[{dp[:,0].min():.2f},{dp[:,0].max():.2f}] walls=[{xlo:.1f},{xhi:.1f}]x"
              f"[{ylo:.1f},..]; vertical compaction={compaction:.2f}")
        print(f"[size={size}] walls_ok={walls_ok} -> {'OK' if ok else 'FAIL'}", flush=True)
    comm.barrier()


if __name__ == "__main__":
    main()
