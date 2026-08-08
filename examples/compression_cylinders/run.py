"""RESEARCH confined-compression workflow inspired by an Abaqus example.

The script regenerates a user-supplied cylinder pack as F-bar Quad4 disks,
places them in a rigid box, and compacts them with a stepped lid using smoothed
friction and implicit dynamics. The adapted mesh, material, and solver do not
constitute an Abaqus reproduction.

    PYTHONPATH=. python examples/compression_cylinders/run.py [n_cyl]

The final check covers finiteness, compaction, and the reported rigid wall/lid
gaps. It is not a complete mutual-contact or physical-validation certificate.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from build_model import boundary_edges, disk_mesh, extract_geometry  # noqa: E402

from coupfe import (  # noqa: E402
    InertiaOperator,
    neo_hookean_kernel_props,
    solve_dynamics,
)
from coupfe.mesh import KernelMeshView  # noqa: E402
from coupfe.operators.base import Residual, Tangent  # noqa: E402
from coupfe.operators.contact import (  # noqa: E402
    DeformableBarrierContact2D, HalfSpace, RigidBarrierContact)
from coupfe.operators.element_group import ElementGroup  # noqa: E402
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel  # noqa: E402

_FOR = os.path.join(os.path.dirname(__file__), "neo_fbar_q4.for")


class ConstForce:
    def __init__(self, gd, f):
        self.gd = np.asarray(gd, int); self.f = np.asarray(f, float)

    def residual(self, U, s, t, dt):
        return Residual(self.gd, -self.f)

    def tangent(self, U, s, t, dt):
        return Tangent(np.array([], int), np.array([], int), np.array([]))

    def commit(self, U, s, t, dt):
        return s


def build_pack(cyls, *, n_disk=3, shrink=0.92):
    """Mesh each cylinder as an F-bar disk; assemble a global mesh.

    Returns dict with nodes, per-material element lists, boundary edges/nodes,
    per-node material id, and the cylinder index of each node.
    """
    all_nodes, rub_el, ste_el, bedges, bnodes, body_ids = [], [], [], [], [], []
    off = 0
    for ci, c in enumerate(cyls):
        nd, q = disk_mesh(c["center"], c["R"] * shrink, n=n_disk)
        be = boundary_edges(q, nd) + off
        all_nodes.append(nd)
        body_ids.extend([ci] * len(nd))
        (rub_el if c["material"] == "RUBBER" else ste_el).extend((q + off).tolist())
        bedges.extend(be.tolist())
        bnodes.extend(sorted(set(be.ravel().tolist())))
        off += len(nd)
    nodes = np.vstack(all_nodes)
    return dict(nodes=nodes, rub=np.array(rub_el, int), ste=np.array(ste_el, int),
                bedges=np.array(bedges, int), bnodes=np.array(sorted(set(bnodes)), int),
                body=np.asarray(body_ids, dtype=int))


def min_pair_gap(nodes, cyls, shrink):
    """Smallest surface-surface gap between regenerated disks (>0 ⇒ separated)."""
    C = np.array([c["center"] for c in cyls]); R = np.array([c["R"] for c in cyls]) * shrink
    g = np.inf
    for i in range(len(cyls)):
        d = np.linalg.norm(C[i + 1:] - C[i], axis=1) - (R[i] + R[i + 1:])
        if len(d):
            g = min(g, d.min())
    return g


def main(n_cyl=8):
    cyls_all, box, params = extract_geometry()
    # Use the n_cyl lowest disks as a bounded demonstration subset.
    cyls = sorted(cyls_all, key=lambda c: c["center"][1])[:n_cyl]
    xL, xR, yB = box
    shrink = 0.92
    gap0 = min_pair_gap(np.zeros((1, 2)), cyls, shrink)
    pack = build_pack(cyls, n_disk=3, shrink=shrink)
    nodes = pack["nodes"]; ndof = len(nodes) * 2

    kernel = build_element_kernel(_FOR, "neo_fbar_q4_cc")
    groups = []
    if len(pack["rub"]):
        er = CompiledElement(
            kernel,
            props=neo_hookean_kernel_props(
                params["G_rubber"], params["K_rubber"]
            ),
            dof_per_node=2,
            n_svars=0,
            mcrd=2,
            n_elem=len(pack["rub"]),
        )
        groups.append(ElementGroup(er, nodes, pack["rub"], dof_per_node=2, comps=(0, 1)))
    if len(pack["ste"]):
        es = CompiledElement(
            kernel,
            props=neo_hookean_kernel_props(
                params["G_rubber"] * 1500, params["K_rubber"] * 650
            ),
            dof_per_node=2,
            n_svars=0,
            mcrd=2,
            n_elem=len(pack["ste"]),
        )
        groups.append(ElementGroup(es, nodes, pack["ste"], dof_per_node=2, comps=(0, 1)))

    # Illustrative relative mass for relaxation; this is not a calibrated
    # physical density model.
    nodal = np.full(len(nodes), 1.0)
    M = np.repeat(nodal, 2)
    inertia = InertiaOperator(M, ndof, damping=4.0)
    # gentle body force per node (seats the pack without crushing the soft rubber;
    # absolute density is irrelevant under dynamic relaxation)
    grav = ConstForce(np.arange(ndof), np.tile([0.0, -3.0], len(nodes)))

    bnodes = pack["bnodes"]; mass_b = nodal[bnodes]
    DHAT, KAPPA, FEPS, MU = 0.6, 2.0e2, 5.0e-2, params["mu_fric"]
    walls = [HalfSpace([xL, yB, 0][:2], [0, 1]),       # floor y=yB
             HalfSpace([xL, yB], [1, 0]),               # left wall x=xL
             HalfSpace([xR, yB], [-1, 0])]              # right wall x=xR
    wall_ops = [RigidBarrierContact(nodes, bnodes, w, dof_per_node=2, comps=(0, 1),
                                    dhat=DHAT, kappa=KAPPA, mass=mass_b, mu=MU,
                                    friction_eps=FEPS) for w in walls]
    cyl_contact = DeformableBarrierContact2D(nodes, bnodes, pack["bedges"], dof_per_node=2,
                                             comps=(0, 1), dhat=DHAT, kappa=KAPPA,
                                             mass=mass_b, mu=MU, friction_eps=FEPS,
                                             body_id=pack["body"])

    base_ops = groups + [inertia, grav, cyl_contact, *wall_ops]
    print(f"{len(cyls)} disks ({len(pack['rub'])} rubber + {len(pack['ste'])} steel quads), "
          f"{len(nodes)} nodes; initial min pair gap = {gap0:.3f} (dhat={DHAT})")

    def pen(U):
        p = nodes + U.reshape(len(nodes), 2)
        return (float(p[bnodes, 1].min() - yB), float(p[bnodes, 0].min() - xL),
                float(xR - p[bnodes, 0].max()), float(p[bnodes, 1].max()))

    # --- seat under gravity ---
    U, _ = solve_dynamics(base_ops, np.zeros(ndof), ndof, lambda t: {}, dt=0.02, n_steps=25)
    _, _, _, top0 = pen(U)
    print(f"  seated: pack top y = {top0:.2f}")

    # --- lid compaction: lower a rigid lid HalfSpace onto the pack ---
    # The lid must descend in steps SMALLER than dhat (else it jumps past the
    # barrier band in one increment and the cubic barrier — active only in
    # [0,dhat) — produces no force, i.e. it tunnels through).
    lid_y0 = top0 + 0.8 * DHAT                # start just inside the band
    compress = 0.30 * (top0 - yB)             # target squeeze ~30%
    step = 0.4 * DHAT                          # < dhat per increment
    n_lid = max(6, int(np.ceil(compress / step)))
    min_pen = np.inf
    for k in range(1, n_lid + 1):
        lid_y = lid_y0 - compress * k / n_lid
        lid = RigidBarrierContact(nodes, bnodes, HalfSpace([xL, lid_y], [0, -1]),
                                  dof_per_node=2, comps=(0, 1), dhat=DHAT, kappa=KAPPA,
                                  mass=mass_b, mu=MU, friction_eps=FEPS)
        U, _ = solve_dynamics(base_ops + [lid], U, ndof, lambda t: {}, dt=0.02, n_steps=10)
        pf, pl, pr, top = pen(U)
        min_pen = min(min_pen, pf, pl, pr, lid_y - top)
        print(f"  lid y={lid_y:5.2f}: pack top={top:5.2f}  pen(floor/L/R)="
              f"{pf:+.2f}/{pl:+.2f}/{pr:+.2f}")
    _, _, _, topf = pen(U)
    compaction = (top0 - topf) / max(top0 - yB, 1e-9)
    rigid_gap_ok = min_pen > -1e-3
    ok = rigid_gap_ok and compaction > 0.10 and np.all(np.isfinite(U))
    print(f"  compaction: top {top0:.2f} -> {topf:.2f}  ({100*compaction:.0f}% of pack height)")
    print(f"  reported rigid wall/lid gap check: {rigid_gap_ok} (min gap {min_pen:+.3f})")
    print("OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    raise SystemExit(0 if main(n) else 1)
