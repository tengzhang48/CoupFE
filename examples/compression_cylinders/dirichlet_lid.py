"""RESEARCH Dirichlet-driven deformable-lid study.

A stiff deformable strip supplies moving contact edges while its nodes are
driven downward over multiple increments. The default increment of prescribed
motion is smaller than the contact band; generic Dirichlet motion is not
automatically collision-bounded, so this is not an any-step-size guarantee.

The script checks a scoped no-gross-tunneling condition, a floor gap, and
finiteness on a small case. Floor and side walls remain fixed rigid
``HalfSpace`` obstacles.

It is a mechanism study rather than dense-pack validation or distributed
qualification.

    PYTHONPATH=. python examples/compression_cylinders/dirichlet_lid.py
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
G, K_BULK = 17.95, 89.7
G_LID, K_LID = G * 200, K_BULK * 200          # stiff lid strip
DHAT, KAPPA, FEPS = 0.6, 2.0e2, 5.0e-2


class ConstForce:
    def __init__(self, gd, f):
        self.gd = np.asarray(gd, int); self.f = np.asarray(f, float)
    def residual(self, U, s, t, dt): return Residual(self.gd, -self.f)
    def tangent(self, U, s, t, dt): return Tangent(np.array([], int), np.array([], int), np.array([]))
    def commit(self, U, s, t, dt): return s


def lid_strip(x0, x1, y, thick, n):
    """A thin horizontal strip of Quad4; returns nodes, elems, bottom-row nodes."""
    xs = np.linspace(x0, x1, n + 1)
    nodes = np.array([[x, y] for x in xs] + [[x, y + thick] for x in xs], float)
    elems = [[i, i + 1, n + 1 + i + 1, n + 1 + i] for i in range(n)]
    bottom = np.arange(n + 1)
    return nodes, np.array(elems, int), bottom


def main(n_cyl=6, lid_step=1.5):
    cyls_all, box, _ = extract_geometry()
    cyls = sorted(cyls_all, key=lambda c: c["center"][1])[:n_cyl]
    xL, xR, yB = box
    shrink = 0.92

    # --- assemble cylinders ---
    cnodes, celems, cbedges, cbnodes, cbody = [], [], [], [], []
    off = 0
    for ci, c in enumerate(cyls):
        nd, q = disk_mesh(c["center"], c["R"] * shrink, n=3)
        be = boundary_edges(q, nd) + off
        cnodes.append(nd); celems.extend((q + off).tolist())
        cbody.extend([ci] * len(nd))
        cbedges.extend(be.tolist()); cbnodes.extend(sorted(set(be.ravel().tolist())))
        off += len(nd)
    cnodes = np.vstack(cnodes); n_cnode = len(cnodes)
    cbnodes = np.array(sorted(set(cbnodes)), int); cbedges = np.array(cbedges, int)
    pack_top = cnodes[cbnodes, 1].max()

    # --- lid strip just above the pack ---
    ln, le, lbottom = lid_strip(xL, xR, pack_top + 0.8 * DHAT, 1.0, 12)
    lbottom = lbottom + n_cnode
    le = le + n_cnode
    nodes = np.vstack([cnodes, ln])
    body_id = np.concatenate([
        np.asarray(cbody, dtype=int),
        np.full(len(ln), len(cyls), dtype=int),
    ])
    ndof = len(nodes) * 2
    lid_nodes = np.arange(n_cnode, len(nodes))
    lid_bottom_edges = np.array([[lbottom[i], lbottom[i + 1]] for i in range(len(lbottom) - 1)], int)

    # --- two element groups: cylinders (soft F-bar) + lid (stiff F-bar) ---
    kernel = build_element_kernel(_FOR, "neo_fbar_q4_dl")
    view = KernelMeshView(nodes, np.array(celems + le.tolist(), int), dof_per_node=2)
    ec = CompiledElement(kernel, props=neo_hookean_kernel_props(G, K_BULK),
                         dof_per_node=2, n_svars=0, mcrd=2, n_elem=len(celems))
    el = CompiledElement(kernel, props=neo_hookean_kernel_props(G_LID, K_LID),
                         dof_per_node=2, n_svars=0, mcrd=2, n_elem=len(le))
    gc = ElementGroup(ec, nodes, np.array(celems, int), dof_per_node=2, comps=(0, 1))
    gl = ElementGroup(el, nodes, le, dof_per_node=2, comps=(0, 1))

    nodal = np.full(len(nodes), 1.0)
    M = np.repeat(nodal, 2)
    inertia = InertiaOperator(M, ndof, damping=3.0)
    grav = ConstForce(np.arange(2 * n_cnode), np.tile([0.0, -3.0], n_cnode))  # gravity on cylinders only

    # rigid fixed walls (floor + sides) — fixed obstacles don't tunnel
    walls = [HalfSpace([xL, yB], [0, 1]), HalfSpace([xL, yB], [1, 0]), HalfSpace([xR, yB], [-1, 0])]
    wall_ops = [RigidBarrierContact(nodes, cbnodes, w, dof_per_node=2, comps=(0, 1),
                                    dhat=DHAT, kappa=KAPPA, mass=nodal[cbnodes], mu=0.1, friction_eps=FEPS)
                for w in walls]
    # ONE deformable barrier: cylinder nodes vs (cylinder + lid-bottom) edges
    all_edges = np.vstack([cbedges, lid_bottom_edges])
    contact = DeformableBarrierContact2D(nodes, cbnodes, all_edges, dof_per_node=2, comps=(0, 1),
                                         dhat=DHAT, kappa=KAPPA, mass=nodal[cbnodes], mu=0.1,
                                         friction_eps=FEPS, body_id=body_id)

    def dirichlet(t):
        # The total lid displacement is ramped over N_STEPS prescribed updates.
        frac = t / (DT * N_STEPS)
        d = {}
        for n in lid_nodes:
            d[int(n) * 2 + 0] = 0.0
            d[int(n) * 2 + 1] = -lid_step * frac
        return d

    global DT, N_STEPS
    DT, N_STEPS = 0.02, 20
    ops = [gc, gl, inertia, grav, contact, *wall_ops]
    U, _ = solve_dynamics(ops, np.zeros(ndof), ndof, dirichlet, dt=DT, n_steps=N_STEPS)

    pos = nodes + U.reshape(len(nodes), 2)
    lid_y = pos[lid_bottom_edges.ravel(), 1].min()           # final lid bottom
    top_after = pos[cbnodes, 1].max()
    # Scoped check: the lid edge does not finish grossly below the pack top.
    no_tunnel = lid_y > top_after - DHAT - 1e-3
    floor_ok = pos[cbnodes, 1].min() - yB > -1e-3
    compaction = pack_top - top_after                    # informational (dense packs compact more)
    ok = no_tunnel and floor_ok and np.all(np.isfinite(U))
    print(f"Dirichlet lid: total travel={lid_step}, "
          f"per-step travel={lid_step/N_STEPS:.3f}, dhat={DHAT}")
    print(f"  lid bottom y={lid_y:.2f}, pack top={top_after:.2f} -> no_tunnel={no_tunnel} "
          "(scoped final-geometry check)")
    print(f"  floor gap check={floor_ok}; compaction={compaction:.2f} (informational)")
    print("OK" if ok else "FAIL")
    return ok


DT, N_STEPS = 0.02, 40

if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
