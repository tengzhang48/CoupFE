"""CoupFE side of a cross-check against the ppf-contact-solver (ZOZO, Apache-2.0).

A soft Hex8 box rests on a **rigid floor** (`HalfSpace` — the analog of ppf's
`scene.add.invisible.wall`) under **gravity tilted by θ** (equivalent to a slope
of angle θ).  ppf/IPC **smoothed friction** on the contact resists tangential
sliding.  Run from the repo root::

    PYTHONPATH=. python examples/contact_vs_ppf/coupfe_box_on_floor.py

This is a **qualitative** cross-check (see `README.md`): ppf is single-precision
GPU with a stiffer `snhk` box and g=9.8; CoupFE here is a soft G=1 box with
g=0.4 selected for this scoped setup, so slide *distances* are not comparable.
This script checks only the following behaviors in its stated configurations:

  1. a positive reported gap, conditional on the configured CCD, solver, and
     time-step assumptions;
  2. friction **holds** a box below the slip threshold (interface slip ≪ the
     frictionless slip);
  3. **frictionless slides**.

Self-reports ``OK`` / ``FAIL``.  The observable is the **contact-interface slip**
(mean x-displacement of the bottom face nodes) — not the box centroid, which a
soft box contaminates with elastic shear.
"""
from __future__ import annotations

import numpy as np

from coupfe import InertiaOperator, solve_dynamics
from coupfe.mesh import KernelMeshView
from coupfe.operators.base import Residual, Tangent
from coupfe.operators.contact import HalfSpace, RigidBarrierContact
from coupfe.operators.element_group import ElementGroup
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

_HEX8_FOR = "coupfe/runtime/elements/neo_hookean_hex8_fbar.for"
G, K_BULK, DENSITY = 1.0, 10.0, 1.0
DHAT, KAPPA = 0.04, 2.0e3
GAP0 = 0.5 * DHAT
FRICTION_EPS = 2.0e-3
_KERNEL = None


class ConstForce:
    """External force ``f`` on dofs ``gd`` (residual −f). Body force / gravity."""

    def __init__(self, gd, f):
        self.gd = np.asarray(gd, int)
        self.f = np.asarray(f, float)

    def residual(self, U, s, t, dt):
        return Residual(self.gd, -self.f)

    def tangent(self, U, s, t, dt):
        return Tangent(np.array([], int), np.array([], int), np.array([]))

    def commit(self, U, s, t, dt):
        return s


def _hex8_box(ne, z0):
    xs = np.linspace(0.0, 1.0, ne + 1)
    zs = np.linspace(z0, z0 + 1.0, ne + 1)
    nodes = np.array([[x, y, z] for z in zs for y in xs for x in xs], float)
    nn = ne + 1

    def nid(i, j, k):
        return k * nn * nn + j * nn + i

    elems = [[nid(i, j, k), nid(i+1, j, k), nid(i+1, j+1, k), nid(i, j+1, k),
              nid(i, j, k+1), nid(i+1, j, k+1), nid(i+1, j+1, k+1), nid(i, j+1, k+1)]
             for k in range(ne) for j in range(ne) for i in range(ne)]
    return nodes, np.array(elems, int)


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = build_element_kernel(_HEX8_FOR, "nh_hex8_ppf_xcheck")
    return _KERNEL


def run(theta_deg, mu, *, ne=2, grav=0.4, damp=0.5, n_steps=150, dt=0.02):
    """Box on a tilted-gravity rigid floor. Returns ``(interface_slip, min_gap)``.

    ``interface_slip`` = mean x-displacement of the bottom-face contact nodes.
    ``min_gap`` = minimum z of the bottom nodes (> 0 ⇒ penetration-free).
    """
    nodes, elems = _hex8_box(ne, GAP0)
    nn = ne + 1
    bottom = np.array([j * nn + i for j in range(nn) for i in range(nn)], int)  # k=0
    view = KernelMeshView(nodes, elems, dof_per_node=3)
    ndof = view.ndof
    elem = CompiledElement(_kernel(), props=(G, K_BULK), dof_per_node=3,
                           n_svars=0, mcrd=3, n_elem=len(elems))
    grp = ElementGroup.from_view(view, elem, comps=(0, 1, 2))

    nodal = np.zeros(len(nodes))
    for e in elems:
        vol = float(np.prod(nodes[e].max(0) - nodes[e].min(0)))
        np.add.at(nodal, e, DENSITY * vol / 8.0)
    M = np.repeat(nodal, 3)
    inertia = InertiaOperator(M, ndof, damping=damp)

    th = np.radians(theta_deg)
    gvec = np.array([grav * np.sin(th), 0.0, -grav * np.cos(th)])   # tilted gravity
    fext = np.zeros(ndof)
    for i in range(len(nodes)):
        fext[3 * i:3 * i + 3] = nodal[i] * gvec
    gravity = ConstForce(np.arange(ndof), fext)

    floor = HalfSpace([0, 0, 0], [0, 0, 1])      # rigid floor z=0, normal +z
    contact = RigidBarrierContact(nodes, bottom, floor, dof_per_node=3,
                                  comps=(0, 1, 2), dhat=DHAT, kappa=KAPPA,
                                  mass=nodal[bottom], mu=mu, friction_eps=FRICTION_EPS)

    U, _ = solve_dynamics([grp, inertia, gravity, contact], np.zeros(ndof), ndof,
                          lambda t: {}, dt=dt, n_steps=n_steps)
    slip = float(U[bottom * 3].mean())
    min_gap = float((nodes + U.reshape(len(nodes), 3))[bottom, 2].min())
    return slip, min_gap


def main():
    # decisive contrast at θ=20°: high friction sticks vs frictionless slides
    slip_stick, gap_stick = run(20.0, 2.0)     # μ=2.0 ≫ tan20=0.36 → stick
    slip_slide, gap_slide = run(20.0, 0.0)     # frictionless → slide
    # the model-comparison point (ppf over-holds here; see README)
    slip_thr, gap_thr = run(35.0, 0.5)         # tan35=0.70 > μ=0.5

    print("box on a rigid floor, tilted gravity, ppf-smoothed friction:")
    print(f"  θ=20° μ=2.0 (stick)      : slip={slip_stick:+.4f}  min_gap={gap_stick:+.4f}")
    print(f"  θ=20° μ=0.0 (frictionless): slip={slip_slide:+.4f}  min_gap={gap_slide:+.4f}")
    print(f"  θ=35° μ=0.5 (threshold)   : slip={slip_thr:+.4f}  min_gap={gap_thr:+.4f}")

    penetration_free = min(gap_stick, gap_slide, gap_thr) > 0.0
    holds = abs(slip_stick) < 0.3 * abs(slip_slide)
    slides = abs(slip_slide) > 0.1
    ok = penetration_free and holds and slides
    print(f"  non-penetration={penetration_free}  friction_holds={holds} "
          f"(stick/slide={abs(slip_stick)/max(abs(slip_slide),1e-30):.2f})  "
          f"frictionless_slides={slides}")
    print("OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
