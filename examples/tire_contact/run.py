"""RESEARCH tire-under-gravity contact driver.

This composes a **hollow rubber torus** (mixed u-p Hex8), its inner rim
**held** (the wheel mount), under
**gravity**, resting tread-down on a rigid `HalfSpace` ground with **smoothed Coulomb friction**, solved
by `solve_dynamics`.

Only the mesh currently has a reviewed pytest gate. No retained full-solve
output, equilibrium study, or exact GetFEM source/result supports a
reproduction or validation claim; see `examples/REFERENCES.md`.

The driver self-reports diagnostic gap, patch, and reaction checks. Those
labels help a research rerun but are not a release gate.
"""
from __future__ import annotations

import numpy as np

from coupfe import InertiaOperator, solve_dynamics
from coupfe.mesh import KernelMeshView
from coupfe.operators.base import Residual, Tangent
from coupfe.operators.contact import HalfSpace, RigidBarrierContact
from coupfe.operators.element_group import ElementGroup
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

from examples.tire_contact.mesh import torus_hex_mesh, tread_surface_nodes, rim_surface_nodes

_UP_HEX8_FOR = "examples/neo_hookean_local_pressure_hex8/neohookean_up_hex8_uel.for"

# Geometry
R, R_IN, R_OUT = 1.0, 0.30, 0.45
N_PHI, N_THETA, N_RHO = 48, 16, 2
# Material — soft near-incompressible rubber (εg = ρg·L/G ~ 0.5 → notable sag, still converges)
G, K_BULK, DENSITY = 1.0, 1.0e2, 1.0
GRAV = 0.5
# Contact + dynamics
DHAT, KAPPA = 0.04, 2.0e3
GAP0 = 0.5 * DHAT
MU, FRICTION_EPS = 0.4, 2.0e-3
DT, N_STEPS, DAMP = 0.02, 60, 0.5     # 60 steps reaches a penetration-free resting patch (tested)


class ConstForce:
    """Body force ``f`` on dofs ``gd`` (residual −f)."""
    def __init__(self, gd, f):
        self.gd = np.asarray(gd, int)
        self.f = np.asarray(f, float)
    def residual(self, U, s, t, dt):
        return Residual(self.gd, -self.f)
    def tangent(self, U, s, t, dt):
        return Tangent(np.array([], int), np.array([], int), np.array([]))
    def commit(self, U, s, t, dt):
        return s


def _lumped_mass(nodes, elems, density):
    nodal = np.zeros(len(nodes))
    for e in elems:
        vol = float(np.prod(nodes[e].max(0) - nodes[e].min(0)))   # AABB proxy (uniform-ish hexes)
        np.add.at(nodal, e, density * vol / 8.0)
    return nodal


def solve_tire(grav=GRAV, mu=MU, n_steps=N_STEPS, n_phi=N_PHI, n_theta=N_THETA, n_rho=N_RHO,
               verbose=True):
    nodes, elems = torus_hex_mesh(R, R_IN, R_OUT, n_phi, n_theta, n_rho)
    tread = tread_surface_nodes(nodes, R, R_OUT)
    rim = rim_surface_nodes(nodes, R, R_IN)
    view = KernelMeshView(nodes, elems, dof_per_node=3)
    ndof = view.ndof

    elem = CompiledElement(build_element_kernel(_UP_HEX8_FOR, "tire_up_hex8"),
                           props=(G, K_BULK), dof_per_node=3, n_svars=1, mcrd=3, n_elem=len(elems))
    grp = ElementGroup.from_view(view, elem, comps=(0, 1, 2))

    nodal = _lumped_mass(nodes, elems, DENSITY)
    M = np.repeat(nodal, 3)
    inertia = InertiaOperator(M, ndof, damping=DAMP)

    fext = np.zeros(ndof)
    fext[2::3] = -nodal * grav                                 # gravity, −z, on every node
    gravity = ConstForce(np.arange(ndof), fext)

    z_ground = nodes[:, 2].min() - GAP0                        # floor just below the tread (gap0 > 0)
    floor = HalfSpace([0.0, 0.0, z_ground], [0.0, 0.0, 1.0])
    contact = RigidBarrierContact(nodes, tread, floor, dof_per_node=3, comps=(0, 1, 2),
                                  dhat=DHAT, kappa=KAPPA, mass=nodal[tread],
                                  mu=mu, friction_eps=FRICTION_EPS)

    # Rim held in x,y (upright, no tipping/fore-aft/roll) but FREE in z: gravity drops the tire
    # vertically onto the floor so the tread flattens into a contact patch (a suspension-strut mount).
    rim_dofs = {int(n) * 3 + c: 0.0 for n in rim for c in (0, 1)}
    U, _ = solve_dynamics([grp, inertia, gravity, contact], np.zeros(ndof), ndof,
                          lambda t: rim_dofs, dt=DT, n_steps=n_steps)
    # static-equilibrium operators (NO inertia) + constrained dofs, for the adjoint (Phase 3)
    static_ops = [grp, gravity, contact]
    rim_con = sorted(rim_dofs)

    pos = nodes + U.reshape(len(nodes), 3)
    min_gap = float(pos[tread, 2].min() - z_ground)
    in_contact = int(np.sum(pos[tread, 2] - z_ground < DHAT))
    max_sag = float(-U.reshape(len(nodes), 3)[:, 2].min())     # how far the tire dropped (z)
    if verbose:
        print(f"tire: {len(nodes)} nodes, {len(elems)} Hex8, {len(tread)} tread, {len(rim)} rim")
        print(f"  min tread gap   = {min_gap:+.4e}  (d̂={DHAT}) -> penetration_free={min_gap > 0}")
        print(f"  tread in-contact= {in_contact} / {len(tread)} nodes (contact patch)")
        print(f"  max downward sag= {max_sag:.4e}")
    return dict(U=U, nodes=nodes, elems=elems, tread=tread, rim=rim, z_ground=z_ground,
                min_gap=min_gap, in_contact=in_contact, max_sag=max_sag,
                static_ops=static_ops, ndof=ndof, rim_con=rim_con, grav=grav, fext=fext)


def main():
    r = solve_tire()
    ok = (r["min_gap"] > 0.0) and (r["in_contact"] > 0) and (r["max_sag"] > GAP0)
    print("OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
