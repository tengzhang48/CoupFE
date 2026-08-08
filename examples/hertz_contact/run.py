"""Hertz contact: a quantitative normal-contact check for CoupFE.

A rigid sphere of radius ``R`` indents a finite Hex8 block that approximates an
elastic half-space.  For a rigid indenter, classical Hertz theory gives

``F(delta) = (4/3) E* sqrt(R) delta**(3/2)`` and
``1/E* = (1 - nu**2)/E``.

The retained model uses the same ``E`` and ``nu`` in the finite-element and
analytic calculations.  Its block is deliberately larger than the contact
patch but remains finite, and contact is enforced by discrete nodal penalty
springs.  The force law is therefore the validation observable; the radius of
the outermost active node is reported only as a mesh-resolution diagnostic.

Run from the repository root::

    PYTHONPATH=. python examples/hertz_contact/run.py

The example self-reports ``OK`` or ``FAIL``.  See the adjacent README for the
model boundary and the solver-backed figure.
"""
from __future__ import annotations

import numpy as np

from coupfe import assemble_residual, newton_solve
from coupfe.mesh import KernelMeshView
from coupfe.operators.contact import RigidContact, Sphere
from coupfe.operators.element_group import ElementGroup
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

_HEX8_FOR = "coupfe/runtime/elements/neo_hookean_hex8_fbar.for"

# Material.  The kernel evaluates
#   P = G (F - F^-T) + lambda ln(J) F^-T,
# so its second property is the first Lame coefficient, not the physical bulk
# modulus.  These conversions make the kernel's infinitesimal tangent match the
# E and nu used by the Hertz oracle.
E, NU = 10.0, 0.3
G = E / (2.0 * (1.0 + NU))
LAME_LAMBDA = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
E_STAR = E / (1.0 - NU * NU)

R_SPHERE = 2.0
LX, LY, LZ = 2.5, 2.5, 2.4
NX, NY, NZ = 16, 16, 8
PENALTY = 1.0e3
DELTAS = (0.02, 0.035, 0.05, 0.065, 0.08)


def _block_mesh(nx, ny, nz, Lx, Ly, Lz):
    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    zs = np.linspace(0, Lz, nz + 1)
    nodes = np.array([[x, y, z] for z in zs for y in ys for x in xs], float)
    nnx, nny = nx + 1, ny + 1

    def nid(i, j, k):
        return k * nnx * nny + j * nnx + i

    elems = [
        [
            nid(i, j, k),
            nid(i + 1, j, k),
            nid(i + 1, j + 1, k),
            nid(i, j + 1, k),
            nid(i, j, k + 1),
            nid(i + 1, j, k + 1),
            nid(i + 1, j + 1, k + 1),
            nid(i, j + 1, k + 1),
        ]
        for k in range(nz)
        for j in range(ny)
        for i in range(nx)
    ]
    return nodes, np.array(elems, int)


def hertz_force(delta):
    """Return the rigid-sphere Hertz force for scalar or array ``delta``."""

    return (4.0 / 3.0) * E_STAR * np.sqrt(R_SPHERE) * np.asarray(delta) ** 1.5


def analyze(deltas, force_fe):
    """Return the fitted log-log slope and ``F_FE/F_Hertz`` ratios."""

    deltas = np.asarray(deltas, dtype=float)
    force_fe = np.asarray(force_fe, dtype=float)
    slope, _ = np.polyfit(np.log(deltas), np.log(force_fe), 1)
    return float(slope), force_fe / hertz_force(deltas)


def run_hertz(deltas=DELTAS, *, verbose=False):
    """Solve the retained load series and return fields plus force-law evidence.

    The returned ``snapshot`` is the final solved load.  Contact values are
    discrete nodal penalty reactions, not a reconstructed pressure field.
    ``solve_hertz`` remains the compact tuple-returning compatibility wrapper.
    """

    deltas = np.asarray(tuple(deltas), dtype=float)
    if deltas.ndim != 1 or deltas.size < 2 or not np.all(np.isfinite(deltas)):
        raise ValueError("deltas must contain at least two finite values")
    if np.any(deltas <= 0.0) or np.any(np.diff(deltas) <= 0.0):
        raise ValueError("deltas must be strictly increasing and positive")

    nodes, elems = _block_mesh(NX, NY, NZ, LX, LY, LZ)
    nn = len(nodes)
    top = np.flatnonzero(np.abs(nodes[:, 2] - LZ) < 1.0e-9)
    bottom = np.flatnonzero(np.abs(nodes[:, 2]) < 1.0e-9)
    view = KernelMeshView(nodes, elems, dof_per_node=3)
    ndof = view.ndof
    elem = CompiledElement(
        build_element_kernel(_HEX8_FOR, "nh_hex8_hertz"),
        props=(G, LAME_LAMBDA),
        dof_per_node=3,
        n_svars=0,
        mcrd=3,
        n_elem=len(elems),
    )
    group = ElementGroup.from_view(view, elem, comps=(0, 1, 2))
    dirichlet = {int(node) * 3 + comp: 0.0 for node in bottom for comp in (0, 1, 2)}
    constrained = np.array(sorted(dirichlet), dtype=int)
    cx, cy = LX / 2.0, LY / 2.0

    if verbose:
        print(
            f"Hertz: rigid sphere R={R_SPHERE} on a {NX}x{NY}x{NZ} Hex8 block, "
            f"E*={E_STAR:.3f}"
        )
        print(
            f"{'delta':>8} {'F_FE':>10} {'F_Hertz':>10} {'F_FE/F_H':>9} "
            f"{'active':>7} {'a_node':>8}"
        )

    displacement = np.zeros(ndof)
    cases = []
    final_snapshot = None
    for delta in deltas:
        sphere_center = np.array([cx, cy, LZ - delta + R_SPHERE])
        sphere = Sphere(sphere_center, R_SPHERE)
        contact = RigidContact(
            nodes,
            top,
            sphere,
            dof_per_node=3,
            comps=(0, 1, 2),
            k=PENALTY,
            mu=0.0,
        )
        displacement, _, iterations = newton_solve(
            [group, contact],
            displacement,
            None,
            ndof,
            dirichlet,
            t=1.0,
            dt=1.0,
        )

        equilibrium_residual, _ = assemble_residual(
            [group, contact], displacement, None, 1.0, 1.0, ndof
        )
        free_residual = equilibrium_residual.copy()
        free_residual[constrained] = 0.0
        free_residual_norm = float(np.linalg.norm(free_residual))

        internal_residual, _ = assemble_residual(
            [group], displacement, None, 1.0, 1.0, ndof
        )
        base_vertical_residual = float(np.sum(internal_residual[bottom * 3 + 2]))
        force_fe = base_vertical_residual

        nodal_displacement = displacement.reshape(nn, 3)
        deformed_nodes = nodes + nodal_displacement
        contact_positions = deformed_nodes[top]
        gaps = np.asarray(sphere.gap(contact_positions), dtype=float)
        normals = np.asarray(sphere.normal(contact_positions), dtype=float)
        penetration = np.maximum(-gaps, 0.0)
        normal_reaction = PENALTY * penetration
        vertical_reaction = normal_reaction * (-normals[:, 2])
        active = gaps < 0.0
        radial_position = np.linalg.norm(contact_positions[:, :2] - [cx, cy], axis=1)
        active_node_radius = float(radial_position[active].max()) if active.any() else 0.0
        force_balance_error = abs(
            float(np.sum(vertical_reaction)) - base_vertical_residual
        )
        force_reference = float(hertz_force(delta))

        case = {
            "delta": float(delta),
            "force_fe": force_fe,
            "force_hertz": force_reference,
            "force_ratio": force_fe / force_reference,
            "active_contact_nodes": int(np.count_nonzero(active)),
            "active_node_radius": active_node_radius,
            "hertz_radius": float(np.sqrt(R_SPHERE * delta)),
            "newton_iterations": int(iterations),
            "free_residual_norm": free_residual_norm,
            "base_vertical_residual": base_vertical_residual,
            "force_balance_error": force_balance_error,
            "max_penetration": float(np.max(penetration)),
        }
        cases.append(case)
        final_snapshot = {
            "delta": float(delta),
            "nodes_reference": nodes.copy(),
            "nodes_deformed": deformed_nodes.copy(),
            "displacement": nodal_displacement.copy(),
            "elements": elems.copy(),
            "top_nodes": top.copy(),
            "bottom_nodes": bottom.copy(),
            "contact_gap": gaps.copy(),
            "contact_normal_reaction": normal_reaction.copy(),
            "contact_vertical_reaction": vertical_reaction.copy(),
            "active_contact": active.copy(),
            "sphere_center": sphere_center.copy(),
        }
        if verbose:
            print(
                f"{delta:8.3f} {force_fe:10.4f} {force_reference:10.4f} "
                f"{case['force_ratio']:9.3f} {case['active_contact_nodes']:7d} "
                f"{active_node_radius:8.3f}"
            )

    force_fe = np.array([case["force_fe"] for case in cases])
    active_node_radius = np.array([case["active_node_radius"] for case in cases])
    slope, force_ratios = analyze(deltas, force_fe)
    return {
        "configuration": {
            "youngs_modulus": E,
            "poisson_ratio": NU,
            "shear_modulus": G,
            "lame_lambda": LAME_LAMBDA,
            "hertz_modulus": E_STAR,
            "sphere_radius": R_SPHERE,
            "block_size": (LX, LY, LZ),
            "mesh_shape": (NX, NY, NZ),
            "nodes": nn,
            "elements": len(elems),
            "degrees_of_freedom": ndof,
            "penalty": PENALTY,
        },
        "deltas": deltas,
        "force_fe": force_fe,
        "force_hertz": hertz_force(deltas),
        "force_ratios": force_ratios,
        "fit_slope": slope,
        "active_node_radius": active_node_radius,
        "cases": cases,
        "snapshot": final_snapshot,
    }


def solve_hertz(deltas=DELTAS, *, verbose=False):
    """Return ``(deltas, F_FE, active_node_radius)`` for compatibility."""

    evidence = run_hertz(deltas=deltas, verbose=verbose)
    return evidence["deltas"], evidence["force_fe"], evidence["active_node_radius"]


def main():
    evidence = run_hertz(verbose=True)
    slope = evidence["fit_slope"]
    ratios = evidence["force_ratios"]
    errors_percent = 100.0 * (ratios - 1.0)
    print(
        f"log-log slope = {slope:.3f} (Hertz: 1.500);  "
        f"force error in [{errors_percent.min():+.1f}%, {errors_percent.max():+.1f}%]"
    )
    exponent_ok = abs(slope - 1.5) < 0.08
    force_ok = bool(np.max(np.abs(ratios - 1.0)) < 0.08)
    solve_ok = all(
        case["free_residual_norm"] < 1.0e-7
        and case["force_balance_error"] < 1.0e-6
        for case in evidence["cases"]
    )
    ok = exponent_ok and force_ok and solve_ok
    print(
        f"  exponent near 3/2: {exponent_ok}   force within 8%: {force_ok}   "
        f"equilibrium: {solve_ok}"
    )
    print("OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
