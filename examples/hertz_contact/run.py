"""Hertz contact — the quantitative analytic benchmark for CoupFE normal contact.

A rigid **sphere** (radius R) indents a deformable Hex8 block (a half-space) by a
prescribed approach δ.  Hertz theory gives the normal force in closed form::

    F(δ) = (4/3) · E* · √R · δ^(3/2),     1/E* = (1-ν²)/E   (rigid indenter)

so a log–log fit of F vs δ must have **slope 3/2** and the prefactor must match
`(4/3)E*√R`.  Run from the repo root::

    PYTHONPATH=. python examples/hertz_contact/run.py

This is a genuinely *quantitative* check (unlike the qualitative ppf cross-check
in `examples/contact_vs_ppf/`), but on a **coarse** mesh with a **finite** block,
so the tolerance is loose: a finite-depth block with a fixed base is stiffer than
a true half-space, biasing F **above** Hertz, and a coarse contact patch
under-resolves the pressure.  The robust, mesh-insensitive signal is the
**exponent 3/2**; the prefactor is checked within a loose band.

Self-reports ``OK`` / ``FAIL``.
"""
from __future__ import annotations

import numpy as np

from coupfe import assemble_residual, newton_solve
from coupfe.mesh import KernelMeshView
from coupfe.operators.contact import RigidContact, Sphere
from coupfe.operators.element_group import ElementGroup
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

_HEX8_FOR = "coupfe/runtime/elements/neo_hookean_hex8_fbar.for"

# material (E, ν) -> (G, K) for the neo-Hookean kernel; E* for a rigid indenter
E, NU = 10.0, 0.3
G = E / (2.0 * (1.0 + NU))
K_BULK = E / (3.0 * (1.0 - 2.0 * NU))
E_STAR = E / (1.0 - NU * NU)

R_SPHERE = 2.0
LX, LY, LZ = 2.0, 2.0, 1.5
NX, NY, NZ = 12, 12, 5
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

    elems = [[nid(i, j, k), nid(i+1, j, k), nid(i+1, j+1, k), nid(i, j+1, k),
              nid(i, j, k+1), nid(i+1, j, k+1), nid(i+1, j+1, k+1), nid(i, j+1, k+1)]
             for k in range(nz) for j in range(ny) for i in range(nx)]
    return nodes, np.array(elems, int), nnx, nny


def hertz_force(delta):
    return (4.0 / 3.0) * E_STAR * np.sqrt(R_SPHERE) * delta ** 1.5


def solve_hertz(deltas=DELTAS, *, verbose=False):
    """Indent the block by each δ; return ``(deltas, F_FE, a_FE)`` arrays."""
    nodes, elems, _, _ = _block_mesh(NX, NY, NZ, LX, LY, LZ)
    nn = len(nodes)
    top = np.nonzero(np.abs(nodes[:, 2] - LZ) < 1e-9)[0]      # contact face
    bottom = np.nonzero(np.abs(nodes[:, 2]) < 1e-9)[0]        # fixed base
    view = KernelMeshView(nodes, elems, dof_per_node=3)
    ndof = view.ndof
    elem = CompiledElement(build_element_kernel(_HEX8_FOR, "nh_hex8_hertz"),
                           props=(G, K_BULK), dof_per_node=3, n_svars=0,
                           mcrd=3, n_elem=len(elems))
    grp = ElementGroup.from_view(view, elem, comps=(0, 1, 2))
    dirichlet = {int(n) * 3 + c: 0.0 for n in bottom for c in (0, 1, 2)}
    cx, cy = LX / 2.0, LY / 2.0

    if verbose:
        print(f"Hertz: rigid sphere R={R_SPHERE} on a {NX}x{NY}x{NZ} block, E*={E_STAR:.3f}")
        print(f"{'delta':>8} {'F_FE':>10} {'F_Hertz':>10} {'F_FE/F_H':>9} {'a_FE':>7} {'a_H':>7}")
    U = np.zeros(ndof)
    F_fe, a_fe_list = [], []
    for d in deltas:
        z_c = LZ - d + R_SPHERE                               # sphere indents top by d
        sphere = Sphere([cx, cy, z_c], R_SPHERE)
        contact = RigidContact(nodes, top, sphere, dof_per_node=3,
                               comps=(0, 1, 2), k=PENALTY, mu=0.0)
        U, _, _ = newton_solve([grp, contact], U, None, ndof, dirichlet, t=1.0, dt=1.0)
        # total normal force = the fixed-base vertical reaction (= the contact load)
        R_int, _ = assemble_residual([grp], U, None, 1.0, 1.0, ndof)
        F = abs(float(np.sum(R_int[bottom * 3 + 2])))
        pos = nodes + U.reshape(nn, 3)
        pen = np.array([float(sphere.gap(pos[int(t_)])) < 0 for t_ in top])
        rad = np.sqrt((pos[top, 0] - cx) ** 2 + (pos[top, 1] - cy) ** 2)
        a_fe = float(rad[pen].max()) if pen.any() else 0.0
        F_fe.append(F); a_fe_list.append(a_fe)
        if verbose:
            print(f"{d:8.3f} {F:10.4f} {hertz_force(d):10.4f} "
                  f"{F/max(hertz_force(d),1e-30):9.3f} {a_fe:7.3f} {np.sqrt(R_SPHERE*d):7.3f}")
    return np.asarray(deltas, float), np.asarray(F_fe), np.asarray(a_fe_list)


def analyze(deltas, F_fe):
    """Return ``(loglog_slope, F_FE/F_Hertz ratios)``."""
    slope, _ = np.polyfit(np.log(deltas), np.log(F_fe), 1)
    return float(slope), F_fe / hertz_force(deltas)


def main():
    deltas, F_fe, _ = solve_hertz(verbose=True)
    slope, ratios = analyze(deltas, F_fe)
    print(f"log-log slope = {slope:.3f} (Hertz: 1.500);  "
          f"F_FE/F_Hertz in [{ratios.min():.2f}, {ratios.max():.2f}]")
    exponent_ok = abs(slope - 1.5) < 0.15
    prefactor_ok = 0.7 < np.median(ratios) < 1.6     # loose: coarse mesh + finite depth
    ok = exponent_ok and prefactor_ok
    print(f"  exponent≈3/2: {exponent_ok}   prefactor within loose band: {prefactor_ok}")
    print("OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
