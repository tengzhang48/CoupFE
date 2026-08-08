"""Serial 3D deformable contact + smoothed FRICTION: two soft Hex8 blocks, the top sheared. ``python run.py``.

The single-process companion to ``examples/mpi_smoke/distributed_dynamics_3d_friction.py``. Modest
gravity seats the top **F-bar Hex8** block onto the bottom one (a sustained normal load → a sustained
barrier normal force ``λ_n``), and the top face is dragged sideways (``+x``). ppf/IPC **smoothed
friction** on the 3D tangent plane (``P = I − n⊗n``) at the interface resists the slide: with ``μ>0``
the interface slips noticeably LESS than the frictionless block. Contact runs on the numba narrow-phase
+ LBVH broad-phase.

`friction_eps` is chosen on the interface-slip scale so the regularized
Gauss-Newton tangent remains usable; see `docs/theory/contact_dynamics.md`.

Self-check: ``μ>0`` interface slip < 0.7× the frictionless slip (friction acts) AND penetration-free.
Prints ``OK`` / ``FAIL``.
"""
from __future__ import annotations

import numpy as np

from coupfe import InertiaOperator, neo_hookean_kernel_props, solve_dynamics
from coupfe.mesh import KernelMeshView
from coupfe.operators.base import Residual, Tangent
from coupfe.operators.contact3d import DeformableBarrierContact3D, point_triangle_coeff_unclassified
from coupfe.operators.element_group import ElementGroup
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

_HEX8_FOR = "coupfe/runtime/elements/neo_hookean_hex8_fbar.for"
NE = 2
G, K_BULK, DENSITY, GRAV = 1.0, 10.0, 1.0, 0.4          # εg = ρgL/G = 0.4 (seats, still converges)
DHAT, KAPPA = 0.04, 2.0e3                               # stiff barrier (inertia regularizes under dynamics)
GAP0 = 0.5 * DHAT                                       # start inside the band → gravity seats it
MU, SHEAR, FRICTION_EPS = 0.8, 0.05, 2.0e-3            # friction holds; eps ≳ slip → tight convergence
DT, N_STEPS, DAMP = 0.02, 50, 3.0


class ConstForce:                                       # external force f on dofs gd (residual −f)
    def __init__(self, gd, f):
        self.gd = np.asarray(gd, dtype=int); self.f = np.asarray(f, dtype=float)

    def residual(self, U, state, t, dt):
        return Residual(self.gd, -self.f)

    def tangent(self, U, state, t, dt):
        return Tangent(np.array([], int), np.array([], int), np.array([]))

    def commit(self, U, state, t, dt):
        return state


def _hex8_block(ne, z0):
    xs = np.linspace(0.0, 1.0, ne + 1); zs = np.linspace(z0, z0 + 1.0, ne + 1)
    nodes = np.array([[x, y, z] for z in zs for y in xs for x in xs], dtype=float)
    nn = ne + 1

    def nid(i, j, k):
        return k * nn * nn + j * nn + i

    elems = [[nid(i, j, k), nid(i+1, j, k), nid(i+1, j+1, k), nid(i, j+1, k),
              nid(i, j, k+1), nid(i+1, j, k+1), nid(i+1, j+1, k+1), nid(i, j+1, k+1)]
             for k in range(ne) for j in range(ne) for i in range(ne)]
    return nodes, np.array(elems, dtype=int)


def _surface_grid(ne, offset, z_pick):
    nn = ne + 1; k = 0 if z_pick == "min" else ne
    return np.array([[offset + k*nn*nn + j*nn + i for i in range(nn)] for j in range(nn)])


def _triangulate(grid):
    nn = grid.shape[0]; tris = []
    for j in range(nn - 1):
        for i in range(nn - 1):
            a, b, c, d = grid[j, i], grid[j, i+1], grid[j+1, i+1], grid[j+1, i]
            tris += [[a, b, d], [b, c, d]]
    return np.array(tris, dtype=int)


def _lumped_mass(nodes, elems, density, ndof):
    M = np.zeros(ndof); nodal = np.zeros(len(nodes))
    for e in elems:
        vol = float(np.prod(nodes[e].max(0) - nodes[e].min(0)))
        np.add.at(nodal, e, density * vol / 8.0)
    for c in range(3):
        M[c::3] = nodal
    return M


def _min_signed_interface_gap(U, nodes, secondary, faces):
    """Minimum oriented secondary-to-primary gap for the block interface."""
    pos = nodes + U.reshape(len(nodes), 3)
    min_gap = np.inf
    for v in secondary:
        v = int(v)
        closest_distance = np.inf
        closest_signed_gap = np.inf
        for f in faces:
            w = point_triangle_coeff_unclassified(pos[v], pos[f[0]], pos[f[1]], pos[f[2]])
            closest = w[0] * pos[f[0]] + w[1] * pos[f[1]] + w[2] * pos[f[2]]
            separation = pos[v] - closest
            distance = float(np.linalg.norm(separation))
            if distance >= closest_distance:
                continue

            reference_normal = np.cross(nodes[f[1]] - nodes[f[0]],
                                        nodes[f[2]] - nodes[f[0]])
            deformed_normal = np.cross(pos[f[1]] - pos[f[0]],
                                       pos[f[2]] - pos[f[0]])
            reference_area = float(np.linalg.norm(reference_normal))
            deformed_area = float(np.linalg.norm(deformed_normal))
            if reference_area <= 1.0e-14 or deformed_area <= 1.0e-14:
                return -np.inf
            if float(deformed_normal @ reference_normal) < 0.0:
                deformed_normal = -deformed_normal
            deformed_normal /= deformed_area
            closest_distance = distance
            closest_signed_gap = float(separation @ deformed_normal)
        min_gap = min(min_gap, closest_signed_gap)
    return float(min_gap)


def _build():
    nb, eb = _hex8_block(NE, 0.0); Nb = len(nb)
    nt, et = _hex8_block(NE, 1.0 + GAP0)
    nodes = np.vstack([nb, nt]); elems = np.vstack([eb, et + Nb])
    faces = _triangulate(_surface_grid(NE, 0, "max"))
    secondary = _surface_grid(NE, Nb, "min").ravel()
    base = np.arange((NE + 1) ** 2)
    top_face = _surface_grid(NE, Nb, "max").ravel()
    top_block = np.arange(Nb, len(nodes))
    return nodes, elems, secondary, faces, base, top_face, top_block


def _run(mu, nodes, elems, secondary, faces, base, top_face, top_block):
    view = KernelMeshView(nodes, elems, dof_per_node=3); ndof = view.ndof
    elem = CompiledElement(build_element_kernel(_HEX8_FOR, "neo_hex8_serial_fr"),
                           props=neo_hookean_kernel_props(G, K_BULK),
                           dof_per_node=3, n_svars=0, mcrd=3, n_elem=len(elems))
    grp = ElementGroup.from_view(view, elem, comps=(0, 1, 2))
    M = _lumped_mass(nodes, elems, DENSITY, ndof)
    inertia = InertiaOperator(M, ndof, damping=DAMP)
    gravity = ConstForce(top_block * 3 + 2, -M[top_block * 3 + 2] * GRAV)     # seat the top block
    contact = DeformableBarrierContact3D(nodes, secondary, faces, np.zeros((0, 2), int),
                                         dof_per_node=3, dhat=DHAT, kappa=KAPPA, mu=mu,
                                         friction_eps=FRICTION_EPS)

    def dirichlet(frac):                                 # base fixed; top face dragged +x (z,y free)
        d = {int(n) * 3 + c: 0.0 for n in base for c in (0, 1, 2)}
        for n in top_face:
            d[int(n) * 3 + 0] = SHEAR * frac
        return d

    operators = [grp, inertia, gravity, contact]
    U = np.zeros(ndof)
    min_signed_gap = np.inf
    for step in range(1, N_STEPS + 1):
        # Advance exactly one accepted step so the public non-penetration
        # evidence covers the trajectory rather than only the final state.
        U, _ = solve_dynamics(
            operators,
            U,
            ndof,
            dirichlet(step / N_STEPS),
            dt=DT,
            n_steps=1,
        )
        min_signed_gap = min(
            min_signed_gap,
            _min_signed_interface_gap(U, nodes, secondary, faces),
        )
    return U, float(min_signed_gap)


def main():
    mesh = _build()
    nodes, _, secondary, faces = mesh[0], mesh[1], mesh[2], mesh[3]
    U_fric, min_gap = _run(MU, *mesh)
    U_free, _ = _run(0.0, *mesh)
    slip_fric = float(np.abs(np.mean(U_fric[secondary * 3 + 0])))
    slip_free = float(np.abs(np.mean(U_free[secondary * 3 + 0])))
    held = slip_fric < 0.7 * slip_free
    penetration_free = min_gap > 0.0
    ok = held and penetration_free and slip_free > 1e-3
    print(f"interface slip: μ={MU} -> {slip_fric:.4e},  μ=0 -> {slip_free:.4e}  (ratio {slip_fric/max(slip_free,1e-30):.2f})")
    print(f"min signed interface gap over trajectory (μ>0) = {min_gap:.3e} (d̂={DHAT})  held={held} penetration_free={penetration_free}  -> {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
