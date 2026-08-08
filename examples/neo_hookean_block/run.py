"""Solve a 2D neo-Hookean block under uniaxial stretch through ``newton_solve``.

    python examples/neo_hookean_block/run.py

Reports the converged Newton iterations, the tip stretch, and the mean lateral
contraction — cross-checked against the analytic traction-free lateral stretch of a
homogeneous plane-strain neo-Hookean block (the example's independent oracle).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from block import DEFAULT_PROPS, uniaxial_problem  # noqa: E402

from coupfe import solve_increments  # noqa: E402


def lateral_stretch_analytic(lam, props):
    """Traction-free lateral stretch ``lam_t`` of a homogeneous plane-strain block.

    Plane strain (F_33 = 1), F = diag(lam, lam_t, 1).  The lateral PK1 component
    must vanish: P_22 = G(lam_t - 1/lam_t) + lambda ln(lam*lam_t) / lam_t = 0,
    where ``lambda = K - 2G/3`` for the public physical bulk modulus ``K``.
    Solve for lam_t by 1D Newton — this is the independent reference the FE
    result is gauged on.
    """
    G, K = props
    lame_lambda = K - 2.0 * G / 3.0
    lt = 1.0
    for _ in range(50):
        J = lam * lt
        P22 = G * (lt - 1.0 / lt) + lame_lambda * np.log(J) / lt
        dP22 = (G * (1.0 + 1.0 / lt**2)
                + lame_lambda * (1.0 - np.log(J)) / lt**2)
        step = P22 / dP22
        lt -= step
        if abs(step) < 1e-14:
            break
    return lt


def main(nx=6, ny=6, Lx=1.0, Ly=1.0, stretch=0.20, props=DEFAULT_PROPS):
    nodes, elems, group, dirichlet = uniaxial_problem(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, stretch=stretch, props=props)
    ndof = len(nodes) * 2
    # Finite strain → ramp the prescribed stretch over a few load increments.
    U, nit = solve_increments([group], np.zeros(ndof), ndof, dirichlet, n_steps=4)

    ux, uy = U[0::2], U[1::2]
    lam = 1.0 + stretch
    # Mean lateral stretch from the top edge's y-displacement.
    top = np.nonzero(np.abs(nodes[:, 1] - Ly) < 1e-9)[0]
    lam_t_fe = 1.0 + uy[top].mean() / Ly
    lam_t_ref = lateral_stretch_analytic(lam, props)

    print(f"=== neo-Hookean block {nx}x{ny}, props (G,K)={props}, stretch={stretch} ===")
    print(f"  Newton iterations         : {nit}")
    print(f"  axial stretch  lam        : {lam:.6f}")
    print(f"  lateral stretch (FE mean) : {lam_t_fe:.6f}")
    print(f"  lateral stretch (analytic): {lam_t_ref:.6f}")
    print(f"  relative error            : {abs(lam_t_fe - lam_t_ref) / lam_t_ref:.2e}")
    print(f"  max |u_x|, |u_y|          : {np.abs(ux).max():.4f}, {np.abs(uy).max():.4f}")
    return U


if __name__ == "__main__":
    main()
