"""Method of Manufactured Solutions (MMS) — element convergence-rate gate.

`verify()` checks that a material tangent is self-consistent, invariants check
selected kinematics and signs, and a compiled smoke test checks execution.
Those checks do not establish the convergence behavior of the complete element
machinery: the physical shape-function gradients, quadrature, Jacobian mapping,
and assembly. A wrong B-matrix or under-integrated rule can still form a
consistent operator and produce a solution.

MMS catches it by the **convergence rate**. Manufacture an exact field, derive
the source that makes it an exact solution, solve on a refining mesh, and check
the discrete error decays at the *theoretical* order. A wrong B-matrix /
quadrature / assembly shows up immediately as a broken rate (or no convergence).

The demonstrator below solves the manufactured Poisson problem

    -div(grad u) = f ,   u = sin(pi x) sin(pi y) ,   f = 2 pi^2 sin(pi x) sin(pi y)

on the unit square (u = 0 on the boundary) with Q4 elements, using the package's
own `shape_quad4`, `map_grad_2d`, and Gauss rules — so a bug in *those* trips the
rate. Bilinear Q4 gives O(h^2) L2 convergence.
"""

from __future__ import annotations

import numpy as np

__all__ = ["assert_convergence_rate", "poisson_quad4_l2_error"]


def assert_convergence_rate(hs, errors, *, expected, atol=0.3,
                            name="MMS convergence"):
    """Assert the observed convergence rate matches the theoretical order.

    Fits the slope of ``log(error)`` vs ``log(h)``.  A wrong B-matrix /
    quadrature / assembly degrades the rate away from ``expected``.  Returns the
    fitted rate.
    """
    hs = np.asarray(hs, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if hs.ndim != 1 or hs.shape != errors.shape or hs.size < 2:
        raise AssertionError(
            f"{name}: need matching 1-D hs/errors of length >= 2 (to fit a "
            f"slope); got hs shape {hs.shape}, errors shape {errors.shape}.")
    for label, arr in (("mesh sizes h", hs), ("errors", errors)):
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            raise AssertionError(
                f"{name}: {label} must be finite and positive before taking "
                f"logs; got {arr}. (A non-finite/non-positive error means the "
                f"solve produced no usable solution — singular system, NaN.)")
    slope = float(np.polyfit(np.log(hs), np.log(errors), 1)[0])
    if abs(slope - expected) > atol:
        raise AssertionError(
            f"{name}: observed rate {slope:.2f} != expected {expected} "
            f"(+/-{atol}); errors={errors}. A wrong B-matrix, an under-"
            f"integrated quadrature rule, or a bad Jacobian/assembly shows up "
            f"here even though verify()/compile/solve all pass.")
    return slope


def poisson_quad4_l2_error(n, gauss=None, broken_bmatrix=False):
    """L2 error of the Q4 FE solution of the manufactured Poisson problem on an
    ``n x n`` unit-square mesh.

    Uses the package's ``shape_quad4`` + ``map_grad_2d`` + ``gauss`` rule (default
    2x2), so a bug in the element kinematics breaks the returned error's decay.

    ``broken_bmatrix=True`` is the broken control: it uses the parent-domain
    shape derivatives ``dshxi`` in place of the physical gradient ``dsh`` (i.e.
    forgets the Jacobian mapping) — a classic wrong-B-matrix bug. On a refining
    mesh the stiffness is then off by O(h^2) and the error DIVERGES, so the
    convergence gate catches it.
    """
    if int(n) != n or n < 2:
        raise ValueError(
            f"poisson_quad4_l2_error: mesh size n must be an integer >= 2 "
            f"(an n=1 mesh has no interior nodes); got {n!r}.")
    n = int(n)
    from ..core.reference_assembly import gauss_2d_2x2, map_grad_2d, shape_quad4
    if gauss is None:
        gauss = gauss_2d_2x2
    xi_gp, w_gp = gauss()
    pi = np.pi
    nn = n + 1

    def nid(i, j):
        return j * nn + i

    X = np.array([[i / n, j / n] for j in range(nn) for i in range(nn)])
    ndof = nn * nn
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    elems = [(ei, ej) for ej in range(n) for ei in range(n)]
    conns = {(ei, ej): [nid(ei, ej), nid(ei + 1, ej),
                        nid(ei + 1, ej + 1), nid(ei, ej + 1)]
             for (ei, ej) in elems}

    for e in elems:
        conn = conns[e]
        coords = X[conn].T                                   # (2,4)
        ke = np.zeros((4, 4))
        fe = np.zeros(4)
        for g in range(len(w_gp)):
            sh, dshxi = shape_quad4(xi_gp[g, 0], xi_gp[g, 1])
            dsh, detJ, _, stat = map_grad_2d(dshxi, coords, 4)
            if stat == 0:
                raise AssertionError("degenerate element Jacobian")
            wj = detJ * w_gp[g]
            grad = dshxi if broken_bmatrix else dsh   # broken: skip Jacobian map
            xg, yg = sh @ coords[0], sh @ coords[1]
            f = 2.0 * pi * pi * np.sin(pi * xg) * np.sin(pi * yg)
            for a in range(4):
                fe[a] += f * sh[a] * wj
                for b in range(4):
                    ke[a, b] += (grad[a, 0] * grad[b, 0]
                                 + grad[a, 1] * grad[b, 1]) * wj
        for a in range(4):
            F[conn[a]] += fe[a]
            for b in range(4):
                K[conn[a], conn[b]] += ke[a, b]

    # Dirichlet u = 0 on the boundary (the manufactured field vanishes there).
    bnd = {nid(i, j) for j in range(nn) for i in range(nn)
           if i in (0, n) or j in (0, n)}
    free = [d for d in range(ndof) if d not in bnd]
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

    err2 = 0.0
    for e in elems:
        conn = conns[e]
        coords = X[conn].T
        ue = u[conn]
        for g in range(len(w_gp)):
            sh, dshxi = shape_quad4(xi_gp[g, 0], xi_gp[g, 1])
            _, detJ, _, _ = map_grad_2d(dshxi, coords, 4)
            wj = detJ * w_gp[g]
            xg, yg = sh @ coords[0], sh @ coords[1]
            uh = sh @ ue
            uex = np.sin(pi * xg) * np.sin(pi * yg)
            err2 += (uh - uex) ** 2 * wj
    return float(np.sqrt(err2))
