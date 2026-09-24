"""Thick spherical shell under internal pressure, meshed in the r-z meridian.

The shell is solved with generated axisymmetric kernels (:mod:`kernels`); the
inner surface ``rho = A`` is loaded by a prescribed cavity pressure through
:class:`coupfe.AxisymmetricCavity`, ``z = 0`` is a symmetry plane and ``r = 0``
the axis. References, both independent of the finite-element code:

* nearly incompressible: the closed-form incompressible neo-Hookean relation
  ``p = G [2/lb + 1/(2 lb^4) - 2/la - 1/(2 la^4)]`` with ``la = a/A``,
  ``lb = b/B`` and ``b^3 = B^3 + a^3 - A^3``;
* compressible: a spherically symmetric boundary-value solve of the same
  Yeoh law with ``scipy.integrate.solve_bvp``, using the principal-stretch
  stresses of :func:`principal_pk1`.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp

from coupfe import AxisymmetricCavity, assemble_residual, newton_solve
from coupfe.operators.axisymmetric import boundary_edges

import kernels


def principal_pk1(props, l1, l2, l3):
    """Nominal stresses ``dW/d lambda_i`` of the Yeoh law for principal stretches.

    ``props = (C10, C20, C30, K)``. Written independently of the generated
    kernels: ``W = sum C_i0 (I1b - 3)**i + K/2 (J - 1)**2``.
    """
    C10, C20, C30, K = props
    J = l1 * l2 * l3
    I1 = l1 * l1 + l2 * l2 + l3 * l3
    Jm23 = J ** (-2.0 / 3.0)
    x = Jm23 * I1 - 3.0
    dW = C10 + 2.0 * C20 * x + 3.0 * C30 * x * x
    vol = K * (J - 1.0) * J
    return tuple(dW * Jm23 * (2.0 * li - 2.0 * I1 / (3.0 * li)) + vol / li for li in (l1, l2, l3))


def mapped_block(element, s, t, mapping=None):
    """Tensor-product block on parameters ``s`` x ``t``, optionally mapped to (r, z).

    Quadratic elements place mid nodes at parametric midpoints. Cells are
    counterclockwise for an orientation-preserving mapping. Unused serendipity
    centre nodes are removed.
    """
    quadratic = element in ("quad8", "quad8r", "tri6")
    s, t = np.asarray(s, float), np.asarray(t, float)
    if quadratic:
        s = np.sort(np.concatenate([s, 0.5 * (s[1:] + s[:-1])]))
        t = np.sort(np.concatenate([t, 0.5 * (t[1:] + t[:-1])]))
    m = 2 if quadratic else 1
    S, T = np.meshgrid(s, t, indexing="ij")
    P = np.column_stack([S.ravel(), T.ravel()])
    X = P if mapping is None else np.column_stack(mapping(P[:, 0], P[:, 1]))
    nt = len(t)

    def idx(i, j):
        return i * nt + j

    cells = []
    for i in range(0, len(s) - 1, m):
        for j in range(0, nt - 1, m):
            c = [idx(i, j), idx(i + m, j), idx(i + m, j + m), idx(i, j + m)]
            if element == "quad4":
                cells.append(c)
            elif element in ("quad8", "quad8r"):
                cells.append(c + [idx(i + 1, j), idx(i + 2, j + 1), idx(i + 1, j + 2), idx(i, j + 1)])
            elif element == "tri3":
                cells += [[c[0], c[1], c[2]], [c[0], c[2], c[3]]]
            elif element == "tri6":
                mid = idx(i + 1, j + 1)
                cells += [[c[0], c[1], c[2], idx(i + 1, j), idx(i + 2, j + 1), mid],
                          [c[0], c[2], c[3], mid, idx(i + 1, j + 2), idx(i, j + 1)]]
            else:
                raise ValueError(f"unknown element {element!r}")
    cells = np.asarray(cells)
    used = np.unique(cells)
    remap = np.full(len(X), -1)
    remap[used] = np.arange(len(used))
    return X[used], remap[cells]


def edge_family(element):
    return "quad8" if element == "quad8r" else element


def shell_mesh(element, A, B, n_rho, n_theta):
    """Quarter meridian ``A <= rho <= B``, ``0 <= theta <= pi/2`` from the r axis.

    Returns nodes, cells and the cavity wall edges ordered counterclockwise
    around the cavity (the reverse of the solid's inner boundary).
    """
    X, cells = mapped_block(element, np.linspace(A, B, n_rho + 1),
                            np.linspace(0.0, np.pi / 2, n_theta + 1),
                            lambda rho, th: (rho * np.cos(th), rho * np.sin(th)))
    X[np.abs(X) < 1e-14 * B] = 0.0
    edges = boundary_edges(cells, edge_family(element))
    radius = np.hypot(X[edges[:, :2], 0], X[edges[:, :2], 1])
    inner = edges[np.isclose(radius, A).all(axis=1)]
    cavity = inner[:, [1, 0] + ([2] if inner.shape[1] == 3 else [])]
    return X, cells, cavity


def solve_inflation(element, formulation, props, p_end, *, A=1.0, B=2.0,
                    n_rho=6, n_theta=8, steps=12):
    """Pressure-controlled inflation; returns the deformed inner radius and details."""
    X, cells, cavity = shell_mesh(element, A, B, n_rho, n_theta)
    nn = len(X)
    dpn = kernels.layout(element, formulation)
    pdof, ndof = dpn * nn, dpn * nn + 1
    solid = kernels.element_group(X, cells, element, formulation, props)
    cav = AxisymmetricCavity(X, cavity, pdof, dof_per_node=dpn)
    bc = {dpn * k: 0.0 for k in np.nonzero(X[:, 0] == 0.0)[0]}
    bc.update({dpn * k + 1: 0.0 for k in np.nonzero(X[:, 1] == 0.0)[0]})
    bc.update({g: 0.0 for g in kernels.unused_pressure_dofs(cells, element, formulation, nn)})
    probe = int(np.nonzero((X[:, 1] == 0.0) & np.isclose(X[:, 0], A))[0][0])
    U = np.zeros(ndof)
    iterations = 0
    for step in range(1, steps + 1):
        d = dict(bc)
        d[pdof] = p_end * step / steps
        U, _, nit = newton_solve([solid, cav], U, None, ndof, d, rtol=1e-10, atol=1e-12,
                                 maxit=40, line_search="admissible", predictor="tangent")
        iterations += nit
        R, _ = assemble_residual([solid, cav], U, None, 1.0, 1.0, ndof)
        R[list(d)] = 0.0
        if not np.linalg.norm(R) < 1e-8 * max(1.0, p_end):
            raise RuntimeError(f"{element}/{formulation}: step {step} did not converge")
    return {"a": A + U[dpn * probe], "U": U, "solid": solid, "cavity": cav,
            "iterations": iterations, "nodes": nn, "elements": len(cells)}


def incompressible_pressure(G, A, B, a):
    """Closed-form inflation pressure of an incompressible neo-Hookean shell."""
    b = (B ** 3 + a ** 3 - A ** 3) ** (1.0 / 3.0)
    la, lb = a / A, b / B
    return G * (2 / lb + 1 / (2 * lb ** 4) - 2 / la - 1 / (2 * la ** 4))


def bvp_inner_radius(props, A, B, p):
    """Deformed inner radius from a spherically symmetric solve of the Yeoh law."""
    def pk(lr, lt):
        Prr, Ptt, _ = principal_pk1(props, lr, lt, lt)
        return Prr, Ptt

    def fun(R, y):
        r, rp = y
        lt = r / R
        Prr, Ptt = pk(rp.astype(complex), lt.astype(complex))
        h = 1e-30
        d_lr = pk(rp + 1j * h, lt + 0j)[0].imag / h
        d_lt = pk(rp + 0j, lt + 1j * h)[0].imag / h
        rpp = -(2.0 * (Prr.real - Ptt.real) / R + d_lt * (rp - lt) / R) / d_lr
        return np.vstack([rp, rpp])

    def bc(ya, yb):
        Pa = pk(np.array([ya[1]]), np.array([ya[0] / A]))[0][0]
        Pb = pk(np.array([yb[1]]), np.array([yb[0] / B]))[0][0]
        return np.array([Pa + p * (ya[0] / A) ** 2, Pb])   # P_rr(A) = -p lambda_theta^2

    R = np.linspace(A, B, 400)
    with np.errstate(invalid="ignore"):   # solve_bvp probes inadmissible trial states
        sol = solve_bvp(fun, bc, R, np.vstack([R * 1.1, np.ones_like(R)]), tol=1e-9, max_nodes=400000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return float(sol.sol(A)[0])
