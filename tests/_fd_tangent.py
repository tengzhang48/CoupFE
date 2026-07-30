"""Finite-difference tangent consistency helper for compiled element kernels.

Used by tests that need an independent check that AMATRX (or the standard-
sign K returned by CompiledElement) matches the Jacobian of the residual.
"""
from __future__ import annotations

import numpy as np


def fd_jacobian(residual, x0, h=1.0e-6):
    """Central-difference Jacobian of ``residual(x)`` at ``x0``.

    Parameters
    ----------
    residual : callable(x) -> ndarray
        Must return a 1D residual vector.
    x0 : ndarray
        State vector around which to differentiate.
    h : float
        Finite-difference step.

    Returns
    -------
    J : ndarray, shape (m, n)
        ``J[i, j] ≈ d residual_i / d x_j``.
    """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    R0 = np.asarray(residual(x0), dtype=float).ravel()
    m = len(R0)
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[j] += h
        x_minus[j] -= h
        R_plus = np.asarray(residual(x_plus), dtype=float).ravel()
        R_minus = np.asarray(residual(x_minus), dtype=float).ravel()
        J[:, j] = (R_plus - R_minus) / (2.0 * h)
    return J


def check_tangent_consistency(elem, ei, coords, U, DU, h=1.0e-6, rtol=1.0e-5,
                              atol=1.0e-8):
    """Return (K_fd, K_elem, rel_error, abs_error) for a CompiledElement.

    The element's ``element_rk`` already returns the standard-sign residual
    ``R = -RHS`` and tangent ``K = AMATRX``.  For an incremental formulation
    ``U = U_old + DU``, the algorithmic tangent is ``K = dR/d(DU)`` with
    ``U_old`` held fixed.  We therefore finite-difference ``DU`` while moving
    ``U`` with it: ``U' = U_old + DU'``.
    """
    U = np.asarray(U, dtype=float)
    DU = np.asarray(DU, dtype=float)
    U_old = U - DU

    def residual_increment(DU_):
        U_ = U_old + np.asarray(DU_, dtype=float)
        R, _ = elem.element_rk(ei, coords, U_, DU_)
        return np.asarray(R, dtype=float).ravel()

    R0, K_elem = elem.element_rk(ei, coords, U, DU)
    K_elem = np.asarray(K_elem, dtype=float)
    K_fd = fd_jacobian(residual_increment, DU, h=h)
    diff = np.abs(K_fd - K_elem)
    denom = np.maximum(np.abs(K_elem), 1.0)
    rel_error = float(np.max(diff / denom))
    abs_error = float(np.max(diff))
    return K_fd, K_elem, rel_error, abs_error
