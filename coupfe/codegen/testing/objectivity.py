"""Material frame-indifference (objectivity) under a superposed rotation.

Objectivity is a property of the constitutive RESPONSE, so it needs two
evaluations — at ``F`` and at ``Q @ F`` for a rotation ``Q`` — not a single
(F, stress) pair.  To stay backend-agnostic, the caller performs both
evaluations with its own calling convention and feeds the two outputs here; the
harness checks that each quantity transformed by the correct rule.

The subtle part is that history must transform consistently,
NOT just ``F``.  Under a superposed spatial rotation ``x -> Q x`` the
intermediate (plastically relaxed) configuration does NOT move, so:

    F        (two-point ref->current)      ->  Q @ F
    Fe       (two-point inter->current)     ->  Q @ Fe
    Fp       (two-point ref->inter)         ->  Fp          (INVARIANT)
    Mandel M (intermediate config)          ->  M           (INVARIANT)
    PK1 P    (two-point ref->current)       ->  Q @ P
    Cauchy   sigma (spatial)                ->  Q sigma Q^T
    Kirchhoff tau, B = F F^T (spatial sym)  ->  Q (.) Q^T

So when re-running at ``Q@F``, the OLD plastic/intermediate state is passed
UNCHANGED (Fp_old is invariant); a spatial old-state tensor (e.g. B_old) is
rotated by ``Q (.) Q^T``.  A model that wrongly rotates ``Fp`` (treats the
intermediate config as spatial), or returns a non-objective stress, is caught
by ``assert_state_invariant`` / ``assert_objective_pk1`` respectively.

These checks only test the transformation rules — they assume the caller drove
both evaluations consistently.  Pair with ``finite_strain`` checks for the
kinematics.
"""

from __future__ import annotations

import numpy as np

from ._util import require_finite

__all__ = [
    "assert_objective_pk1",
    "assert_objective_cauchy",
    "assert_state_invariant",
    "assert_state_corotational",
]


def _check(resid, scale, rtol, atol, name, msg):
    if resid > rtol * scale + atol:
        raise AssertionError(
            f"{name}: {msg}  (||.|| = {resid:.3e}, relative "
            f"{resid / max(scale, 1e-300):.3e}).")


def assert_objective_pk1(P, P_super, Q, *, rtol=1e-8, atol=1e-10,
                         name="objective PK1"):
    """Assert ``P(Q F) == Q @ P(F)`` (PK1 is a two-point ref->current tensor)."""
    P = np.asarray(P, float)
    P_super = np.asarray(P_super, float)
    Q = np.asarray(Q, float)
    require_finite(name, P=P, P_super=P_super, Q=Q)
    expected = Q @ P
    resid = float(np.linalg.norm(P_super - expected))
    _check(resid, float(np.linalg.norm(expected)), rtol, atol, name,
           "P(QF) != Q P(F) — the PK1 stress is not frame-indifferent")
    return resid


def assert_objective_cauchy(sigma, sigma_super, Q, *, rtol=1e-8, atol=1e-10,
                            name="objective Cauchy"):
    """Assert ``sigma(Q F) == Q sigma(F) Q^T`` (Cauchy is a spatial tensor)."""
    sigma = np.asarray(sigma, float)
    sigma_super = np.asarray(sigma_super, float)
    Q = np.asarray(Q, float)
    require_finite(name, sigma=sigma, sigma_super=sigma_super, Q=Q)
    expected = Q @ sigma @ Q.T
    resid = float(np.linalg.norm(sigma_super - expected))
    _check(resid, float(np.linalg.norm(expected)), rtol, atol, name,
           "sigma(QF) != Q sigma(F) Q^T — the Cauchy stress is not objective")
    return resid


def assert_state_invariant(A, A_super, *, rtol=1e-8, atol=1e-10,
                           name="intermediate-config state invariance"):
    """Assert an intermediate-config quantity is UNCHANGED under superposed Q.

    Use for ``Fp`` and the Mandel stress: the relaxed/intermediate configuration
    does not move with a superposed spatial rotation, so these are invariant.  A
    model that rotates ``Fp`` (treats it as spatial) fails here because its
    history transformation is inconsistent.
    """
    A = np.asarray(A, float)
    A_super = np.asarray(A_super, float)
    require_finite(name, A=A, A_super=A_super)
    resid = float(np.linalg.norm(A_super - A))
    _check(resid, float(np.linalg.norm(A)), rtol, atol, name,
           "an intermediate-config quantity (e.g. Fp) changed under a "
           "superposed rotation — it should be invariant")
    return resid


def assert_state_corotational(A, A_super, Q, *, rtol=1e-8, atol=1e-10,
                              name="spatial state corotation"):
    """Assert a spatial symmetric tensor corotates: ``A(Q F) == Q A(F) Q^T``.

    Use for spatial state such as ``B = F F^T`` or a spatial back-stress.
    """
    A = np.asarray(A, float)
    A_super = np.asarray(A_super, float)
    Q = np.asarray(Q, float)
    require_finite(name, A=A, A_super=A_super, Q=Q)
    expected = Q @ A @ Q.T
    resid = float(np.linalg.norm(A_super - expected))
    _check(resid, float(np.linalg.norm(expected)), rtol, atol, name,
           "a spatial tensor did not corotate as Q (.) Q^T")
    return resid
