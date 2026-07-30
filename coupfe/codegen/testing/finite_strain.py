"""Backend-agnostic finite-strain constitutive invariants.

These checks operate on raw NumPy(-compatible) arrays — ``F``, ``Fe``, ``Fp``,
``P``, ``M``, Cauchy ``sigma`` — NOT on ``coupfe.codegen`` model objects.  The same
primitives therefore validate the ``coupfe.codegen`` reference, the CoupLAM JAX
model, and CoupMPM C++ outputs read into arrays.  They are *kinematic invariants
and stress-mapping identities*, independent of the constitutive law, so they
catch the bug class that CS-vs-FD / f2py / feacheap structurally cannot: those
faithfully differentiate, mirror, or integrate the code that exists — even when
that code encodes the wrong equation.

What each check PROVES and does NOT prove
-----------------------------------------
``assert_multiplicative_split``
    PROVES the returned ``(Fe, Fp)`` reproduce ``F`` (kinematic consistency).
    Does NOT prove the elastic/plastic split is physically correct — only that
    ``F == Fe @ Fp``.  Catches factor-order bugs such as the Anand
    ``Fe = Fe_tr @ inv(Fp_new) @ Fp_old`` (correct only for commuting/coaxial
    states, wrong under non-coaxial loading).

``assert_isochoric_plastic``
    PROVES ``det(Fp) == 1``.  Apply ONLY to isochoric flow (J2 / von Mises).
    Dilatant models (Anand, Cam-Clay, Nor-Sand) legitimately have
    ``det(Fp) != 1`` — do not call it there.  Catches the linearized update
    ``Fp_new = (I + dgamma*N) @ Fp_old`` masquerading as ``expm(dgamma*N)``.

``assert_pk1_from_mandel`` / ``assert_cauchy_from_mandel``
    PROVE the returned stress is the correct push-forward of the returned
    Mandel stress through the returned ``(Fe, Fp)``.  Do NOT prove the Mandel
    stress itself is right (that needs an independent oracle).  The PK1 form
    catches the Anand H1 bug (pulling back with ``Fp_old^-T`` after updating
    ``Fp``); the Cauchy form is the right check for models that return Cauchy
    stress (CoupLAM/CoupMPM), where H1 is structurally absent.

Stress-mapping convention (isotropic elasticity)
------------------------------------------------
With ``Fe = Re @ Ue`` (right polar) and the symmetric Mandel stress ``M`` coaxial
with ``Ce = Fe^T Fe``::

    Cauchy   sigma = det(Fe)^-1 * Re @ M @ Re^T
    Kirchhoff tau  = det(Fp)   * Re @ M @ Re^T        (= J*sigma)
    PK1      P     = J*sigma@F^-T = det(Fp) * Re @ M @ inv(Ue) @ inv(Fp)^T

The ``det(Fp)`` (plastic Jacobian) factor in ``P`` matters only for dilatant
flow; for isochoric flow ``det(Fp)=1`` and it drops out.
"""

from __future__ import annotations

import numpy as np

from ._util import require_finite

__all__ = [
    "polar_right",
    "assert_multiplicative_split",
    "assert_isochoric_plastic",
    "assert_pk1_from_mandel",
    "assert_cauchy_from_mandel",
]


def polar_right(Fe):
    """Right polar decomposition ``Fe = Re @ Ue`` (Re orthogonal, Ue SPD)."""
    Fe = np.asarray(Fe, dtype=float)
    U, s, Vt = np.linalg.svd(Fe)
    Re = U @ Vt
    Ue = (Vt.T * s) @ Vt
    return Re, Ue


def _resid_norm(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b)), float(max(np.linalg.norm(b), 1.0))


def assert_multiplicative_split(F, Fe, Fp, *, rtol=1e-10, atol=1e-12,
                                name="multiplicative split"):
    """Assert ``F == Fe @ Fp`` (the defining kinematic identity).

    Returns the residual norm.  Raises ``AssertionError`` on violation.
    """
    F = np.asarray(F, dtype=float)
    Fe = np.asarray(Fe, dtype=float)
    Fp = np.asarray(Fp, dtype=float)
    require_finite(name, F=F, Fe=Fe, Fp=Fp)
    resid, scale = _resid_norm(F, Fe @ Fp)
    if resid > rtol * scale + atol:
        raise AssertionError(
            f"{name}: F != Fe @ Fp  (||F - Fe@Fp|| = {resid:.3e}, "
            f"relative {resid / scale:.3e}).  The returned elastic/plastic "
            f"gradients do not reproduce F — a kinematic factor-order bug "
            f"(e.g. Fe = Fe_tr @ inv(Fp_new) @ Fp_old, correct only when "
            f"Fp_old commutes with the plastic increment).")
    return resid


def assert_isochoric_plastic(Fp, *, atol=1e-10, name="isochoric plastic flow"):
    """Assert ``det(Fp) == 1`` (isochoric / volume-preserving plasticity).

    Only valid for J2 / von Mises.  Returns ``det(Fp)``.
    """
    Fp = np.asarray(Fp, dtype=float)
    require_finite(name, Fp=Fp)
    detFp = float(np.linalg.det(Fp))
    if abs(detFp - 1.0) > atol:
        raise AssertionError(
            f"{name}: det(Fp) = {detFp:.12f} != 1  (off by {detFp - 1.0:.3e}). "
            f"Plastic flow is not volume preserving — e.g. a linearized update "
            f"Fp_new = (I + dgamma*N) @ Fp_old instead of expm(dgamma*N). "
            f"(If the model is intentionally dilatant, do not use this check.)")
    return detFp


def assert_cauchy_from_mandel(sigma, Fe, M, *, rtol=1e-8, atol=1e-10,
                              name="Cauchy from Mandel"):
    """Assert ``sigma == det(Fe)^-1 * Re @ M @ Re^T`` (Re from polar(Fe)).

    The right check for models that return **Cauchy** stress (CoupLAM/CoupMPM).
    ``M`` is the symmetric intermediate-config Mandel stress.  Returns residual.
    """
    sigma = np.asarray(sigma, dtype=float)
    Fe = np.asarray(Fe, dtype=float)
    M = np.asarray(M, dtype=float)
    require_finite(name, sigma=sigma, Fe=Fe, M=M)
    Re, _ = polar_right(Fe)
    Je = float(np.linalg.det(Fe))
    sigma_expected = (Re @ M @ Re.T) / Je
    resid, scale = _resid_norm(sigma, sigma_expected)
    if resid > rtol * scale + atol:
        raise AssertionError(
            f"{name}: sigma != det(Fe)^-1 Re M Re^T  (||.|| = {resid:.3e}, "
            f"relative {resid / scale:.3e}).  The Cauchy stress is inconsistent "
            f"with the returned Mandel stress and elastic gradient.")
    return resid


def assert_pk1_from_mandel(P, F, Fe, M, *, rtol=1e-8, atol=1e-10,
                           name="PK1 from Mandel"):
    """Assert ``P == det(F) * sigma @ inv(F)^T``, ``sigma = det(Fe)^-1 Re M Re^T``.

    The right check for models that return **PK1** stress (coupfe.codegen).  This is
    the universal PK1<->Cauchy identity ``P = J sigma F^-T`` evaluated from the
    returned ``(F, Fe, M)``; when ``F = Fe @ Fp`` it equals
    ``det(Fp) Re M inv(Ue) Fp^-T``.  Because it is built from ``F`` directly:

    - an ``F != Fe @ Fp`` inconsistency is caught (``F`` is actually used);
    - the ``det(F)/det(Fe) = det(Fp)`` plastic-Jacobian factor is enforced, so
      an implementation that OMITS it fails on a **dilatant** path (where
      ``det(Fp) != 1``) while still passing on isochoric paths;
    - the Anand **H1** bug is caught — a model that pulls back with ``Fp_old^-T``
      after updating ``Fp`` returns a ``P`` that disagrees with this mapping.

    Returns the residual norm.  Pair with ``assert_multiplicative_split`` to also
    certify the kinematics the mapping assumes.
    """
    P = np.asarray(P, dtype=float)
    F = np.asarray(F, dtype=float)
    Fe = np.asarray(Fe, dtype=float)
    M = np.asarray(M, dtype=float)
    require_finite(name, P=P, F=F, Fe=Fe, M=M)
    Re, _ = polar_right(Fe)
    sigma = (Re @ M @ Re.T) / float(np.linalg.det(Fe))
    P_expected = float(np.linalg.det(F)) * sigma @ np.linalg.inv(F).T
    resid, scale = _resid_norm(P, P_expected)
    if resid > rtol * scale + atol:
        raise AssertionError(
            f"{name}: P != det(F) sigma F^-T  (||.|| = {resid:.3e}, relative "
            f"{resid / scale:.3e}).  The PK1 push-forward is inconsistent with "
            f"the returned (F, Fe, Mandel) — e.g. pulling back with Fp_old^-T "
            f"after updating Fp (Anand H1), an omitted det(Fp) factor on a "
            f"dilatant path, or F != Fe @ Fp.")
    return resid
