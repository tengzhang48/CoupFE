"""Backend-agnostic operator-sign / block-definiteness gate.

Generalizes the ``OperatorSignWarning`` screen (``core/verify.py``) and
``tests/test_phase_flux_operator.py`` into a reusable assertion on raw
coefficients, so it can gate any scalar storage/flux pair — phase-field damage,
species transport, the electroneutral electric potential — across backends.

The package convention is ``r = storage*eta - flux . grad(eta)`` (strong form
``storage + Div(flux) = 0``).  A correct diffusive pair has
``flux = -Df grad(field)`` with ``Df`` positive-definite, so EVERY eigenvalue
of ``sym(dflux/dgrad)`` has the opposite sign to ``dstorage/dfield`` (the
single-field block ``s*M - d_i*L`` is then sign-definite in every eigen-
direction).  Any eigenvalue sharing the sign of ``dstorage/dfield`` is an
anti-diffusive direction — the phase-flux/LCE/Li sign trap.  The check looks at
the FULL matrix and ALL eigenvalues, so anisotropic and 3D sign errors (a bad
z-direction, or a small positive eigenvalue hiding behind a large good one) are
caught, not just isotropic 2D flips.

This is an operator-level test: it sees a convention error that CS-vs-FD cannot,
because CS-vs-FD faithfully differentiates the wrong-signed flux.
"""

from __future__ import annotations

import numpy as np

from ._util import require_finite

__all__ = [
    "assert_scalar_gradient_block_definite",
    "assert_diffusive_flux",
    "assert_coupled_field_scale_balance",
]


def assert_diffusive_flux(dflux_dgradfield, *, tol=1e-9, name="diffusive flux"):
    """Assert a flux is diffusive: ``sym(dflux/dgrad)`` has no anti-diffusive
    (positive) eigen-direction.

    The right check for a **storage-free** field equation (quasi-static /
    Laplace-type, e.g. the electroneutral electric potential) where
    ``assert_scalar_gradient_block_definite`` cannot be used — there is no
    storage term to set the reference sign.  With the convention
    ``r = storage*eta - flux.grad(eta)``, well-posedness needs
    ``flux = -Df grad(field)`` with ``Df`` positive-definite, i.e.
    ``dflux/dgrad`` negative-definite.  A positive eigenvalue is an
    anti-diffusive direction (a flipped flux sign).  Returns the eigenvalues.
    """
    require_finite(name, dflux_dgradfield=dflux_dgradfield)
    Df = np.atleast_2d(np.real(np.asarray(dflux_dgradfield, dtype=float)))
    if Df.shape[0] != Df.shape[1]:
        raise ValueError(
            f"{name}: dflux_dgradfield must be square, got shape {Df.shape}")
    eigs = np.linalg.eigvalsh(0.5 * (Df + Df.T))
    dmax = float(np.max(np.abs(eigs)))
    if dmax < 1e-300:
        raise AssertionError(f"{name}: dflux/dgrad is ~zero; no diffusion to judge.")
    bad = eigs[eigs > tol * dmax]                  # clearly positive = anti-diffusive
    if bad.size:
        raise AssertionError(
            f"{name}: flux is NOT diffusive — {bad.size} eigenvalue(s) of "
            f"sym(dflux/dgrad) are positive: {np.array2string(bad, precision=3)}. "
            f"With r = storage*eta - flux.grad(eta), a diffusive flux must be "
            f"-Df grad(field) with Df positive-definite (dflux/dgrad negative-"
            f"definite). A positive eigenvalue is an anti-diffusive (flipped-sign) "
            f"direction.")
    return eigs


def assert_scalar_gradient_block_definite(dstorage_dfield, dflux_dgradfield, *,
                                          tol=1e-9,
                                          name="scalar gradient operator"):
    """Assert the assembled single-field block is sign-definite (diffusive).

    Parameters
    ----------
    dstorage_dfield : float
        ``d(storage)/d(field)`` — the reaction/storage stiffness coefficient.
    dflux_dgradfield : float or (2,2)/(3,3) array
        ``d(flux)/d(grad field)`` as the model returns it.  For a correct
        diffusive operator this is negative-definite (flux = -Df grad f).
    tol : float
        Relative eigenvalue tolerance for the indefinite test.

    Returns the eigenvalues of ``sym(dflux_dgradfield)``.  Raises
    ``AssertionError`` if ANY eigen-direction is anti-diffusive.

    Criterion: per eigen-direction ``i`` of ``sym(Df)`` with diffusivity
    ``d_i``, the operator ``s*M - d_i*L`` is diffusive (definite) iff ``d_i``
    has the OPPOSITE sign to ``s``.  An eigenvalue sharing the sign of ``s`` is
    an anti-diffusive direction.  The check examines EVERY eigenvalue of the
    FULL matrix (not a 2x2 slice, not only the dominant one), so anisotropic
    and 3D sign errors — a bad z-direction, or a small positive eigenvalue
    hiding behind a large good one — are caught.
    """
    require_finite(name, dstorage_dfield=dstorage_dfield,
                   dflux_dgradfield=dflux_dgradfield)
    s_val = float(np.real(dstorage_dfield))
    Df = np.atleast_2d(np.real(np.asarray(dflux_dgradfield, dtype=float)))
    if Df.shape[0] != Df.shape[1]:
        raise ValueError(
            f"{name}: dflux_dgradfield must be square, got shape {Df.shape}")

    eigs = np.linalg.eigvalsh(0.5 * (Df + Df.T))   # ALL eigenvalues, FULL matrix
    dmax = float(np.max(np.abs(eigs)))
    if abs(s_val) < 1e-300 or dmax < 1e-300:
        raise AssertionError(
            f"{name}: degenerate coefficients (dstorage/dfield = {s_val:.3e}, "
            f"max|eig(dflux/dgrad)| = {dmax:.3e}); cannot judge the sign.")

    # anti-diffusive directions: eigenvalues that share the sign of s_val
    same_sign = np.sign(s_val) * eigs
    bad = eigs[same_sign > tol * dmax]
    if bad.size:
        raise AssertionError(
            f"{name}: storage/flux pair has {bad.size} ANTI-DIFFUSIVE "
            f"direction(s) — eigenvalue(s) of sym(dflux/dgrad) sharing the sign "
            f"of dstorage/dfield (={s_val:.3e}): "
            f"{np.array2string(bad, precision=3)}.  The convention is "
            f"r = storage*eta - flux.grad(eta), so a diffusive flux must be "
            f"-Df grad(field) with Df positive-definite: EVERY eigenvalue of "
            f"sym(dflux/dgrad) must have the opposite sign to dstorage/dfield. "
            f"Likely a flipped flux sign in one or more directions.")
    return eigs


def assert_coupled_field_scale_balance(K, field_dofs, *, max_ratio=1e6,
                                       nondim=None,
                                       name="coupled field scale balance"):
    """Gate the inter-field SCALE DISPARITY of a coupled multiphysics tangent.

    A coupled tangent (u-mu gel swelling, u-A magneto-mechanics, u-T thermo-
    mechanics, u-c-phi hydrogen fracture) assembled in SI units can have wildly
    different per-field block magnitudes — e.g. a momentum block ~1e6 (shear
    modulus G) next to a chemical-transport block ~1e-13 (M = D*c/RT).  Two
    failures follow, BOTH invisible to the usual screens:

    - A Newton gated on a single global ``||R||`` is dominated by the strong
      field and SILENTLY under-resolves the weak one.  Each step stops ~2 iters
      early on the weak field; the under-resolution COMPOUNDS over a transient
      (the gel case matched ABAQUS at 1 h and was 16 % low by 6 h).  The fix is
      field-wise convergence OR non-dimensionalizing the weak form.
    - The conditioning ratio (here ~5e17) is harmless to a DIRECT solver (it
      equilibrates internally) but KILLS an iterative solver (CG/GMRES/AMG).

    Neither is caught by CS-vs-FD (the tangent is self-consistent), by a single
    direct solve (it just works, with a tiny — and misleading — residual), or by
    a smoke test.  This gate is PROACTIVE: it reads the disparity off the element
    tangent alone, before any transient or solver choice.

    Parameters
    ----------
    K : (n, n) array
        Assembled element / system tangent.
    field_dofs : dict[str, sequence[int]]
        DOF indices belonging to each field, e.g.
        ``{"u": [0, 1, 3, 4], "mu": [2, 5]}`` for a node-major u-mu element.
    max_ratio : float
        Largest acceptable ratio of per-field block-diagonal magnitudes (median
        ``|K_ii|`` over each field's DOFs).  Default ``1e6`` — well above any
        legitimate physical coupling (e.g. osmotic/shear ~25) but far below a
        unit pathology.
    nondim : (row_scale, col_scale) of per-DOF arrays, optional
        A declared non-dimensionalization ``K_hat = diag(row_scale) @ K @
        diag(col_scale)`` (``row_scale`` from the equation scales, ``col_scale``
        from the variable scales — for a gel, ``col_scale[mu] = 1/RT``).  When
        given, the gate is applied to ``K_hat`` instead of ``K`` — i.e. it
        VERIFIES the declared non-dim actually brings the block ratio to O(1),
        PROVING the scaling conditions the operator rather than trusting a magic
        constant.  A non-dim that fails to balance raises.

    Returns
    -------
    dict with ``ratio`` and per-field ``block_mag`` (and, when ``nondim`` is
    given, ``ratio_nondim`` / ``block_mag_nondim``).

    Raises ``AssertionError`` if the (possibly non-dimensionalized) block ratio
    exceeds ``max_ratio``.
    """
    require_finite(name, K=K)
    K = np.atleast_2d(np.real(np.asarray(K, dtype=float)))
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"{name}: K must be square, got shape {K.shape}")
    if len(field_dofs) < 2:
        raise ValueError(
            f"{name}: need >=2 fields to judge a scale disparity, got "
            f"{list(field_dofs)}")

    def _block_mags(M):
        diag = np.abs(np.diag(M))
        mags = {}
        for f, idx in field_dofs.items():
            d = diag[np.asarray(idx, dtype=int)]
            d = d[d > 0]
            if d.size == 0:
                raise AssertionError(
                    f"{name}: field '{f}' has an all-zero tangent diagonal; "
                    f"cannot judge its block scale.")
            mags[f] = float(np.median(d))
        return mags

    block_mag = _block_mags(K)
    strong = max(block_mag, key=block_mag.get)
    weak = min(block_mag, key=block_mag.get)
    ratio = block_mag[strong] / block_mag[weak]
    result = {"ratio": ratio, "block_mag": block_mag}

    advice = (
        f"A Newton gated on a single global ||R|| will be '{strong}'-dominated "
        f"and SILENTLY under-resolve '{weak}' (the error compounds over a "
        f"transient); the conditioning also breaks iterative solvers. Fix: "
        f"non-dimensionalize the weak form (column-scale each field DOF by its "
        f"characteristic scale, e.g. mu by 1/RT) OR use field-wise convergence "
        f"(gate each field on its own residual/reference).")

    if nondim is None:
        if ratio > max_ratio:
            raise AssertionError(
                f"{name}: coupled tangent block-scale ratio {ratio:.2e} > "
                f"{max_ratio:.0e} — '{strong}' ({block_mag[strong]:.2e}) vs "
                f"'{weak}' ({block_mag[weak]:.2e}). {advice}")
        return result

    row_scale = np.asarray(nondim[0], dtype=float).ravel()
    col_scale = np.asarray(nondim[1], dtype=float).ravel()
    if row_scale.shape != (K.shape[0],) or col_scale.shape != (K.shape[0],):
        raise ValueError(
            f"{name}: nondim row/col scales must each have length {K.shape[0]}, "
            f"got {row_scale.shape} and {col_scale.shape}")
    Kn = (row_scale[:, None] * K) * col_scale[None, :]
    block_mag_nd = _block_mags(Kn)
    ratio_nd = max(block_mag_nd.values()) / min(block_mag_nd.values())
    result["ratio_nondim"] = ratio_nd
    result["block_mag_nondim"] = block_mag_nd
    if ratio_nd > max_ratio:
        raise AssertionError(
            f"{name}: the DECLARED non-dimensionalization does NOT condition the "
            f"operator — post-nondim block ratio {ratio_nd:.2e} > {max_ratio:.0e} "
            f"(raw {ratio:.2e}). The characteristic scales are wrong or "
            f"incomplete; the field blocks remain unbalanced after scaling.")
    return result
