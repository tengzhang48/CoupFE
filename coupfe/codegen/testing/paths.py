"""Deterministic non-coaxial finite-strain states.

The point of these generators is NOT random coverage — it is to construct
states where the principal directions rotate and the plastic increment does
not commute with the prior plastic gradient, so that diagonal/commuting matrix
mistakes (the Anand M1/H1 class, masked by axisymmetric tests) cannot hide.

All generators are deterministic (no RNG) so results are reproducible and the
broken-control tests are stable.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rotation",
    "simple_shear",
    "rotated_stretch",
    "noncoaxial_state",
]


def _expm_sym(A):
    """Matrix exponential of a SYMMETRIC matrix via eigendecomposition.

    Exact for symmetric ``A`` (the only case used here), numpy-only so the
    harness adds no dependency.
    """
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    return (V * np.exp(w)) @ V.T


def rotation(theta, axis="z"):
    """Proper rotation by ``theta`` (radians) about a coordinate axis."""
    c, s = np.cos(theta), np.sin(theta)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    raise ValueError(f"axis must be x/y/z, got {axis!r}")


def simple_shear(gamma):
    """Simple shear ``F = I + gamma * e1⊗e2`` (a non-coaxial, isochoric path)."""
    F = np.eye(3)
    F[0, 1] = gamma
    return F


def rotated_stretch(stretches=(1.15, 0.95, 1.0), theta=0.6, axis="z"):
    """``F = R @ diag(stretches)`` — stretch then rotate (non-symmetric F)."""
    R = rotation(theta, axis)
    return R @ np.diag(np.asarray(stretches, dtype=float))


def noncoaxial_state(*, theta=0.6, stretches=(1.15, 0.95, 1.0),
                     fp_shear=0.08, dp_angle=0.9, dp_mag=0.12,
                     dilatant=False, dil_rate=0.05, dt=1.0):
    """A non-coaxial finite-strain plasticity state.

    Returns a dict with:
      ``F``        total deformation gradient (rotation + stretch, non-symmetric)
      ``Fp_old``   a non-diagonal prior plastic gradient (``det = 1``)
      ``Dp``       a plastic rate that does NOT commute with ``Fp_old``
      ``Fp_new``   exact exponential update ``expm(dt*Dp) @ Fp_old``
      ``dilatant`` whether the plastic rate has a volumetric part
      ``dt``

    By construction ``Fp_old`` and ``Dp`` have different principal frames, so
    the (wrong) order ``Fe = Fe_tr @ inv(Fp_new) @ Fp_old`` differs from the
    (correct) ``Fe = F @ inv(Fp_new)`` — i.e. the multiplicative-split invariant
    discriminates here but would not on a diagonal/coaxial state.

    With ``dilatant=True`` the plastic rate gets a volumetric part
    (``Dp += dil_rate*I``), so ``det(Fp_new) = exp(3*dt*dil_rate) != 1`` — the
    state needed to exercise the plastic-Jacobian (``det(Fp)``) factor in the
    PK1 mapping (an omitted factor passes when isochoric, fails when dilatant).
    """
    F = rotated_stretch(stretches, theta)

    # Non-diagonal prior plastic gradient (volume preserving): unimodular shear.
    Fp_old = np.eye(3)
    Fp_old[0, 1] = fp_shear
    Fp_old[1, 2] = 0.5 * fp_shear
    Fp_old = Fp_old / np.linalg.det(Fp_old) ** (1.0 / 3.0)

    # Deviatoric symmetric plastic rate in a DIFFERENT frame than Fp_old.
    R = rotation(dp_angle, axis="z")
    core = np.diag([1.0, -0.5, -0.5])
    Dp = dp_mag * (R @ core @ R.T)            # symmetric, traceless -> isochoric
    if dilatant:
        Dp = Dp + dil_rate * np.eye(3)        # volumetric part -> det(Fp_new) != 1

    Fp_new = _expm_sym(dt * Dp) @ Fp_old
    return {"F": F, "Fp_old": Fp_old, "Dp": Dp, "Fp_new": Fp_new,
            "dilatant": dilatant, "dt": dt}
