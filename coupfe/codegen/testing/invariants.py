"""Thermodynamic and structural invariants: dissipation and tangent symmetry.

These complement the kinematic/stress-mapping checks in ``finite_strain.py``.
Both are gated on a declared convention: the dissipation
check needs an explicit work-conjugate pair, and the symmetry check must only
fire when the model declares symmetry is physically expected.
"""

from __future__ import annotations

import numpy as np

from ._util import require_finite

__all__ = [
    "assert_plastic_dissipation_nonneg",
    "assert_tangent_major_symmetry",
]


def assert_plastic_dissipation_nonneg(mandel, plastic_rate, *, rtol=1e-10,
                                      name="plastic dissipation"):
    """Assert the plastic dissipation ``M : Dp >= 0`` (reduced dissipation ineq).

    Parameters
    ----------
    mandel : (3,3) array
        Mandel stress ``M`` on the elastic intermediate configuration.
    plastic_rate : (3,3) array
        Plastic rate of deformation ``Dp`` (symmetric part of
        ``Lp = Fdot_p Fp^-1``); for the slip models here
        ``Dp = sum_a gamma_dot_a (sym(s_a x m_a) + beta m_a x m_a)``.

    The second law requires ``M : Dp >= 0``.  This is the pair declared by
    ``RegimeManifest.conjugate_pair == "mandel:plastic_rate"`` (Anand's
    dilatancy restriction tau - beta*sigma_n >= 0 is exactly this being
    non-negative).  A flipped flow direction or slip-rate sign makes it
    negative.  Returns ``M : Dp``.

    NOTE: this is the Mandel:Dp pair specifically.  Other formulations declare
    other pairs (P:Fdot, sigma:D, sigma:deps) — pass the matching tensors; do
    not assume one universal dissipation formula.
    """
    M = np.asarray(mandel, dtype=float)
    Dp = np.asarray(plastic_rate, dtype=float)
    require_finite(name, mandel=M, plastic_rate=Dp)
    diss = float(np.tensordot(M, Dp))                 # M : Dp = sum_ij M_ij Dp_ij
    scale = float(np.linalg.norm(M) * np.linalg.norm(Dp))
    if diss < -rtol * max(scale, 1.0):
        raise AssertionError(
            f"{name}: M : Dp = {diss:.3e} < 0  (relative {diss / max(scale, 1e-300):.3e}). "
            f"The plastic dissipation is NEGATIVE — a second-law violation, e.g. "
            f"a flipped plastic flow direction or slip-rate sign, or a plastic "
            f"rate that is not the gradient of the yield/flow potential.")
    return diss


def assert_tangent_major_symmetry(K, *, expected=True, rtol=1e-8, atol=1e-10,
                                  name="tangent major symmetry"):
    """Assert ``K == K^T`` when (and only when) symmetry is expected.

    ``expected`` should come from ``RegimeManifest.tangent_symmetric``.
    Associated / hyperelastic tangents are symmetric; non-associated plasticity,
    coupled multiphysics, and some push-forward/Jaumann paths are legitimately
    UNsymmetric — checking symmetry there gives FALSE failures.  When
    ``expected=False`` this never raises (it just returns the measured asymmetry
    for information).  Returns the relative asymmetry ``||K-K^T|| / ||K||``.
    """
    K = np.asarray(K, dtype=float)
    require_finite(name, K=K)
    scale = max(float(np.linalg.norm(K)), 1.0)
    asym = float(np.linalg.norm(K - K.T)) / scale
    if expected and asym > rtol + atol / scale:
        raise AssertionError(
            f"{name}: tangent is NOT major-symmetric (||K-K^T||/||K|| = "
            f"{asym:.3e}) but the model declares symmetry is expected.  Either "
            f"a tangent assembly bug, or the manifest should set "
            f"tangent_symmetric=False (non-associated / coupled / Jaumann).")
    return asym
