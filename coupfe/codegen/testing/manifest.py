"""RegimeManifest — a model's declared validation envelope and conventions.

The harness applies only the checks a model *claims* are physically expected.
Two review cautions make this necessary:

  - **Major symmetry is not universal.**  Associated / hyperelastic tangents
    are symmetric, but non-associated plasticity, coupled multiphysics UELs, and
    some finite-strain push-forward / Jaumann paths are legitimately UNsymmetric.
    The symmetry check must be gated by ``tangent_symmetric``.
  - **Dissipation needs an explicit work-conjugate pair.**  There is no single
    universal dissipation formula; the model declares which pair its
    dissipation check uses via ``conjugate_pair``.

The remaining flags are descriptive — they document the declared envelope and
seed later regime-enforcement (asserting a model actually ENTERS the regime it
claims, the zhang_soga gap) and edge-case generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["RegimeManifest"]


@dataclass(frozen=True)
class RegimeManifest:
    """Declared validation envelope for a constitutive model.

    Parameters
    ----------
    name : str
        Model identifier.
    conjugate_pair : str
        The stress/rate work-conjugate pair the dissipation check uses, e.g.
        ``"mandel:plastic_rate"`` (M : Dp), ``"pk1:Fdot"`` (P : Fdot),
        ``"cauchy:D"`` (sigma : D), ``"smallstrain:strainrate"``
        (sigma : deps).  No universal formula — the caller passes the matching
        pair to ``assert_plastic_dissipation_nonneg``.
    tangent_symmetric : bool
        Whether major symmetry of the tangent is physically expected.  False
        for non-associated plasticity, coupled multiphysics, some Jaumann/
        push-forward paths.  Gates ``assert_tangent_major_symmetry``.
    finite_strain, dilatant, rate_dependent, noncoaxial_capable, gradient : bool
        Descriptive envelope flags.
    coupled_fields : tuple[str, ...]
        Extra solution fields beyond displacement (e.g. ``("phi", "mu", "T")``).
    field_scales : dict[str, float]
        Characteristic (non-dimensionalization) scale per field — the value used
        to make that field's variable O(1), e.g. ``{"u": L, "mu": R*T}`` for a
        gel.  Seeds ``assert_coupled_field_scale_balance``'s non-dim check: the
        column scale of a coupled DOF is ``1/field_scales[field]``.  A badly-
        scaled coupled tangent (SI units, a momentum block ~1e6 next to a
        transport block ~1e-13) silently under-resolves the weak field under a
        global-norm Newton and wrecks iterative-solver conditioning; these scales
        are how the harness proves a non-dim conditions it.
    notes : str
        Free-form provenance / caveats.
    """

    name: str
    conjugate_pair: str = "mandel:plastic_rate"
    tangent_symmetric: bool = True
    finite_strain: bool = True
    dilatant: bool = False
    rate_dependent: bool = False
    noncoaxial_capable: bool = True
    gradient: bool = False
    coupled_fields: tuple = field(default_factory=tuple)
    field_scales: dict = field(default_factory=dict)
    notes: str = ""


# Reference manifests for the models this harness already exercises.  These
# document the conventions the checks must respect (note the differing symmetry
# expectation: J2 associated -> symmetric; Anand non-associated dilatant
# rate-dependent -> NOT symmetric).
J2_FEFP = RegimeManifest(
    name="J2_FeFp",
    conjugate_pair="mandel:plastic_rate",
    tangent_symmetric=True,         # associated J2, hyperelastic Hencky
    finite_strain=True,
    dilatant=False,
    rate_dependent=False,
)

ANAND_2025 = RegimeManifest(
    name="anand_2025_rock",
    conjugate_pair="mandel:plastic_rate",
    tangent_symmetric=False,        # non-associated (dilatant), rate-dependent
    finite_strain=True,
    dilatant=True,
    rate_dependent=True,
    gradient=True,                  # gradient-damage regularization
    coupled_fields=("d",),
    notes="Coulomb-Mohr slip with beta dilatancy; explicit forward-Euler.",
)

# Chester-Anand gel: a coupled u-mu transient where the SI-unit block disparity
# (momentum ~G=1e6 vs transport ~M=D*c/RT~1e-13) is the failure mode that
# assert_coupled_field_scale_balance exists to catch.  The non-dim scales below
# bring it to O(1): displacement by the domain length L, chemical potential by
# R*T (mu/RT ~ O(1)).  Coupled UELs are legitimately UNsymmetric.
CHESTER_ANAND_GEL = RegimeManifest(
    name="chester_anand_gel",
    tangent_symmetric=False,        # coupled multiphysics u-mu
    finite_strain=True,
    rate_dependent=True,            # backward-Euler solvent transport
    coupled_fields=("mu",),
    field_scales={"u": 2.5e-3, "mu": 8.3145 * 298.0},   # L (m), R*T (J/mol)
    notes="Neo-Hookean + Flory-Huggins, condensed local pressure; transport "
          "M=D*cR/RT is ~1e-13 in SI -> needs field-wise convergence (direct) "
          "or non-dimensionalization (iterative).",
)
