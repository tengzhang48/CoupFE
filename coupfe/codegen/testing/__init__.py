"""coupfe.codegen validation harness — backend-agnostic constitutive checks.

Reusable invariants, path generators, and operator gates that test
*constitutive correctness*, not just consistency.  They operate on raw arrays
(``F``, ``Fe``, ``Fp``, ``P``, ``M``, ...), so the same checks validate the
coupfe.codegen reference, the CoupLAM JAX model, and CoupMPM C++ outputs.

Design rule (see VALIDATION_HARNESS_PLAN_2026-06-13.md): a check is only trusted
once it has a *broken control* — it must FAIL when a known bug is reintroduced.
See ``tests/test_validation_harness.py`` for the broken controls that gate each
primitive here.

These checks deliberately do NOT cover everything (no objectivity-with-history,
no property-based fuzzing yet — both deferred).  Each function's docstring states
what it proves and what it does not.
"""

from .finite_strain import (
    assert_cauchy_from_mandel,
    assert_isochoric_plastic,
    assert_multiplicative_split,
    assert_pk1_from_mandel,
    polar_right,
)
from .invariants import (
    assert_plastic_dissipation_nonneg,
    assert_tangent_major_symmetry,
)
from .manifest import RegimeManifest
from .element_convergence import (
    assert_convergence_rate,
    poisson_quad4_l2_error,
)
from .objectivity import (
    assert_objective_cauchy,
    assert_objective_pk1,
    assert_state_corotational,
    assert_state_invariant,
)
from .operators import (
    assert_coupled_field_scale_balance,
    assert_diffusive_flux,
    assert_scalar_gradient_block_definite,
)
from .paths import (
    noncoaxial_state,
    rotated_stretch,
    rotation,
    simple_shear,
)

__all__ = [
    # finite_strain
    "assert_multiplicative_split",
    "assert_isochoric_plastic",
    "assert_pk1_from_mandel",
    "assert_cauchy_from_mandel",
    "polar_right",
    # invariants
    "assert_plastic_dissipation_nonneg",
    "assert_tangent_major_symmetry",
    # objectivity
    "assert_objective_pk1",
    "assert_objective_cauchy",
    "assert_state_invariant",
    "assert_state_corotational",
    # manifest
    "RegimeManifest",
    # mms (element convergence-rate gate)
    "assert_convergence_rate",
    "poisson_quad4_l2_error",
    # operators
    "assert_scalar_gradient_block_definite",
    "assert_diffusive_flux",
    "assert_coupled_field_scale_balance",
    # paths
    "noncoaxial_state",
    "rotated_stretch",
    "rotation",
    "simple_shear",
]
