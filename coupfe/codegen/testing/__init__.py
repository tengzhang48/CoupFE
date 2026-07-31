"""coupfe.codegen validation harness — backend-agnostic constitutive checks.

Reusable invariants, path generators, and operator gates test constitutive
properties rather than only implementation consistency. They operate on raw
arrays (``F``, ``Fe``, ``Fp``, ``P``, ``M``, ...), so an application can use
them with another backend after translating its outputs to the documented
array convention.

For an important claim, add an independent oracle and, where practical, a
deliberately broken control. See ``skills/testing.md`` for the public evidence
guidance.

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
