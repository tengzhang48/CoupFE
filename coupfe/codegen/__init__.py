"""
coupfe.codegen — Python-to-Fortran code generator for Abaqus UMAT/UEL.

Users write constitutive models and weak forms in Python. The package:
  1. Provides tangent-consistency checks for supported declarations.
  2. Generates a self-contained Fortran ``.for`` file that runs in
     Abaqus or, for supported elements, the native CoupFE runtime.

Capability summary
------------------
What you get when you ``import coupfe.codegen as au``:

    Materials & weak forms
        au.Material, au.SmallStrainMaterial
        au.WeakForm
        au.VectorField, au.ScalarField, au.LocalScalar

    Tensor algebra (CS-safe; usable in both real and complex codepaths)
        from coupfe.codegen.core.tensor import det, inv, exp, log, sqrt, trace,
                                           sym, dev, eig, logm, expm, ...

    Plasticity / soil helpers
        au.small_strain_plasticity   (mean_stress, q_mises, ...)
        au.soil                      (MCC, Nor-Sand helpers)

    UEL generators
        au.generate_uel(problem, path, element=..., formulation=...)
            built-in element configurations include Quad4, Quad8/Quad8R,
            Tet4/Tet4R, Hex8, and Hex20
            formulation ∈ {'standard', 'fbar_mechanics',
                           'local_pressure'}
        au.generate_uel_local_pressure(weakform, output_path, ...)

    UMAT generators
        au.generate_umat(material, path)            # finite-strain
        au.generate_umat(material, path, matrix_backend="iterative")
            # opt-in fixed-count iterative sqrtm/logm/expm backend
        au.generate_small_strain_umat(material, path)
            # small-strain materials may define self._helper(...) methods;
            # the generator emits each helper as a Fortran subroutine.
            # See docs/api.md and the public code-generation examples.

    Hand-written Fortran sidecars (small-strain UMAT)
        @au.fortran_helper(inputs=[...], outputs=[...], subroutine="...")
        def _my_helper(self, x):
            ...
        au.generate_small_strain_umat(..., extra_fortran_files=["helper.for"])
            # Python body runs in verify(); generator emits a CALL to the
            # named Fortran subroutine and appends the sidecar verbatim.

    Element / mesh metadata
        au.ElementConfig, au.ELEMENT_CONFIGS

    Abaqus .inp scaffold
        au.UELModelConfig, au.ScaffoldReport
        au.generate_inp_scaffold(...)
        au.write_job_inp(...)
        au.ensure_coupled_dummy_material(...)

    Verification
        material.verify()                           # raises VerificationError
        au.VerificationError                        # for try/except in tests

    Symbolic tangent (prototype, optional)
        au.SymbolicTangent

Minimal example
---------------

    import coupfe.codegen as au
    from coupfe.codegen.core.tensor import det, inv, log

    class NeoHookean(au.Material):
        props = dict(G=1.0, K=100.0)
        def stress_PK1(self, F):
            J = det(F)
            return self.G * (F - inv(F).T) + self.K * log(J) * inv(F).T

    model = NeoHookean(G=0.5, K=50.0)
    model.verify()                                    # raises if any block fails
    au.generate_umat(model, "neo_hookean_umat.for")

For the current public API and support boundary, see ``docs/api.md`` and
``docs/capabilities.md``. Runnable generator examples are indexed in
``examples/README.md`` and their evidence is recorded in
``examples/REFERENCES.md``.
"""

from .core.material import Material, SmallStrainMaterial
from .core.weakform import WeakForm
from .core.fields import VectorField, ScalarField, LocalScalar
from .core.verify import VerificationError
from .core.fortran_helper import fortran_helper
from .core import tensor
from .core import small_strain_plasticity
from .core import soil
from .generators.element_config import ElementConfig, ELEMENT_CONFIGS
from .generators.inp_scaffold import UELModelConfig, ScaffoldReport

# Legacy generator-lineage/API marker retained for generated-header
# compatibility. It is intentionally independent of the installable CoupFE
# distribution version in ``coupfe.__version__``.
GENERATOR_API_VERSION = "0.1.0"
__version__ = GENERATOR_API_VERSION


def generate_uel(*args, **kwargs):
    """Generate an Abaqus UEL; imported lazily to keep core imports light."""
    from .generators.uel_gen import generate_uel as _generate_uel
    return _generate_uel(*args, **kwargs)


def generate_uel_local_pressure(*args, **kwargs):
    """Generate a Quad4/Hex8 UEL with element-local pressure condensation.

    This is a scoped research path. For a standard mixed ``u,p,mu``
    formulation, use ``generate_uel(..., formulation='standard')``.
    """
    from .generators.uel_local_pressure import (
        generate_uel_local_pressure as _f,
    )
    return _f(*args, **kwargs)


def generate_umat(*args, **kwargs):
    """Generate an Abaqus UMAT; imported lazily to keep core imports light."""
    from .generators.umat_gen import generate_umat as _generate_umat
    return _generate_umat(*args, **kwargs)


def generate_small_strain_umat(*args, **kwargs):
    """Generate a small-strain Abaqus UMAT."""
    from .generators.umat_gen import (
        generate_small_strain_umat as _generate_small_strain_umat,
    )
    return _generate_small_strain_umat(*args, **kwargs)


def generate_inp_scaffold(*args, **kwargs):
    """Generate Abaqus .inp include scaffold files for a UEL job."""
    from .generators.inp_scaffold import generate as _generate
    return _generate(*args, **kwargs)


def write_job_inp(*args, **kwargs):
    """Write a top-level Abaqus include driver for scaffolded UEL jobs."""
    from .generators.inp_scaffold import write_job_inp as _write_job_inp
    return _write_job_inp(*args, **kwargs)


def ensure_coupled_dummy_material(*args, **kwargs):
    """Add required density/specific-heat cards to coupled dummy materials."""
    from .generators.inp_scaffold import (
        ensure_coupled_dummy_material as _ensure_coupled_dummy_material,
    )
    return _ensure_coupled_dummy_material(*args, **kwargs)


# SymbolicTangent is a prototype/optimization path; importing it is
# behind a property-style accessor so a missing sympy install does not
# break ``import coupfe.codegen as au`` for users on the default path.
def __getattr__(name):
    if name == "SymbolicTangent":
        from .core.symbolic_tangent import SymbolicTangent
        return SymbolicTangent
    raise AttributeError(f"module 'coupfe.codegen' has no attribute {name!r}")


__all__ = [
    # materials and weak forms
    "Material", "SmallStrainMaterial", "WeakForm",
    "VectorField", "ScalarField", "LocalScalar",
    # tangent verification
    "VerificationError",
    # hand-written Fortran helper sidecars
    "fortran_helper",
    # tensor algebra and plasticity helpers (modules)
    "tensor", "small_strain_plasticity", "soil",
    # UEL/UMAT generators (UEL-focused; the shared engine in umat_gen is kept)
    "generate_uel", "generate_uel_local_pressure",
    "generate_umat", "generate_small_strain_umat",
    # element / mesh metadata
    "ElementConfig", "ELEMENT_CONFIGS",
    # Abaqus .inp scaffold
    "UELModelConfig", "ScaffoldReport",
    "generate_inp_scaffold", "write_job_inp",
    "ensure_coupled_dummy_material",
    # symbolic tangent prototype (lazy import via __getattr__)
    "SymbolicTangent",
]
