"""CoupFE — a small, validated finite-element scaffold.

One element definition → a complex-step kernel that runs both inside Abaqus (as a
UEL/UMAT) and standalone here.  The core is the operator contract; meshing, BCs,
loading, and time integration are thin, AI-writable glue validated by the harness.

See ``docs/DESIGN.md`` for the architecture and ``docs/roadmap.md`` for the plan.
"""

from coupfe.operators.base import (
    Operator,
    Residual,
    Tangent,
    complex_step_tangent,
)
from coupfe.operators.element_group import ElementGroup, GroupState
from coupfe.operators.inertia import InertiaOperator
from coupfe.assembly.assemble import (
    assemble_residual,
    assemble_tangent,
    newton_solve,
    solve_increments,
    solve_dynamics,
    solve_dynamics_adaptive,
)
from coupfe.mesh import (
    Circle,
    KernelMeshView,
    Plane,
    Sphere,
    check_positive_jacobian,
    uniform_refine_quad,
)
from coupfe.constraints import (
    ConstraintRelation,
    ConstraintTransform,
    compile_affine_constraints,
)
from coupfe.model import Model, Result
from coupfe.materials import NeoHookean

# The compiled-element runtime needs numpy.f2py + a Fortran compiler at *build*
# time (to compile an element .for once).  It is imported lazily so the pure-Python
# operator contract works without the toolchain installed.
try:  # pragma: no cover - environment-dependent
    from coupfe.runtime.compiled_element import (
        CompiledElement,
        build_element_kernel,
    )
    _HAVE_RUNTIME = True
except Exception:  # noqa: BLE001 - any import failure ⇒ runtime simply unavailable
    CompiledElement = None  # type: ignore
    build_element_kernel = None  # type: ignore
    _HAVE_RUNTIME = False

__version__ = "0.0.1"

__all__ = [
    "Operator",
    "Residual",
    "Tangent",
    "complex_step_tangent",
    "ElementGroup",
    "GroupState",
    "CompiledElement",
    "build_element_kernel",
    "assemble_residual",
    "assemble_tangent",
    "newton_solve",
    "solve_increments",
    "solve_dynamics",
    "solve_dynamics_adaptive",
    "InertiaOperator",
    "KernelMeshView",
    "Circle",
    "Sphere",
    "Plane",
    "uniform_refine_quad",
    "check_positive_jacobian",
    "ConstraintRelation",
    "ConstraintTransform",
    "compile_affine_constraints",
    "Model",
    "Result",
    "NeoHookean",
]
