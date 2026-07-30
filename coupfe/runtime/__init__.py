"""CoupFE compiled-element runtime — f2py kernels for the fast assembly path.

A single residual definition is emitted once as a self-contained Fortran ``.for``
(consistent tangent derived inside the kernel by complex step) and compiled here.
``build_element_kernel`` builds the importable module; ``CompiledElement`` drives it
in one batched call per group.  CoupFE never imports the research lab at runtime.
"""

from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

__all__ = ["CompiledElement", "build_element_kernel"]
