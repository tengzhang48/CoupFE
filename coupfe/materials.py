"""Material specs for the model-setup pipeline.

A material is a thin spec that knows how to turn a mesh (region) into an
:class:`~coupfe.operators.element_group.ElementGroup` — i.e. which compiled element kernel
to build and its props/state layout. The kernel is built once (f2py) and cached. The
runtime is imported lazily so that importing this module needs no Fortran toolchain.
"""

from __future__ import annotations

import functools
import os

_ELEM_DIR = os.path.join(os.path.dirname(__file__), "runtime", "elements")


def neo_hookean_kernel_props(G, K):
    """Convert physical ``(G, K)`` to the retained kernels' raw ``(G, lambda)``.

    The retained core kernels under ``coupfe/runtime/elements`` evaluate
    ``P = G(F - F^-T) + lambda*ln(J)*F^-T``.  Consequently their second raw
    property is the first Lame coefficient, while the public material API uses
    the physical small-strain bulk modulus ``K = lambda + 2*G/3``.
    """
    G = float(G)
    K = float(K)
    return G, K - 2.0 * G / 3.0


@functools.lru_cache(maxsize=8)
def _build(for_path, module_name):
    from coupfe.runtime.compiled_element import build_element_kernel
    return build_element_kernel(for_path, module_name)


class NeoHookean:
    """Compressible neo-Hookean parameterized by physical ``G`` and ``K``.

    Here ``K`` is the small-strain bulk modulus.  The retained raw Fortran
    core kernel uses ``(G, lambda)`` in
    ``P = G(F - F⁻ᵀ) + lambda ln(J) F⁻ᵀ``; this wrapper performs the
    conversion before constructing the compiled element.

    Maps to the vendored CoupFE-native ``neo_hookean_q4_fbar_native.for``
    kernel, preserving the F-bar formulation used by this convenience material
    before the native backend was introduced.
    The parallel ``neo_hookean_q4.for`` Abaqus UEL is retained for export and
    backend-parity checks, not used as the default standalone runtime.
    """

    def __init__(self, G, K):
        self.props = neo_hookean_kernel_props(G, K)
        self.n_svars = 0
        self.mcrd = 2
        self._for = os.path.join(_ELEM_DIR, "neo_hookean_q4_fbar_native.for")
        self._module = "coupfe_pipeline_neo_q4"

    def element_group(self, view, elem_set, comps):
        from coupfe.operators.element_group import ElementGroup
        from coupfe.runtime.compiled_element import CompiledElement
        n_elem = (view.n_elem if elem_set is None
                  else int(len(view.elem_sets[elem_set])))
        elem = CompiledElement(_build(self._for, self._module), props=self.props,
                               dof_per_node=view.dof_per_node, n_svars=self.n_svars,
                               mcrd=self.mcrd, n_elem=n_elem)
        return ElementGroup.from_view(view, elem, comps=comps, elem_set=elem_set)
