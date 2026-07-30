"""A bulk element group as a CoupFE :class:`~coupfe.operators.base.Operator`.

An :class:`ElementGroup` is one element *type* driven over its own element list: it
holds a compiled kernel (:class:`~coupfe.runtime.compiled_element.CompiledElement`)
and the mesh subset, and turns the kernel's one batched ``(R, K)`` call into a global
``Residual`` / ``Tangent`` in COO form.  It owns no physics — the physics lives in the
kernel's residual; the group only gathers element DOFs and scatters the result.  Mixed
multi-material meshes compose by listing several groups (each writing its own
components) into the same ``newton_solve`` call.

Global DOF layout (the multi-material pattern, mirroring the lab's ``assemble_groups``):
the global system keeps a **uniform** ``dof_per_node`` (e.g. 3 = u_x, u_y, mu).  Each
group declares which per-node **components** its element touches via ``comps`` — a
u-only rubber element uses ``comps=(0, 1)`` and never writes the mu rows/cols; a u-mu
gel element uses ``comps=(0, 1, 2)``.  A node's global DOF is ``node*dof_per_node +
comp``.  A component that no group provides an equation for carries no row and must be
pinned (Dirichlet) by the caller, exactly as Abaqus treats an inert temperature DOF.

State protocol (see ``docs/DESIGN.md`` / ``skills/pitfalls.md``): every residual/
tangent evaluation starts from the **committed** state — here the step-start
displacement ``U_prev`` (so the kernel forms ``DU = U - U_prev`` for rate/history
terms) and the kernel's committed ``svars``.  Producing a residual/tangent never
mutates committed state; the candidate is returned in ``state_trial`` and the internal
``svars`` are written **only** by :meth:`commit`, after the global solve accepts ``U``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from coupfe.operators.base import Residual, Tangent
from coupfe.runtime.compiled_element import CompiledElement


@dataclass
class GroupState:
    """Committed state an :class:`ElementGroup` reads each evaluation.

    ``U_prev`` is the step-start global displacement vector (so the element sees
    ``DU = U - U_prev``); ``None`` ⇒ start of analysis (``DU = U``).  The element's
    internal state variables live in the :class:`CompiledElement` and are committed
    transactionally by :meth:`ElementGroup.commit`.
    """

    U_prev: Optional[np.ndarray] = None


class ElementGroup:
    """One element type over its element list, as an Operator.

    Parameters
    ----------
    element : CompiledElement
        The compiled kernel; supplies ``element_rk_batch`` and ``commit_group``.
    nodes : (Nnode, ndim) array
        Reference coordinates (total-Lagrangian; fixed).
    elems : (Nelem, nne) int array
        Element connectivity (node indices).
    dof_per_node : int
        The **uniform global** DOFs per node (the whole system's layout).
    comps : sequence of int, optional
        Per-node component indices this element writes.  Defaults to
        ``range(dof_per_node)`` (a single-field group filling every component).
    """

    def __init__(self, element: CompiledElement, nodes, elems, dof_per_node,
                 comps: Optional[Sequence[int]] = None, *, fuse_rk=None):
        self.element = element
        self.nodes = np.asarray(nodes, dtype=float)
        self.elems = np.asarray(elems, dtype=int)
        self.dof_per_node = int(dof_per_node)
        # R/K fusion (see the _rk docstring): the kernel returns R and K
        # together, and Newton assembles residual then tangent at the SAME U,
        # so caching the pair runs the kernel once per iterate instead of
        # twice.  Default ON; historical local speedups are not release
        # benchmark evidence. ``fuse_rk=False`` (or env COUPFE_FUSE_RK=0)
        # restores independent per-call evaluation for
        # schemes that mutate props/state BETWEEN a residual and its paired
        # tangent, matrix-free/residual-only loops, or debugging.
        self.fuse_rk = (os.environ.get("COUPFE_FUSE_RK", "1") != "0"
                        if fuse_rk is None else bool(fuse_rk))
        self._rk_cache = None
        self.comps = (np.arange(self.dof_per_node, dtype=int) if comps is None
                      else np.asarray(comps, dtype=int))
        self.nelem, self.nne = self.elems.shape
        self.ndofel = self.nne * len(self.comps)
        # Global-DOF map (nelem, ndofel): node-major to match the element's
        # [n0c0, n0c1, ..., n1c0, ...] ordering.  gdof = node*dof_per_node + comp.
        self.gm = (self.elems[:, :, None] * self.dof_per_node
                   + self.comps[None, None, :]).reshape(self.nelem, self.ndofel)
        # Precompute the COO index arrays for the local tangent blocks.
        self._rows = np.broadcast_to(
            self.gm[:, :, None], (self.nelem, self.ndofel, self.ndofel)).ravel()
        self._cols = np.broadcast_to(
            self.gm[:, None, :], (self.nelem, self.ndofel, self.ndofel)).ravel()

    @classmethod
    def from_view(cls, view, element, comps=None, elem_set=None):
        """Build an :class:`ElementGroup` over a ``KernelMeshView``.

        ``elem_set`` selects a named element subset (a material region) for the
        multi-material pattern; otherwise the whole mesh. The operator sees only the
        view's compact arrays — never a mesh-backend object.
        """
        elems = (view.elems if elem_set is None
                 else view.elems[view.elem_sets[elem_set]])
        return cls(element, view.nodes, elems, view.dof_per_node, comps)

    # -- helpers -------------------------------------------------------- #
    def _coords(self):
        return self.nodes[self.elems]                      # (nelem, nne, ndim)

    def _gather(self, U, state):
        """Element-local ``(U_g, DU_g)`` for this group's components."""
        U = np.asarray(U, dtype=float)
        U_prev = None if state is None else getattr(state, "U_prev", None)
        DU = U if U_prev is None else U - np.asarray(U_prev, dtype=float)
        return U[self.gm], DU[self.gm]

    # -- Operator contract --------------------------------------------- #
    def _rk(self, U, state):
        """Evaluate the kernel once, returning ``(R_all, K_all)``.

        ``element_rk_batch`` computes BOTH the residual and the complex-step
        tangent in one pass (K is nearly free once R is formed), and Newton
        assembles the residual then the tangent at the SAME ``U`` — so without
        fusion the kernel runs TWICE per iteration and half the work is
        discarded (measured ~40% of a compiled-element solve's runtime).

        When ``fuse_rk`` is on, cache the pair so the paired residual/tangent
        share one evaluation.  The key is ``(U_g, DU_g, props)`` — it must
        include ``props`` because callers legitimately mutate them between
        steps (e.g. an active-stress ``Ta`` per time step); ``commit`` clears
        the cache so committed-state changes can never serve stale K.  The
        line-search trials at ``U + alpha*dU`` change ``U_g`` and correctly
        miss.  ``fuse_rk=False`` skips the cache entirely (independent
        per-call evaluation).
        """
        U_g, DU_g = self._gather(U, state)
        if not self.fuse_rk:
            return self.element.element_rk_batch(self._coords(), U_g, DU_g)
        props = getattr(self.element, "props", None)
        key = (U_g.tobytes(),
               None if DU_g is U_g else DU_g.tobytes(),
               None if props is None else np.asarray(props).tobytes())
        cached = self._rk_cache
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        R_all, K_all = self.element.element_rk_batch(self._coords(), U_g, DU_g)
        self._rk_cache = (key, R_all, K_all)
        return R_all, K_all

    def residual(self, U, state, t, dt) -> Residual:
        """Group residual contribution, scattered to global DOFs ``gm``."""
        R_all, _K_all = self._rk(U, state)
        return Residual(
            gdofs=self.gm.ravel(),
            values=np.asarray(R_all, dtype=float).ravel(),
            state_trial=GroupState(U_prev=None if state is None
                                   else getattr(state, "U_prev", None)),
        )

    def tangent(self, U, state, t, dt) -> Tangent:
        """Group tangent in COO triplets (kernel's complex-step ``dR/dU``)."""
        _R_all, K_all = self._rk(U, state)
        return Tangent(
            rows=self._rows,
            cols=self._cols,
            values=np.asarray(K_all, dtype=float).ravel(),
        )

    def commit(self, U, state, t, dt) -> GroupState:
        """After the solve accepts ``U``: commit the kernel's trial state and
        return the new committed state (the just-accepted ``U`` becomes the next
        step's ``U_prev``)."""
        self.element.commit()
        self._rk_cache = None          # committed state changed -> invalidate
        return GroupState(U_prev=np.asarray(U, dtype=float).copy())
