"""The CoupFE operator contract — the spine of the package.

Everything that contributes to the global nonlinear system is an **Operator**
with one small contract: a bulk element group, a contact set, a constraint, a
load.  An operator is functionally **pure** (same inputs → same outputs) and it
**never commits state implicitly** — a candidate ``state_trial`` is returned and
only committed after the global solve accepts a step.  That discipline is what
makes complex-step tangent columns, line searches, and (later) matrix-free
products correct; see ``docs/DESIGN.md``.

The contract is deliberately at **group granularity** (a batch of like
contributions sharing one kernel), matching how the compiled f2py element kernels
are actually called — one coarse batched call, not a Python loop over elements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class Residual:
    """An operator's residual contribution to the global vector.

    ``values`` are summed into the global residual at ``gdofs`` (duplicate global
    indices are summed, COO-style).  ``state_trial`` is a candidate state and is
    **never** committed by producing it.
    """

    gdofs: np.ndarray
    values: np.ndarray
    state_trial: Any = None


@dataclass
class Tangent:
    """An operator's assembled local tangent in COO triplets.

    ``(rows, cols, values)`` are summed into the global sparse matrix.  Indices
    are global DOFs (the same space as :class:`Residual.gdofs`).
    """

    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    state_trial: Any = None


@runtime_checkable
class Operator(Protocol):
    """A residual-contributing operator over a group of like contributions."""

    def residual(self, U: np.ndarray, state: Any, t: float, dt: float) -> Residual:
        """Residual contribution at solution ``U`` from committed ``state``."""
        ...

    def tangent(self, U: np.ndarray, state: Any, t: float, dt: float) -> Tangent:
        """Assembled local tangent (analytic or complex-step) at ``U``."""
        ...

    def commit(self, U: np.ndarray, state: Any, t: float, dt: float) -> Any:
        """After the global solve accepts ``U``: recompute and return the new
        committed state.  Stateless operators return ``state`` unchanged."""
        ...


def complex_step_tangent(residual_fn, ue: np.ndarray, h: float = 1e-30) -> np.ndarray:
    """Dense local tangent of a pure element residual by complex step.

    ``residual_fn(ue) -> r`` must accept a complex ``ue`` and be analytic (no
    ``abs``/``max``/branches-on-value).  Returns ``dr/due`` exactly (no
    subtraction cancellation).  This is the CoupFE pattern: **one residual is the
    source of truth; the tangent is a derived transformation of it.**
    """
    ue = np.asarray(ue, dtype=complex)
    n = ue.shape[0]
    K = np.empty((n, n))
    for j in range(n):
        up = ue.copy()
        up[j] += 1j * h
        K[:, j] = residual_fn(up).imag / h
    return K
