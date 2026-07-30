"""Shared guards for the validation harness."""

from __future__ import annotations

import numpy as np

__all__ = ["require_finite"]


def require_finite(name, **arrays):
    """Raise ``AssertionError`` if any input has a NaN/Inf entry.

    A diverged or ill-posed state (NaN/Inf stress, tangent, or deformation) is
    itself a defect a validation gate must REJECT — never silently pass.
    Without this guard the downstream comparisons use NaN, and ``NaN > tol`` is
    ``False``, so the assertion would not fire (a false pass).
    """
    for key, a in arrays.items():
        arr = np.asarray(a, dtype=float)
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.count_nonzero(~np.isfinite(arr)))
            raise AssertionError(
                f"{name}: input '{key}' has {n_bad} non-finite value(s) "
                f"(NaN/Inf) — a diverged/ill-posed state. A validation gate "
                f"must reject this, not silently pass.")
