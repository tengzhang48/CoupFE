"""Distributed-mesh M2 gate: curved-boundary convergence in an actual solve.

Independent oracle: the exact axisymmetric field ``u_r = a r + b/r`` on a thick
annulus sector. Refining the re-embedded curved mesh must drive the interior
relative-L2 error down at the Quad4 rate (~h²). The rate itself is the gate — a wrong
element/assembly/mesh would break it.

Honest scope: for Quad4 + Dirichlet BC, re-embedding is a *geometric* improvement
(boundary nodes exactly on the curve — see ``test_mesh.py``), not a solution-accuracy
one; the solve converges ~h² either way. So this gate checks the *convergence rate*,
not a re-embed-vs-not solution gap (there isn't a meaningful one for this case).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "curved_annulus"))

try:
    from annulus import solve_level
    solve_level(1)                       # triggers the f2py kernel build
    _HAVE = True
except Exception as exc:                 # pragma: no cover - toolchain-dependent
    _HAVE = False
    _WHY = str(exc)

pytestmark = pytest.mark.skipif(
    not _HAVE, reason="neo-Hookean kernel unavailable"
    + (f": {_WHY}" if not _HAVE else ""))


def test_curved_boundary_convergence_quad4():
    errs = [solve_level(L)[1] for L in (1, 2, 3)]
    # the error drops monotonically under refinement
    assert errs[1] < errs[0] and errs[2] < errs[1]
    # and substantially (a non-converging discretization would not)
    assert errs[0] > 4.0 * errs[2]
    # asymptotic Quad4 rate ~ 2 (averaged over the two fine steps)
    rate = np.log2(errs[0] / errs[2]) / 2.0
    assert 1.7 < rate < 2.4, f"convergence rate {rate:.2f} not ~2 (errs={errs})"
