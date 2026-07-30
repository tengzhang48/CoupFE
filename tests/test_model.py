"""The model-setup pipeline (P) — the declarative front door.

The same refined block-on-plane problem as ``test_pipeline.py``, but expressed through
``Model`` instead of hand-wiring the mesh/element/contact/BC/solve plumbing. This is the
contract the front door must honour: a handful of declarations → a converged, physical solve.
"""

import numpy as np
import pytest

from coupfe import Model, NeoHookean
from coupfe.operators.contact import HalfSpace

# The element group needs the f2py kernel; skip cleanly if the toolchain is absent.
try:
    NeoHookean(1.0, 10.0).element_group(
        Model.structured(1, 1).view, None, (0, 1))
    _HAVE = True
except Exception as exc:                                  # pragma: no cover
    _HAVE = False
    _WHY = str(exc)

pytestmark = pytest.mark.skipif(
    not _HAVE, reason="neo-Hookean kernel unavailable"
    + (f": {_WHY}" if not _HAVE else ""))


def test_model_block_on_plane():
    m = Model.structured(4, 4, 1.0, 1.0)
    m.refine(levels=1)                                    # 16 → 64 elements
    m.material("block", NeoHookean(G=1.0, K=10.0))
    m.fix("left", x=0.0)                                  # bbox face, auto-detected
    m.prescribe("top", y=-0.05)                           # driven down onto the plane
    m.contact("bottom", HalfSpace([0.0, -0.01], [0.0, 1.0]), k=1.0e4)

    res = m.solve(steps=4)

    assert res.converged                                 # ‖R_free‖ below tol
    assert res.iters < 60                                # load-stepped Newton, sane count
    ybot = res.position("bottom")[:, 1]
    assert ybot.min() > -0.01 - 2e-3                     # contact held — no penetration
    assert np.abs(res.U).max() > 0.01                    # the block actually deformed


def test_model_selectors_and_validation():
    """Bbox faces, explicit indices, and a predicate all resolve; missing material errors."""
    m = Model.structured(2, 2, 1.0, 1.0)                 # 3×3 nodes, midline at y=0.5
    assert set(m._resolve("left")) == {0, 3, 6}          # x == 0 column
    assert np.array_equal(m._resolve([1, 2, 3]), [1, 2, 3])
    midline = m._resolve(lambda X: abs(X[1] - 0.5) < 1e-9)
    assert set(midline) == {3, 4, 5} and np.allclose(m.view.nodes[midline, 1], 0.5)
    with pytest.raises(KeyError):
        m._resolve("nope")
    with pytest.raises(ValueError):
        m.solve()                                        # no material declared
