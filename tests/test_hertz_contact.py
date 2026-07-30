"""Hertz analytic contact benchmark — the quantitative normal-contact gate.

A rigid sphere indenting a deformable block must reproduce Hertz:
``F = (4/3)E*√R δ^{3/2}``.  The mesh-insensitive signature is the **exponent
3/2** (gated tightly); the prefactor is gated in a loose band (a coarse,
finite-depth block is stiffer than a true half-space, biasing F above Hertz).
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

_RUN = Path(__file__).parent.parent / "examples" / "hertz_contact" / "run.py"


def _load():
    spec = importlib.util.spec_from_file_location("hertz_run", _RUN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="needs gfortran")
def test_hertz_force_law():
    ex = _load()
    deltas, F_fe, _ = ex.solve_hertz(deltas=(0.03, 0.05, 0.07))
    slope, ratios = ex.analyze(deltas, F_fe)

    # exponent 3/2 — the mesh-insensitive Hertz signature (tight)
    assert abs(slope - 1.5) < 0.15, f"log-log slope {slope:.3f} not ≈ 3/2"
    # prefactor — loose band (coarse mesh + finite-depth over-stiffening biases F up)
    assert 0.7 < np.median(ratios) < 1.6, f"F_FE/F_Hertz median {np.median(ratios):.2f} off"
    # FE should be ON THE STIFF side of Hertz, not soft (the physical bias direction)
    assert np.median(ratios) > 1.0, "FE softer than Hertz — wrong bias (suspect a bug)"
