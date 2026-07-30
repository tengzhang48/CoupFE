"""CoupFE-side behavior gate used by the ppf interoperability recipe.

This checks CoupFE alone: a box on a rigid floor under tilted gravity is
(1) penetration-free, (2) held by friction below the slip threshold, and
(3) slides when frictionless. It neither executes ppf nor supplies an
independent physics comparison; see `examples/contact_vs_ppf/README.md`.
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

_EX = Path(__file__).parent.parent / "examples" / "contact_vs_ppf" / "coupfe_box_on_floor.py"


def _load():
    spec = importlib.util.spec_from_file_location("ppf_xcheck", _EX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="needs gfortran")
def test_box_on_floor_friction_vs_frictionless():
    ex = _load()
    slip_stick, gap_stick = ex.run(20.0, 2.0, n_steps=100)   # μ=2.0 ≫ tan20 → stick
    slip_slide, gap_slide = ex.run(20.0, 0.0, n_steps=100)   # frictionless → slide

    # (1) non-penetration in BOTH configs (the core contact guarantee)
    assert gap_stick > 0.0, f"penetration with friction: gap={gap_stick}"
    assert gap_slide > 0.0, f"penetration frictionless: gap={gap_slide}"

    # (3) frictionless actually slides (else (2) is vacuous — the broken control)
    assert abs(slip_slide) > 0.1, f"frictionless did not slide: {slip_slide}"

    # (2) friction holds: interface slip ≪ the frictionless slip
    ratio = abs(slip_stick) / abs(slip_slide)
    assert ratio < 0.3, f"friction did not hold: stick/slide={ratio:.2f}"
