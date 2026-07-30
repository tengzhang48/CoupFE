"""RESEARCH friction-threshold sweep inspired by ppf-contact-solver.

ppf-contact-solver's `friction.ipynb` slides an armadillo down a slope of angle
θ = arctan(0.5) (so tan θ = 0.5) with friction μ = 0.51. A historical
headless run was not retained with its environment and output, so it is not
release evidence.

Here we exercise the related CoupFE barrier on a soft Hex8 box on a rigid floor,
sweeping μ across the rigid-block analytic
critical value `μ_crit = tan θ = 0.5`. The observable is the contact-interface slip
(bottom-face x-displacement) normalized by a bracket — frictionless (pure slide) to
μ = 2 (full stick = the elastic-shear floor of the soft box).

Run this to inspect monotonicity, non-penetration, and the effect of
`friction_eps`; retain both environments and outputs before making an upstream
comparison. Soft-box elasticity, damped relaxation, and smoothing mean this is
not a quantitative rigid-block threshold benchmark.

    PYTHONPATH=. python examples/contact_vs_ppf/friction_threshold_sweep.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent


def _load_box():
    spec = importlib.util.spec_from_file_location(
        "cbf", _DIR / "coupfe_box_on_floor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sweep(friction_eps=2.0e-4, mus=(0.30, 0.40, 0.49, 0.51, 0.60)):
    cbf = _load_box()
    cbf.FRICTION_EPS = float(friction_eps)
    theta = np.degrees(np.arctan(0.5))                 # ppf's slope: tan θ = 0.5
    slide = abs(cbf.run(theta, 0.0)[0])                # frictionless = pure slide
    stick = abs(cbf.run(theta, 2.0)[0])                # μ=2 = full stick (shear floor)
    span = max(slide - stick, 1e-12)
    print(f"θ={theta:.2f}°  tanθ=0.50=μ_crit(analytic)   friction_eps={friction_eps:.0e}")
    print(f"bracket: frictionless={slide:.3f}  full-stick={stick:.3f}  "
          "(historical ppf notebook setting: μ=0.51)")
    out = {}
    for mu in mus:
        slip = abs(cbf.run(theta, mu)[0])
        norm = (slip - stick) / span
        out[mu] = norm
        print(f"  μ={mu:.2f}  normalized slip={norm:.2f}  -> {'slide' if norm > 0.3 else 'STICK'}")
    return out


if __name__ == "__main__":
    sweep()
