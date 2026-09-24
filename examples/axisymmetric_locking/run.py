"""Volumetric locking of generated axisymmetric elements in thick-shell inflation.

A neo-Hookean spherical shell (A = 1, B = 2, G = 1) is inflated to p = 0.6 G
with every generated element/formulation pair, at K/G = 100 (nu ~ 0.495) and
K/G = 10 000 (nu ~ 0.49995). The quantity of interest is the inner-surface
displacement a - A, compared with a spherically symmetric reference solve
(K/G = 100) or the closed-form incompressible relation (K/G = 10 000, whose
compressibility correction is O(G/K) ~ 1e-4).

Kernels are generated from the ``kernels.py`` declarations, compiled once with
f2py and cached (``COUPFE_AXI_KERNEL_CACHE`` selects the cache directory).

    PYTHONPATH=. python examples/axisymmetric_locking/run.py
"""

from __future__ import annotations

import os
import sys

from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shell import bvp_inner_radius, incompressible_pressure, solve_inflation  # noqa: E402

CASES = [("quad4", "standard"), ("quad4", "fbar"), ("tri3", "standard"),
         ("quad8", "standard"), ("quad8r", "standard"), ("quad8", "mixed"),
         ("tri6", "standard"), ("tri6", "mixed")]
G, A, B, P = 1.0, 1.0, 2.0, 0.6


def main():
    print(f"Thick-shell inflation, A={A}, B={B}, p={P} G, 6 x 8 cells (triangles: 2 per cell)\n")
    print("| K/G | element | formulation | a/A | error in a - A | Newton iterations |")
    print("|---:|---|---|---:|---:|---:|")
    for ratio in (1.0e2, 1.0e4):
        props = (0.5 * G, 0.0, 0.0, ratio * G)
        if ratio <= 1.0e3:
            a_ref = bvp_inner_radius(props, A, B, P)
        else:
            a_ref = brentq(lambda a: incompressible_pressure(G, A, B, a) - P, A + 1e-9, 1.8 * A)
        for element, formulation in CASES:
            try:
                out = solve_inflation(element, formulation, props, P, A=A, B=B)
            except RuntimeError as exc:   # a strongly locked element can leave Newton's basin
                print(f"| {ratio:g} | {element} | {formulation} | — | not converged ({exc}) | |", flush=True)
                continue
            err = (out["a"] - A) / (a_ref - A) - 1.0
            print(f"| {ratio:g} | {element} | {formulation} | {out['a'] / A:.5f} | {100 * err:+.2f}% "
                  f"| {out['iterations']} |", flush=True)
        print(f"| {ratio:g} | reference | {'BVP' if ratio <= 1e3 else 'incompressible'} | "
              f"{a_ref / A:.5f} | | |")


if __name__ == "__main__":
    main()
