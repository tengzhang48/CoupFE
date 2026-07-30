"""Print the curved-boundary convergence table (with vs without re-embedding).

    python examples/curved_annulus/run.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from annulus import solve_level  # noqa: E402


def _table(reembed):
    print(f"\n  re-embed geometry = {reembed}")
    print(f"  {'level':>5} {'n_elem':>7} {'h':>10} {'rel L2 err':>14} {'rate':>6}")
    prev = None
    for level in range(4):
        h, err, ne = solve_level(level, reembed=reembed)
        rate = "" if prev is None else f"{np.log2(prev / err):6.2f}"
        print(f"  {level:>5} {ne:>7} {h:>10.4f} {err:>14.3e} {rate:>6}")
        prev = err


def main():
    print("Curved-annulus (Lamé-form) convergence — Quad4 expects rate ~2")
    _table(reembed=True)
    _table(reembed=False)
    print("\n  Note: for Quad4 + Dirichlet BC BOTH converge at ~h^2 — re-embedding is a")
    print("  GEOMETRIC improvement (boundary nodes exactly on the curve, see the mesh")
    print("  tests), not a solution-accuracy one here. Its solution payoff appears with")
    print("  higher-order geometry or curved-boundary loads (tractions).")


if __name__ == "__main__":
    main()
