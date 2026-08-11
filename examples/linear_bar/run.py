"""Run the nonlinear bar: fix the left end, ramp a tip load, solve with Newton.

    python examples/linear_bar/run.py
"""

from __future__ import annotations

import numpy as np

try:  # support both ``python run.py`` and package-style imports used by renderers
    from .bar import Bar1D
except ImportError:  # pragma: no cover - direct-script path
    from bar import Bar1D

from coupfe import newton_solve


def solve_bar(n_elem=10, L=2.0, E=100.0, H=40.0, f_tip=8.0):
    """Solve the retained bar and return the actual nodal snapshot."""

    bar = Bar1D(n_elem=n_elem, L=L, E=E, H=H, f_tip=f_tip)
    U0 = np.zeros(bar.ndof)
    U, _, nit = newton_solve([bar], U0, None, bar.ndof, {0: 0.0}, t=1.0)

    return {
        "configuration": {
            "n_elements": int(n_elem),
            "length": float(L),
            "youngs_modulus": float(E),
            "nonlinear_modulus": float(H),
            "tip_force": float(f_tip),
        },
        "nodes_reference": bar.x.copy(),
        "elements": np.column_stack(
            (np.arange(bar.n, dtype=int), np.arange(1, bar.n + 1, dtype=int))
        ),
        "displacement": U.copy(),
        "nodes_deformed": bar.x + U,
        "newton_iterations": int(nit),
        "tip_displacement": float(U[-1]),
        "linear_tip_reference": float(f_tip * L / E),
    }


def main():
    evidence = solve_bar()

    print(f"converged in {evidence['newton_iterations']} Newton iterations")
    print(f"tip displacement      = {evidence['tip_displacement']:.6f}")
    print(f"linear (H=0) reference = {evidence['linear_tip_reference']:.6f}  "
          f"(nonlinear softens/stiffens it)")


if __name__ == "__main__":
    main()
