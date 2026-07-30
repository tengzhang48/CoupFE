"""Run the nonlinear bar: fix the left end, ramp a tip load, solve with Newton.

    python examples/linear_bar/run.py
"""

from __future__ import annotations

import numpy as np

from bar import Bar1D

from coupfe import newton_solve


def main():
    bar = Bar1D(n_elem=10, L=2.0, E=100.0, H=40.0, f_tip=8.0)
    U0 = np.zeros(bar.ndof)
    U, _, nit = newton_solve([bar], U0, None, bar.ndof, {0: 0.0}, t=1.0)

    eps_lin = bar.f_tip / bar.E            # small-strain reference
    print(f"converged in {nit} Newton iterations")
    print(f"tip displacement      = {U[-1]:.6f}")
    print(f"linear (H=0) reference = {eps_lin * 2.0:.6f}  "
          f"(nonlinear softens/stiffens it)")


if __name__ == "__main__":
    main()
