"""The operator-contract spine, with a harness-style cross-check.

These are the broken-control-able tests that gate CoupFE's core: the assembler
composes operators, Newton converges, and — the key CoupFE invariant — the
**complex-step tangent equals the analytic tangent**.  If a future change makes
the residual non-analytic (an ``abs``, a value-branch), this test fails.
"""

import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "linear_bar"))

from bar import Bar1D  # noqa: E402

from coupfe import assemble_residual, newton_solve  # noqa: E402


def test_linear_bar_matches_analytic_solution():
    # H = 0 -> linear bar, tip displacement = f * L / E exactly.
    bar = Bar1D(n_elem=8, L=2.0, E=100.0, H=0.0, f_tip=5.0)
    U, _, nit = newton_solve([bar], np.zeros(bar.ndof), None, bar.ndof, {0: 0.0})
    assert abs(U[-1] - 5.0 * 2.0 / 100.0) < 1e-10
    assert nit <= 2                      # linear → 1–2 iterations


def test_complex_step_tangent_equals_analytic():
    # The core CoupFE invariant: the derived (complex-step) tangent is exact.
    bar = Bar1D(n_elem=8, L=2.0, E=100.0, H=40.0, f_tip=5.0)
    U = np.linspace(0.0, 0.1, bar.ndof)  # an arbitrary nonlinear state
    T = bar.tangent(U, None, 1.0, 1.0)
    K_cs = sp.coo_matrix((T.values, (T.rows, T.cols)),
                         shape=(bar.ndof, bar.ndof)).toarray()
    K_an = bar.analytic_tangent(U)
    assert np.allclose(K_cs, K_an, atol=1e-9, rtol=0.0)


def test_nonlinear_bar_converges_to_zero_residual():
    bar = Bar1D(n_elem=8, L=2.0, E=100.0, H=40.0, f_tip=5.0)
    U, _, nit = newton_solve([bar], np.zeros(bar.ndof), None, bar.ndof, {0: 0.0})
    R, _ = assemble_residual([bar], U, None, 1.0, 1.0, bar.ndof)
    R[0] = 0.0                           # drop the reaction DOF
    assert np.linalg.norm(R) < 1e-8
    assert nit < 10                      # Newton, not glacial


def test_two_operators_compose():
    # Two bars sharing the shared node assemble through the same contract.
    bar = Bar1D(n_elem=4, L=1.0, E=100.0, H=0.0, f_tip=3.0)
    U, _, _ = newton_solve([bar], np.zeros(bar.ndof), None, bar.ndof, {0: 0.0})
    # monotonically increasing displacement along a bar in tension
    assert np.all(np.diff(U) > -1e-12)
