"""Gate the dual-multiplier semismooth-Newton PARTIAL-SLIP friction solver (coupfe.operators.
contact_semismooth, demonstrated in examples/semismooth_friction).

The defining properties: a stick zone and a slip zone coexist at converged static equilibrium; every
stick node has EXACTLY zero tangential slip (a constraint, not a regularized spring); every slip node sits
exactly on the Coulomb cone |p|=μN; the Schur-condensed interface solve returns the same answer as the
direct augmented solve; and with μ=0 nothing can stick (broken control)."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_RUN = _ROOT / "examples" / "semismooth_friction" / "run.py"

pytestmark = pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran unavailable")


def _load():
    spec = importlib.util.spec_from_file_location("ssn_run", _RUN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ssn_run"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def block():
    ex = _load()
    nodes, ndof, K = ex.build()                          # compiles the NeoHookean kernel once
    bot = np.nonzero(np.abs(nodes[:, 1] - nodes[:, 1].min()) < 1e-9)[0]
    bx = bot * 2
    return ex, nodes, ndof, K, bx


def test_example_runs_ok():
    env = dict(os.environ, PYTHONPATH=str(_ROOT))
    out = subprocess.run([sys.executable, str(_RUN)], cwd=str(_ROOT), env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "OK" in out.stdout and "FAIL" not in out.stdout, out.stdout + out.stderr


def test_efficient_solver_matches_reference_and_warmstarts(block):
    """The reusable SemismoothFrictionSolver (factor-once + interface-condensed) must reproduce the
    reference solve to solver precision, on a single load and warm-started along a load path."""
    from coupfe.operators.contact_semismooth import (SemismoothFrictionSolver,
                                                     solve_friction_semismooth)
    ex, nodes, ndof, K, bx = block
    bot = np.nonzero(np.abs(nodes[:, 1] - nodes[:, 1].min()) < 1e-9)[0]
    top = np.nonzero(np.abs(nodes[:, 1] - nodes[:, 1].max()) < 1e-9)[0]
    bxd, byd, txd, tyd = bot * 2, bot * 2 + 1, top * 2, top * 2 + 1
    f = np.zeros(ndof); f[tyd] = -ex.P / len(top)
    fixed = np.concatenate([txd, byd])
    solver = SemismoothFrictionSolver(K, fixed_dofs=fixed, contact_tan_dofs=bxd,
                                      normal_dofs=byd, mu=ex.MU)
    p = None
    for delta in (0.1, 0.3, 0.5):                          # a load path; warm-start p across steps
        vals = np.concatenate([np.full(len(txd), delta), np.zeros(len(byd))])
        ref = solve_friction_semismooth(K, f, fixed_dofs=fixed, fixed_vals=vals,
                                        contact_tan_dofs=bxd, normal_dofs=byd, mu=ex.MU, schur=True)
        out = solver.solve(f, vals, p0=p); p = out.p
        assert out.converged
        assert np.allclose(ref.U, out.U, atol=1e-7)
        assert np.array_equal(ref.stick, out.stick)


def test_partial_slip_exact_stick_and_cone(block):
    ex, nodes, ndof, K, bx = block
    res = ex.solve(nodes, ndof, K, 0.02)                 # press-dominated → central stick, edge slip
    assert res.converged
    assert res.stick.any() and (~res.stick).any()        # genuine partial slip
    assert np.max(np.abs(res.U[bx][res.stick])) < 1e-9   # exact zero-slip stick
    slip = ~res.stick
    assert np.allclose(np.abs(res.p[slip]), ex.MU * res.N[slip], atol=1e-6)  # slip nodes on the cone


def test_shear_regime_shifts_stick_zone(block):
    ex, nodes, ndof, K, bx = block
    res = ex.solve(nodes, ndof, K, 0.8)                  # shear-dominated
    assert res.converged and res.stick.any() and (~res.stick).any()
    assert np.max(np.abs(res.U[bx][res.stick])) < 1e-9


def test_schur_equals_direct(block):
    ex, nodes, ndof, K, bx = block
    a = ex.solve(nodes, ndof, K, 0.8, schur=False)
    b = ex.solve(nodes, ndof, K, 0.8, schur=True)
    assert np.allclose(a.U, b.U, atol=1e-7)
    assert np.allclose(a.p, b.p, atol=1e-7)


def test_mu_zero_nothing_sticks(block):
    """Broken control: no friction ⇒ no node can stick (cap=0 ⇒ all slip, p=0)."""
    ex, nodes, ndof, K, bx = block
    saved = ex.MU
    try:
        ex.MU = 0.0
        res = ex.solve(nodes, ndof, K, 0.05)
        assert res.converged and int(res.stick.sum()) == 0
        assert np.allclose(res.p, 0.0, atol=1e-9)
    finally:
        ex.MU = saved
