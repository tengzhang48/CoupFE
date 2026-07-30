"""Gate the two dual-multiplier-friction studies:
  * the RELAY — differentiable friction-field identifiability via the frozen-active-set adjoint
    (examples/friction_identifiability); needs gfortran (NeoHookean bulk K);
  * FINITE SLIDING — friction as interface plasticity over a re-pairing path with ε_p-style state
    transfer (examples/finite_sliding_friction); pure numpy, no gfortran.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent


def _run(rel, timeout):
    env = dict(os.environ, PYTHONPATH=str(_ROOT))
    out = subprocess.run([sys.executable, str(_ROOT / rel)], cwd=str(_ROOT), env=env,
                         capture_output=True, text=True, timeout=timeout)
    assert out.returncode == 0 and "OK" in out.stdout and "FAIL" not in out.stdout, out.stdout + out.stderr


def test_finite_sliding_friction_runs():
    _run("examples/finite_sliding_friction/run.py", timeout=120)


def test_finite_sliding_state_transfer_is_necessary():
    """Direct property gate (no gfortran): large slip integrates to μN·L, and dropping the state transfer
    makes the friction force collapse at edge crossings (the spurious re-stick)."""
    spec = importlib.util.spec_from_file_location(
        "fsf", _ROOT / "examples" / "finite_sliding_friction" / "run.py")
    fsf = importlib.util.module_from_spec(spec); spec.loader.exec_module(fsf)
    path = np.linspace(0.0, fsf.L, 8000)
    diss_T, _, ft_T = fsf.drag(path, transfer=True)
    diss_NT, _, ft_NT = fsf.drag(path, transfer=False)
    s = len(path) // 4
    assert abs(diss_T - fsf.CAP * fsf.L) / (fsf.CAP * fsf.L) < 0.02      # large slip → μN·L
    assert np.min(np.abs(ft_T[s:])) > 0.95 * fsf.CAP                     # WITH transfer: on the cone
    assert np.min(np.abs(ft_NT[s:])) < 0.3 * fsf.CAP                     # WITHOUT: collapses at crossings


def test_finite_sliding_capstan_runs():
    _run("examples/finite_sliding_capstan/run.py", timeout=120)


def test_capstan_matches_analytic():
    """Finite-sliding friction on a CURVE integrates to the capstan e^{μθ}, and a frozen frame is wrong."""
    spec = importlib.util.spec_from_file_location(
        "cap", _ROOT / "examples" / "finite_sliding_capstan" / "run.py")
    cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)
    for theta_deg, mu in ((90, 0.3), (180, 0.4), (270, 0.25)):
        theta = np.radians(theta_deg)
        ratio, frozen_normal = cap.capstan(theta, mu)
        assert abs(ratio - np.exp(mu * theta)) / np.exp(mu * theta) < 0.02   # reproduces e^{μθ}
        assert frozen_normal > 0.7                                            # frozen frame ⇒ huge normal error


def test_relay_identifiability_runs():
    if shutil.which("gfortran") is None:
        import pytest
        pytest.skip("gfortran unavailable")
    _run("examples/friction_identifiability/run.py", timeout=300)
