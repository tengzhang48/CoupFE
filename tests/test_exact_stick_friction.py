"""Gate the dual-multiplier EXACT-STICK friction example (examples/exact_stick_friction).

Two layers: (1) run the example as a subprocess and assert it self-reports OK; (2) drive the active-set
interface directly and assert the *defining* properties of the dual-multiplier treatment that the
smoothed/return-map forms cannot deliver — interface slip that is EXACTLY zero in stick (a constraint,
not ~F/k_t), the Coulomb force locked at μN to machine precision once sliding, and the stick→slip onset
at the analytic incipient shear. A broken control (zeroed friction) must lose the cap."""
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
_RUN = _ROOT / "examples" / "exact_stick_friction" / "run.py"

pytestmark = pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran unavailable")


def _load():
    spec = importlib.util.spec_from_file_location("exact_stick_run", _RUN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exact_stick_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_example_runs_ok():
    env = dict(os.environ, PYTHONPATH=str(_ROOT))
    out = subprocess.run([sys.executable, str(_RUN)], cwd=str(_ROOT), env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "OK" in out.stdout and "FAIL" not in out.stdout, out.stdout + out.stderr


def test_exact_zero_slip_stick_and_coulomb_cap():
    ex = _load()
    nodes, ndof, K = ex.build_stiffness()
    iface = ex.ExactStickInterface(nodes, ndof, K)
    dstar = ex.MU * ex.P / iface.shear_stiffness()

    # Below the incipient shear: stick, interface slip EXACTLY zero, force strictly below the cone.
    for ratio in (0.25, 0.5, 0.95):
        regime, vt, Ff = iface.step(ratio * dstar)
        assert regime == "stick"
        assert vt == 0.0                                   # a constraint v_t=0, not a regularized creep
        assert Ff < ex.MU * ex.P
        assert Ff == pytest.approx(ratio * ex.MU * ex.P, rel=1e-6)  # linear elastic stick branch

    # Above it: slip, force locked on the Coulomb cone to machine precision, interface sliding.
    for ratio in (1.05, 1.5, 2.0):
        regime, vt, Ff = iface.step(ratio * dstar)
        assert regime == "slip"
        assert vt > 0.0
        assert Ff == pytest.approx(ex.MU * ex.P, abs=1e-12)
        assert iface.last_friction_work < 0.0

    # The closed cone includes its boundary.  A few ULPs of sparse-solver
    # roundoff must not make the active-set state Python/SciPy dependent.
    regime, vt, Ff = iface.step(dstar)
    assert Ff == pytest.approx(ex.MU * ex.P, abs=1e-12)
    assert regime == "stick" and vt == 0.0

    # The roundoff guard must not blur a genuinely supercritical load.
    regime, vt, Ff = iface.step((1.0 + 1.0e-10) * dstar)
    assert regime == "slip" and vt > 0.0
    assert Ff == pytest.approx(ex.MU * ex.P, abs=1e-12)


def test_broken_control_zero_friction_loses_cap():
    """With μ→0 the cone collapses: the interface can never stick under any shear (no cap)."""
    ex = _load()
    nodes, ndof, K = ex.build_stiffness()
    iface = ex.ExactStickInterface(nodes, ndof, K)
    saved = ex.MU
    try:
        ex.MU = 0.0
        regime, vt, Ff = iface.step(1e-3)                  # tiny shear
        assert regime == "slip" and vt > 0.0               # cannot stick — no friction to hold it
    finally:
        ex.MU = saved


def test_broken_control_undercapped_slip_is_observable():
    """The reported slip force must come from equilibrium, not echo ``mu*P``."""
    ex = _load()

    class UnderCappedInterface(ex.ExactStickInterface):
        def _slip_force_distribution(self, stick_reaction, cap):
            return 0.5 * super()._slip_force_distribution(
                stick_reaction, cap
            )

    nodes, ndof, K = ex.build_stiffness()
    iface = UnderCappedInterface(nodes, ndof, K)
    dstar = ex.MU * ex.P / iface.shear_stiffness()
    regime, vt, friction_reaction = iface.step(1.5 * dstar)

    assert regime == "slip" and vt > 0.0
    assert friction_reaction == pytest.approx(0.5 * ex.MU * ex.P, abs=1.0e-12)
    assert abs(friction_reaction - ex.MU * ex.P) > 0.25 * ex.MU * ex.P


def test_broken_control_reversed_friction_does_positive_work():
    """A correct force magnitude cannot conceal a reversed friction direction."""
    ex = _load()

    class ReversedInterface(ex.ExactStickInterface):
        def _slip_force_distribution(self, stick_reaction, cap):
            return -super()._slip_force_distribution(
                stick_reaction, cap
            )

    nodes, ndof, K = ex.build_stiffness()
    iface = ReversedInterface(nodes, ndof, K)
    dstar = ex.MU * ex.P / iface.shear_stiffness()
    with pytest.raises(RuntimeError, match="positive work"):
        iface.step(1.5 * dstar)
