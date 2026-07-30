"""Gate the serial 3D contact examples (run them, assert they self-report OK).

Each compiles the F-bar Hex8 kernel and runs a short dynamics solve through the
3D contact stack; numba/LBVH acceleration is optional and the NumPy fallback is
part of the supported path. The tests skip without gfortran.
"""
import os
import shutil
import subprocess
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
pytestmark = pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran unavailable")


def _run_example(rel_path, timeout):
    env = dict(os.environ, PYTHONPATH=_ROOT)
    out = subprocess.run([sys.executable, rel_path], cwd=_ROOT, env=env,
                         capture_output=True, text=True, timeout=timeout)
    assert out.returncode == 0 and "OK" in out.stdout and "FAIL" not in out.stdout, out.stdout + out.stderr


def test_serial_3d_collision_example():
    """Two F-bar Hex8 blocks collide (v0), penetration-free — the approachable serial companion to the
    distributed 3D collision demo. Exercises ElementGroup + InertiaOperator +
    DeformableBarrierContact3D through serial solve_dynamics."""
    _run_example("examples/contact_3d_blocks/run.py", timeout=400)


def test_serial_3d_friction_example():
    """Two F-bar Hex8 blocks, gravity-seated + the top sheared; ppf smoothed friction holds (μ>0 slip <
    frictionless), penetration-free. The serial companion to the distributed 3D friction demo."""
    _run_example("examples/contact_3d_friction/run.py", timeout=500)
