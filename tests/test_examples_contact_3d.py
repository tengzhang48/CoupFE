"""Gate the serial 3D contact examples (run them, assert they self-report OK).

Each compiles the F-bar Hex8 kernel and runs a short dynamics solve through the
3D contact stack; numba/LBVH acceleration is optional and the NumPy fallback is
part of the supported path. The tests skip without gfortran.
"""
import os
import importlib.util
import shutil
import subprocess
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
pytestmark = pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran unavailable")


def _run_example(rel_path, timeout):
    env = dict(os.environ, PYTHONPATH=_ROOT)
    out = subprocess.run([sys.executable, rel_path], cwd=_ROOT, env=env,
                         capture_output=True, text=True, timeout=timeout)
    assert out.returncode == 0 and "OK" in out.stdout and "FAIL" not in out.stdout, out.stdout + out.stderr
    return out.stdout


def _load_example(rel_path):
    name = "contact_evidence_" + os.path.basename(os.path.dirname(rel_path))
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_serial_3d_collision_example():
    """Two F-bar Hex8 blocks collide (v0), penetration-free — the approachable serial companion to the
    distributed 3D collision demo. Exercises ElementGroup + InertiaOperator +
    DeformableBarrierContact3D through serial solve_dynamics."""
    _run_example("examples/contact_3d_blocks/run.py", timeout=400)


def test_serial_3d_friction_example():
    """Two F-bar Hex8 blocks, gravity-seated + the top sheared; ppf smoothed friction holds (μ>0 slip <
    frictionless), penetration-free. The serial companion to the distributed 3D friction demo."""
    output = _run_example("examples/contact_3d_friction/run.py", timeout=500)
    assert "min signed interface gap over trajectory" in output


@pytest.mark.parametrize("rel_path", [
    "examples/contact_3d_blocks/run.py",
    "examples/contact_3d_friction/run.py",
])
def test_signed_gap_rejects_tunneled_secondary_surface(rel_path):
    """A vertex below the primary face has negative gap despite nonzero distance."""
    ex = _load_example(rel_path)
    bottom_nodes, _ = ex._hex8_block(ex.NE, 0.0)
    n_bottom = len(bottom_nodes)
    top_nodes, _ = ex._hex8_block(ex.NE, 1.0 + ex.GAP0)
    nodes = np.vstack([bottom_nodes, top_nodes])
    faces = ex._triangulate(ex._surface_grid(ex.NE, 0, "max"))
    secondary = ex._surface_grid(ex.NE, n_bottom, "min").ravel()
    displacement = np.zeros(nodes.size)

    separated = ex._min_signed_interface_gap(displacement, nodes, secondary, faces)
    assert separated == pytest.approx(ex.GAP0)

    # Move the secondary surface through the primary by half the activation
    # distance.  The old unsigned metric was +0.5*dhat and therefore passed
    # ``gap > 0``; the oriented metric must expose the crossing as negative.
    displacement[secondary * 3 + 2] = -(ex.GAP0 + 0.5 * ex.DHAT)
    tunneled = ex._min_signed_interface_gap(displacement, nodes, secondary, faces)
    assert abs(tunneled) == pytest.approx(0.5 * ex.DHAT)
    assert tunneled < 0.0
