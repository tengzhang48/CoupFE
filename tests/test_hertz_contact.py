"""Hertz analytic contact benchmark: force law, fields, and equilibrium."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

_RUN = Path(__file__).parent.parent / "examples" / "hertz_contact" / "run.py"
_RENDER = Path(__file__).parent.parent / "examples" / "hertz_contact" / "render.py"
_ASSET = (
    Path(__file__).parent.parent
    / "docs"
    / "assets"
    / "hertz-contact-benchmark.svg"
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="needs gfortran")
def test_hertz_force_law(tmp_path):
    ex = _load(_RUN, "hertz_run")
    evidence = ex.run_hertz(deltas=(0.03, 0.05, 0.07))
    deltas = evidence["deltas"]
    force_fe = evidence["force_fe"]
    slope = evidence["fit_slope"]
    ratios = evidence["force_ratios"]

    assert np.allclose(force_fe / ex.hertz_force(deltas), ratios)
    assert abs(slope - 1.5) < 0.08, f"log-log slope {slope:.3f} not near 3/2"
    assert np.max(np.abs(ratios - 1.0)) < 0.08, (
        f"maximum retained force error {np.max(np.abs(ratios - 1.0)):.1%}"
    )

    # The kernel's ln(J) coefficient has the infinitesimal role of lambda.  A
    # bulk-modulus substitution silently changes the E, nu pair used by Hertz.
    expected_lambda = ex.E * ex.NU / ((1.0 + ex.NU) * (1.0 - 2.0 * ex.NU))
    assert ex.LAME_LAMBDA == pytest.approx(expected_lambda)

    config = evidence["configuration"]
    snapshot = evidence["snapshot"]
    assert config["elements"] == ex.NX * ex.NY * ex.NZ
    assert config["nodes"] == (ex.NX + 1) * (ex.NY + 1) * (ex.NZ + 1)
    assert snapshot["nodes_reference"].shape == (config["nodes"], 3)
    assert snapshot["nodes_deformed"].shape == (config["nodes"], 3)
    assert snapshot["displacement"].shape == (config["nodes"], 3)
    assert snapshot["elements"].shape == (config["elements"], 8)
    assert np.all(np.isfinite(snapshot["nodes_deformed"]))
    assert np.array_equal(
        snapshot["active_contact"], snapshot["contact_gap"] < 0.0
    )
    assert np.all(snapshot["contact_normal_reaction"] >= 0.0)
    assert np.all(snapshot["contact_normal_reaction"][~snapshot["active_contact"]] == 0.0)
    assert float(np.sum(snapshot["contact_vertical_reaction"])) == pytest.approx(
        force_fe[-1], rel=1.0e-8, abs=1.0e-10
    )
    assert np.allclose(snapshot["displacement"][snapshot["bottom_nodes"]], 0.0)
    assert all(case["base_vertical_residual"] > 0.0 for case in evidence["cases"])
    assert all(case["free_residual_norm"] < 1.0e-7 for case in evidence["cases"])
    assert all(case["force_balance_error"] < 1.0e-6 for case in evidence["cases"])

    renderer = _load(_RENDER, "hertz_render")
    figure = renderer.render_hertz_svg(evidence, tmp_path / "hertz.svg")
    svg = figure.read_text(encoding="utf-8")
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "CoupFE · m =" in svg
    assert "active nodal reaction" in svg
    assert "not a mesh/domain-converged pressure solution" in svg
    assert "contact pressure" not in svg.lower()
    assert "nan" not in svg.lower()

    published = _ASSET.read_text(encoding="utf-8")
    assert "Finite-element field at δ = 0.080" in published
    assert "CoupFE · m = 1.533" in published
    assert "Force error +1.3% to +6.2%" in published
    assert "2601 nodes · 2048 Hex8 · 25 active nodes" in published
    assert "active nodal reaction" in published
    assert "contact pressure" not in published.lower()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("gfortran") is None, reason="needs gfortran")
def test_hertz_bulk_modulus_broken_control():
    """The plausible bulk-modulus substitution must fail the force gate."""

    ex = _load(_RUN, "hertz_run_wrong_modulus")
    physical_bulk_modulus = ex.E / (3.0 * (1.0 - 2.0 * ex.NU))
    assert physical_bulk_modulus != pytest.approx(ex.LAME_LAMBDA)
    ex.LAME_LAMBDA = physical_bulk_modulus
    evidence = ex.run_hertz(deltas=(0.03, 0.05, 0.07))
    assert np.max(np.abs(evidence["force_ratios"] - 1.0)) >= 0.08
