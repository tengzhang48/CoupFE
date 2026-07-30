"""Focused gates for the RESEARCH stabilized mixed u-theta Tet4 example."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sympy")

_EXAMPLE_DIR = (
    Path(__file__).resolve().parent.parent / "examples" / "stabilized_tet4"
)
_BUILD_PY = _EXAMPLE_DIR / "build.py"


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "stabilized_tet4_build", _BUILD_PY
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_independent_derivative_and_homogeneous_oracles():
    example = _load_example()
    errors = example.run_reference_checks()
    assert errors["dS_dtheta"] < 1.0e-9
    assert errors["homogeneous"] < 1.0e-12


def test_derivative_oracle_rejects_wrong_log_term_sign():
    """Broken control: the paper's minus-log term must not become plus-log."""
    example = _load_example()
    material = example.ScovazziBlockMaterial()
    state = next(iter(example.verification_states()))
    F = np.asarray(state["F"], dtype=complex)
    thetat = float(state["thetat"])
    theta = 1.0 + thetat
    step = 1.0e-25

    derivative_cs = np.imag(
        example._independent_second_pk(
            F, thetat + 1j * step, material
        )
    ) / step
    _, Cbar_inv, _, _ = material._S_enriched(F, thetat)

    # Deliberately reintroduce a sign error in
    # 3*lambda + 2*mu - 2*lambda*log(theta).
    broken_factor = (
        3.0 * material.lam
        + 2.0 * material.mu
        + 2.0 * material.lam * np.log(theta)
    )
    broken = broken_factor * Cbar_inv / (3.0 * theta)
    scale = np.max(np.abs(derivative_cs))
    broken_relative_error = np.max(np.abs(broken - derivative_cs)) / scale
    assert broken_relative_error > 4.0e-2


def test_material_and_problem_verify_at_non_benign_states():
    example = _load_example()
    problem = example.ScovazziBlockTet4()
    states = list(example.verification_states())

    # WeakForm construction supplies the material with the declared scalar
    # field metadata, so this is the configured material verification path.
    assert problem._mat.verify(
        state=states[0], tol=5.0e-5, verbose=False
    )
    for state in states:
        assert problem.verify(state=state, tol=5.0e-5, verbose=False)


def test_tet4_generation_and_object_compile(tmp_path):
    example = _load_example()
    generated = example.build(tmp_path)
    source = generated.read_text()

    assert generated.name == "scovazzi_block_tet4.for"
    assert "NDOFEL = 16" in source
    assert "INTEGER, PARAMETER :: NGP = 4" in source
    assert "CALL gauss_tet4" in source
    assert "scovazziblockmaterial_stress_PK1" in source
    assert "scovazziblockmaterial_phase_storage" in source
    assert "scovazziblockmaterial_phase_flux" in source

    compiler = shutil.which("gfortran")
    if compiler is None:
        return
    result = subprocess.run(
        [
            compiler,
            "-c",
            "-ffixed-form",
            "-ffixed-line-length-none",
            str(generated),
            "-o",
            str(tmp_path / "scovazzi_block_tet4.o"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_reference_record_is_explicitly_historical():
    record = json.loads((_EXAMPLE_DIR / "reference_result.json").read_text())
    assert record["record_kind"] == (
        "historical_abaqus_published_comparison_metadata"
    )
    assert record["current_coupfe_solver_reproduction"] is False
    assert record["historical_abaqus_result"]["final_center_u3_mm"] == pytest.approx(
        -0.6962435841560364
    )
