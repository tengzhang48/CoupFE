"""Focused public gates for the two existing abaqus_ufl paper ports."""

from __future__ import annotations

import importlib.util
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("sympy")

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mixed_gel_material_has_stress_free_equilibrium_oracle():
    gel = _load(
        "paper_gel_build",
        "examples/gel_chester_anand/u_p_mu_quad8/build.py",
    )
    mat = gel.ChesterAnandGelMaterial()
    identity = np.eye(3, dtype=complex)
    phi0 = float(mat.phi0)
    mu_equilibrium = float(mat.Rgas * mat.theta) * (
        math.log(1.0 - phi0) + phi0 + float(mat.chi) * phi0**2
    )

    np.testing.assert_allclose(
        np.asarray(mat.stress_PK1(identity, 0.0, mu_equilibrium)),
        np.zeros((3, 3)),
        atol=1.0e-13,
    )
    assert abs(complex(mat.pressure_resid(identity, 0.0, mu_equilibrium))) < 1.0e-12
    assert abs(complex(
        mat.solvent_storage(identity, identity, 0.0, 0.0, 1.0)
    )) < 1.0e-12

    # Broken control: an offset chemical potential must violate equilibrium.
    assert abs(complex(
        mat.pressure_resid(identity, 0.0, mu_equilibrium + 1.0)
    )) > 0.99
    assert gel.ChesterAnandUPMuQuad8().verify(
        state=gel.verification_state(),
        verbose=False,
    )


def test_corrosion_material_has_stationary_intact_state_oracle():
    corrosion = _load(
        "paper_corrosion_build",
        "examples/phasefield_corrosion_cui/build.py",
    )
    mat = corrosion.CuiJ2CorrosionMaterial()
    identity = np.eye(3, dtype=complex)
    zero = np.zeros(3, dtype=complex)

    stress, state = mat.stress_PK1(
        identity,
        1.0,
        0.0,
        np.zeros((3, 3), dtype=complex),
        0.0,
        0.0,
        1.0e-3,
        0.0,
        0.0,
        1.0,
    )
    np.testing.assert_allclose(np.asarray(stress), np.zeros((3, 3)), atol=1.0e-12)
    assert abs(complex(state["ep"])) < 1.0e-12
    assert abs(complex(
        mat.phase_storage(identity, 1.0, 1.0, 1.0, 1.0e-3, 1.0)
    )) < 1.0e-12
    np.testing.assert_allclose(mat.phase_flux(identity, 1.0, zero), zero)
    assert abs(complex(mat.species_storage(identity, 1.0, 1.0, 1.0))) < 1.0e-12
    np.testing.assert_allclose(mat.species_flux(identity, 1.0, zero, zero), zero)

    # Broken control: a phase increment at fixed chemistry is not stationary.
    assert abs(complex(
        mat.phase_storage(identity, 0.9, 1.0, 1.0, 1.0e-3, 1.0)
    )) > 1.0
    assert corrosion.CuiJ2CorrosionFull().verify(
        state=corrosion.verification_state(),
        verbose=False,
    )


@pytest.mark.parametrize(
    ("name", "relative_path", "builder", "source_name", "needle"),
    [
        (
            "paper_gel_generate",
            "examples/gel_chester_anand/u_p_mu_quad8/build.py",
            "gel",
            "chester_anand_upmu_quad8_uel.for",
            "chesteranandupmu",
        ),
        (
            "paper_corrosion_generate",
            "examples/phasefield_corrosion_cui/build.py",
            "corrosion",
            "phasefield_corrosion_cui_full_uel.for",
            "CuiJ2Corrosion",
        ),
    ],
)
def test_existing_paper_forms_generate_and_compile(
    tmp_path,
    name,
    relative_path,
    builder,
    source_name,
    needle,
):
    module = _load(name, relative_path)
    if builder == "gel":
        path, _ = module.build_kernel(tmp_path=str(tmp_path))
    else:
        path, _ = module.build_kernel(variant="full", tmp_path=str(tmp_path))

    assert path.name == source_name
    source = path.read_text(encoding="utf-8")
    assert "SUBROUTINE UEL" in source
    assert needle.casefold() in source.casefold()

    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran is not available")
    subprocess.run(
        [
            compiler,
            "-c",
            "-ffixed-form",
            "-ffixed-line-length-none",
            str(path),
            "-o",
            str(tmp_path / f"{builder}.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
