"""Focused public gates for the two hyperelastic Abaqus UMAT examples."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import shutil
import subprocess
import warnings

import numpy as np
import pytest
import coupfe.codegen as au
from coupfe.codegen.core.tensor import eye
from coupfe.codegen.core.verify import VerificationError

try:
    from numpy.exceptions import ComplexWarning
except ImportError:  # NumPy < 1.25
    from numpy import ComplexWarning

_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES = {
    "neo_hookean": (
        _ROOT / "examples" / "neo_hookean_umat",
        "neo_hookean_umat.for",
    ),
    "ogden": (
        _ROOT / "examples" / "ogden_umat",
        "ogden_umat.for",
    ),
}


def _load_example(name):
    example_dir, _ = _EXAMPLES[name]
    spec = importlib.util.spec_from_file_location(
        "abaqus_ufl_{}_build".format(name), example_dir / "build.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _cauchy(model, F):
    F = np.asarray(F, dtype=float)
    first_pk = np.asarray(model.stress_PK1(F), dtype=complex).real
    return first_pk @ F.T / np.linalg.det(F)


def test_neo_hookean_closed_forms_and_broken_control():
    example = _load_example("neo_hookean")
    model = example.NeoHookean()
    shear_modulus, bulk_modulus = example.DEFAULT_PROPS

    stretch = 1.2
    F_uniaxial = np.diag([stretch, 1.0, 1.0])
    expected_uniaxial = np.diag(
        [
            shear_modulus * (stretch - 1.0 / stretch)
            + bulk_modulus * math.log(stretch) / stretch,
            bulk_modulus * math.log(stretch) / stretch,
            bulk_modulus * math.log(stretch) / stretch,
        ]
    )
    assert np.allclose(
        _cauchy(model, F_uniaxial),
        expected_uniaxial,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    gamma = 0.3
    F_shear = np.eye(3)
    F_shear[0, 1] = gamma
    expected_shear = shear_modulus * np.array(
        [
            [gamma**2, gamma, 0.0],
            [gamma, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(
        _cauchy(model, F_shear),
        expected_shear,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    # Broken control: deleting the volumetric term must fail the uniaxial
    # oracle by a wide margin.
    wrong_first_pk = shear_modulus * (
        F_uniaxial - np.linalg.inv(F_uniaxial).T
    )
    wrong_cauchy = (
        wrong_first_pk @ F_uniaxial.T / np.linalg.det(F_uniaxial)
    )
    assert not np.allclose(
        wrong_cauchy, expected_uniaxial, rtol=1.0e-6, atol=1.0e-8
    )


def _ogden_generic_F():
    angles = (0.4, -0.3, 0.7)
    ax, ay, az = angles
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(ax), -math.sin(ax)],
            [0.0, math.sin(ax), math.cos(ax)],
        ]
    )
    rotation_y = np.array(
        [
            [math.cos(ay), 0.0, math.sin(ay)],
            [0.0, 1.0, 0.0],
            [-math.sin(ay), 0.0, math.cos(ay)],
        ]
    )
    rotation_z = np.array(
        [
            [math.cos(az), -math.sin(az), 0.0],
            [math.sin(az), math.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    stretch = np.array(
        [
            [1.15, 0.08, 0.00],
            [0.08, 0.95, 0.05],
            [0.00, 0.05, 1.06],
        ]
    )
    return rotation_z @ rotation_y @ rotation_x @ stretch


def _isochoric_neo_hookean_cauchy(F, mu, bulk_modulus):
    J = np.linalg.det(F)
    bbar = J ** (-2.0 / 3.0) * (F @ F.T)
    dev_bbar = bbar - np.trace(bbar) * np.eye(3) / 3.0
    return (
        mu * dev_bbar + bulk_modulus * math.log(J) * np.eye(3)
    ) / J


def test_ogden_eig_free_closed_forms_and_broken_control():
    example = _load_example("ogden")
    model = example.OgdenOneTerm()
    mu, alpha, bulk_modulus = example.DEFAULT_PROPS

    # Triple-repeated spectrum: the isochoric part is exactly zero.
    dilation = 1.2
    observed = _cauchy(model, dilation * np.eye(3))
    expected = (
        bulk_modulus
        * math.log(dilation**3)
        / dilation**3
        * np.eye(3)
    )
    assert np.allclose(observed, expected, rtol=1.0e-11, atol=1.0e-11)

    # Pair-repeated spectrum with J=1.
    axial_stretch = 1.6
    stretches = np.array(
        [
            axial_stretch,
            1.0 / math.sqrt(axial_stretch),
            1.0 / math.sqrt(axial_stretch),
        ]
    )
    powers = stretches**alpha
    expected_principal = (
        2.0 * mu / alpha * (powers - powers.sum() / 3.0)
    )
    observed = _cauchy(model, np.diag(stretches))
    assert np.allclose(
        np.diag(observed),
        expected_principal,
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    assert np.max(np.abs(observed - np.diag(np.diag(observed)))) < 1.0e-11

    # At alpha=2, the spectral model must equal the eig-free isochoric
    # neo-Hookean expression for a generic non-isochoric state.
    F = _ogden_generic_F()
    alpha_two = example.OgdenOneTerm(mu=mu, alpha=2.0, K=bulk_modulus)
    expected = _isochoric_neo_hookean_cauchy(F, mu, bulk_modulus)
    assert np.allclose(
        _cauchy(alpha_two, F), expected, rtol=1.0e-10, atol=1.0e-12
    )

    # Broken control: omitting J^(-2/3) from bbar must be rejected.
    J = np.linalg.det(F)
    b = F @ F.T
    dev_b = b - np.trace(b) * np.eye(3) / 3.0
    wrong = (
        mu * dev_b + bulk_modulus * math.log(J) * np.eye(3)
    ) / J
    assert not np.allclose(wrong, expected, rtol=1.0e-6, atol=1.0e-8)


@pytest.mark.parametrize("name", ("neo_hookean", "ogden"))
def test_model_verifies_at_nontrivial_state(name):
    example = _load_example(name)
    class_name = "NeoHookean" if name == "neo_hookean" else "OgdenOneTerm"
    model = getattr(example, class_name)()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComplexWarning)
        assert model.verify(state=example.verification_state(), verbose=False)


def test_verifier_preserves_complex_finite_difference_components():
    class ComplexContainerStress(au.Material):
        def stress_PK1(self, F):
            return complex(1.0) * (F - eye(3))

    class BrokenComplexSlope(au.Material):
        def stress_PK1(self, F):
            return (1.0 + 1.0e-3j) * (F - eye(3))

    with warnings.catch_warnings():
        warnings.simplefilter("error", ComplexWarning)
        assert ComplexContainerStress().verify(verbose=False)
    with pytest.raises(
        VerificationError, match=r"d\(stress_PK1\)/d\(F\)"
    ):
        BrokenComplexSlope().verify(verbose=False)


@pytest.mark.parametrize("name", ("neo_hookean", "ogden"))
def test_generated_source_is_byte_identical(name, tmp_path):
    example = _load_example(name)
    example_dir, source_name = _EXAMPLES[name]
    generated = example.generate_source(tmp_path, verify=False)
    retained = example_dir / source_name

    assert generated.read_bytes() == retained.read_bytes()
    source = generated.read_text()
    assert "SUBROUTINE UMAT" in source
    assert "Generated by coupfe.codegen" in source
    assert "Original declaration Copyright (c) 2026 Teng Zhang, MIT." in source


@pytest.mark.parametrize("name", ("neo_hookean", "ogden"))
def test_generated_source_compiles(name, tmp_path):
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran not available")

    example_dir, source_name = _EXAMPLES[name]
    source = example_dir / source_name
    output = tmp_path / source.with_suffix(".o").name
    result = subprocess.run(
        [
            compiler,
            "-c",
            "-ffixed-form",
            "-ffixed-line-length-none",
            str(source),
            "-o",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file()
