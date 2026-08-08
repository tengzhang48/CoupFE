"""Physical-parameter gates for the public neo-Hookean convenience material."""

from __future__ import annotations

import numpy as np
import pytest

from coupfe import NeoHookean, neo_hookean_kernel_props
from examples.neo_hookean_inelastic_local_pressure_quad4.build import (
    NeoHookeanInelasticUP,
)
from examples.neo_hookean_local_pressure_hex8.build import (
    NeoHookeanUP as NeoHookeanUPHex8,
)
from examples.neo_hookean_local_pressure_quad4.build import (
    NeoHookeanUP as NeoHookeanUPQuad4,
)
from examples.neo_hookean_mixed.build import NeoHookeanMixed
from examples.neo_hookean_umat.build import NeoHookean as NeoHookeanUMAT
from examples.phasefield_fracture_uel.build import PhaseFieldFractureMaterial
from examples.scalar_diffusion_uel.build import HeatDiffusionMaterial
from examples.thermo_mechanics_quad8.build import ThermoMechanicalMaterial
from examples.uel_scaffold_quad4.build import NeoHookean as NeoHookeanScaffold


def _raw_cauchy(F, props):
    """Evaluate the retained raw ``(G, lambda)`` material-point contract."""
    G, lame_lambda = props
    J = np.linalg.det(F)
    FinvT = np.linalg.inv(F).T
    P = G * (F - FinvT) + lame_lambda * np.log(J) * FinvT
    return P @ F.T / J


def _small_strain_moduli(props, h=1.0e-6):
    """Recover shear and bulk moduli from independent centered perturbations."""
    return _small_strain_moduli_from_cauchy(
        lambda F: _raw_cauchy(F, props), h=h
    )


def _small_strain_moduli_from_cauchy(cauchy, h=1.0e-6):
    """Recover moduli from a material-independent numerical perturbation."""
    eye = np.eye(3)

    shear_plus = eye.copy()
    shear_minus = eye.copy()
    shear_plus[0, 1] += h
    shear_minus[0, 1] -= h
    shear = (
        cauchy(shear_plus)[0, 1]
        - cauchy(shear_minus)[0, 1]
    ) / (2.0 * h)

    sigma_plus = cauchy((1.0 + h) * eye)
    sigma_minus = cauchy((1.0 - h) * eye)
    mean_plus = np.trace(sigma_plus) / 3.0
    mean_minus = np.trace(sigma_minus) / 3.0
    # The centered volumetric-strain increment is 6h: tr(+h I)-tr(-h I).
    bulk = (mean_plus - mean_minus) / (6.0 * h)
    return shear, bulk


@pytest.mark.parametrize("G,K", ((1.0, 10.0), (3.5, 7.25)))
def test_public_neo_hookean_recovers_physical_shear_and_bulk_moduli(G, K):
    raw_props = neo_hookean_kernel_props(G, K)
    assert raw_props == pytest.approx((G, K - 2.0 * G / 3.0))
    assert NeoHookean(G=G, K=K).props == pytest.approx(raw_props)

    shear, bulk = _small_strain_moduli(raw_props)
    assert shear == pytest.approx(G, rel=2.0e-9, abs=1.0e-10)
    assert bulk == pytest.approx(K, rel=2.0e-9, abs=1.0e-10)


def test_raw_bulk_modulus_bypass_is_a_broken_control():
    """Passing physical K directly to the raw lambda slot must miss the K gate."""
    G, K = 1.0, 10.0
    shear, broken_bulk = _small_strain_moduli((G, K))

    assert shear == pytest.approx(G, rel=2.0e-9, abs=1.0e-10)
    assert broken_bulk == pytest.approx(K + 2.0 * G / 3.0, rel=2.0e-9)
    assert not np.isclose(broken_bulk, K, rtol=1.0e-3, atol=0.0)


def _cauchy_from_pk1(stress_PK1, F):
    P = np.asarray(stress_PK1(F), dtype=complex).real
    return P @ F.T / np.linalg.det(F)


@pytest.mark.parametrize(
    "material,stress",
    (
        (NeoHookeanUMAT(G=1.0, K=10.0), lambda m, F: m.stress_PK1(F)),
        (NeoHookeanScaffold(G=1.0, K=10.0), lambda m, F: m.stress_PK1(F)),
        (
            HeatDiffusionMaterial(G=1.0, K=10.0),
            lambda m, F: m.stress_PK1(F, 0.0),
        ),
        (
            ThermoMechanicalMaterial(G=1.0, K=10.0),
            lambda m, F: m.stress_PK1(F, 0.0),
        ),
        (
            PhaseFieldFractureMaterial(G=1.0, K=10.0, kappa=0.0),
            lambda m, F: m.stress_PK1(F, 0.0),
        ),
    ),
    ids=("umat", "uel-scaffold", "diffusion", "thermo", "phasefield"),
)
def test_codegen_declarations_recover_physical_moduli(material, stress):
    def cauchy(F):
        return _cauchy_from_pk1(lambda state: stress(material, state), F)

    shear, bulk = _small_strain_moduli_from_cauchy(cauchy)
    assert shear == pytest.approx(1.0, rel=2.0e-9, abs=1.0e-10)
    assert bulk == pytest.approx(10.0, rel=2.0e-9, abs=1.0e-10)


@pytest.mark.parametrize(
    "material,pressure,stress",
    (
        (
            NeoHookeanUPQuad4(G=1.0, K=10.0),
            lambda _m, F, lam: lam * np.log(np.linalg.det(F)),
            lambda m, F, p: m.stress_PK1(F, p),
        ),
        (
            NeoHookeanUPHex8(G=1.0, K=10.0),
            lambda _m, F, lam: lam * np.log(np.linalg.det(F)),
            lambda m, F, p: m.stress_PK1(F, p),
        ),
        (
            NeoHookeanInelasticUP(G=1.0, K=10.0, J_inel=1.0),
            lambda _m, F, lam: lam * np.log(np.linalg.det(F)),
            lambda m, F, p: m.stress_PK1(F, p, 1.0),
        ),
        (
            NeoHookeanMixed(G=1.0, K=10.0),
            lambda _m, F, lam: lam * (np.linalg.det(F) - 1.0),
            lambda m, F, p: m.stress_PK1(F, p),
        ),
    ),
    ids=("local-q4", "local-hex8", "inelastic-local-q4", "mixed-q8"),
)
def test_pressure_elimination_recovers_physical_moduli(
    material, pressure, stress
):
    lame_lambda = material.K - 2.0 * material.G / 3.0

    def cauchy(F):
        p = pressure(material, F, lame_lambda)
        return _cauchy_from_pk1(lambda state: stress(material, state, p), F)

    shear, bulk = _small_strain_moduli_from_cauchy(cauchy)
    assert shear == pytest.approx(1.0, rel=2.0e-9, abs=1.0e-10)
    assert bulk == pytest.approx(10.0, rel=2.0e-9, abs=1.0e-10)
