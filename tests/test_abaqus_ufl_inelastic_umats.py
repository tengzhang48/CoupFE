"""Focused gates for the public J2 and SLS small-strain examples."""

from __future__ import annotations

import importlib.util
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[1]


def _load(name, relative_path):
    path = _ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_j2_uel_elastic_branch_matches_hooke_and_preserves_state():
    example = _load(
        "coupfe_j2_uel_elastic",
        "examples/j2_plasticity_uel/build.py",
    )
    problem = example.J2Quad4()
    model = problem._mat
    deformation_gradient = np.eye(3) + np.diag([5.0e-4, 0.0, 0.0])
    strain = (
        0.5 * (deformation_gradient + deformation_gradient.T) - np.eye(3)
    )
    epsp_old = np.zeros((3, 3))
    alpha_old = 0.0

    sigma, state = model.stress_PK1(
        deformation_gradient, epsp_old, alpha_old, 0.1
    )
    sigma = np.real(np.asarray(sigma, dtype=complex))
    epsp_new = np.real(np.asarray(state["epsp"], dtype=complex))
    alpha_new = float(np.real(state["alpha"]))

    mu = model.E / (2.0 * (1.0 + model.nu))
    lam = (
        model.E
        * model.nu
        / ((1.0 + model.nu) * (1.0 - 2.0 * model.nu))
    )
    sigma_expected = (
        lam * np.trace(strain) * np.eye(3) + 2.0 * mu * strain
    )
    deviator = sigma - np.trace(sigma) * np.eye(3) / 3.0
    q_new = np.sqrt(1.5 * np.sum(deviator * deviator))

    np.testing.assert_allclose(sigma, sigma_expected, rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(epsp_new, epsp_old)
    assert alpha_new == alpha_old
    assert q_new < model.sigma_y

    equation_sigma = problem.momentum_equation(
        None, deformation_gradient, epsp_old, alpha_old, 0.1
    )
    np.testing.assert_allclose(
        np.real(np.asarray(equation_sigma, dtype=complex)),
        sigma_expected,
        rtol=0.0,
        atol=1.0e-12,
    )
    equation = problem.equations["momentum_equation"]
    assert equation["field_vars"] == ["F"]
    assert equation["param_vars"] == ["dt"]

    oracle = example.analytic_state_update(
        strain,
        epsp_old,
        alpha_old,
        model.E,
        model.nu,
        model.sigma_y,
        model.H,
    )
    np.testing.assert_allclose(oracle[0], sigma_expected, rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(oracle[1], epsp_old)
    assert oracle[2] == alpha_old


def test_j2_uel_plastic_return_reaches_yield_surface_and_updates_state():
    example = _load(
        "coupfe_j2_uel_plastic",
        "examples/j2_plasticity_uel/build.py",
    )
    model = example.J2SmallStrainPlasticity()
    old = example.verification_state()
    strain = 0.5 * (old["F"] + old["F"].T) - np.eye(3)
    epsp_old = old["epsp"]
    alpha_old = old["alpha"]

    mu = model.E / (2.0 * (1.0 + model.nu))
    lam = (
        model.E
        * model.nu
        / ((1.0 + model.nu) * (1.0 - 2.0 * model.nu))
    )
    elastic_trial = strain - epsp_old
    sigma_trial = (
        lam * np.trace(elastic_trial) * np.eye(3)
        + 2.0 * mu * elastic_trial
    )
    s_trial = sigma_trial - np.trace(sigma_trial) * np.eye(3) / 3.0
    q_trial = np.sqrt(1.5 * np.sum(s_trial * s_trial))
    phi_trial = q_trial - (model.sigma_y + model.H * alpha_old)
    assert phi_trial > 0.0

    dgamma = phi_trial / (3.0 * mu + model.H)
    flow_direction = 1.5 * s_trial / q_trial
    sigma_expected = sigma_trial - 2.0 * mu * dgamma * flow_direction
    epsp_expected = epsp_old + dgamma * flow_direction
    alpha_expected = alpha_old + dgamma

    sigma, state = model.stress_PK1(
        old["F"], epsp_old, alpha_old, old["dt"]
    )
    sigma = np.real(np.asarray(sigma, dtype=complex))
    epsp_new = np.real(np.asarray(state["epsp"], dtype=complex))
    alpha_new = float(np.real(state["alpha"]))

    np.testing.assert_allclose(sigma, sigma_expected, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(epsp_new, epsp_expected, rtol=0.0, atol=1.0e-14)
    assert math.isclose(alpha_new, alpha_expected, rel_tol=0.0, abs_tol=1.0e-15)
    assert alpha_new > alpha_old
    assert abs(np.trace(epsp_new)) < 1.0e-14

    s_new = sigma - np.trace(sigma) * np.eye(3) / 3.0
    q_new = np.sqrt(1.5 * np.sum(s_new * s_new))
    yield_new = model.sigma_y + model.H * alpha_new
    assert math.isclose(q_new, yield_new, rel_tol=1.0e-13, abs_tol=1.0e-10)

    # The returned stress must be elastic stress evaluated from the new state.
    elastic_new = strain - epsp_new
    sigma_from_state = (
        lam * np.trace(elastic_new) * np.eye(3) + 2.0 * mu * elastic_new
    )
    np.testing.assert_allclose(sigma, sigma_from_state, rtol=0.0, atol=1.0e-10)

    oracle = example.analytic_state_update(
        strain,
        epsp_old,
        alpha_old,
        model.E,
        model.nu,
        model.sigma_y,
        model.H,
    )
    np.testing.assert_allclose(oracle[0], sigma_expected, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(oracle[1], epsp_expected, rtol=0.0, atol=1.0e-14)
    assert math.isclose(oracle[2], alpha_expected, rel_tol=0.0, abs_tol=1.0e-15)

    # Broken control: the former s/q direction and scaled-alpha update leave
    # this same active state materially outside the updated yield surface.
    old_direction = s_trial / q_trial
    old_sigma = sigma_trial - 2.0 * mu * dgamma * old_direction
    old_s = old_sigma - np.trace(old_sigma) * np.eye(3) / 3.0
    old_q = np.sqrt(1.5 * np.sum(old_s * old_s))
    old_alpha = alpha_old + np.sqrt(2.0 / 3.0) * dgamma
    assert old_q - (model.sigma_y + model.H * old_alpha) > 100.0


def test_j2_uel_generated_kernel_commits_consistent_state(tmp_path):
    if not all(shutil.which(tool) for tool in ("gfortran", "meson", "ninja")):
        pytest.skip("native Fortran build toolchain is unavailable")

    from coupfe.runtime.compiled_element import CompiledElement

    example = _load(
        "coupfe_j2_uel_generated",
        "examples/j2_plasticity_uel/build.py",
    )
    build_dir = tmp_path / "j2-native"
    _source, module = example.build_kernel(
        tmpdir=str(build_dir),
        module_name="coupfe_test_j2_uel_state",
    )
    problem = example.J2Quad4()
    element = CompiledElement(
        module,
        props=example.DEFAULT_PROPS,
        dof_per_node=2,
        n_svars=4 * example._PER_GP,
        mcrd=2,
        n_elem=1,
        dt=0.1,
        state_schema=problem._mat.state_schema,
    )

    old = example.verification_state()
    example.set_element_state(element, old["epsp"], old["alpha"])
    committed_old = element.svars.copy()
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    displacement = np.column_stack(
        (0.02 * coords[:, 0], np.zeros(coords.shape[0]))
    ).ravel()

    residual = element.element_r(0, coords, displacement, displacement)
    assert np.all(np.isfinite(residual))
    np.testing.assert_array_equal(element.svars, committed_old)

    strain = np.diag([0.02, 0.0, 0.0])
    _sigma_expected, epsp_expected, alpha_expected = (
        example.analytic_state_update(
            strain,
            old["epsp"],
            old["alpha"],
            *example.DEFAULT_PROPS,
        )
    )
    for gp in range(4):
        epsp_trial, alpha_trial = example.unpack_gp_state(
            element.svars_trial[0], gp
        )
        np.testing.assert_allclose(
            epsp_trial, epsp_expected, rtol=0.0, atol=1.0e-14
        )
        assert math.isclose(
            alpha_trial, alpha_expected, rel_tol=0.0, abs_tol=1.0e-15
        )

    element.commit()
    for gp in range(4):
        epsp_committed, alpha_committed = example.get_gp_state(element, gp)
        np.testing.assert_allclose(
            epsp_committed, epsp_expected, rtol=0.0, atol=1.0e-14
        )
        assert math.isclose(
            alpha_committed, alpha_expected, rel_tol=0.0, abs_tol=1.0e-15
        )


def test_j2_closed_form_covers_elastic_plastic_and_broken_control():
    example = _load(
        "coupfe_small_strain_j2",
        "examples/small_strain_j2_umat/build.py",
    )
    model = example.SmallStrainJ2()
    assert model.verify(verbose=False)

    errors = example.reference_errors(model)
    assert errors["elastic_tau_abs"] < 1.0e-12
    assert errors["elastic_ep_abs"] < 1.0e-14
    assert errors["plastic_tau_abs"] < 1.0e-10
    assert errors["plastic_ep_abs"] < 1.0e-10
    assert errors["plastic_ep"] > 0.0

    # Broken control: dropping H from the return denominator must disagree
    # materially with the exact consistency solution.
    numerator = (
        2.0
        * math.sqrt(3.0)
        * model.G
        * example.FINAL_TENSOR_SHEAR
        - model.sigma_y
    )
    ep_without_hardening_in_denominator = numerator / (3.0 * model.G)
    _, ep_reference = example.closed_form_monotonic_shear(
        example.FINAL_TENSOR_SHEAR, model
    )
    assert abs(ep_without_hardening_in_denominator - ep_reference) > 1.0e-3


def test_sls_discrete_relaxation_limits_and_broken_control():
    example = _load(
        "coupfe_small_strain_sls",
        "examples/small_strain_viscoelastic_umat/build.py",
    )
    model = example.SmallStrainViscoelastic()
    assert model.verify(verbose=False)

    errors = example.reference_errors(model)
    assert errors["max_stress_abs"] < 1.0e-13
    assert errors["max_viscous_strain_abs"] < 1.0e-13

    instantaneous = (
        2.0
        * example.STEP_TENSOR_SHEAR
        * (model.G_inf + model.G_v)
    )
    equilibrium = 2.0 * example.STEP_TENSOR_SHEAR * model.G_inf
    assert equilibrium < errors["final_stress"] < errors["first_stress"]
    assert errors["first_stress"] < instantaneous

    coarse_error, fine_error = example.continuous_limit_errors()
    assert fine_error < 0.6 * coarse_error

    # A non-default relaxation time must flow through the diagnostic; this
    # catches the former hard-coded tau=0.5 oracle.
    slower_model = example.SmallStrainViscoelastic(tau=1.25)
    slower_coarse, slower_fine = example.continuous_limit_errors(slower_model)
    assert slower_fine < slower_coarse
    assert not np.allclose(
        (slower_coarse, slower_fine),
        (coarse_error, fine_error),
        rtol=1.0e-12,
        atol=1.0e-15,
    )

    # Broken control: an explicit dashpot recursion must not match the
    # backward-Euler discrete oracle.
    ratio = example.TIME_INCREMENT / model.tau
    eps_v_explicit = 0.0
    for _ in range(example.N_INCREMENTS):
        eps_v_explicit += ratio * (
            example.STEP_TENSOR_SHEAR - eps_v_explicit
        )
    explicit_stress = (
        2.0 * example.STEP_TENSOR_SHEAR * model.G_inf
        + 2.0
        * model.G_v
        * (example.STEP_TENSOR_SHEAR - eps_v_explicit)
    )
    exact_stress = example.discrete_shear_stress(
        example.N_INCREMENTS, model
    )
    assert abs(explicit_stress - exact_stress) > 1.0e-4


@pytest.mark.parametrize(
    (
        "name",
        "relative_build",
        "committed_name",
        "origin_example",
        "source_needles",
    ),
    [
        (
            "j2_regeneration",
            "examples/small_strain_j2_umat/build.py",
            "small_strain_j2.for",
            "small_strain_j2_umat",
            (
                "NSTATV = 1",
                "SUBROUTINE smallstrainj2_stress_update",
                "ep_old = STATEV(1)",
                "STATEV(1) = DBLE(ep_new_z)",
            ),
        ),
        (
            "sls_regeneration",
            "examples/small_strain_viscoelastic_umat/build.py",
            "small_strain_viscoelastic.for",
            "small_strain_viscoelastic_umat",
            (
                "NSTATV = 9",
                "SUBROUTINE smallstrainviscoelastic_stress_update",
                "eps_v_old(1,1) = STATEV(1)",
                "STATEV(9) = DBLE(eps_v_new_z(3,3))",
            ),
        ),
    ],
)
def test_generated_umats_are_current_and_compile(
    tmp_path,
    name,
    relative_build,
    committed_name,
    origin_example,
    source_needles,
):
    example = _load(name, relative_build)
    regenerated = example.generate(tmp_path / committed_name)
    committed = _ROOT / Path(relative_build).parent / committed_name

    assert regenerated.read_bytes() == committed.read_bytes()
    source = regenerated.read_text()
    assert "SUBROUTINE UMAT" in source
    assert "compression-positive user API" in source
    assert (
        "Declaration origin: tengzhang48/abaqus_ufl {}".format(
            origin_example
        )
        in source
    )
    assert "Commit: 0f525339db1aad70e9f8f4825a02c1164f0da7a0" in source
    assert (
        "Original declaration Copyright (c) 2026 Teng Zhang, MIT."
        in source
    )
    assert (
        "Regenerated by CoupFE; see README.md for scope and license link."
        in source
    )
    for needle in source_needles:
        assert needle in source

    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran not available")
    result = subprocess.run(
        [
            compiler,
            "-c",
            "-ffixed-form",
            "-ffixed-line-length-none",
            str(regenerated),
            "-o",
            str(tmp_path / "{}.o".format(name)),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
