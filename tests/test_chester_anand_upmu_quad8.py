"""Phase 10 Batch 1.5: Chester-Anand three-field u-p-mu Quad8 gel."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

sympy = pytest.importorskip("sympy")  # codegen dep

from coupfe.codegen.core.reference_assembly import assemble_element
from coupfe.codegen.generators.uel_gen import generate_element, generate_uel
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

_BUILD_PY = (
    Path(__file__).parent.parent
    / "examples"
    / "gel_chester_anand"
    / "u_p_mu_quad8"
    / "build.py"
)


def _load_example_module():
    spec = importlib.util.spec_from_file_location("chester_build", _BUILD_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Q8_NODES = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.5, 0.0],
    [1.0, 0.5],
    [0.5, 1.0],
    [0.0, 0.5],
], dtype=float)


def _build_state_vectors(F, p_field, mu_field, grad_mu, F_old_factor):
    """Build U and DU for the Quad8 node-major u-p-mu layout."""
    F2 = np.asarray(F, dtype=float)[:2, :2]
    p = np.asarray(p_field, dtype=float)
    mu = np.asarray(mu_field, dtype=float)
    gmu = np.asarray(grad_mu, dtype=float)
    mu0 = mu[0] - gmu[0] * Q8_NODES[0, 0] - gmu[1] * Q8_NODES[0, 1]

    U = np.zeros(28)
    DU = np.zeros(28)
    idx = 0
    for a, X in enumerate(Q8_NODES):
        u = (F2 - np.eye(2)) @ X
        U[idx:idx + 2] = u
        DU[idx:idx + 2] = (1.0 - F_old_factor) * u
        idx += 2
        if a < 4:
            U[idx] = p[a]
            DU[idx] = p[a] * 0.1  # small pressure increment
            idx += 1
        U[idx] = mu[a]
        DU[idx] = mu[a] - (mu0 + gmu[0] * X[0] + gmu[1] * X[1])
        idx += 1
    return U, DU


def test_material_verifies():
    ex = _load_example_module()
    assert ex.ChesterAnandUPMuQuad8().verify(
        state=ex.verification_state(), verbose=False
    )


def test_native_generation_smoke(tmp_path):
    ex = _load_example_module()
    out = tmp_path / "chester_upmu_q8_native.for"
    generate_element(ex.ChesterAnandUPMuQuad8(), str(out), element='Quad8',
                     formulation='standard', backend='native')
    src = out.read_text().upper()
    assert "SUBROUTINE COUPFE_ELEMENT_RK" in src
    assert "SUBROUTINE UEL" not in src


def test_native_vs_uel_sign_convention(tmp_path):
    """Native residual = weak form = -UEL RHS; native K = UEL AMATRX."""
    if not shutil.which("gfortran"):
        pytest.skip("gfortran not available")

    ex = _load_example_module()
    problem = ex.ChesterAnandUPMuQuad8()
    props = np.asarray(ex.DEFAULT_PROPS, dtype=float)
    state = ex.verification_state()
    dt = float(state["dt"])

    native_for = tmp_path / "chester_upmu_q8_native.for"
    uel_for = tmp_path / "chester_upmu_q8_uel.for"
    generate_element(problem, str(native_for), element='Quad8',
                     formulation='standard', backend='native')
    generate_uel(problem, str(uel_for), element='Quad8',
                 formulation='standard', mat_prefix="chesteranandupmu")

    mod_n = build_element_kernel(
        str(native_for), "chester_upmu_q8_native_sign",
        workdir=str(tmp_path / "build_native"))
    mod_u = build_element_kernel(
        str(uel_for), "chester_upmu_q8_uel_sign",
        workdir=str(tmp_path / "build_uel"))

    F = state["F"]
    p = np.array([1.5, 2.0, 2.5, 1.0])
    mu = np.array([0.8, 1.0, 1.2, 1.1, 0.9, 1.05, 1.15, 0.95])
    grad_mu = state["grad_mu"][:2]
    U, DU = _build_state_vectors(F, p, mu, grad_mu, 0.98)

    native_elem = CompiledElement(mod_n, props=props, dof_per_node=4, n_svars=0,
                                  mcrd=2, n_elem=1, dt=dt, backend='native')
    uel_elem = CompiledElement(mod_u, props=props, dof_per_node=4, n_svars=0,
                               mcrd=2, n_elem=1, dt=dt, backend='abaqus_uel')

    R, K = native_elem.element_rk(0, Q8_NODES, U, DU)
    rhs, amatrx = uel_elem.element_rk(0, Q8_NODES, U, DU)

    assert np.allclose(R, rhs, atol=1e-10, rtol=0.0)
    assert np.allclose(K, amatrx, atol=1e-10, rtol=0.0)

    u_slice = slice(0, 16)
    p_slice = slice(16, 20)
    mu_slice = slice(20, 28)
    assert np.allclose(R[u_slice], rhs[u_slice], atol=1e-10, rtol=0.0)
    assert np.allclose(R[p_slice], rhs[p_slice], atol=1e-10, rtol=0.0)
    assert np.allclose(R[mu_slice], rhs[mu_slice], atol=1e-10, rtol=0.0)
    assert np.allclose(K[u_slice, :], amatrx[u_slice, :], atol=1e-10, rtol=0.0)
    assert np.allclose(K[p_slice, :], amatrx[p_slice, :], atol=1e-10, rtol=0.0)
    assert np.allclose(K[mu_slice, :], amatrx[mu_slice, :], atol=1e-10, rtol=0.0)

    assert not np.allclose(R, -rhs, atol=1e-6, rtol=0.0)


def test_native_vs_reference_assembly(tmp_path):
    """Native kernel matches the independent Python reference assembly."""
    if not shutil.which("gfortran"):
        pytest.skip("gfortran not available")

    ex = _load_example_module()
    problem = ex.ChesterAnandUPMuQuad8()
    props = np.asarray(ex.DEFAULT_PROPS, dtype=float)
    state = ex.verification_state()
    dt = float(state["dt"])

    native_for = tmp_path / "chester_upmu_q8_native.for"
    generate_element(problem, str(native_for), element='Quad8',
                     formulation='standard', backend='native')
    mod = build_element_kernel(
        str(native_for), "chester_upmu_q8_native_ref",
        workdir=str(tmp_path / "build"))

    F = state["F"]
    p = np.array([1.5, 2.0, 2.5, 1.0])
    mu = np.array([0.8, 1.0, 1.2, 1.1, 0.9, 1.05, 1.15, 0.95])
    grad_mu = state["grad_mu"][:2]
    U, DU = _build_state_vectors(F, p, mu, grad_mu, 0.98)

    elem = CompiledElement(mod, props=props, dof_per_node=4, n_svars=0,
                           mcrd=2, n_elem=1, dt=dt, backend='native')
    R, K = elem.element_rk(0, Q8_NODES, U, DU)

    rhs_ref, amatrx_ref = assemble_element(
        problem, Q8_NODES.T, U, DU, dt, props, element="quad8"
    )

    assert np.allclose(R, -rhs_ref, atol=1e-10, rtol=0.0)
    assert np.allclose(K, amatrx_ref, atol=1e-10, rtol=0.0)
