"""Phase 10 Batch 2.11: Cui phase-field corrosion Quad8R."""
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

_BUILD_PY = Path(__file__).parent.parent / "examples" / "phasefield_corrosion_cui" / "build.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("cui_build", _BUILD_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Q8R_NODES = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.5, 0.0],
    [1.0, 0.5],
    [0.5, 1.0],
    [0.0, 0.5],
], dtype=float)


def _build_state_vectors(state):
    """Build U and DU for the Quad8R node-major u-phi-c layout."""
    F2 = np.asarray(state["F"], dtype=float)[:2, :2]
    gphi = np.asarray(state["grad_phi"], dtype=float)[:2]
    gc = np.asarray(state["grad_c"], dtype=float)[:2]

    phi0 = state["phi"] - gphi[0] * 0.5 - gphi[1] * 0.5
    c0 = state["c"] - gc[0] * 0.5 - gc[1] * 0.5

    U = np.zeros(32)
    DU = np.zeros(32)
    for a, X in enumerate(Q8R_NODES):
        u = (F2 - np.eye(2)) @ X
        phi = phi0 + gphi[0] * X[0] + gphi[1] * X[1]
        c = c0 + gc[0] * X[0] + gc[1] * X[1]
        base = 4 * a
        U[base:base + 2] = u
        U[base + 2] = phi
        U[base + 3] = c
        DU[base:base + 2] = u
        DU[base + 2] = phi - state["phi_old"]
        DU[base + 3] = c - state["c_old"]
    return U, DU


def _svars_from_state(state, schema, n_gp):
    """Flatten per-GP state values into the SVARS layout."""
    per_gp = sum(entry['size'] for entry in schema.values())
    svars = np.zeros(n_gp * per_gp, dtype=float)
    for gp in range(n_gp):
        base = gp * per_gp
        for name, entry in schema.items():
            off = base + entry['offset']
            val = state[f"{name}_old"]
            if entry['size'] == 1:
                svars[off] = float(val)
            else:
                svars[off:off + entry['size']] = np.asarray(val, dtype=float).ravel()
    return svars


def test_material_verifies():
    ex = _load_example_module()
    assert ex.CuiJ2CorrosionFull().verify(
        state=ex.verification_state(), verbose=False)


def test_uel_generates_and_compiles(tmp_path):
    if not shutil.which("gfortran"):
        pytest.skip("gfortran not available")

    ex = _load_example_module()
    problem = ex.CuiJ2CorrosionFull()
    props = np.asarray(ex.DEFAULT_PROPS, dtype=float)
    state = ex.verification_state()
    dt = float(state["dt"])
    schema = problem._mat.state_schema

    uel_for = tmp_path / "cui_q8r_full_uel.for"
    generate_uel(problem, str(uel_for), element='Quad8R', formulation='standard')
    mod = build_element_kernel(str(uel_for), "cui_q8r_full_compile",
                               workdir=str(tmp_path / "build"))

    U, DU = _build_state_vectors(state)
    elem = CompiledElement(
        mod, props=props, dof_per_node=4, n_svars=60,
        mcrd=2, n_elem=1, dt=dt, backend='abaqus_uel',
        state_schema=schema,
    )
    elem.svars[0, :] = _svars_from_state(state, schema, n_gp=4)
    R, K = elem.element_rk(0, Q8R_NODES, U, DU)
    assert np.all(np.isfinite(R))
    assert np.all(np.isfinite(K))


def test_production_variant_compiles(tmp_path):
    if not shutil.which("gfortran"):
        pytest.skip("gfortran not available")

    ex = _load_example_module()
    problem = ex.CuiJ2Corrosion()
    props = np.asarray(ex.DEFAULT_PROPS, dtype=float)
    state = ex.verification_state()
    dt = float(state["dt"])

    uel_for = tmp_path / "cui_q8r_prod_uel.for"
    generate_uel(problem, str(uel_for), element='Quad8R', formulation='standard')
    mod = build_element_kernel(str(uel_for), "cui_q8r_prod_compile",
                               workdir=str(tmp_path / "build"))

    U, DU = _build_state_vectors(state)
    elem = CompiledElement(
        mod, props=props, dof_per_node=4, n_svars=60,
        mcrd=2, n_elem=1, dt=dt, backend='abaqus_uel',
        state_schema=problem._mat.state_schema,
    )
    R, K = elem.element_rk(0, Q8R_NODES, U, DU)
    assert np.all(np.isfinite(R))
    assert np.all(np.isfinite(K))


def test_native_vs_uel_sign_convention(tmp_path):
    if not shutil.which("gfortran"):
        pytest.skip("gfortran not available")

    ex = _load_example_module()
    problem = ex.CuiJ2CorrosionFull()
    props = np.asarray(ex.DEFAULT_PROPS, dtype=float)
    state = ex.verification_state()
    dt = float(state["dt"])
    schema = problem._mat.state_schema

    native_for = tmp_path / "cui_q8r_native.for"
    uel_for = tmp_path / "cui_q8r_uel.for"
    generate_element(problem, str(native_for), element='Quad8R',
                     formulation='standard', backend='native')
    generate_uel(problem, str(uel_for), element='Quad8R',
                 formulation='standard')

    mod_n = build_element_kernel(str(native_for), "cui_q8r_native_sign",
                                 workdir=str(tmp_path / "build_native"))
    mod_u = build_element_kernel(str(uel_for), "cui_q8r_uel_sign",
                                 workdir=str(tmp_path / "build_uel"))

    U, DU = _build_state_vectors(state)
    svars = _svars_from_state(state, schema, n_gp=4)

    native_elem = CompiledElement(
        mod_n, props=props, dof_per_node=4, n_svars=60,
        mcrd=2, n_elem=1, dt=dt, backend='native',
        state_schema=schema,
    )
    uel_elem = CompiledElement(
        mod_u, props=props, dof_per_node=4, n_svars=60,
        mcrd=2, n_elem=1, dt=dt, backend='abaqus_uel',
        state_schema=schema,
    )
    native_elem.svars[0, :] = svars
    uel_elem.svars[0, :] = svars

    assert native_elem.has_residual_only
    R, K = native_elem.element_rk(0, Q8R_NODES, U, DU)
    state_from_joint = native_elem.svars_trial.copy()
    R_only = native_elem.element_r(0, Q8R_NODES, U, DU)
    state_from_residual = native_elem.svars_trial.copy()

    # Both native entries start from the same committed state. The residual
    # and trial state must therefore be identical, including for this
    # time-dependent, stateful Quad8R material.
    np.testing.assert_array_equal(R_only, R)
    np.testing.assert_array_equal(state_from_residual, state_from_joint)

    R_batch, _ = native_elem.element_rk_batch(
        Q8R_NODES[None, :, :], U[None, :], DU[None, :]
    )
    state_from_joint_batch = native_elem.svars_trial.copy()
    R_only_batch = native_elem.element_r_batch(
        Q8R_NODES[None, :, :], U[None, :], DU[None, :]
    )
    state_from_residual_batch = native_elem.svars_trial.copy()
    np.testing.assert_array_equal(R_only_batch, R_batch)
    np.testing.assert_array_equal(
        state_from_residual_batch, state_from_joint_batch
    )

    rhs, amatrx = uel_elem.element_rk(0, Q8R_NODES, U, DU)

    assert np.allclose(R, rhs, atol=1e-9, rtol=0.0)
    assert np.allclose(K, amatrx, atol=1e-9, rtol=0.0)
    assert not np.allclose(R, -rhs, atol=1e-6, rtol=0.0)


def test_state_commit_round_trip(tmp_path):
    if not shutil.which("gfortran"):
        pytest.skip("gfortran not available")

    ex = _load_example_module()
    problem = ex.CuiJ2CorrosionFull()
    props = np.asarray(ex.DEFAULT_PROPS, dtype=float)
    state = ex.verification_state()
    dt = float(state["dt"])
    schema = problem._mat.state_schema

    native_for = tmp_path / "cui_q8r_native_commit.for"
    uel_for = tmp_path / "cui_q8r_uel_commit.for"
    generate_element(problem, str(native_for), element='Quad8R',
                     formulation='standard', backend='native')
    generate_uel(problem, str(uel_for), element='Quad8R',
                 formulation='standard')

    mod_n = build_element_kernel(str(native_for), "cui_q8r_native_commit",
                                 workdir=str(tmp_path / "build_native"))
    mod_u = build_element_kernel(str(uel_for), "cui_q8r_uel_commit",
                                 workdir=str(tmp_path / "build_uel"))

    U, DU = _build_state_vectors(state)
    svars = _svars_from_state(state, schema, n_gp=4)

    native_elem = CompiledElement(
        mod_n, props=props, dof_per_node=4, n_svars=60,
        mcrd=2, n_elem=1, dt=dt, backend='native',
        state_schema=schema,
    )
    uel_elem = CompiledElement(
        mod_u, props=props, dof_per_node=4, n_svars=60,
        mcrd=2, n_elem=1, dt=dt, backend='abaqus_uel',
        state_schema=schema,
    )
    native_elem.svars[0, :] = svars
    uel_elem.svars[0, :] = svars

    native_elem.element_rk(0, Q8R_NODES, U, DU)
    uel_elem.element_rk(0, Q8R_NODES, U, DU)
    native_elem.commit()
    uel_elem.commit()

    np.testing.assert_allclose(native_elem.svars, uel_elem.svars, atol=1e-10)


def test_bad_init_broken_control(tmp_path):
    """A pathological init (xL -> 0) must not agree with the declared init."""
    if not shutil.which("gfortran"):
        pytest.skip("gfortran not available")

    ex = _load_example_module()
    problem = ex.CuiJ2CorrosionFull()
    props = np.asarray(ex.DEFAULT_PROPS, dtype=float)
    state = ex.verification_state()
    dt = float(state["dt"])
    schema = problem._mat.state_schema

    uel_for = tmp_path / "cui_q8r_uel_init.for"
    generate_uel(problem, str(uel_for), element='Quad8R', formulation='standard')
    mod = build_element_kernel(str(uel_for), "cui_q8r_uel_init",
                               workdir=str(tmp_path / "build"))

    U, DU = _build_state_vectors(state)

    # declared init: schema sets xL = L0 > 0 -> physically consistent residual
    elem_decl = CompiledElement(
        mod, props=props, dof_per_node=4, n_svars=60,
        mcrd=2, n_elem=1, dt=dt, backend='abaqus_uel',
        state_schema=schema,
    )
    R_decl, _ = elem_decl.element_rk(0, Q8R_NODES, U, DU)
    assert np.all(np.isfinite(R_decl))

    # pathological init: tiny positive xL bypasses the <=0 fallback and blows up
    # the phase-storage term.  The residual is still finite but wildly wrong.
    elem_bad = CompiledElement(
        mod, props=props, dof_per_node=4, n_svars=60,
        mcrd=2, n_elem=1, dt=dt, backend='abaqus_uel',
        state_schema=schema,
    )
    elem_bad.svars.fill(0.0)
    # Offsets from schema: xL is offset 12 within each per-GP block.
    per_gp = sum(entry['size'] for entry in schema.values())
    for gp in range(4):
        elem_bad.svars[0, gp * per_gp + schema['xL']['offset']] = 1.0e-12
    R_bad, _ = elem_bad.element_rk(0, Q8R_NODES, U, DU)
    assert np.all(np.isfinite(R_bad))
    rel_err = np.linalg.norm(R_bad - R_decl) / (np.linalg.norm(R_decl) + 1e-30)
    assert rel_err > 1.0e-3


def test_reference_assembly_residual(tmp_path):
    if not shutil.which("gfortran"):
        pytest.skip("gfortran not available")

    ex = _load_example_module()
    problem = ex.CuiJ2CorrosionFull()
    props = np.asarray(ex.DEFAULT_PROPS, dtype=float)
    state = ex.verification_state()
    dt = float(state["dt"])
    schema = problem._mat.state_schema

    uel_for = tmp_path / "cui_q8r_uel_ref.for"
    generate_uel(problem, str(uel_for), element='Quad8R', formulation='standard')
    mod = build_element_kernel(str(uel_for), "cui_q8r_uel_ref",
                               workdir=str(tmp_path / "build"))

    U, DU = _build_state_vectors(state)
    svars = _svars_from_state(state, schema, n_gp=4)

    elem = CompiledElement(
        mod, props=props, dof_per_node=4, n_svars=60,
        mcrd=2, n_elem=1, dt=dt, backend='abaqus_uel',
        state_schema=schema,
    )
    elem.svars[0, :] = svars
    R, K = elem.element_rk(0, Q8R_NODES, U, DU)

    rhs_ref, amatrx_ref = assemble_element(
        problem, Q8R_NODES.T, U, DU, dt, props, STATEV=svars,
        element="quad8r"
    )

    assert np.allclose(R, -rhs_ref, atol=1e-7, rtol=0.0)
    assert np.allclose(K, amatrx_ref, atol=1e-7, rtol=0.0)


def test_uel_fd_tangent_consistency(tmp_path):
    if not shutil.which("gfortran"):
        pytest.skip("gfortran not available")

    from _fd_tangent import check_tangent_consistency

    ex = _load_example_module()
    problem = ex.CuiJ2CorrosionFull()
    props = np.asarray(ex.DEFAULT_PROPS, dtype=float)
    state = ex.verification_state()
    dt = float(state["dt"])
    schema = problem._mat.state_schema

    uel_for = tmp_path / "cui_q8r_uel_fd.for"
    generate_uel(problem, str(uel_for), element='Quad8R', formulation='standard')
    mod = build_element_kernel(str(uel_for), "cui_q8r_uel_fd",
                               workdir=str(tmp_path / "build"))

    U, DU = _build_state_vectors(state)
    svars = _svars_from_state(state, schema, n_gp=4)

    elem = CompiledElement(
        mod, props=props, dof_per_node=4, n_svars=60,
        mcrd=2, n_elem=1, dt=dt, backend='abaqus_uel',
        state_schema=schema,
    )
    elem.svars[0, :] = svars
    _, _, rel_err, _ = check_tangent_consistency(
        elem, 0, Q8R_NODES, U, DU, h=1.0e-6, rtol=1.0e-5, atol=1.0e-8)
    assert rel_err < 1.0e-5


def test_uel_fd_broken_control_species_flux_tangent_block(tmp_path):
    """Flip only the d(species_flux)/d(grad_c) tangent block."""
    if not shutil.which("gfortran"):
        pytest.skip("gfortran not available")

    from _fd_tangent import check_tangent_consistency

    ex = _load_example_module()
    problem = ex.CuiJ2CorrosionFull()
    props = np.asarray(ex.DEFAULT_PROPS, dtype=float)
    state = ex.verification_state()
    dt = float(state["dt"])
    schema = problem._mat.state_schema

    clean_for = tmp_path / "cui_q8r_uel_clean.for"
    corrupt_for = tmp_path / "cui_q8r_uel_corrupt.for"
    generate_uel(problem, str(clean_for), element='Quad8R', formulation='standard')

    text = clean_for.read_text()
    target = "dspecies_flux_dgrad_c(i,k) = AIMAG(jRz(i)) / CS_H"
    assert target in text, "expected species-flux tangent block not found"
    text = text.replace(target, "dspecies_flux_dgrad_c(i,k) = -AIMAG(jRz(i)) / CS_H", 1)
    corrupt_for.write_text(text)

    U, DU = _build_state_vectors(state)
    svars = _svars_from_state(state, schema, n_gp=4)

    # clean kernel passes
    mod_clean = build_element_kernel(str(clean_for), "cui_q8r_uel_clean_fd",
                                     workdir=str(tmp_path / "build_clean"))
    elem_clean = CompiledElement(
        mod_clean, props=props, dof_per_node=4, n_svars=60,
        mcrd=2, n_elem=1, dt=dt, backend='abaqus_uel',
        state_schema=schema,
    )
    elem_clean.svars[0, :] = svars
    _, _, rel_clean, _ = check_tangent_consistency(
        elem_clean, 0, Q8R_NODES, U, DU, h=1.0e-6, rtol=1.0e-5, atol=1.0e-8)
    assert rel_clean < 1.0e-5

    # corrupted kernel fails
    mod_corrupt = build_element_kernel(str(corrupt_for), "cui_q8r_uel_corrupt_fd",
                                       workdir=str(tmp_path / "build_corrupt"))
    elem_corrupt = CompiledElement(
        mod_corrupt, props=props, dof_per_node=4, n_svars=60,
        mcrd=2, n_elem=1, dt=dt, backend='abaqus_uel',
        state_schema=schema,
    )
    elem_corrupt.svars[0, :] = svars
    _, _, rel_corrupt, _ = check_tangent_consistency(
        elem_corrupt, 0, Q8R_NODES, U, DU, h=1.0e-6, rtol=1.0e-5, atol=1.0e-8)
    assert rel_corrupt > 1.0e-3
