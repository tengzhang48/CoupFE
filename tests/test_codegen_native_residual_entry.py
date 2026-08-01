"""Gates for the native residual-only code-generation entry.

Native generated elements contain the existing joint ``coupfe_element_rk``
entry and a residual-only ``coupfe_element_r`` twin.  These checks keep the
joint entry stable, prove that the twin retains the same residual statements,
and ensure that Abaqus UEL generation remains separate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import numpy as np
import pytest

pytest.importorskip("sympy")

import coupfe.codegen as au  # noqa: E402
from coupfe.codegen import tensor  # noqa: E402
from coupfe.codegen.core.tensor import det, inv, log  # noqa: E402
from coupfe.codegen.generators.uel_gen import generate_element  # noqa: E402
from coupfe.runtime.compiled_element import (  # noqa: E402
    CompiledElement,
    build_element_kernel,
)


ROOT = Path(__file__).resolve().parent.parent
F_BAR_KERNEL = (
    ROOT / "coupfe" / "runtime" / "elements" / "neo_hookean_hex8_fbar.for"
)
Q4_F_BAR_KERNEL = (
    ROOT
    / "coupfe"
    / "runtime"
    / "elements"
    / "neo_hookean_q4_fbar_native.for"
)
STANDARD_KERNEL = (
    ROOT / "coupfe" / "runtime" / "elements" / "neo_hookean_q4_native.for"
)
ABAQUS_STANDARD_KERNEL = (
    ROOT / "coupfe" / "runtime" / "elements" / "neo_hookean_q4.for"
)

_RK_END = "      END SUBROUTINE coupfe_element_rk\n"
_R_HEADER = "C     coupfe_element_r (residual-only twin)"
_R_END = "      END SUBROUTINE coupfe_element_r\n"

# SHA-256 of each reviewed joint-entry prefix. The standard Quad4 and F-bar
# Hex8 values predate this feature; the F-bar Quad4 value locks the new native
# counterpart of the retained Abaqus UEL formulation. Merely editing the
# residual twin cannot mask joint-kernel drift.
_RK_PREFIX_SHA256 = {
    "hex8_fbar": "25a6106cd983fb96da5252be99e29faa02df4372f261e34e700eb42444812dd0",
    "quad4_fbar": "5bc72906c1c656731b9f64469ac93681c8f0a68d1699d6c74f898f75e5d597a7",
    "standard": "df85745128c69c9724b28f08d7eae0d7a5ff463e8e4bc2024ea5e10a9456398e",
}


class NeoHookean(au.Material):
    """Source form used to generate the long-standing public Quad4 kernel."""

    props = dict(G=1.0, K=100.0)

    def stress_PK1(self, F):
        J = tensor.det(F)
        return (
            self.G * (F - tensor.inv(F).T)
            + self.K * tensor.log(J) * tensor.inv(F).T
        )


NeoHookeanStandard = NeoHookean


class NeoQuad4(au.WeakForm):
    material = NeoHookeanStandard

    def define_fields(self):
        self.u = au.VectorField("u", degree=1)

    def momentum_equation(self, v, F):
        return self.material.stress_PK1(F)


class NeoHookean(au.Material):
    """Source form used to generate the public F-bar Hex8 kernel."""

    props = dict(G=0.5, K=50.0)

    def stress_PK1(self, F):
        J = det(F)
        FinvT = inv(F).T
        return self.G * (F - FinvT) + self.K * log(J) * FinvT


NeoHookeanFBar = NeoHookean


class NeoHex8(au.WeakForm):
    material = NeoHookeanFBar
    ndim = 3

    def define_fields(self):
        self.u = au.VectorField("u", degree=1)

    def momentum_equation(self, v, F):
        return self.material.stress_PK1(F)


class NeoQuad4FBar(au.WeakForm):
    material = NeoHookeanFBar

    def define_fields(self):
        self.u = au.VectorField("u", degree=1)

    def momentum_equation(self, v, F):
        return self.material.stress_PK1(F)


class StatefulNeoHookean(au.Material):
    """Small stateful law used only to gate the F-bar state contract."""

    props = dict(G=0.5, K=50.0)
    state_vars = dict(alpha=0.25)

    def stress_PK1(self, F, alpha_old, dt):
        J = det(F)
        FinvT = inv(F).T
        scale = 1.0 + alpha_old
        stress = scale * (
            self.G * (F - FinvT) + self.K * log(J) * FinvT
        )
        alpha_new = alpha_old + dt * log(J)
        return stress, {"alpha": alpha_new}


class StatefulNeoQuad4FBar(au.WeakForm):
    material = StatefulNeoHookean

    def define_fields(self):
        self.u = au.VectorField("u", degree=1)

    def momentum_equation(self, v, F):
        return self.material.stress_PK1(F)


def _generate(tmp_path, kind, backend="native"):
    if kind == "hex8_fbar":
        problem, element, formulation = NeoHex8(), "Hex8", "fbar_mechanics"
    elif kind == "quad4_fbar":
        problem, element, formulation = (
            NeoQuad4FBar(),
            "Quad4",
            "fbar_mechanics",
        )
    else:
        problem, element, formulation = NeoQuad4(), "Quad4", "standard"
    out = tmp_path / f"neo_{kind}_{backend}.for"
    generate_element(
        problem,
        str(out),
        element=element,
        formulation=formulation,
        backend=backend,
    )
    return out.read_text()


def _entry_blocks(source):
    rk_end = source.index(_RK_END) + len(_RK_END)
    r_start = source.index(_R_HEADER, rk_end)
    r_end = source.index(_R_END, r_start) + len(_R_END)
    return source[:rk_end], source[r_start:r_end]


def _without_residual_entry(source):
    _rk_source, r_source = _entry_blocks(source)
    return source.replace(r_source, "", 1)


@pytest.mark.parametrize(
    ("kind", "vendored"),
    (
        ("standard", STANDARD_KERNEL),
        ("quad4_fbar", Q4_F_BAR_KERNEL),
        ("hex8_fbar", F_BAR_KERNEL),
    ),
)
def test_native_generation_is_regenerable_and_rk_is_unchanged(
    tmp_path, kind, vendored
):
    generated = _generate(tmp_path, kind)
    assert generated == vendored.read_text()

    rk_source, _ = _entry_blocks(generated)
    digest = hashlib.sha256(rk_source.encode()).hexdigest()
    assert digest == _RK_PREFIX_SHA256[kind]


@pytest.mark.parametrize("kind", ("standard", "quad4_fbar", "hex8_fbar"))
def test_residual_twin_keeps_r_and_removes_tangent_work(tmp_path, kind):
    generated = _generate(tmp_path, kind)
    rk_source, r_source = _entry_blocks(generated)

    # R declarations, zeroing, and assembly statements remain in the same order.
    assert [line for line in r_source.splitlines() if "R(" in line] == [
        line for line in rk_source.splitlines() if "R(" in line
    ]
    assert "SVARS_OUT(i) = SVARS_IN(i)" in r_source

    assert "K(NDOFEL,NDOFEL)" not in r_source
    assert "_cs_tangent" not in r_source
    assert "_cs_tangents" not in r_source
    assert "K assembly" not in r_source
    assert "Tangent arrays" not in r_source

    if kind.endswith("fbar"):
        assert "Atang" not in r_source
        assert "dJbar_du" not in r_source
        assert "Q_all" not in r_source
        assert "inv33d" not in r_source


@pytest.mark.parametrize("kind", ("standard", "quad4_fbar", "hex8_fbar"))
def test_abaqus_generation_has_no_native_residual_entry(tmp_path, kind):
    abaqus_source = _generate(tmp_path, kind, backend="abaqus_uel")
    assert "SUBROUTINE UEL(" in abaqus_source
    assert "SUBROUTINE coupfe_element_r(" not in abaqus_source


def test_vendored_abaqus_kernel_is_byte_unchanged(tmp_path):
    out = tmp_path / "neo_q4_abaqus_fbar.for"
    generate_element(
        NeoQuad4(G=0.5, K=50.0),
        str(out),
        element="Quad4",
        formulation="fbar_mechanics",
        backend="abaqus_uel",
        mat_prefix="neohookean",
    )
    assert out.read_text() == ABAQUS_STANDARD_KERNEL.read_text()


_Q4_COORDS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float
)
_Q4_U = np.array([0.0, 0.0, 0.05, 0.01, 0.07, -0.02, 0.01, -0.03])

_H8_COORDS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=float,
)
_H8_U = np.column_stack(
    (
        0.04 * _H8_COORDS[:, 0] + 0.01 * _H8_COORDS[:, 1],
        -0.02 * _H8_COORDS[:, 1] + 0.005 * _H8_COORDS[:, 2],
        0.03 * _H8_COORDS[:, 2],
    )
).ravel()


@pytest.mark.parametrize(
    ("kind", "kernel", "coords", "u", "mcrd", "dof_per_node"),
    (
        ("standard", STANDARD_KERNEL, _Q4_COORDS, _Q4_U, 2, 2),
        ("quad4_fbar", Q4_F_BAR_KERNEL, _Q4_COORDS, _Q4_U, 2, 2),
        ("hex8_fbar", F_BAR_KERNEL, _H8_COORDS, _H8_U, 3, 3),
    ),
)
def test_compiled_residual_is_bit_identical_to_joint_entry(
    tmp_path, kind, kernel, coords, u, mcrd, dof_per_node
):
    if not all(shutil.which(tool) for tool in ("gfortran", "meson", "ninja")):
        pytest.skip("native Fortran build toolchain is unavailable")

    build_dir = tmp_path / kind
    module = build_element_kernel(
        str(kernel), f"coupfe_test_native_r_{kind}", workdir=str(build_dir)
    )
    element = CompiledElement(
        module,
        props=(0.5, 50.0),
        dof_per_node=dof_per_node,
        n_svars=0,
        mcrd=mcrd,
        n_elem=1,
    )
    assert element.has_residual_only

    du = 0.25 * u
    r_joint, _ = element.element_rk(0, coords, u, du)
    r_only = element.element_r(0, coords, u, du)
    assert np.array_equal(r_only, r_joint)

    r_joint_batch, _ = element.element_rk_batch(
        coords[None, :, :], u[None, :], du[None, :]
    )
    r_only_batch = element.element_r_batch(
        coords[None, :, :], u[None, :], du[None, :]
    )
    assert np.array_equal(r_only_batch, r_joint_batch)


def test_quad4_fbar_native_matches_retained_abaqus_uel(tmp_path):
    """The new native convenience kernel preserves the prior F-bar behavior."""
    if not all(shutil.which(tool) for tool in ("gfortran", "meson", "ninja")):
        pytest.skip("native Fortran build toolchain is unavailable")

    native_module = build_element_kernel(
        str(Q4_F_BAR_KERNEL),
        "coupfe_test_q4_fbar_native",
        workdir=str(tmp_path / "q4-fbar-native"),
    )
    uel_module = build_element_kernel(
        str(ABAQUS_STANDARD_KERNEL),
        "coupfe_test_q4_fbar_uel",
        workdir=str(tmp_path / "q4-fbar-uel"),
    )
    common = dict(
        props=(0.5, 50.0),
        dof_per_node=2,
        n_svars=0,
        mcrd=2,
        n_elem=1,
    )
    native = CompiledElement(native_module, **common)
    uel = CompiledElement(uel_module, **common)
    du = 0.25 * _Q4_U

    r_native, k_native = native.element_rk(0, _Q4_COORDS, _Q4_U, du)
    r_uel, k_uel = uel.element_rk(0, _Q4_COORDS, _Q4_U, du)
    np.testing.assert_allclose(r_native, r_uel, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(k_native, k_uel, rtol=0.0, atol=1.0e-12)


def test_stateful_fbar_residual_and_trial_state_match_joint_single_and_batch(
    tmp_path,
):
    if not all(shutil.which(tool) for tool in ("gfortran", "meson", "ninja")):
        pytest.skip("native Fortran build toolchain is unavailable")

    source = tmp_path / "stateful_q4_fbar_native.for"
    problem = StatefulNeoQuad4FBar()
    generate_element(
        problem,
        str(source),
        element="Quad4",
        formulation="fbar_mechanics",
        backend="native",
    )
    module = build_element_kernel(
        str(source),
        "coupfe_test_stateful_q4_fbar",
        workdir=str(tmp_path / "stateful-q4-fbar-build"),
    )
    element = CompiledElement(
        module,
        props=(0.5, 50.0),
        dof_per_node=2,
        n_svars=4,
        mcrd=2,
        n_elem=2,
        dt=0.2,
        state_schema=problem._mat.state_schema,
    )
    element.svars[0] = np.array([0.10, 0.20, 0.30, 0.40])
    element.svars[1] = np.array([0.15, 0.25, 0.35, 0.45])

    r_joint, _ = element.element_rk(0, _Q4_COORDS, _Q4_U, 0.25 * _Q4_U)
    state_joint = element.svars_trial.copy()
    r_only = element.element_r(0, _Q4_COORDS, _Q4_U, 0.25 * _Q4_U)
    state_only = element.svars_trial.copy()
    np.testing.assert_array_equal(r_only, r_joint)
    np.testing.assert_array_equal(state_only, state_joint)

    coords = np.repeat(_Q4_COORDS[None, :, :], 2, axis=0)
    u = np.stack((_Q4_U, 0.7 * _Q4_U))
    du = 0.25 * u
    r_joint_batch, _ = element.element_rk_batch(coords, u, du)
    state_joint_batch = element.svars_trial.copy()
    r_only_batch = element.element_r_batch(coords, u, du)
    state_only_batch = element.svars_trial.copy()
    np.testing.assert_array_equal(r_only_batch, r_joint_batch)
    np.testing.assert_array_equal(state_only_batch, state_joint_batch)


def test_pre_residual_entry_native_kernel_builds_and_falls_back(tmp_path):
    if not all(shutil.which(tool) for tool in ("gfortran", "meson", "ninja")):
        pytest.skip("native Fortran build toolchain is unavailable")
    old_kernel = tmp_path / "neo_q4_pre_r.for"
    old_kernel.write_text(_without_residual_entry(STANDARD_KERNEL.read_text()))
    module = build_element_kernel(
        str(old_kernel),
        "coupfe_test_native_pre_r_fallback",
        workdir=str(tmp_path / "old-native-build"),
    )
    element = CompiledElement(
        module,
        props=(0.5, 50.0),
        dof_per_node=2,
        n_svars=0,
        mcrd=2,
        n_elem=1,
    )
    assert not element.has_element_r
    assert not element.has_element_r_batch
    r_joint, _ = element.element_rk(0, _Q4_COORDS, _Q4_U, 0.25 * _Q4_U)
    r_fallback = element.element_r(0, _Q4_COORDS, _Q4_U, 0.25 * _Q4_U)
    assert np.array_equal(r_fallback, r_joint)
    r_joint_batch, _ = element.element_rk_batch(
        _Q4_COORDS[None, :, :],
        _Q4_U[None, :],
        (0.25 * _Q4_U)[None, :],
    )
    r_fallback_batch = element.element_r_batch(
        _Q4_COORDS[None, :, :],
        _Q4_U[None, :],
        (0.25 * _Q4_U)[None, :],
    )
    assert np.array_equal(r_fallback_batch, r_joint_batch)


def test_abaqus_uel_residual_fallback_preserves_adapter_sign(tmp_path):
    if not all(shutil.which(tool) for tool in ("gfortran", "meson", "ninja")):
        pytest.skip("native Fortran build toolchain is unavailable")
    module = build_element_kernel(
        str(ABAQUS_STANDARD_KERNEL),
        "coupfe_test_abaqus_r_fallback",
        workdir=str(tmp_path / "abaqus-build"),
    )
    element = CompiledElement(
        module,
        props=(0.5, 50.0),
        dof_per_node=2,
        n_svars=0,
        mcrd=2,
        n_elem=1,
    )
    assert element.backend == "abaqus_uel"
    assert not element.has_residual_only
    r_joint, _ = element.element_rk(0, _Q4_COORDS, _Q4_U, 0.25 * _Q4_U)
    r_fallback = element.element_r(0, _Q4_COORDS, _Q4_U, 0.25 * _Q4_U)
    assert np.array_equal(r_fallback, r_joint)
    r_joint_batch, _ = element.element_rk_batch(
        _Q4_COORDS[None, :, :],
        _Q4_U[None, :],
        (0.25 * _Q4_U)[None, :],
    )
    r_fallback_batch = element.element_r_batch(
        _Q4_COORDS[None, :, :],
        _Q4_U[None, :],
        (0.25 * _Q4_U)[None, :],
    )
    assert np.array_equal(r_fallback_batch, r_joint_batch)
