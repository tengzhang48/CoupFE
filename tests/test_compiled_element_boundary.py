"""Backend-boundary gates for :mod:`coupfe.runtime.compiled_element`.

The CoupFE-native ABI and the Abaqus UEL ABI are parallel targets. These
tests use strict fake f2py modules so the native path fails immediately if an
Abaqus-only argument leaks into it, without requiring a Fortran compiler.
"""

from __future__ import annotations

import numpy as np

from coupfe.runtime.compiled_element import CompiledElement


class _NativeModule:
    def __init__(self):
        self.arg_counts = []

    @staticmethod
    def _joint(svars, _coords, u, _du, _props, _time, _dtime):
        r = np.asarray(u, dtype=float).copy()
        if r.ndim == 1:
            return r, np.eye(r.size), np.asarray(svars, dtype=float).copy()
        ndof, nelem = r.shape
        k = np.repeat(np.eye(ndof)[:, :, None], nelem, axis=2)
        return r, k, np.asarray(svars, dtype=float).copy()

    @staticmethod
    def _residual(svars, _coords, u, _du, _props, _time, _dtime):
        return (np.asarray(u, dtype=float).copy(),
                np.asarray(svars, dtype=float).copy())

    def drive_native(self, *args):
        self.arg_counts.append(("rk", len(args)))
        return self._joint(*args)

    def drive_native_batch(self, *args):
        self.arg_counts.append(("rk_batch", len(args)))
        return self._joint(*args)

    def drive_native_r(self, *args):
        self.arg_counts.append(("r", len(args)))
        return self._residual(*args)

    def drive_native_batch_r(self, *args):
        self.arg_counts.append(("r_batch", len(args)))
        return self._residual(*args)


class _UELModule:
    def __init__(self):
        self.lflags = []

    def drive_uel(self, *args):
        assert len(args) == 16
        svars, _coords, u = args[:3]
        self.lflags.append(np.asarray(args[9], dtype=int).copy())
        rhs = -np.asarray(u, dtype=float)
        return rhs, np.eye(rhs.size), np.asarray(svars).copy(), args[8]

    def drive_uel_batch(self, *args):
        assert len(args) == 15
        svars, _coords, u = args[:3]
        self.lflags.append(np.asarray(args[9], dtype=int).copy())
        rhs = -np.asarray(u, dtype=float)
        ndof, nelem = rhs.shape
        amatrx = np.repeat(np.eye(ndof)[:, :, None], nelem, axis=2)
        return rhs, amatrx, np.asarray(svars).copy(), args[8]


def _data(nelem=2):
    coords = np.array([
        [[0.0], [1.0]],
        [[1.0], [2.0]],
    ])[:nelem]
    u = np.array([[0.2, -0.1], [0.4, 0.3]])[:nelem]
    return coords, u, 0.25 * u


def test_native_runtime_has_no_abaqus_call_arguments():
    module = _NativeModule()
    element = CompiledElement(
        module,
        props=(2.0,),
        dof_per_node=1,
        n_svars=1,
        mcrd=1,
        n_elem=2,
        backend="native",
    )
    coords, u, du = _data()

    r, k = element.element_rk(0, coords[0], u[0], du[0])
    r_only = element.element_r(0, coords[0], u[0], du[0])
    rb, kb = element.element_rk_batch(coords, u, du)
    rb_only = element.element_r_batch(coords, u, du)

    np.testing.assert_array_equal(r, u[0])
    np.testing.assert_array_equal(r_only, r)
    np.testing.assert_array_equal(rb, u)
    np.testing.assert_array_equal(rb_only, rb)
    assert k.shape == (2, 2)
    assert kb.shape == (2, 2, 2)
    assert module.arg_counts == [
        ("rk", 7),
        ("r", 7),
        ("rk_batch", 7),
        ("r_batch", 7),
    ]
    assert not hasattr(element, "_uel_lflags")


def test_uel_adapter_is_isolated_normal_static_joint_call():
    module = _UELModule()
    element = CompiledElement(
        module,
        props=(2.0,),
        dof_per_node=1,
        n_svars=1,
        mcrd=1,
        n_elem=2,
        backend="abaqus_uel",
    )
    coords, u, du = _data()

    r, _k = element.element_rk(0, coords[0], u[0], du[0])
    rb, _kb = element.element_rk_batch(coords, u, du)

    np.testing.assert_array_equal(r, u[0])
    np.testing.assert_array_equal(rb, u)
    assert len(module.lflags) == 2
    for lflags in module.lflags:
        np.testing.assert_array_equal(lflags, [1, 1, 1, 0, 0, 0])
    assert not element.has_residual_only
