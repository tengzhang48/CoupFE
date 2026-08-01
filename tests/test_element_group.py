"""Harness-style gates for the compiled neo-Hookean :class:`ElementGroup`.

Three independent oracles, in the CoupFE spirit (a check is only trusted once it
**fails** on a reintroduced bug — each has a broken control):

1. **Constant-stress patch test.**  An affine displacement ``u = (F-I) X`` imposed on
   the boundary must be reproduced exactly at every interior node (constant stress ⇒
   the affine field is the exact discrete solution).  Independent of the element's
   internals.  Broken control: a *non-affine* boundary field is NOT reproduced.

2. **The core CoupFE invariant.**  The assembled COO tangent equals a finite-
   difference of the assembled residual.  Broken control: a transposed/scaled tangent
   fails the same check (so the test has teeth on a real tangent bug).

3. **Newton converges** to a zero residual on a finite-strain uniaxial solve, and the
   FE lateral contraction matches the analytic traction-free stretch (a second,
   physical oracle).
"""

import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp

EX = os.path.join(os.path.dirname(__file__), "..", "examples", "neo_hookean_block")
sys.path.insert(0, EX)

pytest.importorskip("numpy")

from block import (  # noqa: E402
    DEFAULT_PROPS, make_group, structured_quad_mesh, uniaxial_problem)
from run import lateral_stretch_analytic  # noqa: E402

from coupfe import (  # noqa: E402
    assemble_residual, assemble_tangent, newton_solve, solve_increments)
from coupfe.operators.element_group import ElementGroup  # noqa: E402


# A compiled kernel is required; skip cleanly if the Fortran toolchain is absent.
try:
    _G = make_group(*structured_quad_mesh(2, 2))
    _HAVE_KERNEL = True
except Exception as exc:  # pragma: no cover - environment-dependent
    _HAVE_KERNEL = False
    _WHY = str(exc)

pytestmark = pytest.mark.skipif(
    not _HAVE_KERNEL,
    reason="neo-Hookean kernel did not compile (needs numpy.f2py + gfortran)"
    + (f": {_WHY}" if not _HAVE_KERNEL else ""))


def _affine_bc(nodes, F):
    """Dirichlet dict imposing ``u = (F-I) X`` on the bounding-box boundary."""
    tol = 1e-9
    x, y = nodes[:, 0], nodes[:, 1]
    on = ((np.abs(x - x.min()) < tol) | (np.abs(x - x.max()) < tol)
          | (np.abs(y - y.min()) < tol) | (np.abs(y - y.max()) < tol))
    bnd = np.nonzero(on)[0]
    A = F - np.eye(2)
    d = {}
    for n in bnd:
        u = A @ nodes[n]
        d[n * 2 + 0] = float(u[0])
        d[n * 2 + 1] = float(u[1])
    return d, bnd


def test_constant_stress_patch():
    """Affine boundary field ⇒ exact affine interior (constant-stress patch test)."""
    nodes, elems = structured_quad_mesh(4, 3, 1.3, 0.9)
    group = make_group(nodes, elems)
    ndof = len(nodes) * 2
    F = np.array([[1.08, 0.04], [0.0, 0.95]])           # arbitrary affine gradient
    d, _ = _affine_bc(nodes, F)
    U, nit = solve_increments([group], np.zeros(ndof), ndof, d, n_steps=2)
    A = F - np.eye(2)
    u_exact = (nodes @ A.T).ravel()
    assert np.allclose(U, u_exact, atol=1e-9, rtol=0.0)
    assert nit <= 12                                     # total over the increments

    # Broken control: a non-affine boundary field is NOT reproduced as the affine one.
    d_bad = dict(d)
    some = list(d_bad)[0]
    d_bad[some] += 0.05
    U_bad, _ = solve_increments([group], np.zeros(ndof), ndof, d_bad, n_steps=2)
    assert not np.allclose(U_bad, u_exact, atol=1e-6)


def _assembled_residual(group, U, ndof):
    R, _ = assemble_residual([group], U, None, 1.0, 1.0, ndof)
    return R


def test_assembled_tangent_matches_fd_of_residual():
    """Core invariant: assembled tangent == FD of the assembled residual."""
    nodes, elems = structured_quad_mesh(3, 2, 1.1, 0.7)
    group = make_group(nodes, elems)
    ndof = len(nodes) * 2
    rng = np.random.default_rng(0)
    U = 0.05 * rng.standard_normal(ndof)                # arbitrary nonlinear state

    K = assemble_tangent([group], U, None, 1.0, 1.0, ndof).toarray()

    # Central finite difference of the assembled residual, column by column.
    h = 1e-6
    Kfd = np.empty((ndof, ndof))
    for j in range(ndof):
        Up, Um = U.copy(), U.copy()
        Up[j] += h
        Um[j] -= h
        Kfd[:, j] = (_assembled_residual(group, Up, ndof)
                     - _assembled_residual(group, Um, ndof)) / (2.0 * h)
    scale = max(np.abs(K).max(), 1.0)
    assert np.allclose(K, Kfd, atol=1e-5 * scale, rtol=0.0)

    # Broken control: a transposed tangent (non-symmetric F-bar block ⇒ K != K^T)
    # fails the same FD check, so the test has teeth on a real assembly bug.
    if not np.allclose(K, K.T, atol=1e-8 * scale):
        assert not np.allclose(K.T, Kfd, atol=1e-5 * scale, rtol=0.0)


def test_newton_converges_and_matches_analytic_lateral():
    """Finite-strain uniaxial solve converges to R≈0; lateral stretch matches theory."""
    nx = ny = 6
    Lx = Ly = 1.0
    stretch = 0.20
    nodes, elems, group, dirichlet = uniaxial_problem(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, stretch=stretch)
    ndof = len(nodes) * 2
    # 20% finite strain → ramp over load increments (a single step overshoots into
    # element inversion; see solve_increments).
    U, nit = solve_increments([group], np.zeros(ndof), ndof, dirichlet, n_steps=4)

    # Residual at the free DOFs is ~0.
    R = _assembled_residual(group, U, ndof)
    free = np.ones(ndof, dtype=bool)
    free[np.array(sorted(dirichlet))] = False
    assert np.linalg.norm(R[free]) < 1e-7
    assert nit < 40                                      # total over 4 increments

    uy = U[1::2]
    top = np.nonzero(np.abs(nodes[:, 1] - Ly) < 1e-9)[0]
    lam_t_fe = 1.0 + uy[top].mean() / Ly
    lam_t_ref = lateral_stretch_analytic(1.0 + stretch, DEFAULT_PROPS)
    assert abs(lam_t_fe - lam_t_ref) / lam_t_ref < 5e-3


def test_rk_fusion_bit_identical_and_fewer_kernel_calls():
    """R/K fusion (default ON) is bit-identical to independent evaluation AND
    runs the compiled kernel strictly fewer times over a full Newton solve.

    Broken control: with fusion OFF the kernel is called ~2x more (residual +
    tangent evaluated separately at each iterate).
    """
    import coupfe.runtime.compiled_element as CE

    orig = CE.CompiledElement.element_rk_batch

    def run(fuse):
        calls = {"n": 0}

        def counted(self, *a, **k):
            calls["n"] += 1
            return orig(self, *a, **k)

        CE.CompiledElement.element_rk_batch = counted
        try:
            nodes, elems, group, dirichlet = uniaxial_problem(
                nx=6, ny=6, Lx=1.0, Ly=1.0, stretch=0.20)
            group.fuse_rk = fuse
            ndof = len(nodes) * 2
            U, _ = solve_increments([group], np.zeros(ndof), ndof, dirichlet,
                                    n_steps=4)
            return U, calls["n"]
        finally:
            CE.CompiledElement.element_rk_batch = orig

    U_on, n_on = run(True)
    U_off, n_off = run(False)
    assert np.array_equal(U_on, U_off), "fusion changed the solution"
    assert n_on < n_off, f"fusion did not reduce kernel calls ({n_on} vs {n_off})"
    # residual+tangent at a shared iterate fuse; line-search residuals still miss
    assert n_off >= 1.5 * n_on


def test_rk_fusion_invalidates_on_prop_change():
    """The fusion key includes props: mutating a material prop between a
    residual and its paired tangent must NOT serve a stale tangent.

    This is the correctness guard for schemes (e.g. per-step active stress)
    that change props during a solve — the key/commit invalidation is what
    makes fusion safe to leave ON by default.
    """
    nodes, elems, group, dirichlet = uniaxial_problem(
        nx=4, ny=4, Lx=1.0, Ly=1.0, stretch=0.05)
    ndof = len(nodes) * 2
    U = np.zeros(ndof)
    for g, v in dirichlet.items():
        U[g] = v

    props0 = np.asarray(group.element.props, dtype=float).copy()
    R1 = group.residual(U, None, 0.0, 1.0)                # fills the cache
    # perturb a prop (shear modulus) and ask for the tangent at the SAME U
    group.element.props = props0.copy()
    group.element.props[0] *= 1.5
    K_pert = group.tangent(U, None, 0.0, 1.0).values
    # independent reference at the perturbed props
    group.element.props = props0.copy()
    group.element.props[0] *= 1.5
    group.fuse_rk = False
    K_ref = group.tangent(U, None, 0.0, 1.0).values
    assert np.allclose(K_pert, K_ref), "stale tangent served after prop change"
    # sanity: the perturbed tangent actually differs from the base one
    group.element.props = props0.copy()
    K_base = group.tangent(U, None, 0.0, 1.0).values
    assert not np.allclose(K_pert, K_base)


def test_explicit_split_mode_uses_residual_only_then_joint_tangent(monkeypatch):
    nodes, elems = structured_quad_mesh(2, 2)
    joint_group = make_group(nodes, elems)
    element = joint_group.element
    assert element.has_element_r_batch
    group = ElementGroup(
        element,
        nodes,
        elems,
        dof_per_node=2,
        evaluation_mode="split",
    )
    calls = {"r": 0, "rk": 0}
    original_r = element.element_r_batch
    original_rk = element.element_rk_batch

    def counted_r(*args, **kwargs):
        calls["r"] += 1
        return original_r(*args, **kwargs)

    def counted_rk(*args, **kwargs):
        calls["rk"] += 1
        return original_rk(*args, **kwargs)

    monkeypatch.setattr(element, "element_r_batch", counted_r)
    monkeypatch.setattr(element, "element_rk_batch", counted_rk)
    U = np.zeros(len(nodes) * 2)

    residual = group.residual(U, None, 0.0, 1.0)
    assert calls == {"r": 1, "rk": 0}
    tangent = group.tangent(U, None, 0.0, 1.0)
    assert calls == {"r": 1, "rk": 1}
    assert residual.values.shape == (len(elems) * 8,)
    assert tangent.values.shape == (len(elems) * 64,)


def test_element_group_rejects_implicit_auto_selection():
    nodes, elems = structured_quad_mesh(1, 1)
    with pytest.raises(ValueError, match="must be 'joint' or 'split'"):
        ElementGroup(
            make_group(nodes, elems).element,
            nodes,
            elems,
            dof_per_node=2,
            evaluation_mode="auto",
        )


def test_joint_mode_retains_one_cached_call_per_paired_iterate(monkeypatch):
    nodes, elems = structured_quad_mesh(2, 2)
    group = make_group(nodes, elems)
    calls = {"rk": 0}
    original = group.element.element_rk_batch

    def counted(*args, **kwargs):
        calls["rk"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(group.element, "element_rk_batch", counted)
    U = np.zeros(len(nodes) * 2)
    group.residual(U, None, 0.0, 1.0)
    group.tangent(U, None, 0.0, 1.0)
    assert calls["rk"] == 1


class _StatefulElementProbe:
    """Minimal stateful batch whose trial state identifies its last iterate."""

    props = np.array([1.0])
    has_element_r_batch = True

    def __init__(self):
        self.svars = np.zeros((1, 1))
        self.svars_trial = self.svars.copy()
        self.evaluated = []

    def element_r_batch(self, coordinates, displacement, increment):
        value = float(displacement[0, 0])
        self.evaluated.append(value)
        self.svars_trial[0, 0] = value
        return np.array([[value]])

    def element_rk_batch(self, coordinates, displacement, increment):
        return self.element_r_batch(coordinates, displacement, increment), np.ones(
            (1, 1, 1)
        )

    def commit(self):
        self.svars = self.svars_trial.copy()


def test_stateful_commit_recomputes_trial_at_accepted_iterate():
    element = _StatefulElementProbe()
    group = ElementGroup(
        element,
        np.array([[0.0]]),
        np.array([[0]]),
        dof_per_node=1,
        evaluation_mode="split",
    )
    group.residual(np.array([9.0]), None, 0.0, 1.0)
    assert element.svars[0, 0] == 0.0
    assert element.svars_trial[0, 0] == 9.0

    state = group.commit(np.array([2.0]), None, 0.0, 1.0)
    assert element.evaluated[-1] == 2.0
    assert element.svars[0, 0] == 2.0
    np.testing.assert_array_equal(state.U_prev, [2.0])


def test_two_groups_compose_multimaterial_layout():
    """Two groups over disjoint elements, mixed-DOF layout, assemble into one system.

    Both groups here are u-only (comps=(0,1)) but the global layout carries an extra
    inert component (dof_per_node=3) — exercising the multi-material pattern: each
    group writes only its components and the spare DOF must be pinned by the caller.
    """
    nodes, elems = structured_quad_mesh(4, 2, 2.0, 1.0)
    elems = np.asarray(elems)
    cx = nodes[elems].mean(axis=1)[:, 0]
    left_e = elems[cx <= 1.0]
    right_e = elems[cx > 1.0]
    dpn = 3                                              # u_x, u_y, + 1 inert comp
    gL = make_group(nodes, left_e, dof_per_node=dpn, comps=(0, 1))
    gR = make_group(nodes, right_e, dof_per_node=dpn, comps=(0, 1))
    ndof = len(nodes) * dpn

    tol = 1e-9
    left = np.nonzero(np.abs(nodes[:, 0]) < tol)[0]
    right = np.nonzero(np.abs(nodes[:, 0] - 2.0) < tol)[0]
    corner = np.nonzero((np.abs(nodes[:, 0]) < tol) & (np.abs(nodes[:, 1]) < tol))[0]
    d = {}
    for n in left:
        d[n * dpn + 0] = 0.0
    for n in corner:
        d[n * dpn + 1] = 0.0
    for n in right:
        d[n * dpn + 0] = 0.3
    for n in range(len(nodes)):                          # pin the inert component
        d[n * dpn + 2] = 0.0

    U, _, nit = newton_solve([gL, gR], np.zeros(ndof), None, ndof, d)
    R, _ = assemble_residual([gL, gR], U, None, 1.0, 1.0, ndof)
    free = np.ones(ndof, dtype=bool)
    free[np.array(sorted(d))] = False
    assert np.linalg.norm(R[free]) < 1e-7
    assert nit < 12
    ux = U[0::dpn]
    assert ux.max() > 0.1                                # block actually stretched
