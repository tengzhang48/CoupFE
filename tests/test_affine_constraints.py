"""Focused gates for mesh-independent affine algebra and serial solve wiring."""

import importlib
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from coupfe import (
    ConstraintRelation,
    ConstraintTransform,
    Residual,
    Tangent,
    compile_affine_constraints,
    newton_solve,
    solve_dynamics,
    solve_dynamics_adaptive,
    solve_increments,
)


def test_affine_reduction_matches_independent_kkt_system():
    rng = np.random.default_rng(12)
    raw = rng.standard_normal((7, 7))
    tangent = raw.T @ raw + 2.0 * np.eye(7)
    force = rng.standard_normal(7)
    relations = (
        ConstraintRelation(5, (1, 2), (0.25, 0.75), 0.20, "weighted"),
        ConstraintRelation(6, (5,), (1.0,), -0.05, "chained"),
    )
    transform = compile_affine_constraints(
        7,
        relations,
        dirichlet={0: 0.3},
    )
    reduced_tangent, reduced_force = transform.reduce_linear_system(
        tangent, force
    )
    reduced_solution = spla.spsolve(reduced_tangent, reduced_force)
    full_solution = transform.lift(reduced_solution)

    # Independent oracle: solve the full saddle-point system instead of using
    # the elimination implementation under test.
    constraint_matrix, prescribed = transform.relation_matrix()
    kkt = sp.bmat(
        [
            [sp.csr_matrix(tangent), constraint_matrix.T],
            [constraint_matrix, None],
        ],
        format="csr",
    )
    kkt_solution = spla.spsolve(kkt, np.r_[force, prescribed])[:7]

    assert np.allclose(full_solution, kkt_solution, atol=2.0e-13, rtol=0.0)
    assert np.max(np.abs(transform.constraint_error(full_solution))) < 2.0e-14
    assert np.max(np.abs((constraint_matrix @ transform.P).toarray())) < 1.0e-14
    assert np.allclose(
        constraint_matrix @ transform.offset,
        prescribed,
        atol=1.0e-14,
        rtol=0.0,
    )


def test_restriction_preserves_virtual_work():
    rng = np.random.default_rng(18)
    transform = compile_affine_constraints(
        5,
        (
            ConstraintRelation(3, (0, 1), (0.4, 0.6)),
            ConstraintRelation(4, (3,), (1.0,), 0.2),
        ),
    )
    reduced_increment = rng.standard_normal(transform.reduced_ndof)
    residual = rng.standard_normal(transform.full_ndof)
    full_increment = transform.project_increment(reduced_increment)

    assert np.dot(full_increment, residual) == pytest.approx(
        np.dot(
            reduced_increment,
            transform.restrict_residual(residual),
        ),
        abs=2.0e-14,
    )


def test_compiler_is_deterministic_and_supports_fully_prescribed_system():
    relations = (
        ConstraintRelation(1, (0,), (2.0,), 0.25, "slave"),
    )
    first = compile_affine_constraints(2, relations, dirichlet={0: -0.5})
    second = compile_affine_constraints(2, relations, dirichlet={0: -0.5})

    assert first.sha256 == second.sha256
    assert first.reduced_ndof == 0
    assert first.lift(np.zeros(0)) == pytest.approx([-0.5, -0.75])
    reduced_tangent, reduced_force = first.reduce_linear_system(
        np.eye(2), np.zeros(2)
    )
    assert reduced_tangent.shape == (0, 0)
    assert reduced_force.shape == (0,)


def test_reduce_guess_projects_slave_values_without_a_linear_solve():
    transform = compile_affine_constraints(
        5,
        (
            ConstraintRelation(3, (0, 1), (0.4, 0.6), 0.1),
            ConstraintRelation(4, (3,), (1.0,), 0.2),
        ),
        dirichlet={2: -0.5},
    )
    inconsistent = np.array([1.0, 2.0, 99.0, -20.0, 40.0])
    reduced = transform.reduce_guess(inconsistent)
    projected = transform.lift(reduced)

    assert reduced == pytest.approx(inconsistent[[0, 1]])
    assert projected[:3] == pytest.approx([1.0, 2.0, -0.5])
    assert np.max(np.abs(transform.constraint_error(projected))) < 1.0e-14
    with pytest.raises(ValueError, match="full_guess"):
        transform.reduce_guess(np.zeros(4))
    with pytest.raises(ValueError, match="dq"):
        transform.project_increment(np.zeros(3))


def test_offset_scaling_preserves_relation_topology_and_rejects_nonfinite_scale():
    relation = ConstraintRelation(3, (0, 2), (0.25, 0.75), 0.4, "jump")
    scaled = relation.scaled_offset(0.25)

    assert scaled == ConstraintRelation(
        3,
        (0, 2),
        (0.25, 0.75),
        0.1,
        "jump",
    )
    with pytest.raises(ValueError, match="scale must be finite"):
        relation.scaled_offset(np.inf)


def test_compiler_rejects_malformed_relations_conflicts_and_cycles():
    with pytest.raises(ValueError, match="same length"):
        ConstraintRelation(2, (0,), ())
    with pytest.raises(ValueError, match="repeat"):
        ConstraintRelation(2, (0, 0), (0.5, 0.5))
    with pytest.raises(ValueError, match="finite"):
        ConstraintRelation(2, (0,), (np.nan,))
    with pytest.raises(TypeError, match="ConstraintRelation"):
        compile_affine_constraints(3, [(2, 0)])
    with pytest.raises(TypeError, match="ConstraintRelation"):
        solve_increments(
            [],
            np.zeros(3),
            3,
            {},
            n_steps=1,
            constraints=[(2, 0)],
        )
    with pytest.raises(ValueError, match="out of range"):
        compile_affine_constraints(3, [ConstraintRelation(3)])
    with pytest.raises(ValueError, match="multiple"):
        compile_affine_constraints(
            3,
            [
                ConstraintRelation(2, (0,), (1.0,)),
                ConstraintRelation(2, (1,), (1.0,)),
            ],
        )
    with pytest.raises(ValueError, match="cycle"):
        compile_affine_constraints(
            3,
            [
                ConstraintRelation(1, (2,), (1.0,)),
                ConstraintRelation(2, (1,), (1.0,)),
            ],
        )
    with pytest.raises(ValueError, match="already an MPC slave"):
        compile_affine_constraints(
            3,
            [ConstraintRelation(2, (0,), (1.0,))],
            dirichlet={2: 0.0},
        )


@pytest.mark.parametrize(
    "invalid_index",
    [True, np.bool_(False), "1", 1.0, 1.9, np.float64(2.0)],
)
def test_constraint_indices_reject_lossy_or_noninteger_inputs(invalid_index):
    with pytest.raises(TypeError, match="slave DOF must be an integer"):
        ConstraintRelation(invalid_index)
    with pytest.raises(TypeError, match="master DOF must be an integer"):
        ConstraintRelation(2, (invalid_index,), (1.0,))
    with pytest.raises(TypeError, match="ndof must be an integer"):
        compile_affine_constraints(invalid_index)
    with pytest.raises(TypeError, match="Dirichlet DOF must be an integer"):
        compile_affine_constraints(3, dirichlet={invalid_index: 0.0})


def test_constraint_indices_accept_numpy_integer_scalars():
    relation = ConstraintRelation(np.int64(2), (np.int32(1),), (1.0,))
    transform = compile_affine_constraints(np.int64(3), (relation,))

    assert relation.slave == 2
    assert relation.masters == (1,)
    assert transform.full_ndof == 3


class _LinearSystem:
    """Small full-space operator used to gate reduced Newton plumbing."""

    def __init__(self, force):
        self.force = np.asarray(force, dtype=float)
        self.max_step_calls = []
        self.committed = None

    def residual(self, U, state, t, dt):
        values = np.asarray(U, dtype=float) - self.force
        return Residual(np.arange(len(values)), values)

    def tangent(self, U, state, t, dt):
        dofs = np.arange(len(self.force))
        return Tangent(dofs, dofs, np.ones(len(dofs)))

    def max_step(self, U, dU, t):
        self.max_step_calls.append(
            (np.asarray(U).copy(), np.asarray(dU).copy(), float(t))
        )
        return 0.5 if len(self.max_step_calls) == 1 else 1.0

    def commit(self, U, state, t, dt):
        self.committed = np.asarray(U).copy()
        return {"accepted_full_solution": self.committed}


def test_reduced_newton_uses_solver_policy_full_increment_and_full_commit(
    monkeypatch,
):
    assembly = importlib.import_module("coupfe.assembly.assemble")
    policy_solve = assembly.linear_solve
    reduced_shapes = []

    def recording_solve(tangent, force):
        reduced_shapes.append(tangent.shape)
        return policy_solve(tangent, force)

    monkeypatch.setattr(assembly, "linear_solve", recording_solve)
    operator = _LinearSystem([0.0, 2.0, 0.0])
    relation = ConstraintRelation(2, (1,), (1.0,), 0.0, "equal")

    solution, state, nit = newton_solve(
        [operator],
        np.zeros(3),
        None,
        3,
        {0: 0.0},
        constraints=(relation,),
        t=2.5,
    )

    assert solution == pytest.approx([0.0, 1.0, 1.0])
    assert state[0]["accepted_full_solution"] == pytest.approx(solution)
    assert operator.committed.shape == (3,)
    assert nit == 3
    assert reduced_shapes == [(1, 1), (1, 1)]
    assert operator.max_step_calls
    for _full_u, full_increment, time in operator.max_step_calls:
        assert full_increment.shape == (3,)
        assert full_increment[0] == pytest.approx(0.0)
        assert full_increment[2] == pytest.approx(full_increment[1])
        assert time == pytest.approx(2.5)


class _ThermalBar:
    """Independent 1-D thermoelastic chain for a physical affine-BC gate."""

    def __init__(self, n=5, length=2.0, modulus=120.0, thermal_strain=0.012):
        self.x = np.linspace(0.0, length, n + 1)
        self.modulus = float(modulus)
        self.thermal_strain = float(thermal_strain)
        self.ndof = n + 1

    def stress(self, U):
        return self.modulus * (
            np.diff(U) / np.diff(self.x) - self.thermal_strain
        )

    def residual(self, U, state, t, dt):
        residual = np.zeros(self.ndof, dtype=np.asarray(U).dtype)
        for element, length in enumerate(np.diff(self.x)):
            stress = self.modulus * (
                (U[element + 1] - U[element]) / length
                - self.thermal_strain
            )
            residual[element : element + 2] += stress * np.array([-1.0, 1.0])
        return Residual(np.arange(self.ndof), residual)

    def tangent(self, U, state, t, dt):
        rows, cols, values = [], [], []
        for element, length in enumerate(np.diff(self.x)):
            element_tangent = self.modulus / length * np.array(
                [[1.0, -1.0], [-1.0, 1.0]]
            )
            dofs = (element, element + 1)
            for local_row, row in enumerate(dofs):
                for local_col, col in enumerate(dofs):
                    rows.append(row)
                    cols.append(col)
                    values.append(element_tangent[local_row, local_col])
        return Tangent(
            np.asarray(rows),
            np.asarray(cols),
            np.asarray(values),
        )

    def commit(self, U, state, t, dt):
        return np.asarray(U).copy()


def test_incremental_affine_jump_recovers_thermal_free_expansion_and_control():
    bar = _ThermalBar()
    jump = bar.thermal_strain * (bar.x[-1] - bar.x[0])
    free_expansion = (
        ConstraintRelation(
            bar.ndof - 1,
            (0,),
            (1.0,),
            jump,
            "thermal box jump",
        ),
    )
    solution, _ = solve_increments(
        [bar],
        np.zeros(bar.ndof),
        bar.ndof,
        {0: 0.0},
        constraints=free_expansion,
        n_steps=3,
    )

    assert solution == pytest.approx(
        bar.thermal_strain * bar.x,
        abs=2.0e-14,
    )
    assert np.max(np.abs(bar.stress(solution))) < 2.0e-12
    transform = compile_affine_constraints(
        bar.ndof,
        free_expansion,
        dirichlet={0: 0.0},
    )
    assert np.max(np.abs(transform.constraint_error(solution))) < 1.0e-14

    # Broken physical setup: a zero jump is a fixed box, not free expansion.
    fixed_box = (
        ConstraintRelation(bar.ndof - 1, (0,), (1.0,), 0.0),
    )
    constrained, _ = solve_increments(
        [bar],
        np.zeros(bar.ndof),
        bar.ndof,
        {0: 0.0},
        constraints=fixed_box,
        n_steps=2,
    )
    assert constrained == pytest.approx(np.zeros(bar.ndof), abs=2.0e-14)
    assert bar.stress(constrained) == pytest.approx(
        np.full(bar.ndof - 1, -bar.modulus * bar.thermal_strain),
        abs=2.0e-12,
    )


def test_incremental_callable_owns_nonproportional_constraint_schedule():
    operator = _LinearSystem([0.0, 0.0])
    fractions = []

    def schedule(fraction):
        fractions.append(fraction)
        return (
            ConstraintRelation(1, (0,), (1.0,), fraction**2),
        )

    solution, _ = solve_increments(
        [operator],
        np.zeros(2),
        2,
        {0: 0.0},
        constraints=schedule,
        n_steps=4,
    )

    assert fractions == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert solution == pytest.approx([0.0, 1.0])


def test_dynamics_fail_closed_before_affine_state_or_predictor_updates():
    relation = (ConstraintRelation(1, (0,), (1.0,)),)

    with pytest.raises(
        NotImplementedError,
        match="constraint-projected inertial/state update",
    ):
        solve_dynamics(
            [],
            np.zeros(2),
            2,
            {},
            dt=0.1,
            n_steps=1,
            constraints=relation,
        )
    with pytest.raises(
        NotImplementedError,
        match="constraint-projected inertial/state update",
    ):
        solve_dynamics_adaptive(
            [],
            np.zeros(2),
            2,
            {},
            t_end=0.1,
            dt_init=0.1,
            constraints=relation,
        )


def test_affine_symbols_are_top_level_public_api():
    assert ConstraintTransform.__module__ == "coupfe.constraints.affine"
    assert compile_affine_constraints.__module__ == "coupfe.constraints.affine"
