"""PETSc gate for joint versus residual/tangent-split distributed assembly."""
from __future__ import annotations

import numpy as np
import pytest


PETSc = pytest.importorskip("petsc4py.PETSc")

from coupfe.assembly.distributed import solve_distributed  # noqa: E402


def test_distributed_split_matches_joint_and_calls_residual_path():
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    size = comm.getSize()
    elements = np.array([[0, 1], [1, 2]], dtype=int)
    local_ids = np.flatnonzero(np.arange(len(elements)) % size == rank)
    my_gm = elements[local_ids]
    my_coordinates = np.zeros((len(local_ids), 2, 1))
    stiffness = np.array([[1.0, -1.0], [-1.0, 1.0]])
    calls = {"joint": 0, "residual": 0}

    def joint_batch(_coordinates, displacement, _increment):
        calls["joint"] += 1
        return (
            np.einsum("ij,ej->ei", stiffness, displacement),
            np.broadcast_to(
                stiffness, (len(displacement), 2, 2)
            ).copy(),
        )

    def residual_batch(_coordinates, displacement, _increment):
        calls["residual"] += 1
        return np.einsum("ij,ej->ei", stiffness, displacement)

    def dirichlet(fraction):
        return {0: 0.0, 2: float(fraction)}

    common = dict(
        ndof=3,
        my_gm=my_gm,
        my_coords=my_coordinates,
        dof_per_node=1,
        batch_fn=joint_batch,
        dirichlet_fn=dirichlet,
        n_steps=1,
        pc="jacobi",
        ksp_type="cg",
        forcing=False,
        rtol=1.0e-12,
        tol=1.0e-12,
    )
    joint_solution, joint_info = solve_distributed(
        **common,
        residual_batch_fn=residual_batch,
        evaluation_mode="joint",
    )
    calls_after_joint = calls.copy()
    split_solution, split_info = solve_distributed(
        **common,
        residual_batch_fn=residual_batch,
        evaluation_mode="split",
    )

    np.testing.assert_allclose(joint_solution, [0.0, 0.5, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(split_solution, joint_solution, atol=1.0e-12)
    assert joint_info["evaluation_mode"] == "joint"
    assert split_info["evaluation_mode"] == "split"
    assert calls_after_joint["residual"] == 0
    # A rank may own no element when the MPI size exceeds this two-element
    # mesh. Every rank with local work must use the residual callback; ranks 0
    # and 1 provide that gate at every tested size.
    if len(local_ids):
        assert calls["residual"] > 0


def test_distributed_explicit_split_requires_residual_callback():
    with pytest.raises(ValueError, match="requires residual_batch_fn"):
        solve_distributed(
            1,
            np.empty((0, 1), dtype=int),
            np.empty((0, 1, 1)),
            1,
            lambda *_args: (np.empty((0, 1)), np.empty((0, 1, 1))),
            lambda _fraction: {},
            1,
            evaluation_mode="split",
        )


def test_distributed_rejects_implicit_auto_selection():
    with pytest.raises(ValueError, match="must be 'joint' or 'split'"):
        solve_distributed(
            1,
            np.empty((0, 1), dtype=int),
            np.empty((0, 1, 1)),
            1,
            lambda *_args: (np.empty((0, 1)), np.empty((0, 1, 1))),
            lambda _fraction: {},
            1,
            evaluation_mode="auto",
        )
