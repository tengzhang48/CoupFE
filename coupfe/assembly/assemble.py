"""Compose operators into a global system and solve it with Newton.

This is the minimal CoupFE driver: it knows nothing about elements, materials,
or contact — only the :class:`~coupfe.operators.base.Operator` contract.  Any
mix of operators (bulk element groups, contact, constraints, loads) assembles
and solves through the same path.  Linear solves route through
:func:`coupfe.assembly.factored.linear_solve` (scipy for small systems, PETSc
MUMPS above the size threshold — ONE policy, see that module); the MPI runtime
adds distribution but the composition contract is identical.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from coupfe.operators.base import Operator

from .factored import linear_solve


def _call_max_step(op, U, dU, t, dt=None):
    """Call an operator's optional ``max_step`` with time and optional step size.

    The extended signature ``max_step(U, dU, t, dt)`` lets an operator CCD-bound a
    predictor jump that spans a physical time interval (e.g. a moving obstacle).  The
    Newton line search uses ``dt=None`` because the correction is evaluated at the fixed
    step end time ``t``.
    """
    ms = getattr(op, "max_step", None)
    if ms is None:
        return 1.0
    if dt is not None:
        try:
            return float(ms(U, dU, t, dt))
        except TypeError:
            pass
    try:
        return float(ms(U, dU, t))
    except TypeError:
        return float(ms(U, dU))


def assemble_residual(operators: Iterable[Operator], U, state, t, dt, ndof):
    """Sum every operator's residual contribution into the global vector."""
    R = np.zeros(ndof, dtype=U.dtype)
    trials = []
    for op in operators:
        c = op.residual(U, state, t, dt)
        np.add.at(R, c.gdofs, c.values)
        trials.append(c.state_trial)
    return R, trials


def assemble_tangent(operators: Iterable[Operator], U, state, t, dt, ndof):
    """Sum every operator's COO tangent into the global sparse matrix."""
    rows, cols, vals = [], [], []
    for op in operators:
        T = op.tangent(U, state, t, dt)
        rows.append(np.asarray(T.rows))
        cols.append(np.asarray(T.cols))
        vals.append(np.asarray(T.values))
    K = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ndof, ndof),
    ).tocsr()
    return K


def _apply_dirichlet(K, R, con):
    """Identity-row Dirichlet: constrained rows become e_g, R already zeroed."""
    K = K.tolil()
    for g in con:
        K.rows[g] = [g]
        K.data[g] = [1.0]
    return K.tocsr(), R


def _newton_solve_affine(
    operators,
    U0,
    state,
    ndof,
    dirichlet,
    constraints,
    *,
    t,
    dt,
    rtol,
    maxit,
):
    """Newton-solve in the exact affine space ``U = P q + offset``.

    Operators continue to assemble and commit in full space.  Only the global
    residual, tangent, and Newton increment cross the mesh-agnostic transform.
    """

    from coupfe.constraints import compile_affine_constraints

    transform = compile_affine_constraints(
        ndof,
        constraints,
        dirichlet=dirichlet,
    )
    q = transform.reduce_guess(np.asarray(U0, dtype=float))
    U = transform.lift(q)

    R0 = None
    nit = 0
    for nit in range(1, maxit + 1):
        R_full, _ = assemble_residual(operators, U, state, t, dt, ndof)
        R = transform.restrict_residual(R_full)
        rn = float(np.linalg.norm(R))
        if R0 is None:
            R0 = max(rn, 1e-300)
        if rn < rtol * R0 or rn < 1e-14:
            break

        K_full = assemble_tangent(operators, U, state, t, dt, ndof)
        K = transform.reduce_tangent(K_full)
        dq = np.asarray(linear_solve(K, -R)).reshape(-1)
        if dq.shape != (transform.reduced_ndof,) or not np.all(np.isfinite(dq)):
            raise RuntimeError(
                "affine-constraint Newton solve produced an invalid increment"
            )
        dU = transform.project_increment(dq)

        # Preserve the established time-aware CCD/max-step path.  Constraint
        # reduction changes coordinates, so each operator must see the lifted
        # full-space increment rather than dq.
        alpha = 1.0
        for op in operators:
            alpha = min(alpha, _call_max_step(op, U, dU, t))

        for _ in range(25):
            trial_q = q + alpha * dq
            trial_U = transform.lift(trial_q)
            Rt_full, _ = assemble_residual(
                operators,
                trial_U,
                state,
                t,
                dt,
                ndof,
            )
            rt = float(
                np.linalg.norm(transform.restrict_residual(Rt_full))
            )
            if np.isfinite(rt) and rt < rn:
                break
            alpha *= 0.5
        q = q + alpha * dq
        U = transform.lift(q)

    new_state = [op.commit(U, state, t, dt) for op in operators]
    return U, new_state, nit


def newton_solve(operators, U0, state, ndof, dirichlet, *, t=1.0, dt=1.0,
                 rtol=1e-9, maxit=60, constraints=None):
    """Newton solve composing any operators, with Dirichlet elimination.

    Parameters
    ----------
    operators : list of Operator
    U0 : initial guess (ndof,)
    state : committed state (any object the operators understand)
    dirichlet : dict {global_dof: value}
    constraints : iterable of ConstraintRelation, optional
        Exact scalar affine relations at the current load/time.  Dirichlet and
        affine equations are compiled into ``U=Pq+offset`` and the nonlinear
        solve runs in ``q``.  Applications remain responsible for constructing
        relations from their mesh/domain semantics.

    Returns ``(U, new_state, n_iters)``.
    """
    if constraints is not None:
        return _newton_solve_affine(
            operators,
            U0,
            state,
            ndof,
            dirichlet,
            constraints,
            t=t,
            dt=dt,
            rtol=rtol,
            maxit=maxit,
        )

    U = np.asarray(U0, dtype=float).copy()
    con = np.array(sorted(dirichlet), dtype=int)
    for g, v in dirichlet.items():
        U[g] = v
    is_con = np.zeros(ndof, dtype=bool)
    is_con[con] = True

    R0 = None
    nit = 0
    for nit in range(1, maxit + 1):
        R, _ = assemble_residual(operators, U, state, t, dt, ndof)
        R[is_con] = 0.0
        rn = float(np.linalg.norm(R))
        if R0 is None:
            R0 = max(rn, 1e-300)
        if rn < rtol * R0 or rn < 1e-14:
            break
        K = assemble_tangent(operators, U, state, t, dt, ndof)
        K, R = _apply_dirichlet(K, R, con)
        dU = linear_solve(K, -R)
        # Step bound (CCD): an operator may cap the step length — e.g. barrier contact,
        # so the line search never steps a node through the obstacle. Start at the tightest
        # bound across operators (default 1.0), then backtrack.
        alpha = 1.0
        for op in operators:
            alpha = min(alpha, _call_max_step(op, U, dU, t))
        # Backtracking line search: a full finite-strain Newton step from a
        # distorted state (e.g. a prescribed boundary displacement while the
        # interior is still zero) can overshoot into element inversion → ln(J) of
        # J<0 → NaN. Damp alpha until the residual is finite and decreasing.
        for _ in range(25):
            Rt, _ = assemble_residual(operators, U + alpha * dU, state, t, dt, ndof)
            Rt[is_con] = 0.0
            rt = float(np.linalg.norm(Rt))
            if np.isfinite(rt) and rt < rn:
                break
            alpha *= 0.5
        U = U + alpha * dU

    new_state = [op.commit(U, state, t, dt) for op in operators]
    return U, new_state, nit


def solve_increments(
    operators,
    U0,
    ndof,
    dirichlet,
    *,
    n_steps=4,
    constraints=None,
    **newton_kw,
):
    """Ramp the Dirichlet BCs over ``n_steps`` load increments, warm-started.

    A finite-strain load applied in a single Newton solve from a zero interior
    usually falls outside Newton's basin (the first tangent can be indefinite → the
    step overshoots → an element inverts → ``ln(J)`` of ``J<0`` → NaN). Incremental
    loading with warm-start is the standard remedy. Returns ``(U, total_iters)``.

    A static ``constraints`` iterable describes target relations: their affine
    offsets are ramped with the same load fraction while coefficients remain
    fixed.  For a non-proportional schedule, pass a callable ``fraction ->
    relations``; its returned offsets are used as current values.

    NOTE: this passes ``state=None`` each increment, which is correct for
    history-free (e.g. hyperelastic) operators. Path-dependent operators need the
    committed state carried between increments — a follow-up tied to giving each
    operator its own state slot (see ``docs/roadmap.md`` workstream B).
    """
    U = np.asarray(U0, dtype=float).copy()
    total = 0
    target_constraints = (
        None
        if constraints is None or callable(constraints)
        else tuple(constraints)
    )
    if target_constraints is not None:
        from coupfe.constraints import ConstraintRelation

        if not all(
            isinstance(relation, ConstraintRelation)
            for relation in target_constraints
        ):
            raise TypeError(
                "constraints must contain ConstraintRelation objects"
            )
    for k in range(1, n_steps + 1):
        fraction = k / n_steps
        dk = {g: v * fraction for g, v in dirichlet.items()}
        if callable(constraints):
            current_constraints = constraints(fraction)
        elif target_constraints is None:
            current_constraints = None
        else:
            current_constraints = tuple(
                relation.scaled_offset(fraction)
                for relation in target_constraints
            )
        U, _, nit = newton_solve(
            operators,
            U,
            None,
            ndof,
            dk,
            constraints=current_constraints,
            **newton_kw,
        )
        total += nit
    return U, total


def solve_dynamics(
    operators,
    U0,
    ndof,
    dirichlet,
    *,
    dt,
    n_steps,
    constraints=None,
    **newton_kw,
):
    """Implicit **backward-Euler** dynamics. ``operators`` must include an
    :class:`~coupfe.operators.inertia.InertiaOperator` (its ``commit`` advances velocity); the
    element/contact/load operators compose unchanged. ``dirichlet`` is a dict (static BCs) or a
    callable ``t -> {dof: value}`` (time-varying). Returns ``(U, info)``.

    Each step Newton-solves ``M/dt²(u−û) + F_int(u) + F_contact(u) − F_ext = 0`` for ``u``,
    warm-started from the inertial predictor ``û``. Backward-Euler is dissipative *on purpose*:
    its numerical damping regularizes non-smooth contact, and — with the load held — it doubles
    as **dynamic relaxation** to a quasistatic equilibrium.

    Exact affine constraints are deliberately unsupported here: a correct
    dynamics path must project the inertial predictor, velocity/history, and
    any time-varying affine offset consistently.
    """
    if constraints is not None:
        raise NotImplementedError(
            "affine constraints are not supported by solve_dynamics; "
            "a constraint-projected inertial/state update is required"
        )

    U = np.asarray(U0, dtype=float).copy()
    total = 0
    for step in range(1, n_steps + 1):
        t = step * dt
        bc = dirichlet(t) if callable(dirichlet) else dict(dirichlet)
        # warm-start from the inertial predictor of any inertia operator present
        U_prev = U                                    # last accepted (non-penetrating) state
        U_pred = U
        for op in operators:
            if hasattr(op, "predictor"):
                U_pred = np.asarray(op.predictor(dt), dtype=float)
                break
        # CCD-bound the PREDICTOR jump too — not just the Newton increment. A fast inertial
        # predictor û = u_prev + dt·v_prev can teleport a node straight through a barrier band
        # (dt·v > gap) before Newton ever runs; bounding it from the last accepted state with
        # the same max_step keeps the warm start penetration-free. (No-op when separated → α=1.)
        dU_pred = U_pred - U_prev
        alpha = 1.0
        for op in operators:
            alpha = min(alpha, _call_max_step(op, U_prev, dU_pred, t, dt))
        U = (U_prev + alpha * dU_pred).copy()
        U, _, nit = newton_solve(operators, U, None, ndof, bc, t=t, dt=dt, **newton_kw)
        total += nit
    return U, {"steps": n_steps, "t": n_steps * dt, "total_newton": total}


def solve_dynamics_adaptive(operators, U0, ndof, dirichlet, *, t_end, dt_init,
                            dt_min=None, dt_max=None, growth=1.2, cut=0.5,
                            maxit=None, constraints=None, **newton_kw):
    """Adaptive-timestep implicit backward-Euler dynamics.

    Same operator contract as :func:`solve_dynamics`, but the step size is chosen
    on the fly: it is **cut** when Newton does not converge (or the CCD bound
    collapses) and **grown** when convergence is easy. This is the dynamics
    analogue of the quasistatic adaptive load stepping in
    ``examples/ring_compress/reproduce.py``, and it is the key to making hard
    contact/impact stable without hand-tuning ``dt``.

    Parameters
    ----------
    t_end : float
        Target physical time. The solver steps until ``t >= t_end``.
    dt_init : float
        Initial attempted step size.
    dt_min : float, optional
        Minimum allowable step; abort if a step would shrink below this.
        Default ``dt_init / 1e6``.
    dt_max : float, optional
        Largest allowable step. Default ``dt_init``.
    growth : float
        Multiplier on a successful step.
    cut : float
        Multiplier on a failed step.
    maxit : int, optional
        Newton-iteration threshold for step acceptance (defaults to the
        ``maxit`` passed in ``newton_kw``).

    Returns
    -------
    (U, info)
        ``info`` contains ``steps``, ``t``, ``total_newton``, ``n_cuts``,
        ``n_grows``, ``dt_init``, ``dt_min``, ``dt_max``.
    """
    if constraints is not None:
        raise NotImplementedError(
            "affine constraints are not supported by "
            "solve_dynamics_adaptive; a constraint-projected "
            "inertial/state update is required"
        )

    U = np.asarray(U0, dtype=float).copy()
    if dt_min is None:
        dt_min = dt_init / 1.0e6
    if dt_max is None:
        dt_max = dt_init
    if maxit is None:
        maxit = newton_kw.get("maxit", 60)
    t = 0.0
    step = 0
    total = 0
    n_cuts = 0
    n_grows = 0
    dt = float(dt_init)
    dt_min = float(dt_min)
    dt_max = float(dt_max)

    def _bc_at(time):
        return dirichlet(time) if callable(dirichlet) else dict(dirichlet)

    while t < t_end - 1e-14:
        dt_try = min(dt, t_end - t)
        step += 1
        t_try = t + dt_try
        bc = _bc_at(t_try)

        # warm-start from inertial predictor, CCD-bound the predictor jump
        U_prev = U
        U_pred = U
        for op in operators:
            if hasattr(op, "predictor"):
                U_pred = np.asarray(op.predictor(dt_try), dtype=float)
                break
        dU_pred = U_pred - U_prev
        alpha = 1.0
        for op in operators:
            alpha = min(alpha, _call_max_step(op, U_prev, dU_pred, t_try, dt_try))
        # A CCD bound that collapsed to zero leaves no Newton basin (the state is
        # already on the obstacle).  Fall back to a tiny positive step and let the
        # line search / cut logic handle it rather than freezing forever.
        if alpha <= 0.0:
            alpha = 1.0e-12
        U_try = (U_prev + alpha * dU_pred).copy()

        try:
            U_new, _, nit = newton_solve(operators, U_try, None, ndof, bc,
                                         t=t_try, dt=dt_try, maxit=maxit,
                                         **newton_kw)
        except Exception:
            nit = maxit + 1

        # If Newton exceeded the allotted iterations (or produced a non-finite
        # update) we treat the step as failed and cut the timestep.
        if nit > maxit or not np.isfinite(U_new).all():
            dt *= cut
            n_cuts += 1
            if dt < dt_min:
                raise RuntimeError(
                    f"adaptive dynamics: step collapsed below dt_min={dt_min} "
                    f"at t={t:.6g}"
                )
            step -= 1
            continue

        # accept step
        U = U_new
        t = min(t_try, t_end)
        total += nit
        if dt * growth <= dt_max and dt_try == dt:
            dt *= growth
            n_grows += 1
        else:
            dt = min(dt, dt_max)

    return U, {
        "steps": step,
        "t": t,
        "total_newton": total,
        "n_cuts": n_cuts,
        "n_grows": n_grows,
        "dt_init": dt_init,
        "dt_min": dt_min,
        "dt_max": dt_max,
    }
