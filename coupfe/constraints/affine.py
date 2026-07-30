"""Exact affine multi-point constraints by sparse elimination.

The full solution is represented as ``U = P @ q + offset``.  Applications can
keep assembling in the full space and reduce a linear system as
``P.T @ K @ P`` and ``P.T @ (f - K @ offset)``.  Each scalar relation has the
form

``U[slave] = sum(coefficients[j] * U[masters[j]]) + offset``.

This mesh-independent algebra is adapted from CoupFE commit
``70ea06355ecf55cecb5ae01c55a377c03879470b``.  Mesh matching, periodic graph
construction, and application-specific boundary semantics deliberately live
outside Core.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import operator
from typing import Iterable, Mapping, Optional

import numpy as np
import scipy.sparse as sp


def _dof_index(value, *, name: str) -> int:
    """Return an actual integer index without lossy coercion.

    ``operator.index`` accepts Python and NumPy integer scalars while rejecting
    strings and floats. Booleans implement that protocol for historical Python
    reasons, so reject them explicitly.
    """

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class ConstraintRelation:
    """One scalar affine constraint equation.

    ``masters=()`` represents an exact prescribed value.  A relation is setup
    data rather than a physical element or constitutive state.
    """

    slave: int
    masters: tuple[int, ...] = ()
    coefficients: tuple[float, ...] = ()
    offset: float = 0.0
    label: str = ""

    def __post_init__(self):
        slave = _dof_index(self.slave, name="slave DOF")
        masters = tuple(
            _dof_index(value, name="master DOF") for value in self.masters
        )
        coefficients = tuple(float(value) for value in self.coefficients)
        object.__setattr__(self, "slave", slave)
        object.__setattr__(self, "masters", masters)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "offset", float(self.offset))
        object.__setattr__(self, "label", str(self.label))
        if len(masters) != len(coefficients):
            raise ValueError("masters and coefficients must have the same length")
        if len(set(masters)) != len(masters):
            raise ValueError("a constraint relation cannot repeat a master DOF")
        if not np.isfinite(self.offset) or not np.all(np.isfinite(coefficients)):
            raise ValueError("constraint coefficients and offset must be finite")

    def scaled_offset(self, factor: float) -> "ConstraintRelation":
        """Return the same linear relation with its target offset scaled.

        This is the load-increment operation for a target affine constraint:
        coefficients remain fixed while the prescribed jump is ramped.  A
        non-proportional schedule should construct its relations explicitly.
        """

        factor = float(factor)
        if not np.isfinite(factor):
            raise ValueError("constraint offset scale must be finite")
        return ConstraintRelation(
            self.slave,
            self.masters,
            self.coefficients,
            factor * self.offset,
            self.label,
        )


def _relations_tuple(relations) -> tuple[ConstraintRelation, ...]:
    if relations is None:
        return ()
    result = tuple(relations)
    if not all(isinstance(relation, ConstraintRelation) for relation in result):
        raise TypeError("constraints must contain ConstraintRelation objects")
    return result


@dataclass(frozen=True)
class ConstraintTransform:
    """Compiled full-to-reduced affine map ``U = P q + offset``."""

    P: sp.csr_matrix
    offset: np.ndarray
    independent_dofs: np.ndarray
    relations: tuple[ConstraintRelation, ...]
    sha256: str

    @property
    def full_ndof(self) -> int:
        return int(self.P.shape[0])

    @property
    def reduced_ndof(self) -> int:
        return int(self.P.shape[1])

    def lift(self, q) -> np.ndarray:
        """Map reduced coordinates into the full affine solution space."""

        q = np.asarray(q)
        if q.shape != (self.reduced_ndof,):
            raise ValueError(f"q must have shape ({self.reduced_ndof},)")
        return np.asarray(self.P @ q).reshape(-1) + self.offset

    def project_increment(self, dq) -> np.ndarray:
        """Map a reduced increment to full space without adding the offset."""

        dq = np.asarray(dq)
        if dq.shape != (self.reduced_ndof,):
            raise ValueError(f"dq must have shape ({self.reduced_ndof},)")
        return np.asarray(self.P @ dq).reshape(-1)

    def reduce_guess(self, full_guess) -> np.ndarray:
        """Extract reduced coordinates from a full-space warm start.

        Every independent DOF is an identity row of ``P``, so its full value is
        exactly the corresponding reduced coordinate.  Lifting the result
        projects any inconsistent slave values onto the affine space without a
        second linear solve.
        """

        full_guess = np.asarray(full_guess)
        if full_guess.shape != (self.full_ndof,):
            raise ValueError(f"full_guess must have shape ({self.full_ndof},)")
        return np.asarray(full_guess[self.independent_dofs]).copy()

    def restrict_residual(self, residual) -> np.ndarray:
        """Restrict a full residual with the work-conjugate map ``P.T``."""

        residual = np.asarray(residual)
        if residual.shape != (self.full_ndof,):
            raise ValueError(f"residual must have shape ({self.full_ndof},)")
        return np.asarray(self.P.T @ residual).reshape(-1)

    def reduce_tangent(self, tangent) -> sp.csr_matrix:
        """Return the reduced tangent ``P.T @ tangent @ P``."""

        tangent = sp.csr_matrix(tangent)
        if tangent.shape != (self.full_ndof, self.full_ndof):
            raise ValueError(
                "tangent shape is incompatible with the constraint transform"
            )
        return (self.P.T @ tangent @ self.P).tocsr()

    def reduce_linear_system(self, tangent, force):
        """Reduce ``K U = f`` to ``(P.T K P) q = P.T (f-K offset)``."""

        tangent = sp.csr_matrix(tangent)
        force = np.asarray(force)
        if force.shape != (self.full_ndof,):
            raise ValueError(f"force must have shape ({self.full_ndof},)")
        return self.reduce_tangent(tangent), self.restrict_residual(
            force - tangent @ self.offset
        )

    def relation_matrix(self):
        """Return the compiled relation matrix ``C`` and right side ``g``."""

        rows, cols, data, rhs = [], [], [], []
        for row, relation in enumerate(self.relations):
            rows.append(row)
            cols.append(relation.slave)
            data.append(1.0)
            for master, coefficient in zip(
                relation.masters, relation.coefficients
            ):
                rows.append(row)
                cols.append(master)
                data.append(-coefficient)
            rhs.append(relation.offset)
        matrix = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(len(self.relations), self.full_ndof),
        ).tocsr()
        return matrix, np.asarray(rhs, dtype=float)

    def constraint_error(self, full_solution) -> np.ndarray:
        """Evaluate ``C @ full_solution - g`` for every compiled relation."""

        full_solution = np.asarray(full_solution)
        if full_solution.shape != (self.full_ndof,):
            raise ValueError(f"full_solution must have shape ({self.full_ndof},)")
        matrix, rhs = self.relation_matrix()
        return np.asarray(matrix @ full_solution).reshape(-1) - rhs


def compile_affine_constraints(
    ndof: int,
    relations: Iterable[ConstraintRelation] = (),
    *,
    dirichlet: Optional[Mapping[int, float]] = None,
) -> ConstraintTransform:
    """Compile acyclic scalar relations into an exact sparse affine transform.

    Dirichlet values are folded into the same transform.  A relation slave
    cannot also be prescribed directly; applications should prescribe an
    independent representative instead.
    """

    ndof = _dof_index(ndof, name="ndof")
    if ndof < 0:
        raise ValueError("ndof must be non-negative")
    supplied = list(_relations_tuple(relations))
    relation_by_slave = {}
    for relation in supplied:
        if relation.slave < 0 or relation.slave >= ndof:
            raise ValueError("constraint slave DOF is out of range")
        if any(master < 0 or master >= ndof for master in relation.masters):
            raise ValueError("constraint master DOF is out of range")
        if relation.slave in relation_by_slave:
            raise ValueError(
                f"DOF {relation.slave} has multiple constraint equations"
            )
        relation_by_slave[relation.slave] = relation

    for dof, value in ({} if dirichlet is None else dict(dirichlet)).items():
        dof = _dof_index(dof, name="Dirichlet DOF")
        value = float(value)
        if dof < 0 or dof >= ndof or not np.isfinite(value):
            raise ValueError("Dirichlet DOF/value is invalid")
        if dof in relation_by_slave:
            raise ValueError(
                f"DOF {dof} is already an MPC slave; "
                "prescribe a representative master instead"
            )
        relation = ConstraintRelation(dof, offset=value, label="dirichlet")
        supplied.append(relation)
        relation_by_slave[dof] = relation

    independent = np.array(
        [dof for dof in range(ndof) if dof not in relation_by_slave],
        dtype=int,
    )
    reduced_index = {
        int(dof): index for index, dof in enumerate(independent)
    }
    memo = {}
    visiting = set()

    def expand(dof):
        if dof in memo:
            return memo[dof]
        if dof in visiting:
            raise ValueError("constraint dependency graph contains a cycle")
        if dof not in relation_by_slave:
            result = ({dof: 1.0}, 0.0)
            memo[dof] = result
            return result
        visiting.add(dof)
        relation = relation_by_slave[dof]
        expression = {}
        constant = relation.offset
        for master, coefficient in zip(
            relation.masters, relation.coefficients
        ):
            if coefficient == 0.0:
                continue
            master_expression, master_constant = expand(master)
            constant += coefficient * master_constant
            for independent_dof, weight in master_expression.items():
                expression[independent_dof] = (
                    expression.get(independent_dof, 0.0)
                    + coefficient * weight
                )
        visiting.remove(dof)
        expression = {
            key: value for key, value in expression.items() if value != 0.0
        }
        result = (expression, constant)
        memo[dof] = result
        return result

    rows, cols, data = [], [], []
    offset = np.empty(ndof, dtype=float)
    for dof in range(ndof):
        expression, offset[dof] = expand(dof)
        for independent_dof, weight in expression.items():
            rows.append(dof)
            cols.append(reduced_index[independent_dof])
            data.append(weight)
    matrix = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(ndof, len(independent)),
    ).tocsr()

    supplied.sort(key=lambda item: item.slave)
    canonical = [
        {
            "slave": relation.slave,
            "masters": relation.masters,
            "coefficients": relation.coefficients,
            "offset": relation.offset,
            "label": relation.label,
        }
        for relation in supplied
    ]
    payload = json.dumps(
        {"ndof": ndof, "relations": canonical},
        sort_keys=True,
        separators=(",", ":"),
    )
    transform = ConstraintTransform(
        matrix,
        offset,
        independent,
        tuple(supplied),
        hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )

    constraint_matrix, rhs = transform.relation_matrix()
    if constraint_matrix.shape[0]:
        product = constraint_matrix @ matrix
        if product.nnz and np.max(np.abs(product.data)) > 1.0e-12:
            raise RuntimeError(
                "compiled constraint transform does not satisfy C @ P = 0"
            )
        if not np.allclose(
            constraint_matrix @ offset,
            rhs,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                "compiled constraint offset does not satisfy C @ offset = g"
            )
    return transform
