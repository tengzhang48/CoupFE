"""Mesh-independent algebra for exact affine multi-point constraints."""

from coupfe.constraints.affine import (
    ConstraintRelation,
    ConstraintTransform,
    compile_affine_constraints,
)

__all__ = [
    "ConstraintRelation",
    "ConstraintTransform",
    "compile_affine_constraints",
]
