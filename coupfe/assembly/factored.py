"""The serial linear-solver policy module: direct (scipy / PETSc MUMPS) + opt-in iterative.

ALL serial solver policy lives here (`skills/performance.md`, "Linear solvers"):
:func:`linear_solve` for one-shot solves (``newton_solve``/``solve_dynamics``
route through it), :func:`factored_lu` for the FACTOR-ONCE / SOLVE-MANY workload
of the condensed contact solvers (one factorization, many back-solves, including
dense BLOCKS — the interface-column cache ``X = K_ff^{-1} S^T``).

Standalone on purpose (no coupfe imports): usable from any operator module
without import cycles.

Historical local measurements motivated the configurable direct-solver
threshold.  They are not retained release benchmark evidence; remeasure on the
target problem and environment before drawing a performance conclusion.  The
interface of both direct backends mirrors ``scipy.sparse.linalg.splu``:
``.solve(b)`` accepts a vector or a dense ``(n, k)`` block.

Known solver considerations:

* A zero pivot or failed preconditioner at contact engagement can indicate a
  singular or nearly singular model, including a contact-only configuration
  with no bulk stiffness. Check the model and null modes before changing the
  solver.
* Parallel factorization and iterative methods have environment- and
  tolerance-dependent comparison floors. This module is serial (``COMM_SELF``);
  distributed qualification is described in `skills/distributed.md`.
* Fallbacks emit a ``RuntimeWarning`` rather than changing backends silently.
  On an ill-conditioned contact system, any reported update still needs a
  residual and finiteness check.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import scipy.sparse.linalg as spla

_LINEAR_SOLVER = os.environ.get("COUPFE_LINEAR_SOLVER", "petsc").lower()

# Historical local measurements motivated this default threshold.  Treat it as
# a configurable policy, not portable performance evidence.  Override with
# COUPFE_FACTORED_PETSC_MIN_N=0 to force PETSc at any size
# (COUPFE_LINEAR_SOLVER=scipy still forces scipy everywhere).
_PETSC_MIN_N = int(os.environ.get("COUPFE_FACTORED_PETSC_MIN_N", "20000"))


class _ScipyLU:
    """splu wrapper (splu.solve already handles 1D and 2D right-hand sides)."""

    backend = "scipy-superlu"

    def __init__(self, K):
        self._lu = spla.splu(K.tocsc())

    def solve(self, b):
        return self._lu.solve(b)


class _PetscLU:
    """PETSc serial direct factorization (COMM_SELF), factored eagerly.

    Prefers MUMPS, then superlu_dist, else PETSc's built-in LU. Dense
    multi-RHS blocks use a conservative per-column KSP loop over the same
    factorization rather than requiring an optional matrix-solve path.
    """

    def __init__(self, K):
        from petsc4py import PETSc

        self._PETSc = PETSc
        Kc = K.tocsr()
        self.n = Kc.shape[0]
        self._A = PETSc.Mat().createAIJWithArrays(
            (self.n, self.n),
            (Kc.indptr.astype(PETSc.IntType),
             Kc.indices.astype(PETSc.IntType), Kc.data),
            comm=PETSc.COMM_SELF,
        )
        self._A.assemble()
        self._ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        self._ksp.setOperators(self._A)
        self._ksp.setType("preonly")
        pc = self._ksp.getPC()
        pc.setType("lu")
        self.backend = "petsc-lu"
        for fst in ("mumps", "superlu_dist"):      # robust direct factors if configured
            try:
                pc.setFactorSolverType(fst)
                self.backend = f"petsc-{fst}"
                break
            except Exception:
                continue
        self._ksp.setUp()                          # factor NOW (factor-once semantics)
        self._x = self._A.createVecRight()
        self._b = self._A.createVecRight()

    def _solve_vec(self, b):
        self._b.setArray(np.ascontiguousarray(b, dtype=float))
        self._ksp.solve(self._b, self._x)
        if self._ksp.getConvergedReason() < 0:
            raise RuntimeError(
                "PETSc KSP failed: reason %d" % self._ksp.getConvergedReason())
        return self._x.getArray().astype(float).copy()

    def _solve_block(self, B):
        n, k = B.shape
        # Keep factor-once semantics with a portable per-column KSP path.
        X = np.empty((n, k), dtype=float)
        for j in range(k):
            X[:, j] = self._solve_vec(B[:, j])
        return X

    def solve(self, b):
        b = np.asarray(b, dtype=float)
        if b.ndim == 1:
            return self._solve_vec(b)
        return self._solve_block(b)

    def destroy(self):
        for obj in ("_x", "_b", "_ksp", "_A"):
            m = getattr(self, obj, None)
            if m is not None:
                try:
                    m.destroy()
                except Exception:
                    pass
                setattr(self, obj, None)

    def __del__(self):
        self.destroy()


def factored_lu(K, *, prefer="auto"):
    """Factor ``K`` once; the returned object solves 1D vectors and 2D blocks.

    ``prefer`` controls the backend policy:

    * ``"scipy"`` — use SciPy's sparse LU. This is a reproducibility or
      environment choice, not an any-size performance claim.
    * ``"auto"`` — use SciPy below ``COUPFE_FACTORED_PETSC_MIN_N`` and try
      PETSc above it. The threshold is configurable and should be remeasured
      for the target matrices and environment.
    * ``"petsc"`` — force PETSc at any size.

    ``COUPFE_LINEAR_SOLVER=scipy`` forces scipy everywhere (same convention as
    :func:`linear_solve`); PETSc errors fall back to scipy with a
    ``RuntimeWarning`` (see the module docstring — a factorization failure is
    often a MODEL problem, e.g. contact on a degenerate/no-bulk system).
    """
    if _LINEAR_SOLVER != "scipy" and prefer != "scipy":
        if prefer == "petsc" or K.shape[0] >= _PETSC_MIN_N:
            try:
                return _PetscLU(K)
            except Exception as exc:
                warnings.warn(
                    f"PETSc factorization failed ({exc!r}); falling back to "
                    "scipy splu. If this system carries contact, check the "
                    "configuration and constrained null modes.",
                    RuntimeWarning, stacklevel=2)
    return _ScipyLU(K)


def linear_solve(K, b, *, prefer="auto"):
    """One-shot solve of ``K x = b`` under the same backend policy as
    :func:`factored_lu`: scipy ``spsolve`` below the size threshold (and for
    ``prefer="scipy"``), PETSc MUMPS above (``prefer="petsc"`` forces PETSc;
    ``COUPFE_LINEAR_SOLVER=scipy`` forces scipy everywhere).  This is THE
    solver entry point for one-shot solves — ``newton_solve``/``solve_dynamics``
    route through it; do not hand-roll PETSc KSP setups in examples.  For
    repeated solves with the SAME matrix use :func:`factored_lu` (factor once);
    for very large well-conditioned bulk systems see :func:`iterative_solve`.
    """
    if _LINEAR_SOLVER != "scipy" and prefer != "scipy":
        if prefer == "petsc" or K.shape[0] >= _PETSC_MIN_N:
            try:
                return _PetscLU(K).solve(np.asarray(b, dtype=float))
            except Exception as exc:
                warnings.warn(
                    f"PETSc direct solve failed ({exc!r}); falling back to "
                    "scipy spsolve; verify the fallback result is finite and "
                    "satisfies the requested residual tolerance.",
                    RuntimeWarning, stacklevel=2)
    return spla.spsolve(K.tocsr(), b)


def iterative_solve(K, b, *, ksp_type="gmres", pc_type=None, rtol=1e-8,
                    max_it=2000):
    """OPT-IN iterative solve (PETSc Krylov + algebraic multigrid), serial.

    ``pc_type=None`` (default) tries **hypre** (BoomerAMG) and falls back to
    **gamg** if hypre is not in the PETSc build. Backend crossover and scaling
    depend on the matrix and environment; profile before selecting a production
    policy.

    This is a candidate for large bulk systems when the selected Krylov method
    and preconditioner match the matrix. Contact terms can introduce contrast,
    nonsymmetry, and near-null modes; qualify those systems separately. For
    serial-versus-rank work, set comparison tolerances from the actual solver
    tolerances and conditioning (see ``skills/distributed.md``).

    Raises on non-convergence instead of silently degrading — if the iteration
    stalls, switch to :func:`linear_solve` (direct) rather than loosening
    ``rtol``.
    """
    from petsc4py import PETSc

    Kc = K.tocsr()
    n = Kc.shape[0]
    pcs = (pc_type,) if pc_type is not None else ("hypre", "gamg")
    last_exc = None
    for pc in pcs:
        A = PETSc.Mat().createAIJWithArrays(
            (n, n),
            (Kc.indptr.astype(PETSc.IntType), Kc.indices.astype(PETSc.IntType),
             Kc.data),
            comm=PETSc.COMM_SELF,
        )
        A.assemble()
        rhs = PETSc.Vec().createWithArray(
            np.ascontiguousarray(b, dtype=float), comm=PETSc.COMM_SELF)
        x = A.createVecRight()
        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        ksp.setOperators(A)
        ksp.setType(ksp_type)
        ksp.setTolerances(rtol=rtol, max_it=max_it)
        ksp.getPC().setType(pc)
        try:
            ksp.solve(rhs, x)
            reason = ksp.getConvergedReason()
            its = ksp.getIterationNumber()
            if reason < 0:
                raise RuntimeError(
                    f"iterative solve diverged (pc={pc}, reason {reason}, "
                    f"{its} its) — use linear_solve (direct) rather than "
                    "loosening rtol")
            return x.getArray().astype(float).copy()
        except Exception as exc:
            last_exc = exc
        finally:
            for obj in (ksp, x, rhs, A):
                obj.destroy()
    raise last_exc


def make_fieldsplit_solver(field_components, dof_per_node, *, ksp_type="gmres",
                           split_type="additive", sub_pc="gamg",
                           rtol=1e-10, atol=1e-50, maxit=2000):
    """FieldSplit (block) preconditioner factory for COUPLED multi-field systems.

    FieldSplit separates a node-major coupled system by field and assigns a
    preconditioner to each block. Scale each equation meaningfully first:
    FieldSplit addresses block structure, not a badly scaled residual.

    ``field_components``: ``[(name, [comp_idx, ...]), ...]`` mapping each field
    to its per-node DOF component indices in the NODE-MAJOR interleaved layout
    (e.g. 2D displacement + concentration: ``[("u", [0, 1]), ("c", [2])]`` with
    ``dof_per_node=3``). ``split_type`` may be ``"additive"``
    (block-Jacobi-like), ``"multiplicative"``, or ``"schur"`` (two fields
    only). ``sub_pc`` selects a PETSc preconditioner available in the active
    build, such as ``"gamg"``, ``"hypre"``, or ``"lu"``.

    Returns ``linear_solve(K, b) -> x`` (drop-in for the driver's pluggable
    solver).  Raises on divergence instead of silently degrading.
    """
    from petsc4py import PETSc
    import scipy.sparse as _sp

    stype = {"additive": PETSc.PC.CompositeType.ADDITIVE,
             "multiplicative": PETSc.PC.CompositeType.MULTIPLICATIVE,
             "schur": PETSc.PC.CompositeType.SCHUR}[split_type]

    def fieldsplit_solve(K, b):
        A = _sp.csr_matrix(K)
        n = A.shape[0]
        n_node = n // dof_per_node
        node = np.arange(n_node)
        mat = PETSc.Mat().createAIJWithArrays(
            (n, n),
            (A.indptr.astype(PETSc.IntType), A.indices.astype(PETSc.IntType),
             A.data),
            comm=PETSc.COMM_SELF,
        )
        mat.assemble()
        rhs = PETSc.Vec().createWithArray(
            np.ascontiguousarray(b, dtype=float), comm=PETSc.COMM_SELF)
        x = mat.createVecRight()
        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        iss = []
        try:
            ksp.setOperators(mat)
            ksp.setType(ksp_type)
            pc = ksp.getPC()
            pc.setType("fieldsplit")
            pc.setFieldSplitType(stype)
            fields = []
            for name, comps in field_components:
                dofs = np.sort(np.concatenate(
                    [node * dof_per_node + c for c in comps])).astype(
                        PETSc.IntType)
                is_ = PETSc.IS().createGeneral(dofs, comm=PETSc.COMM_SELF)
                iss.append(is_)
                fields.append((name, is_))
            pc.setFieldSplitIS(*fields)
            pc.setUp()
            for sub in pc.getFieldSplitSubKSP():
                sub.getPC().setType(sub_pc)
            ksp.setTolerances(rtol=rtol, atol=atol, max_it=maxit)
            ksp.solve(rhs, x)
            reason = ksp.getConvergedReason()
            if reason < 0:
                raise RuntimeError(
                    f"FieldSplit KSP diverged (reason {reason}, "
                    f"split={split_type}, sub_pc={sub_pc}) — check per-field "
                    "equation scaling before switching solvers")
            return x.getArray().astype(float).copy()
        finally:
            for obj in iss + [ksp, x, rhs, mat]:
                try:
                    obj.destroy()
                except Exception:
                    pass

    return fieldsplit_solve
