# Review: linear solvers + assembly (CoupFE-core, 2026-07-02)

Closing summary of the solver/assembly review. All items landed on `main` with
tests and docs; the historical sibling-worktree port is complete.

## What was reviewed and fixed

1. **Random per-example solver setups → ONE policy module.** `newton_solve` used bare `spsolve`;
   examples (esp. `cattaneo_3d`) hand-rolled PETSc KSP setups. Consolidated into
   `coupfe/assembly/factored.py`: `linear_solve` (one-shot, scipy small / MUMPS 3D-above-~20k),
   `factored_lu` (factor-once for condensed contact), with `newton_solve`/`solve_dynamics` routed
   through it. Gated in `tests/test_factored.py`.

2. **Iterative + FieldSplit added, opt-in.** `iterative_solve` (hypre-first, gamg fallback — measured
   2-4× gamg, beats direct on 2D SPD at ~90k DOFs) for large well-conditioned bulk;
   `make_fieldsplit_solver` ported from `abaqus_ufl.fe` (validated on coupled u-c-phi) for multifield.

3. **The R/K double-evaluation (the serious one).** The compiled kernel returns R and K together
   (like the Abaqus `UEL`'s `RHS`+`AMATRX`; `abaqus_ufl.fe` preserves this in a single-pass
   `assemble`). CoupFE's `Operator` split into `residual()`/`tangent()` ran the kernel TWICE per
   Newton iterate. Fixed with a fusion cache in `ElementGroup` (`fuse_rk`, default ON; robust key +
   `commit` invalidation; `COUPFE_FUSE_RK=0` to disable). Measured: neo-Hookean block 64→24 kernel
   calls, bit-identical; cardiac 1.77×. Gated (bit-identical + prop-change staleness control).

## Load-bearing conclusions (the corrections that matter)

- **A claim without its measured bound is a future wrong decision.** "2D → SuperLU at any size" and
  "gamg as the AMG default" were both overclaims, corrected with measured anchors.
- **Structure decides, not dimensionality.** The thin-walled LV factors cheaply — MUMPS is *slower*
  there than SuperLU at every benchmark size. The 3D→MUMPS heuristic is for BULK 3D.
- **Profile assembly-vs-solve before any "the solver is slow" conclusion.** In cardiac the linear
  solve was 14-21%; the real lever was the redundant kernel eval.
- **The Abaqus UEL is joint-by-design; follow it.** A contract that evaluates a joint (R,K) kernel
  through two calls silently doubles the hot cost. Long-term: a combined
  `Operator.residual_and_tangent()`; the cache is the bit-identical bridge.

## Deliberately deferred / out of scope
- Combined `Operator.residual_and_tangent()` (the proper contract-level fix; cache bridges it now).
- Residual-only kernel path (skip the CS tangent when only R is needed — helps residual-only /
  line-search-heavy solves; the compiled UEL currently always returns both).
- Distributed/MPI solver policy (separate; `superlu_dist`, `skills/distributed.md`).

## Where it lives
- Code: `coupfe/assembly/factored.py`, `coupfe/operators/element_group.py`.
- Tests: `tests/test_factored.py`, `tests/test_element_group.py`.
- Why: `docs/lessons_learned.md` (2026-07-02), `skills/performance.md` ("The solver ladder",
  "Linear solvers", assembly-vs-solve), `skills/pitfalls.md` (joint-kernel rule).
- Port status: complete; the code and regression-test locations above are the
  durable record.
