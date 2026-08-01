# CoupFE performance / acceleration skill

How to make an operator fast **without premature compilation**. The operator contract is
acceleration-agnostic — an operator returns `(R, COO)` regardless of *how* it's computed —
so you can select an implementation per operator without changing the public contract.

## The ladder — escalate after measuring a bottleneck

1. **Try vectorized NumPy first.** Replace uniform per-element or per-node
   Python loops with batched array operations when that makes the code clearer
   and the profile supports it.
2. **numba for the irregular loops that don't vectorize** — search, narrow-phase, tree/BVH
   traversal. `@njit` and `prange` can be practical compiled paths when
   numba's supported subset fits the kernel. Caveats:
   array-based data (no Python
   objects in the hot loop — e.g. a BVH as arrays) and a one-time JIT warmup.
3. **Use f2py Fortran for regular dual-backend element kernels.** This preserves
   the native/Abaqus ABI goal where the supported generator path applies.
4. **Treat another compiled or GPU backend as a separate design decision.** It
   needs a measured target, a stable data contract, parity tests, packaging,
   and a maintainable CI environment.

## The discipline

- **Measure before escalating.** A Python loop over a small subset (contact
  nodes or a surface) may be adequate; element-wide loops are common profiling
  targets. Measure the actual case.
- **The contract isolates acceleration.** Each operator's residual/tangent can be
  numpy / vectorized / numba / f2py / Rust **independently** — speed up the bottleneck
  operator and leave the rest. No global rewrite.
- **Compile the hot / regular / parallel-backend work; keep dynamic / non-smooth / orchestration in
  Python** until the loop is the measured bottleneck. (Bulk elements → Fortran; contact force
  → vectorized numpy; contact *search* → numba; everything else → Python.)
- **For large 3D direct solves, distinguish assembly from factorization.** A
  sparse factorization may dominate batched element assembly. Before
  shrinking a mesh or rewriting assembly, time one assembled matrix with the intended linear solver.
- **Match element evaluation to callback intent.** Generated native kernels
  expose a joint R/K path and, for current sources, a residual-only path.
  Joint caching avoids duplicate tangents at paired callbacks; split evaluation
  avoids constructing tangents for line-search and acceptance-only residuals.
  Compare both with identical thread/rank settings and retain call counts.

## Linear-solver policy

All the serial linear-solve policy lives in `coupfe/assembly/factored.py`.
Its configurable thresholds are defaults, not portable performance evidence;
profile assembly, setup, factorization, and solve time on the target problem.

- `newton_solve` / `solve_dynamics` route through `linear_solve(K, b)` automatically —
  nothing to configure for a normal run.
- Factor-once / solve-many workloads (condensed contact, repeated back-solves
  against a fixed `K`) use
  `factored_lu(K, prefer="scipy"|"auto"|"petsc")`. `auto` applies the
  configurable size policy; an explicit preference is an application decision,
  not a portable performance claim.
- MPI: `solve_distributed(..., solver=..., pc=...)` is a separate path; see
  `skills/distributed.md`. Start with one OpenMP/BLAS thread per rank and change
  that only after profiling the actual configuration.
- Env knobs: `COUPFE_LINEAR_SOLVER=scipy` forces scipy everywhere;
  `COUPFE_FACTORED_PETSC_MIN_N` moves the size threshold.

**Do not copy hand-rolled PETSc KSP setup from example code.** New code imports
from `coupfe.assembly.factored`; if a
genuinely new solver need appears (iterative, preconditioned, GPU), extend
`factored.py` so the policy stays in one place.

### Iterative solves

`iterative_solve(K, b)` provides an opt-in GMRES/AMG path. Its suitability
depends on the matrix, scaling, and preconditioner.

- Contact terms can create high contrast, nonsymmetry, and near-null modes that
  require a formulation-specific preconditioner. The current examples often
  use direct solvers; that is not a universal policy for all contact problems.
- A direct-solver failure at contact engagement can indicate a singular model
  or missing bulk stiffness. Inspect null modes and configuration before
  treating a backend change as the fix.

### The solver ladder

| Rung | Use | Verification note |
|---|---|---|
| scipy direct (SuperLU) | small systems and portable fallback | profile fill-in and factorization memory |
| PETSc MUMPS direct | serial sparse-direct candidate when SciPy fill-in dominates | remeasure the crossover on the target matrix |
| Krylov + AMG | large, well-conditioned SPD/scalar blocks | require convergence and `rtol`-sensitivity checks |
| FieldSplit + per-block solvers | coupled multifield systems | scale each equation and verify the block split |
| Distributed direct | tight 1-vs-N gates at moderate size | use a reproducible supported backend |
| Distributed iterative | larger distributed systems | rank-independent iteration behavior is a useful health check |

Common traps:

- **Symmetric Dirichlet for CG**: eliminate rows AND columns (`zeroRowsColumns`) to keep the
  operator SPD — plain row-zeroing silently breaks CG.
- **Solver-aware 1-vs-N tolerances:** iterative agreement is limited by KSP
  tolerances and conditioning; a direct solve can provide a tighter small-case
  comparison when the backend is available.
- **Rank-dependent iteration behavior** should trigger an ownership,
  preconditioner, and conditioning audit.
- **A zero pivot at contact engagement** may indicate a degenerate or
  underconstrained model; diagnose it rather than assuming either model or
  solver is solely responsible.
- **Ill-conditioned large grids have a direct-solver roundoff floor.** Two
  direct solvers can disagree near that floor; diagnose conditioning and the
  residual before calling it a defect.
- **Threads:** start with `OMP_NUM_THREADS=1` and matching BLAS settings under
  `mpirun`. Retain a profile before selecting a different thread/rank balance.
