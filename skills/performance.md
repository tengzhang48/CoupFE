# CoupFE performance / acceleration skill

How to make an operator fast **without premature compilation**. The operator contract is
acceleration-agnostic — an operator returns `(R, COO)` regardless of *how* it's computed —
so you pick the tool per operator, and no framework fights you (the FEniCSx pain).

> **Evidence note:** timings, crossover sizes, and speedups in this dated skill
> are historical local measurements. Raw logs and a locked machine/software
> environment were not retained for the release candidate. Use them as
> profiling hypotheses, not public benchmarks; rerun and archive the evidence
> before citation.

## The ladder — escalate only when you have MEASURED a bottleneck

1. **Vectorize numpy first.** numpy is not slow; a Python *loop* is. Most operators
   vectorize: the rigid-contact force + complex-step tangent does 50k nodes in ~20 ms once
   the per-node loop is gone. Replace per-element / per-node Python loops with batched array
   ops before reaching for anything compiled.
2. **numba for the irregular loops that don't vectorize** — search, narrow-phase, tree/BVH
   traversal. `@njit` is near-C, `prange` is multicore, `numba.cuda` is GPU. It is
   **production-viable for most problems** (≤ ~1e5–1e6 features), *not* just a prototype, and
   it stays in Python — one language, full control, clean here because we own the operator
   (unlike injecting numba into FEniCSx's FFCx pipeline). Caveats: array-based data (no Python
   objects in the hot loop — e.g. a BVH as arrays) and a one-time JIT warmup.
3. **f2py Fortran for the hot, REGULAR, DUAL-HOME inner loops** — the element kernels (hot
   over *all* elements, and they must also run as Abaqus UELs). Fixed-format `.for`,
   complex-step tangent inside.
4. **Rust / C++ — escalation only**: the extreme scale (ppf-contact-solver does 180M contacts
   in Rust + CUDA — portable algorithms if we go there), GPU-heavy work where mature CUDA
   matters, or a distributable binary. **Not** fixed-format Fortran for irregular/tree code
   (great for kernels, awkward at trees).

## The discipline

- **Measure before escalating; never pre-compile.** A Python loop over a *subset* (contact
  nodes, a surface) is often fine; a loop over *all elements* is not. Profile the real cost.
- **The contract isolates acceleration.** Each operator's residual/tangent can be
  numpy / vectorized / numba / f2py / Rust **independently** — speed up the bottleneck
  operator and leave the rest. No global rewrite.
- **Compile the hot / regular / dual-home; keep dynamic / non-smooth / orchestration in
  Python** until the loop is the measured bottleneck. (Bulk elements → Fortran; contact force
  → vectorized numpy; contact *search* → numba; everything else → Python.)
- **The full-control advantage:** because we own the loops and the data layout, every rung of
  this ladder is available cleanly — which is exactly what made it tricky in FEniCSx and is
  easy here.
- **For large 3D direct solves, distinguish assembly from factorization.** The 3D Cattaneo/Abaqus
  mesh (77,760 DOF) assembled with the batched f2py element path in seconds, but SciPy/SuperLU
  factorization took minutes per Newton solve and made the run look "too big." PETSc/MUMPS exposed
  the real nonlinear behavior (`ksp_its=1`; contact active-set chatter at stiff penalty). Before
  shrinking a mesh or rewriting assembly, time one assembled matrix with the intended linear solver.

## Linear solvers — ONE policy, don't hand-roll

Doc contract for this section (Teng, 2026-07-02): **simple, informative, correct** — one
table, measured anchors on every claim, bounds stated. That is sufficient for an agent to
choose the right solver; anything more detailed belongs in the individual project.

All the serial linear-solve policy lives in `coupfe/assembly/factored.py`. The rule
(measured, this box): **scipy** for small systems and for 2D meshes up to the measured
~130k DOFs (PETSc setup overhead only slows them; NOT an any-size claim — see the ladder
below), **PETSc MUMPS** for 3D systems above ~20k DOFs (SuperLU's 3D fill-in wall:
400 s vs MUMPS ~45 s factor at 96k DOFs).

- `newton_solve` / `solve_dynamics` route through `linear_solve(K, b)` automatically —
  nothing to configure for a normal run.
- Factor-once / solve-many workloads (condensed contact, repeated back-solves against a
  fixed `K`) use `factored_lu(K, prefer="scipy"|"auto"|"petsc")` — the caller passes the
  dimensionality hint, because fill-in behaviour is what decides and only the call site
  knows the mesh.
- MPI: `solve_distributed(..., solver=..., pc=...)` — a separate path, see `skills/distributed.md`
  (`OMP_NUM_THREADS=1` under mpirun; threaded BLAS is for SERIAL MUMPS only).
- Env knobs: `COUPFE_LINEAR_SOLVER=scipy` forces scipy everywhere;
  `COUPFE_FACTORED_PETSC_MIN_N` moves the size threshold.

**Do not copy hand-rolled PETSc KSP setup from historical example code.** The
private, withheld Cattaneo investigation predates this module and is not in the
public artifact. New code imports from `coupfe.assembly.factored`; if a
genuinely new solver need appears (iterative, preconditioned, GPU), extend
`factored.py` so the policy stays in one place.

### Iterative solves — opt-in, and know why you're opting in

`iterative_solve(K, b)` (gmres+gamg, same module) exists for very large WELL-CONDITIONED bulk
systems past the direct wall (3D beyond ~1-2M DOFs on this box). Two documented cautions:

- **Contact-stiffened systems stay DIRECT.** Penalty/barrier terms create high-contrast entries
  AMG handles poorly, and a Krylov tolerance conflates linear-solver error with contact
  diagnostics — the direct solve's `ksp_its=1` is exactly what isolated the real active-set
  chatter in the cattaneo-3d investigation. Validation / 1-vs-N invariant work also needs exact
  solves (`skills/distributed.md`).
- **A direct-solver "failure" at contact engagement is usually the MODEL.** The recorded
  MUMPS/superlu_dist zero-pivot errors (`docs/lessons_learned.md` 2026-06-23,
  `docs/dev/contact_experiments.md`) came from a degenerate no-bulk test config; an iterative
  solver "working" on such a system is a breadcrumb, not a fix. The consolidated module now
  WARNS on PETSc→scipy fallback so this failure class is visible, not silent.

### The solver ladder — common knowledge across the lab projects

Surveyed 2026-07-02 across CoupFE and its application implementations. Detailed external
setups stay in their respective repositories; this repository's implementation is
`coupfe/assembly/factored.py`. What is common is the ladder and the traps:

| Rung | When | Measured anchor |
|---|---|---|
| scipy direct (SuperLU) | small anything; 2D **at the sizes we've measured (≲130k DOFs)** — NOT an any-size claim; 2D *contact* stays direct for exactness, not speed | PETSc setup overhead dominates small systems; but at 490k-DOF 2D Poisson SuperLU takes 7.3 s vs hypre 0.9 s — re-measure past ~10⁵ DOFs |
| PETSc MUMPS direct | 3D above ~20k DOFs; anything needing exact solves | 96k DOFs: 400 s (SuperLU) → ~45 s (MUMPS), CoupFE |
| Krylov + AMG — **hypre first, gamg fallback** (the `iterative_solve` default) | LARGE well-conditioned SPD/scalar blocks — already wins 2D SPD at ~90k DOFs | this box: hypre 0.13/0.36/0.90 s vs gamg 0.44/0.75/1.73 s vs SuperLU 0.56/2.5/7.3 s at 90k/250k/490k; lab: constant 7 its, mesh-independent, ~20× MUMPS at 79k |
| FieldSplit + AMG per block | coupled multifield (u-c-phi, phi-T) — never hand one ill-scaled coupled matrix to AMG whole | **now in core: `make_fieldsplit_solver`** (ported from abaqus_ufl.fe, gated in `tests/test_factored.py`; EDA runs the same pattern); scale each equation to O(1) first (the mu/RT lesson) |
| Distributed direct | 1-vs-N gates, moderate size | `superlu_dist`, never parallel MUMPS (non-reproducible pivoting) |
| Distributed CG+GAMG | large distributed SPD | rank-independent iteration count = the healthy-PC signal (EDA PDN) |

Cross-project traps (each cost someone a debugging cycle):

- **Symmetric Dirichlet for CG**: eliminate rows AND columns (`zeroRowsColumns`) to keep the
  operator SPD — plain row-zeroing silently breaks CG (EDA).
- **Two tolerance regimes for 1-vs-N**: iterative matches serial only to the KSP `rtol`;
  reproducible direct (`superlu_dist`) matches to machine precision — use direct for the tight
  gate (EDA + `skills/distributed.md`).
- **A rank-DEPENDENT iteration count** flags a distributed-state or preconditioner bug, not a
  convergence quirk (EDA).
- **Contact-stiffened matrices stay direct** (this repo, above); **zero pivot at contact
  engagement = model problem** (degenerate/no-bulk config).
- **Ill-conditioned large grids have a direct-solver roundoff FLOOR** — two exact solvers can
  legitimately disagree at that floor; it is the conditioning, not a bug (EDA PDN).
- **Threads**: `OMP_NUM_THREADS=1` (and OPENBLAS/MKL) under `mpirun`; threaded BLAS helps only
  serial MUMPS.
