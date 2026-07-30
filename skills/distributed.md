# CoupFE distributed-FE development skill

How to add and verify distributed (MPI) code. Read alongside `skills/testing.md` and
`docs/dev/distributed_mesh.md`.

## Scope: regular meshes (do not build the worst case)

CoupFE targets relatively regular meshes (structured / block-structured). On those,
distribution is **block/coordinate partition + structured ghost halos +
embarrassingly-parallel uniform refinement** — no ParMETIS graph partitioning, no parallel
AMR / hanging-node constraints, no dynamic load balancing. **Do not** build those; for a
genuinely irregular geometry, put **gmsh + DMPlex behind the `KernelMeshView` contract**
(the kernels never change). We *use* the general machinery there, we don't *rebuild* it.

## The one gate: the 1-vs-N invariant (verify in two layers)

A distributed result must equal the serial result. Prove it twice:

1. **In-process (no MPI).** Partition → owned/ghost `LocalMesh` → assemble each part's owned
   cells → scatter to global → **sum == serial**; and the gather/reduce adjoint
   `⟨Gu,v⟩ = ⟨u,Gᵀv⟩`. Catches decomposition / local↔global-numbering bugs cheaply.
   (`tests/test_distribute.py`.)
2. **Real MPI.** `serial == N-rank` to **machine precision** (~1e-15) at **≥2 rank counts**.
   Catches comm / PETSc-numbering bugs the in-process layer can't see. Run via subprocess
   (mpirun can't run inside pytest), guarded by mpirun/petsc4py availability.
   (`tests/test_mpi_distributed.py`, `examples/mpi_smoke/`.)

Never claim "parallel works" from a single rank count, a loose tolerance, or a serial-only
test.

## Memory-local generation

Each rank generates **only its block** — compute connectivity on the fly from
`(rank, dims)`; never build the global mesh on any rank. That is the
10M-DOF-without-gathering lever, and on a structured mesh it is trivial.

## PETSc mechanics (petsc4py only — never import mpi4py)

- Assemble with **global** indices + `ADD_VALUES` + `Mat/Vec.assemble()`; PETSc sums the
  off-process contributions, so you do not hand-manage ghosts for *assembly* correctness.
- Symmetric Dirichlet: `Mat.zeroRowsColumns(rows, diag, x, b)`; match the serial reference's
  reduced system so the comparison is exact.
- Gather for the serial compare: `PETSc.Scatter.toZero(vec)`.
- Always `OMP_NUM_THREADS=1` (and OPENBLAS/MKL) under mpirun — oversubscription otherwise.

## Signals to read

- A **rank-independent** iteration count is healthy. A **rank-dependent** one flags a
  distributed-state or preconditioner bug.
- A monotonic N× slowdown with more ranks is a **setup** bug (ranks × threads vs cores, the
  launch line), not a fundamental property — audit that first.

## Scalable preconditioning (PETSc): FieldSplit + the rigid-body near-null-space
The 1-vs-N gate proves *correctness*; making the KSP iteration count **flat with problem size** is
a separate problem, and for coupled / elastic blocks it has specific recipes (validated in the
CoupFE-EDA distributed work — the FieldSplit/near-null-space the core scopes as the next step).

- **Coupled multi-field → `PCFIELDSPLIT` (GAMG per scalar field).** ASM/block-Jacobi have no coarse
  grid → iterations grow with size (impractical at millions of DOF); plain GAMG **mis-coarsens** an
  interleaved `(φ,T)` block. Split into scalar fields and GAMG each one → mesh-independent (`745→17`
  iters, flat over a 16× size range). Two traps that *look like* "won't converge":
  - **The FieldSplit index sets must be each rank's LOCALLY-OWNED field DOFs**, not the global set:
    `loc = arange(rs, re); is_phi = loc[loc % 2 == 0]`. Passing the global IS → GMRES stagnates with
    a *wrong* answer.
  - **Use a full-load Newton loop, not load-stepping-with-one-linear-solve**, for a nonlinearity
    (e.g. a quadratic Joule `σ|∇φ|²`) — one solve per step converges the KSP but not the fixed point.
    `solve_distributed` does Newton internally; a hand-rolled FieldSplit driver must do it explicitly.
- **An elasticity (displacement) block needs the rigid-body near-null-space, or GAMG is useless.**
  Plain GAMG on a u-block: ~**565** iters; with the 6 rigid-body modes attached: ~**21**, and
  mesh-independent. Budget for it from the start. Recipe: `MatNullSpace.createRigidBody(coords_vec)`
  (coords as a block-size-3 Vec) → `Au.setNearNullSpace(ns)` on the u sub-block
  (`ksp_u.getOperators()[0]`), **after** `ksp.setUp()` (sub-blocks don't exist before it), **before**
  the solve (GAMG sets up lazily). `PC.setCoordinates()` *breaks* GAMG here — use the explicit
  null-space. A fully-prescribed (Dirichlet-everywhere) field's block is the identity → GAMG can't
  coarsen it (`jacobi` instead).
- **Assemble owned-row CSR; do NOT use `setValuesCOO` if you solve more than once per process.**
  PETSc's `setPreallocationCOO`/`setValuesCOO` (MPIAIJ) **corrupts global state** — a *confirmed
  upstream bug* (≈30-line pure-petsc4py Laplacian reproducer, fails identically on pip-wheel 3.25.2
  AND conda-forge 3.24.2). The *first* solve after a COO assembly works; *every subsequent* one fails
  (GAMG stagnates reason −3; ILU/bjacobi PC-setup reason −11; `MatConvert` segfaults), so Newton
  diverges at iter 1 — even though the matrix matches `setValues` to ~1e-15 (it's memory corruption,
  not a value bug). The fast *and* safe path is the textbook one: a **node-aligned row partition**
  (rank owns nodes `[na,nb)` → rows), evaluate the elements **touching** owned nodes (owned + 1 ghost
  layer) in one batched `element_rk_batch`, keep the **owned-row** triplets, build a local
  `scipy.sparse` CSR (vectorized, sums duplicates), `createAIJ(csr=...)` — plain AIJ, no off-process
  routing, no COO, no Python loop (≈7× faster than a per-element `setValues` stamp loop).
- **Gate the MECHANISM, not just convergence.** Assert an absolute `ksp_its` *bound* (e.g. `< 40`) —
  that proves the near-null-space / FieldSplit is doing its job. "KSP-converged" alone can hide both
  traps above (the serial==N-rank gate against an independent oracle is what catches them).

## The production path (DONE — `coupfe/assembly/distributed.py`)

`solve_distributed(...)` is the production nonlinear-FE distributed solve: finite-strain
load-stepped Newton + line search across ranks, serial == N-rank to **machine precision**
(`examples/mpi_smoke/distributed_neohookean.py`, gated 2/4 ranks). Key points for extending it:

- **The operator contract is the SERIAL interface; it does not go distributed.**
  `assemble_residual`/`assemble_tangent` take the *global* `U` and return *global* `gdofs` —
  an O(ndof) call. Memory-local distributed can't use it. The distributed unit is instead the
  **batched element evaluator** `CompiledElement.element_rk_batch(coords, U, DU) -> (R, K)`
  (the same f2py call the `ElementGroup` scatters serially). `solve_distributed` takes that
  `batch_fn` + the rank's partition (`my_gm`, `my_coords`), not an operator list. So: a new
  *element* is distributed for free (same batch_fn); a non-element operator (contact!) needs
  its own distributed contribution — don't expect the serial `Operator` to carry over.
- **Build-then-partition keeps the global numbering identical** to the serial solve, so the
  1-vs-N compare is *exact*. `element_partition(view, rank, size)` does this. Memory-local
  structured generation (each rank builds only its block) is the separate scale optimization
  — same per-rank assembly logic, different mesh source.
- **An exact (direct) linear solve makes the invariant tight.** `pc="lu", solver="mumps"`
  → serial == N-rank to ~1e-16 (a clean correctness signal). An iterative PC only matches to
  its `rtol`; use direct for the *gate*, `gamg`/`gmres` for *scale*.
- **No f2py build race across ranks** — each rank is a separate process, builds the kernel in
  its own tempdir, imports by module name into its own `sys.modules`. Concurrent gfortran
  compiles, no shared-file conflict. (Each rank pays one build; cached thereafter in-process.)
- **History-free (hyperelastic) is the right FIRST distributed target**: the residual uses
  total `U`, so the serial (`DU=U`) vs distributed (`DU=U−U_prev_step`) convention difference
  doesn't affect the result — it isolates *distributed correctness* from the *state-handoff*
  question. Path-dependent elements add per-element state commit + carrying state across
  increments (tracked, like the serial `solve_increments` note).

Faithful extensions still tracked from the lab's `solve_steps_mpi_local`: coupled-field
`PCFieldSplit` (+ AMG near-null-space coords), diagonal scaling, the per-element state commit,
and `partition_unstructured` + `local_ndof` for non-contiguous partitions.

## Distributed rigid contact (DONE — `solve_distributed(..., contact=...)`)

Rigid-obstacle contact (frictionless or Coulomb friction) wires in as a **node-local**
per-rank contribution — each contact node handled by exactly one rank (the one owning its
first DOF), which holds its friction stick-state. No cross-rank coupling (the analytical
obstacle is known everywhere). Two rules that bit us:
- **A collective op must be called by ALL ranks.** Reading contact positions uses a PETSc
  `VecScatter` (collective). With the even DOF split, all boundary contact nodes can land on
  rank 0, leaving others with none — if those ranks skip the scatter, the others **hang**.
  Guard only the local `setValues`; call the scatter/assemble/reduce unconditionally (let the
  kernel return empties for a rank with no contact nodes).
- **Use a REPRODUCIBLE parallel solver — `superlu_dist`, not MUMPS.** MUMPS's parallel
  pivoting/scheduling is non-reproducible (~1e-12 run-to-run); a smooth solve hides it, but an
  ill-conditioned mode (e.g. a slipping frictional node ≈ zero tangential stiffness) amplifies
  it into a visible, non-repeatable ~1e-3. `superlu_dist` solves the *same* non-smooth problem
  rank-independently and run-to-run repeatably to **1.3e-16**. (Both handle the asymmetric
  friction tangent correctly — it's purely a reproducibility difference.) This is the default
  in `solve_distributed`.

## Distributed DEFORMABLE contact (DONE — `deformable_contact=…`)

Unlike rigid (node-local), a deformable pair couples a secondary *vertex* with a primary *edge/face*
that after partitioning can live on **different ranks**. The recipe (one shared helper,
`_DistDeformableContact`, used by both the quasistatic `solve_distributed` and the dynamics
`solve_dynamics_distributed`):
- **Replicate the contact surface** (secondary ∪ edge/face nodes) to every rank each iteration via a
  `VecScatter` — O(surface) ≪ ndof, NOT an all-gather of U.
- Each rank assembles the contact COO for the pairs it **owns** (2D / 3D vertex-face: by the
  secondary vertex's first DOF; 3D edge-edge: by the first node of the pair's first edge, via the
  operator's `owns_edge_pair`) with **global** dof indices; PETSc off-process `ADD_VALUES` routes the
  off-rank stencil nodes. Each pair is assembled exactly once.
- **Global CCD** (barrier): each rank's local point-edge/point-triangle `max_step` is reduced to the
  global minimum with a 1-entry-per-rank `Vec.min()` (the limiting pair can be on any rank).
- **Penalty** runs quasistatic; the **penetration-free barrier** needs **dynamics**
  (`solve_dynamics_distributed`: node-local inertia `M/dt²` + Rayleigh damping + gravity, ghosted
  bulk, CCD-bounded predictor) — it does NOT converge quasistatically (the residual-norm line search
  stalls at the projection flip; ppf has no energy-merit line search, it relies on dynamics).
- **Friction** rides the dynamics with no new distributed code (the helper threads `mu`; the
  path-dependent `_x0` is rank-independent because U is). **3D** is the same helper subclassed; the
  cross-rank assembly + global CCD are verified to machine precision. All rank-independent (1-vs-N).
- **Gotcha:** verify with a real (bulk-backed) system or a static cross-rank assembly check — a
  no-bulk / mass-point system is degenerate (direct LU fails at engagement, PETSc SEGVs on empty
  objects); that's a missing-element artifact, not a contact bug.

## Dual-multiplier friction in parallel — assessment (good for SMALL interfaces, not the scale path)

Is the dual-multiplier (`contact_semismooth`) "good in parallel"? **Yes for a small
contact interface (the common case), no as a route to scale** — and that's a deliberate decision, not a
missing feature.

- **What parallelizes (verified).** The method condenses to the interface: `G = S Kff⁻¹ Sᵀ`,
  `H = Knf Kff⁻¹ Sᵀ`. The only expensive, parallelizable work is the `nc` bulk solves that build
  `Kinv_ST = Kff⁻¹ Sᵀ` — distribute those over a PETSc KSP (`superlu_dist`), the same path the rest of
  CoupFE uses. `examples/mpi_smoke/distributed_dual_multiplier.py` does exactly this and is
  **rank-independent**: 1-vs-{2,4} ranks agree to `~1e-14` (the active set is identical because the
  bulk solve is reproducible — use `superlu_dist`, never MUMPS).
- **The design: bulk parallel, interface REPLICATED.** `G` (dense `nc×nc`) is gathered to every rank
  and the active-set iteration runs **redundantly** on the (tiny) interface. Negligible when
  `nc ≪ ndof` — and contact is a *surface*, so `nc ~ N^{1/2}` (2D) or `N^{2/3}` (3D). For ordinary
  contact this is the right trade: bulk dominates and scales, interface is free.
- **The ceiling (why it's not the scale path).** The interface is **dense** (`G` formed by `nc`
  backsolves; solve is `O(nc²)`–`O(nc³)`) and **replicated** (serial on every rank, no parallelism in
  the active set). A *large/fine* contact interface makes the dense-Schur formation + the replicated
  active-set the bottleneck — neither scales. The HPC-scalable dual-multiplier is **not** dense Schur:
  it's **FETI/TFETI** (multipliers = the FETI dual, a coarse problem for global coupling) + **MPRGP**
  (parallel bound-constrained QP) — what **PERMON/FLLOP** does (`docs/dev/permon_assessment.md`). That
  scales, but is **forward-only** (not differentiable — our edge).
- **Decision (`docs/dev/dual_multiplier_strategy.md`):** don't parallelize the dense-Schur
  dual-multiplier for scale. The **smoothed (ppf) friction is our scalable distributed path** —
  node-local residual/tangent on each pair, no dense interface, no replicated active set, already
  rank-independent to many ranks (the deformable-contact section above). The dual-multiplier's value is
  the **differentiable small-strain partial-slip niche** (fretting/joints; `nc` small), where the
  parallel bulk-solve already suffices. Forward-at-HPC-scale → PERMON; relay for gradients.

## When a parallel result surprises you, swap ONE thing — don't theorize

A distributed number you can't fully explain (a rank-dependent count, a non-repeatable
solution, an N× slowdown) is a setup/tooling issue until proven otherwise. **Vary one
component and watch what moves**: the linear solver (MUMPS↔superlu_dist — isolated the
repeatability bug in one run), the rank count, OMP threads, the regime. A plausible mechanism
("non-smooth ⇒ non-deterministic") is *not* evidence — demand a check that would fail if the
guess were wrong, and cross-check against what you already know. Repeatability is
non-negotiable; never rationalize its absence. See `docs/lessons_learned.md`.
