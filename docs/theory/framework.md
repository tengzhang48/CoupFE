# Theory — the core framework (element, Newton, distributed, mesh)

The mathematical basis of the non-contact pieces of CoupFE. Companion to
`docs/theory/contact_dynamics.md` (contact/barrier/friction/dynamics) and the narrative
`docs/lessons_learned.md`. Notation as in the companion: `u` DOFs, `R(u)` residual,
`K = ∂R/∂u` tangent; Newton solves `K δu = −R`.

## 1. The operator contract and one-residual / complex-step tangent

Everything is a residual-contributing operator with `(residual, tangent, commit)`. The driver
sums contributions and knows nothing about elements/materials/contact — only the contract:
```
R(u) = Σₒ Rₒ(u) ,   K(u) = Σₒ Kₒ(u)
```
**One residual is the source of truth; the tangent is derived from it by complex step**
`Kᵢⱼ = Im(Rᵢ(u + i·h·eⱼ))/h`, `h ~ 1e-30` — exact (no cancellation) for analytic `R`. A
hand-coded tangent is a second source of truth that drifts; we never keep one. `commit`
recomputes-and-returns accepted state (never mutate committed state mid-solve). The contract is
**acceleration-agnostic**: an operator returns `(R, COO)` however computed (numpy / f2py /
numba / Rust), so speed is a per-operator choice, not a framework rewrite.

## 2. The finite-strain element

**Reference configuration, total-Lagrangian.** The bulk stays in the reference frame; the
stress measure is the first Piola–Kirchhoff `P`. For a hyperelastic material `P = ∂Ψ/∂F`, with
`F = I + ∂u/∂X` the deformation gradient. The element residual is the internal force
`Rₑ = ∫ Bᵀ P dV` (one source of truth); its tangent is the **complex-step** of `Rₑ` — so the
geometric + material stiffness come for free, with no hand-coded `∂P/∂F`.

**Compiled + dual-home.** The element is a fixed-format Fortran `.for` (`neo_hookean_q4.for`)
that runs *both* as an Abaqus UEL and standalone here via f2py + the `drive_uel` wrapper.
`CompiledElement.element_rk_batch(coords, U, DU) → (R, K)` evaluates the **whole element group
in one f2py call** (~95× a per-element Python loop) — this batched evaluator is the unit the
`ElementGroup` scatters to global COO, and (see §4) the unit distributed-memory uses. Abaqus
sign convention (`RHS = −R`, `AMATRX = K`) is handled in the wrapper so the operator sees the
standard `(R, K)`.

## 3. The Newton driver — basin control

A finite-strain load applied in *one* Newton solve from a zero interior usually falls outside
Newton's basin: the first tangent can be indefinite, the full step overshoots, an element
inverts (`det F = J < 0`), and `ln(J)` → NaN. Two remedies, both in the core driver:

- **Load incrementation** (`solve_increments`) — ramp the Dirichlet data over warm-started
  increments so each Newton solve starts near its root.
- **Backtracking line search** (`newton_solve`) — damp `α` (halving) until the trial residual
  is *finite and decreasing*. This is what catches the `ln(J<0)` NaN. A subtlety: **finer
  meshes make the first-iteration distortion worse** (a smaller boundary element sees a larger
  local strain for the same boundary displacement), so the same step that converged coarse can
  diverge fine until the line search is in place.

Dirichlet is imposed by identity rows (`Kᵍ = eᵍ`, residual zeroed). **Coupled convergence is
field-wise**, not a single global norm: split `‖R‖` by field and divide each by its block scale
(a momentum-dominated global norm hides an under-converged transport field — the gel
under-swell saga). Count iters/step: 3–5 means the tangent is fine and a wrong "converged but
drifting" result is the *gate*, not the model.

## 4. Distributed-memory solve

**Domain decomposition.** Partition elements (coordinate sort-and-chunk along the longest
axis: contiguous, balanced, small ghost halos). A node is **owned** by the lowest-id part
touching it; others see it as a one-layer **ghost**.

**Memory-local generation is the scale lever.** Each rank generates ONLY its block — compute
connectivity on the fly from `(rank, dims)`; never build the global mesh on any rank. That is
what enables ~10M DOFs without gathering.

**Assembly.** Each rank assembles its owned cells into a *distributed* PETSc `Mat`/`Vec` with
**global indices + `ADD_VALUES`**; `assemble()` sums the off-process contributions, so ghost
reduction for *assembly correctness* is automatic — the memory win is in not building the
global mesh, not in hand-managing ghosts. `U` is ghosted to each rank via a `VecScatter` (no
all-gather).

**The contract is serial; the element is the distributed unit.** `assemble_residual`/
`assemble_tangent` take the *global* `u` and return *global* gdofs — an O(ndof) interface that
cannot go distributed. So `solve_distributed` is built on the **batched element evaluator**
(`element_rk_batch`) over a rank's partition, not the serial operator list. Consequence: a new
*element* distributes for free; a non-element operator (contact) needs its own per-rank
contribution.

**Correctness gate — the 1-vs-N invariant.** Gathered to one process, the distributed solution
equals the serial solve to **machine precision**, independent of rank count, verified in two
layers: in-process (owned/ghost sum == serial) and real MPI (1/2/4 ranks). A **rank-independent
iteration count** (e.g. CG 16 iters at 1/2/4 ranks) is a health signal; a rank-dependent one
flags a distributed-state/preconditioner bug. **Build-then-partition keeps the global numbering
identical so the compare is exact** (memory-local generation is the separate scale step).

**Solver reproducibility (a hard-won caveat).** A *direct* solve makes the invariant
machine-precision, but **MUMPS is non-reproducible in parallel** (dynamic pivoting varies
run-to-run by ~1e-12) — an ill-conditioned mode then amplifies that into a visible difference.
Default to a **reproducible** parallel solver (`superlu_dist`); MUMPS only when repeatability
isn't required. Run with `OMP_NUM_THREADS=1` to avoid oversubscription.

## 5. Mesh — refinement and convergence

`KernelMeshView` is the narrow contract between meshing and physics: operators see only compact
`nodes/elems/labels` arrays, never a DMPlex/forest object, so the kernels are backend-agnostic.

**Curved-boundary refinement.** Boundary *classification* travels with the mesh
(`node_geometry` → a `GeometryBackend`), so uniform refinement re-embeds new boundary nodes on
the true curve (`project`/`normal`), gated by a positive-Jacobian check.

**What re-embedding buys (honest result).** For **linear elements + Dirichlet BC**, the
curved-annulus convergence (Lamé-form oracle) is **~h² with or without** re-embedding — so
re-embedding is a *geometric* improvement (boundary nodes exactly on the curve), **not** a
*solution-accuracy* one. Its solution payoff appears with higher-order geometry (affine geometry
caps a high-order solution) or curved-boundary loads (a faceted boundary has wrong
normals/area). A convergence study is only meaningful once the measured quantity is the
*discretization* error (relative-L2, real `1/r` curvature, load small enough that the
finite-strain floor sits well below the discretization error, read at the asymptotic levels).

**Scope.** Regular meshes (block/coordinate partition, structured halos, embarrassingly-
parallel uniform refinement, no hanging nodes); application repositories may delegate the
worst case to Gmsh + DMPlex *behind* the `KernelMeshView` contract. Core owns the contract,
not the mesh-software adapter or its domain-specific labels.

## 6. The model pipeline (glue, not theory)

`Model` is a thin declarative layer over the contract — collect operators + build the Dirichlet
dict + drive `solve_increments`/`solve_dynamics`. **No physics.** Meshing, BCs, loading, time
integration, output are per-problem glue kept out of the core (the harness validates them).
