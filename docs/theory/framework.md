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
For supported generated elements, **one residual is the source of truth and the
tangent is derived from it by complex step**,
`Kᵢⱼ = Im(Rᵢ(u + i·h·eⱼ))/h`, with a small `h`. This avoids subtraction
cancellation for analytic `R`. Other operators may use analytic or semismooth
tangents, which require independent consistency tests. The contract is
acceleration-agnostic: an operator returns `(R, COO)` however it is computed,
so an implementation can be optimized without changing assembly.

## 2. The finite-strain element

**Reference configuration, total-Lagrangian.** The bulk stays in the reference frame; the
stress measure is the first Piola–Kirchhoff `P`. For a hyperelastic material `P = ∂Ψ/∂F`, with
`F = I + ∂u/∂X` the deformation gradient. The element residual is the internal force
`Rₑ = ∫ Bᵀ P dV` (one source of truth); its tangent is the complex-step of
`Rₑ`. When the complete residual path is analytic, this includes the material
and geometric dependence without maintaining a separate hand-coded tangent.

**Compiled, parallel targets.** A supported element declaration can generate a
CoupFE-native kernel (`coupfe_element_rk`, plus an optional
`coupfe_element_r`) and an Abaqus UEL. They implement the same declared weak
form through different solver interfaces; the native ABI does not carry
Abaqus procedure flags. `CompiledElement.element_rk_batch(coords, U, DU) →
(R, K)` evaluates a whole native element group in one f2py call. This batched
evaluator is the unit `ElementGroup` scatters to global COO and the unit used
by the distributed bulk path. The in-process UEL wrapper remains a narrow
normal-static compatibility adapter for focused parity checks and selected
research examples. It converts Abaqus's `RHS = −R` convention so callers see
the standard `(R, K)`; Abaqus itself supplies its UEL procedure flags in an
Abaqus analysis.

## 3. The Newton driver — basin control

A finite-strain load applied in one Newton solve from a poor initial guess can
fall outside Newton's basin: a trial step may invert an element
(`det F = J < 0`) and make a logarithmic material response nonfinite. Two
controls are available in the core driver:

- **Load incrementation** (`solve_increments`) — ramp the Dirichlet data over warm-started
  increments so each Newton solve starts near its root.
- **Backtracking line search** (`newton_solve`) — reduce `α` while seeking a
  finite, decreasing trial residual. This improves basin control but is not a
  formulation-specific `det(F)` feasibility guarantee.

Dirichlet conditions are imposed by identity rows (`Kᵍ = eᵍ`, residual
zeroed). Coupled problems should also inspect residuals by field and on
physically meaningful scales; a single global norm can hide a weakly scaled
equation. The current generic driver returns an iteration count, not a
convergence flag, and calls operator commit after its loop. It therefore does
not by itself provide accepted-step orchestration for every path-dependent use.

## 4. Distributed-memory solve

**Domain decomposition.** The regular-mesh helper partitions elements by a
deterministic coordinate ordering. A node is owned by the lowest-id part
touching it; other touching parts treat it as a ghost. General graph-quality or
minimal-halo claims are outside this helper.

**Memory-local generation is the scale lever.** When a structured generator is
used, each rank can build only its local block from `(rank, dims)`, avoiding a
global mesh allocation. Build-then-partition is also provided for smaller
problems and exact numbering comparisons.

**Assembly.** Each rank assembles its owned cells into a *distributed* PETSc `Mat`/`Vec` with
**global indices + `ADD_VALUES`**; `assemble()` sums the off-process contributions, so ghost
reduction for *assembly correctness* is automatic — the memory win is in not building the
global mesh, not in hand-managing ghosts. `U` is ghosted to each rank via a `VecScatter` (no
all-gather).

**The contract is serial; the element is the distributed unit.** `assemble_residual`/
`assemble_tangent` take the *global* `u` and return *global* gdofs — an O(ndof) interface that
cannot go distributed. So `solve_distributed` is built on the **batched element evaluator**
(`element_rk_batch`) over a rank's partition, not the serial operator list. A
non-element operator such as contact needs its own per-rank contribution. A new
history-free, single-field element can reuse this path when
it satisfies the batch evaluator's layout and state assumptions; distribution
is not inferred from a serial element test alone.

**Correctness gate — the 1-vs-N invariant.** A distributed path should be
compared with its serial oracle at more than one rank count, with tolerances
appropriate to the chosen linear solver. In-process ownership and reduction
tests catch numbering and assembly defects before an MPI run. A
rank-independent iteration trend is a useful preconditioner diagnostic, not an
independent physics validation. The current repository ships rerunnable MPI
programs but no retained final-revision multi-rank qualification record.

**Solver reproducibility.** Available direct and iterative backends depend on
the PETSc build, and parallel pivoting or Krylov tolerances can change the
comparison floor. Record the backend, tolerances, rank count, and environment
with a result. Run with `OMP_NUM_THREADS=1` unless the target configuration has
been deliberately profiled otherwise.

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

## 6. The model pipeline

`Model` is a thin declarative layer over the contract: it collects operators,
builds the Dirichlet map, and invokes a driver. Application-specific meshing,
boundary semantics, loading, and output remain outside the numerical core.
Those parts are important model code and require their own tests.
