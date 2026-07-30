# Distributed mesh generation + refinement — design note (new algorithm)

The goal: reach ~10M DOF **without ever holding the fine mesh on one rank**. Full
treatment: the plan, §26 (`docs/standalone_gpu_plan.md`).

## What exists vs what's new
- **Exists in the legacy lab (port source, not current core):** the distributed *solve* — `abaqus_ufl`
  `solve_steps_mpi_local`, `partition_rcb`/`partition_unstructured` distribute and
  solve an *existing* mesh (rank-independent to 16 ranks). Basic gmsh generation
  (`fe/mesh.py`).
- **New (this note):** building/refining a large mesh **in parallel**, re-embedding
  curved geometry, the compact kernel-view abstraction, and no-gather I/O.

## Scope decision: target regular meshes (application-owned DMPlex escape hatch)

A *general* parallel mesh (DMPlex, p4est) must handle the worst case — arbitrary
unstructured topology, adaptive refinement with hanging nodes, dynamic load balancing —
and that is exactly what makes it large. **CoupFE deliberately does not.** Most research
(and the consulting target) uses **relatively regular meshes** — structured /
block-structured / forest — where the distributed problem collapses:

- **partition** = block/coordinate decomposition (no ParMETIS graph partitioner);
- **ghost halos** = structured neighbours (simple, predictable);
- **uniform refinement** = embarrassingly parallel (cell→children + halo, no cross-rank
  coordination);
- **no hanging nodes** (uniform), **no dynamic repartitioning**.

So the regular-mesh distributed path is simple *and* scalable without the general
machinery — and the lab already has the pieces (`partition_structured_quad`,
`solve_steps_mpi_local`). **M3b on regular meshes is therefore a small step, not a
DMPlex-scale build.**

A powerful combination falls out: **structured topology + curved geometry via M1/M2
re-embedding** (e.g. the annulus is structured in (r,θ), curved in (x,y)) covers a large
share of real research meshes — plates, blocks, cylinders, annuli, mapped domains — with
the *simple* parallel path and *correct* curved boundaries.

**Escape hatch:** for the occasional genuinely irregular geometry, an application
repository may take an unstructured mesh (for example from Gmsh), use DMPlex if its
scale requires it, and convert the result to the same `KernelMeshView` contract. The
kernels never change. CoupFE core does not own or depend on Gmsh, `meshio`, DMPlex, or
the application's physical-label conventions.

**Ownership update (2026-07-29):** EDA and cardiac keep their mesh-software adapters
beside their domain geometry and metadata. If both eventually need identical conversion
mechanics, extract a separate optional interoperability package; do not add those
dependencies to core.

## The pipeline
```
coarse topological mesh + AUTHORITATIVE geometry
   → semantic labels/classification
   → distribute while still small
   → parallel nested refinement (deterministic under rank count)
   → re-embed new boundary nodes in the geometry + quality repair
   → owned/ghost DOFs, hierarchy, compact kernel views
   → bulk (+ contact) operator setup
```
Key principle: **the coarse mesh is topology, not the curved boundary.** Refining a
faceted boundary only makes smaller facets on the wrong surface. Curvature/sharp
features come from a separate **geometry backend** (analytical / CAD / NURBS /
signed-distance / discrete), and feature classification (vertex/curve/surface) must
survive distribution and refinement.

## The two contracts to define first
1. **`KernelMeshView`** — compact, rank-local arrays the operators consume (owned-cell
   connectivity, reference coords, owned+ghost DOF maps, labels, gather/scatter for
   constraints, quadrature-state offsets, contact-feature lists, comm plans). **No
   generated kernel ever sees DMPlex cones/closures/sections.** An application adapter
   hides DMPlex/DMForest (or another mesh provider) behind this.
2. **`GeometryBackend`** — `evaluate/project/normal/derivatives/contains` per entity;
   projection preserves classification (corner→vertex, edge→its curve, face→its face).

## Stages
- **M1 — contracts.** `MeshPlan`, `KernelMeshView`, `GeometryBackend` schemas;
  ownership/ghost/label/constraint semantics; build manifest. Serial → kernel view.
- **M2 — uniform refinement + analytical geometry.** One low-order family (Quad4/Hex8
  to match the runtime); project new boundary nodes to plane/cylinder/sphere; Jacobian
  & boundary-error checks. Demonstrate curved-boundary convergence.
- **M3 — MPI distribution + compact views.** Distribute *before* large refinement;
  owned/ghost DOFs + comm plans; verify serial == N-rank residual/`R`. Target 1M DOF
  without global mesh replication.
- **M4 — 10M bulk benchmark.** Scalable I/O/checkpoint (no rank-0 gather); a
  hierarchy-based / distributed preconditioner; strong/weak-scaling baseline. **This
  is the evidence for the "more scalable than Abaqus" claim.**
- **M5 — application-owned CAD/NURBS adapters feeding the existing contracts.**
  M6 — contact surfaces on the refined curved mesh. M7 — local (region)
  refinement + hanging-node constraints
  (`R_i = Cᵀ R_e`, `K_i = Cᵀ K_e C`; the kernel view carries the gather/scatter `C`).
- Runtime adaptivity (history transfer) is a separate later research track.

**Status: M1 ✓, M2 ✓, M3 ✓, M3b ✓ (core).** `coupfe/mesh/distribute.py` partitions a view
into owned/ghost memory-local `LocalMesh`es (coordinate sort-chunk; lowest-touching-part
owns a node); the invariant *sum of owned-cell contributions == serial* is gated
in-process (`tests/test_distribute.py`, incl. the gather/reduce adjoint), and real-MPI
smokes show serial == N-rank to ~1e-15 at 1/2/4 ranks (gated via subprocess in
`tests/test_mpi_distributed.py`):
- **M3 — distributed assembly** (`examples/mpi_smoke/distributed_residual.py`).
- **M3b (core) — memory-local distributed SOLVE** (`examples/mpi_smoke/distributed_solve.py`):
  each rank generates ONLY its block (no global mesh), assembles a distributed PETSc
  matrix, and PETSc KSP solves it — serial == N-rank with a **rank-independent** CG count
  (16 its at 1/2/4 ranks).

**Remaining:** wire the operator contract + the f2py element + nonlinear Newton onto this
same pattern — the distributed nonlinear-FE *production* solve — which reuses the lab's
proven `solve_steps_mpi_local`. Below was the serial M1/M2 record:

**Earlier — M1 ✓, M2 ✓ (serial).** `coupfe/mesh` ships `GeometryBackend` +
`Circle`/`Sphere`/`Plane`, `KernelMeshView`, `uniform_refine_quad` (with re-embedding),
`check_positive_jacobian`, and `ElementGroup.from_view` (M1, `tests/test_mesh.py`); the
curved-annulus Lamé study converges at **~h²** (M2, `examples/curved_annulus` +
`tests/test_curved_convergence.py`). **Finding:** for Quad4 + Dirichlet, re-embedding is
a *geometric* improvement (boundary nodes exactly on the curve), not a solution-accuracy
one — the solve converges ~h² with or without it; the solution payoff needs higher-order
geometry or curved-boundary *loads* (tractions). Next: **M3** — MPI distribution + the
adjoint-communication invariant (the first genuinely "distributed" step).

## Invariants (the harness for meshing)
Per backend and rank count: one owner per cell; every needed DOF owned-or-ghost;
owner→ghost broadcast and ghost→owner reduce are adjoint (`⟨Gu,v⟩=⟨u,Gᵀv⟩`); labels
& geometry classification preserved through refine+redistribute; positive reference
Jacobians; residual/`Jv` invariant under rank count, partitioner, and ordering; restart
reproduces solution + state; **no production path gathers the fine mesh on one rank.**

## Why this is "next" for the new-algorithm track
It underwrites the headline scale (~10M DOF) and the scalability claim, it's fully
greenfield (the lab never built parallel refinement), and the `KernelMeshView`
contract is what lets every operator — bulk and contact — stay backend-agnostic. The
distributed *solve* it feeds is already proven; this closes the gap to the mesh.
