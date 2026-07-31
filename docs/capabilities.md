# CoupFE capability status

This matrix describes the current public source. It distinguishes a shipped
implementation from a qualified use case and avoids inferring distributed
coverage from serial coverage.

Legend: **yes** = implemented with public tests or a clearly scoped public
example; **partial** = implementation exists with a stated limitation;
**no** = not implemented in the current public API.

> The serial public suite exercises the shipped source. MPI programs are
> included under `examples/mpi_smoke/`, but this release does not publish a
> retained final-revision multi-rank qualification record. No retained
> large-scale benchmark or release-grade scaling record is published.

## Solve and state

| Capability | Serial | Distributed (MPI) |
|---|---|---|
| Operator composition and sparse assembly | **yes** | **partial** — dedicated distributed drivers, not the generic serial `Operator` loop |
| Load-stepped Newton with line search | **yes** | **partial** — history-free, single-field driver |
| Implicit backward-Euler dynamics | **yes** | **partial** — node-local inertia plus the documented force/contact hooks |
| Path-dependent element state and accepted-step commit | **partial** — compiled-element evaluation and commit are separate, but generic `newton_solve` has no convergence flag before commit and `solve_increments` is history-free | **no** |
| Coupled/multifield forms | **partial** — public codegen examples and a FieldSplit helper; evidence is formulation-specific | **partial** — preconditioning utility exists, but the generic distributed driver is single-field |
| Exact affine reduction `U = Pq + U0` | **yes** for quasistatic `newton_solve` and `solve_increments` | **no** |
| Affine constraints in fixed/adaptive dynamics | **no** — rejected explicitly | **no** |
| Sparse linear-solver policy | **partial** — automatic selection covers direct backends; Krylov/FieldSplit paths are explicit opt-ins | **partial** — solver choices depend on the PETSc build and driver |

## Contact

| Capability | Serial | Distributed (MPI) |
|---|---|---|
| Rigid penalty contact | **yes**, 2-D and 3-D obstacles | **partial** — node-local rigid path in `solve_distributed` |
| Return-map Coulomb friction on rigid penalty contact | **yes** | **partial** — node-local rigid path |
| Rigid cubic barrier and collision step bound | **yes** | **no** |
| Deformable 2-D penalty contact | **yes** | **partial** — deformable specification in the quasistatic driver |
| Deformable 2-D cubic barrier | **yes**, including an all-primitive option | **partial** — dynamics path with surface replication |
| Deformable 3-D vertex-face and edge-edge barrier contact | **yes**; public 3-D block examples are tested | **partial** — cross-rank implementation exists, without a retained final-revision MPI record |
| Smoothed friction on deformable barrier contact | **yes**, 2-D and 3-D | **partial** — deformable dynamics path |
| Return-map or persistent friction on deformable contact | **partial** — opt-in modes; support differs between 2-D and 3-D paths | **no** |
| Semismooth exact-stick friction | **partial** — small-scale linear-bulk research solver with lagged normal data | **no** general distributed solver |
| Self-contact | **partial** — 2-D penalty and a focused 3-D barrier operator; no qualified public 3-D end-to-end workflow | **no** |
| Broad phase | **yes** — spatial hash in 2-D and optional numba LBVH/grid paths in 3-D | **partial** — replicated-surface strategy |
| GPU contact | **no** | **no** |

The contact implementations cover different mathematical models. A passing
penalty, barrier, return-map, or semismooth example does not qualify the other
families. Exact-stick studies are intentionally separate from the smoothed
barrier/dynamics path.

## Code generation and elements

| Capability | Status |
|---|---|
| Python declaration to Abaqus UEL | **yes** for the element/formulation combinations exposed by `coupfe.codegen` |
| Python declaration to native element kernel | **yes**, with narrower geometry coverage than the Abaqus UEL generator |
| Abaqus UMAT generation | **yes**; four public material-point/codegen examples cover Neo-Hookean, Ogden, small-strain J2, and standard-linear-solid viscoelasticity |
| Hosting an arbitrary compiled UMAT inside CoupFE | **no** |
| Deterministic regeneration and generated-source compilation | **yes** for the examples whose tests state that scope |
| Named state schema and tensor history | **yes** in the compiled-element/codegen path |
| F-bar generation | **partial** — implemented for the supported backends; no general no-locking or inversion-robustness claim |
| Element-local condensed pressure | **partial**, shipped as research examples rather than a default formulation |
| Mixed and coupled paper-form declarations | **partial**, with example-specific implementation checks rather than broad model validation |
| Native geometry families | **partial** — the native standalone ABI is primarily Quad4-oriented |
| Compiled UEL geometry families | **partial** — public examples exercise Quad4, Quad8/Quad8R, Tet4, Hex8, and selected mixed layouts; coverage differs by formulation |

Native/UEL parity, reference assembly, and tangent consistency establish
implementation properties. They do not independently validate a constitutive
model, benchmark interpretation, or parameter set.

## Mesh and interoperability

| Capability | Status |
|---|---|
| Neutral in-memory mesh bridge | **yes** — `KernelMeshView` carries one homogeneous fixed-width cell block, named node/element sets, and geometry classification |
| Structured Quad4 generation and uniform refinement | **yes** |
| Analytic re-embedding | **partial** — `Circle`, `Sphere`, and `Plane` are the supplied geometry backends |
| Positive-Jacobian check | **partial** — current check is for supported 2-D geometry, not a general 3-D quality metric |
| Deterministic owned/ghost partitioning | **yes** for regular meshes |
| General graph partitioning or adaptive refinement | **no** |
| General Gmsh, CAD, DMPlex, `meshio`, VTU, XDMF, Exodus, or CGNS adapter | **no** in core |
| General Abaqus `.inp` importer | **no**; examples contain only narrow, explicitly scoped readers/scaffolds |
| Mixed cell blocks and facet topology in `KernelMeshView` | **no** |

Applications own authoritative geometry, mesh-tool integration, physical
labels, periodic matching, and file-format semantics. Core consumes the compact
array contracts after that translation.

## Evidence and maturity limits

- Examples labeled **READY** are supported only for the role stated in
  [`examples/REFERENCES.md`](../examples/REFERENCES.md).
- Examples labeled **RESEARCH** demonstrate an implemented path but are not a
  general validation or production-support claim.
- External-solver parity is claimed only when the input, environment, result,
  and redistribution authority are documented. Several workflows therefore
  require user-supplied data and make no bundled comparison claim.
- There is no retained large-scale benchmark, hardware-qualified scaling
  record, GPU backend, or broad real-device validation suite.
- CoupFE is early-stage research software; external adoption has not yet been
  documented by this project.

High-level future priorities are in [`roadmap.md`](roadmap.md). They are not
current capabilities.
