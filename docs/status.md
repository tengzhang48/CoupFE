# CoupFE status

CoupFE is active alpha research software. The current public tree contains a
working operator-based finite-element core, build-time code generation, a
compiled-element runtime, serial and PETSc/MPI solve paths, contact building
blocks, regular-mesh utilities, and a curated example and test suite.

## Available now

- composable residual, tangent, and commit operators;
- load-stepped Newton solves and implicit backward-Euler dynamics;
- separate evaluation and commit interfaces for compiled element state;
- serial quasistatic affine-constraint reduction;
- native and Abaqus UEL element generation, plus Abaqus UMAT generation;
- Quad4-oriented native mesh/refinement utilities and compact mesh views;
- deterministic regular-mesh partitioning and MPI assembly/solve entry points;
- two- and three-dimensional contact, search, friction, and focused exact-stick
  studies; and
- examples with explicit evidence and provenance categories.

The precise serial, distributed, contact, code-generation, and mesh boundary is
maintained in [`capabilities.md`](capabilities.md). Public calls are listed in
[`api.md`](api.md).

## Qualification boundary

The shipped serial test suite exercises the public source tree. Some MPI paths
require an external PETSc/MPI environment and are provided as rerunnable smoke
programs; this release does not publish a retained final-revision multi-rank
qualification record.

Several examples are intentionally labeled **RESEARCH**. They demonstrate an
implemented path without claiming broad physical validation, production
support, or an independently reproduced external result. Workflows that depend
on licensed or user-supplied data state that requirement locally.

There is no retained large-scale benchmark or release-grade scaling record.
Historical timings and development observations are not part of the public
evidence boundary.

## Current limitations

- no general CAD, Gmsh, DMPlex, Abaqus-input, or results-file adapter in core;
- no distributed affine-constraint transform or distributed path-dependent
  element-state commit;
- the generic `newton_solve` return value has no convergence flag before the
  driver calls operator commit, and `solve_increments` is history-free;
- distributed coupled-field and contact coverage is narrower than serial
  coverage;
- native kernel geometry support is narrower than the compiled UEL path;
- no GPU backend; and
- no claim of broad real-device or production engineering validation.

See [`roadmap.md`](roadmap.md) for high-level priorities and
[`../examples/REFERENCES.md`](../examples/REFERENCES.md) before treating an
example as validation evidence.
