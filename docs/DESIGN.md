# CoupFE design

## Scope

CoupFE is a compact finite-element scaffold for developing and testing custom
operators, elements, materials, and contact formulations. It is not a general
PDE platform or a complete engineering-analysis product.

Mature finite-element frameworks are broad for good reason: they provide
reusable meshing, boundary-condition, time-integration, I/O, solver, and
edge-case support. CoupFE chooses a narrower boundary. Core owns reusable
numerical contracts and algorithms; applications own model setup, domain
semantics, mesh-tool integration, and output policy.

AI coding agents can accelerate development of application code and adapters,
but that work is not trivial. Units, boundary conditions, state transitions,
loading protocols, and data mappings remain correctness-critical. Generated or
AI-assisted code requires domain review, independent evidence, and executable
tests just like any other contribution.

## Core boundary

Core includes:

- the `Operator` contract and compositional assembly;
- nonlinear increments and implicit dynamics;
- explicit compiled-element evaluation and commit interfaces;
- exact affine-constraint reduction;
- compact array-level mesh contracts and regular-mesh utilities;
- serial and PETSc/MPI solve paths;
- contact operators and search primitives; and
- build-time UEL/UMAT and native-kernel generation.

Core does not provide a general Gmsh, CAD, DMPlex, Abaqus-input, or results-file
adapter. Application packages translate their own geometry, labels, material
regions, boundary conditions, and file formats into the small core contracts.
Shared adapters can be factored into optional packages when repeated use
justifies their maintenance and dependency cost.

## Operator contract

Every physical contribution implements the same interface:

```text
residual(U, state, t, dt) -> Residual
tangent(U, state, t, dt)  -> Tangent
commit(U, state, t, dt)   -> new_state
```

Bulk element groups, contact sets, inertia, and loads compose through this
contract. The global driver does not need application-specific knowledge of
their physics.

For smooth residuals, complex-step differentiation can derive consistent
tangents without maintaining a second hand-written formulation. Discrete
choices such as active sets, search results, and branch decisions must be held
fixed or handled by an appropriate nonsmooth method during differentiation.

Stateful operators are designed around a transactional rule: trial residual
and tangent evaluations start from committed state and do not modify it, while
commit is a separate operation. The compiled-element interface follows that
separation. Generic driver orchestration does not yet enforce accepted-step
history in every failure and multi-increment path; the exact boundary is
documented in [`capabilities.md`](capabilities.md).

## Constraints

Exact affine constraints use a separate transformation,

```text
U = P q + U0,
```

so physical operators continue to assemble and commit in full coordinates.
Core compiles scalar relations and performs the algebraic reduction. An
application remains responsible for constructing meaningful relations from its
mesh and boundary semantics. The current qualified path is serial quasistatic
solve; fixed and adaptive dynamics reject affine constraints explicitly, and
the MPI drivers do not consume the transform.

## One formulation, two backends

A supported element declaration can generate Fortran for two execution
environments:

- an Abaqus UEL, called by Abaqus with Abaqus-owned procedure data such as
  ``LFLAGS``; and
- a native kernel loaded by the standalone CoupFE runtime, called through
  CoupFE's residual/tangent interfaces without Abaqus procedure flags.

These are parallel backends, not a call chain. The native ABI exposes the
quantities CoupFE needs directly: a joint residual/tangent entry and, where
generated, an optional residual-only entry. The in-process UEL wrapper is a
narrow normal-static compatibility path for focused implementation-parity
tests and selected research examples; it does not emulate general Abaqus
procedure sequencing.

Supported material declarations can also generate Abaqus UMAT source.

Using one formulation helps detect code-generation, sign, ordering,
state-transfer, and ABI drift across backends. Backend agreement is an
implementation check, not an independent physical oracle. Each model still
needs evidence appropriate to its claims, such as an analytic limit, a
separately implemented invariant, a convergence study, or a properly sourced
reference result.

Code generation is a build-time facility. The runtime consumes generated
kernels and does not require SymPy during a normal solve. A generated Abaqus
UMAT is an output target; CoupFE does not currently host arbitrary compiled
UMATs as its material runtime.

## Mesh and application integration

`KernelMeshView` is the neutral bridge into core. It carries zero-based NumPy
coordinates and connectivity plus named sets and geometry classification.
Regular Quad4 generation, uniform refinement, and deterministic owned/ghost
partitioning are provided. More general mesh topology, CAD association,
periodic node matching, mixed cell blocks, and vendor-format translation remain
application responsibilities.

This division is architectural, not a statement that integration work is
unimportant. Application adapters should be small where possible, but they
must make their assumptions explicit and be tested against the authoritative
geometry and model definition.

## Validation discipline

The public test suite combines several evidence types:

- analytic or independently implemented checks;
- discretization and convergence checks;
- finite-difference or complex-step consistency checks;
- generated-source compilation and deterministic regeneration;
- native/UEL implementation parity; and
- deliberately broken controls for selected failure modes.

These categories answer different questions and must not be conflated.
Consistency between two implementations that share a formulation does not by
itself validate that formulation. Example-specific evidence and provenance are
recorded in [`examples/REFERENCES.md`](../examples/REFERENCES.md), and the exact
support boundary is summarized in [`capabilities.md`](capabilities.md).

## Maturity

CoupFE is alpha research software. It has useful tested building blocks, but it
does not claim complete element coverage, general geometry support, broad
engineering validation, or production-scale performance. No retained
large-scale benchmark or release-grade scaling record is currently published.
