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
- the native compiled-element ABI, loading, evaluation, and commit interfaces;
- exact affine-constraint reduction;
- compact array-level mesh contracts and regular-mesh utilities;
- serial and PETSc/MPI solve paths;
- contact operators and search primitives; and
- declaration verification and source generation for supported native elements
  and explicitly scoped external-solver exports.

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

## Native execution and external source export

A supported element declaration can generate Fortran for CoupFE's native
compiled-element ABI. This is the element path used by CoupFE's serial and
PETSc/MPI solves: the standalone runtime loads the kernel and calls CoupFE's
residual/tangent interfaces without Abaqus procedure flags.

Selected declarations can separately emit Abaqus/Standard UEL source for an
external Abaqus workflow. Abaqus owns and executes the full procedure context,
including its call sequence and data such as ``LFLAGS``. CoupFE's in-process UEL
adapter makes only a narrow normal-static joint call for focused
implementation-parity tests and selected research examples; it is not a
general Abaqus procedure host or a supported general CoupFE solve path.

The native ABI exposes the quantities CoupFE needs directly: a joint
residual/tangent entry and, where generated, an optional residual-only entry.
Comparing that native implementation with the corresponding Abaqus-interface
export can detect code-generation, sign, ordering, state-transfer, and ABI
drift. Because both share a formulation, the comparison is an implementation
check, not an independent physical oracle.

Supported material declarations can also emit Abaqus/Standard UMAT source as
an external export artifact. CoupFE does not host or call UMATs in its native
runtime.

Each model still needs evidence appropriate to its claims, such as an analytic
limit, a separately implemented invariant, a convergence study, or a properly
sourced reference result.

Source generation is an optional preparation step. The runtime consumes native
compiled-element kernels and does not require SymPy during a normal solve.
Abaqus UEL and UMAT files remain external export targets. The limited UEL
parity adapter described above is the sole in-process exception; UMAT has no
CoupFE runtime adapter.

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
- native/Abaqus-interface export parity; and
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
