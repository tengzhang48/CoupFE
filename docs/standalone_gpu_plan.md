# CoupFE standalone runtime and acceleration direction

This document records the current public architecture and the conditions under
which further runtime acceleration would be justified. It replaces a long,
dated internal plan. Detailed design explorations, discarded alternatives, and
the original chronology remain available in Git history.

## Current position

CoupFE is a small finite-element core, not a general meshing or PDE platform.
It owns the correctness-critical contracts needed to define an element, compose
operators, solve a nonlinear system, and validate the result. Application
repositories own domain geometry, mesh-file adapters, named-boundary policy,
loads, workflow orchestration, and result presentation.

The supported execution model is CPU-first:

- Python coordinates model construction, operator composition, and solvers.
- NumPy and SciPy provide the compact serial reference path.
- PETSc provides sparse linear algebra and distributed-memory execution where
  the capability is explicitly implemented and tested.
- Generated Fortran supplies dense element-kernel work through a stable ABI.
- Optional acceleration is introduced only behind a tested reference path.

This architecture is intentionally modest. A clear, reviewable CPU
implementation is the oracle against which any optimized implementation must
be checked.

## Core boundaries

The durable core abstractions are:

- `Operator`: contributes residual, tangent, optional step bounds, and state
  commit behavior.
- `ElementGroup`: maps a compiled element over a homogeneous element set and
  scatters its local contributions.
- `CompiledElement`: drives batched native or Abaqus-compatible Fortran kernels.
- `KernelMeshView`: a neutral, zero-based array contract for coordinates,
  connectivity, sets, and limited geometry metadata.
- Solver functions: assemble and solve composed operators without embedding
  application-specific physics.

The core should not absorb an external mesh framework simply to reduce adapter
code. CAD, Gmsh, ventricular meshes, package geometry, periodic matching, and
similar domain semantics belong in the consuming application. An adapter ends
at `KernelMeshView`; repeated neutral functionality can be promoted only after
multiple applications demonstrate the same need.

## Build-time code generation

`coupfe.codegen` is an optional build-time facility. A material and weak form
can emit self-contained Fortran for two element backends:

- the CoupFE native element ABI; and
- the Abaqus UEL ABI.

The standalone runtime does not import a model-definition repository or SymPy
to evaluate an already generated element. Generated kernels are ordinary
runtime assets with explicit properties, dimensions, DOF layout, and state
size. Abaqus UMAT generation is also available, but CoupFE does not imply a
generic standalone UMAT host; a standalone structural solve needs a complete
element or operator.

Backend parity is an implementation check. It can expose ABI, sign, ordering,
or state-transfer drift, but it is not independent validation of the physical
formulation.

## Residual, tangent, and state

The residual is the primary statement of each smooth physical contribution.
Where complex-step differentiation is used, every operation on the perturbed
path must preserve the imaginary component. Non-smooth choices such as contact
search, active-set classification, and stick/slip selection are frozen from the
real iterate and differentiated only on the selected smooth branch.

The intended state protocol is transactional:

1. Every trial evaluation reads the same committed state.
2. Residual and tangent evaluation produce trial state without mutating the
   committed state.
3. Line-search and rejected iterations discard their trial state.
4. The solver commits exactly once after accepting a step.

This protocol is required for reproducible tangents and path-dependent models.
The compiled-element interface separates evaluation from commit, but the
generic drivers do not yet enforce accepted-only external-state commit in every
failure path, and `solve_increments` remains history-free. Distributed support
must also be claimed separately; a working serial state path does not establish
distributed state ownership or commit behavior.

## Contact

Contact remains an operator rather than an element formulation. Search,
closest-feature data, collision bounds, friction history, and active-set state
have lifecycles different from bulk quadrature state. Keeping this boundary
allows bulk kernels to stay simple while contact implementations evolve behind
the same residual/tangent/commit contract.

Each contact mode needs its own evidence. Penalty, barrier, smoothed friction,
return-map friction, and multiplier methods have different conditioning,
feasibility, and state semantics. A passing test for one mode cannot qualify
another.

## Distributed CPU execution

The serial operator interface accepts global vectors and is not itself a
distributed-memory abstraction. The useful distributed unit is the batched
element evaluator: each rank owns elements, ghosts the nodal values it needs,
and inserts local contributions into PETSc matrices and vectors with global
indices.

Distributed qualification requires more than successful execution. It should
cover ownership, ghost updates, off-process accumulation, global step bounds,
state ownership where applicable, and equality with the serial result within a
stated numerical tolerance. Performance claims additionally require retained
inputs, environment details, raw output, and converged equivalent solutions.

## GPU and matrix-free research

GPU and matrix-free execution are future research directions, not current
release promises. They should be pursued only when retained profiling on a
representative, redistributable problem shows that the existing CPU path is the
limiting factor.

Before starting an accelerated backend, require:

1. a reproducible CPU baseline with independent correctness evidence;
2. a measured bottleneck that the proposed backend can address;
3. a viable preconditioner for the target problem class;
4. a data-ownership design that avoids accidental host/device traffic;
5. CPU/accelerated parity tests, including state and failure paths; and
6. a CI environment capable of keeping the backend from silently rotting.

Matrix-free `Jv` is useful only with a practical preconditioner. For coupled,
contact-rich, or strongly ill-conditioned systems, finding that preconditioner
may be the primary research problem. Until it exists, an assembled tangent with
an appropriate direct or block-preconditioned solve is the honest baseline.

An accelerated implementation should reuse stable contracts rather than fork
the model semantics. The reference residual, state protocol, element manifest,
and validation cases must remain common across backends.

## Near-term priorities

The public engineering sequence is:

1. Keep the serial and PETSc CPU paths reproducible and claim-bounded.
2. Harden native/UEL generation, compilation, packaging, and ABI tests.
3. Expand stateful and coupled coverage only with independent oracles and
   transactional-state tests.
4. Keep mesh and geometry adapters in application repositories while improving
   the neutral handoff contract when demonstrated needs recur.
5. Qualify contact modes individually, including feasibility and state transfer.
6. Retain final-revision distributed evidence before making scale claims.
7. Revisit GPU or matrix-free work only when the evidence gates above are met.

## Development and review

AI coding agents can accelerate implementation, test generation, documentation,
and comparison of alternatives. They do not replace formulation review,
dimensional analysis, provenance checks, numerical validation, or domain-expert
judgment. Agent-produced changes are held to the same source-authority, broken-
control, independent-oracle, and reproducibility requirements as human-written
changes.

The objective is not to accumulate backends. It is to preserve one small,
understandable system whose supported paths are backed by inspectable evidence.
