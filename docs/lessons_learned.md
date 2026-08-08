# CoupFE engineering lessons

This document collects durable engineering lessons from CoupFE development. It
replaces a dated internal diary of experiments and superseded plans. The detailed
chronology remains in Git history; this version keeps guidance that should survive
changes in examples, hardware, and implementation details.

## Keep the core small

A small stable contract is more valuable than a broad framework with ambiguous
ownership. Core owns operator composition, element execution, nonlinear solution,
neutral mesh arrays, and correctness gates. Applications own geometry acquisition,
mesh formats, labels, loads, domain policy, and reporting.

Do not add a dependency to core merely because an application already uses it.
Translate data into `KernelMeshView` and keep the adapter beside the application.
Promote code only when multiple consumers demonstrate a neutral abstraction.

## Make the residual the source of truth

For smooth physics, define the residual first and derive or verify the tangent
against it. A separately maintained residual and stiffness inevitably drift.
Complex-step differentiation is powerful only when the entire perturbed path is
complex-safe; a cast, comparison, absolute value, or unsupported helper can
silently erase the derivative.

## Treat state transactionally

Every nonlinear trial should start from committed state. An evaluation may
return trial state, but it must not mutate the committed copy. Rejected Newton
steps, line-search probes, and individual complex-step columns must not affect
the next evaluation. A driver for path-dependent models must commit once, after
the global step is accepted. The current generic drivers do not yet enforce
that rule in every nonconvergence and multi-increment path; see
[`capabilities.md`](capabilities.md).

Test state layouts with values that expose ordering errors. Symmetric tensors can
hide row-major/column-major mistakes; include a non-symmetric tensor state when
the ABI supports tensor history. Check both the committed state and a downstream
quantity such as stress, because agreement in one does not prove the other path.

Serial state support does not automatically extend to MPI. Distributed state
needs explicit ownership, ghosting, commit, restart, and rank-invariance tests.

## Separate residual-only work from tangent work deliberately

A generated tangent can dominate element cost because it evaluates multiple
complex-step directions and assembles a dense local block. Caching a joint R/K
result is still the right choice when a solver requests residual and tangent at
the same iterate. It is wasteful for rejected line-search trials, independent
acceptance checks, and exact-equilibrium steps that request only a residual.

Native generation therefore keeps two entries from the same residual source:
the established joint R/K entry and a residual-only twin with tangent work
removed. Callers select ``joint`` or ``split`` explicitly; ``joint`` remains
the default. Do not infer a universal speedup: retain call counts and
representative timings, and test residual plus trial-state parity. Before
committing a path-dependent state, reevaluate at the accepted iterate so the
last rejected trial cannot be promoted.

## Separate consistency from validation

Useful consistency checks include:

- complex-step tangent versus finite differences;
- generated kernel versus a reference assembly;
- native versus Abaqus-backend parity; and
- serial versus distributed equality.

These catch implementation drift, but shared equations can be wrong in the same
way on both sides. A physics claim needs an independent oracle: an analytic
solution, patch test, manufactured solution, energy/work balance, independently
implemented invariant, or well-provenanced external reference.

Every important gate should have a broken control. Deliberately reintroduce a
representative defect and confirm that the test fails. A test that has never
failed may not exercise the behavior its name suggests.

Run the code. Static review cannot reveal compiler, ABI, nonlinear-basin, state-
transfer, or environment failures. Start with focused element and operator tests,
then execute the end-to-end path associated with the claim.

Review findings need the same evidence discipline. Reproduce the alleged
failure from the equations and representative numbers, then classify whether it
belongs to the solver, an example, an extractor, or only a misleading comment.
Do not preserve a false positive merely because a review labeled it major: an
independent calculation can confirm the behavior while rejecting the proposed
explanation or severity.

Test the invariant named by the claim, not a convenient surrogate:

- a plastic return map must finish on its declared yield surface, not merely
  have a tangent consistent with its own update;
- a public bulk modulus must recover that bulk modulus in the infinitesimal
  tangent, even if the raw kernel ABI accepts the first Lame coefficient;
- a penetration claim needs an oriented signed gap or collision history, not an
  unsigned closest-point distance; and
- a global Coulomb cap must be reconstructed from assembled reactions, not
  returned as the expected cap by the branch being tested.

When a defect crosses an API boundary, inventory callers before changing it.
Convert once at a named boundary and document the raw ABI; silent reinterpretation
can fix one example while breaking external callers or a previously correct
benchmark.

## Establish evidence before tuning

Begin with units, signs, boundary conditions, and dimensionless groups. Confirm
that the requested state is physically feasible before changing solver options.
A nonlinear failure caused by element inversion, impossible loading, or an
unresolved weak field is not repaired by a different linear solver.

Use load incrementation and an appropriate line search for finite-strain solves.
Inspect constrained modes when the tangent is singular. For coupled systems,
evaluate convergence per field on meaningful scales; a small unscaled residual
in a weak equation can still correspond to a large solution error.

Dynamic relaxation can approximate a quasistatic state only when loading is
ramped, then held until inertia and damping contributions are negligible relative
to the stated energy or force scale. A transient snapshot is not a quasistatic
oracle merely because displacement appears stable.

Convergence studies must test refinement trends, not tune a band around one mesh.

## Match an analytic benchmark at the constitutive tangent

An analytic oracle and a finite-element model must use the same effective
parameters, not merely variables with familiar names. For the retained Hertz
example, the compiled law is
`P = G(F - F^-T) + lambda ln(J) F^-T`; the coefficient of `ln(J)` has the
infinitesimal role of the first Lamé coefficient `lambda`. Supplying the physical
bulk modulus there silently changes both the effective Young's modulus and
Poisson ratio seen by the contact calculation.

Derive the small-strain tangent of a finite-strain law before translating
`E` and `nu`, and preserve that conversion beside the call site. A symbol such
as `K` in an implementation is not evidence that it means physical bulk
modulus. Include a structural broken control that substitutes the plausible
wrong convention and verify that the benchmark rejects it.

After matching the material, separate the remaining errors. Check domain size,
mesh resolution across the expected contact patch, penalty sensitivity,
nonlinear residual, and global force balance independently. A boolean active
set is often a poor footprint extractor: one marginal outer node or ring can
set the reported radius while carrying negligible reaction. Define extractors
from the quantity that supports the claim and study their refinement behavior.

Apply the same discipline to figures. Plot solved fields and reactions directly,
state the deformation scale, and label discrete nodal forces as nodal forces.
Do not relabel them as pressure or stress unless a documented reconstruction
and an appropriate convergence check establish that quantity.

## Keep code generation at build time

`coupfe.codegen` defines materials and weak forms and emits self-contained
Fortran. The runtime loads the generated native or Abaqus-compatible element and
does not need symbolic tooling or a model-source checkout.

The compiled ABI uses explicit dimensions, DOF layout, properties, and state
size. Do not recover runtime metadata by introspecting build-time objects. Test
the sign convention, node/field ordering, Gauss-point state layout, and both
single-element and batched calls.

A UEL is a complete element. A UMAT is a material-point routine and needs an
element host to supply kinematics, quadrature, and assembly. Generating a UMAT
does not imply that CoupFE can run it as a standalone structural model.

## Keep contact outside the bulk element

Contact has search, active-set, collision, and history lifecycles that differ
from bulk quadrature. Model it as an operator so it can compose with the same
solver without complicating every element ABI.

Freeze non-smooth decisions from the real iterate before differentiating a
contact branch. Do not complex-step through closest-feature search, min/max
classification, or stick/slip switches.

Collision bounds must cover every motion applied before residual evaluation,
including predictors and moving obstacles, not only the final Newton increment.
Feasibility of the bulk map and feasibility of contact gaps are equally important.

For surface contact, test orientation, action/reaction balance, incident-feature
exclusion, and surface-aware penetration. Verify search and narrow-phase
primitives independently before embedding them in a difficult nonlinear solve.

Friction models are not interchangeable. State clearly whether a path uses a
smooth rate regularization, a return map, or a multiplier method; document what
is committed across steps and how state is transferred when pairing changes.

## Design distribution around local work

The global serial operator interface is not the unit to distribute. The natural
unit is a batched element or contact contribution with explicit global indices.
Each rank owns local work, ghosts required vector entries, and lets PETSc combine
off-process contributions.

Test new distributed primitives in isolation before a full solve. Verify owned
and ghost data, reduction semantics, global collision bounds, and equality with
the serial result. A rank-equality check is evidence of distribution correctness,
not independent validation of the underlying physics.

Do not describe a capability as distributed merely because one neighboring path
uses MPI. State, coupled fields, constraints, contact modes, and dynamics each
need their own distributed qualification.

## Optimize from retained profiles

Choose an implementation technique that matches the computation:

- vectorized NumPy for uniform dense array operations;
- compiled loops for branch-heavy per-feature geometry;
- Fortran for dense element kernels that also need a stable external ABI; and
- PETSc for sparse and distributed linear algebra.

Profile first and retain the input, environment, raw output, and convergence
evidence. Do not publish speedups from unmatched or non-converged runs. Reuse
factorizations when the matrix is unchanged, and reject silent iterative-solver
failure.

GPU and matrix-free work should begin only after profiling shows a relevant
bottleneck and the target problem has a credible preconditioning strategy.
Backend novelty is not evidence of usefulness or correctness.

## Use AI assistance with engineering controls

AI agents help with bounded implementation, search, tests, and documentation.
They do not replace mechanics expertise, formulation review, dimensional analysis,
license/provenance review, or independent validation.

Keep tasks scoped, review diffs, preserve unrelated work, and require the same
tests and evidence for agent-authored code as for any other contribution. An
agent’s confidence is not a release criterion.

## Document claims at their evidence boundary

Mark historical plans as historical. Distinguish implementation checks,
research demonstrations, and supported validation. Link claims to retained,
rerunnable evidence and record known omissions beside the claim.

Before release, ask:

1. Is the source and redistribution authority clear?
2. Does the claimed path run from a clean installation?
3. Is there an independent oracle and a broken control?
4. Are state, signs, layouts, and failure paths tested?
5. Are distributed or performance claims backed by retained equivalent runs?
6. Are application-specific adapters still outside the core?
7. Could a reader distinguish current support from future research?

If any answer is no, narrow the claim or keep the work in development history.
