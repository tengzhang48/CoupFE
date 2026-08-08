---
name: coupfe-development
description: Engineering workflow for implementing, debugging, testing, documenting, and presenting CoupFE operators, materials, solvers, contact examples, and validation evidence. Use when changing CoupFE source, tests, benchmark claims, or solver-backed figures.
---

# CoupFE development skill

Read this before adding an operator, a material, or a solver feature. It ships
**with** the code so contributors and AI coding agents can reuse the project's
current engineering guidance. It reduces avoidable mistakes but does not replace
formulation review or model-specific verification. Pair it with
`skills/pitfalls.md`.

## How to add an element/material operator

1. **Write the residual first.** For a supported generated element, put the
   physical statement in one residual definition rather than duplicating it in
   a separately maintained stiffness routine.
2. **Derive or independently check the tangent.** Use complex step
   (`complex_step_tangent`) for analytic residuals. Analytic and semismooth
   tangents are also valid when their consistency and branch assumptions are
   tested explicitly.
3. **Declare state explicitly.** Return `state_trial`; never mutate committed
   state. Implement `commit` to recompute-and-return the accepted state.
4. **Make the differentiated path complex-analytic** (see `pitfalls.md`).
   Discrete choices need a separately stated treatment.
5. **Add a harness check, not just a smoke test.** Cross-check the complex-step
   tangent against an analytic reference (as `tests/test_operator_contract.py`
   does), run patch/energy tests, and use field-scale and sign checks appropriate
   to coupled equations. Consistency (CS-vs-FD) is **not** correctness; use an
   independent oracle for a physical claim.

## How to debug a coupled solver that "converges but is wrong"

The fingerprint and diagnostic workflow live in `pitfalls.md` ("coupled
convergence gate"). Split ‖R‖ by field, scale each block meaningfully, and check
whether error compounds over time. Treat this as a way to distinguish possible
convergence and scaling defects from formulation or boundary-condition defects,
not as a substitute for investigating either.

## Companion skills

- `skills/preflight.md` — use before a substantial new simulation: analyze
  dimensionless groups and timescales, then dry-run boundary conditions,
  loading, and material scales before drawing a method-limitation conclusion.
- `skills/pipeline.md` — the model-setup pipeline `Model` (a declarative front
  door suitable for human- or AI-assisted application setup): keep it a thin
  no-physics layer; extend via an operator/material, then expose convenience.
- `skills/pitfalls.md` — codified failure modes (complex-step safety, state protocol, the
  coupled convergence gate, contact gotchas).
- `skills/testing.md` — how to write tests that catch bugs (independent oracle + broken
  control; run it, don't loosen tolerances).
- `skills/distributed.md` — distributed/MPI development (the 1-vs-N invariant, memory-local
  generation, petsc4py mechanics).
- `skills/performance.md` — the acceleration ladder (vectorize → numba → f2py → Rust/C++),
  measure before escalating, the contract is acceleration-agnostic.
- `skills/contact.md` — contact + dynamics (freeze the discrete / complex-step the smooth;
  cubic barrier + CCD; capacity-vs-conditioning; dynamics as the substrate + dynamic relaxation).
  For an analytic contact benchmark, also read `skills/testing.md`; derive the
  material-parameter mapping from the implemented tangent before tuning contact
  or solver parameters.

The math behind it all is in `docs/theory/` (`contact_dynamics.md`, `framework.md`).

## How to work on the dual-multiplier contact / friction line

1. **Match the existing public interface.** `contact_semismooth.py` is the
   reference: factor the bulk stiffness once, condense to the contact-tangential
   interface, return a `FrictionResult` with `(U, p, stick, N, iters, residual,
   converged)`. A new solver should be a drop-in replacement.
2. **Keep alternative research solvers outside core.** Validate them on a
   clearly labeled research branch or repository; promote only a reusable,
   maintainable method with an explicit scope decision.
3. **Validate with a sweep, not a spot check.** See `skills/testing.md`. Gate
   physical invariants, load/friction sweeps, and any adjoint against finite
   differences.

## What stays application-specific

Core may provide generic solvers, time-integration algorithms, affine-constraint
algebra, and compact mesh operations. Domain geometry acquisition, mesh-format
translation, selection of physical boundary conditions, model-specific loading
schedules, and result presentation remain application responsibilities. Keep
those policies in the application or example layer (with AI assistance where
useful) and test their assumptions there.

The canonical home for simple model setup is **`Model`** (`coupfe/model.py`) — the declarative
front door (`Model.structured(...).material(...).fix(...).prescribe(...).contact(...).solve()`).
It is a *thin layer over the operator contract* (collect operators + build the Dirichlet
dict + drive `solve_increments`); it contains **no physics**. Add a new physics capability
as an operator/material; expose its problem-setup convenience on `Model`, never the reverse.
