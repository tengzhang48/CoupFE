# CoupFE development skill

Read this before adding an operator, a material, or a solver feature. It ships
**with** the code on purpose: the pitfalls below are hard-won, and codifying them
is what lets a custom operator be built correctly the first time. Pair it with
`skills/pitfalls.md`.

## How to add an element/material operator

1. **Write the residual — it is the only source of truth.** Put the entire
   physics (including the material/constitutive response) in one pure function
   `residual(U, state, t, dt)`. No physics anywhere else.
2. **Do not write a tangent.** Derive it from the residual by complex step
   (`complex_step_tangent`). A hand-coded stiffness is a second source of truth
   that will drift from the residual.
3. **Declare state explicitly.** Return `state_trial`; never mutate committed
   state. Implement `commit` to recompute-and-return the accepted state.
4. **Make the residual complex-analytic** (see `pitfalls.md`). This is the one
   real constraint complex step imposes.
5. **Add a harness check, not just a smoke test.** Cross-check the complex-step
   tangent against an analytic reference (as `tests/test_operator_contract.py`
   does), run patch/energy tests, and — for coupled fields — the operator gates
   (block-definiteness, the coupled-scale balance). Consistency (CS-vs-FD) is
   **not** correctness; you need an independent oracle.

## How to debug a coupled solver that "converges but is wrong"

This is common and almost never the physics. The fingerprint and the fix live in
`pitfalls.md` ("coupled convergence gate"). Short version: split ‖R‖ by field,
divide each by its block scale to get the *solution* error, check whether the
error compounds over time (→ it's the convergence gate, not the model), and gate
each field on its own scale (field-wise convergence). Count Newton iters/step:
3–5 means the tangent is fine and the gate is the bug.

## Companion skills (read the relevant one before working in that area)
- `skills/preflight.md` — **READ FIRST for any new simulation**: dimensionless analysis
  (timescales via `eigsh(K, M)`, derive ramp/damping/dt/penalty from the groups) + a dry
  run that tests the BCs, loading protocol, and material scales BEFORE the production run;
  no "method limitation" conclusion without the pre-flight artifacts.
- `skills/pipeline.md` — the model-setup pipeline `Model` (the declarative front door / AI-glue
  target): keep it a thin no-physics layer; extend via operator/material then expose sugar.
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

The math behind it all is in `docs/theory/` (`contact_dynamics.md`, `framework.md`).

## How to work on the dual-multiplier contact / friction line

1. **Match the existing public interface.** `contact_semismooth.py` is the
   reference: factor the bulk stiffness once, condense to the contact-tangential
   interface, return a `FrictionResult` with `(U, p, stick, N, iters, residual,
   converged)`. A new solver should be a drop-in replacement.
2. **Keep alternative research solvers outside core.** Validate them on a
   dedicated private research line; promote only a reusable, mainstream method
   with an explicit product decision.
3. **Validate with a sweep, not a spot check.** See `skills/testing.md`. Gate
   physical invariants, load/friction sweeps, and any adjoint against finite
   differences.

## What NOT to add to the core

Meshing, BC application, loading schedules, time integration, output — these are
per-problem glue. Write them in the example/driver layer (AI-assisted is fine),
and let the harness validate them. Keeping them out of the core is the point.

The canonical home for that glue is **`Model`** (`coupfe/model.py`) — the declarative
front door (`Model.structured(...).material(...).fix(...).prescribe(...).contact(...).solve()`).
It is a *thin layer over the operator contract* (collect operators + build the Dirichlet
dict + drive `solve_increments`); it contains **no physics**. Add a new physics capability
as an operator/material; expose its problem-setup convenience on `Model`, never the reverse.
