# CoupFE testing skill — how to write tests that actually catch bugs

Read this before adding a test. CoupFE's value is trust, and tests are the trust.
The default instinct — "compile it, run it, the number looks plausible" — is
exactly what lets a wrong element ship. These rules are how a test earns trust.

## The two non-negotiables

1. **Test against an INDEPENDENT oracle, not for self-consistency.** A
   complex-step-vs-finite-difference check, a "the tangent is the derivative of
   the residual" check, a smoke run — these are *consistency*. They pass on a
   wrong-but-consistent residual (a flipped sign, a wrong convention, a bad
   material constant). You must compare against something derived **independently**
   of the code under test:
   - a closed-form/analytic solution (e.g. the bar's `f·L/E`, the block's analytic
     traction-free lateral stretch);
   - a **patch test** (an affine field is the exact discrete solution of a
     constant-stress problem — reproduced exactly or the element is wrong);
   - an energy/work balance;
   - a manufactured solution;
   - an **operator gate** (block-definiteness, diffusive-flux sign, the
     coupled-scale balance) — a *physical property*, checked from the tangent alone.
   Cross-backend equality of code generated from the same residual is valuable
   ABI/implementation parity, but it is not an independent physics oracle and
   must be paired with one of the checks above.
2. **Every test ships a BROKEN CONTROL.** A test is only trusted once you have
   watched it FAIL on a reintroduced bug. Add the bug (a non-affine BC, a
   transposed tangent, a flipped sign) inline and assert the test rejects it. A
   green test with no broken control proves nothing — it may be asserting `True`.

## Two checks that belong on almost every element/operator

- **`complex-step tangent == analytic tangent`** (or `== FD of the assembled
  residual`). This is the CoupFE core invariant; if a change makes the residual
  non-analytic, this catches it. (Consistency-only, so pair it with an oracle.)
- **An independent physical result** (a solution field, a reaction, a stretch)
  vs a closed form.

See `tests/test_operator_contract.py` (the bar) and `tests/test_element_group.py`
(the compiled neo-Hookean) for the pattern: patch test + tangent-vs-FD + analytic
physical oracle, each with a broken control.

**A stored-tensor element's `kernel == reference_assembly` gate must inject a
NON-SYMMETRIC `Fp_old`** (and assert it really is non-symmetric). Tensor STATEV
is column-major; a symmetric state (`epsp`, `Cp`) is order-invariant, so a
symmetric-only gate passes even when the layout is transposed — a *vacuous* test
of the round-trip. The path is only exercised by a **tensor-typed, non-symmetric**
state: storing a non-symmetric quantity as *scalars* (as `lce_quad4` does with
`Fv11..Fv21`) skips the reshape and won't catch it either. The 2026-06-25
`reference_assembly` transpose bug survived every prior element because none had a
tensor-typed *and* non-symmetric state (`epsp` is symmetric; `Fv` is scalar-packed). The
discriminating signal: **committed state matches the oracle but the stress does
not** ⇒ the bug is in a stress-only path (here `polar(Fe)` / the STATEV read),
not the return map. A private finite-strain-J2 regression exercises this axis,
but that example/test is not in the first public artifact pending provenance.

## Validation ladder summary

Climb the ladder in order; do not use a solver run to debug a Python equation
error. The full ladder is in `skills/model_development.md`:

1. Python reference / material-point checks.
2. `problem.verify()` (tangent consistency).
3. Generated Fortran compile.
4. f2py single-element check (`CompiledElement.element_rk` vs `reference_assembly`).
5. Solver run (`newton_solve` / `solve_increments`).
6. Abaqus-UEL export validation (when Abaqus is available).

A model is not validated just because step 3 passes. Every rung needs either an
independent oracle or a deliberately broken control.

### Verifying coupled / multi-field elements (and the harness direction)

`reference_assembly` is an *independent* re-assembler, but historically it
hand-enumerated assembly cases (`row_asm × col_asm × field-kind × wrt_kind`), so a
**new weak-form term shape** could hit an unfilled cell. Two consequences for how you
verify a coupled element:

- It now handles a **value-residual depending on a scalar field's gradient** (e.g. a
  source/storage term in one equation that depends on `grad_phi`/`grad_c` of another —
  the case the Li 5-field element needed). If you add an element with a coupling the
  table doesn't cover, it **raises `NotImplementedError("unhandled … pattern … use
  B-matrix assembly")`** rather than silently dropping a block (a wrong K). Treat that
  raise as "the oracle needs a branch," not "your element is broken."
- The **trigger is a term *shape*, not a field count** — "doesn't work at N fields" is
  almost always shorthand for "has a coupling pattern the oracle never met."

**Direction (do this for any new/post-processed element now):** add a **generic FD/CS
tangent-consistency check `K ≈ ∂R/∂U`** (difference the whole element residual w.r.t.
each DOF). It needs zero coupling knowledge, works for any number of fields, and is the
*only* tangent gate that catches **hand-written tangents** (e.g. the axisymmetric gel's
post-processed hoop blocks, which the complex-step auto-tangent does **not** cover).
Pair it with an R check (patch test / MMS / residual-only reference) since FD-consistency
checks K against R, not R itself. See `docs/dev/verification_harness_redesign.md` — the
plan is to retire the tangent case-table in favour of this generic check + property gates.

## Solver tests: sweep parameters, don't spot-check

A dual-multiplier / LCP / active-set friction solver can pass at `delta = 0.02`
and `delta = 0.8` yet fail at `delta = 0.735` because the active set is near a
transition and the iteration basin changes. For any new solver:

- **Sweep the control parameter** (load level, displacement, friction
  coefficient) and compare every point to an independent oracle (the semismooth
  engine, an analytic stick/slip onset, or a closed-form limit).
- **Include a broken control** that disables the safeguard: e.g. run the same
  sweep with no damping / high damping / warm-start reuse and assert it either
  diverges or lands on the wrong basin somewhere. The test is only trusted when
  the safeguarded version passes and the broken version fails.
- **Check the active-set diagnostics**, not just the displacement. A wrong
  solution can still satisfy `|p| = mu N` on slip nodes and `v_t = 0` on stick
  nodes; the discriminating signal is agreement of the stick/slip partition with
  the oracle.

## Run it — never trust a static review, and don't loosen the tolerance

A statically-reviewed-but-unexecuted port *looks* correct and silently isn't.
(Real example in this repo's history: a migrated element passed the patch test and
the tangent invariant but its uniaxial test hit a singular matrix at runtime — a
Newton-basin/element-inversion issue no read-through could surface.) Always
execute, and when a test fails, **diagnose, do not just loosen the tolerance**:

- Singular/NaN tangent? Find the null mode (`eigvalsh` of the constrained block):
  a smooth translation/rotation = an unconstrained **rigid-body mode** (fix the
  BCs); a checkerboard = an **hourglass/spurious** element mode; a *negative*
  eigenvalue that appears only mid-solve = the step overshot into **element
  inversion** (load-step + line-search, don't blame the element).
- Trace the residual **per Newton iteration** to see where it blows up.
- Finite-strain problems need **load incrementation** (`solve_increments`) and a
  **line search** — a full step applied from a zero interior overshoots, and finer
  meshes make it worse (smaller boundary element → larger local strain).

## The validation registry — test broadly, release narrowly

`validation/` is the internal confidence layer: register many lab models
(`@register` → build a problem, solve through `coupfe`, compare to an independent
oracle, return a `ValidationResult`), run them as parametrized pytest cases. This
is separate from the curated public `examples/`. We validate against as many
models as possible; we ship only a clean few. When you port a lab model, add it
here with its oracle and a note on what it exercises (state, coupling, staggering).

## Tolerances

Set tolerances from the *expected* discretization/round-off error, not from
"what makes it pass." A tangent-vs-FD check is FD-limited (~`1e-5·scale`); an
analytic-solution check on a converged solve is round-off-limited; a coarse-mesh
physical comparison carries discretization error — state which, and why the number
is what it is.

## Convergence gates: assert the TREND, don't band a wandering result

A coarse-mesh physical comparison must assert the result **converges** to the oracle
under refinement (the error *shrinks*), not that each mesh lands inside a loose band. A
band wide enough to pass a coarse mesh also passes a *non-converging* solve, and the gate
then certifies nothing. (A retired research Hertz benchmark "passed" a ±15% band while
the load drifted **+18%** with refinement and the half-width crept *up* — the solution
wasn't converging; the loose gate hid it. Root cause: a starved contact patch on a uniform
mesh.) The discipline:

- **Print the QoIs across the refinement and look at the trend *before* writing the assert.**
  A passing gate tells you nothing until you've seen the ratios actually converge.
- **Gate the trend:** `err(finest) < err(coarsest)` (and small), or `|QoI(2h)−QoI(h)|`
  shrinking. Refine until the QoI is mesh-stable, *then* set a tight tolerance.
- **A tolerance reverse-engineered to pass is a smell** — a ±75% "gate" is not a gate.
- **Extract from the right quantity.** A naive extractor (`max|x|` for a contact half-width)
  carries a bias that *grows* with refinement; fit the physical profile (Hertz pressure is a
  semi-ellipse → `p²` linear in `x²`) and gate the fit's **R²** too — that validates the
  *shape*, not just two scalars.
- **Reject `converged=False`.** A gate that checks only a residual band accepts a *non-converged*
  iterate and reports its tuned numbers as passing. Assert the solver converged, not
  just that some residual is small.
- **Test a mesh FINER than the one you tuned to.** A result tuned to look right at `nx≤240` blew up at
  `nx=360`. Make the *stability* check span past your tuning resolution (e.g. `|ratio(h)−ratio(2h)|`
  small at the **two finest**), so a coarser-stop tune can't sneak through.
- **When a converged QoI is *stable but wrong*, suspect the model/extraction, not the solver — and keep
  digging past the first plausible cause.** Use controlled parameter sweeps to distinguish
  a numerical defect from a mismatch between the FE setup and the analytic oracle's
  assumptions. A stable-wrong number is progress, but verify the proposed cause.

## Validation honesty — separate the claim from the match
What you *claim* a test proves matters as much as that it passes. Five refinements:

- **Prefer a STRUCTURAL-zero broken control over a scaling one.** A control like "halve the rate ⇒
  halve the decay" can itself break under a legitimate discretization bias (a backward-Euler rate bias
  failed it at 2%), giving a false alarm. `k=0 ⇒ no decay` is **exact** regardless of step — pick the
  control the discretization can't perturb.
- **Separate the *rigorous* claim from the *benchmark* claim.** "Self-consistent / matches a closed
  form to machine precision" and "matches a published figure within a few %" are different statements —
  never call the second one *exact*. State which you're making.
- **Cross-check a published parameter set across ≥2 independent sources, and recompute it yourself,
  before baking it in.** Single-source transcription is a silent error mode (a dropped zero in a
  material constant survives every self-consistent test). The parameters are an input to verify, not a
  given.
- **For a machine-precision gate, pick an oracle the element represents EXACTLY.** A linear field
  `u=(λ−1)X` is reproduced by a trilinear Hex8 to ~2e-13 — a real correctness gate; a curved/analytic
  oracle is discretization-limited (a few %) and only ever a *benchmark-within-discretization*. Don't
  conflate the two.
- **Report a genuinely singular QoI at a FIXED mesh, with the singularity named.** A re-entrant-corner
  stress (or any singular field) *grows* with refinement — it has no mesh-converged value. Pin the
  mesh, name the singularity, and report the non-singular limit (a bulk-RMS, a binned average) as the
  convergent quantity instead.
- **Same mesh is not cross-engine parity.** The 3D Cattaneo/Abaqus comparison used the exact uploaded
  mesh and still was not a validation gate: Abaqus `C3D8R` + hard/augmented contact is not CoupFE
  F-bar/full-integration Hex8 + finite nodal penalty. For an Abaqus comparison, list and match the
  element formulation, integration/hourglass control, contact enforcement, load history, and
  post-processing extractor before claiming a solver error or a pass.
