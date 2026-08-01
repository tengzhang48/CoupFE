# CoupFE pitfalls

This guide collects failure modes that are useful when extending the current
CoupFE scaffold. They are engineering cautions, not a claim that every possible
model or solver has been covered. Pair them with the checks in
[`testing.md`](testing.md) and the scope table in
[`../docs/capabilities.md`](../docs/capabilities.md).

## Complex-step differentiation

CoupFE can obtain tangents by complex-step differentiation. The residual must
therefore remain analytic in every perturbed variable. On that path, avoid:

- `abs`, `max`, `min`, clipping, or sign tests on a perturbed value;
- extracting the real part before completing the residual;
- changing a branch or active set in response to the imaginary perturbation;
- conjugating contractions such as `conjg(z) * z`; and
- eigenvector-based formulas at repeated or nearly repeated eigenvalues unless
  their behavior has been justified for the constitutive model.

When a nonsmooth choice is unavoidable, choose it at the real base point and
hold it fixed while differentiating the selected smooth branch. A
complex-step/finite-difference comparison is useful, but both methods can agree
on a consistently wrong residual. Add an independent mechanics or analytic
check whenever one is available.

For matrix functions such as `logm` and `sqrtm`, the iterative backend avoids
explicit eigenvector reconstruction and is usually the safer starting point for
complex-step work. It still has numerical tolerances and a finite accuracy
floor. If a finite-difference comparison fails, sweep the finite-difference
step before changing the acceptance tolerance.

## Generated-code constraints

The Python reference path supports more expressions than the current Fortran
translator. Generation, compilation, and one execution test are therefore
separate gates.

### Prefer explicit products for squares

In generated complex arithmetic, a generic power may be implemented through a
logarithm. For a quantity that can be exactly zero, write `z * z` instead of
`z**2`. Include zero and symmetric states in generated-kernel tests.

### Keep translated expressions simple

- Assign determinants and inverse transposes to temporaries before returning a
  tensor expression. This makes generated-source inspection easier and avoids
  ambiguity about scalar and tensor shapes.
- Use `0.5 * (A + A.T)` where the translator does not support `sym`.
- Keep translated helper functions type-consistent across their call sites.
- Remember that Fortran names are case-insensitive: `P` and `p` collide.
- Use translator-supported statement-form branches based on the real base
  value, rather than branching on a complex perturbation.

The supported operation set evolves. Treat a successful Python `verify()` as a
reference-path result, not proof that a generated kernel will compile.

### Fields and state variables

Generated material methods have a strict signature and state layout:

- follow the generator's documented field/state argument order;
- list every declared old-state argument, even if the method recomputes it;
- return updated state in the form accepted by the translator;
- avoid passing one helper both scalar and tensor versions of the same
  argument; and
- pack tensor state with the generator's column-major layout.

A symmetric test tensor cannot reveal a row-major/column-major mismatch. Use a
nonsymmetric tensor in at least one state round-trip test.

The present generator handles structural orientation most reliably when it is
represented through tensor quantities used by the constitutive equations.
Compile and exercise any new vector-state representation before relying on it.

### Native and Abaqus signs

The native kernel presents a weak-form residual `R` and tangent `dR/du`.
An Abaqus UEL uses the Abaqus `RHS`/`AMATRX` convention. `CompiledElement`
normalizes the supported generated backends to CoupFE's `(R, K)` convention;
code that reads or edits emitted Fortran must still distinguish the two.

For scalar equations, derive the weak form before mapping it to the generator's
storage-and-flux tuple. The identifier `flux` denotes the coefficient required
by that tuple; it need not match a paper's sign convention for physical flux.
Keep `OperatorSignWarning` enabled and add a nonuniform gradient test for each
new transport or phase-field formulation.

## Feasibility is separate from energy

An energy or residual does not by itself stop a trial step from leaving its
valid domain.

- Bulk finite-strain models generally require `det(F) > 0`. Formulations that
  also use an averaged Jacobian may require a bound on that quantity as well.
- Barrier contact requires a feasible positive starting gap and a step bound
  that prevents crossing the obstacle.
- A line search or step limiter must enforce the conditions relevant to the
  particular formulation; the current generic Newton driver does not infer all
  of them automatically.

If a residual becomes nonfinite, localize the problem by operator or element
group before changing loads, boundary conditions, or tolerances.

## Transactional state

Path-dependent operators require a clear accepted-state protocol:

- evaluate every residual and tangent from the same committed state;
- do not let one differentiation column update the state seen by another;
- do not commit line-search trials; and
- commit once, using a final real evaluation, after the driver accepts the
  global step.

The generic solver helpers do not yet provide a complete increment-level
transaction manager for every stateful generated element. This limitation is
called out in the capabilities and API documentation. Application drivers must
make their state policy explicit.

## Coupled systems

A single unscaled residual norm can hide an unconverged field when blocks use
very different physical units or characteristic magnitudes. Before relying on a
coupled transient:

- inspect residuals and tangent blocks by field;
- select problem-derived scales or nondimensional variables;
- use field-aware convergence criteria where necessary; and
- distinguish nonlinear convergence from linear-solver conditioning.

Cross-backend agreement is helpful only when the models, loading history,
discretization, and convergence criteria are aligned.

## Contact

Contact search and active-set selection are nonsmooth. Choose the candidate
feature, active branch, and stick/slip state at the real iterate; differentiate
only the selected smooth evaluation.

Other common checks are:

- state the gap and normal convention, then verify equal-and-opposite reactions;
- orient boundary facets consistently and test that the undeformed contact
  residual is zero when the bodies are separated;
- use a positive stiffness proxy where a penalty or barrier rule requires a
  stiffness scale;
- start barrier examples with a positive gap and use an appropriate collision
  or step bound;
- account for obstacle motion over the full step rather than only at the final
  time; and
- treat distributed contact search as a spatial communication problem, not as
  an automatic consequence of the bulk mesh halo.

Options such as normalized barrier shapes, elastic stiffness contributions,
kinematic gap floors, and obstacle thickness change the mathematical model.
Do not reuse parameters across those choices without checking their units and
meaning. See [`contact.md`](contact.md) for the currently shipped interfaces.

## Contact examples and comparisons

A successful nonlinear solve is not by itself a validation result. For a
contact comparison:

- refine or grade the mesh where the contact patch is expected;
- measure patch width from meaningful reactions rather than every nominally
  active node;
- reproduce the loading history used by a path-dependent reference solution;
- match material, interface, dimensional, and boundary assumptions; and
- retain the input, environment, raw output, and acceptance rule for any result
  described as validation.

Some CoupFE examples are deliberately labeled `RESEARCH`: they exercise useful
paths but do not yet carry that complete evidence package.

## Dynamics and moving obstacles

Adaptive dynamics with barrier contact combines several independent controls:
the nonlinear iteration limit, obstacle motion, collision bounds, admissible
gaps, mass and damping scales, and the load/ramp history. Pass those controls
explicitly through application drivers and test their effect separately.

Dynamic relaxation also needs a stated settling criterion. A visually slow
ramp or a completed time interval does not establish a quasistatic state; use a
problem-appropriate kinetic-energy or residual measure and include a hold when
needed.

## Joint residual/tangent kernels

A compiled element may compute residual and tangent in one kernel call, whereas
the public `Operator` contract exposes separate `residual()` and `tangent()`
methods. `ElementGroup` can reuse a paired evaluation through its one-entry
cache. The cache key and invalidation rules matter when properties or committed
state change.

Current native generators also emit a residual-only twin. Use explicit
``evaluation_mode="split"`` when residual-only callbacks (line-search trials,
acceptance checks, matrix-free loops) are common; keep ``"joint"`` when paired
residual/tangent requests dominate. Require the caller to choose explicitly;
do not switch evaluation policy merely because an R-only symbol is present.
Disable fusion only for algorithms that intentionally
mutate inputs between paired calls or while diagnosing cache behavior.

Residual-only and joint entries must start from the same committed state and
produce identical residual/trial state. Reevaluate a stateful element at the
accepted iterate before commit; the most recent callback may have been a
rejected line-search trial.

## Validation principle

Compilation, smoke tests, and derivative consistency cover different failure
classes. Add physical properties and independent oracles: patch tests, energy
balances, sign and definiteness checks, manufactured solutions, or a carefully
matched external solver. A regression is strongest when a deliberately broken
control demonstrates that the check detects the intended error.
