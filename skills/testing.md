# CoupFE testing guide

Use this guide when adding a test or changing the evidence claimed by an
example. The public policy is [`../validation/README.md`](../validation/README.md),
and example-specific claims are recorded in
[`../examples/REFERENCES.md`](../examples/REFERENCES.md).

## Distinguish the evidence type

Different checks answer different questions:

- **Consistency:** complex-step versus finite difference, residual versus
  tangent, generated kernel versus reference assembly, or native versus UEL
  output from the same declaration.
- **Independent oracle:** analytic solution, manufactured solution, patch test,
  energy/work balance, independently derived invariant, or authorized external
  result with a reproducible extractor.
- **Convergence:** error or a quantity of interest follows the expected trend
  under mesh, time-step, or nonlinear-tolerance refinement.
- **Integration:** generated source compiles, an installed artifact imports, or
  a documented solver path executes.
- **Broken control:** a deliberately incorrect sign, layout, state transition,
  or boundary condition is rejected by the proposed gate.

Consistency is important but does not prove the equations are physically
correct. A physical-validation claim needs independent evidence. A build-time
example may stop at consistency and compilation if it is labeled as a codegen
proof rather than validation.

## Development ladder

Use the smallest layer that can expose the suspected defect:

1. Evaluate the Python material or residual on characteristic states.
2. Run declaration-level derivative checks.
3. Generate and compile the Fortran source.
4. Compare one compiled element with reference assembly.
5. Exercise a small boundary-value problem through a public driver.
6. Compare another backend only after matching conventions and inputs.

Do not debug a solver trajectory before the material-point and element layers
are understood. Conversely, static review and compilation cannot expose ABI,
state-transfer, nonlinear-basin, or environment defects; execute the path that
supports the claim.

## Tangent checks

For an analytic residual, compare the local or assembled tangent with a finite
difference or an independently derived tangent. State the finite-difference
step study and tolerance; one arbitrary perturbation size can hide truncation
or roundoff error.

A post-processed, analytic, or semismooth tangent needs its own consistency
gate. Do not assume the complex-step generator covers a tangent modified after
residual differentiation.

`reference_assembly` is a useful implementation oracle, but it enumerates
supported weak-form patterns. If it raises `NotImplementedError` for a new term
shape, extend the oracle or use a coupling-agnostic residual-difference check;
do not silently drop the missing block.

## State and ABI checks

Path-dependent tests should verify both the state array and a downstream
quantity such as stress or residual. Include a non-symmetric tensor state when
testing tensor layout: symmetric tensors can hide row/column-major transposes.

Check that:

- repeated trial evaluations start from the same committed state;
- line-search and derivative probes do not commit;
- the caller commits only an accepted step;
- generated and native state layouts agree; and
- re-pairing contact state is rotated or transferred under the documented rule.

The current generic-driver limitations for accepted-step state are part of the
test scope; see `docs/capabilities.md`.

## Nonlinear and coupled tests

Test more than one parameter point near branch or active-set changes. Sweep the
load, friction coefficient, material parameter, or time step relevant to the
method, and inspect diagnostics as well as displacement.

For coupled equations, report residuals on meaningful per-field scales. A
single global norm can hide a weakly scaled equation. Test field index maps,
block structure, and any FieldSplit configuration separately from the full
solve.

When a solve fails, inspect constrained null modes, element Jacobians, per-field
residuals, contact gaps, and the line-search/CCD step before changing a
tolerance. Load incrementation and line search help finite-strain problems, but
they do not repair an impossible model or missing constraint.

## Mesh, time-step, and extraction convergence

A coarse-mesh band is not a convergence study. Record the quantity of interest
over several resolutions and check that the error or inter-level difference
decreases in the expected regime. Include a level beyond the one used to choose
parameters when practical.

The extractor is part of the evidence. For contact width, reaction, peak
stress, or a singular field, document how the value is computed and whether it
has a mesh-converged meaning. Reject a nonconverged solver state before
evaluating a benchmark band.

Set tolerances from the expected source of error:

- roundoff for an exactly represented algebraic identity;
- finite-difference error for a numerical derivative;
- discretization error for a continuum solution; or
- solver tolerance and conditioning for a serial-versus-rank comparison.

## Cross-backend and external comparisons

For an Abaqus or other solver comparison, match and record:

- equations and sign conventions;
- element interpolation, integration, and stabilization/hourglass treatment;
- contact enforcement and friction law;
- mesh, loads, boundary conditions, and time history; and
- post-processing and quantity-of-interest extraction.

Sharing a mesh alone does not establish parity. Agreement between two outputs
generated from the same declaration primarily checks implementation. External
data also need clear redistribution or user-supplied boundaries.

## Broken controls

Use a broken control when it materially improves discrimination. Good controls
are structural: remove a required term, flip a flux sign, transpose state
layout, or violate a boundary relation. Avoid a control whose expected failure
depends on a narrow numerical tuning.

Not every small unit test needs an embedded broken implementation. Important
validation gates should demonstrate sensitivity to the defect class they are
meant to catch.

## Public evidence checklist

Before strengthening a public claim:

1. Run the exact source or installed-artifact path being claimed.
2. State the evidence category and non-scope.
3. Add an independent oracle for a physical-validation claim.
4. Add a broken control when it improves discrimination.
5. Record optional tools, data, environment, and skip conditions.
6. Update `examples/REFERENCES.md` and the capability matrix.
7. Keep raw distributed or performance output with the exact revision before
   citing it.

Useful public patterns include `tests/test_operator_contract.py`,
`tests/test_element_group.py`, `tests/test_curved_convergence.py`, and the
focused contact and paper-form tests named in the example references ledger.
