# CoupFE public test and evidence policy

The executable public tests live in [`../tests/`](../tests/). This directory
contains documentation only; it is not an importable `validation` package or a
model registry.

The worked response to the 2026-08-08 public-example audit is recorded in
[`example_review_2026-08-08.md`](example_review_2026-08-08.md).

Run the default public suite from the repository root with:

```bash
python -m pytest -q
```

The project configuration excludes tests marked `slow` by default. Optional
code-generation, compiled-element, numba, PETSc, MPI, or external-data paths
require their corresponding tools and environments. Long workflows can be run
explicitly with:

```bash
python -m pytest -q -ra -m slow
```

## What the public tests cover

The shipped suite includes focused checks for:

- operator assembly, nonlinear solves, dynamics, and exact affine constraints;
- compiled element groups, state behavior, and the declarative model pipeline;
- regular-mesh refinement, geometry checks, and in-process distribution;
- two- and three-dimensional contact primitives and selected end-to-end
  examples;
- return-map, persistent, finite-sliding, and semismooth friction studies;
- UEL/UMAT generation, deterministic generated source, compilation, state, and
  reference-assembly behavior for the examples that claim those checks; and
- analytic or independently implemented checks for selected examples such as
  the linear bar, curved annulus, finite-block Hertz normal-force check, and
  material-point paths.

`examples/mpi_smoke/` contains rerunnable MPI programs. They are useful for
qualifying a concrete PETSc/MPI installation, but this release does not publish
a retained final-revision multi-rank result.

## Evidence categories

Tests in this repository answer different questions:

- **Analytic or independent oracle:** compares a measured quantity with a
  separately derived result or invariant.
- **Convergence evidence:** checks the expected trend under refinement.
- **Implementation consistency:** compares a tangent with a numerical
  derivative, a generated kernel with reference assembly, or native and UEL
  backends generated from the same form.
- **Compilation and integration:** verifies that generated source builds and a
  documented execution path runs.
- **Broken control:** demonstrates that a known incorrect variant fails a
  discriminating check.

Implementation consistency is not independent physical validation when both
sides share the same equations. A passing build or tangent check also does not
establish convergence of a boundary-value problem.

## Adding evidence

Add a normal pytest case under `tests/` and keep its claim narrow:

1. state the equations, units, conventions, and supported regime;
2. identify whether the check is analytic, independent, consistency-based,
   convergence-based, or an integration check;
3. use an independent oracle when making a physical-validation claim;
4. include a broken control when it materially improves discrimination;
5. state optional tools or user-supplied inputs explicitly; and
6. update [`../examples/REFERENCES.md`](../examples/REFERENCES.md) when the test
   changes an example's evidence boundary.

Do not turn a skipped optional path, an unretained run, or agreement between two
generated backends into a stronger public claim than the evidence supports.

## Responding to a review finding

Treat a review note as a hypothesis to reproduce, not as either an instruction
to edit blindly or a reason to defend the current code.  Use this sequence:

1. Pin the reviewed revision and preserve unrelated local changes.
2. Restate the claimed invariant numerically at the smallest useful layer.
3. Reproduce it with a calculation independent of the implementation and its
   existing oracle.
4. Classify the failure as core behavior, example behavior, evidence/extraction,
   documentation, or a review assertion that is not reproducible.
5. Identify every caller that shares the affected convention or raw ABI before
   changing a public parameter meaning.
6. Make the smallest complete correction and retain compatibility explicitly;
   do not silently reinterpret an established raw-kernel property tuple.
7. Add a direct regression for the physical invariant and a broken control for
   the reported defect class.  Do not assert a value returned solely for the
   purpose of satisfying the same assertion.
8. Regenerate committed artifacts when their declaration changes, then run the
   focused tests, the default suite, release/site checks, and applicable
   compiled or end-to-end paths.
9. Record what was confirmed, what was disproved, the resulting claim boundary,
   and any compatibility work deliberately deferred.

This process intentionally separates diagnosis from repair.  A derivative
check may confirm the derivative of an incorrect constitutive map; a positive
unsigned distance may coexist with surface crossing; and two names such as
``K`` and ``lambda`` may hide a public/raw-ABI mismatch even when every
self-consistency test passes.
