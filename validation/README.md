# CoupFE public test and evidence policy

The executable public tests live in [`../tests/`](../tests/). This directory
contains documentation only; it is not an importable `validation` package or a
model registry.

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
