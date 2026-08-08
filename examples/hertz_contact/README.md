# Rigid-sphere Hertz contact

This example is CoupFE's quantitative normal-contact check. A rigid sphere
indents a compressible neo-Hookean Hex8 block, and the resulting base reaction
is compared with the classical elastic-half-space relation

\[
F = \frac{4}{3} E^* \sqrt{R}\,\delta^{3/2}, \qquad
E^* = \frac{E}{1-\nu^2}.
\]

The retained calculation uses `E = 10`, `nu = 0.3`, `R = 2`, a
`2.5 × 2.5 × 2.4` block, and a `16 × 16 × 8` Hex8 mesh. The base is fixed in
all three displacement components; the sides and top are traction free except
for frictionless contact with the sphere. Five prescribed approaches span
`delta = 0.02` to `0.08`.

The compiled kernel evaluates
`P = G(F - F^-T) + lambda ln(J) F^-T`. Its second property is therefore the
first Lamé coefficient
`lambda = E nu / ((1 + nu)(1 - 2 nu))`, not the physical bulk modulus. Using
that conversion keeps the finite-element infinitesimal tangent consistent with
the `E` and `nu` used by the Hertz oracle.

## Run and render

From the repository root, after installing the runtime dependencies and a
Fortran compiler:

```bash
PYTHONPATH=. python examples/hertz_contact/run.py
PYTHONPATH=. python examples/hertz_contact/render.py
```

The solver command writes no files. The explicit render command regenerates
[`../../docs/assets/hertz-contact-benchmark.svg`](../../docs/assets/hertz-contact-benchmark.svg)
from a fresh checked solve.

The retained result is:

| Approach `delta` | CoupFE force | Hertz force | `F_FE / F_Hertz` |
|---:|---:|---:|---:|
| 0.020 | 0.05937 | 0.05861 | 1.013 |
| 0.035 | 0.13744 | 0.13568 | 1.013 |
| 0.050 | 0.23748 | 0.23167 | 1.025 |
| 0.065 | 0.36452 | 0.34338 | 1.062 |
| 0.080 | 0.49196 | 0.46886 | 1.049 |

The five-point log-log slope is `1.533`, compared with the Hertz exponent
`1.500`. The focused regression uses `delta = 0.03, 0.05, 0.07`, checks the
force to within 8%, checks the exponent to within 0.08, and independently
checks free-DOF equilibrium and base/contact force balance. An opt-in slow
broken control substitutes physical bulk modulus for the required Lamé
coefficient and verifies that the same force gate rejects the mismatch.

## What the figure shows

The left panel is the actual final finite-element solution at `delta = 0.08`:
the exterior Hex8 faces are drawn at true deformation scale and colored by
nodal downward displacement. Gold markers identify active contact nodes and
vary in size with their vertical nodal reactions. They are **not** a
reconstructed contact pressure. The right panel compares all five solved forces
with the Hertz law on logarithmic axes.

## Evidence boundary

This is a finite-block, discrete-node penalty calculation, not a converged
half-space contact solution. The fixed base and finite lateral extent stiffen
the response; the smallest analytic contact radius spans only about 1.3 surface
elements; and the active set changes by nodal rings. The reported
`active_node_radius` is consequently a mesh diagnostic, not a validated contact
radius or pressure-footprint measurement. Penalty stiffness, mesh, and domain
studies are required before using this setup for a convergence claim.

The comparison also places a finite-strain neo-Hookean model against the
infinitesimal Hertz relation. Within those declared limits, the example checks
the normal-force law and equilibrium of the implemented retained setup. It does
not establish general contact accuracy, friction accuracy, stress accuracy, or
device-level validation.

The analytic reference follows K. L. Johnson, *Contact Mechanics*, Cambridge
University Press, 1985, ISBN `978-0-521-34796-9`. The SVG is generated entirely
from project-authored model data and solver output.
