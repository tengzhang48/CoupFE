# CoupFE contact vs ppf-contact-solver — a cross-check

An upstream interoperability and behavior cross-check for CoupFE's
deformable-contact + smoothed friction against [ppf-contact-solver][ppf]
(ZOZO, Apache-2.0). CoupFE adapts parts of the same algorithm lineage, so this
is not an independent physics validation. The external side is a separate
single-precision CUDA implementation.

The scenario both run: a **box on a rigid floor under gravity tilted by θ**
(equivalent to a slope of angle θ), with smoothed friction `μ`. ppf's
`scene.add.invisible.wall` ≡ CoupFE's `HalfSpace`.

- CoupFE side: `coupfe_box_on_floor.py` (in the test suite; needs `gfortran`).
- ppf side: `ppf_reference.py` (needs ppf built + a CUDA GPU; **not** in CI).

## This is a QUALITATIVE cross-check, by design

The two solvers are *not* unit-matched and should not be compared by magnitude:

| | ppf | CoupFE |
|---|---|---|
| precision | single (GPU) | double (CPU) |
| elasticity | `snhk` (stiff) | neo-Hookean, **G=1 (soft)** |
| gravity | g = 9.8 | **g = 0.4** in the current scoped CoupFE setup |
| observable | box centroid x | bottom-face **interface slip** |

The different units and models mean slide distances are not a quantitative
comparison. A retained rerun can compare only the explicitly stated qualitative
controls below.

## Rerun and evidence boundary

A fresh retained rerun can check non-penetration and the qualitative direction
of friction in both implementations. Record the upstream commit, build options,
GPU and CPU hardware, parameters, raw logs, and output hashes. Do not infer a
quantitative match from the different observables in the table above.

Both solvers use regularized rate-form friction, so neither is an exact-Coulomb
oracle near the transition. Clean qualitative controls are a clearly subcritical
case, a frictionless case, and non-penetration. Report the observed values only
from a retained rerun.

For a *quantitative* friction comparison you want a known-answer benchmark
(Cattaneo–Mindlin partial slip) with CoupFE's **return-map (exact Coulomb)**
friction, not the smoothed model. For a quantitative *normal* contact check see
`examples/hertz_contact/` (the analytic Hertz benchmark).

[ppf]: https://github.com/st-tech/ppf-contact-solver/tree/8b7740b032131aeeb46f51d882c96e09b171acc8
