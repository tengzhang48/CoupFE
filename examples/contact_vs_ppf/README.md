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
| gravity | g = 9.8 | **g = 0.4** (`εg=ρgL/G ≲ 0.5` or no converged equilibrium) |
| observable | box centroid x | bottom-face **interface slip** |

So slide *distances* are meaningless to compare. What must agree is the
**behaviour**.

## Historical, unretained observations

The table below records development observations whose raw external log and
environment were not retained. It is useful context for rerunning the recipe,
not current release evidence.

| config | ppf (centroid_x, min gap) | CoupFE (slip, min gap) | agree |
|---|---|---|---|
| **non-penetration** (every config) | min_y `> 0` (even sliding 17.7) | min_gap `≈ +0.033` | ✅ **yes** |
| friction holds below threshold | θ=20° μ=0.5 → held `+0.026` | θ=20° μ=2.0 → stick `+0.073` | ✅ both stick |
| frictionless slides | θ=35° μ=0 → `+17.7` | θ=20° μ=0 → `+0.40` (5.5× the stick) | ✅ both slide |

**What a fresh retained rerun could check:** non-penetration and the qualitative
direction of friction in both implementations. The historical numbers above do
not establish those as release claims.

## Model nuance (the "ppf isn't exact Coulomb" caveat, made concrete)

At **θ=35°, μ=0.5** (tan 35° = 0.70 > 0.5, so ideal Coulomb says *slide*):

- **ppf over-holds** — centroid `+0.054` (essentially stuck).
- **CoupFE slides** — interface slip `+0.26`.

Both use a *regularized rate-form* friction (not exact Coulomb), so neither is
the ground truth near the threshold; they just regularize differently. The
clean, model-agnostic comparison points are therefore **well below threshold
(both stick)** and **frictionless (both slide)** — plus **non-penetration always**.

For a *quantitative* friction comparison you want a known-answer benchmark
(Cattaneo–Mindlin partial slip) with CoupFE's **return-map (exact Coulomb)**
friction, not the smoothed model. For a quantitative *normal* contact check see
`examples/hertz_contact/` (the analytic Hertz benchmark).

[ppf]: https://github.com/st-tech/ppf-contact-solver/tree/8b7740b032131aeeb46f51d882c96e09b171acc8
