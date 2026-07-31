# CoupFE contact and dynamics guide

Use this guide when changing contact operators, search, collision bounds,
friction state, or implicit dynamics. The mathematical background is in
[`docs/theory/contact_dynamics.md`](../docs/theory/contact_dynamics.md); the
authoritative implementation boundary is
[`docs/capabilities.md`](../docs/capabilities.md).

Contact examples are intentionally prominent in the public tree because they
exercise geometry, nonlinear solution, state, search, and solver integration
together. A passing example qualifies only the formulation and scale it states.

## Choose the formulation explicitly

| Need | Current path | Important boundary |
|---|---|---|
| Simple rigid contact | `RigidContact` penalty operator | Penetration is controlled by stiffness, not eliminated as a constraint. |
| Rigid barrier contact | `RigidBarrierContact` plus `max_step` | Requires a feasible start and a driver that honors the collision bound. |
| 2-D deformable penalty contact | `DeformableContact2D` | Pairing, orientation, and surface exclusions remain model responsibilities. |
| 2-D or 3-D deformable barrier contact | `DeformableBarrierContact2D` / `DeformableBarrierContact3D` | Current end-to-end demonstrations use implicit dynamics; broad self-contact support is partial. |
| Smoothed friction on a barrier | `mu` and `friction_eps` on the barrier operators | Regularized stick and lagged/PSD tangent assumptions must be part of the claim. |
| Return-map or persistent friction | `friction_kt` and, where supported, `friction_persistent` | Support differs between the 2-D and 3-D operators and requires correct commit/pair-transfer logic. |
| Exact-stick research study | `SemismoothFrictionSolver` and the exact-stick examples | Small-scale, linear-bulk or lagged-normal scope; not the general distributed contact path. |

Penalty, barrier, return-map, smoothed, and semismooth formulations are not
interchangeable. State the model used before comparing iteration counts,
penetration, slip, or reactions.

## Operator rules

1. Define the sign convention once. CoupFE assembles an energy-gradient-style
   residual; physical force is its negative. Derive normal signs from the
   stated gap and energy rather than intuition.
2. Separate discrete and smooth work. Determine active pairs, closest features,
   and stick/slip branches from the real iterate, then differentiate only the
   selected smooth branch.
3. Do not mutate committed friction state during residual, tangent, line-search,
   or complex-step evaluations. Commit after the caller accepts the step.
4. Treat state transfer across re-pairing as a physical operation. Project a
   persistent tangential quantity into the new frame and test dissipation and
   reversal memory.
5. Expose `max_step(U, dU, ...)` for a barrier path. The bound must cover every
   motion applied before residual evaluation, including an inertial predictor
   or moving obstacle.

The current generic Newton and increment drivers do not enforce accepted-step
history in every path. Check `docs/capabilities.md` before using them with a new
path-dependent contact state.

## Search and geometry

Test broad phase and narrow phase separately.

- A broad phase must return a conservative superset of nearby candidates. Test
  it against brute force on small randomized and degenerate surfaces.
- A narrow phase must cover face interiors, edges, vertices, parallel or nearly
  parallel features, and zero-length/zero-area defensive cases.
- Self-contact needs incident-feature and same-body exclusions appropriate to
  the model. Excluding too much misses contact; excluding too little can add
  self-forces from adjacent primitives.
- Surface orientation affects signed gaps, normal direction, and reaction
  signs. Include reversed-orientation broken controls.
- Freeze pairing only when the formulation states that per-step lag. An
  all-primitive barrier and a closest-feature node-to-segment model have
  different pairing semantics.

The optional numba and LBVH implementations are accelerators, not separate
oracles. Keep a readable NumPy reference where practical and add parity tests
over representative and degenerate geometry before making an equivalence
claim.

## Collision bounds and feasibility

A barrier energy alone does not prove nonpenetration. The collision bound is
the feasibility mechanism, and it assumes the current configuration is already
valid. Test at least:

- a clearly separated pair, where the full step is allowed;
- an approaching pair, where the bound is strictly between zero and one;
- a receding pair;
- a curved or multi-feature case whose linear estimate needs a safety check;
- a distributed case where the limiting pair is not owned by rank zero; and
- an invalid starting gap, which should be diagnosed rather than silently
  presented as protected by CCD.

Bulk feasibility is separate. A collision-safe step can still invert an
element; finite-strain applications may need a determinant or strain bound as
well.

## Friction state and tangents

For smoothed friction, document the smoothing length relative to the expected
step displacement. Smaller regularization is not automatically more accurate:
it can make the tangent poorly conditioned and the apparent stick response
step-size dependent.

For a return map, test both stick and slip branches, the cone inequality,
tangent asymmetry where expected, and the transition between branches. For
persistent finite-sliding state, also test edge crossing, frame rotation,
reversal, and nonnegative dissipation over a closed loading history.

Semismooth exact-stick examples are useful for studying stick/slip onset and
frozen-active-set derivatives. Their small-interface results should not be
generalized to large three-dimensional contact without a different scalable
interface solver.

## Dynamics and dynamic relaxation

The implicit backward-Euler path can regularize difficult contact transitions
and dissipate transients. That does not make every final frame quasistatic.

A dynamic-relaxation claim needs a staged load protocol:

1. ramp the load slowly relative to a computed structural timescale;
2. hold the load;
3. monitor kinetic energy, static residual, contact feasibility, and reaction;
4. repeat with a smaller time step or slower ramp; and
5. report the convergence criteria with the result.

Reject a state that merely looks settled. Damping and inertia must be small on
the stated force or energy scale at the sampling point.

## Distributed contact

Rigid analytical contact is node-local and can be assigned by DOF ownership.
Deformable contact couples surface entities across ranks; the current path
replicates surface positions, assigns each candidate pair to one owner,
assembles with global indices, and reduces the collision bound globally.

Every collective must be called on all ranks, even when one rank has no local
contact entity. Qualify the surface assembly, global minimum reduction, and
solution separately. The public MPI programs are rerunnable diagnostics; there
is no retained final-revision multi-rank record in this release.

See [`skills/distributed.md`](distributed.md) for environment and evidence
requirements.

## Validation ladder

Build contact evidence from small independent checks before an end-to-end
collision:

1. gap, closest-point, normal, and derivative checks;
2. residual/tangent consistency on a frozen branch;
3. action/reaction balance and orientation controls;
4. collision-bound checks;
5. analytic or independently derived limits such as Hertz or capstan relations;
6. mesh/time-step/refinement trends where discretization matters; and
7. an end-to-end example with explicit convergence and feasibility gates.

The public contact entries and their evidence are indexed in
[`examples/README.md`](../examples/README.md) and
[`examples/REFERENCES.md`](../examples/REFERENCES.md). Do not convert a
qualitative external comparison into a validation claim without the exact
model, data authority, run record, and extractor.

## Performance

Profile broad phase, narrow phase, element assembly, and linear solve
separately. Vectorized NumPy is often suitable for uniform node-local work;
compiled loops help branch-heavy feature geometry; sparse factorization can
still dominate both. Record hardware, software, solver settings, convergence,
and the input with any performance number.

The ppf-contact-solver-derived geometry and smoothed-friction work is credited
in [`NOTICE`](../NOTICE). Preserve that provenance when moving or translating
the relevant algorithms.
