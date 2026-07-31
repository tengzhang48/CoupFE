# Theory: contact, friction, and implicit dynamics

This page summarizes the formulations implemented in CoupFE. The operational
guide is [`../../skills/contact.md`](../../skills/contact.md), the exact support
boundary is [`../capabilities.md`](../capabilities.md), and upstream adaptation
credits are recorded in [`../../NOTICE`](../../NOTICE).

Notation: `u` is the global displacement vector, `R(u)` is the assembled
residual, and `K = ∂R/∂u` is its tangent. CoupFE uses an energy-gradient
sign convention, so physical force is `-R`. For a rigid obstacle, `g(x)` is the
signed gap (`g > 0` when separated) and `n = ∂g/∂x` is the outward normal.

## Operator and tangent boundary

Bulk elements, contact, inertia, and loads compose through the same
`(residual, tangent, commit)` operator interface. Their tangents need not be
constructed the same way.

For an analytic residual, complex step can provide

```text
K_ij = Im(R_i(u + i h e_j)) / h.
```

It avoids subtraction cancellation when the entire perturbed path preserves
the imaginary component. Absolute values, real casts, branch changes, search,
and active-set decisions require special treatment. Contact code may instead
use an explicit analytic, Gauss–Newton, or semismooth tangent. When complex step
is used on a piecewise-smooth contact branch, select the discrete branch from
the real iterate and hold it fixed during differentiation.

## Penalty normal contact

For penetration `g < 0`, a quadratic penalty energy is

```text
Pi_c = 1/2 k g^2,
R_c  = k g n,
K_c  = k n ⊗ n + k g ∂n/∂u.
```

The active set is selected from the real iterate. Penalty contact is simple and
allows a small penetration whose size depends on stiffness, load, mesh, and
conditioning. It is not an exact constraint.

## Incremental penalty–Coulomb friction

The return-map path uses a tangential stick spring and a Coulomb cap. With
`P_t = I - n ⊗ n` and tangential increment `Delta s`,

```text
f_t^trial = P_t f_t^committed + k_t Delta s,
cap       = mu |f_n|.
```

The update is

```text
stick: |f_t^trial| <= cap  ->  f_t = f_t^trial
slip:  |f_t^trial| >  cap  ->  f_t = cap f_t^trial / |f_t^trial|.
```

The stick/slip branch is frozen while its tangent is evaluated. The resulting
tangent can be nonsymmetric because normal and tangential responses are
coupled. The committed tangential force is path-dependent state. Re-pairing a
finite-sliding contact therefore also requires an explicit rule for rotating or
transferring that state.

The interface separates evaluation and commit, but a driver must still invoke
commit only after accepting a step. The current generic-driver limitation is
documented in [`../capabilities.md`](../capabilities.md).

## Smoothed barrier friction

The barrier path also offers a smoothed, semi-implicit friction model adapted
from the IPC/ppf-contact-solver family. Let `x0` be the step-start contact
configuration, `dx` the relative contact-point displacement, and
`P = I - n ⊗ n`. With smoothing length `epsilon`,

```text
lambda = mu lambda_n / max(epsilon, |P dx|),
R_t    = lambda P dx,
K_t    = lambda P.
```

For `|P dx| >= epsilon`, the residual magnitude reaches the Coulomb cap. Below
that threshold, the model behaves like a regularized tangential spring. The
implemented tangent holds the normal direction and normal-force magnitude
fixed and drops their derivatives, producing a symmetric positive-semidefinite
Gauss–Newton approximation.

This formulation has useful computational structure but makes explicit
approximations:

- the normal data are lagged within the friction differentiation;
- exact static stick is replaced by a smoothing-length-dependent creep; and
- the PSD tangent changes the Newton path relative to the full derivative.

Accuracy and conditioning therefore depend on time step, smoothing length,
normal loading, and the problem. Exact-stick questions belong to the separately
scoped return-map or semismooth studies.

The relevant ppf-contact-solver source lineage is Apache-2.0 and identified in
`NOTICE`; CoupFE adapts the algorithms rather than claiming backend identity.

## Cubic barrier and collision bound

Inside an activation band `d < dhat`, the cubic energy used by the barrier path
is

```text
B(d)   = kappa/3 (dhat - d)^3,
B'(d)  = -kappa (dhat - d)^2,
B''(d) = 2 kappa (dhat - d).
```

Unlike a logarithmic barrier, this polynomial energy remains finite at zero
distance. That improves one aspect of conditioning but means the energy alone
does not enforce nonpenetration.

Feasibility comes from `max_step(u, du)`, which limits an approaching update.
For a flat obstacle, a linear estimate is

```text
g(alpha) = g0 + alpha (Delta x . n),
alpha <= eta [-g0 / (Delta x . n)]
```

when `Delta x . n < 0`, with safety factor `eta < 1`. Curved and deformable
features use additional closest-feature and conservative-advancement checks.

A collision-bound claim requires all of the following:

- the starting configuration is feasible;
- every relevant candidate pair is included;
- the bound covers predictors, moving obstacles, and Newton corrections; and
- the chosen driver honors the returned global minimum.

If any condition is absent, the barrier should not be described as an
unconditional nonpenetration guarantee. Bulk element inversion is a separate
feasibility problem.

## Implicit dynamics and relaxation

The backward-Euler driver solves, per step,

```text
R(u) = M/dt^2 (u - u_hat)
     + F_int(u) + F_contact(u) - F_ext = 0,
u_hat = u_prev + dt v_prev,
v     = (u - u_prev) / dt.
```

Optional mass-proportional damping adds a term proportional to
`M (u - u_prev) / dt`. Inertia and damping can regularize difficult contact
transitions, which is why the shipped deformable-barrier demonstrations use
implicit dynamics. This is an implementation choice and not a theorem that a
quasistatic contact solve cannot work.

Dynamic relaxation is a claim about a protocol, not just an integrator. Ramp
the load relative to a computed structural timescale, hold it, and sample only
after kinetic energy, inertial/damping forces, the static residual, and the
quantity of interest satisfy stated tolerances. Repeat with a smaller time step
or slower ramp before calling the state quasistatic.

The optional gap-dependent stiffness contribution used by some barrier paths
has the simplified CoupFE form `s = kappa + M/d^2`. It increases capacity as a
positive gap becomes small, but must be regularized numerically and does not
replace the collision bound. It is inspired by ppf contact-stiffness ideas; it
is not a complete reimplementation of every upstream stiffness term.

## Three-dimensional deformable contact

Three-dimensional surface contact uses two primitive families:

- vertex–face, based on point–triangle closest distance; and
- edge–edge, for contacts not represented by a vertex–face pair.

Closest-point weights map each primitive contribution to its nodal stencil. The
cubic-barrier residual uses the closest-distance gradient, while additive
continuous collision detection supplies a safe-step estimate. These geometry
and collision algorithms were adapted from ppf-contact-solver and modified for
CoupFE; see `NOTICE`.

The broad phase constructs conservative AABB candidates. An optional numba
LBVH implementation is available, with a uniform-grid fallback. NumPy and
numba implementations require explicit parity tests before an equivalence or
performance claim.

`DeformableBarrierContact3D` combines vertex–face and edge–edge contributions
and supports the scoped smoothed-friction path. Exact vertex/edge coincidences
can be represented by more than one primitive stream and may double-count a
contribution; this is a documented limitation rather than a general
deduplication guarantee.

## Evidence boundary

Primitive derivative tests, serial/numba agreement, and serial/rank agreement
check implementation properties. They do not independently validate a contact
model or external benchmark. Use the analytic and example-specific evidence in
[`../../examples/REFERENCES.md`](../../examples/REFERENCES.md), and keep penalty,
barrier, smoothed-friction, return-map, and semismooth claims separate.
