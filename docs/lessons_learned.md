# CoupFE lessons learned

> **Release-evidence boundary:** exact timings, speedups, iteration counts, and
> scale figures in this dated narrative are historical local observations
> without retained raw logs and a locked environment. Rerun them before
> citation. Functional and correctness gates are tracked separately.

Dated, narrative lessons from building CoupFE. The codified, broken-control-tested
failure modes live in `skills/pitfalls.md`; this file is the "why we did it this way"
record. Newest first.

## 2026-07-02 — The operator-contract split double-evaluated the element kernel; the Abaqus UEL is joint-by-design (Teng)

**Serious performance regression, now fixed.** The compiled element kernel
(`element_rk_batch`, driving the generated Fortran `SUBROUTINE UEL`) returns the residual
`R` (Abaqus `RHS`) and the tangent `K` (Abaqus `AMATRX`) TOGETHER in one element call.
Because `K` is obtained through complex-step residual evaluations, `R` is inexpensive once
`K` has been formed; `K` is not inexpensive when only `R` is needed. CoupFE's `Operator` contract
splits evaluation into two methods, `ElementGroup.residual()` and `.tangent()`, and Newton
calls both at the SAME `U`. Each re-ran the full kernel and discarded half its output, so
**every Newton iteration evaluated the element kernel twice.** Measured share of a
compiled-element solve: assembly ~82% (residual 53% + tangent 29%), of which roughly half
was redundant.

**Root cause (Teng's diagnosis): `abaqus_ufl_lab` is right because it strictly follows the
Abaqus setup.** The Abaqus `UEL` computes `RHS` and `AMATRX` in ONE subroutine call — joint
by design. The lab's `abaqus_ufl.fe` driver preserves that exactly: `newton_solve` →
`assemble(...)` → `R, K` in a single pass, one `element_fn` call per element per iteration.
CoupFE regressed it when it wrapped the joint kernel in a SPLIT operator contract to compose
element + contact + inertia operators. The composability is worth keeping; the split
evaluation was an accident of the abstraction, not a requirement of it.

**Fix (`coupfe/operators/element_group.py`):** a one-entry `_rk` cache so the paired
residual/tangent at one iterate share a single kernel call. Key = `(U_g, DU_g, props)` —
`props` is in the key because callers legitimately mutate them per step (active stress `Ta`
in cardiac), and `commit()` clears the cache so committed-state changes can never serve
stale `K`; line-search trials at `U + αdU` change `U_g` and correctly miss. **Toggle
`fuse_rk` (default ON; env `COUPFE_FUSE_RK=0`)** keeps both regimes: ON restores the
Abaqus/lab joint evaluation for standard Newton; OFF gives independent per-call evaluation
for schemes that mutate props/state between a residual and its paired tangent,
residual-only/matrix-free loops, or debugging.

Verified: neo-Hookean block 64 → 24 kernel calls, **bit-identical** (0.0 diff);
cardiac Case A 77 s → 44 s (1.77×), full u(t) history bit-identical; a prop-change
staleness broken-control gates the key (`tests/test_element_group.py`).

Takeaways: (1) when a kernel returns R and K jointly (as the Abaqus UEL does), a contract
that evaluates them separately silently doubles the hot cost — the lab's single-pass
`assemble` is the reference shape; (2) the *proper* long-term fix is a combined
`Operator.residual_and_tangent()` on the contract (mirroring `assemble`), with the cache as
the minimal, bit-identical bridge; (3) profile assembly-vs-solve before any "the solver is
slow" conclusion — here the linear solve was only 14% and MUMPS would have been *slower* on
the thin-walled LV (structure, not dimensionality).

## 2026-07-02 — Solver guidance for AI agents: ONE policy module + simple/informative/CORRECT docs beat detailed instructions (Teng)

The solver story had gone random: `newton_solve` used bare `spsolve`, five `cattaneo_3d`
scripts each hand-rolled a different PETSc KSP setup, the branch helpers were never
re-homed, and the sibling projects (abaqus_ufl.fe, CoupFE-EDA) each carried their own
setups. An agent landing anywhere would invent yet another. The consolidation
(`coupfe/assembly/factored.py` + the "solver ladder" in `skills/performance.md`) worked,
and HOW it worked is the lesson — Teng's framing: **AI agents are now powerful enough
that simple, informative, and correct guidance is sufficient to choose the right
solver.** The three words carry weight:

- **Simple** — one module, one table, two env knobs. Detailed per-case setups stay in the
  individual projects; only the LADDER and the traps are shared. An agent doesn't need a
  solver manual; it needs the decision rule and where the boundary is.
- **Informative** — every rung carries its MEASURED anchor (SuperLU 400 s vs MUMPS ~45 s
  at 96k 3D DOFs; hypre 2-4x gamg and beating direct on 2D SPD at ~90k; the lab's
  constant-7-iterations AMG). Numbers, not adjectives — an agent can extrapolate from an
  anchor; it can only obey an adjective.
- **Correct** — which is where the first draft failed twice, caught by Teng: "2D at any
  size -> SuperLU" was an overclaim (measured only to ~130k; at 490k 2D Poisson SuperLU
  is 8x slower than hypre), and gamg was documented as the AMG default when hypre had
  been measured better in the lab. **A claim without its measured bound is a future
  wrong decision.** State the bound, state the counter-anchor, and say what the real
  reason is (2D contact stays direct for EXACTNESS, not speed).

Corollaries: port-don't-reimplement paid again (`make_fieldsplit_solver` came from
abaqus_ufl.fe with its validation history, gaining only house hardening — COMM_SELF,
destroy-in-finally, raise-with-guidance); and a solver policy is TESTABLE — policy
selection, all-backend agreement, and divergence-must-raise are gated in
`tests/test_factored.py`, so the guidance cannot silently rot.

## 2026-07-02 — Pre-flight is mandatory in the AI era: dimensionless analysis + a dry run of the SETUP (Teng)

Codifying Teng's directive after the ring-compression episode below: **a dimensionless
analysis and a dry run of the simulation — testing the boundary conditions, the loading
protocol, and the material properties — are critical in the AI era.** Generating a
runnable simulation is now cheap; the dominant failure mode is a wrong setup that runs
fine, gets debugged as if the physics or the solver were at fault, and ends in a false
"method limitation" that pollutes the record for the next agent.

The two disciplines, in order (full checklist: `skills/preflight.md`):

1. **Dimensionless analysis FIRST.** Identify the governing groups and compute the
   structural timescales NUMERICALLY (`eigsh(K, M)` is one call) before choosing any
   ramp time, damping, dt, or penalty. Derive every protocol number from the groups.
   The ring failed exactly here: a "slow" 40 s ramp against a computed T_1 = 126 s
   fundamental period, damping chosen without ever forming zeta = alpha/(2*omega_1).
2. **A dry run that tests the SETUP, not the physics**: rigid-body-mode count and
   reaction sums for the BCs (with a deliberately-broken BC as the control); the
   loading driver logged and inspected against intent, with a KE monitor wired in and
   sampling allowed only at settled states; material scales re-derived from the
   ASSEMBLED model (mass, patch modulus) so unit slips can't hide.

The gate this imposes: no production run, no validation claim, and no
"method/solver limitation" conclusion until the pre-flight artifacts (group table +
dry-run log) exist. A surprising result — especially one where gentler loading looks
worse — is a setup bug until the pre-flight proves otherwise.

## 2026-07-02 — CORRECTION: the ring diagnosis required a settled-state protocol

This is a historical development record. The proprietary input provenance,
environment, and raw comparison run were not retained, so the numerical
observations in this investigation are not first-release validation evidence.

The 2026-06-30 lesson below concluded that `solve_dynamics` "does not collapse onto the
quasistatic branch" for the ring compression. That conclusion was WRONG — the textbook
example of closing a problem by citing a limitation instead of disproving it. Nobody had
computed the ring's timescales:

- The free ring's fundamental (ovalization) mode is `omega_1 = 0.0498 rad/s`
  (`eigsh(K, M)`), i.e. **T_1 = 126 s**. The "slow" 40 s ramp was 3x FASTER than one
  period — impulsive loading. Even the 400 s check ran at `zeta = alpha/(2*omega_1) =
  0.01` — 100x underdamped — and sampled mid-flight with **no hold phase at all**.
- Every reported pathology follows: the non-monotonic RF2 (snapshots of an oscillating
  transient), the wrong equator U1, and "slower ramps separate" (the ring caught
  mid-rebound at the sampling time — a physically inverted result that should have been
  read as a SETUP bug, not a method limit).
- "A hold with higher damping corrupts the reaction because `αM·v` dominates" is only
  true WHILE the ring moves. As `v -> 0` the damping force vanishes identically, so the
  settled state solves `F_int + F_contact = 0` exactly — the settled sample is an
  equilibrium and is dt-independent.

**The implemented protocol:** staged ramp-hold-settle, with damping selected
from the computed structural timescale and sampling permitted only after a
stated kinetic-energy threshold. A private development check once compared
this path with external values, but that input/output record is incomplete and
the test is not in the reviewed public test partition. Re-run and retain the
full evidence before reporting agreement.

**Takeaways.** (1) Dimensionless analysis FIRST: compute `omega_1` before choosing ramp
time and damping — "slow" and "low damping" are meaningless without the structural
timescale. (2) Sample a dynamic-relaxation run only at a KE-gated settled state, never
at the end of a ramp. (3) A result where SLOWER loading looks worse is a protocol/setup
bug by default. (4) Dynamic relaxation can be used for a settled equilibrium
study only when the kinetic-energy and residual gates demonstrate that inertia
and damping no longer control the reported state.

## 2026-06-30 — `solve_dynamics` is a transient solver, not a quasistatic solver [CORRECTED 2026-07-02 — see the entry above; the root-cause claim below is wrong]

This superseded entry sampled the end of a ramp while the ring was still
oscillating and misdiagnosed the resulting transient as a method limitation.
The durable lesson is narrower: `solve_dynamics` is a transient integrator, and
a dynamic-relaxation study must establish the structural timescale, include a
hold, and gate the reported state on kinetic energy and equilibrium residual.
Keep `reproduce_dynamics.py` as a RESEARCH contact-method workflow until its
external input and full rerun evidence are retained.

## 2026-06-30 — Rigid-barrier ppf-derived implementation coverage (non-GPU)

After three commits on `cattaneo-3d` the rigid-barrier contact path now covers the
non-GPU parts of the ppf-contact-solver recipe:

1. **Geometry-normalized cubic barrier** (`ppf_norm=True`) — decouples activation distance
   from force magnitude.
2. **Adaptive stiffness** (`mass`) plus optional `elastic_op` — completes
   `s = wᵀ(K_elast + M/g²)w` beyond the scalar `κ` approximation.
3. **Time-aware CCD** for moving obstacles (`MovingHalfSpace`/`MovingSphere`) — samples the
   obstacle trajectory at `t − dt + α·dt` during predictor jumps.
4. **Kinematic obstacles** (`kinematic=True`) — floors the effective gap at
   `constraint_tol * dhat`, preventing a moving wall from sticking/slamming.
5. **Shell pass-through** (`thickness`) — ignores nodes that have tunneled deeper than the
   obstacle thickness.

**Evidence boundary.** Passing public tests cover the shipped barrier
primitives and smaller contact examples. The historical ring comparison lacks
a retained raw run and is not validation evidence; do not use it to claim
cross-engine agreement.

The remaining unported ppf items are 3D self-contact broad-phase/LBVH/ACCD and bulk `J>0`
strain limiting — both outside the plain-Python rigid-barrier path.

## 2026-06-30 — ppf-style kinematic obstacles and pass-through thickness

The next ppf port after the geometry-normalized barrier and time-aware CCD was the obstacle
metadata that keeps prescribed-motion walls from sticking and that lets thin shells pass
through.

**What ppf does.** ppf's `Floor`/`Sphere` structs carry a `kinematic` flag and a `thickness`.
A kinematic floor uses `gap = max(gap, constraint_tol * ghat)` when evaluating the push
constraint, so a node pressed by a moving wall never sees an unbounded repulsive force. A
finite thickness means the obstacle is a shell: if penetration exceeds `thickness`, the
constraint is skipped entirely.

**Fix.** Added `kinematic`, `thickness`, and `constraint_tol` to `HalfSpace`, `Sphere`,
`MovingHalfSpace`, and `MovingSphere`. `RigidBarrierContact` forwards them to
`rigid_barrier_eval`:
- `kinematic=True` floors the effective gap at `constraint_tol * dhat` for both the force
  and the CCD bound, preventing stick/slam behavior.
- `thickness > 0` removes nodes whose penetration depth exceeds it from the active set.

**Tests.** `tests/test_contact_barrier.py` now checks that a kinematic floor caps the barrier
force and that a shell with finite thickness ignores deep-penetration nodes while still
repelling shallow ones.

## 2026-06-30 — ppf-aligned barrier shape + time-aware moving-obstacle CCD

Re-reading the reference [ppf-contact-solver](https://github.com/st-tech/ppf-contact-solver)
pointed to two remaining gaps in CoupFE's barrier contact: the cubic-barrier shape was not
geometry-normalized, and the CCD step bound did not follow a moving obstacle over the predictor
interval.

**What ppf does.** The ppf cubic barrier separates *shape* from *stiffness*: the shape carries a
`2/d̂` factor, so the force is `−(2/d̂)·s·(d̂−d)² n` with `s = wᵀ(K_elast + M/g²)w`. In CoupFE the
shape coefficient `1/3` was folded into `κ`, which made `κ` a shape+stiffness hybrid and tied the
force magnitude to `d̂`.

**Fix 1: `ppf_norm=True`.** `rigid_barrier_eval` and `RigidBarrierContact` now accept `ppf_norm`.
When enabled, the barrier uses the normalized cubic shape (`2/d̂`), so `κ` becomes a true contact
stiffness (force/length) and the adaptive `s = κ + M/d²` term matches the ppf recipe. Existing
calls default to `False` and are byte-identical.

**Fix 2: time-aware CCD.** The old `_MovingObstacleContact` workaround updated `obs.p` to the
obstacle's position at the *target* time before `max_step`, but the predictor CCD still evaluated
the gap at that single end position — a node could in principle tunnel during the step and end
separated. Now `HalfSpace`/`Sphere` accept an optional time argument, and `MovingHalfSpace` /
`MovingSphere` carry a callable trajectory. The driver's `_call_max_step` forwards both `t` and
`dt`; `RigidBarrierContact.max_step` evaluates the obstacle at `t − dt + α·dt` during the safety
bisection, so the bound respects the actual relative motion over the predictor interval.

**Example update.** `examples/ring_compress/reproduce_dynamics.py` now uses `MovingHalfSpace` for
the top platen. The barrier run remains stable (33 steps, RF2 ≈ 13292) and the reaction is reported
at the final platen position.

**Tests added.** `tests/test_contact_barrier.py` gates the normalized force magnitude, the
load-balance equilibrium with `ppf_norm=True`, and the time-aware CCD against a rising floor.

## 2026-06-30 — Dynamic relaxation of the same ring with `solve_dynamics` / ppf-style barrier contact

Adding a separated dynamics version (`examples/ring_compress/reproduce_dynamics.py`) of the
`ring_compress` example drove home the difference between penalty contact and the ppf-style
cubic barrier (`RigidBarrierContact`) in a production-like setting.

**The failure path (barrier).** A first attempt using the same fixed-dt driver that worked for
the penalty case hung: Newton iterations climbed, the CCD `alpha` collapsed to ~`2^-30`
(≈9.3e-10), and the solve made no progress. Two separate issues compounded:
1. **Initial gap was zero** on the bottom plate — ring outer radius = 10 and the bottom
   `HalfSpace` was at `y = -10`. The barrier's CCD bound at a touching node is `0`, so the
   predictor/Newton step is clamped to near zero and the barrier force is evaluated at `d ≈ 0`
   where `M/d²` explodes. **Fix:** offset *both* plates by `±dhat` so every contact node starts
   with gap `> 0`.
2. **`maxit` was not being passed through.** `solve_dynamics_adaptive` accepted a `maxit`
   argument but forwarded only `**newton_kw` to `newton_solve`, so `newton_solve` used its
   default `maxit=60`. The step-statistics looked "converged" but the example was silently
   doing far more work than intended, and the barrier case occasionally hit the ceiling.
   **Fix:** pass `maxit=maxit` explicitly.
3. **The moving-plate wrapper only updated the obstacle position in `residual/tangent/commit`,
   not in `max_step`.** The predictor CCD evaluates `max_step` *before* the first residual of
   the step, so the CCD bound was computed against the *previous* plate position. A node could
   be allowed to jump through the moving plate. **Fix:** update `obs.p = motion(t)` inside the
   wrapper's `max_step` and add a `_call_max_step` helper that forwards time `t` to operators
   that accept it.
4. **`M/d²` divides by zero when the gap is driven to machine precision.** Even with CCD the
   gap can become tiny; `mass / d²` overflowed to `inf` and poisoned the Newton solve.
   **Fix:** floor `d²` in the barrier force and friction stiffness (`np.maximum(d², 1e-24)`).
5. **CCD bound collapsed to zero when a node was already close to the obstacle.** The safety
   bisection kept halving `alpha` until it underflowed. **Fix:** introduce a `gap_floor` in
   `RigidBarrierContact.max_step` (absolute `1e-4` plus a fraction of `dhat`) and clamp the
   returned `alpha` to `1e-12` in the driver as an emergency fallback.

**Tuning the barrier for dynamic relaxation.** The ppf-style barrier is much stiffer than the
penalty contact. With the original `dhat=0.05`, `kappa=1e3`, `mu=0.5`, `friction_eps=1e-4` the
barrier case needed ~60 Newton iterations per step and ran for minutes. Practical settings for
this demo turned out to be `dhat=0.20`, `kappa=5e1`, `friction_eps=1e-1` — a much softer barrier
band and a larger friction smoothing width, which lets the dynamic-relaxation solve accept steps
in ~20 iterations. The reaction is not directly comparable to the penalty reaction because the
barrier acts at finite gap.

**Historical transient observation (not validation).** Early no-hold runs
sampled an oscillating state, so their reaction values cannot establish a
quasistatic or external match. They did reveal that excessive
mass-proportional damping can dominate the elastic reaction and that obstacle
extent must match the intended setup. The corrected research workflow derives
ramp and damping from the structural timescale, includes a hold, and requires a
settled-state gate.

The key operational rule: **for the barrier, run under dynamics (`solve_dynamics*`), keep the
contact nodes initially separated, and CCD-bound *every* position update including the predictor
and the moving obstacle.**

## 2026-06-29 — `RigidContact` compression of a soft ring collapses unless the load step is adaptive

Reproducing Abaqus `ring_compress.inp` exposed the difference between Abaqus's
surface-to-surface hard contact + automatic incrementation and CoupFE's node-to-obstacle
penalty (`RigidContact`) with fixed increments.

This is a historical development investigation. The public release does not
include the proprietary deck or a retained raw comparison run, so it supports
the adaptive-stepping lesson but not an Abaqus-agreement claim.

**The failure.** With a fixed top-plate displacement increment (e.g. 0.1 over 40 steps)
the soft neo-Hookean ring (G=2, K=20, r_inner=8, r_outer=10) collapses asymmetrically:
the top pole slides inward, only a handful of nodes stay active, and the total vertical
reaction drops to ~0. The deformation looks nothing like the symmetric flattened-ring
solution Abaqus produces.

**Why Abaqus doesn't collapse.** The `.sta` file shows automatic time incrementation
(ATT=2 cutback on the first increment) and **severe-discontinuity iterations** (SDIs).
Abaqus enforces hard contact as a constraint, and when the active set doesn't resolve
it cuts the increment. The small increments keep the Newton iteration in the basin of
the symmetric solution.

**The CoupFE fix.** `RigidContact` is a penalty method, not a constraint solver, but the
same increment-control idea works:
- Wrap `solve_increments` in an adaptive stepper that **cuts the load increment when
  Newton exceeds a tolerance** and **grows it when convergence is easy**.
- Use a penalty stiffness (`k=1e4`, `k_t=1e4`) that is stiff enough to keep penetration
  below ~5e-6 but soft enough that the return-map friction cap (`μ·k·|g|`) is meaningful.
  With `k=1e6` the penetration is ~1e-7, the friction cap per node is ~0.025, and the
  ring can slide along the plate instead of sticking.
- Do **not** enforce symmetry with extra y-axis BCs; let the adaptive stepping keep the
  solution symmetric. The only Dirichlet BCs should be the Abaqus anchors (nodes 1 and 6
  fixed in x).

**Sliding-step limitation.** `RigidContact`'s return-map friction tracks stick anchors in
the spatial frame, so a horizontally moving rigid obstacle does **not** generate tangential
friction. The workaround used in `examples/ring_compress/reproduce.py` is to identify the
nodes in contact with the top plate at the end of compression and prescribe the Abaqus
plate displacement (2 units) directly on those nodes. This is the stick-limited equivalent
of the Abaqus sliding step.

The development run recovered a symmetric branch with adaptive stepping, but
its numerical comparison values are not release evidence and must be
re-established from retained inputs and outputs before citation.

## 2026-06-28 — `ElementGroup` runs a coupled element natively only with UNIFORM dofs (every field on every node)

A core-architecture constraint worth stating plainly (surfaced building a mixed u-p element). The
`ElementGroup` assembler uses a uniform `comps` mask — **every field must live on every node**
(uniform dof/node). Consequences for mixed/coupled elements:

- **Taylor-Hood (e.g. pressure on corner nodes only) cannot be expressed** by the uniform-`comps`
  `ElementGroup` — non-uniform dofs/node. (It would need a per-node field mask the engine doesn't have.)
- **Equal-order P1-P1 (u and p both degree-1 → uniform dof/node) runs unchanged** on the existing
  mesh — and it is LBB-stable *here* because the near-incompressibility constraint `J−1−p/κ` carries a
  `p/κ` term that **is** Bochev-Dohrmann pressure stabilization (the stabilized equal-order pair, not a
  raw unstable one). So the engine's uniform-dof limitation and the stabilized-equal-order FE fact line
  up: the pair the engine *can* express is also the one that's stable.

This is distinct from the **element-local condensed-pressure** u-p (the `local_pressure` generator),
where `p` is a per-element internal variable with **zero global DOFs** — that sidesteps the uniform-dof
question entirely and is the route when you want a u-only global system.

## 2026-06-27 — Multiplicative-split stress must be w.r.t. the reference config, not the intermediate one

Adding the `J_inel` hook to `local_pressure` was almost right: the generator forms
`J_e = J / J_inel` and the pressure equation `p = K·avg(ln J_e)` is correct. The trap was in
`stress_PK1`: with `F = F_e·F_inel` and `ψ_R = J_inel·ψ_e(F_e)`, the reference PK1 is

    P = J_inel · P_e · F_inel⁻ᵀ

For isotropic `F_inel = J_inel^(1/d)·I` this reduces to `P = J_inel^((d-1)/d)·P_e` — `√J_inel`
in 2D, `J_inel^(2/3)` in 3D. Returning `P_e` alone is wrong by exactly that factor. It is easy to
miss because `F_e = F / J_inel^(1/d)` already divides by the stretch once; the same factor
reappears when mapping stress back to the reference volume.

The deeper lesson: **a zero-stress special state cannot verify stress scaling.** The free-
expansion gate (`F = F_inel`, `p ≈ 0`, `stress ≈ 0`) is blind to a multiplicative stress error
because `√J_inel·0 = 0`. So is a complex-step/FD consistency test, which only checks that the
code differentiates the *wrong* stress consistently with itself. A pressure-only oracle is also
blind. The correct gate is a **stress invariance / magnitude check**: compare two states with the
same `F_e` and `J_e` but different `J_inel` (one inelastic, one pure-mechanical), and assert the
residuals scale by the reference-mapping factor. That is the material-point invariance pattern
(`abaqus_ufl_lab/tests/material_point/harness.py`).

Scope reminder: the `J_inel` hook itself is intentionally limited to a prescribed/passed
inelastic Jacobian here. Driving `J_inel` by a coupling field (thermal concentration, growth
multiplier, etc.) is the next multiphysics step and stays a separate task.
## 2026-06-25 — N-body node-to-segment contact needs consistent edge winding; and: a slow small problem is a BUG, diagnose it

The distributed 16-disk pack would not converge (`rnorm ~ 1e3`, maxit every step, 285 s
or timeout) while the 8-disk pack converged in seconds. Teng kept pushing — *"why is it
so slow for such a small problem? something is wrong"* — and he was right twice over.

**The two wrong turns (mine).** I first blamed **F-bar** (the element), then a **singular
velocity field** (the load) — and I even *asserted* a velocity-field "fix" I never re-ran.
Both were wrong, and the lesson is process: **a 384-DOF dynamics problem taking minutes is
not a property, it is a bug.** The decisive diagnostic was boring and cheap — split the
residual at the **undeformed, zero-load state**: `|R_bulk| = 0` but `|R_contact| = 777`.
A non-zero contact force at the *rest* configuration is impossible for real physics, so the
bug was localized to the contact in one step — after I'd wasted two turns guessing.
The physics-first rule here means *isolate the operator*, not *guess which physics*.

**The bug.** `DeformableBarrierContact2D` is a node-to-segment barrier whose signed gap is
`d = (e×r)/L` — `> 0` when the secondary is on the **left** of edge `a→b`. It's a *two-body*
operator: the user orients the single primary surface. I used it for **N-body mutual
contact** (one operator over the union of all disk boundaries), and my `boundary_edges`
extracted free edges with **inconsistent winding**. A node near a wrongly-wound neighbour
edge gets a **negative** gap → the barrier reads a *deep penetration* → a huge spurious
force at rest → divergence. The 8-disk pack dodged it by luck (no triggering geometry); the
16-disk set's big R=7.2 disk (long edges, loose broad-phase AABB) triggered it.

**The fix.** Orient every body's boundary loop **outside-on-left** in `boundary_edges`
(`cross(b−a, midpoint−centroid) > 0`, else swap) — then `|R_contact|@rest = 0` and the
16-disk converges in **10 s** (was 285 s/divergent), rank-independent to **1.78e-15**. Plus
a defensive, opt-in `body_id=` on the operator (a node never contacts *any* edge of its own
body, not just the 1-ring incident ones; default-off, byte-identical to before — 32 contact
tests unchanged). Regression: `tests/test_multibody_contact.py` (separated ⇒ zero rest
force; a flipped-winding **broken control** that inflates the force; body_id self-exclusion).

**Fallout to correct:** the earlier "strong-scaling 1.14× / problem-too-small" claim came
from these **non-converged** runs (maxit every step), so it was measuring divergence, not
scaling — it's void; a real benchmark must be re-run now that the pack converges.

## 2026-06-25 — A moving rigid obstacle that bypasses the CCD = missing physics, not a tuning knob

Building the cylinder-compaction example (`examples/compression_cylinders`), the rigid lid would
**tunnel through** the pack and the "fix" was to lower it in steps `< dhat`. Teng pushed: *is that
missing physics or wrong physics?* It is **missing physics**, and the small-step rule is a
**breadcrumb** masking it.

CoupFE's non-penetration is a two-part guarantee: the **cubic barrier** repels within `[0,dhat)`, and
**CCD `max_step`** bounds the *Newton step* so a deforming body never crosses the obstacle. The
barrier alone does **not** restore from `gap<0` (it is zero outside `[0,dhat)`); penetration is
prevented *only* because the CCD never lets a step reach `gap<0`. A **moving rigid obstacle** (the
lid) is moved *externally* — I recreated the `HalfSpace` lower each increment — so its motion
**bypasses the CCD**. When it jumps `>dhat` it lands already-penetrated, in the dead zone where the
barrier gives no force ⇒ it passes straight through. `skills/contact.md` already states the rule
("any position update that bypasses the line search — predictor, warm-start, BC ramp — must go
through the same CCD bound, or it tunnels"); **a moving obstacle is simply the 4th case**, and the
**Hertz indenter used the same recreate-the-obstacle workaround**.

- **The breadcrumb** = "steps `< dhat`". It works only because each step stays inside the barrier
  band; it is not a guarantee and it caps how fast a displacement-controlled platen can move.
- **The proper fix** = give the rigid obstacle a velocity/per-step displacement and fold its motion
  into the gap **and** `max_step` (bound the *relative* body-obstacle approach over the step) — then
  a displacement-controlled lid/indenter/platen is penetration-free at any step size. This is a real
  capability gap (displacement-controlled rigid tooling is ubiquitous: indentation, compaction,
  forming). Tracked as the next contact item.

**The other "bug" was NOT missing physics — it was my setup.** The gravity that "crushed" the disks
was an *arbitrary nodal mass = 1 + hand-tuned body force*, i.e. **wrong units**, not an engine flaw.
`ConstForce`/gravity is correct; the fix is consistent units (real density → lumped mass →
`force = mass·g`; here `εg = ρgL/G ≈ 6.5e-5`, so real gravity is gentle and just seats the pack).
Classifying the two correctly matters: one is a feature to build (moving-obstacle CCD), the other is
a units discipline.
## 2026-06-24 — In the AI era, jump straight to the best option *when you understand the problem well*

A recurring pattern this session, and Teng's explicit call: when we genuinely understand a problem —
the algorithm is known and we have a trusted reference — **skip the prototype and implement the best
version directly.** The cost of a throwaway intermediate (write it the easy way, get it "working," then
rewrite it for the real target) is mostly wasted in the AI era, where writing the real version is not
the bottleneck. Concrete instances that paid off:

- **numba-native from the start**, not pure-Python-then-port. A pure-Python kernel "works" but lives in
  a different env (numba's nopython subset rejects `einsum`/`column_stack`/closures), so it would just
  be rewritten. We wrote the @njit kernels directly → 284×, bit-exact, no rewrite.
- **BVH ported from ppf's LBVH template directly** (node layout + query), not a hand-rolled "simple"
  tree first. The simple version would diverge from ppf's structure and block the GPU escalation.
- **edge-edge + ACCD ported straight from the ppf-faithful numpy** to numba — no intermediate.

The hard precondition — and the thing that makes this NOT reckless — is **genuine understanding,
verified against the reference first.** Before porting edge-edge/ACCD I re-read ppf's
`distance.hpp`/`accd.hpp` and confirmed our numpy was an exact port (and that ppf does *no* edge-edge
mollifier). "Jump to the best option" only works *after* "check the reference." Jumping without that is
the opposite trap — building the wrong thing fast. So the rule is two-sided:
**understand + verify the reference,
THEN implement the target directly** — and keep the bit-for-bit gate (numpy oracle) so "direct" never
means "unchecked."

## 2026-06-24 — numba is the right accelerator for contact (284× bit-exact); BVH from ppf's template

Profiling (measured, not guessed) put the dominant contact cost in the **Python per-pair barrier loop**:
~80 µs/pair, of which >99% is interpreter + numpy-dispatch overhead (the actual arithmetic is tens of
flops on 3-vectors + small matrices). Three durable findings:

- **numba, not Fortran, for contact.** The dual-home (run-inside-Abaqus) rationale that justifies
  fixed-format Fortran for *element* kernels does NOT apply to contact — contact only ever runs in
  CoupFE standalone. The kernels are already numpy → `@njit` compiles them with near-zero rewrite,
  stays in Python (prange / `numba.cuda` later), and is far more maintainable. Measured: the vertex-face
  cubic-barrier + ppf-smoothed-friction kernel went **77.6 → 0.27 µs/pair = 284×**, and **bit-for-bit
  vs the numpy oracle** (max rel R 6.6e-17 / K 4.0e-17 with friction+mass; R exactly 0.0 at mu=0). The
  pattern: **numpy stays the bit-exact ORACLE; numba is the production path; gate bit-for-bit.**
- **Write it numba-native from the start — don't prototype in pure Python then re-port.** numba rejects
  / mis-compiles `einsum`/`column_stack`/`np.outer`-style helpers, so the @njit kernels are explicit
  scalar arithmetic. A pure-Python prototype would just have to be rewritten for "the new env" (numba's
  nopython subset). Teng's call, and right.
- **BVH broad-phase ported from ppf's LBVH template** (`bvh_numba.py`), replacing the uniform grid
  (which degrades on non-uniform meshes). Build = Morton-sorted **iterative** midpoint split; node
  layout + stack query are ppf-compatible (so the GPU Karras build is a direct escalation). Gated vs
  brute force to **exactly** the AABB-overlap set (miss=0, extra=0) across uniform / flat-sheet /
  non-uniform-clustered / duplicate / degenerate / single / empty configs.

Two traps hit and worth remembering:
- **A recursive `@njit` + `cache=True` can stale-SEGFAULT** after a signature change (numba loads a
  stale compiled binary). Fix: clear `__pycache__`, AND prefer an **iterative** build (work-stack) over
  recursion for cached numba — no recursive-caching fragility. (The BVH build is iterative for this.)
- **A tightened broad-phase exposed a LATENT CCD bug** the loose grid had been masking:
  `max_step` queried candidates with band `dhat`, but CCD must
  catch a face a node could *sweep through* over the step even if it's > `dhat` at the current config.
  Fix: the CCD broad-phase band is **`dhat + 2·reach`** (swept motion), while the barrier broad-phase
  stays `dhat` (current config). The grid's accidental looseness had compensated; tightening made the
  band explicit. Also: **`dhat` must scale with element size** — a fixed `dhat` on a fine mesh inflates
  the candidate count (it's a parameter pathology, not a broad-phase failure).

Cost-structure note vs ppf: ppf pairs a *loose* box-overlap broad-phase (BVH) with a *cheap GPU*
narrow-phase (1 thread/pair); we tightened the broad-phase (AABB-distance prune, ~55→~20 cand/vertex)
because our narrow-phase, even at 0.27 µs/pair, is CPU — spend a little more pruning to save narrow-phase
calls. Same conservative-superset spirit, opposite balance (our per-pair cost is opposite). `docs/dev/
contact.md` + `contact_experiments.md`.

## 2026-06-24 — Smoothed (ppf/IPC) friction: tune `friction_eps` to the slip scale, not arbitrarily small

The end-to-end 3D distributed friction demo (two Hex8 blocks sheared) wouldn't converge — the Newton
solve stalled at `|R| ~ 1e-4` (60 iters) even though the friction was clearly *working* (it reduced
interface slip). The cause is specific to the smoothed-friction **Gauss-Newton tangent** and only
shows up under a real multi-iteration solve (the unit friction tests call the kernel directly, so they
never saw it). The smoothed law is `λ = μ λ_n / max(ε, ‖P·dx‖)` with friction force `λ(P·dx)` and the
ppf tangent **drops `dλ`** (one of the three deliberate ppf approximations — Gauss-Newton, PSD, free
robustness). That dropped term matters differently in two regimes:

- **Moderate slip-per-step (`ut` a few × `ε`):** `λ = μλ_n/ut`, so the dropped `dλ·(P·dx)` is
  *comparable* to the kept `λP` term and injects a spurious flow-direction stiffness → Newton converges
  only **linearly to a residual floor**. This is the awkward band — *and it is a MODERATE-slip
  phenomenon, not a large-slip one*: at *much* larger `ut` the whole friction tangent `~μλ_n/ut → 0`,
  friction becomes a near-**constant** force, and it converges fine again.
- **Stick (`ut < ε`):** `λ = μλ_n/ε` is constant in `ut` → `dλ ≈ 0` → the tangent is **near-exact** →
  tight (quadratic-ish) convergence.

So the default tiny `friction_eps=1e-4` put a `~1.5e-3` interface slip deep in the plateau → stall.
Setting `friction_eps = 2e-3` (just **above** the slip) moved it to near-stick → `4` iters, `|R|=6e-9`.
Larger `ε` softens friction (stick spring), so restore holding strength with `μ` — but `μ` has its own
ceiling: too-stiff `μλ_n/ε` re-stalls the solve (`μ=1.0` here) for no real gain (the held-ratio had
already hit its **elastic floor** — even fully stuck, the secondary rides the *other* block's elastic
shear). The convergent operating point is `ε ≳ ut` (near-stick) + `μ` for strength. Rule: **pick
`friction_eps` from the expected interface slip**, not as a "make it sharp" knob — the tangent's
*consistency*, not the physics, is what's failing. `docs/dev/contact_experiments.md` (2026-06-24).

**The deeper tension (convergence ↔ accuracy), and why ppf trades it the way it does.** This is not a
bug to fix — it is the *defining* trade of the ppf/IPC smoothed model, and it is a *good* trade for the
right reasons:
- ppf **PSD-projects every Hessian** (barrier + friction) → every Newton step is a guaranteed descent
  direction → the solve is **unconditionally robust with NO energy-merit line search** (IPC needs
  `energy()` on every operator for its Armijo filter — a heavy contract ppf *abandons*). Robustness
  over exactness: a solver that must never fail on millions of contacts prefers "always a safe step."
- The accuracy it gives up is mostly in the **tangent, not the residual.** The barrier/friction
  **forces (residual) are exact**; the dropped `dλ` only affects the *path/rate* of convergence (and,
  if it converges, *not* the fixed point). The only true residual approximations — the lagged `λ_n,n`
  and the mollified cone `max(ε,·)` — are **tunable and vanishing** (`→0` as `ε→0` and iterations→∞).
  So "accuracy sacrifice" = a dial, not a fixed error. The catch is precisely our tension: dialing
  accuracy (small `ε`, plateau) costs convergence.
- ppf spends its accuracy budget **where the physics is a hard constraint** (closest-feature geometry
  + CCD non-penetration are *exact*, and non-penetration is *guaranteed*, unlike penalty's
  stiffness-dependent overlap) and **economizes where the physics is already empirical** (the exact
  stick-slip transition of a Coulomb law — itself an approximation of reality). Regularizing an
  already-approximate friction law by a small `ε` is a far cheaper "accuracy loss" than getting the
  geometry or non-penetration wrong.
- We get a *better* version of the trade than ppf: ppf runs **single precision on GPU** (throughput);
  we ported the same geometry/barrier/friction to **double-precision CPU + a direct solver**, so we pay
  none of the float32 cost — our only sacrifice is the friction *tangent* (the moderate-slip convergence
  band) and the regularized stick.

**Fair calibration (corrected after discussion — beware the "limitation" reflex).**
I first wrote *"we don't have robust large-slip kinetic
friction"* — **that is wrong, and it collapsed under one push.** The smoothed residual reproduces
`f = μN` at **all** slip magnitudes (`|f| = μλ_n` for `ut > ε`, opposing the slip direction), and large
*steady* slip is the model's **easy** regime (`tangent ~ μλ_n/ut → 0` → friction → a near-constant
force). The genuine, narrow costs are **not** about large slip: (1) **stick is a regularized creep**
(`~ε`), not exact lock — a *stick*-side issue; (2) the **moderate slip-per-step convergence band** above
(tunable via `ε`/sub-step/damping); (3) an `O(dt)` **direction lag** for *turning* slip (straight
sliding is exact); (4) it cannot represent **rate-weakening** stick-slip instability — but neither can
constant-`μ` Coulomb, so that is a *model* choice, not a smoothing defect. **Prefer the return-map
(`RigidContact`, exact stick, non-symmetric) when you need exact static stick / sharp transitions / no
cyclic creep — NOT for large slip per se.** (A richer law — rate-and-state / Stribeck — is the orthogonal
move when the *physics* of large-slip friction is rate-dependent; it also regularizes the tangent for
real, turning the numerical `ε` into a model.)

## 2026-06-23 — Check the physics (element formulation + dimensionless numbers) BEFORE the solver

The 3D two-block dynamics demo wouldn't converge, and I spent a long detour on the *solver* — PSD-
projecting the bulk Hessian, a residual line search, a max_dx cap, the ppf recipe — before Teng
redirected: *"check the physics first; what element do you use? we don't need a 200× difference."* He
was right on every count, and the actual causes were all physics:

1. **Element formulation.** I'd generated a **standard (full-integration) Hex8**, which
   **volumetrically locks** (artificially over-stiff). The fix is an **F-bar Hex8**
   (`generate_element(..., element='hex8', formulation='fbar_mechanics')`, FD-tangent-verified). The
   2D demo converged because its Quad4 was already F-bar. *A standard low-order element under a
   nearly-/moderately-incompressible finite-strain material is a locking trap — use F-bar (or
   local-pressure).*
2. **Dimensionless gravity.** The gravitational strain `εg = ρ g L / G` was **2** (`GRAV=2,
   L=ρ=G=1`) — the soft block was *crushed* (≈37% strain), which has no converged static equilibrium.
   The decisive evidence: **even serial `solve_dynamics` with a line search FAILED at εg=2 for BOTH
   κ=2000 and κ=50** (`|R|≈0.45`, 1000s of iters). Setting `εg ≲ 0.2–0.4` → converges in 2–3 Newton
   iters. My PSD/line-search machinery was treating a symptom of a non-physical load.
3. **Stiffness ratio.** `κ/G = 2000` (the "200×") over-stiffens and ill-conditions; match it,
   `κ ~ K_bulk`. And the adaptive `s = κ + M/gap²` **over-repels** at a small gap (use a fixed κ for a
   gentle rest; reserve adaptive `s` for hard impacts).
4. **Setup.** A gravity-*balanced* resting contact is finicky in 3D (strong load crushes, gentle load
   lets the block oscillate out of the band). A **collision via initial velocity** (no gravity) is
   robust — and the CCD guarantees penetration-free every step by construction, so no sustained load
   is even needed; track the *minimum gap over the trajectory* as the honest evidence.

The lesson: when a contact-dynamics solve won't converge, **diagnose the physics first** — the
element formulation (locking?) and the dimensionless groups (`εg`, `κ/G`, impact `v/c`) — before
reaching for PSD-projection / line-search / trust-region solver machinery. A non-physical
configuration (a crushing load, a 200× stiffness, a locking element) has no good numerical fix.
This matches the lab's "matched κ ~ bulk stiffness" lesson.

## 2026-06-23 — Don't build a degenerate test to dodge a missing piece; verify the new primitive in isolation

Bringing contact to 3D distributed, the new cross-rank machinery was cheap — `_DistDeformableContact3D`
*subclasses* the 2D helper and overrides only `__init__`, because the substantive methods
(`surface_U`/`assemble_into`/`ccd_alpha`/`commit`) are op-agnostic (good factoring paying off again).
The one genuinely-new idea: edge-edge has **no natural "secondary"** to partition by (2D vertex-face
does), so I added an `owns_edge_pair` predicate on the first node of each pair's first edge — applied
in **both** `_contributions` *and* `max_step` so the per-rank CCD min reduces to the correct global
min. All of it verified rank-independent to machine precision.

The real lesson was the *test* I almost shipped. To exercise the full 3D distributed **dynamics solve**
I lacked a 3D bulk (Hex8/Tet4) element, so I made the "bodies" pure mass points dropped on a floor —
**no bulk**. That config is *degenerate*: the system matrix is mostly-diagonal inertia + tiny contact
off-diagonals, which (a) **both** direct solvers (superlu_dist *and* mumps) fail to factor at first
barrier engagement (zero pivot → `du=inf`), and (b) **intermittently SEGVs** PETSc (~1-in-4) on the
empty/size-0 objects. I chased it for a while (iterative solver helped; a ghost-scatter guard fixed
`-n≥2`) before recognizing these are artifacts of the **missing element**, not the contact code — the
*serial* `solve_dynamics` of the identical config works fine.

The right move (and the one I'd already written down this session): **verify the new distributed
PRIMITIVE in isolation** — a solver-free static cross-rank assembly check (R, K·v, and the global CCD
vs the serial full operator, rank-independent to `0.0`) — and **defer** the end-to-end dynamics solve
to when the real element exists (then the system is well-conditioned, as in 2D, and the direct solver
just works). A flaky test of a degenerate stand-in is worse than no test: it neither exercises the
real path nor stays green. Don't contort the *setup* to avoid a missing dependency; test what you
actually have, and name the dependency. The iterative-solver/SEGV-guard band-aids were the
breadcrumb pointing at "this configuration is the wrong test.")

## 2026-06-23 — Path-dependent state is rank-independent for free once U is; good factoring makes the next feature "no new code"

After distributed dynamics landed, distributed deformable **friction** turned out to need **zero new
MPI code**. Two reasons, both worth internalizing:

1. **Factor the cross-rank machinery once.** The shared `_DistDeformableContact` helper (surface
   replication + owned-secondary COO + off-process `ADD_VALUES` + global-CCD `Vec.min`) already
   threaded `mu`/`friction_eps` into each rank's owned-secondary `DeformableBarrierContact2D` and
   advanced the step-start friction reference `_x0` in `commit`. So enabling friction was passing a
   parameter, not writing a distributed assembler. Factoring the genuinely-hard part into one place
   (used by both the quasistatic and dynamics drivers) is what made the follow-on free — and it
   avoids the divergence risk of a second copy.

2. **Rank-independence of path-dependent state follows from rank-independence of U.** Friction is
   stateful (`_x0` accumulates over steps), which *sounds* like it could differ across partitions.
   It can't, here: the per-step displacement U is rank-independent to machine precision (the dynamics
   1-vs-N), and `_x0 = X + U` at each commit, so the friction reference evolves identically on every
   rank → the whole path-dependent trajectory is rank-independent (`max|u_N−u_1| ~ 5e-16`). This is
   the same argument as the distributed *rigid* friction. The general principle: a path-dependent
   internal variable that is a deterministic function of the (rank-independent) solution inherits its
   rank-independence — you don't need a separate cross-rank reconciliation for the state, only for
   the solve. The verification still earns it: gate the *state-bearing* run with 1-vs-N, not just a
   stateless one.

So the full deformable-contact stack — penalty, barrier, friction — now runs distributed, and the
friction step was the cheapest of the three precisely because the first two were factored well.

## 2026-06-23 — Check the reference implementation before building; the residual-norm line search is the wrong merit for non-smooth contact

Before building the "barrier energy-merit line search" I'd proposed to unblock the distributed
barrier solve, Teng said: check ppf-contact-solver first. Reading its Newton loop settled it —
**ppf has no energy-merit (Armijo) line search at all.** It avoids needing one with the IPC
projected-Newton recipe: PSD-projected Hessian (analytic eigensystems → guaranteed descent
direction), a max-displacement step cap (trust region, not Armijo), a CCD filter, and an
incremental-potential (**dynamics**) formulation that makes the local energy strongly convex; then
it takes the step *unconditionally*. So the thing I was about to build didn't need to exist.

Then the decisive experiments (logged in `docs/dev/contact_experiments.md`): I added a PSD
(Gauss-Newton) tangent option and tried to converge the quasistatic deformable barrier three ways —
PSD + residual-norm line search, the full ppf recipe **without** dynamics, and force-driven. **All
three stalled.** The lesson in two parts:

1. **The residual-norm line search is the wrong merit for non-smooth contact.** A PSD tangent gives
   an energy-*descent* direction, but `‖R‖` is not monotone along it at the node-to-segment
   projection flip, so a `‖R‖`-decrease backtracking shrinks the step to zero and stalls. Don't gate
   the step on `‖R‖`. The right merits are the *energy* (IPC filter line search) or *nothing* (ppf:
   PSD + step cap + dynamics).
2. **Dynamics is the substrate, not a fallback.** The quasistatic deformable barrier converges only
   with dynamics or a true energy-Armijo search; our serial `solve_dynamics` already converges it,
   and ppf is dynamic for exactly this reason. So the distributed barrier's path is **distributed
   dynamics** (reuse the working serial dynamics + the cross-rank assembly + global-CCD reduction),
   not forcing quasistatic convergence — which also matches our own earlier "dynamics is the
   substrate for robust contact" conclusion.

Meta: this is the project's "PORT, don't re-derive" discipline applied to the
*solver strategy*, not just kernels — read how the proven system does it
before writing a new mechanism. And: when a fix "should obviously work" (PSD ⇒ descent ⇒ converges),
run the experiment — the missing piece (the merit function / dynamics) only showed up by trying it.

## 2026-06-22 — "Representative coverage" can mean "least-verified coverage": the dedup trap

Reviewing the Phase-10 deduplicated ports — 3 elements chosen as *representatives* covering NEW API axes
(2.7 Li battery = 5-field max; 2.10 Chester-Anand local-pressure = static **condensation** ABI edge;
3.4 axisymmetric gel = axisymmetric **hoop** kinematics). The forms `verify()` (complex-step vs FD),
they generate + compile, no regression. But the gate strength is *inverted* against the risk: the
exact axis each element was picked to cover is the **least** verified, because the independent oracle
(`reference_assembly`, the harness's teeth) doesn't generalize to that new axis:
- **Li (5-field):** `test_native_vs_reference_assembly` is **xfailed** ("indexing not generalized to
  5 fields"). So the only independent check is the native↔UEL **sign** gate — but both backends come
  from the *same* weak form, so that's code-vs-itself for *physics*; Li's physics is `verify()`-only.
- **Chester (condensation):** intrinsically **UEL-only** (needs Abaqus `UVARM` → not f2py-runnable),
  so native/oracle gates legitimately can't apply — but the **condensation logic lives in
  `generate_uel_local_pressure`, which `verify()` never exercises**. Net: the condensed R/K is gated
  only by "it compiles." The ABI edge it was chosen to cover is untested.
- **Axisym (hoop):** the hoop strain `F(3,3)` + hoop stress + **hand-derived hoop tangent blocks**
  are spliced into the generated Fortran by a *text post-processor* (`_make_axisymmetric`) — AFTER
  codegen, so the complex-step **auto-tangent guarantee (the whole point of the system) does NOT cover
  them**. Gated only by smoke (finite RHS/AMATRX + radial-residual≠0). A wrong sign / missing `r_gp`
  weight / wrong tangent block passes silently.

This is the consistency-not-correctness and operator-level-gate lesson exactly — and
it is structural, not sloppiness: when
you **dedup a zoo to representatives by API-axis novelty, you select for the cases your oracle can't
reach**. Fixes are cheap and element-agnostic: (1) an **FD tangent-consistency** check
(`AMATRX ≈ -dRHS/dU` by finite difference) gates *any* generated/post-processed UEL incl. hand-written
tangents — the single highest-value missing gate for the axisym element; (2) **byte/structural compare
to the validated lab `.for`** gives a real oracle for UEL-only elements (Chester) without running them;
(3) **extend `reference_assembly` to the new axis** (n-field indexing) so the representative actually
gets the harness it was meant to stress. Lesson: budget oracle-extension *per new axis* when dedup'ing,
or the representative ships consistency-gated only. Corollary: a **core generator change**
(`has_state` detection, `LocalScalar` overlap) motivated by an element with **no R/K correctness gate**
(Chester) must be regression-verified via the **full suite + the canonical stateful elements'
round-trip/oracle** (J2/LCE/Hussein), NOT via the motivating element.

## 2026-06-22 — Verify the new distributed PRIMITIVE in isolation; don't let an orthogonal serial bug mask it

Building distributed deformable contact, the **penalty** version went end-to-end cleanly: cross-rank
assembly (surface replication + owned-secondary COO with global dofs + PETSc off-process
`ADD_VALUES`), verified rank-independent both at the assembly level (R exact, K·v ~1e-13) and as a
**full two-block solve** against an *independent* serial oracle (operator-level
`solve_increments([ElementGroup, DeformableContact2D])` — a different assembly path, not
`solve_distributed` grading itself). The genuinely-new distributed primitive for the **barrier**
(penetration-free) version is the **global CCD step bound**: each rank's point-edge `max_step` is
local, but the limiting pair can be on any rank, so the Newton step bound is a *collective minimum*
(petsc4py-only via a 1-entry-per-rank `Vec.min()` — never mpi4py).

The barrier *full solve* didn't converge — and the instinct is "the distributed code is wrong." It
isn't: I traced the Newton residual and **it stalls identically at one rank (serial)**. The cause is
a residual-norm line search collapsing `ls_α→0` at the node-to-segment **projection flip** (the
closest edge / ξ changes between iterations → the frozen-projection "Newton" direction transiently
*increases* `‖R‖`, so a monotone-decrease line search rejects every step). That is a **serial
contact-solver** issue (fix = energy-merit, CCD-filtered line search — barrier contact is energy
minimization, where `‖R‖` is non-monotone along the step but the energy is monotone), **orthogonal to
MPI**. The lesson: when a new distributed feature rides on a finicky serial kernel, **verify the new
primitive directly** (here: the global CCD min equals the serial full-surface `max_step`, `|diff|=0`,
< 1 so it bit — rank-independent at 1/2/4) instead of burying it inside a full solve whose failure
has an unrelated, serial cause. Decoupling the verification (a) ships a passing, honest artifact now,
and (b) localizes the real fix to where it belongs (the serial line search). When that lands, the
distributed barrier solve works with **zero additional MPI code**. Profile or trace
*which layer* fails before "fixing" the wrong one.

## 2026-06-22 — Know the scaling model: ppf is single-GPU (not MPI); contact pairs are embarrassingly parallel

Checked ppf's parallelism before assuming: it is **single-GPU CUDA** — "both contact and elasticity
on the GPU", **180M contacts on one GPU**, single precision, **LBVH** broad-phase; **no MPI / NCCL /
multi-GPU** anywhere. So ppf scales by *thread parallelism on one device*, while CoupFE scales by
**MPI** across ranks/nodes (the `abaqus_ufl` heritage). These are **different, complementary axes**:
MPI when the *mesh* exceeds one node (but then contact's spatial proximity couples ranks — the hard
distributed broad-phase); GPU when the *contact count* is the wall (no cross-rank problem — it's all
in one device's memory). Don't conflate "parallel" — name the axis.

**Forward path (for future agents/improvement).** The contact **per-pair kernels** (distance
coefficients, barrier residual/tangent, ACCD) are *embarrassingly parallel* — each candidate pair is
independent (the numpy versions are literally per-pair loops). So: build correctness in **numpy
first** (it's the verified oracle + already O(N) with the grid broad-phase, fine for serial/modest
research = most use), then port the *same per-pair math* for speed — **Fortran (f2py, dual-home,
double precision)** or **GPU (CUDA, ppf-style: one thread/pair + LBVH; that's literally ppf, so we'd
port its kernels directly)**. The **operator contract** stays the serial O(ndof) spine and does not
change; only the kernel implementation moves, gated bit-for-bit against the numpy oracle. Correctness
first, speed second — don't write the Fortran/GPU version until scale demands it.

## 2026-06-22 — Don't re-derive battle-tested geometry; PORT it (the from-scratch version missed cases)

For 3D deformable contact I started hand-rolling the point-triangle kernel — a plane-distance gap +
a barycentric foot test, marking *foot-outside* as inactive. Teng stopped it: *"check the
ppf-contact-solver; that code has been tested by many; we don't need to write everything from
scratch."* He was right, and concretely so: the hand-rolled version was **wrong** — a 3D
point-triangle contact does **not** require the foot inside the face; when the foot leaves the
triangle the closest feature is an *edge or a vertex*, which my "inactive if foot outside" logic
silently dropped (missed contact → penetration). ppf's `contact/distance.hpp` does the full
closest-feature **classification** (`point_triangle_distance_coeff_unclassified` falls back to
point-edge → point-vertex; `edge_edge` likewise), and `accd.hpp` does additive CCD — both reviewed
and exercised on huge problems. Porting them faithfully to numpy (verified against an independent
brute-force closest-distance oracle, *incl.* the degenerate regions) is **faster and more correct**
than deriving from scratch.

**Lesson: AI is good at writing code, but "can write it" ≠ "should re-derive it."** For
*error-prone, well-trodden* kernels — geometric predicates, distance/closest-feature, CCD, robust
orientation, quadrature, special functions — find the battle-tested implementation, port the
*algorithm* (not the stack), and verify the port against an independent oracle. Re-derivation is for
the genuinely *new* algorithm (our barrier+dynamics+friction *integration*, the operator contract);
the standard sub-routines underneath it are a port. The tell that you're in port-territory: the math
has known degenerate cases a fresh derivation will quietly get wrong (it did).

## 2026-06-22 — Broad-phase with band = d̂ is a *transparent* O(N²)→O(N) swap

The contact closest-edge search was brute-force O(N²) (every vertex × every edge). The spatial-hash
broad phase (`contact_search.candidate_pairs`) replaces it, and the key realization is that with the
search band set to the contact band **`d̂`**, the swap is *provably identical*, not approximate:
an **active** node-segment pair has point-segment distance `< d̂` (active ⇔ line gap `< d̂` with the
foot on the segment ⇔ point-segment distance `< d̂`), so it's in the band-`d̂` candidate *superset*
⇒ same closest edge ⇒ same force; and a vertex with **no** in-band edge contributes nothing either
way. So the active set and forces are byte-identical (the deformable barrier + friction suites pass
unchanged) — we got the O(N) win for free on the hot residual/tangent path, with the brute-force as
the correctness oracle. The subtlety that does *not* transfer: **`max_step`/CCD needs a band `> d̂`**
(it must catch an edge the vertex could *cross this step*, not just one currently within `d̂`), so it
can't reuse the band-`d̂` set verbatim — left brute-force for now, wired with a reach margin later.
Uniform grid (not BVH) first: FE features are roughly uniform, so a grid is simpler and O(N); BVH is
the escalation for non-uniform meshes.

## 2026-06-22 — Check what actually runs distributed before calling it a gap

Asked "can contact-friction run under MPI?", the honest first answer was "no, it's serial." True for
the *new ppf smoothed/barrier* friction — but Teng pushed: *the previous friction model can work
with MPI, can you check?* He was right. `solve_distributed(contact=…)` runs the **penalty return-map
Coulomb friction** (`rigid_penalty_eval`) node-local, and running it confirmed **rank-independence to
1.6e-16** (1-rank vs 2-rank). So MPI friction *exists* — it's just the older penalty model, not the
barrier one. The real shape of the gap: **two contact stacks have diverged** — a rich *serial*
operator stack (barrier/deformable/dynamics/smoothed-friction) and a narrow *distributed* stack
(rigid penalty + return-map friction, quasistatic, node-local). Before stating a
capability gap, *run the thing* — a
"serial-only" claim was half-wrong, and the precise version ("the **smoothed** friction is serial;
the **penalty** friction is distributed + verified") is far more useful for planning.

## 2026-06-23 — The verification harness must not be a second assembler (physics ≠ numerics)

The Li 2026 5-field port (`u, phi, c, T, d`) failed with an `IndexError` in
`reference_assembly`, originally labelled "indexing not generalized to 5 fields." Both the
label and my first instinct (patch the table) treated the symptom. Two corrections, both
worth keeping:

1. **It was a coupling *pattern*, not a field count.** The crash was a value-assembled source
   term (the `T` equation's `pressure_storage`) depending on the **gradient of another scalar
   field** (`grad_phi`, `grad_c`) — the `value × grad`, `wrt_kind='vector'` cell of the tangent
   table, which only had the `matrix` (F = grad u) version. "Doesn't work at N fields" is almost
   always shorthand for "has a term shape the code never met." The fix used the *same* chain rule
   as F = grad u (`∂grad_s[l]/∂s[b] = dsh[b,l]`; the column is the scalar DOF `s[b]`) — `grad_s`
   is a derived quantity, never an independent variable.

2. **The real problem is architectural, and it's in the harness, not the engine.** The weak form
   and the generator were *correct* — the generated Fortran's R and K matched once the oracle was
   fixed. The generator differentiates generically (complex-step over DOFs), exactly as FFCx does
   for a UFL form. The gap was only in `reference_assembly`, which is a *second, hand-written
   assembler* that re-encodes every coupling as a case table — so it falls behind real
   multiphysics. **We mixed physics into numerics in the harness.** FEniCSx has no such limit
   because it has *one* generic assembler; we already have that genericity in the generator and
   accidentally dropped it in the verification layer.

The direction (docs/dev/verification_harness_redesign.md): keep the weak form as the source of
truth + generic auto-diff generation; verify with a **generic FD/CS tangent-consistency check
`K ≈ ∂R/∂U`** (zero coupling knowledge, any field count — and the *only* gate that catches a
**hand-written** tangent, e.g. the axisym gel's post-processed hoop blocks), a **residual-only
reference + patch/MMS** for R-physics, and **operator-level gates** for well-posedness/authoring
guidance. Interim, the oracle now **raises** on an unhandled pattern instead of silently dropping
a block (a wrong K) — incompleteness made self-reporting. Gate on properties,
not a parallel re-derivation; this also matches the
"representative coverage = least-verified coverage" lesson.

## 2026-06-21 — CCD must bound the PREDICTOR, not just the Newton step

Deformable–deformable barrier contact (`DeformableBarrierContact2D`) was penetration-free
in every *static* probe — `max_step` correctly capped a near-crossing step in isolation —
yet a dropped block sank one full `d̂` band into the floor under `solve_dynamics`. The CCD
was only applied to the Newton **increment**; `solve_dynamics` then overwrote `U` with the
**unbounded inertial predictor** `û = u_prev + dt·v_prev` *before* Newton. Once `dt·v` grew
past the gap, the predictor *teleported* the node through the barrier band, Newton started
already-penetrated, and the closest-edge normal flipped → no recovery. Fix: CCD-bound the
predictor jump from the last accepted state with the same `max_step` (a no-op when
separated). **Lesson: any position update that bypasses the line search — predictor,
warm-start, BC ramp — must go through the same CCD bound, or it tunnels.** (IPC bounds every
position update for exactly this reason.) Bonus: with the predictor bounded, a too-soft
*fixed* κ no longer penetrates a hard impact at all — so the value of adaptive `M/d²` is
*capacity/conditioning* (without it the solve pins at gap≈0 and thrashes ~18× more Newton
iterations), **not** non-penetration. Re-derived the adaptive-stiffness test around that.

## 2026-06-21 — A deforming surface needs a surface-aware penetration oracle

The two-block test first "failed" with a −1.4 penetration that did not exist: the oracle
compared each top-block node's `y` to the **global max** of the strip-top `y`. But the
strip top *deforms* — it dips in the middle under the contact load — so a middle node
resting correctly on the dipped surface read as below the un-dipped corners. The operator's
own per-pair signed gaps were all **positive**. Fix: the independent oracle must evaluate
the **deformed surface beneath each node** (`np.interp` over the deformed strip-top nodes),
not a flat/extremal proxy. When both sides move, "penetration" is only meaningful against
the *current* opposing surface — a rigid-floor oracle silently lies once the floor bends.

## 2026-06-21 — A node-to-line gap has an analytic, action-reaction gradient

For the deformable barrier, the signed gap `d = (e×r)/L` (`e=xb−xa`, `r=xs−xa`) to the edge
**line** is independent of the foot parameter ξ — the perpendicular distance to a line does
not depend on where the foot falls; ξ only gates the active region and is *not* needed for
the force. Its analytic gradient `∂d/∂[xs,xa,xb]` has `∂d/∂xs = n` exactly and the three
nodal gradients **sum to zero** (translation invariance), which makes the assembled barrier
force action-reaction (Σforce = 0, Στorque = 0) *by construction* — verified, not assumed.
Building the force from that analytic gradient (rather than a hand-split `(1−ξ, ξ)` like the
penalty) keeps it complex-step-differentiable for the **consistent** tangent, which came out
symmetric to machine precision (it *is* an energy Hessian) and matched FD.

## 2026-06-21 (later) — Adaptive κ is a capacity-vs-conditioning problem; dynamics is the fix

Tried Ando-style adaptive barrier stiffness. Naive version (`κ = c·k_bulk/d̂`, set once) made
convergence **worse**: it matched *stiffness* (conditioning) but the cubic's max force `κd̂²`
fell *below* the load, so the gap collapsed. **Stiffness-matching and load-capacity are two
different constraints.** Reverted it (kept the 5 barrier tests green).

Reading the ppf source (`barrier/barrier.cu::compute_stiffness`) showed the real recipe: the
geometry-normalized cubic barrier is scaled per-contact by `stiff_k = wᵀ(K_elast + M/g²)w` —
elasticity gives conditioning, and an **inertia term `M/g²` (→∞ as the gap closes, recomputed
each eval)** gives capacity. That `M/g²` is what my frozen κ lacked — **and it is inertial**,
so in a *quasistatic* solve (no mass) the capacity bound is unavoidable. The capacity pain is a
**quasistatic artifact**, not a barrier flaw. (Full analysis: `docs/dev/ppf_contact_analysis.md`.)

**Conclusion — make implicit dynamics the substrate for robust contact.** Inertia supplies the
barrier capacity *and* regularizes the non-smooth stick/slip + active-set transitions that stall
a quasistatic Newton (this is why ppf/IPC are dynamic). Added `InertiaOperator` (lumped mass,
backward-Euler `M/dt²(u−û)`, `commit` advances `v=(u−u_prev)/dt`) + a `solve_dynamics` driver —
the element/contact/load operators are driver-agnostic and **compose unchanged**; the only new
code is the time loop. Verified: free fall gives `v=−g t` *exactly* (backward Euler), and a
loaded mass–spring **relaxes to the quasistatic `u=F/k`** — so the dissipative integrator
doubles as **dynamic relaxation**, a robust way to reach static equilibria too.

**Contact-on-dynamics works (and confirms the thesis).** A neo-Hookean block dropped under
gravity via `solve_dynamics` (no Dirichlet BC — the mass makes the free body non-singular)
free-falls, the cubic barrier + CCD keep it penetration-free, and with mass-proportional
(Rayleigh) damping it **rests on the floor at gap > 0** (`tests/test_contact_dynamics.py`). Two
observations: (1) the `M/dt²` diagonal regularizes the per-step Newton (it converges where the
quasistatic solve stalled); (2) a *hard* impact with a *fixed* κ still diverges — the remaining
fragility that the ppf-style adaptive stiffness fixes.

**Adaptive stiffness (the ppf `M/g²` capacity) — now DONE, and it works.** `RigidBarrierContact`
takes a per-node `mass` and uses `s = κ + M/d²` (the inertial `M/d²` term → ∞ as the gap
closes). The decisive test (`tests/test_contact_dynamics.py`): a HARD drop with a deliberately
**too-soft `κ=10`** — fixed-κ crushes straight through the floor (gap **−1.83**), but the *same*
κ with the `M/d²` term **rests at gap +0.006**, settled, converged. Two things the implementation
taught: (a) the `M/d²` doesn't just add capacity, it gives a strong **recovery** force when the
fast inertial predictor `û` overshoots into penetration (the cubic stays NaN-free at `d<0`, and
`M/d²>0` there pushes back); (b) it needs `M`, so it only makes sense under the **dynamic
driver** — confirming, end-to-end, why dynamics is the substrate. (The fuller `wᵀK_elast w`
conditioning term over `κ` is the remaining refinement; `κ` as the baseline + `M/d²` capacity is
already robust.)

## 2026-06-21 — Penetration-free contact (Stage 4): cubic barrier, NOT the IPC log-barrier

Built rigid penetration-free contact (`RigidBarrierContact` + `rigid_barrier_eval` + a CCD
step bound wired into the driver line search). The key design call — and Teng flagged the
log-barrier as wrong before I'd finished — is **use a cubic barrier, not the IPC log-barrier**:

- **The log-barrier `-(d̂-d)²ln(d/d̂)` is ill-conditioned and fragile.** Its stiffness
  `b'' ~ 1/d → ∞` as the gap closes — and an ill-conditioned tangent is exactly what
  *amplifies solver non-reproducibility* (the MUMPS lesson below). It also `NaN`s for any
  `d ≤ 0`, so it demands strict feasibility on every iterate. Wrong fit for a solver where we
  care about repeatability and robustness.
- **The cubic barrier `(κ/3)(d̂-d)³` is bounded and robust.** Stiffness `B'' = 2κ(d̂-d)` is
  bounded (≤ 2κd̂); it's C² (C¹ force); it's a polynomial so complex step is exact and it
  **never NaNs**, even at `d ≤ 0` (a penetrating node just gets a large *finite* push-out).
- **Non-penetration comes from CCD, not from the energy → ∞.** The line-search step bound
  (`max_step`, exposed by the operator and consumed in `newton_solve`) keeps every iterate's
  gap > 0; the barrier only supplies a smooth, well-conditioned force. So we don't *need* the
  ill-conditioned ∞, and the method degrades gracefully if CCD is imperfect (curved obstacle).

**Convergence regime (a real fixed-κ limit, the conditioning theme again):**
- **κ must be ~ the bulk stiffness.** A too-stiff barrier (κ=1e4 vs a soft G=1 bulk, ~80×)
  makes the Newton step ill-conditioned and the line search **stalls — linear, not quadratic,
  convergence** (residual reduces ~10× then sticks; the block barely moves). Matching κ to the
  bulk (κ≈2e2 here) restored **quadratic** convergence (nit=7, |R|~1e-15). Adaptive κ (Ando
  dynamic stiffness) is the principled refinement; fixed κ works when scale-matched.
- **A finite barrier balances a FORCE, not an over-prescribed displacement.** Prescribing a
  boundary *past* the available gap demands infinite reaction — feasible only for the log-∞;
  with a finite cubic the solve correctly *stalls at the wall* (gap→0, no penetration, no
  convergence). So load barrier-contact problems by force, not by driving a boundary through.

Recurring meta-point: **ill-conditioning is the enemy on two fronts at once** — it amplifies
solver non-reproducibility *and* it stalls Newton. Prefer the well-conditioned formulation
(cubic over log; κ matched to the bulk) even when the ill-conditioned one is "more standard."

## 2026-06-21 — Codegen port (UEL-focused) + the multi-agent shared-checkout collision

Re-homed the form→`.for` generator into the build-time-only `coupfe/codegen/` subpackage
(option B, mechanical). Two lessons.

**"UEL-only" couldn't mean deleting `umat_gen.py`.** `uel_gen` imports the shared Python→
Fortran translation *engine* (`FortranTranslator`, `_tensorize_expr`, `_get_source_body`)
**from** `umat_gen.py` — the translator was written for UMAT first and reused for UEL. So a
literal "drop UMAT" would break UEL, and extracting the engine would be a *rewrite* (violating
option B). Resolution: **keep the engine (`umat_gen.py`), drop the genuinely-independent
`uinter_gen.py`, and scope *validation/release* to the UEL path.** Lesson: before "port only
X", grep X's imports — shared engines hide in sibling modules; scope the *release*, not
necessarily the *files*.

**Multi-agent shared checkout — the expensive one (be careful in future).** Codegen
and contact work were meant to proceed "in parallel on different branches", but there was
**one shared git checkout** — and a checkout has ONE branch at a time. The `coupfe/codegen/`
copy landed on the wrong branch, and a later **`git add -A`** swept it into an unrelated
commit, unnoticed until `git ls-files` was inspected. Fix, now in place: **a git
worktree per agent** so branches are truly isolated; **never `git add -A` in a shared
tree** (stage explicit paths); and **verify `git worktree list` / `branch` / `status`
before every commit**. Also: a blocked agent's approval/permission wall is in that agent's
session—another agent cannot clear it.

## 2026-06-20 (later) — Distributed rigid contact + friction: node-local; and a MUMPS non-reproducibility trap

Distributed **rigid-obstacle** contact is node-local: the analytical obstacle is known on
every rank, so a contact node's force depends only on its own position — no cross-rank
coupling. Each contact node is handled by **exactly one rank** (the one owning its first
DOF), which also holds that node's friction stick-state. It's added to `solve_distributed`
as a per-rank contribution (the operator contract is serial — see the entry below).

**The collective-scatter deadlock.** Reading a contact node's `U` uses a PETSc `VecScatter`,
which is **collective**. With PETSc's even DOF split, the boundary contact nodes (lowest DOF
ids) are all owned by rank 0, so other ranks own *zero* contact nodes. Guarding the scatter
with `if len(my_contact_nodes):` made those ranks skip it while rank 0 called it → **hang**.
Rule: every rank must call a collective (scatter/assemble/reduce) unconditionally; guard only
the *local* `setValues`. An empty-input kernel call returning empties is the clean way to keep
the control flow uniform.

**The non-determinism was the SOLVER (MUMPS), not the contact — corrected.** First read: a
mixed stick/slip solve differed by ~7.7e-4 (≈1.5% of `max|U|≈0.05`) between rank counts *and*
run-to-run at 4 ranks, while smooth cases were exact. I wrongly filed it as "non-smooth contact
is fundamentally non-deterministic in parallel." The user pushed back — 1.5% is too large, and
results must be repeatable. Root-causing (vary ONE thing — the linear solver) gave the real
answer:
- parallel **assembly** is bit-exact (smooth cases = `0.0`, not even 1e-16);
- with **superlu_dist** the *same* mixed non-smooth solve is rank-independent AND run-to-run
  repeatable to **1.3e-16** (verified 1/2/4 ranks);
- with **MUMPS** it is non-repeatable run-to-run (~7.7e-4) at >1 rank.
So the cause is **MUMPS parallel non-reproducibility** — its dynamic pivoting/scheduling
varies run-to-run by ~1e-12; an *ill-conditioned mode* (a slipping frictional node has ~zero
tangential stiffness → near-null mode) **amplifies** that into a visible ~1e-3. It is NOT a
property of non-smooth contact: MUMPS even handles the *asymmetric* friction tangent correctly
(`mumps_1rank == superlu_1rank` to 1.5e-16, ruling out a wrong symmetric mode). **Fix: default
to a reproducible solver (`superlu_dist`).** My continuity argument should have been the tell I
was wrong: penalty friction force is *continuous* at the cone, so a marginal stick/slip flip
changes the solution by ~0, **not** 1.5% — the number contradicted my own story, which meant
the story was wrong, not that the number was "fundamental".

**What is verified vs. still open (be honest about the boundary).** VERIFIED by direct
measurement: MUMPS is non-repeatable in parallel here; superlu_dist is exact and repeatable;
MUMPS is correct sequentially. NOT yet verified — the *quantitative mechanism*: I claimed
"~1e-12 solver noise amplified ~1e9× by a near-null slip mode," but did **not** measure MUMPS's
actual per-solve variation or the tangent's condition number, so the 7.7e-4 magnitude is not
fully explained (it could be larger direct MUMPS variation, conditioning, or a near-bifurcation
in the contact state). This does not change the fix or any gate; flagged for a deeper dive if it
ever matters (measure `||A x_mumps - A x_superlu||`, `cond(A)`, and the per-step set changes).

**Two process lessons from this (the important ones):**
1. **Try a different solver before declaring a numerical result "fundamental".** Swapping
   MUMPS→superlu_dist isolated the cause in one run. A surprising number you can't fully
   explain is a reason to vary ONE component (solver, rank count, regime, tolerance) and watch
   what moves — not to wrap it in a plausible narrative. A surprising distributed
   result is a setup/tooling issue until proven otherwise.
2. **Guessing from numerics is dangerous.** "Non-smooth ⇒ non-deterministic" *sounded* right
   and was confidently wrong. A plausible mechanism is not evidence. Demand a check that would
   *fail* if the guess were wrong (here: "then superlu_dist would be non-deterministic too" —
   it wasn't), and cross-check against what you already know (the force-continuity argument).
   Repeatability is non-negotiable for a production solver; never rationalize its absence.

**Friction is iteration-hungry — size the Newton budget for it.** The first load step that
goes from no-contact to contact+friction took **39** Newton iterations; the driver's default
`maxit=30` silently cut it off → an under-converged step → a wrong committed stick state that
propagated (it looked like a distributed bug; it was the iteration cap). Raised the default to
60. The lesson generalizes: a non-smooth operator can need many iterations on the step where
its active/stick/slip set is establishing; budget for it and *check the iteration count*, don't
assume non-convergence is a model error.

**The friction model.** Incremental penalty-Coulomb = plastic return-mapping: an elastic
tangential *stick* spring anchored at the contact point, trial force returned to the cone
`|f_t| ≤ μ|f_n|` on slip. It is **stateful** (committed tangential force per node, advanced in
`commit`, held internally like `CompiledElement.svars`). Complex-step gives the consistent
tangent with the active set *and* the stick/slip set frozen from the real iterate; the
normal-tangential coupling makes it **non-symmetric** (friction is non-associative). The kernel
is `rigid_penalty_eval` — contact's `element_rk_batch` analog (one call → R, K, new-state) —
reused by the serial operator and the distributed solver.

## 2026-06-20 (later) — Production distributed solve: the contract is serial; the element is the distributed unit

Wiring the production distributed Newton (`coupfe/assembly/distributed.py`) clarified a
boundary worth stating sharply: **the operator contract is the *serial* interface and does
not go distributed.** `assemble_residual`/`assemble_tangent` take the *global* `U` and return
*global* `gdofs` — O(ndof) per call. A memory-local distributed solve can't pass a global `U`.
The right distributed unit is the **batched element evaluator** (`element_rk_batch(coords, U,
DU)`) operating on a rank's partition with a ghosted slice of `U`. So `solve_distributed`
takes a `batch_fn` + `(my_gm, my_coords)`, *not* an operator list. Consequence: a new
*element* is distributed for free; a non-element operator (notably **contact**) will need its
own distributed contribution — the serial `Operator` does not carry over.

What made the gate clean:
- **Build-then-partition** (`element_partition`) keeps the global node numbering identical to
  the serial solve, so the 1-vs-N compare is *exact* (not "close after renumbering").
  Memory-local structured generation is the separate scale lever (same assembly, different
  mesh source) — don't conflate correctness with the scale optimization.
- **A direct linear solve makes the invariant machine-precision.** `pc="lu", solver="mumps"`
  → serial == N-rank to **1e-16** at 1/2/4 ranks. That's a *clean* correctness signal; an
  iterative PC would only agree to its `rtol` and muddy the read. Use direct for the gate,
  `gamg`/`gmres` for scale.
- **No f2py build race across ranks** — each rank is a separate process, compiles in its own
  tempdir, imports by module name into its own `sys.modules`. Concurrent gfortran is just
  load, not a conflict. (Each rank pays one build, cached after.)
- **History-free (hyperelastic) first** — the residual uses total `U`, so the serial
  (`DU=U`) vs distributed (`DU=U−U_prev_step`) convention difference is invisible. That
  isolates *distributed correctness* from the *state-handoff* question, which (as in the
  serial `solve_increments`) is the path-dependent follow-up.

It is a port + integration of the lab's proven `solve_steps_mpi_local`, not a new algorithm —
exactly as scoped. FieldSplit / diagonal scaling / per-element commit / unstructured
partition are faithful extensions tracked for when a problem needs them.

## 2026-06-20 (later) — Pipeline P: the front door is glue; gate it against the hand-wired path

**P is a thin layer, not a framework.** `Model` collects operators (element groups from
materials, contact), builds the Dirichlet dict from selectors, and calls `solve_increments`.
It contains **no physics** — every behaviour is already in an operator/material. The
discipline (now in `skills/pipeline.md` and `SKILL.md`): add a capability as an
operator/material first, then expose *setup convenience* for it on `Model` — never the
reverse, never physics in `Model`. If a `Model` method does anything an operator couldn't,
the abstraction has leaked.

**Gate the front door against the hand-wired path.** `tests/test_pipeline.py` solves the
block-on-plane by hand (build view → refine → `CompiledElement` → `ElementGroup.from_view`
→ `RigidContact` → `solve_increments`). `tests/test_model.py` solves the *same* problem in
~6 `Model` declarations and asserts the *same* result (converged, no penetration, deformed).
That pairing is the real test: it proves the front door is a faithful re-expression that
**changes nothing about the solve**, and it documents the API by example. Build the
hand-wired integration test first; make the declarative layer reproduce it.

**Convenient selectors remove the node-set boilerplate.** Resolving a BC target from a
*bbox face* (`"left"`/`"top"`/…, auto-detected from the coordinate extremes), a coordinate
*predicate*, or an *index array* — not just a pre-defined named set — is what makes the
front door concise. Trap caught in the selector test: a predicate must match nodes that
**actually exist** — `y==0.5` finds nothing on a 3×3-node grid (`y ∈ {0, ⅓, ⅔, 1}`); match
the test geometry to the mesh.

## 2026-06-20 (later) — Distributed mesh (M3/M3b): scope is the lever; verify in two layers

**Scope collapses the problem.** A general parallel mesh (DMPlex/p4est) is large because
it handles the worst case — arbitrary unstructured topology, parallel AMR with hanging
nodes, dynamic load balancing. Most research and the consulting target use *relatively
regular* meshes, where distribution collapses to block/coordinate partition (no ParMETIS),
structured ghost halos, embarrassingly-parallel uniform refinement, no hanging nodes, no
repartitioning. CoupFE targets that case; application repositories may put Gmsh+DMPlex
behind the `KernelMeshView` contract while core owns neither the adapter nor its
domain-specific labels. Structured topology + curved geometry via M1/M2 re-embedding (the
annulus is structured in (r,θ)) covers most real meshes.

**Verify distributed code in two layers, both gating the 1-vs-N invariant.**
1. **In-process (no MPI):** owned/ghost decomposition → assemble each part's owned cells →
   scatter to global → *sum == serial*; plus the gather/reduce adjoint `⟨Gu,v⟩=⟨u,Gᵀv⟩`.
   Catches decomposition/numbering bugs cheaply (`tests/test_distribute.py`).
2. **Real MPI (mpirun):** serial == N-rank to **machine precision** (~1e-15) at ≥2 rank
   counts. Catches the comm / PETSc-numbering bugs the in-process layer can't see
   (`tests/test_mpi_distributed.py`, via subprocess). Distributed bugs hide in test gaps;
   never claim "parallel works" from one rank count or a loose tolerance.

**Memory-local generation is the scale lever.** Each rank generates ONLY its block (compute
connectivity on the fly from `(rank, dims)` — never build the global mesh). For structured
meshes that's trivial. The PETSc Mat/Vec is distributed; assembling with *global* indices +
`ADD_VALUES` + `assemble()` lets PETSc sum off-process contributions, so you don't
hand-manage ghosts for assembly correctness — the memory win is in not building the global
mesh.

**A rank-independent solver is a correctness signal.** The distributed CG took 16 iterations
at 1, 2, *and* 4 ranks. A rank-*dependent* count would flag a distributed-state or
preconditioner bug — a surprising rank-dependent result is a setup bug, not a property.

**Mechanics:** petsc4py only (never import mpi4py); `Scatter.toZero` to gather for the serial
compare; `Mat.zeroRowsColumns(rows, diag, x, b)` for symmetric Dirichlet (match the serial
reference's reduced system); `OMP_NUM_THREADS=1` to avoid oversubscription.

## 2026-06-20 — Bootstrap: scaffold-first, one residual, complex-step

CoupFE began as a *clean-room scaffold*, not a fork of the research lab. The spine is
one abstraction — **everything is a residual-contributing operator** (`(residual,
tangent, commit)`) — and one discipline: **one residual is the source of truth; the
tangent is derived from it by complex step.** A 1D nonlinear bar (pure NumPy) proved
the contract before any compiled kernel, with the core invariant test *complex-step
tangent == analytic tangent*. Keeping the core tiny and the harness first-class (the
`skills/` docs ship with the code) is the product, not breadth.

## 2026-06-20 — Verify a ported runtime by RUNNING it, never by static review

The compiled-element port (`CompiledElement`/`ElementGroup`) was done by an agent whose
sandbox **blocked execution**, so it could only review statically. The port *looked*
correct — and it passed the patch test and the tangent==FD invariant — yet the uniaxial
case hit a **singular matrix → NaN** at runtime. Static review cannot see a Newton-basin
failure. The rule (now in `skills/testing.md`): a static port is **not done**; run it,
and when it fails, **diagnose, don't loosen the tolerance**. Diagnosing the null mode
(`eigvalsh` of the constrained block) vs tracing the residual per iteration told apart
"unconstrained RBM" from "step overshot into element inversion" — it was the latter.

## 2026-06-20 — Finite-strain Newton needs load-stepping AND a line search

Applying a 20% stretch in one Newton solve from a zero interior overshoots: the first
tangent is indefinite (a negative eigenvalue), the full step inverts an element, and
`ln(J)` of `J<0` → NaN. Two fixes, both core (not "glue"):
- **`solve_increments`** — ramp the Dirichlet BCs over a few warm-started increments;
- **a backtracking line search** in `newton_solve` — damp `alpha` until the residual is
  finite and decreasing.
A subtlety worth remembering: **finer meshes make the first-iteration distortion
worse** (a smaller boundary element sees a larger local strain for the same boundary
displacement), so the same 5% step that converged on a coarse mesh diverged on a finer
one until the line search was added.

## 2026-06-20 — Geometry re-embedding (M2): geometric, not solution-accuracy, for Quad4+Dirichlet

Curved-boundary convergence on the annulus (Lamé-form oracle) converges at **~h²**
*with or without* re-embedding the boundary onto the circle (the no-reembed error was
even marginally smaller). The honest conclusion: for **linear elements + Dirichlet BC**,
re-embedding is a **geometric** improvement (boundary nodes exactly on the curve — the
M1 mesh tests gate that), **not** a solution-accuracy one. Its *solution* payoff appears
with **higher-order geometry** (where affine geometry caps a high-order solution) or
**curved-boundary loads** (a faceted boundary has wrong normals/area). We gated the
*convergence rate* (the real oracle) and did **not** dress up a re-embed-vs-not solution
gap that isn't there.

## 2026-06-20 — Make a convergence study clean before asserting a rate

The first M2 attempt gave noisy, non-monotonic rates because (a) the field `αr+β/r` was
nearly linear → tiny pre-asymptotic errors near the solver floor, and (b) the chosen
load left a **finite-strain error floor** comparable to the discretization error.
Fixes: widen the radius ratio for real `1/r` curvature, keep the load small (linear
regime, floor well below the discretization error), use a **relative-L2** metric, and
read the rate at the **asymptotic (fine) levels**, skipping the noisy coarse one. A
convergence test is only meaningful once the measured quantity is the discretization
error, not a floor.

## 2026-06-20 — Small traps
- **`..` in a `sys.path` entry breaks `dirname`-based path resolution** — a sibling
  example resolved its kernel `.for` to the wrong directory. Use `os.path.abspath`
  before any `dirname` arithmetic that another module relies on.

## 2026-06-24 — Friction is interface plasticity; "small sliding" is the pairing, not the multiplier
The dual-multiplier / return-map friction **is** J2-like plasticity at the interface: the Coulomb cone
`|f_t| ≤ μN` is the yield surface, the slip is the plastic flow, stick is elastic. So like plasticity it
handles **arbitrarily large slip** via incremental return-mapping — there is **no small-sliding limit in
the constitutive law**. The "small sliding" restriction lives **only in the contact kinematics**: a frozen
pairing (fixed selection `S` — secondary node ↔ a fixed primary segment). For large slip the contact point
**migrates** (the foot crosses to the next segment), so the pairing must update.

Extend to **finite sliding** the way plasticity integrates a large-strain path: **per-step freeze the
pairing** (small per-step slip → the small-sliding linearization is fine, the same restriction ppf's
smoothed friction already has) + **re-pair between steps** (reuse the broad/narrow-phase search) +
**transfer the friction STATE**. That state is the crux: friction needs a **history/memory variable — the
accumulated slip / stick anchor / committed tangential force — exactly the analog of effective plastic
strain `ε_p`** (the carried `f_t` is the back-stress analog). It is what makes friction path-dependent, and
for finite sliding it must **follow the migrating contact point** (an objective slip increment across the
geometry change; the carried `f_t` rotates with the contact frame). RetroMech's `diff_friction.py` already
does the large-slip path return-mapping for a *rigid SDF* (no re-pairing); the deformable–deformable **state
transfer at re-pairing** is the only genuinely new piece — a contained problem, not a general-tangent rebuild.

META (recurring — an instance of "don't close with a limitation"): I repeatedly said "the dual-multiplier is
small-sliding," treating my fixed-`S` *implementation* as a *method* limit. Audit every "X can't do Y" claim
for implementation-scope-vs-method-limit before stating it.

## 2026-06-24 — Label production vs research; keep the production path untouched
A research direction (here: exact-stick / dual-multiplier / finite-sliding / the differentiable relay) can
grow fast and accrete many examples + a solver + opt-in operator modes. Two disciplines keep `main` a clean
*core* without a history rewrite (which a shared `main` forbids): (1) make every research piece **strictly
opt-in and default-off** (`friction_kt`/`friction_persistent` reduce *exactly* to the prior path when unset —
verified by the unchanged production tests), so it never touches the production behaviour; (2) **label it**
— an `examples/README.md` index marks `[prod]` vs `[study]`, and the docs/skills say plainly which friction
mode is the production path (smoothed) vs proofs-of-concept (return-map/persistent/dual-multiplier).
`contact` itself is the core (keep it); the line is production-vs-research *within* contact, not contact-vs-not.

## 2026-06-25 — A symmetric state masks a layout bug; the first non-symmetric state finds it
Building `j2_fefp_uel` (finite-strain J2, Fe-Fp) the compiled kernel disagreed with the
`reference_assembly` oracle by **1e-4 in stress** — yet the committed **state matched an
independent scipy return-map to 1e-14**. That split is the whole diagnosis: a stress-only path
was wrong while the return map was right, so the bug was in `polar(Fe)`/the STATEV read, not the
physics. Root cause: `reference_assembly` flattened tensor STATEV **C-order**, but the Fortran
kernel stores it **column-major** (`idx = offset + j*3 + i`).

**Why it survived every prior element (the "why we still have a bug"):** the trip condition is a
*conjunction* — state declared as a **3×3 tensor** AND **non-symmetric in value** — and no prior
element had both. The tensor-typed states were all `epsp` (plastic *strain*, symmetric by
construction — as is the usual finite-strain choice `Cp = FpᵀFp`), so order-invariant (`M == Mᵀ`):
the `kernel == reference` gate passed *regardless* of layout. `lce_quad4` *does* carry
non-symmetric state (`Fv`, with `Fv12 ≠ Fv21`) — but stores it as **five named scalars**, which
each have their own offset and never hit the tensor `reshape`, so it dodged the bug by
*representation*, not symmetry. `Fp` is the first state that is **both** tensor-typed and
non-symmetric, so it's the first that can distinguish C from F. The bug was latent, not
introduced; the missing thing was a **test that probes the axis** (one more instance of "the first
element to exercise a path finds the latent bug"). Fix = `order='F'` in `reference_assembly`; the regression gate now injects a
non-symmetric `Fp_old` and asserts it is non-symmetric (else the test is vacuous). Take-away: a
consistency gate only tests what its inputs *span* — a symmetric-only state never tests a tensor
layout.

**Two more from the same element.** (1) A private development study suggested
that the isochoric finite-strain-J2 case needed an anti-locking formulation,
but that example is withheld pending public formulation/port provenance. Its
historical stiffness ratio and convergence description are not release
evidence; do not turn them into a default-policy claim without a retained,
qualified rerun. (2) **The framework
verify's FD step (1e-7) false-fails a correct spectral tangent** — iterative `logm`/`expm`/`polar`
have a ~1e-12 noise floor that 1e-7 FD amplifies to ~1e-5. Diagnosed the textbook FD step-size
U-curve (bottoms at rel 1e-8 near `eps≈3e-5`) instead of loosening the tolerance — the CS tangent
the kernel uses is exact; the FD *reference* was the limited side. Verify spectral elements at the
resolvable step.

## 2026-06-25 — A scaling number is meaningful only on a converged, rank-equal solve

A historical local study first timed non-converged fixed-iteration runs, which
produced plausible but meaningless speedups. Later development runs suggested
the expected sub-linear, replicated-contact Amdahl regime, but their raw logs,
revisions, machine topology, and MPI/PETSc environment were not retained.
Therefore no numerical speedup from that study is release evidence.

The durable rule is to gate every performance record on convergence and
rank-1 equality, then retain the complete environment and output. The
distributed driver partitions the bulk but replicates the contact surface, so
only the bulk fraction is expected to scale.

Historical debugging also suggested that the confined lid/walls Newton stalled
as contact-pair count grew while a radial-trap problem and the bulk-only path
continued. The exact thresholds and residuals were not retained as release
evidence. That observation localized the investigation to confined-contact
pairing/CCD behavior and led to the fixes described below; it does not establish
a current scale limit.

## 2026-06-26 — Fine-mesh contact stall was TWO bugs in series; kinetics supplies the barrier capacity

The distributed barrier's fine-mesh non-convergence (the 2026-06-25 "open issue") was **two
pathologies stacked**, and fixing the first **exposed** the second — peel the onion, don't stop at
the first cause.
1. **CCD capacity lock (fixed).** Per-iteration logging (`|R|`, `ccd_alpha`) showed `ccd_alpha`
   pinned at **2⁻³⁰** (= the CCD bisection floor) with `|R|` constant for all 60 Newton iters —
   the step `U.axpy(α,du)` was a no-op, the state frozen. Cause: the distributed barrier ran on a
   **fixed κ** (the wiring never passed the per-node mass), so the ppf adaptive stiffness
   `s = κ + M/d²` was inactive; with only the fixed-κ capacity `~κd̂²`, once the compaction load
   exceeded it a pair's gap collapsed to ≤0 and CCD (which assumes a penetration-free state) floored
   α. **Fix:** thread `mass` into `_DistDeformableContact` → `DeformableBarrierContact2D`. Then
   `ccd_alpha=1.0` and most steps converge in 3–4 iters. **The capacity term `M/d²` is INERTIAL** —
   so *kinetics is what makes ppf-style contact robust* (no mass → no gap-capacity → fragile);
   confirms `ppf_contact_analysis.md`.
2. **Active-set chatter (FIXED — `freeze_pairing`).** With the lock gone (`α=1.0`), some fine-mesh
   steps still hit max-Newton with `|R|` **oscillating between two values** — the node-to-segment
   **closest-edge pairing flips** between iterations (a node at a vertex between two edges). Fix:
   `freeze_pairing=True` — fix each node's closest-edge **per step**, re-search only at `commit()`
   (positions still update with `U`; the gap mask still filters). n_ref=12 went from 12/33 chatter
   steps to **0** (≤4 Newton iters/step, rank-independent 1.3e-15). This is **not a band-aid — it is
   the standard node-to-segment method** (Abaqus/Wriggers/Laursen: the contact search is per-INCREMENT,
   never per-iteration). The old per-iteration re-pairing **half-adopted IPC's "re-evaluate the contact
   set" model without IPC's all-primitive smooth barrier** that makes re-evaluation safe — so we
   inherited the instability without the machinery. **Why it wasn't done at the start:** it was a
   *known, documented* gotcha (`skills/contact.md` listed "freeze the active set per solve — not yet
   done"); it stayed deferred because **every coarse/small test case converged with per-iteration
   re-pairing** — the chatter only appears at fine mesh / vertex configs, invisible until the fine-mesh
   benchmark forced it. Lesson: a *known deferred* robustness item on the default path is a latent bug,
   not a footnote — apply the textbook choice for your formulation up front.

Process lessons: (a) **a residual frozen at the CCD floor (2⁻³⁰) = a capacity/penetration lock**, not
chatter — read the `α` value. (b) **Fixing one contact pathology exposes the next** — the chatter was
real but masked. (c) **Don't trust fast/coarse reproductions:** shrinking N_STEPS to make the repro
quick introduced a *third* artifact (the lid is a Dirichlet BC, not CCD-bounded, so coarse steps
overshoot into the pack) that confounded the diagnosis — the clean signal needed the original
N_STEPS.

## 2026-06-26 — Robust contact is needed *infrastructure* for coupled multiphysics; "novelty" is the wrong lens

Teng's correction, recorded so the framing sticks. I kept judging the contact work by **novelty**
("the barrier isn't new; only differentiability is the moat") — that is the **wrong lens**, on three counts:

1. **The goal is a capability, not a novelty.** What CoupFE needs is a **robust contact solver that
   couples with our multiphysics** (thermal / chemical / phase-field / poromechanics, via the
   operator-contract spine). That value does not depend on being new. And differentiable contact isn't
   uniquely novel anyway — DiffTaichi, Warp, differentiable-IPC already exist. Robust contact is
   **methodology-enabling infrastructure**, exactly the thing not to write off as "plumbing."

2. **Why build it in-house instead of "just use ppf".** ppf is single-physics (pure mechanics), Rust+CUDA;
   it does **not** plug into our coupled PETSc operator engine. We need the robustness **inside** our
   framework so it couples. So ppf is the **template** (all-primitive cubic barrier, ACCD, adaptive
   `M/d²` stiffness, dynamics) — not a dependency, and not something to dismiss building as "reinventing."
   The contact + multiphysics **coupling** is the actual positioning: ppf doesn't couple; Abaqus is general
   but won't live inside our engine. Few tools are robust-contact *and* multiphysics-coupled.

3. **Abaqus-implicit is NOT the robustness bar.** Wide use ≠ robust implicit contact. Abaqus/**Standard
   (implicit)** has real contact-convergence trouble; the standard practitioner fallback is Abaqus/
   **Explicit** — contact is very robust there but slow (conditionally stable, tiny steps). Do not cite
   "Abaqus is mature" as if its implicit contact always converges; it doesn't. (Teng has a stress case.)

Path forward (consistent with the above): invest in genuine robustness using ppf as the template — the next
real step is the **all-primitive barrier** (vertex–edge / edge–edge), which removes the node-to-segment
closest-edge fragility *at the source* rather than patching it with the per-step pairing freeze. Build it
because the coupled engine needs it — not to be novel.

## 2026-06-26 — F-bar NaNs at centroid J̄≤0 (unguarded rescale); bulk feasibility (J>0) is as essential as contact feasibility (gap>0)

A contact solve NaN'd; I burned a long time **sweeping the wrong knobs** (lid, gravity, walls, Poisson —
none fixed it) before the two tools that actually cracked it: a **dimensionless analysis** (which caught a
*separate* real over-load, εg=ρgL/G≈2.7–10.8 ≫ 0.5) and a **bulk-vs-contact residual split** (a `bulk_nan`
check in the assemble) which said "bulk", after which **reading the F-bar code** found it in two minutes.

**Root cause.** Our F-bar rescale `alpha = (J̄/J)^(1/d)` is mathematically correct (de Souza Neto, right
exponent), but **unguarded**: when the **centroid** Jacobian `J̄` crosses 0 (centroid inversion under
bending/shear), `(negative)^(1/2)` → **NaN** — *even while every Gauss-point J is still positive*. It is
**F-bar-specific**: a standard element uses each GP's own J and cannot hit the centroid mode — **verified**:
the standard element ran clean (`bulk_nan=0`) on the exact config where F-bar NaN'd at step 23.

**The general principle (this is the big one).** In an implicit solve, **contact AND bulk both need
FEASIBILITY GUARDS, not just energies.** Contact: `gap > 0` (CCD/ACCD). Bulk: `J > 0` and, for F-bar,
`J̄ > 0` (a J-bound line search / strain limiting). A cubic barrier or an F-bar element is an *energy*;
without a feasibility line search keeping every iterate inside `{gap>0, J>0}`, the iterate leaves the feasible
region and the `log J` / `(J̄/J)^(1/d)` / `M/gap²` NaNs at the boundary. **ppf has BOTH (ACCD + strain
limiting) — that is why ppf is robust and our ports NaN.** The method (F-bar) is robust; *our implementation
lacked the guard* — do not mistake a missing feasibility guard for an unstable method.

**To avoid it (so we don't repeat this):**
- **Any element with a centroid/mean rescale `(X)^(1/d)`** (F-bar, F-bar plasticity) **MUST guard `X>0`** —
  a feasibility line search that prevents the Newton step from driving `J̄≤0`, or at minimum a NaN-safe
  fallback (detect `J̄≤0`, signal a step cut). A raw fractional power of a possibly-negative base is a
  **silent NaN trap** in codegen/Fortran elements.
- **When a NaN appears, LOCALIZE the residual FIRST** (bulk-vs-contact split — one run tells you which
  operator) **before sweeping loads/BCs.** I swept lid/gravity/walls/Poisson; the split + a code read would
  have been minutes.
- **Pair the feasibility guards:** if you have a contact CCD (`gap>0`), you also need a bulk `J>0` guard —
  same idea, two operators. (The strain-limiting / J-bound line search is the bulk analogue of ACCD.)

**Why it's F-bar (not "F-bar is unstable") — and the simple fix, since this IS known in the field.** F-bar is
the gold-standard near-incompressible element (de Souza Neto et al. 1996); it is *not* inherently unstable.
The volumetric rescale `F̄=(J̄/J)^(1/d)·F` samples volume at the **centroid** `J̄` (that's what cures locking),
and the catch is the **fractional power**: if the deformation drives `J̄≤0` (the element folds at its
centroid) then `(J̄)^(1/2)=sqrt(neg)→NaN`. But `J̄≤0` is a **genuine local inversion** — *no* element is
meaningful there; a standard neo-Hookean's `log J` NaNs at `J≤0` too. The standard element merely survived our
test because it inverts on the *per-GP* J while F-bar inverts on the *centroid* J̄, and on our contact path
the centroid crossed zero first. Element inversion (`J≤0`) robustness is a well-studied topic, and the fixes
are standard, simplest first: **(1) a `J>0`/`J̄>0` line search** (scale the Newton step so no Jacobian crosses
0 — the bulk analogue of the contact CCD; keeps F-bar, fails *gracefully* not NaN — the simplest patch);
**(2) an inversion-safe energy** (Stomakhin et al. 2012, energetically-consistent invertible elasticity —
extends the energy smoothly below a threshold J); **(3) mixed u-p** (no centroid rescale → the failure mode
doesn't exist; the more principled near-incompressible element — what we adopted, validated `bulk_nan=0` on
the pack where F-bar NaN'd). So: the *surprise* was an unguarded `(J̄/J)^(1/d)`, the *cleanest* fix is the u-p,
the *quickest* is the J̄>0 line search.

## Architectural decisions (rationale in `docs/status.md`)
- **Historical development decision; not the first-release default.** A June 2026 development round
  preferred mixed u-p over F-bar. The element-local condensed pressure gives
  `p = K·avg(lnJ)` — the **volume average** of the
  volumetric response, i.e. **mean-dilatation / B-bar** (deviatoric full + volumetric averaged), *not* F-bar's
  **centroid** value. So it has **no `(J̄/J)^(1/d)` rescale → no centroid-J̄ NaN** (validated `bulk_nan=0` on
  the disk pack where F-bar NaN'd). That case does not establish general inf-sup or inversion
  robustness (including the Q1-P0 checkerboard caveat). **For
  multiphysics** `u`–`X`–`p` (X = μ/T/c): `p` is **condensed-local** (static condensation → zero global DOFs),
  so it makes the volumetric mechanics robust *inside* the coupling at no global cost — the `local_pressure`
  generator emits the gel `u`–`μ`–`p` when a `μ` field is declared, else pure-mechanical `u`–`p`. F-bar is kept
  only as a fallback (and would need a `J̄>0` line search to fail gracefully). The recorded disk-pack
  convergence is historical development evidence. The coupled gel
  local-pressure examples are withheld from the first release pending a
  formulation citation, provenance, and scoped qualification. The three
  pure-mechanics Quad4/Hex8 variants ship as explicitly unqualified RESEARCH
  codegen workflows; they are not release defaults or validation evidence.
- **Multiphysics local pressure uses the ELASTIC Jacobian `J_e=J/J_inel`, not total `J`** (Teng, 2026-06-26).
  With an inelastic volume change (`F=F_e·F_inel`: swelling `F_s`/thermal `F_θ`/growth `F_g`), the
  stress-free inelastic part must be divided out: `p = K·avg(ln J_e)`. Total `J` gives a **spurious pressure**
  (a freely-swelling gel should have `p=0`, but `K·avg(lnJ)≠0`) — the `F=F_e·F_g` /
  sanity-check-`J_e~O(1)` rule. The gel path already uses `J_e`; pure mechanics is
  the `J_inel=1` case (correct as-is); **a NEW coupling (thermal/growth) through `local_pressure` needs the
  `J_e` split with `J_inel` from the coupling field — a generalization TODO in codegen.**
- **Compiler stays separate** (the form→`.for` generator is build-time, never merged
  into the runtime; keep `abaqus_ufl` now, re-home as `coupfe-gen` later by a mechanical
  rename — never a rewrite).
- **No UMAT runtime** — materials are functions inside element operators; UMAT survives
  only as an Abaqus-backend emit option.
- **Defer on concrete need, not speculatively** — matrix-free `Jv`, GPU, and the
  standalone no-Python Fortran runtime are all deferred (the last likely never).

## 2026-06-30 — 3D Cattaneo/Abaqus diagnostic: same mesh is not same model

The private diagnostic showed that the original `RigidContact` nodal-penalty
path could be exercised on a large uploaded mesh when the linear solve moved
from SciPy/SuperLU to PETSc/MUMPS. It did not establish Abaqus parity. The
example is withheld from the public artifact because its reviewed end-to-end
gate is a strict expected failure and the retained solve is marked
`converged=false`. Its historical force, radius, and stick/slip values must not
be cited as release evidence.

Lessons:
- **Same mesh is necessary but not sufficient.** The uploaded Abaqus deck uses `C3D8R` and Abaqus
  contact enforcement; the CoupFE run used the compiled F-bar/full-integration Hex8 kernel and finite
  nodal penalty. Mesh parity alone cannot prove solver parity.
- **Separate the linear-solver bottleneck from nonlinear contact behavior.** SciPy `spsolve` made one
  78k-DOF factorization take minutes, hiding the actual Newton trace. PETSc/MUMPS made the trace visible:
  `ksp_its=1` exonerates the linear solve; residual plateaus are contact active-set/penalty behavior.
- **Stiff penalty is a nonlinear-model choice, not just a numerical knob.** The original `k=5000`
  chattered; `k=200` progressed and matched radius well, but it changes the contact law. Do not compare a
  softened penalty run to Abaqus hard/augmented contact as if the enforcement were identical.
- **Normalize before diagnosing friction.** Dimensionless ratios can help
  separate normal/contact-law/element-stiffness mismatch from a tangential
  return-map defect, but the diagnostic must converge before those ratios are
  evidence.
- **A contact-radius match does not validate stick-radius extraction.** The nodal stick extractor still
  under-predicted the oracle (`c_fe/c_h=0.834`) despite a resolved Hertz radius. Stick/slip extraction is a
  separate post-processing problem; do not bury it inside the contact-radius success.
- **Close diagnostics honestly.** This example is withheld, not a production
  benchmark. Reopen only with a specific new capability: C3D8R-like reduced
  integration/hourglass control, matched Abaqus penalty contact,
  augmented-Lagrange/hard contact, a deliberately scoped barrier comparison,
  or a better stick-radius estimator. Do not use this nodal-penalty result as
  evidence against the barrier path;
  that path has different feasibility and convergence requirements.
