# CoupFE contact + dynamics skill

> **Release-evidence boundary:** exact timings, speedups, iteration counts, and
> scale figures in this dated engineering guidance are historical local
> observations without retained raw logs and a locked environment. Rerun them
> before citation. Functional and correctness gates are tracked separately.

Read before touching `coupfe/operators/contact.py`, `coupfe/operators/inertia.py`, or the
drivers in `coupfe/assembly`. The math is in `docs/theory/contact_dynamics.md`; this is the
operational discipline + the hard-won pitfalls. Pair with `skills/SKILL.md` and
`skills/pitfalls.md`.

## Why contact is a HARD simulation task (orient before you start)
Contact is not "elasticity plus a wall." It is the place where most FE solves go to die, for
structural reasons — name them so you design around them, not into them:
- **The constraint set is COMBINATORIAL and changes during the solve.** Which nodes touch, and which
  stick vs slip, is a discrete decision re-made every iteration → the residual/tangent are
  **non-smooth**. Newton has no fixed point to converge to while the active set churns (the active-set
  chatter / dual-multiplier cycling we keep meeting).
- **Geometry is a minefield of degenerate cases.** Closest-feature (face↔edge↔vertex, which triangle)
  has sharp branches; a hand-rolled distance/CCD predicate *will* get a degenerate case wrong (missed
  contact → penetration). **Port battle-tested geometry, don't re-derive it.**
- **Non-penetration is a CONSTRAINT, not an energy.** A finite penalty/barrier can be overrun; the
  guarantee comes from a **feasibility filter** (CCD keeps `gap>0`; the bulk analogue keeps `J>0`),
  not from energy→∞. Energies alone are not robust at the boundary.
- **Conditioning is adversarial.** Barrier/penalty stiffness fights the bulk; a slipping node has ~0
  tangential stiffness (a near-null mode) that **amplifies** any solver non-reproducibility
  (the MUMPS-vs-superlu_dist lesson). Reproducibility is non-negotiable and easily lost here.
- **Every model trades robustness against fidelity — there is no free lunch.** Smoothed/barrier is
  robust + penetration-free but *creeps* at stick; exact-stick (return-map / dual-multiplier) nails the
  stick but is fragile and narrow. Choosing wrong wastes weeks.
- **Validation is genuinely hard.** Analytic oracles are few (Hertz, Cattaneo) and carry hidden
  assumptions (the Dundurs-β trap — "compare like with like"); extraction is brittle (never `max|x|`).

The two design responses that make it tractable, both baked into CoupFE: **(1) dynamics as the
substrate** (inertia regularizes the non-smooth barrier + supplies capacity; quasistatic Newton stalls
on the indefinite barrier — see below), and **(2) freeze the discrete decision from the REAL iterate**
(active set, stick/slip, closest feature) and complex-step only the smooth branch.

## Which contact model to use (we have ~8 operators / 3 normal families × 3 friction fidelities)
Pick by **what you need**, not by what's newest. The families:

| Normal treatment | Operators | Non-penetration | When |
|---|---|---|---|
| **Penalty** | `RigidContact`, `DeformableContact2D`, `SurfaceContact2D` (self) | by stiffness (not guaranteed) | simplest; rigid obstacle; distributed node-local |
| **Cubic barrier + CCD** (ppf/IPC) | `RigidBarrierContact`, `DeformableBarrierContact2D` (+ 2D all-primitive), `DeformableBarrierContact3D` | **guaranteed** (CCD) | the production path — 3D, large-deformation, robust; needs **dynamics** |
| **Dual-multiplier** (constraint) | `SemismoothFrictionSolver` | exact KKT | **2D-only**, small-strain; the *differentiable* exact-stick niche |

Friction fidelities (on the penalty/barrier path): **smoothed** (ppf, rate-form, the default —
distributed, 2D+3D, but creeps at stick) · **return-map** (`friction_kt`, exact cone, near-exact
stick, 2D + 3D vertex-face) · **persistent** (finite-sliding exact stick). The **dual-multiplier**
gives machine-zero stick + a differentiable relay, 2D only.

**Decision guide:**
| If you need… | Use |
|---|---|
| Default: 3D / large-deformation / robust / scalable / distributed | **cubic barrier + smoothed friction** on `solve_dynamics` |
| An exact stick zone / partial-slip stick radius (not large slip) | **return-map friction** (`friction_kt`) on the barrier |
| 2D exact partial-slip **+ gradients** (fretting / joints) | **semismooth dual-multiplier** + relay |
| Rigid obstacle, simplest, distributed | `RigidContact` (penalty return-map), node-local |

**Principles:** want a *guarantee* (no penetration) → always a **barrier** (CCD), never raw penalty.
Robustness/scale/3D → **smoothed**. An exact-stick *QOI* (slip amplitude, dissipation, stick radius) →
**return-map** (forward) or **dual-multiplier** (2D + differentiable). Dual-multiplier is **2D-only** —
all 3D work runs on the barrier stack. See `docs/dev/dual_multiplier_strategy.md` for the niche.

## How to add / change a contact operator

1. **Residual from an energy; signs from the gradient.** Normal penalty `R = k g n`; cubic
   barrier `R = −κ(d̂−d)² n`; friction `R = f_t` (return-mapped). Physical force = `−R`. Never
   guess a sign — differentiate the potential.
2. **Freeze the discrete decision from the REAL iterate, then complex-step the smooth branch.**
   Active set (`g<0` or `d<d̂`), stick/slip (`|f_t^trial| vs μ|f_n|`), the closest feature — all
   computed from `np.real(x)` and held fixed; only then perturb with `+i·h`. **Never
   complex-step the switch itself.** Use `√(Σ vᵢ²)` not `np.abs`/`np.linalg.norm`.
3. **State is internal and node-local.** Friction's committed `f_t` / stick anchor live on the
   operator (like `CompiledElement.svars`), advanced in `commit`. Node-local ⇒ it distributes
   (one rank owns each contact node's state).
4. **Expose `max_step(U, dU)` for penetration-free contact (CCD).** Return the largest
   `α ∈ (0,1]` keeping every gap > 0. The driver starts the line search at `min(1, minₒₚ
   max_step)`. Non-penetration comes from this, not from the barrier → ∞.
5. **Gate with an independent oracle + a broken control** (not just CS-vs-FD): the analytical
   Coulomb law (stick slope `k_t` → slip plateau `μ|f_n|`), a load-balance at a known
   equilibrium, the non-penetration guarantee, `mu=0`/`adaptive=False` backward-compat.

## Barrier choice — cubic, not log

Use the **cubic** barrier `(κ/3)(d̂−d)³` (bounded stiffness, C², polynomial → no NaN even at
`d≤0`). **Do not** use the IPC log barrier here: its `1/d` stiffness is ill-conditioned (which
amplifies solver non-reproducibility) and NaNs at `d≤0`. Get the non-penetration guarantee from
CCD instead of the energy → ∞.

`RigidBarrierContact(..., ppf_norm=True)` switches to the geometry-normalized cubic used by the
reference ppf-contact-solver: the force is `−(2/d̂)·s·(d̂−d)² n` with `s = κ + M/d²`. This
decouples the activation distance `d̂` from the force magnitude and makes `κ` a true contact
stiffness (force/length), matching the ppf `wᵀ(K+M/g²)w` recipe.

## Stiffness: matched-fixed, or adaptive under dynamics

- A *fixed* κ must satisfy BOTH **capacity** (`κd̂²` > the reaction, else the gap collapses) and
  **conditioning** (κ not ≫ the bulk, else the line search stalls — linear, not quadratic). For
  a static problem, hand-match κ to the bulk and size it to the load.
- **Adaptive κ is a capacity-vs-conditioning problem, not a one-liner.** A naive "match κ to the
  bulk stiffness" controls conditioning but ignores capacity → it made convergence *worse*. The
  correct recipe (ppf) scales the geometry-normalized barrier by `wᵀ(K_elast + M/g²)w`, whose
  `M/g²` capacity term is **inertial** — so it is well-posed only **under dynamics**. Use
  `ppf_norm=True` so the cubic shape carries the `2/d̂` factor; pass the bulk operator(s) as
  `elastic_op` to have `RigidBarrierContact` add the normal-direction `nᵀK_elastn` estimate to
  `s` (completing the ppf recipe beyond the scalar `κ` approximation).

## Dynamics is the substrate for robust contact

- Add `InertiaOperator` (lumped mass via `lumped_mass`, backward-Euler `M/dt²(u−û)`) and solve
  with `solve_dynamics`. The element/contact/load operators compose unchanged.
- The `M/dt²` diagonal **regularizes the per-step Newton** (converges where quasistatic stalls);
  inertia regularizes non-smooth transitions; and it supplies the barrier's `M/g²` capacity.
- **No Dirichlet BC needed** for a free body — the mass makes the system non-singular under load.
- **Dynamic relaxation:** turn on Rayleigh damping (`InertiaOperator(..., damping=α)`) and hold
  the load → the dissipative integrator settles to the quasistatic equilibrium. A robust way to
  reach static solutions too (it's a *trick for quasistatics*, not only for transients).
- Load barrier-contact problems by **force**, not by driving a boundary *through* the gap (a
  finite barrier can't supply infinite reaction → it correctly stalls at the wall).
- **CCD must bound the PREDICTOR, not just the Newton increment.** `solve_dynamics` warm-starts
  each step from the inertial predictor `û = u_prev + dt·v_prev`; if `dt·v > gap` that jump
  *teleports* a node through the barrier band before Newton runs. The driver CCD-bounds the
  predictor from the last accepted state (same `max_step`, a no-op when separated). **General
  rule: any position update that bypasses the line search — predictor, warm-start, BC ramp,
  or a MOVING RIGID OBSTACLE — must go through the same CCD bound, or it tunnels.** A
  displacement-controlled lid/indenter/platen moved by recreating its `HalfSpace`/`Sphere`
  lower each step bypasses the CCD; if it jumps `>dhat` it lands already-penetrated, in the
  barrier's dead zone (zero force outside `[0,dhat)`), and passes through. Today's workaround is
  "steps `< dhat`" (used by `compression_cylinders`' lid and the Hertz indenter) — a breadcrumb,
  not a guarantee.

  **Solution:** use a time-aware obstacle (`MovingHalfSpace(position(t), normal)` or
  `MovingSphere(center(t), R)`) and the CCD bound now samples the obstacle trajectory at the
  fraction `α` along the predictor step (`t_a = t − dt + α·dt`). The driver forwards both the
  target time `t` and the step size `dt` to `max_step`, so the predictor jump is bounded against
  the *actual* relative motion, not just the end position. `examples/ring_compress/reproduce_dynamics.py`
  uses `MovingHalfSpace` for the top platen; the old `_MovingObstacleContact` wrapper is kept
  only to update the stored `obs.p` for the penalty contact path.

  (Once predictor-bounded, even a too-soft *fixed* κ stays penetration-free — so adaptive
  `M/d²` is about capacity/conditioning, not non-penetration: without it the solve pins at
  gap≈0 and burns many-fold more Newton iterations.)

## Deformable–deformable contact (node-to-segment, 2D)

`DeformableBarrierContact2D` / `deformable_barrier_eval` — both sides are DOFs (secondary node +
2 primary-edge nodes all carry the cubic barrier force). The discipline that makes it clean:

- **The gap to the edge LINE is `d = (e×r)/L` and is independent of the foot ξ** (`e=xb−xa`,
  `r=xs−xa`). ξ only gates the active region (`0≤ξ≤1`, foot on the segment) and is *not* in the
  force. Activate on `d<d̂ AND 0≤ξ≤1`; a node off the segment end is a point-point case (skip).
- **Build the force from the analytic gap-gradient, not a hand-split `(1−ξ,ξ)`.** `∂d/∂xs = n`
  exactly and the three nodal gradients **sum to zero** (translation invariance) → the assembled
  force is action-reaction (Σforce=0, Στorque=0) *by construction*. Verify that, don't assume it.
- Because the gradient is analytic, the barrier force is **complex-step-differentiable** → the
  *consistent* 6×6 tangent (normal-rotation geometry, not just rank-1 `n⊗n`); it comes out
  **symmetric to machine precision** — it *is* an energy Hessian, so check symmetry as a gate.
- **Point-edge CCD** (`max_step`): the secondary is colinear with the moving edge when
  `e(α)×r(α)=0`, a quadratic in α (both linear in α) → smallest root in (0,1] with the foot on
  the segment, eta-capped, plus a bisection net re-checking the *true* signed gap.
- Two bodies = compose **two** operators (A-nodes vs B-edges *and* B-nodes vs A-edges). Closest
  edge is brute-force (→ numba/Rust at scale). Self-contact = a search-based sibling (TODO),
  exactly as `SurfaceContact2D` is to `DeformableContact2D`.
- **N-body mutual contact (one operator over the UNION of all bodies' nodes+edges) demands
  CONSISTENT edge winding.** Since `d=(e×r)/L>0` only when the secondary is on the LEFT of `a→b`,
  every body's boundary loop must be oriented the same way (**outside-on-left**: orient so
  `cross(b−a, mid−centroid)>0`). A single wrongly-wound edge gives a neighbour node a *negative*
  gap → the barrier reads a deep penetration → a **huge spurious force at the undeformed rest
  state → divergence** (the 16-disk cylinder pack, 2026-06-25; the 8-disk dodged it by luck).
  **Diagnose: `|R_contact|` at U=0 must be 0** — a force at rest is impossible for real physics,
  so it localizes the bug to contact instantly (don't blame the element or the load). Belt-and-
  suspenders: the opt-in `body_id=` (per-node body id) excludes *all* same-body edges, not just
  the 1-ring incident ones. `examples/compression_cylinders/build_model.py:boundary_edges`,
  `tests/test_multibody_contact.py`.
- **Penetration oracle must be surface-aware when both sides deform.** Comparing a node to a
  flat/extremal proxy (e.g. global-max strip-top y) lies once the opposing surface bends —
  interpolate the *deformed* surface beneath each node (the operator's own per-pair gap is the
  truth; a phantom "penetration" usually means the oracle, not the contact).

## 3D deformable contact + friction (`contact3d.py`, `DeformableBarrierContact3D`)

Two primitives — **vertex-face** (point-triangle) + **edge-edge** — each a cubic barrier over a
12-DOF stencil, with ACCD `max_step`. The geometry (distance coeffs, closest-feature
classification, ACCD) is **ported from ppf**, not re-derived (see the Don'ts). Each barrier's gap
gradient is `grad = Bᵀn`, where `B`'s blocks are the **signed closest-point weights**: vertex-face
`[1,−w₀,−w₁,−w₂]` (barycentric), edge-edge `[a₀,a₁,−b₀,−b₁]`.

- **3D friction = the 2D smoothed model on the tangent plane `P = I − n⊗n`** (`_smoothed_friction_3d`,
  shared by both primitives). Force `λ(P·dx)`, `λ = μ λ_n / max(ε,‖dx‖)`, opposes the relative
  tangential slip `dx = Σ cᵢ(Xᵢ−X0ᵢ)` of the contact-point pair since the step start; residual `Bᵀ`-
  mapped, tangent `λ BᵀP B`.
- **Reuse the barrier's weights for friction — don't recompute a separate Jacobian.** Because
  `Σ cᵢ = 0`, action-reaction (`Σf=0`) and zero-friction-under-rigid-co-translation hold **by
  construction** (verify, don't assume). The tangent is symmetric PSD (Gauss-Newton; the same
  lagged-`λ_n`/mollified-cone/PSD-projection approximations as 2D).
- Kernels take `mu/eps/X0` (step-start stencil positions); `mu=0` or `X0=None` is **byte-identical**
  to frictionless. The operator gains `mu/friction_eps` and tracks step-start `_x0` (advanced in
  `commit`) — stateful, semi-implicit. Gate it like 2D: `|f_s|=μλ_n` at full slip, stick(sub-ε)→slip,
  zero under rigid/normal motion, PSD tangent, and the stateful round-trip (drag → friction →
  vanishes after `commit`).
- **Dedup is NOT needed** (checked ppf): ppf does no type-classification/dedup and no edge-edge
  mollifier (unclassified closest distance + cubic barrier; same `i<j`+shared-vertex exclusion). The
  coincidence double-count is benign and present in ppf too — don't build a dedup.
- **3D distributed contact development path (cross-rank assembly + global CCD):**
  `_DistDeformableContact3D` (3D
  subclass of the 2D helper) — vertex-face partitioned by the secondary vertex, edge-edge by
  `DeformableBarrierContact3D(owns_edge_pair=…)`; off-process `ADD_VALUES` routing + global-CCD
  `Vec.min`. Historical development gates recorded rank invariance
  (`distributed_3d_primitives`); a retained final-revision MPI rerun is
  pending. NOTE: verify it by
  the **static cross-rank assembly** (solver-free) — a no-bulk (mass-point) dynamics solve is
  degenerate (direct LU fails at engagement; PETSc SEGVs on empty objects). The full end-to-end 3D
  *dynamics* solve needs a 3D **bulk element** (then the system is well-conditioned, like 2D).
- **End-to-end 3D distributed dynamics development path** (with the vendored
  **F-bar Hex8**): two demos have historical 1/2/4-rank development records —
  `distributed_dynamics_3d_blocks` (blocks
  collide, frictionless) and `distributed_dynamics_3d_friction` (blocks seated + sheared, μ>0 friction
  holds, slip 0.64× frictionless). The full deformable-contact stack — bulk + barrier + friction —
  now runs distributed in 2D AND 3D.
- **Smoothed-friction convergence:** tune `friction_eps` to the interface **slip scale** (not tiny).
  In the slip plateau (`ut ≫ eps`) the Gauss-Newton tangent drops `dλ` (comparable to the kept `λP`) →
  Newton floors at moderate `|R|`; set `eps ≳ ut` (near-stick) and use `μ` for strength (which has its
  own convergence ceiling). See `docs/dev/contact_experiments.md` (2026-06-24).
- **Acceleration — match the tool to the STRUCTURE (the key principle).** numba pays off ONLY where
  numpy is forced into a Python per-element loop; where numpy *vectorizes*, it is already near-optimal
  and numba adds nothing. Concretely:
  - **3D narrow-phase (vertex-face + edge-edge + ACCD) → numba.** Per-pair **data-dependent branching**
    (closest-feature classification face/edge/vertex, degenerate fallbacks, edge-edge alternating
    projection) — each pair takes a different branch → can't vectorize → the numpy code was a Python
    per-pair loop. `@njit` removes that interpreter loop and remains bit-exact
    against the NumPy oracle; remeasure before making a speedup claim.
  - **Return-map friction (`rigid_penalty_eval`) + 2D smoothed/barrier → vectorized numpy.** Node-local
    vs one obstacle, identical ops per node, non-smoothness = global masks → already loop-free. **Do NOT
    numba them** — it would remove nothing (non-bottleneck; profile first).
  - **Element kernels → Fortran** (dual-home: must also run inside Abaqus — a requirement contact does
    NOT have, so contact uses numba, not Fortran).
  - Always keep the **numpy version as the bit-for-bit ORACLE**; gate numba ⇄ numpy to machine
    precision. (Full table: `docs/api.md` "Performance".)
- Implemented: numba narrow-phase (vertex-face + edge-edge + ACCD) + **LBVH broad-phase** (ppf template,
  iterative — a recursive `@njit`+`cache=True` can stale-segfault); **self-contact** (3D barrier +
  smoothed friction, numba incident-exclusion `self_contact=True`). Focused
  operator gates pass, but the full hairpin subprocess exceeds its bounded gate
  and is withheld from the first release, so 3D self-contact remains partial.
  Remaining 3D: GPU (`numba.cuda` / ppf Karras LBVH) at scale; rate-and-state
  friction when a problem needs it.

## Exact-stick friction line (research-grade — `docs/dev/dual_multiplier_strategy.md`)
Three friction fidelities, increasing in stick fidelity. **The smoothed model is the production path; the
rest are gated proofs-of-concept** for the differentiable / fretting / knot directions — do not assume they
scale or are wired to the distributed path.
- **Friction = interface plasticity** (cone = yield, slip = plastic flow). So it handles arbitrarily large
  slip via incremental return-mapping — "small sliding" is only the *pairing*, not the multiplier
  ([[friction-is-interface-plasticity lesson, docs/lessons_learned.md]]).
- **return-map** (`friction_kt`): exact cone, correct slip tangent (no plateau floor), 2D + 3D vertex-face.
- **persistent** (`friction_persistent`): carries the committed tangential force (εₚ analog) across steps,
  re-framed onto the current tangent plane at re-pairing → finite-sliding exact stick (held contact KEEPS
  its force; per-step FORGETS it). 2D + 3D. Verified: capstan reproduces `e^{μθ}`.
- **dual-multiplier** —
  `contact_semismooth.SemismoothFrictionSolver` (Alart-Curnier **semismooth Newton**):
  exact stick, per-node partial slip, Schur-condensed, factor-once, and
  small-strain/small-sliding. The **relay** (frozen-active-set adjoint) makes it
  differentiable for friction-field identification. PETSc bulk-solve parallel;
  PERMON remains the forward-QP scale reference (`docs/dev/permon_assessment.md`).
- **Where it earns its keep:** micro-slip / dissipation at the stick-slip boundary over cycles (fretting,
  joints, knots), all small-strain. NOT large-deformation/gross-sliding (that's smoothed ppf).

## Broad-phase search (`contact_search.candidate_pairs`)

The closest-edge scan is O(N²) brute force; the broad phase is a **uniform spatial hash** (bin each
edge into every cell its `d̂`-expanded AABB overlaps, query each vertex's cell). It is a **superset**
of the true within-`d̂` set by construction — no contact missed; brute force is the oracle.
- **Band = `d̂` ⇒ the swap is *identical*, not approximate**, on the residual/tangent path: an active
  pair has point-segment distance `< d̂` ⇒ it's in the band-`d̂` superset ⇒ same closest edge + force;
  a vertex with no in-band edge contributes nothing either way. So wire it there freely (O(N²)→O(N)).
- **`max_step`/CCD needs band `> d̂`** (it must catch an edge the vertex could *cross this step*) — do
  **not** reuse the band-`d̂` set for CCD; it needs a per-step-reach margin. Still brute force today.
- **Uniform grid first, BVH only for non-uniform** meshes; numba is the accel (`@njit`/`prange`), Rust
  the extreme. The search is self-contained (geometry in → pairs out), no PETSc — reusable distributed.

## Distributed contact (recap; full detail in `skills/distributed.md`)

> Exact rank counts, tolerances, and invariance figures below are historical
> development observations without retained final-revision MPI logs. They
> describe the intended gates, not current release qualification; rerun them on
> the selected public revisions before citation.

Node-local: each contact node handled by the rank owning it. **Every rank must call a collective
op** (the `VecScatter`) even with zero owned contact nodes — guarding it deadlocks the others.
**Use a reproducible parallel solver (`superlu_dist`, the default), not MUMPS** — MUMPS's
parallel pivoting is non-reproducible and an ill-conditioned contact mode amplifies it.
- **Rigid distributed contact** is node-local (obstacle known everywhere → no cross-rank coupling);
  historical development gates for **penalty return-map friction**
  (`solve_distributed(contact=…)`) recorded rank invariance.
- **Distributed DEFORMABLE contact** (`solve_distributed(deformable_contact=…)`) is the genuinely
  cross-rank case: a pair couples a secondary *vertex* with a primary *edge* that may live on
  different ranks. Replicate the contact **surface** (O(surface) ≪ ndof, a `VecScatter`, not an
  all-gather of U); each rank assembles its **owned** secondaries with global dof indices; PETSc
  off-process `ADD_VALUES` routes the edge-node contributions. A historical
  **penalty full-solve** gate compared ranks with an independent serial oracle.
  **Barrier** adds one new distributed primitive —
  the **global CCD step bound** (each rank's `max_step` reduced to the global min, petsc4py
  `Vec.min()`, never mpi4py). The cross-rank logic is one shared helper (`_DistDeformableContact`),
  used by both the quasistatic and dynamics drivers.
- **Distributed DYNAMICS** (`solve_dynamics_distributed`) is how the **penetration-free deformable
  barrier** runs distributed — it does NOT converge quasistatically (residual-norm line search stalls
  at the projection flip; ppf has no energy-merit line search, it relies on dynamics + a PSD/CCD step
  — see `docs/dev/contact_experiments.md`). Implicit backward-Euler incremental potential: node-local
  inertia `M/dt²` + Rayleigh damping + gravity, ghosted bulk, the shared contact helper, a
  **CCD-bounded predictor**, and the global-CCD step. The historical two-block
  gate recorded penetration-free, rank-invariant runs; the serial-dynamics
  oracle is a sanity
  number (~1e-2 drift — ppf-style no-backtrack vs serial backtrack — not an exact gate).
- **Distributed deformable friction development path** (smoothed) — runs under
  MPI and has historical rank-invariance gates. It needed **no
  new distributed plumbing** (only a parameter, `mu` in the spec): the shared helper already threads
  `mu`/`friction_eps` into each rank's owned-secondary op (friction is extra residual/tangent on the
  *same* pairs the barrier already routes cross-rank) and advances `_x0` in `commit`. The hard
  distributed parts (cross-rank assembly, global-CCD, dynamics) were built for the barrier; friction
  reuses them. Rank-independent because the per-step U is, so the path-dependent `_x0`=X+U follows
  identically on every rank. Scope by model: penalty deformable ✓, barrier primitives ✓, barrier
  full-solve (dynamics) ✓, deformable friction ✓ — the full deformable-contact stack now runs
  distributed.

## Convergence gotchas (diagnose a contact-solve stall HERE first, before the linear solver)
The contact stack converges, but the failure modes are specific — match the symptom:
- **Friction stalls at `|R| ~ 1e-4`, won't tighten (many Newton iters).** The smoothed (ppf/IPC)
  friction tangent is **Gauss-Newton — it drops `dλ`**. This bites at **MODERATE slip-per-step**
  (`ut` a few × `friction_eps`): `λ=μλ_n/ut` and the dropped term is *comparable* to the kept `λP`
  (spurious flow-direction stiffness), so Newton is only linear and floors. Fix: set
  `friction_eps ≳ expected slip` (near-stick, `dλ≈0`, near-exact tangent). NOTE — **this is NOT a
  large-slip limitation**: the residual reproduces `f = μN` at all slip, and large *steady* slip is the
  *easy* regime (`tangent ~ μλ_n/ut → 0` → friction → a near-constant force). The genuine narrow costs
  are stick-side (creep `~ε`, not exact lock), this moderate-slip band, and an `O(dt)` direction lag for
  *turning* slip. `μ` sets holding strength but has a ceiling — too-stiff `μλ_n/eps` re-stalls. Reach
  for the **return-map** (`RigidContact`, exact stick, non-symmetric) when you need exact static stick /
  sharp transitions — **not** for large slip (rate-dependent large-slip physics → a rate-and-state law).
- **Barrier stalls quasistatically.** It *only* converges under dynamics (inertia regularizes the
  indefinite barrier tangent) or a true energy-Armijo line search — never a residual-norm line search.
  Use `solve_dynamics*`, not `solve_increments`/`solve_distributed`, for the barrier.
- **Newton step frozen, `|R|` constant for all iters, `ccd_alpha ≈ 2⁻³⁰` (≈9.3e-10).** That α is the
  CCD bisection **floor** ⇒ a pair is at **gap ≤ 0** (penetrating) at the iterate, so CCD can't keep it
  positive and clamps the whole step to ~0 → `U` doesn't move, residual is constant. Cause: **fixed-κ
  capacity** (`~κd̂²`) exceeded by the load → the gap collapsed. Fix: pass `mass=` so the barrier uses the
  **adaptive `s=κ+M/d²`** (the inertial `M/d²` → ∞ as gap → 0 = unbounded capacity); needs dynamics. This
  is a *capacity* lock, NOT chatter — read the α value to tell them apart.
- **Solve oscillates without descending, `ccd_alpha=1.0` (no friction needed).** **Active-set / closest-
  feature chatter** — the closest edge/feature (node↔edge, which edge at a vertex) is re-found *every
  iteration*, so a node at a vertex between two edges flips its pairing each iterate → R oscillates. **Fix
  (DONE 2026-06-26): `freeze_pairing=True`** on `DeformableBarrierContact2D` (+ the distributed spec) — the
  node→closest-edge topology is fixed per step and re-searched only at `commit()`. This is the **standard
  node-to-segment approach** (Abaqus/Wriggers/Laursen: contact search is per-increment, not per-iteration).
  Per-iteration re-pairing — the old default — half-adopted IPC's "re-evaluate the set" *without* IPC's
  all-primitive smooth barrier that makes it safe. (IPC/ppf don't freeze; they avoid the discrete
  closest-edge choice entirely. For *our* node-to-segment barrier, freeze is the textbook fix.)
- **Slow / ill-conditioned, or flings a body out.** `κ`: match `κ ~ K_bulk` (100×+ ratio
  ill-conditions); adaptive `s=κ+M/gap²` over-repels at a small gap (fixed `κ` for a gentle rest).
- **"It just won't converge" — check the LOAD first.** `εg=ρgL/G ≳ 0.5` (or any non-physical
  deformation) has no converged equilibrium; that's physics, not the solver. A held-top + barrier
  overlap can't sustain a normal load (relaxes to `gap=d̂`, `λ_n≈0`) — friction needs sustained gravity.
- **Scale/solver limit.** Direct only (`superlu_dist`, NOT MUMPS — non-reproducible on slip near-null
  modes). Indefinite barrier + non-symmetric friction tangents have no iterative preconditioner yet →
  large 3D contact is direct-bound (the open scaling problem).

## `RigidContact` quasistatic compression: use adaptive load stepping

`ring_compress` ships as a RESEARCH workflow. Its proprietary input deck, raw
external run, and complete environment record are not retained, so the
historical comparison numbers below are tuning notes rather than release
validation.

`RigidContact` (penalty + return-map friction) can exercise soft-body
compression between rigid plates with adaptive load stepping. A fixed
increment can drive the Newton iteration onto an asymmetric collapse branch
(the ring top pole slides inward, contact shrinks to a few nodes, reaction
approaches zero). The example uses a user-level stepper that cuts on
non-convergence and grows on convergence.

Practical recipe from `examples/ring_compress/reproduce.py`:
- Penalty `k` and tangential stiffness `k_t` should be **matched to the load scale**, not
  maximally large. With `k=1e6` the penetration is ~1e-7, the return-map friction cap
  `μ·k·|g|` is ~0.025 per node, and the body slides along the plate instead of sticking.
  With `k=1e4` penetration is ~5e-6, the friction cap is ~50× larger, and the symmetric
  compressed solution is recovered.
- Start with a modest increment (e.g. `|Δy|=0.05` for the top plate). Accept the step if
  `solve_increments` converges in ≤ ~20 Newton iterations; otherwise halve the increment and
  retry. Grow the increment by ~1.2× after successful steps, capped at the initial size.
- Keep only the physical Dirichlet BCs (Abaqus anchors). Do **not** add symmetry BCs to
  suppress collapse — that masks the real issue. Adaptive stepping is the fix.

## Dynamic-relaxation ring compression: penalty vs ppf-style barrier (`examples/ring_compress/reproduce_dynamics.py`)

A separated dynamics version of the ring compression example uses `solve_dynamics` /
`solve_dynamics_adaptive` and runs both `RigidContact` (penalty) and
`RigidBarrierContact` (cubic barrier) against the same moving top plate.
Historical no-hold runs sampled an oscillating transient and must not be cited
as quasistatic agreement. The useful protocol lesson is to compute the
structural timescale, use a ramp-hold-settle sequence, and sample only after a
kinetic-energy convergence check.

**Barrier path.** Much stiffer and more sensitive:
- The barrier requires **dynamics** (it does not converge quasistatically).
- The contact nodes must start with **positive gap** — offset both plates by `±dhat`.
- The moving plate must update in **`max_step`** as well as in residual/tangent/commit, or the
  predictor can tunnel through it.
- `M/d²` must be floored to avoid overflow as the gap shrinks toward machine precision.
- The barrier band needs to be soft enough for dynamic relaxation: for this demo
  `dhat=0.20`, `kappa=5e1`, `friction_eps=1e-1` lets the adaptive solver accept steps in
  ~20 Newton iterations.

The example ships as RESEARCH because the code illustrates a staged
ramp-hold-settle workflow, but the input-deck provenance and raw comparison
record are incomplete. Its historical percent differences are not a public
validation claim. See the dated investigation in `docs/lessons_learned.md`;
compute the structure's `omega_1` before choosing ramp time and damping, and
sample settled states only.

## Moving rigid obstacles and `RigidContact` friction

`RigidContact`'s return-map friction stores stick anchors in the **spatial frame** (the
committed node position `x_prev` and committed tangential force `f_t`). A *moving* rigid
obstacle therefore does **not** generate the tangential friction needed to drag a body with
it — the solver sees the node's motion relative to its own previous spatial position, not
relative to the obstacle.

Workarounds:
- For a known obstacle displacement (e.g. the Abaqus top plate shifted horizontally by 2
  units), identify the ring nodes in contact with the obstacle and prescribe that
  displacement directly on those nodes. This is the stick-limited equivalent of frictional
  dragging; see `examples/ring_compress/reproduce.py`.
- For a true moving-obstacle friction model, use the barrier stack with dynamics (the
  obstacle's velocity enters the relative slip) or a dual-multiplier solver for 2D
  small-strain problems.

## Rigid 3D Cattaneo / Abaqus comparison discipline

`examples/cattaneo_3d` is excluded from the public artifact: its reviewed
end-to-end gate is a strict expected failure and the retained solve is marked
non-converged. The private diagnostic still supplies useful comparison
discipline:

- The recorded Abaqus comparison path is **nodal penalty `RigidContact`**. Do not use its result to
  accept or reject the separate ppf/barrier track; that method needs its own scoped comparison and
  convergence criteria.
- Same mesh is not same model. Abaqus used `C3D8R`; the CoupFE run used the compiled
  F-bar/full-integration Hex8 kernel. Abaqus hard/augmented contact is also not the same enforcement
  as finite nodal penalty.
- Do not cite its historical radius, load, or stick/slip values. Re-establish
  them only after a converged, refinement-qualified rerun with retained inputs
  and outputs.
- A high penalty (`k=5000`) caused active-set chatter; a lower penalty is a model change, not just
  a solver setting. If Abaqus parity is needed, match contact enforcement (Abaqus penalty settings or
  CoupFE augmented/hard contact) before tuning.
- Stick-radius extraction is a separate post-processing problem. A correct Hertz radius does not
  validate nodal stick/slip classification; use radial/profile interpolation and report transition
  resolution.

## Don'ts
- Don't complex-step a discrete switch; freeze it.
- Don't use the log barrier (ill-conditioned, NaNs).
- Don't ship a half-baked adaptive κ (the naive stiffness-match makes it worse — verify on a
  hard case before claiming it helps).
- Don't expect machine-precision quasistatic capacity from a finite barrier — that's what
  dynamics (or a hand-sized κ) is for.
- Don't gate correctness on CS-vs-FD alone — use an independent physical oracle.
- Don't let the predictor / warm-start / BC ramp bypass the CCD bound (it tunnels).
- Don't measure penetration against a flat proxy when the opposing surface deforms (phantom fails).
- Don't carry a stateful operator (`InertiaOperator`, friction) across two runs — build a fresh
  one per run, or its end-state poisons the next solve.
- **Don't re-derive battle-tested geometry/CCD from scratch — PORT it.** 3D distance/closest-feature
  (point-triangle, edge-edge) and CCD have known degenerate cases a fresh derivation quietly gets
  wrong (a hand-rolled point-triangle marked *foot-outside* as inactive → missed edge/vertex contact
  → penetration). Port `ppf-contact-solver` `contact/distance.hpp` (`*_unclassified` coeffs) +
  `accd.hpp` and verify against an independent oracle (brute-force closest distance). Re-derive only
  the genuinely new *integration* (barrier+dynamics+friction, the operator contract); the standard
  sub-kernels under it are a port. (`coupfe/operators/contact3d.py`.)

## Quantitative contact benchmark recipe (by convergence)

To validate a contact solver against an analytic oracle:
- **Grade the mesh at the contact** (`parabolic_block_mesh(beta=...)`) — a small patch is
  starved on a uniform mesh and the load won't converge.
- **Gate the physical reaction or traction directly** rather than an implementation proxy.
- **Fit the expected distribution** (for Hertz, `p²` is linear in `x²`) to obtain robust
  half-width and peak-pressure estimates plus an R² shape check; don't trust
  `max|x|` or a single peak node.
- **Gate convergence, not a band** — `b`/`p0`/load mesh-stable, error shrinking
  (`skills/testing.md`). Test a mesh **finer than the one you tuned to** and reject
  `converged=False`; otherwise a tuned or unfinished result can look validated.
- **Match the loading history for path-dependent friction.** Reproduce the oracle's
  normal-then-tangential sequence instead of replacing it with a one-shot solve.
- **Compare like with like.** Match material, interface, and dimensional assumptions
  to the analytic model before calling a stable deviation a solver bug.
