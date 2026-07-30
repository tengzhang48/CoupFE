# Contact — design note (new algorithm)

> **Release-evidence boundary:** exact timings, speedups, iteration counts, and
> scale figures in this dated design record are historical local observations
> without retained raw logs and a locked environment. Rerun them before
> citation. Functional and correctness gates are tracked separately.

Contact is a **separate operator**, not an element. It implements the same
`(residual, tangent, commit)` contract, contributes `R_contact` / `J_contact·v` to
the global system, and owns a *changing* interaction graph (which surface feature
touches which) that a fixed-connectivity element never has. Reference for the full
treatment: the plan, §25 (`docs/standalone_gpu_plan.md`).

## Plan & decisions (2026-06-20)

**Serial first.** The contract isolates contact as a separate `(R, tangent, state)`
operator, so the *tricky* mechanics (gap/normal conventions, the active set, the
non-smooth tangent, CCD) are nailed **serially** with harness gates before any
distribution. Most contact research/consulting is serial or modest-scale anyway.

**Implementation languages follow the workload (the contract is language-agnostic).**
Element kernels = **Fortran** (`.for`, f2py — hot regular array math + Abaqus dual-home).
Rigid / regular contact force+tangent = **vectorized numpy** (50k nodes in ~20 ms; the
complex-step tangent vectorizes — no Python node-loop). The contact **search** (BVH
broad-phase, point-triangle / edge-edge narrow-phase, CCD) is irregular tree/spatial code:
**numba is production-viable for the search for most problems** — `@njit` compiles the
irregular broad/narrow-phase loops to near-C (multicore via `prange`, even GPU via
`numba.cuda`), and it stays in Python, so we keep one language + full control (clean here,
unlike fighting FFCx in FEniCSx). It is the **default** search tool, not just a prototype;
for the vast majority of contact problems (≤ ~10⁵–10⁶ features) it is fast enough. The
caveats are array-based data (no Python objects in the hot loop — a BVH as arrays, standard)
and a one-time JIT warmup. **Rust/C++ is the escalation**, reserved for the *extreme* scale
(ppf-contact-solver does 180M contacts in Rust + CUDA — so if we go there we can port its
algorithms directly), GPU-heavy work where mature CUDA matters, or a distributable binary.
The search/manager is self-contained (geometry in → contact records / COO out), so it needs
**no** PETSc interop regardless of language. **Not** fixed-format Fortran for the search
(great for kernels, awkward at trees).

**Keep the production distributed solver contact-extensible (a design rule, recorded
now).** Distributed contact is unlike the bulk: it couples surfaces by **spatial
proximity** (contacting DOFs can live on different ranks), with **dynamic sparsity**,
needing a *separate* spatial comm layer + a global CCD min-reduction. So when the
production distributed solver is built, keep `K_bulk` (fixed sparsity) **separable** from
a dynamic `K_contact` (two PETSc mats / `MatNest`, or rebuild only the contact COO) —
**never** a monolithic rigidly-preallocated bulk+contact matrix — and prefer the
**matrix-free contact `Jv`** route (no global contact matrix → no distributed
dynamic-sparsity problem). Serial contact is unaffected; this rule is for the later
distributed co-design.

**Stage sequence:**
- **Stage 1 ✓:** rigid analytical obstacle (`HalfSpace`/`Sphere`), frictionless **penalty**,
  vectorized; complex-step tangent with the active set **frozen** from the real iterate.
  `coupfe/operators/contact.py::RigidContact` (`tests/test_contact.py`).
- **Stage 2 ✓:** deformable **node-to-segment** penalty (2D) — `DeformableContact2D`; force
  distributed equal-and-opposite to the edge nodes by `(1-ξ, ξ)`; rank-1 PSD tangent
  `k d⊗d` (frozen normal/projection = small-sliding); brute-force closest-edge search.
- **Stage 3 ✓ (general surface contact):** `SurfaceContact2D` — any vertex vs its closest
  NON-ADJACENT edge, so **multi-body and SELF-contact** (adjacency-excluded), reusing the
  node-to-segment penalty; brute-force candidates (`tests/test_contact.py`: self-contacting
  quad loop + multi-body + exclusion).
- **Friction ✓ (rigid):** incremental penalty-Coulomb on `RigidContact` (`mu>0`, `k_t`) —
  an elastic stick spring anchored at the contact point, return-mapped to the cone
  `|f_t| ≤ μ|f_n|`; **stateful** (committed tangential force, advanced in `commit`);
  complex-step tangent with the active set *and* stick/slip set frozen (non-symmetric,
  non-associative). Kernel `rigid_penalty_eval` (`tests/test_contact_friction.py`:
  stick→slip law, CS-vs-FD tangent, integrated shear, `mu=0` backward-compat).
- **Distributed rigid contact ✓:** `solve_distributed(..., contact=...)` — node-local
  per-rank contribution (each contact node on the rank owning it; friction state there).
  Rank-independent AND run-to-run repeatable to **machine precision (1.3e-16)** for real mixed
  stick/slip — **with a reproducible solver (`superlu_dist`, the default)**. MUMPS is
  non-reproducible in parallel and the slip near-null mode amplifies it (~1e-3); that's a
  *solver* trap, not a contact-physics property (lessons_learned.md, skills/distributed.md).
  `tests/test_mpi_distributed.py::test_distributed_friction_rank_independent`.
- **Distributed DEFORMABLE contact ✓ (penalty; the new cross-rank work):**
  `solve_distributed(..., deformable_contact={...})`. Unlike rigid contact (node-local, obstacle
  known everywhere), a deformable pair couples a secondary *vertex* with a primary *edge* that
  after partitioning can live on a **different rank**. Approach: **replicate the contact surface**
  (secondary ∪ edge nodes — O(surface) ≪ ndof, a VecScatter, not an all-gather of U) to every rank
  each iteration; each rank assembles the contact COO for the secondaries it **owns** (by first-DOF
  ownership → each pair once) with **global** dof indices; **PETSc off-process `ADD_VALUES` routes**
  the edge-node contributions cross-rank. The serial `DeformableContact2D` operator is reused
  verbatim. Verified rank-independent two ways: assembly (`distributed_deformable_residual`, R exact
  + K·v ~1e-13 at 1/2/4 ranks) and a **full solve** of two neo-Hookean blocks loaded only through
  the contact interface (`distributed_deformable_solve`) — gathered U matches an **independent**
  serial oracle (operator-level `solve_increments([ElementGroup, DeformableContact2D])`, a different
  assembly path) to ~1e-8 and is identical across rank counts. `tests/test_mpi_distributed.py`:
  `test_distributed_deformable_residual`, `test_distributed_deformable_solve`.
- **Distributed deformable BARRIER (penetration-free) ✓ — via distributed DYNAMICS:** the new
  distributed primitive is the **global CCD step bound** — each rank's point-edge `max_step` is
  local, but the limiting pair can be on any rank, so the line-search bound is the **global minimum**
  across ranks (petsc4py-only reduction via a 1-entry-per-rank `Vec.min()`, **never mpi4py**),
  verified rank-independent in isolation (`distributed_deformable_barrier`: global CCD == serial
  full-surface `max_step`, `|diff|=0`, < 1 so it constrains). The full barrier *solve* does **not**
  converge **quasistatically** (a serial, MPI-orthogonal issue: the residual-norm line search stalls
  at the node-to-segment projection flip — and ppf has no energy-merit line search; it relies on
  dynamics + a PSD/CCD-bounded step — `docs/dev/contact_experiments.md`). So the path is **distributed
  dynamics**: `solve_dynamics_distributed(...)` brings the implicit backward-Euler incremental
  potential to the distributed solver (node-local inertia `M/dt²` + Rayleigh damping + gravity,
  ghosted bulk, the **shared** cross-rank contact helper `_DistDeformableContact`, a **CCD-bounded
  predictor**, and the global-CCD step bound). The inertia regularizes the non-smooth contact and
  supplies the barrier's capacity. Verified (`distributed_dynamics_barrier`,
  `test_distributed_dynamics_barrier_rank_independent`): two neo-Hookean blocks collide under gravity
  through the cross-rank barrier, **penetration-free** (min gap stays in `(0, d̂)`), Newton converges
  (`|R|~1e-13`), and the gathered U is **rank-independent to machine precision** (`max|u_N−u_1| ~
  3e-16` at 1/2/4 ranks). (The serial `solve_dynamics` oracle is a *sanity* check, not an exact gate
  — the distributed driver is ppf-style with no residual backtracking, so two valid backward-Euler
  trajectories drift ~`1e-2`; 1-vs-N is the rigorous gate.) `_DistDeformableContact` is the single
  source for the cross-rank contact logic, shared by the quasistatic and dynamics drivers.
- **Distributed deformable FRICTION ✓ (rides the dynamics):** the ppf smoothed friction on the
  distributed barrier. It runs under MPI (rank-independent), and needed **no new distributed
  plumbing** — only a parameter (`mu`). The shared
  `_DistDeformableContact` already threads `mu`/`friction_eps` into each rank's owned-secondary
  `DeformableBarrierContact2D` (so the friction residual/tangent assemble cross-rank with the
  barrier) and advances the step-start friction reference `_x0` in `commit`. Friction is
  path-dependent, but because the per-step U is rank-independent the `_x0` evolves identically on
  every rank → the answer is rank-independent (the same argument as the distributed rigid friction).
  Verified (`distributed_dynamics_friction`, `test_distributed_dynamics_friction_rank_independent`):
  two blocks held by gravity, the top sheared sideways — the μ>0 interface slip is **0.38×** the
  frictionless slip (friction holds), converged + penetration-free, and the gathered μ>0 U is
  rank-independent to **machine precision** (`max|u_N−u_1| ~ 5e-16`). So the full deformable-contact
  stack — penalty, barrier, friction — now runs distributed.
- **Stage 4 — penetration-free (rigid) ✓:** `RigidBarrierContact` + `rigid_barrier_eval`
  (**cubic** barrier `(κ/3)(d̂-d)³`, bounded stiffness, no NaN — chosen over the IPC log-barrier
  for conditioning; lessons_learned.md) + a **CCD step bound** `max_step` consumed by the
  `newton_solve` line search → no iterate penetrates. Verified: repulsive force + CS-vs-FD
  tangent, CCD keeps gap>0, single-node load-balance, integrated block non-penetration
  (`tests/test_contact_barrier.py`). Fixed κ must be ~ the bulk stiffness (too stiff → line
  search stalls); adaptive κ (Ando) is the refinement.
- **Implicit dynamics ✓ (the substrate for robust contact):** `InertiaOperator` (lumped mass,
  backward-Euler) + `solve_dynamics`. After reading the ppf source, adaptive κ is a
  *capacity-vs-conditioning* problem whose clean fix (`stiff_k = wᵀ(K_elast + M/g²)w`) needs
  the **inertial** `M/g²` term — i.e. dynamics; quasistatic has no such capacity. So dynamics
  is the substrate: inertia gives barrier capacity + regularizes non-smooth transitions, and
  backward-Euler doubles as dynamic relaxation for quasistatics. Operators are driver-agnostic
  and compose unchanged. (`tests/test_dynamics.py`; `docs/dev/ppf_contact_analysis.md`.)
- **Contact-on-dynamics ✓ (basic):** a neo-Hookean block dropped under gravity via
  `solve_dynamics` (no Dirichlet — the mass regularizes the free body) free-falls, the cubic
  barrier + CCD keep it penetration-free, and with Rayleigh damping it **rests on the floor at
  gap > 0** (`tests/test_contact_dynamics.py`). The `M/dt²` diagonal regularizes the per-step
  Newton (converges where quasistatic stalled). Remaining fragility: a *hard* impact with a
  *fixed* κ still diverges.
- **Adaptive stiffness ✓ (ppf `M/g²` capacity):** `RigidBarrierContact(mass=...)` uses
  `s = κ + M/d²` — the inertial `M/d²` → ∞ as the gap closes (gap-dependent capacity, no
  hand-tuned κ; also a strong *recovery* force if the inertial predictor overshoots into
  penetration). Decisive test: a hard drop where a too-soft fixed `κ=10` crushes through (gap
  −1.83) but the *same* κ with `M/d²` rests at gap +0.006 (`tests/test_contact_dynamics.py`).
  Needs `M` → dynamic driver only. (Refinement: the `wᵀK_elast w` conditioning term over `κ`.)
- **Deformable–deformable barrier ✓ (2D node-to-segment):** `DeformableBarrierContact2D` +
  `deformable_barrier_eval` — the penetration-free version of `DeformableContact2D`. Both sides
  are DOFs: the secondary node *and* the two primary-edge nodes get the cubic barrier force, so
  two deformable bodies cannot interpenetrate. The signed gap to the edge **line** has an analytic,
  action-reaction gradient `_seg_grad_d` (the three nodal gradients sum to zero), so the barrier
  force is complex-step-differentiable → the **consistent** (normal-rotation) 6×6 tangent, verified
  CS-vs-FD + symmetric to machine precision (true energy Hessian). A **point-edge CCD**
  (`max_step`: quadratic colinearity root `e(α)×r(α)=0` + bisection safety net) stops a secondary
  crossing its edge. Adaptive `s = κ + M/d²` carries over (per secondary node). Two-body =
  compose two operators (A-nodes vs B-edges *and* vice-versa); brute-force closest-edge search.
  Verified: kernel repulsion + Σforce/Στorque = 0, CS-vs-FD tangent, CCD non-crossing, and an
  integrated **two deformable blocks** drop where both deform and rest at gap > 0 (broken control:
  no contact → interpenetrates). `tests/test_contact_deformable_barrier.py`.
- **CCD bounds the PREDICTOR too (fix in `solve_dynamics`):** the per-step inertial predictor
  `û = u_prev + dt·v_prev` is now CCD-bounded from the last accepted state, not just the Newton
  increment. A fast predictor (`dt·v > gap`) was teleporting a node straight through a barrier
  band before Newton ran. Side effect (good): a too-soft *fixed* κ no longer penetrates a hard
  impact (CCD holds for any κ) — adaptive `M/d²` is now shown to be about *capacity/conditioning*
  (it pins at gap≈0 and the solve thrashes ~18× more Newton iters without it), not non-penetration.
- **Friction on the barrier ✓ (ppf/IPC smoothed, semi-implicit) — F1/F2/F3:** added to *both*
  `RigidBarrierContact` and `DeformableBarrierContact2D` (theory `docs/theory/contact_dynamics.md`
  §3b). The barrier supplies `λ_n = s(d̂−d)²`; with `P = I−n⊗n` and slip `dx` from the step start,
  `λ = μλ_n/max(ε,‖P·dx‖)` gives residual `λ(P·dx)` and a **symmetric-PSD** tangent `λP` (`λ_n, n`
  frozen → analytic Gauss-Newton, no complex step). *Why this and not the return map:* simpler,
  stateless (reference = step-start position, dynamics already has it), PSD-conditioned — adopted
  from `ppf-contact-solver` `friction.hpp`. **Rigid (F1):** `RigidBarrierContact(mu, friction_eps)`;
  **deformable (F3):** friction opposes the *relative* slip of the secondary vs the moving foot,
  distributed by `J=[I,−(1−ξ)I,−ξI]` (action-reaction, PSD `λJᵀPJ`). `μ=0` byte-identical/stateless.
  Gates (`tests/test_contact_barrier_friction.py`, `tests/test_contact_deformable_friction.py`):
  analytic stick→slip law, `|f_t|≤μλ_n`, PSD-symmetric friction tangent, no-friction-under-
  co-translation, and **under dynamics (F2)** a block sticks below `μ·weight` / slides above
  (broken control `μ=0` slides). Caveat: it's *rate-form* (x₀ per step) → bounded sub-creep, not
  perfect static stick (a return-map anchor would be needed for that); penalty `RigidContact` keeps
  the exact return-map for the penalty path.
- **Subsequent Stage 4 work ✓:** 3D vertex-face/edge-edge contact, numba narrow phase,
  LBVH broad phase, 3D barrier self-contact, distributed deformable barrier/friction,
  and opt-in return-map/persistent friction are implemented and gated. The current
  remaining limitations are maintained in `docs/capabilities.md`.

## Exact-stick friction: smoothed vs return-map vs dual-multiplier (2026-06-24)
Three ways to enforce Coulomb friction, increasing in stick fidelity and in machinery:

1. **Smoothed (ppf/IPC)** — `f_t = μλ_n · pdx/max(eps,‖pdx‖)`, PSD Gauss-Newton tangent `λJᵀPJ`.
   Reproduces `f=μN` and is *unconditionally* robust (drops `dλ`), but stick is a regularized plateau:
   residual creep `~μN·eps`, never exactly zero. **The Gauss-Newton tangent is FD-inconsistent in slip**:
   at saturation its tangential-stiffness entry is `λ=μλ_n/u_t` (≈16 in the pilot) while the true
   derivative is **0** — this overstated slip-direction stiffness *is* the plateau floor.
2. **Return-map (radial-return)** — elastic stick spring `k_t·slip` projected onto the cone (cap `μλ_n`).
   **Forces are exact Coulomb** (verified: stick `=k_t·slip`, slip `=μλ_n`, transition at `slip=μλ_n/k_t`),
   and its **slip tangent `(μλ_n/u_t)(P−t̂⊗t̂)` is correct** — the saturated-direction entry is **0,
   matching FD** (fixes the smoothed's floor). Implemented and gated as `friction_kt=` on
   `DeformableBarrierContact2D` and the vertex-face path of `DeformableBarrierContact3D`.
   But stick is still elastic:
   micro-slip `~F/k_t`, and pushing `k_t→∞` for exact stick reintroduces a **`k̃_t = k_t L/E`
   conditioning wall**. (Both modes deliberately **freeze the barrier normal/`λ_n`/foot** in the friction
   tangent → both show the *same* ~40% full-FD mismatch on the 6×6 block; that is the frozen-normal, not
   a bug, and the smoothed converges fine with it.)
3. **Dual multiplier (Alart-Curnier / active-set)** — stick is a **constraint `v_t=0`** whose Lagrange
   multiplier is the friction force; slip puts the multiplier on the cone `|p_t|=μN`. Active set switches
   on the **utilization `η=|p_t|/(μN)`** (stick iff `η≤1`). Stick is then **exactly zero-slip** (machine
   precision) with **no `k_t` and no conditioning knob** — the only remaining parameter (the AC
   augmentation `r`) affects convergence rate, not the answer. This is the exact form; `examples/
   exact_stick_friction` demonstrates + gates it (machine-zero stick, Coulomb cap, analytic incipient
   onset `δ*=μP/k_shear`).

**Why the example uses GLOBAL Coulomb, and the partial-slip lesson.** A uniform block has no static
*partial* slip: it is all-stick until `Σ|p_t|=μΣN`, then gross-slides (force-control above that has **no
static equilibrium** — it accelerates; hence the example is *displacement*-controlled). Per-node partial
slip (Cattaneo: a stick core + slip annulus) needs **non-uniform pressure**. We confirmed the physics —
under shear the bonded **corners** carry tangential reactions exceeding `μN` while the interior sticks, so
slip *does* initiate at the edges — **but the bonded sharp corner is a stress singularity**: the naive
"switch corner to slip, free its dof, re-solve" **cascades** (an under-constrained free node slides to
garbage `v_t≫δ`). That is exactly what the **consistent semismooth Newton on `(u,p)`** is for and the
switch-and-resolve is not.

**Per-node partial slip — built (semismooth Newton, `coupfe.operators.contact_semismooth`).** The
Alart-Curnier semismooth Newton solves the saddle-point `[[K,−Sᵀ],[A_cu,A_cp]] (du,dp) = −(R_u,C)` with
the generalized Jacobian (stick → constraint `v_t=0`, `A_cp=0`; slip → `p=±μN`, `A_cp=1`). It resolves a
stick zone + slip zone consistently and **does not cascade** at the singular corner (converges in ~12
iters, res ~1e-11), giving exact `v_t=0` at every stick node and `|p|=μN` at every slip node. Two regimes
verified: small shear = press-dominated (Poisson bulge slips the edges outward, centre sticks); large
shear = shear-dominated (the stick zone shifts toward the leading side, Cattaneo-like). The **Schur
complement** condenses the interior DOFs to a small interface system `(contact tangential dofs,
multipliers)` — verified to return the same `(U,p)` as the direct augmented solve. Remaining: **linear
bulk `K`** + a **lagged/clamped normal** (no coupled Signorini, no nonlinear outer Newton), it is a
**standalone solver not yet wired into `solve_dynamics`/the distributed path**, and a smooth-geometry
**quantitative Cattaneo gate** (analytic stick radius) is still open — the flat bonded corner is a
mesh-dependent singularity (fine for the qualitative demo; the *global* criterion sidesteps it for the
clean quantitative example).

## Self-contact (3D barrier + smoothed friction, 2026-06-24)
One connected body whose surface touches *itself* (folding, crumpling, cloth, tissue). Enabled by
`DeformableBarrierContact3D(self_contact=True)` — the same vertex-face + edge-edge barriers, but with
**incident-exclusion**: a surface vertex must not contact the faces incident to it (gap ≡ 0). Without it
every vertex "contacts" its own faces and both the barrier AND the CCD `max_step` blow up on the
flat/unloaded body (an incident pair pins the time-of-impact at 0 → nothing can move).

- **vertex-face**: drop candidate pairs where the vertex is a node of the face — a **numba** kernel
  (`contact3d_numba.vf_drop_incident_nb`, per-pair branchy → the numba sweet spot) applied inside
  `vertex_face_candidates(self_contact=True)`, so the filter reaches BOTH `_contributions` (barrier) and
  `max_step` (CCD). This is the **1-ring** exclusion (the vertex's own faces); faces in the 1-ring that do
  NOT contain it sit ≥ one element-height away, so for `d̂` < element size this is complete. A deeper
  topological neighborhood (2-ring) is not filtered — fine for thin sheets, revisit for thick folds.
- **edge-edge**: already excludes shared-vertex pairs (`exclude_shared=True`); that is the edge analog.
- **friction**: the ppf smoothed model rides on the self-pairs unchanged (a self-pair is just a
  vertex-face / edge-edge pair from the same body). This is the **right model for self-contact** — many
  dynamic forming/breaking pairs, large sliding, robustness over exact stick — which is what ppf/IPC use;
  the exact-stick (return-map / semismooth) modes are reserved for single well-defined interfaces.

Why the friction *model* was the easy call and the geometry the real work: see the capabilities note.
Two-body contact is unaffected (it passes disjoint vertex/face sets, so no pair is ever incident).
Focused private-development checks cover incident exclusion, a flat-sheet
broken control, and the active operator. The hairpin end-to-end driver exceeds
its 600-second bound and is excluded from the public artifact rather than
presented as a passing gate. Remaining: a bounded end-to-end case, distributed
support, deeper-neighborhood qualification, and a smooth-geometry quantitative
gate.

## Configuration
Bulk stays **reference / total-Lagrangian** (PK1). Contact is evaluated in the
**current** configuration — gap, normal, closest-point projection are spatial. The
two coexist through the operator contract; the bulk is never converted to a
current-config formulation.

## Tangent / differentiation policy
Do **not** complex-step through the contact algorithm — it contains nonsmooth /
discrete operations (candidate-pair creation, closest-feature switches, open/contact
activation, `max`/`min`, CCD root selection). Use an **algorithmic / generalized**
contact tangent. Two explicit modes (the manifest records which):
- `psd` — the robust rank-one normal Hessian `φ''(d) n⊗n` (a Gauss-Newton/projected
  approximation; SPD-friendly for CG);
- `consistent` — adds the projection/normal/geometry derivatives (better Newton, may
  be indefinite/nonsymmetric → MINRES/GMRES).
Complex-step may still *verify* a smooth contact sub-kernel with the active set,
feature pair, and stick/slip **frozen** from the real iterate.

## Stages (increasing geometric + nonlinear difficulty)
1. **Rigid analytical SDF** (plane/sphere/cylinder), frictionless penalty/barrier.
   `n = ∇g/‖∇g‖`. **Largely a PORT** — RetroMech `contact.py` already has SDFs +
   node-to-SDF penalty, rigid-limit verified. → bucket-2 port.
2. **Deformable point-to-surface** — linear triangular primary facets, secondary
   points; `g_n = (x_s − x_m(ξ))·n_m`. **New.**
3. **Full feature contact** — point-triangle, edge-edge, point-edge, point-point
   (large sliding + self-contact); stable global feature IDs, deterministic owners.
   **New.**
4. **Cubic barrier + CCD** — `φ(d)=2/(3d̂)(d̂−d)³`; an additive continuous-collision
   step bound `α ≤ η·α_CCD` fed into the PETSc line search (MPI: global `min`
   reduction). Penetration-free. **New** — the core algorithmic piece.
5. **Friction** — regularized first, then stateful Coulomb (committed slip, return
   mapping, stick/slip set). Transactional state like a constitutive update. **New.**
6. **MPI / self-contact** — spatial (not adjacency) communication: local BVH →
   exchange bounding regions → remote feature ghosts → deterministic ownership →
   reduce to DOF owners. **New.**

## Learn from [`ppf-contact-solver`](https://github.com/st-tech/ppf-contact-solver)
Adopt the **algorithm**, not the stack: BVH broad phase, point-triangle/edge-edge
narrow phase, cubic barrier, additive CCD, PSD rank-one Hessian, dynamic contact
sparsity. **Do not** copy: their custom single-precision GPU CG (keep PETSc KSP/SNES;
double precision), their whole solver. Track the upstream commit in the manifest.

## Contract integration
- `R_contact`/COO `J_contact` summed alongside bulk (the assembler already composes
  operators; contact just adds dynamic COO records).
- `contact_max_step(U, dU) → α_max` plugs into `newton_solve`'s line search (CCD
  before energy/residual globalization).
- Transactional state: search records are trial; committed friction/multiplier
  history commits only after the increment is accepted.

## Stiffness scale
Never set the barrier stiffness from the raw (possibly indefinite) bulk tangent. Use
a **positive proxy** `k_c = ŵᵀ K_proxy ŵ`, `K_proxy ⪰ 0` (material-only or
positive-projected), with configurable bounds. Expose `fixed` / `elasticity_scaled`
/ `effective_dynamic` policies.

## Split of work
Stage 1 ports from RetroMech (bucket 2). Stages 2–6 are the new-algorithm effort
(this note's owner), built to the depth a real engagement needs.

## Broad-phase search (scheduled — the next scale piece)

**Why now.** Every contact path today (`DeformableBarrierContact2D`, `SurfaceContact2D`, the
friction) finds candidates by **brute force O(N²)** (every vertex × every edge). That caps contact
at small meshes *serially*, and it is the prerequisite for distributed contact (you can't
distribute a quadratic search — you distribute *candidate generation*). So it is item #1 on the
scale roadmap (`docs/capabilities.md`), before any distributed-contact work.

**Design (decided).** A self-contained `ContactSearch`: positions + surface vertices + surface
edges + band `d̂` → candidate `(vertex, edge)` pairs within `d̂` (+ a safety margin). It owns no
PETSc/MPI state (geometry in → pairs out), so it drops into the serial operators now and becomes
the local kernel of the distributed spatial layer later.

- **Structure: uniform spatial grid / hash first** (not BVH). FE surface features are roughly
  uniform in size, so a uniform grid with cell ≥ `max_edge_len + d̂` is simple, cache-friendly,
  and O(N) for the common case; **BVH is the escalation** for highly non-uniform feature sizes
  (adaptive meshes). Conservative cell size ⇒ **no candidate within `d̂` is ever missed** (the
  brute-force result is the correctness oracle).
- **Tooling: vectorized numpy → numba.** numpy first (correct + decent), then `@njit`/`prange`
  for the hot bin/scan loop (the recorded default; Rust/CUDA only at the extreme). Array-based
  bins (no Python objects in the hot loop).
- **Integration: transparent.** The operators consume candidate pairs instead of scanning all
  edges; the active-set/gap/force/CCD logic is unchanged. Self-contact reuses it with the
  adjacency exclusion (vertex not an endpoint of its candidate edge).

**Phased build (each gated; brute-force is the oracle):**
- **S1 ✓ — uniform-grid broad-phase (numpy).** `coupfe/operators/contact_search.py::candidate_pairs`
  returns per-vertex candidate edges; **superset** of the brute-force within-`d̂` set by construction
  (no missed contact). `tests/test_contact_search.py`: superset (uniform + jittered), incident
  exclusion, far-bodies-don't-bridge, ~linear-not-quadratic scaling.
- **S2 ✓ — wired into `DeformableBarrierContact2D`.** `_pairs` restricts the closest-edge scan to
  the band-`d̂` candidates. **Provably identical to brute force** (an active pair has point-segment
  distance < `d̂` ⇒ it's in the band-`d̂` superset ⇒ same closest edge + force; a vertex with no
  in-band edge contributes nothing either way) → O(N²)→O(N) on the hot residual/tangent path with
  the deformable barrier + friction suites passing **unchanged** (19 tests).
- **S2b ✓ — `max_step` (CCD) broad-phased.** Band = `d̂ + 2·reach` (`reach` = max nodal
  displacement) so it catches an edge the secondary could *cross this step*; each secondary is
  CCD-checked against **all** its candidates, not just the closest (a correctness upgrade — the old
  closest-only `max_step` could miss a crossing of a non-closest edge). So `DeformableBarrierContact2D`
  is now fully O(N) (residual/tangent + CCD). `tests/test_contact_deformable_barrier.py::
  test_ccd_catches_non_closest_crossing_edge`.
- **S2c ✓ — penalty operators broad-phased.** `SurfaceContact2D` (self-contact) + `DeformableContact2D`
  now use `candidate_pairs` (a `search_band`, default = max edge length). Penalty has no fixed `d̂`,
  so the band is a *heuristic* covering penetration depth + reach (identical to brute force while
  penetrations stay within band — unlike the barrier's *provable* band-`d̂`). The broad-phase now
  covers **all 2D contact operators**. (`tests/test_contact_search.py::
  test_penalty_operators_broadphase_equals_bruteforce`.)
- **numba 3D — DONE (2026-06-24).** The dominant cost was the Python **per-pair barrier loop**
  (~80 µs/pair, >99% interpreter/dispatch overhead). Ported numba-native (`contact3d_numba.py`):
  `tri_barrier_eval_nb` (vertex-face cubic barrier + ppf smoothed friction) + `closest_faces_nb`,
  gated **bit-for-bit** vs the numpy oracle (rel R 6.6e-17) — **284×** (0.27 µs/pair). And a numba
  **LBVH** (`bvh_numba.py`, ppf `lbvh` template; Morton-sorted *iterative* midpoint split, ppf-compatible
  node layout → GPU Karras = direct escalation) replaces the uniform grid (non-uniform-robust), gated
  EXACT vs brute force (miss=0, extra=0). `numpy = oracle, numba = production` (grid + per-pair numpy =
  the identical fallback). **Fortran NOT used for contact** — the dual-home (in-Abaqus) rationale that
  justifies Fortran for element kernels does not apply (contact is CoupFE-only). 2D `candidate_pairs`
  numba + numba edge-edge/ACCD are the remaining follow-ups.
- **Later:** 3D (vertex-face / edge-edge), BVH for non-uniform meshes, and the distributed spatial
  broad-phase (exchange bounding regions → remote feature ghosts → deterministic owners) — which
  is then item #2/#3 on the scale roadmap.

## 3D deformable contact (Stage 5 — PORTED from ppf-contact-solver)

3D needs two primitives (2D had one): **point-triangle** (vertex-face) and **edge-edge** (skew
segments, new in 3D). The robust closest-feature geometry + CCD are error-prone with many degenerate
cases, so we **port** ppf's battle-tested code rather than re-derive (a hand-rolled point-triangle
marked foot-outside as inactive → missed edge/vertex contact; see `lessons_learned.md`).

- **`coupfe/operators/contact3d.py`** — faithful numpy ports (Apache-2.0):
  - `point_edge` / `point_triangle` / `edge_edge` **distance coefficients** + `*_unclassified`
    closest-feature classification (face→edge→vertex; the 4-endpoint edge-edge fallback) [ppf
    `contact/distance.hpp`].
  - `tri_barrier_eval` / `edge_edge_barrier_eval` — cubic barrier on the robust closest-point gap
    `|p_a−p_b|`, residual `−s(d̂−gap)²·[weighted n]`, **PSD** rank-1 tangent, action-reaction
    (Σforce=0, Στorque=0). Penetration-free is the CCD's job (unsigned gap), as in ppf.
  - **ACCD** `accd_toi` + `point_triangle_toi` / `edge_edge_toi` (conservative advancement, centerize
    + max-relative-velocity) [ppf `contact/accd.hpp`].
- **Verified** (`tests/test_contact3d.py`, 9): closest distance == brute-force oracle in all regions,
  both barriers action-reaction+PSD, off-edge degenerate active, ACCD prevents crossing (both
  primitives; broken control = full step crosses).
- **3D broad-phase ✓** (`vertex_face_candidates` + `edge_edge_candidates`, uniform 3D grid + AABB,
  superset-verified) and the **3D operator ✓** `DeformableBarrierContact3D` (both primitives over the
  broad phase + ACCD `max_step`; integrated test: a vertex dropped onto a fixed triangle rests above
  it, no penetration).
- **3D friction ✓** (`tests/test_contact3d.py`, +6): the ppf/IPC smoothed model on the contact
  normal's 2D tangent plane `P = I − n⊗n`, a faithful generalization of the 2D node-to-segment
  friction. Both primitives (`tri_barrier_eval`, `edge_edge_barrier_eval`) gain `mu/eps/X0`; the
  friction force `λ(P·dx)` (`λ = μ λ_n / max(ε,‖dx‖)`) opposes the relative tangential slip since the
  step start, mapped to the 12 DOFs by the kinematic Jacobian `B` whose blocks are the same signed
  weights the barrier uses (`Σcᵢ=0` → action-reaction + zero friction under rigid co-translation, by
  construction). Symmetric-PSD tangent `λ BᵀP B`; shared helper `_smoothed_friction_3d`. The operator
  `DeformableBarrierContact3D` gains `mu`/`friction_eps` and tracks step-start positions `_x0`
  (advanced in `commit`); `mu=0` byte-identical/stateless. Verified: `|f_s|=μλ_n` at full slip,
  stick(sub-ε)→slip(plateau), action-reaction, zero under rigid/normal motion, PSD tangent, and the
  stateful operator (friction resists a tangential drag, then vanishes after `commit` advances `_x0`).
  This completes the implemented two-body 3D vertex-face/edge-edge friction
  path; it is not a claim that self-contact, every degeneracy, or every
  distributed configuration is fully qualified.
- **NOT needed — degenerate dedup:** checking ppf settled this — ppf does **no** type-classification/
  dedup and **no** edge-edge mollifier (unclassified closest distance + cubic barrier per candidate
  pair; same `i<j` + shared-vertex exclusion we use). The mild double-count at exact
  vertex-vertex/vertex-edge coincidences is benign and present in ppf too; building a dedup would
  diverge from ppf for a measure-zero gain. Our 3D edge-edge is a faithful ppf port.
- **3D distributed contact ✓ (cross-rank assembly + global CCD):** `_DistDeformableContact3D` (a
  subclass of the 2D helper — the substantive methods are op-agnostic and inherited) brings 3D
  contact to the distributed solvers. Both 3D primitives are partitioned: **vertex-face** by the
  secondary vertex (passed as the op's `vertices`), **edge-edge** by the first node of each pair's
  first edge (`DeformableBarrierContact3D(owns_edge_pair=…)`, applied in both `_contributions` and
  `max_step`), so every pair is assembled by exactly one rank; PETSc off-process `ADD_VALUES` routes
  the off-rank stencil nodes; the CCD bound reduces per-rank `max_step` to the global min (`Vec.min`).
  `solve_dynamics_distributed` dispatches 3D when the spec has `"faces"`. Verified rank-independent in
  isolation (`distributed_3d_primitives`, `test_distributed_3d_primitives`): with an active
  vertex-face AND edge-edge pair, gathered R + K·v == the serial full operator (both `0.0` diff) and
  the global CCD == serial `max_step` (`|diff|=0`, `<1`) at 1/2/4 ranks.
- **3D distributed dynamics end-to-end ✓ (with a bulk element):** a stencil-only / mass-point (no-bulk)
  system is degenerate for the solve (direct LU fails at engagement; PETSc SEGVs on the empty objects),
  so the end-to-end run needs a real 3D **bulk element**. With the vendored **F-bar Hex8** the system is
  well-conditioned (as in 2D) and `solve_dynamics_distributed(deformable_contact={...,"faces":…})` runs
  it directly. Two demos, both rank-independent to machine precision at 1/2/4 ranks:
  `distributed_dynamics_3d_blocks` (two Hex8 blocks **collide**, frictionless) and
  `distributed_dynamics_3d_friction` (two Hex8 blocks seated + **sheared**, μ>0 friction holds — slip
  0.64× frictionless). So the **full deformable-contact stack — bulk + barrier + friction — now runs
  distributed in 2D AND 3D**, all rank-independent (`test_distributed_dynamics_3d_blocks_rank_independent`,
  `test_distributed_dynamics_3d_friction_rank_independent`).
- **Friction convergence note (smoothed/ppf under a Newton solve):** tune `friction_eps` to the
  interface **slip scale**, not arbitrarily small. In the slip plateau (`ut ≫ eps`) the Gauss-Newton
  friction tangent drops `dλ`, which there is comparable to the kept `λP` term → Newton floors at a
  moderate `|R|` (no tight convergence); set `eps ≳ ut` (near-stick, `dλ≈0`, near-exact tangent) and
  restore holding strength via `μ` (which has its own convergence ceiling — too-stiff `μλ_n/eps`
  re-stalls). See `docs/dev/contact_experiments.md` (2026-06-24).
- **Remaining 3D:** degenerate vertex-vertex / vertex-edge dedup (benign, ppf does the same — deferred);
  a serial non-flat geometry exercising edge-edge friction at scale; the Fortran/GPU per-pair kernel
  port for production scale (numpy is the verified oracle).

## Parallelization: MPI vs GPU, and the Fortran/GPU port path (for future agents)

**ppf is single-GPU CUDA, NOT MPI.** Re-checked: ppf "runs both contact and elasticity solvers on
the GPU," reaches **180M contacts on one modern NVIDIA GPU** in *single precision*, with an **LBVH**
(radix-sort GPU broad-phase). There is **no MPI / NCCL / multi-GPU** in it. So ppf scales by *thread
parallelism on one device*, not by distributing across ranks.

This is a different scaling axis from CoupFE's, and the two are **complementary**:
| | CoupFE (today) | ppf |
|---|---|---|
| Parallelism | **MPI** (petsc4py) across ranks/nodes — the bulk FE solve | **single-GPU CUDA** threads |
| Precision | double (direct/iterative PETSc) | single |
| Scales by | many nodes / big memory (huge meshes) | one big GPU (huge *contact counts*) |
| Contact cross-rank | the hard part (spatial proximity couples ranks → the distributed broad-phase) | a non-issue (all in one GPU's memory) |

Pick MPI when the *mesh* is too big for one node; pick GPU when the *contact count* is the wall.
For distributed contact CoupFE still needs the cross-rank spatial layer (the surface-replication
broad-phase is the start); ppf sidesteps that entirely by staying on one device.

**The Fortran/GPU port path — and why it's clean.** The contact **per-pair kernels** (distance
coefficients, the barrier residual/tangent, ACCD) are **embarrassingly parallel**: each candidate
pair is an *independent* unit of work (the numpy versions are literally a Python loop over pairs).
So the current numpy `contact3d.py` is the **verified reference / oracle**, and two ports follow the
same per-pair math with no algorithmic change:
- **Fortran (f2py)** — vectorize/OpenMP the per-pair loop, same dual-home story as the element
  kernels (`CompiledElement`). Moderate speedup, stays in the CoupFE runtime, double precision.
- **GPU (CUDA, ppf-style)** — one thread per pair + an LBVH broad-phase; this is *literally ppf*, so
  if we go there we port its kernels directly (we already ported its math to numpy). Extreme scale.

The discipline that makes this safe: the **operator contract** (`residual/tangent/commit/max_step`)
is the serial O(ndof) spine and does **not** change; only the per-pair kernel implementation moves
to Fortran/GPU, gated against the numpy oracle (bit-for-bit on the same pairs). Build correctness in
numpy first (done), port for speed second (deferred until scale demands it — the numpy + grid
broad-phase is already O(N) and fine for serial/modest-scale research, which is most of the use).
