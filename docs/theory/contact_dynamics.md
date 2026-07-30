# Theory — contact, barrier, friction, and implicit dynamics

The mathematical formulation behind CoupFE's contact + dynamics. Companion to the operational
guide `skills/contact.md`, the design note `docs/dev/contact.md`, the ppf study
`docs/dev/ppf_contact_analysis.md`, and the narrative `docs/lessons_learned.md`.

Notation: `u` = global DOFs, `R(u)` = residual (the source of truth), `K = ∂R/∂u` = tangent.
Physical force = `−R` (the residual is the energy gradient). For a contact node, `x = X + u`
is the current position, `g(x)` the signed gap (>0 separated), `n(x)` the outward obstacle
normal (`∂g/∂x = n`).

## 1. The operator contract and the complex-step tangent

Every contribution — bulk element, contact, inertia, load — is an operator with
`(residual, tangent, commit)`. The global system sums them; one driver solves any mix.

**One residual is the source of truth; the tangent is derived from it by complex step:**
```
K_ij = Im( R_i(u + i·h·e_j) ) / h ,      h ~ 1e-30
```
which is exact to machine precision (no subtraction cancellation) **provided `R` is analytic**.
The only analyticity constraint: use `√(Σ vᵢ²)` not `|v|`, and avoid branch cuts.

**Non-smooth physics** (contact active set, stick/slip, contact-pair creation) is handled by
**freezing the discrete decision from the real iterate**, then complex-stepping the resulting
*smooth branch*. This yields the consistent tangent of that branch — never complex-step the
switch itself.

## 2. Normal contact — penalty

Penalty energy for penetration, `Π_c = ½ k g²` when `g < 0`. Residual and tangent:
```
R = ∂Π_c/∂u = k g n ,        K = k n⊗n + k g ∂n/∂u
```
The active set `{g < 0}` is frozen from the real iterate; within it the force `k g n` is
analytic, so complex step recovers `K` including the curved-obstacle geometric term `k g ∂n/∂u`
automatically. Signs are unambiguous because they come from the energy gradient.

## 3. Friction — incremental penalty-Coulomb (return mapping)

An elastic *stick* spring anchored at the contact point, return-mapped to the Coulomb cone —
the exact analogue of plastic return mapping, hence **stateful**.

Per step, tangential slip increment `Δs = P_t(x − x_prev)` (projected to the current tangent
plane `P_t = I − n⊗n`). Trial tangential force:
```
f_t^trial = P_t f_t^committed + k_t Δs ,        cap = μ |f_n| = μ k |g|
```
- **stick** (`|f_t^trial| ≤ cap`): `f_t = f_t^trial`;
- **slip** (`|f_t^trial| > cap`): `f_t = cap · f_t^trial / |f_t^trial|`.

Contact residual = `k g n + f_t`. The active set *and* the stick/slip set are frozen; within
the frozen branch the force is analytic (in slip `|f_t^trial| > 0`), so complex step gives the
consistent tangent. Friction is **non-associative** → the tangent is **non-symmetric** (the
solver must handle it). The committed `f_t` is per-node state advanced in `commit`; it is
node-local, so it distributes (one rank owns each contact node's state).

## 3b. Friction on the barrier — smoothed / lagged (ppf / IPC style)

The return map (§3) is the right model on the *penalty* path. On the **barrier** path (§4) we use
the smoothed, semi-implicit friction of IPC / the `ppf-contact-solver`, which is *simpler* and
gives a **symmetric, PSD** tangent. It reuses what the barrier already produces.

The barrier gives a normal-force magnitude `λ_n = |f_n| = s(d̂−d)²` (with the adaptive
`s = κ + M/d²`, §7). Let `dx = x − x₀` be the tangential slip **measured from the step start**
`x₀` (no sliding anchor — under implicit dynamics `x₀` is just `u_prev`, §6), and `P = I − n⊗n`
the tangent-plane projection. Then, with `ε` a small slip tolerance (`friction_eps`):
```
λ   = μ λ_n / max(ε, ‖P·dx‖)
R_t = λ (P·dx)            (friction residual contribution; physical force = −R_t opposes the slip)
K_t = λ P                (friction tangent — symmetric PSD)
```
One smooth expression *is* the whole stick/slip law:
- **slip** (`‖P·dx‖ ≥ ε`): `|R_t| = μ λ_n` — full kinetic Coulomb along the slip direction;
- **stick** (`‖P·dx‖ < ε`): `|R_t| = μ λ_n ‖P·dx‖ / ε` — a linear spring;

and `|R_t| ≤ μ λ_n` always — the cone is enforced by the mollifier `max(ε, ·)`, with **no branch,
no return map, no cone projection**.

**Why this beats the return map (and why ppf scales).**
- **No stateful anchor.** The reference is the step-start position `x₀` (dynamics already carries
  it); the return map's *sliding* anchor + committed `f_t` are gone.
- **SPD tangent.** `λ_n` and `n` are held fixed while forming `R_t`/`K_t` (the IPC "semi-implicit /
  PSD-projected" friction — `K_t = λP` drops `∂λ/∂x`, `∂n/∂x`), so `K_t = λP ⪰ 0` is **symmetric**
  and CG-friendly. The return map's normal–tangential coupling is non-symmetric (§3).
- **Capacity rides the barrier for free.** `λ_n = s(d̂−d)²` carries the adaptive `M/d²` (§7), and
  `λ_n → 0` at the band edge, so friction fades smoothly with light contact. The only knob is `ε`.
- **Analytic** — closed-form `R_t`, `K_t`; no complex step needed for friction.

**Why it is so simple — and the approximations.** The simplicity is the *payoff* of one idea:
Coulomb friction is **non-associative** (the slip direction is not normal to the cone), so it is
**not the gradient of any potential** — which is precisely what makes the *exact* friction tangent
non-symmetric (§3). The smoothed model **approximates friction by an associative, potential-derived
law** `D(u_t) = μλ_n · φ(‖u_t‖)`; once friction is a potential, its gradient `R_t` and Hessian `K_t`
are trivial and symmetric. Three approximations buy that potential structure:

1. **Lag `λ_n`, `n` (semi-implicit).** Holding the cone cap and normal fixed within the friction
   differentiation **decouples** friction from the normal problem (removes the normal–tangential
   cross terms) → a potential in `u_t` alone. *Error:* a one-step-lagged `λ_n`; shrinks with `dt`,
   or vanishes if the lag is iterated to a fixed point. **This one affects the converged answer**
   (unless iterated).
2. **Mollify the cone (`max(ε, ‖P·dx‖)`).** True stick is exactly zero slip (and the static force
   is indeterminate — non-smooth). The `max(ε,·)` replaces stick with a stiff *reversible* creep up
   to `~ε`. *Error:* `~ε` of fake elastic slip in stuck regions; smaller `ε` → nearer true Coulomb
   but stiffer. **Affects the model.** (The return map's `k_t` stick spring is the *same*
   approximation, parameterized as a stiffness instead of a length.)
3. **PSD-project the Hessian (`K_t = λP`, drop `∂λ/∂x`).** In the slip regime the *true* Hessian is
   `μλ_n/‖u_t‖·(P − t̂t̂ᵀ)` — rank-deficient (no stiffness along the slip direction). Using the full
   `λP` adds back `λ t̂t̂ᵀ` → a clean SPD over-estimate. This is **Gauss–Newton**: it changes only
   the Newton *path*, not the fixed point (`R_t` is the exact smoothed residual), so it costs strict
   quadratic convergence but **not** accuracy — free robustness.

So this model is **not "more approximate" than the return map (§3)** — it bundles differently: both
regularize stick (#2), but the return map keeps the *exact, non-symmetric* tangent (and freezes the
stick/slip set), whereas this swaps that for **lag (#1) + PSD (#3)** → a symmetric, SPD, stateless,
analytic tangent. The trade: it gives up exact Coulomb coupling and strict quadratic convergence,
and gains symmetry (CG-friendly), no state, and rock-solid conditioning at scale. **Under implicit
dynamics (§6) the lag is exactly the regime where it is stable** — why it fits the barrier+dynamics
path. For a tight quasistatic stick problem needing exact Coulomb, the non-symmetric exact tangent
is the fallback. Source: `ppf-contact-solver`
`crates/ppf-cts-solver/src/cpp/energy/model/friction.hpp` (+ call site `.../contact/contact.cu`),
Apache-2.0 — adopt the algorithm, not the stack.

## 4. Penetration-free contact — the cubic barrier

Barrier energy on the active band `d < d̂` (`d̂` = activation distance), 0 beyond:
```
B(d) = (κ/3)(d̂−d)³ ,   B'(d) = −κ(d̂−d)² ,   B''(d) = 2κ(d̂−d)
```
Residual `R = B'(d) n = −κ(d̂−d)² n` (repulsive), bounded stiffness `B'' ≤ 2κd̂` (→0 at `d̂`).
C² (so the force is C¹ → continuous tangent), and **polynomial → never NaN**, even at `d ≤ 0`
(a penetrating node gets a large *finite* push-out).

**Why cubic, not the IPC log barrier** `−(d̂−d)² ln(d/d̂)`: the log → ∞ as `d→0` (self-
guaranteeing non-penetration) but its stiffness `~ 1/d` is **unboundedly ill-conditioned** and
it `NaN`s for `d ≤ 0`. An ill-conditioned tangent both amplifies solver non-reproducibility
and stalls Newton (see §8). The cubic trades the energy-∞ guarantee for bounded conditioning —
and recovers the non-penetration guarantee from CCD instead.

## 5. CCD — the continuous-collision step bound

Non-penetration is enforced by **bounding the line-search step**, not by the energy → ∞.
`max_step(u, du)` returns the largest `α ∈ (0,1]` keeping every contact node's gap positive
along `u + α du`:
```
gap(α) ≈ g₀ + α (Δx·n)        (linear; exact for a flat obstacle)
α ≤ η · (−g₀ / (Δx·n))        for approaching nodes (Δx·n < 0), η ≈ 0.9
```
with a bisection safety net re-checking the true gap (covers curved obstacles). The driver
starts the line search at `min(1, minₒₚ max_step)` then backtracks — so **no iterate ever
penetrates**, regardless of barrier shape. (The cubic also degrades gracefully if CCD is
imperfect, because it stays finite under penetration.)

## 6. Implicit dynamics — backward-Euler

Per time step, in the incremental-potential (backward-Euler) form with predictor
`û = u_prev + dt·v_prev`:
```
R(u) = M/dt² (u − û) + F_int(u) + F_contact(u) − F_ext = 0 ,     v = (u − u_prev)/dt
```
solved by Newton each step (warm-started from `û`). Mass-proportional **Rayleigh damping**
adds `+ α M v = α M (u − u_prev)/dt`. The inertia is a plain operator (`M/dt²` diagonal); the
element/contact/load operators compose with it **unchanged**.

Three reasons dynamics is the right substrate for robust contact:
1. **Conditioning** — the `M/dt²` diagonal regularizes the per-step Newton (small `dt` → mass-
   dominated, well-conditioned), so it converges where the quasistatic solve stalled.
2. **Regularization** — inertia smooths the non-smooth stick/slip and active-set transitions.
3. **Capacity** — it supplies the barrier's gap-dependent capacity term (§7).

**Dynamic relaxation:** with a load held and damping on, the dissipative integrator settles to
the **quasistatic** equilibrium — a robust way to reach static solutions too.

## 7. Adaptive stiffness — capacity vs. conditioning

A *fixed* κ faces two competing constraints:
- **capacity** — the cubic's maximum force `κd̂²` must exceed the contact reaction, else the gap
  collapses to the wall (lower bound on κ);
- **conditioning** — the barrier stiffness must not dwarf the bulk, else the Newton step is
  ill-conditioned and the line search stalls (upper bound on κ).

The ppf resolution (read from `barrier/barrier.cu::compute_stiffness`): keep the barrier
*geometry-normalized* (κ ≡ 2/d̂, no free κ) and scale its force/Hessian per contact by
```
stiff_k = wᵀ (K_elast + M/g²) w
```
where `w` is the (weighted, normalized) contact direction, `K_elast` the local **elasticity
Hessian** projected onto `w` (→ conditioning), and `M/g²` an **inertia** term that → ∞ as the
gap closes (→ capacity), recomputed every evaluation. Conditioning from elasticity, capacity
from inertia.

**Crucially `M/g²` is inertial** — it exists only under dynamics. In a *quasistatic* solve
(`M = 0`) `stiff_k → wᵀK_elast w` and the capacity bound is unavoidable: that is why a naive
quasistatic "match κ to the bulk stiffness" fails (it controls conditioning but not capacity),
and why **dynamics is the substrate** that makes adaptive stiffness well-posed.

## 8. Cross-cutting themes

- **Ill-conditioning is the enemy on two fronts** — it amplifies solver non-reproducibility
  (e.g. MUMPS parallel pivoting; use a reproducible solver like superlu_dist) *and* stalls
  Newton. Prefer the well-conditioned formulation (cubic over log; κ matched / adaptive; the
  `M/dt²` regularization) even when the ill-conditioned one is "more standard."
- **Freeze the discrete, complex-step the smooth.** Active set, stick/slip, contact pairs are
  frozen from the real iterate; the per-branch force is analytic and complex-stepped.
- **Composability** — element + contact + friction + barrier + inertia + load are all the same
  contract, so quasistatic Newton, load-stepped Newton, and implicit dynamics are three drivers
  over the *same* operators.

## 9. 3D deformable contact — ported from ppf-contact-solver

3D needs **two** contact primitives where 2D needed one (node-to-segment):
- **point-triangle** (a vertex vs a triangular face) — the analog of point-segment;
- **edge-edge** (two skew segments' closest approach) — *new in 3D*; without it, contacts that no
  vertex-face pair catches are missed.

The robust geometry (closest-feature with all degenerate fall-throughs) and the CCD are
error-prone and well-trodden, so they are **ported faithfully from `ppf-contact-solver`**
(`contact/distance.hpp`, `accd.hpp`, `aabb.hpp`, Apache-2.0), not re-derived (a hand-rolled
point-triangle that required the foot *inside* the face silently drops edge/vertex contact — see the
lesson). `coupfe/operators/contact3d.py`.

**Distance coefficients (closest-point weights).** Instead of a plane distance, both primitives use
the **closest-point distance** `gap = |p_a − p_b|` where `p_a = Σ wᵃ·(feature A nodes)`,
`p_b = Σ wᵇ·(feature B nodes)`. The weights come from the ported `*_distance_coeff_unclassified`:
- point-triangle → barycentric `(w₀,w₁,w₂)`; if the foot leaves the face it falls back to the
  closest **edge** → **vertex** (so a vertex-edge / vertex-vertex contact is found, not dropped).
- edge-edge → `(a₀,a₁,b₀,b₁)` via a 2×2 solve + a 4-step alternating-projection refinement + a
  four-endpoint-case fallback when the interior solution leaves `[0,1]`.

**Barrier.** Mirrors 2D on the closest-point gap: with `n = (p_a−p_b)/gap`, the residual over the
feature's nodes is `−s(d̂−gap)²·grad`, `grad = [n weighted by ∂p/∂node]` (e.g. point-triangle
`[n, −w₀n, −w₁n, −w₂n]`); PSD tangent `2s(d̂−gap)·gradᵀgrad`. Action-reaction holds by construction
(Σforce = 0; Στorque = 0 since `p_a−p_b ∥ n`). The gap is **unsigned** (a distance), so
penetration-free is the **CCD's** job, not a signed recovery — exactly as in ppf.

**ACCD (additive continuous collision).** `accd_toi` is conservative advancement: centerize the
feature points (translation invariance), compute the max pairwise relative speed `u_max`, then
advance the time-of-impact `toi += (d−target)/u_max` until the closest distance reaches the band
`target = offset + reduction·(d₀−offset)`. Returns the max safe step fraction; the operator's
`max_step` is the **min toi over all candidate pairs** (point-triangle + edge-edge).

**Broad phase.** Per-feature AABB with a `d̂` margin (ppf `aabb.hpp`); ppf uses an LBVH on the GPU,
we use a **uniform 3D grid** (BVH is the extreme-scale escalation). Two pair streams: vertex-face
and edge-edge, each a *superset* of the true within-`d̂` set.

**Operator.** `DeformableBarrierContact3D` composes vertex-face (closest face per vertex) +
edge-edge (active non-adjacent pairs) over the broad phase, assembles the 12-DOF residual/tangent,
and `max_step` = the min ACCD toi. **3D friction** is §3b's smoothed model on the contact normal's
2D tangent plane (`P = I − n⊗n`): the friction force `λ(P·dx)` (`λ = μ λ_n / max(ε,‖dx‖)`, `λ_n` =
the barrier normal force) opposes the relative tangential slip `dx` of the contact-point pair since
the step start, mapped to the 12 DOFs by the kinematic Jacobian `B` whose blocks are the same signed
weights the barrier uses for the gap gradient (`grad = Bᵀn`): vertex-face `[1,−w₀,−w₁,−w₂]`,
edge-edge `[a₀,a₁,−b₀,−b₁]`. Because `Σ cᵢ = 0`, the friction is action-reaction (`Σf = 0`) and
vanishes under a rigid co-translation by construction; the tangent `λ BᵀP B` is symmetric PSD (the
same Gauss-Newton / lagged-`λ_n` / mollified-cone approximations as 2D). The operator tracks the
step-start positions `_x0` (advanced in `commit`); `μ=0` is byte-identical/stateless. MVP caveat:
mild double-count at exact vertex-vertex/vertex-edge coincidences — but **ppf does the same** (it
uses the unclassified closest distance per candidate pair, no type-classification/dedup, no
edge-edge mollifier — the cubic barrier avoids the log-barrier's parallel-edge gradient blowup). It
is benign (measure-zero; CCD still guarantees non-penetration), so we match ppf and do not dedup.
