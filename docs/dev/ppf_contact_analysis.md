# ppf-contact-solver analysis → implications for CoupFE contact

Grounded in the actual source of ZOZO's
[`ppf-contact-solver`](https://github.com/st-tech/ppf-contact-solver)
(Apache-2.0), read 2026-06-21. It is our
algorithm reference for robust contact. This note records what it does and what it means for us.

For a reproducible source baseline corresponding to CoupFE's adaptations, use
upstream commit `8b7740b032131aeeb46f51d882c96e09b171acc8`. Retained evidence
does not prove that this was the exact checkout read in 2026, so the repository
does not make that stronger claim. The source mapping is:

- `contact.py` → `barrier/cubic.hpp`, `barrier/barrier.cu`,
  `energy/model/friction.hpp`, `contact/contact.cu`, and the 2-D use of
  `contact/distance.hpp`;
- `contact3d.py` → `contact/distance.hpp` and `contact/accd.hpp`, with the
  barrier/friction sources above;
- `contact3d_numba.py` → a Numba translation of those `contact3d.py`
  adaptations; and
- `bvh_numba.py` → `contact/aabb.hpp`, `lbvh/lbvh.hpp`, `lbvh/lbvh.cu`, and
  `data.hpp`, with its midpoint build explicitly retained as a local
  simplification.

## What ppf is
GPU (CUDA) + Rust, **penetration-free** contact for FEM deformables (shells/solids/rods),
**single-precision on the GPU**, scales to ~180M contacts, with strict strain-limiting.
C++/CUDA core in `crates/ppf-cts-solver/src/cpp/`: modules `contact`, `barrier`, `energy`,
`eigenanalysis`, `strainlimiting`, `lbvh` (GPU BVH broad-phase), `plasticity`, `solver`
(matrix-free CSR). It is an **implicit dynamics** solver.

## The cubic barrier (`barrier/cubic.hpp`) — confirms our choice
```
energy(g)    = -2(g-ĝ)³ / (3ĝ)        # g = gap, ĝ = activation distance ("ghat")
gradient(g)  = -2(g-ĝ)² / ĝ
curvature(g) =  4(1 - g/ĝ)            # bounded; → 0 at ĝ, = 4 at g=0
```
Default `barrier = "cubic"` (choices cubic/quad/log). Bounded curvature, polynomial (no NaN).
**It is geometry-normalized — there is NO free stiffness κ** (κ ≡ 2/ĝ is baked in). The
stiffness is applied separately.

## The dynamic stiffness (`barrier/barrier.cu::compute_stiffness`) — the key idea
The barrier gradient/Hessian are scaled by a **per-contact scalar** (`contact.cu:213`):
```
stiff_k = wᵀ (K_elast + M/g²) w
```
- `K_elast` = the **actual local elasticity Hessian** blocks of the contact participants
  (`hess(index[i], index[j])`), projected onto the contact direction `w` (barycentric-weighted,
  normalized). → matches the barrier to the material stiffness (**conditioning**).
- `+ M/g²` = an **inertia term that blows up as the gap `g → 0`**, recomputed every evaluation.
  → automatic stiffening / unbounded **capacity** at close range. (Static-side mass 0 is
  replaced by the max participant mass so a rigid obstacle still "feels inertia"
  — `barrier.cu:64`.)

So: **conditioning from elasticity, capacity from inertia/`g²`.** The two requirements that
fought each other in our quasistatic attempt are reconciled *by dynamics*.

## Why our quasistatic adaptive κ failed (the lesson)
We tried `κ = c·k_bulk/d̂` and **froze it** (once-set). That matched stiffness (conditioning)
but had **no gap-dependent capacity** — the cubic's max force `κd̂²` was below the load, so the
gap collapsed. The missing piece is precisely ppf's `M/g²` term (recomputed, → ∞ as `g→0`).
But that term is **inertial** — in a true quasistatic solve there is no mass, so
`stiff_k → wᵀK_elast w` and the capacity bound (`~ stiffness·d̂`) is unavoidable. The capacity
problem is a **quasistatic artifact**, not a barrier-design flaw.

## Implementation outcome: dynamics is the contact substrate
ppf (and IPC) are dynamic for good reasons — inertia (a) supplies the `1/g²` barrier capacity
for free, and (b) regularizes the non-smooth stick/slip and active-set transitions that stall a
quasistatic Newton. CoupFE now has an **implicit-dynamics driver** and lumped
mass operator as peers to the quasistatic Newton. The contact
operators (`RigidContact`, `RigidBarrierContact`, CCD `max_step`) are driver-agnostic
(`residual/tangent/commit/max_step`) and plug in unchanged; the adaptive
stiffness `stiff_k = wᵀ(K_elast + M/g²)w` works as intended. Keep quasistatic too (same
operators, fixed matched κ) for genuinely static problems; a lighter stop-gap is a pseudo-mass
/ dynamic-relaxation `c/g²` capacity term without full time integration.

## CPU adaptation completed; extreme-scale work deferred

The CPU/Numba path now includes LBVH broad phase, ACCD for deformable contact,
implicit dynamics, and barrier/friction integration. Deferred work is the GPU
LBVH/per-pair port, broader Hessian projection where needed, and a matrix-free
GPU solver—only when scale justifies that separate qualification.

## Superseded next steps

The former first two steps—implicit dynamics and barrier/friction with adaptive
stiffness—are implemented and covered by focused tests. Quasistatic remains for
static loading with a fixed matched κ. Future work is qualification at larger
scale, not completion of those CPU features.
