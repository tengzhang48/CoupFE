# Contact: accuracy, convergence, and the Abaqus comparison (standing note + benchmark plan)

Status: **standing assessment + a benchmark plan whose timing is deliberately open.** The favorable
claims below are **hypotheses to be measured**, not established results — "more robust / more scalable
than Abaqus" must be *benchmarked into evidence, not asserted* (project rule). This note exists so the
comparison is framed correctly when we do run it, and so the honest gaps are visible now.

Companion reading: `docs/dev/contact_experiments.md` (the measured tuning chains), the friction
convergence lesson in `docs/lessons_learned.md` (2026-06-24), and `docs/dev/ppf_contact_analysis.md`
(why we adopted the ppf model).

---

## 1. What our method is, and what it trades

Our contact is a faithful **numpy port of ppf-contact-solver** (ZOZO): a **cubic barrier** + **CCD**
step bound for guaranteed non-penetration, and **ppf/IPC smoothed friction** (PSD Gauss-Newton). The
deliberate trade (detailed in the lesson):

- **Exact:** closest-feature geometry, CCD, and the barrier/friction **forces (residual)**.
- **Guaranteed:** non-penetration (CCD step bound), at every step — *not* stiffness-dependent overlap.
- **Approximated (tunable, vanishing):** the friction **tangent** (drops `dλ` → PSD descent, no energy
  line search), and the stick–slip transition (mollified by `friction_eps`).
- **Cost of the approximation (calibrated — NOT a large-slip deficiency):** the smoothed residual
  reproduces `f = μN` at **all** slip magnitudes, and large *steady* slip is the **easy** regime
  (`tangent ~ μλ_n/ut → 0` → friction → a near-constant force). The real costs are narrow and live
  elsewhere: (1) a **Newton convergence-sensitivity band at *moderate* slip-per-step** (`ut` a few × `ε`,
  where the dropped `dλ` injects spurious flow-direction stiffness) — tunable via `ε`/sub-step/damping;
  (2) **stick is a regularized creep** (`~ε`), not exact lock; (3) an `O(dt)` direction lag for *turning*
  slip. Use the **return-map** friction (`RigidContact`, exact stick, non-symmetric) when you need exact
  static stick / sharp transitions / no cyclic creep — **not** for large slip. (A rate-and-state /
  Stribeck law is the orthogonal move when large-slip friction is genuinely *rate-dependent*.)

We run this **double precision on CPU + a direct solver**, so we pay none of ppf's single-precision
(GPU) cost; our only sacrifice vs. an exact-tangent method is the friction tangent + the stick-slip
transition.

---

## 2. How good are we vs. Abaqus? (honest, un-benchmarked)

Abaqus/Standard contact is the 40-year industry gold standard: surface-to-surface, penalty /
augmented-Lagrange / direct Lagrange enforcement, finite-sliding, general 3D self-contact, friction,
thermal/electrical contact, mature direct + iterative solvers, massive parallel scale. We are a young
research stack. The honest matrix:

| Axis | Abaqus/Standard | CoupFE contact | Read |
|---|---|---|---|
| **Non-penetration** | penalty leaks ∝ 1/stiffness; "hard"/Lagrange enforces it at a convergence cost | **guaranteed** by CCD, every step | **us (structural advantage)** |
| **Large-deformation self-contact** | known to *chatter*; needs stabilization/contact controls | IPC-class barrier — the regime IPC/ppf were *invented* for | **us (hypothesis — the upside)** |
| **Parallel reproducibility** | not guaranteed bit-identical across rank counts | **rank-independent to machine precision** (1e-16) | **us** |
| **Differentiable / inverse / coupled** | contact is a black box | operator contract + dual-home UEL → differentiable | **us (the moat)** |
| **Scale (DOF, # contacts)** | 10M+ DOF contact routine | tiny demos (≈16 hexes); **direct-bound, un-benchmarked** | **Abaqus (big gap)** |
| **Generality** | many elements, thermal/electrical, surface smoothing, corner/edge handling | vertex-face + edge-edge, flat-surface-tested | **Abaqus** |
| **Large-slip kinetic friction (`f=μN`)** | penalty/Lagrange handles it | **reproduced** — residual exact at all slip; large *steady* slip is the *easy* regime | **wash** |
| **Exact static stick / sharp stick-slip transition** | exact (Lagrange/aug-Lagrange) | smoothed default = regularized creep (`~ε`); the **return-map** path gives exact stick | **Abaqus (mild — we have the path)** |
| **Maturity / messy real cases** | 40 yrs hardened (sharp corners, intermittent, mixed) | research-stage | **Abaqus** |
| **Validation** | decades of experiments + benchmarks | unit/integration tests only | **Abaqus** |

**The reframe (your point, and it's the right one): Abaqus *also* has well-known contact convergence
pain** — "too many attempts," negative-eigenvalue warnings, contact chattering, and the routine need
for *automatic stabilization / contact damping / contact controls* to get complicated contact to
converge at all. So the comparison is **not** "us (fragile) vs. Abaqus (flawless)." It is **two methods
with different failure modes**, and ours has a *structural* robustness advantage exactly where Abaqus
is weakest: **guaranteed non-penetration + PSD descent in large-deformation / self-contact**, with no
hand-tuned stabilization. Our convergence difficulties (the *moderate*-slip friction band, active-set
chatter) are real but **bounded and diagnosable** (see the convergence notes); Abaqus's are real and often require
expert babysitting. The honest summary: **we are likely competitive-to-better on robustness in the
modern large-deformation regime, and clearly behind on scale, generality, and maturity.** All of that
is a hypothesis until benchmarked.

---

## 3. Benchmark plan (TIMING DELIBERATELY OPEN)

We will compare — but **not yet**, and that is a considered choice (a premature benchmark on a tiny,
narrow stack would mis-measure both methods). What to run *when the time is right*:

**Cases (each has an Abaqus answer + an independent oracle):**
1. **Hertzian contact** (sphere-on-half-space): contact pressure vs. the *analytic* Hertz solution —
   accuracy, not just agreement-with-Abaqus.
2. **Block under friction** (prescribed normal + tangential): friction force vs. Coulomb `μN`. Large
   *steady* slip should reproduce `μN` cleanly (the easy regime); the honest stress tests of our
   smoothed default are the **stick / stick-slip transition / cyclic-reversal** cases (where it creeps
   `~ε` and the return-map gives exact stick) and the **moderate-slip + stiff-normal** convergence band.
3. **Self-contact / large deformation** (a buckling/folding structure, ironing, or a compressed
   foam): the regime where Abaqus chatters — measure *robustness* (iterations, failures, stabilization
   needed) and non-penetration, not just the final shape. **This is where we expect to look best.**
4. **A 3D multi-body collision at scale** (many elements): parallel scaling + reproducibility.

**Metrics (robustness-first, speed last):**
- **Accuracy:** vs. the analytic/independent oracle (Hertz, Coulomb), not Abaqus-as-truth.
- **Non-penetration:** max interpolation/overclosure (we guarantee 0; Abaqus penalty leaks).
- **Robustness:** Newton iterations, # failed increments, **whether stabilization/contact-damping was
  required** to converge (a fair, telling axis — Abaqus often needs it; we aim not to).
- **Reproducibility:** 1-vs-N rank result drift (we = machine precision).
- **Scale / wall-clock:** *reported, not headlined* — speed is hardware-dependent;
  the methodological story is robustness +
  exactness + differentiability.

**Prerequisites that make it the right time (the triggers, not a date):**
- The Fortran/GPU **per-pair kernel port** lands (numpy is the verified oracle today, but too slow for
  a fair scale benchmark) — so we benchmark a *production-shaped* solver, not the reference oracle.
- Enough **generality** to express the cases (≥1 more element type; a non-flat self-contact case).
- A concrete **need** pulls it: a paper claim, a customer/consulting question, or a design decision
  that the comparison would settle. Benchmark to answer a question, not for its own sake.

Until those, the standing position is: **the favorable claims are hypotheses; cite the structural
arguments (guaranteed non-penetration, PSD descent, exact parallel reproducibility, differentiability)
as *reasons to expect* an advantage in the large-deformation regime, and say plainly that scale,
generality, and validation are open.**

---

## 4. What would change this assessment
- A large-deformation self-contact case where Abaqus converges (with stabilization) and **we do not** →
  our robustness advantage is narrower than claimed; investigate (active-set chatter? barrier capacity?).
- A scale benchmark where direct-solver memory/time caps us well below Abaqus → the iterative-
  preconditioner-for-contact problem is on the critical path, not deferred.
- An **exact-static-stick / sharp stick-slip** case that matters (locking, cyclic friction) →
  prioritize the **return-map** (exact stick) path. (Large slip itself is *not* the trigger — the
  smoothed model reproduces `f = μN` there; rate-dependent large-slip physics → a rate-and-state law.)
