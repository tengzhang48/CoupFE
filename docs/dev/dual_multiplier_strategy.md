# Dual-multiplier (exact-stick) friction — where it's worth it, and the generalization decision

Standing strategy note (2026-06-24). Companion to `docs/dev/contact.md` (the three friction modes) and
`docs/dev/contact_vs_abaqus_benchmark.md`. Written to settle a recurring question: **should we invest time
generalizing the semismooth-Newton dual-multiplier solver into a real contact solver, and if so, toward
what?** Conclusion up front: **not toward forward generality; only toward differentiable exact partial-slip
for a fretting/joint niche — and only if that application is in scope.**

## What we have

`coupfe/operators/contact_semismooth.py` — Alart–Curnier (1991) semismooth Newton + Schur-condensed
interface. Exact Coulomb: stick is a **constraint** `v_t=0` (its Lagrange multiplier is the friction
force), slip puts the multiplier on the cone `|p|=μN`; per-node partial slip resolved consistently
in the bounded linear-bulk/fixed-pair example (`examples/semismooth_friction`
and its focused gates). This is a research implementation of a published
method, not a novel method or a claim of equivalence to a commercial solver.

Reference: P. Alart and A. Curnier, “A mixed formulation for frictional contact problems prone to
Newton like solution methods,” *Computer Methods in Applied Mechanics and Engineering* 92(3),
353–375 (1991), DOI `10.1016/0045-7825(91)90022-X`.

## Honest assessment — two axes

- **Strength inside the demonstrated model:** algebraic stick, a Coulomb-cone
  multiplier, and coexisting stick/slip nodes can be resolved without a
  penalty stiffness. Focused examples check these properties; they do not
  establish accuracy across contact geometries, material laws, or load paths.
- **As a *contact solver*: narrow.** Linear / small-strain bulk (`K` frozen at `u=0`, no finite-strain
  re-linearization), small-sliding (fixed contact pairing `S`), **lagged/known normal** (no coupled
  Signorini multiplier), rigid/fixed counter-surface, standalone driver (not in `solve_dynamics` /
  distributed), no barrier/CCD penetration-free guarantee.

Contrast: the **ppf smoothed stack** (the mainstream CoupFE path) has much of the generality the dual-multiplier lacks
— finite strain, large sliding (re-pairs each step), deformable–deformable, 3D, distributed, barrier+CCD,
self-contact — but is **approximate on stick** (rate-form, creeps `~μN·eps` per step).

## Where the dual-multiplier may be useful (a narrow research hypothesis)

Exact stick may be advantageous when **the quantity of interest is the micro-slip amplitude
or dissipation at the stick–slip boundary, accumulated over cycles** — exactly where smoothed creep
can bias the result:

| Problem | Potential exact-stick QOI | Risk with smoothed/penalty treatment |
|---|---|---|
| **Fretting fatigue** | partial-slip annulus (stick core + slip ring); slip amplitude → crack nucleation | smears the stick radius `c/a` |
| **Bolted / frictional joints** | micro-slip hysteresis loop area (energy dissipated/cycle), joint softening | mis-sizes the loop |
| **Cyclic / ratcheting / fretting wear** | accumulated slip over many cycles | spurious creep accumulates |
| **Cattaneo–Mindlin validation** | closed-form stick radius | can't hit the analytic radius cleanly |

For gross sliding, crash, cloth, and large-deformation biomechanics, the
barrier/smoothed path is presently the more complete implementation. Which
formulation is more accurate remains problem- and evidence-dependent.

## The insight that changes the cost calculus

Many candidate fretting/joint studies use small-strain bulk models, which may
make the current linear-elastic scope useful. That assumption must be checked
for a selected application; it does not remove the fixed-pair, lagged-normal,
and coupled-contact limitations.

## The generalization decision

- **Toward a general forward exact-contact solver → NO.** That competes with Abaqus' mature
  Lagrange-multiplier friction on its home turf, where we have no edge. Dead end; rebuilds a known method.
- **Toward differentiable exact partial slip for fretting/joints → a candidate
  research direction.** The combination may be underserved, but that is a
  positioning hypothesis requiring a current literature/tool survey. The
  checked-in solver has no semismooth adjoint and does not establish this
  capability.

### Outcome that buys something
- **Inverse identification** of friction fields in fretting/joints from measured slip or dissipation.
- **Gradient-based design** of frictional joints (maximize dissipation / minimize fretting damage) with an
  *exact* stick–slip model.

These are potential outcomes, not current capabilities or claims about what
other tools cannot produce.

### Work required (scope still to be established)
Small-strain bulk is done. The real work is: (1) the **differentiable adjoint through the semismooth active
set** (extend the RetroMech path-adjoint from return-map to the AC multiplier); (2) a **Cattaneo–Mindlin
validation** (the analytic stick radius); (3) a small/moderate-sliding contact kinematics adequate for
fretting amplitudes. A focused project.

## Decision gate

Invest **only if** a fretting-fatigue / frictional-joint application is in scope for CoupFE. If yes → the
differentiable-partial-slip path is high-value; scope it concretely. If CoupFE's targets stay
large-deformation / biomech → the dual-multiplier remains a **proof + friction-model testbed**, ppf stays
production, and the time goes to the **Abaqus benchmark** (validate the scalability claim) instead.

Either way: **do not generalize it for forward generality.** Its value is the exact active set, and the
exact active set pays off in *inverse/design*, not forward simulation.

## Method lineage — where this sits in the literature (2026-06-25)

Recurring confusion ("I can't find this method named anywhere"): it's because (a) we run **two different
contact tracks** that belong to **different families**, and (b) the family our exact-stick solver belongs to
goes by **at least five names**.

### We have two tracks, only one is in this family
- **Track A — the barrier/penalty stack** (`DeformableBarrierContact2D`/`...3D`, the ppf cubic barrier; the
  production default, distributed). This is a **smooth-penalty / barrier** method (IPC-style: a potential
  that → ∞ at contact, Newton + line-search + CCD, **no multiplier, no active set**). It is *not* a
  semismooth/active-set method. Its nearest literature cousin is the **interior-point / barrier** route
  (e.g. contact-as-constrained-minimization → IPOPT in FreeFEM), not the methods below.
- **Track B — `contact_semismooth.py`** (the dual-multiplier exact-stick solver). This **is** the
  semismooth-Newton active-set family, verbatim: Alart–Curnier NCP residual, generalized (semismooth)
  Jacobian whose **stick/slip branches are the active set**, Lagrange multiplier = the dual.

### Track B is one method under five names
By **Hintermüller–Ito–Kunisch (SIAM J. Optim. 2003)**, *the primal-dual active-set strategy IS a semismooth
Newton method*. So these are the same family seen from different angles — search one, miss the others:
**primal-dual active set** · **semismooth Newton** · **generalized Newton** (Renard) · **NCP-function /
nonsmooth Newton** (Alart–Curnier, Fischer–Burmeister) · plain **active set** (broadest; not every
active-set method is semismooth Newton — the equivalence is for PDAS-with-the-NCP/max-function).

Turnkey implementations of *exactly* Track B's method:
- **PETSc** `SNESVINEWTONSSLS` (semismooth) / `SNESVINEWTONRSLS` (reduced-space active set) for VIs with
  variable bounds — anything on SNES (FEniCS/Firedrake, deal.II, MOOSE) can drive it.
- **deal.II** step-41 (obstacle, PDAS, cites HIK 2003) / step-42 (elastoplastic contact).
- **GetFEM** (Renard): Alart–Curnier + generalized Newton — the closest turnkey academic twin of ours.
- **Kratos** `ContactStructuralMechanicsApplication`: dual-mortar + active-set/NCP (Mataix Ferrándiz,
  Cornejo).
- **Wohlmuth** dual-mortar PDAS; **Gitterle–Popp–Gee–Wall** finite-deformation frictional mortar with
  semismooth Newton + **consistent linearization** (in-house BACI → open successor **4C**).
- **PermonQP / FLLOP** (PETSc): the dual-multiplier posed as a **QP** (MPRGP / TFETI) at HPC scale.

Commercial codes (Abaqus/Standard, ANSYS, Marc, COMSOL) use the **same effective active-set update** but
brand it by the *enforcement* scheme (penalty / augmented-Lagrangian / Lagrange-multiplier) wrapped in an NR
loop that re-evaluates contact status each iteration. Abaqus' **"severe discontinuity iterations"** are the
visible analog of the active-set change.

### Alternative QP/LCP formulations

Alternative QP/LCP contact formulations remain a private research direction.
They are not shipped in CoupFE core; the retained core path is the
Alart–Curnier semismooth solver described above.

### What is standard, and what is ours
- **Published foundation:** the forward algorithm follows the
  Alart–Curnier semismooth-Newton/PDAS family. The focused gates check this
  implementation only within its bounded model; they do not make it a
  commercial-solver equivalent or a generally qualified contact method.
- **Possible future edge — differentiability:** applying an adjoint to the
  Alart–Curnier residual is a research direction, not a checked-in capability.
  The separate return-map relay example has a local grad-vs-finite-difference
  gate, but that result must not be attributed to the semismooth solver.
- **Our gaps (be honest):** we **lag the normal force N** — so the tangential semismooth Newton is exact but
  the normal↔tangential coupling is staggered, **not** the consistent linearization of
  Gitterle–Popp–Gee–Wall; plus serial, small-strain bulk, node-to-segment (not mortar). Same method, behind
  on scale / generality / consistency.
