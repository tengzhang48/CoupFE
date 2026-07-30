# Build log — reproducing a contact benchmark (the GetFEM tire)

A **worked process**, written so a future contributor can follow the *method*, not just the result. The
companion `docs/dev/contact_benchmark_plan.md` is the *what*; this is the *how* — the model decisions,
the implementation, and the analysis/verification at each step, with the reusable lesson called out.

> **Release status: RESEARCH.** Only the torus-mesh construction currently has
> a reviewed pytest gate. The solve, stress, and sensitivity numbers below are
> historical development observations without a retained raw run, environment,
> or exact GetFEM source/result record. Phase marks mean “implemented,” not
> release-validated, and no value in this log should be cited as a benchmark
> match.
>
> **The one-line method:** build the simulation **bottom-up and verify each layer independently before
> composing** — element → mesh → contact-solve → analysis → gradients. Contact solves are where bugs
> hide; a wrong element or mesh is nearly undiagnosable *through* a failing contact Newton, so each
> layer earns an independent gate first. (Why contact is hard: `skills/contact.md`.)

## The order that works (generalizable)
1. **Pick the model honestly and state the caveats** — don't over-claim a reproduction.
2. **Element first, verified independently** (compile ≠ correct).
3. **Mesh, verified** (positive Jacobian + exact dimensions) before any solve.
4. **The solve = contact on dynamics** (the robust substrate).
5. **Analysis** — the QOI (Von Mises) + an honest comparison to the reference.
6. **The edge** — differentiable sensitivity (what the reference can't do).

Threaded through all of it: **search before building** (don't rebuild what exists), **independent
oracle per layer**, and the convergence-gate discipline (`skills/testing.md`).

---

## Phase 0 — the element, implemented as RESEARCH (`f511a6a`)
**Model decision.** Rubber is near-incompressible. The default anti-locking element, **F-bar**, is
*wrong here*: its `(J̄/J)^{1/d}` rescale NaNs when the centroid `J̄` crosses 0 under bending/shear —
exactly the tire's regime (`skills/pitfalls.md`). So: **mixed u-p** (element-constant condensed
pressure `p = K·avg(lnJ)`, mean dilatation), condensed to a u-only system. The current checks do
not establish general inf-sup or inversion robustness; the element ships only
as a RESEARCH workflow, not a supported/default formulation.

**Implementation.** The 3D sibling of `examples/neo_hookean_local_pressure_quad4`: the material
(`stress_PK1`, `pressure_resid`) is dimension-agnostic (3×3 `F`); only `ndim=3` and
`ELEMENT_CONFIGS["hex8"]` change. `examples/neo_hookean_local_pressure_hex8/build.py` generates +
compiles it (NDOFEL=24, one condensed-p SVAR). *Search-before-build win:* `hex8` was already in
`ELEMENT_CONFIGS` and the local-pressure generator already supports Hex8 — no codegen work needed.

**Internal development checks (not in the public test partition).**
`tests/test_neo_hookean_local_pressure_hex8.py` checks:
- **pressure condensation** — uniform `F` ⇒ condensed `p == K·log(det F)` (analytic, independent);
- **undeformed is stress-free** with a symmetric (energy-Hessian) tangent;
- **isochoric bulk-modulus independence** — at an isochoric `F`
  (`det=1` ⇒ `p=0`) the internal force is unchanged across the tested `K`
  values. This is an implementation invariant, not proof of general inf-sup,
  anti-locking, or inversion robustness.

**Lesson.** Verify the element on an *independent* property (analytic value, an invariant) before
trusting it. The codegen `verify()` only checks tangent self-consistency (CS-vs-FD) — a *consistent*
element can still be *wrong*.

---

## Phase 2a — the mesh, reviewed gate (`examples/tire_contact/mesh.py`)
**Model decision.** A **hollow thick-walled torus** standing in the xz-plane (axle along y), tread
down. Tire-like, meshes cleanly (an annular cross-section has no polar-centre degeneracy a solid disk
would), and robust. **Linear Hex8 + a few layers through the wall** (here 2) is the adaptation for
*linear* elements — a single linear layer can't bend (GetFEM gets bending from one Q2 layer; we trade
order for stability + a finer mesh).

**Implementation.** A structured parametric torus —
`node(φ,θ,ρ) = ((R+ρcosθ)cosφ, ρsinθ, (R+ρcosθ)sinφ)` — φ around the wheel, θ around the tube, ρ
through the wall; φ,θ periodic. Hex corners wound for a **positive Jacobian** (auto-flipped + checked).
*A parametric structured mesh is the right tool for a regular torus — gmsh is for arbitrary CAD.*

**Analysis (verify the mesh before solving).** `python examples/tire_contact/mesh.py`:
`2560 Hex8`, outer radius `1.450 = R+r_out` ✓, width `0.900 = 2r_out` ✓, lowest `z = −1.450 =
−(R+r_out)` ✓ (the ground plane), positive Jacobian corner-wise (`check_positive_jacobian`) ✓, plus a
`tread_surface_nodes` extractor (1280 outer nodes) for the contact set.

**Lesson.** Check exact dimensions *and* element validity (`min signed Jacobian > 0`) on **every**
generated mesh, before any solve — a silently-degenerate or mis-sized mesh costs days through a
contact solve.

---

## Phase 2b — contact on dynamics, implemented but ungated (`examples/tire_contact/run.py`)
**Model decision.** Tire as a `mixed-u-p Hex8` `ElementGroup`; rigid-ground **`HalfSpace`** just below
the tread; **`RigidBarrierContact`** (cubic barrier + CCD + ppf-smoothed friction) on the tread nodes;
gravity as a body force; all on **`solve_dynamics`** (dynamic relaxation — the barrier doesn't converge
quasistatically). Mirrors `examples/contact_vs_ppf/coupfe_box_on_floor.py`, swapping in the torus, the
mixed-u-p element, and the held rim.

**Historical modelling observation.** A development run first held the rim
**fully fixed** (x,y,z) — a "mounted wheel." Result: gravity made the tire *ovalize* and the bottom
moved **up** (gap grew 0.02→0.037), **no contact patch**. With the rim fixed everywhere the tire can't
translate down, so gravity just squashes the cross-section and *lifts* the tread. **Fix: hold the rim
in x,y only, free in z** — the axle then acts like a *suspension strut* (upright, no tipping/fore-aft/
roll, since fixing rim-x,y also kills the roll-about-y mode) while the whole tire **drops** onto the
floor. A rerun with retained outputs is required before reporting the resulting
sag, patch size, or penetration as evidence.

**Implementation.** Soft near-incompressible rubber (`G=1, K/G=100`), `grav=0.5`,
`dhat=0.04`, `kappa=2e3`, `mass=` on the tread nodes (the adaptive-capacity
barrier), `mu=0.4` smoothed friction, `damp=0.5`, `dt=0.02`, 60 steps. Element reused from the committed
`.for` (no regeneration).

**Evidence gap.** The driver reports minimum gap, barrier-band population, and
sag, but no reviewed full-solve gate or retained result currently establishes
their expected values. Add a bounded run with a convergence/residual gate,
hashed output, and environment record before treating those quantities as a
contact benchmark.

**Lesson.** A contact "setup" that seems physical can be subtly wrong (rim-fixed lifts the tread); the
**deformed shape** diagnoses it instantly (the bottom moved the wrong way), not a scalar residual — the
same lesson as the cardiac Case-B "apex pinned" mode (`docs/lessons_learned.md`). **Run it and look at
the displacement, not just the convergence flag.**

## Phase 2c — Von Mises post-processing, implemented but ungated (`vonmises.py`, `analyze.py`)
**Model/analysis decision.** The element kernel computes stress internally but returns only R/K, so
recover the field **independently in Python** from the converged `U` — same mixed-u-p law
(`P = G(F−F⁻ᵀ)+p F⁻ᵀ`, `p = K·avg(lnJ)`, `σ = J⁻¹PFᵀ`, `σ_vm = √(3/2 s:s)`). Independent recovery
doubles as a cross-check of the element physics.

**Implementation.** `element_vonmises(nodes,elems,U,G,K)` (per-element, Gauss-averaged) +
`write_vtk(...)` (a 25-line legacy-VTK of the **deformed** mesh + per-cell von Mises for ParaView — no
built-in writer in the core, and we keep it that way: I/O is per-problem glue, `docs/DESIGN.md`).

**Analysis boundary.** `analyze.py` can compute peak location and
contact-/rim-band stress summaries from a saved result. Historical values
suggested a contact-region concentration, but no exact GetFEM source revision,
parameters, reference data, or retained CoupFE output supports a comparison.
Treat the script as a post-processing capability, not a GetFEM validation.

**Lesson.** When the kernel hides the QOI (here, stress), reconstruct it **independently** — you get the
analysis output *and* a cross-check of the simulation's physics for free. And a few lines of VTK beat a
dependency for the visual comparison.

**Workflow status.** The element → mesh → contact-dynamics → von Mises path is
implemented, but only the mesh layer is in the reviewed public gate. A
qualified benchmark needs a bounded solve, retained output/environment, mesh
refinement, and an exact open-source reference record.

## Phase 3 — differentiable sensitivity  ◐ (`sensitivity.py`) — an HONEST negative result + the right path
**Goal.** `d(compliance C=f_ext·U)/dG` via an **adjoint** (one solve for all parameters — what
GetFEM/Kratos can't give), validated against a re-solve FD.

**What we tried.** The cheap route: a **static adjoint at the resting state** — `K=∂R_static/∂U`
(bulk+contact, no inertia), `Kᵀλ=f_ext`, `dC/dG=−λᵀ∂R/∂G` (`∂R/∂G` a cheap residual-FD on the bulk).
Key insight that *should* make it tractable: the forward needs dynamics (indefinite barrier tangent),
but the adjoint is a single linear solve, where an indefinite K is fine.

**The negative result (and it's the valuable part).** It does **not** validate: adjoint `−0.52` vs FD
`−0.05`, because **`|R_static(U*)| ≈ 0.2`, not 0** — the dynamics resting state is **not a static
equilibrium**, even after hard settling (`DAMP=2`, 110 steps; it didn't shrink). The static adjoint
*assumes* `R_static=0`, so it's invalid for a barrier-**dynamics** solve. *Run it and check the
assumption (`|R_static|`), don't trust the gradient.*

**The lesson — differentiability ⟂ forward-robustness here.** The smoothed-barrier-on-dynamics is the
*robust forward* path, but it has **no clean static equilibrium**, so there's no cheap static adjoint.
The two correct routes: (1) the **dynamic adjoint** (through the time-stepping, carrying inertia/damping)
— proper but substantial; or (2) a contact formulation with a **clean static KKT** — the
**dual-multiplier**, whose frozen-active-set adjoint *is* validated (`examples/friction_identifiability`,
grad-vs-FD ~1e-10, 2D small-strain). So **our validated differentiable contact is the dual-multiplier**;
the 3D smoothed-barrier tire's differentiability is the scoped extension (the dynamic adjoint). This
reinforces `docs/dev/dual_multiplier_strategy.md`: the dual-multiplier earns its keep precisely *because*
it solves to a clean KKT (differentiable), where the robust smoothed path does not.

---

## Honest caveats (carry into the writeup)
Linear Hex8 ≠ GetFEM's Q2/order-2 geometry. Until the exact upstream source,
parameters, outputs, and refinement evidence are retained, this is a
capability-oriented RESEARCH workflow—not a GetFEM reproduction, validation,
or forward-contact superiority claim (`docs/DESIGN.md`, Positioning).
