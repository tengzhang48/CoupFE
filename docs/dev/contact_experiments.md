# Contact / dynamics — key experiments log (what we tested, what it established)

A record of the **decisive experiments** behind the contact + dynamics design choices, so a future
agent can trust (or re-run) the conclusions instead of re-deriving them. Each entry: the *question*,
the *setup*, the *result*, and the *technique it verified*. Narrative "why" lives in
`docs/lessons_learned.md`; codified pitfalls in `skills/pitfalls.md`; this file is the evidence.

Newest first within each theme. Reproduce under `PYTHONPATH=$PWD OMP_NUM_THREADS=1 mamba run -n
coupfe python -m pytest <test>` (MPI ones via `mpirun`).

---

## Quasistatic barrier convergence + the ppf line-search question (2026-06-23)

**Q: Does ppf-contact-solver use an energy-merit (Armijo) line search? Do we need one?**
- **Setup.** Read ppf's Newton loop (`crates/ppf-cts-solver/src/cpp/main/main.cu` 820–973) + energy
  models; grep for any energy/residual decrease / reject / backtracking.
- **Result.** ppf does **no** merit line search at all. It uses: (1) **PSD-projected Hessian**
  (analytic eigensystems `eigsys/` + `solve_symm_eigen3x3`; `inflate.hpp` reassembles a "projected
  Hessian via rank-1 outer products, keep positive modes") → the Newton direction is guaranteed
  descent; (2) a **max-displacement step cap** `toi_recale = min(1, max_dx/max‖dx‖)` (trust-region,
  not Armijo); (3) a **CCD filter** `toi`; (4) **incremental potential (dynamics)** — inertia
  `M/dt²` makes the local energy strongly convex. Then it takes `x −= toi_recale·toi·dx`
  **unconditionally**. ppf's friction (`energy/model/friction.hpp`) is byte-for-byte our model
  (`P=I−nnᵀ`, `λ=μ·contact/max(min_dx,‖P·dx‖)`, grad `λ(P·dx)`, hess `λP`).
- **Verified.** There is no energy-merit line search to port. Robust contact = PSD Hessian + step
  cap + CCD + dynamics.

**Q: Will a PSD (Gauss-Newton) tangent alone make the quasistatic deformable barrier converge?**
- **Setup.** Serial two neo-Hookean blocks, top driven onto bottom through a node-to-segment
  `DeformableBarrierContact2D` (`κ=2e3`, `d̂=0.04`). Added an opt-in PSD tangent option (rank-1
  `2sg·gd⊗gd`, dropping the indefinite geometric `−sg²·∂gd/∂X` term). Compared three solvers.
- **Result.** All three **stall**: (a) PSD tangent + standard *residual-norm* line search →
  `α→0` (`max_nit=60`, `|R|≈6e-3`); (b) full ppf recipe **without dynamics** (PSD + CCD + `max_dx`
  cap, no backtrack) → `|R|≈1e-2` at 400 iters; (c) **force-driven** (feasible load, below the cubic
  capacity `s·d̂²`) → `|R|≈1.3` at 80 iters.
- **Verified.** The quasistatic deformable barrier converges only with **dynamics** OR a true
  **energy-Armijo (IPC filter)** line search — a *residual-norm* or *no-merit* line search will not.
  Two consequences: (1) the "barrier energy-merit line search" is not a thing ppf needs; (2) the
  robust path for the distributed barrier is **distributed dynamics** (reuse the already-converging
  serial `solve_dynamics`), not forcing quasistatic convergence. (Confound noted: driving a
  *displacement through the gap* with a finite cubic barrier is itself infeasible — see "finite
  barrier balances a force" below.)
- **Status.** PSD-tangent option reverted from the tree pending the dynamics build (it is what ppf's
  PSD Hessian corresponds to and will return there); the finding stands on its own.

**Q: Does distributed dynamics make the deformable barrier converge (path B)? Is it rank-independent?**
- **Setup.** `solve_dynamics_distributed` — implicit backward-Euler incremental potential brought to
  the distributed solver (node-local inertia `M/dt²` + Rayleigh damping + gravity, ghosted bulk, the
  **shared** cross-rank contact helper, a **CCD-bounded predictor**, and the global-CCD step bound).
  Two neo-Hookean blocks collide under gravity through a cross-rank cubic barrier;
  `examples/mpi_smoke/distributed_dynamics_barrier.py` at 1/2/4 ranks.
- **Result.** Newton **converges** every step (`|R|~7e-14`, ~4 iters); **penetration-free** (min gap
  stays in `(0, d̂)`); gathered U is **rank-independent to machine precision** (`max|u2−u1|=2.8e-16`,
  `max|u4−u1|=3.1e-16`). The consistent (indefinite) deformable-barrier tangent is fine here — the
  inertia `M/dt²` regularizes it (no PSD projection needed; the serial dynamics test already showed
  this). The serial `solve_dynamics` oracle differs by ~`6.8e-3` — *expected*, not a bug: the
  distributed driver is ppf-style (CCD bound, **no residual backtracking**) while serial
  `newton_solve` backtracks, so two valid backward-Euler trajectories drift over 30 steps; the
  machine-precision 1-vs-N is the rigorous gate.
- **Verified.** Dynamics is the substrate (ppf-confirmed): the penetration-free deformable barrier
  works **distributed** via dynamics, rank-independent, with no energy-merit line search. The
  cross-rank contact + global-CCD machinery is shared (`_DistDeformableContact`) between the
  quasistatic and dynamics drivers. `tests/test_mpi_distributed.py::
  test_distributed_dynamics_barrier_rank_independent`.

---

## Barrier conditioning, stiffness, and dynamics

**Q: Cubic barrier vs IPC log barrier?**
- **Result.** The log barrier's `1/d` stiffness is ill-conditioned (amplifies the MUMPS-style
  non-reproducibility) and NaNs at `d≤0`. The **cubic** `(κ/3)(d̂−d)³` has bounded stiffness
  `2κ(d̂−d)`, is C², polynomial → no NaN even on penetration (gives a finite recovery force).
- **Verified.** Use the cubic barrier; non-penetration is the CCD's job, not energy→∞.

**Q: How stiff must a fixed κ be (quasistatic)?**
- **Result.** `κ ~ bulk stiffness` converges in ~7 Newton iters; too-stiff → line-search stall,
  linear not quadratic. A *finite* barrier balances a **force**, not an over-prescribed
  **displacement**: prescribing a boundary motion that needs more reaction than the capacity
  `s·d̂²` is infeasible and stalls "at the wall" (no log-∞ to push back).
- **Verified.** Match κ to bulk stiffness; load barrier problems by force (or via dynamics), not by
  driving a boundary through the gap.

**Q: Does a naive frozen adaptive κ (`κ = c·k_bulk/d̂`) help? (negative result)**
- **Result.** No — it made convergence **worse**. It matched the local stiffness scale but gave the
  cubic barrier **no load capacity**: the max cubic force `κ·d̂²` fell below the applied load, so the
  gap collapsed. Reverted (the 5 barrier tests still pass without it). Reading ppf's source then
  clarified *why*: ppf's "dynamic stiffness" is a per-contact scalar `wᵀ(K_elast + M/g²)w` scaling
  the barrier (`barrier.cu::compute_stiffness`) — `K_elast` (local elasticity Hessian projected on
  the contact direction) supplies **conditioning**, and `M/g²` (inertial, →∞ as `g→0`, recomputed)
  supplies **capacity**. The `M/g²` is exactly what a frozen κ lacks — and it is *inertial*, so the
  capacity bound is a quasistatic artifact.
- **Verified.** Don't ship a stiffness heuristic that only matches the *scale*; capacity comes from
  the inertial `M/g²` term, which needs dynamics. Verify a stiffness change on a **hard** case
  before claiming it helps (a "matched" κ looked right and was worse).

**Q: What does the adaptive stiffness `s = κ + M/d²` actually buy (decisive test)?**
- **Setup.** Hard drop with a deliberately **too-soft** `κ=10`, under dynamics.
- **Result.** Fixed-κ crushes **through** the floor (gap `−1.83`); the **same κ + M/d²** rests at
  gap `+0.006`, converged. After the predictor-CCD fix (below), even too-soft fixed κ no longer
  *penetrates* — but without `M/d²` the solve pins at gap≈0 and thrashes **18× more** Newton iters
  (16674 vs 907).
- **Verified.** The inertial `M/d²` term is about **capacity + conditioning**, not non-penetration
  (CCD owns that); it needs dynamics (`M`). `tests/test_contact_dynamics.py`.

---

## CCD / continuous collision

**Q: Is bounding the Newton step enough for non-penetration?**
- **Result.** No. `solve_dynamics` overwrote `U` with the *unbounded* inertial predictor
  `û = u_prev + dt·v_prev` **before** Newton; once `dt·v > gap` it **teleported** a node through the
  barrier band. Fix: CCD-bound the predictor jump from the last accepted state (same `max_step`,
  no-op when separated).
- **Verified.** **Any** position update that bypasses the line search (predictor / warm-start / BC
  ramp) must pass through the CCD bound or it tunnels (IPC does this too). `tests/
  test_contact_deformable_barrier.py`.

---

## Broad-phase search

**Q: Is the spatial-hash broad phase an approximation?**
- **Result.** For residual/tangent with **band = d̂** it is **provably identical**, not approximate:
  an active pair has point-segment distance `< d̂` ⇒ it is in the band-`d̂` superset ⇒ same closest
  edge + force; a vertex with no in-band edge contributes nothing either way. For **CCD/`max_step`**
  the band must be `d̂ + 2·reach` (it must catch an edge the vertex could *cross this step*). The
  deformable barrier + friction suites pass **unchanged** after the O(N²)→O(N) swap.
- **Verified.** Wire the broad phase into residual/tangent freely; give CCD a per-step-reach margin.
  `tests/test_contact_search.py`.

---

## Distributed correctness + determinism

**Q: Is non-smooth (stick/slip) contact non-deterministic in parallel?**
- **Setup.** Real mixed stick/slip frictional contact, 1 vs N ranks, MUMPS vs superlu_dist.
- **Result.** With **MUMPS** the solution varied run-to-run (~1e-3): parallel pivoting is
  non-reproducible (~1e-12) and the **slip near-null mode** (a slipping node has ~0 tangential
  stiffness) amplifies it. With **superlu_dist** it is rank-independent **and** repeatable to
  **1.3e-16**. `mumps_1rank == superlu_1rank` to 1.5e-16 (rules out a wrong-symmetric-mode story).
- **Verified.** The non-determinism was a **solver** property (MUMPS), **not** physics. Default to
  the reproducible `superlu_dist`. Vary one component before declaring a numerical result
  fundamental. `tests/test_mpi_distributed.py::test_distributed_friction_rank_independent`.

**Q: How do we gate distributed correctness in general?**
- **The 1-vs-N invariant.** Gather the distributed solution to one rank; it must equal the serial
  solve to *solver precision*, independent of rank count. Used for the bulk solve, the cross-rank
  deformable-contact **assembly** (R exact, K·v ~1e-13), the **penalty full solve** (vs an
  *independent* serial oracle, ~1e-8), and the **global CCD** reduction (`Vec.min` == serial
  full-surface `max_step`, `|diff|=0`). `tests/test_mpi_distributed.py`.

---

## 3D distributed contact + the no-bulk degeneracy (2026-06-23)

**Q: Does the 2D cross-rank machinery generalize to 3D (vertex-face + edge-edge)?**
- **Setup.** `_DistDeformableContact3D` (subclass of the 2D helper; op-agnostic methods inherited):
  vertex-face partitioned by the secondary vertex, edge-edge by `owns_edge_pair` (first node of the
  pair's first edge, applied in `_contributions` AND `max_step`). Static cross-rank assembly check
  (`distributed_3d_primitives`) with an active vertex-face + edge-edge pair.
- **Result.** Rank-independent **to machine precision** (R, K·v, and the global CCD all `0.0` diff at
  1/2/4 ranks; CCD `=0.2 < 1` so it constrains). Both primitives' cross-rank routing + the global
  `Vec.min` CCD are correct.
- **Verified.** 3D distributed contact (assembly + global CCD) works; the same surface-replication +
  owned-pair COO + off-process `ADD_VALUES` pattern as 2D. `test_distributed_3d_primitives`.

**Q: Can we test the full 3D distributed dynamics solve with mass points (no bulk element)?**
- **Result. No — the no-bulk system is degenerate for the solve.** Free vertices dropped on a fixed
  floor gives a mostly-diagonal inertia matrix + tiny contact off-diagonals. At first barrier
  engagement, **both direct solvers fail** (superlu_dist *and* mumps → zero pivot / `KSP_DIVERGED_PC_FAILED`
  → `du=inf`), and PETSc **intermittently SEGVs** (~1-in-4) on the empty/degenerate objects (a
  size-0 bulk scatter; a guard fixed the `-n≥2` case but `-n1` still flaked). An *iterative* solver
  (jacobi+gmres) converges (gap matched the serial 0.0184, 1-vs-N `=0.0`), but the SEGV made it a
  flaky test.
- **Verified / lesson.** These are artifacts of a **missing volumetric element**, not the contact
  code (serial `solve_dynamics` of the same config works). So verify 3D distributed contact by the
  **static cross-rank assembly** above (robust, solver-free), and defer the full end-to-end 3D
  dynamics solve to when a 3D **bulk element** (Hex8/Tet4) is wired — then the system is
  well-conditioned (like 2D) and the direct solver works. Don't build a fragile no-bulk dynamics
  test to dodge the missing element.

## Friction (2D + 3D, ppf smoothed model)

**Q: Does the smoothed friction give true static stick?**
- **Result.** No — it is rate-form (the slip reference `x0` resets each step), so it admits a
  bounded sub-`ε` **creep**, not perfect stick. The correct assertion is `stick ≪ slip ≪
  no-friction`, not `stick == 0`. (Measure slip at the **contact** nodes, not the whole-block mean —
  the latter includes elastic shear and reads as false drift.)
- **Verified.** `tests/test_contact_barrier_friction.py`, `tests/test_contact_deformable_friction.py`.

**Q: Does 3D friction generalize the 2D model faithfully?**
- **Setup.** Sliding vertex over a face / edge over edge, within `d̂`.
- **Result.** `|f_s| = μ·λ_n` at full slip; stick(sub-`ε` linear)→slip(plateau); friction forces
  sum to zero over the stencil (**action-reaction**) and vanish under a rigid co-translation **by
  construction** (the signed weights satisfy `Σcᵢ=0`); tangent `λ BᵀP B` symmetric PSD; `μ=0`
  byte-identical; the operator is stateful (drag → friction → vanishes after `commit` advances
  `_x0`). `tests/test_contact3d.py` (6 friction tests).
- **Verified.** 3D friction = the 2D smoothed model on the tangent plane `P=I−n⊗n`, reusing the
  barrier's own closest-point weights — no separate Jacobian.

**Q: Does the smoothed friction work on the DISTRIBUTED deformable barrier (cross-rank)?**
- **Setup.** `solve_dynamics_distributed` with `deformable_contact={"kind":"barrier","mu":0.4}`; two
  blocks held by gravity, the top sheared sideways; `distributed_dynamics_friction.py` at 1/2/4 ranks.
- **Result.** It needed **zero new MPI code** — the shared `_DistDeformableContact` already threads
  `mu` into each rank's owned-secondary op and advances `_x0` in `commit`. The μ>0 interface slip is
  **0.38×** the frictionless slip (friction holds), converged + penetration-free, and the gathered
  μ>0 U is **rank-independent to machine precision** (`max|u2−u1|=5.4e-16`).
- **Verified.** Friction is path-dependent, but because the per-step U is rank-independent the `_x0`
  reference evolves identically on every rank → distribution doesn't change the answer (same argument
  as the distributed rigid friction). `tests/test_mpi_distributed.py::
  test_distributed_dynamics_friction_rank_independent`.

---

## 3D edge-edge / degenerate dedup — does ppf classify and dedup? (2026-06-23)

**Q: Should we add contact-pair classification/dedup (and an edge-edge mollifier) to fix the mild
double-count at exact vertex-vertex/vertex-edge coincidences?**
- **Setup.** Read ppf's contact narrow phase (`contact/contact.cu`, `distance.hpp`, `accd.hpp`); grep
  for any dedup / distance-type classification / mollifier.
- **Result.** **No.** ppf applies the barrier per candidate pair using the **unclassified** closest
  distance everywhere — **no** type-classification, **no** dedup pass, **no** edge-edge mollifier
  (the `mollif`/parallel-guard hits are all in the *bending*/dihedral energy, not contact). Its
  edge-edge candidate generation is the same as ours: `edge_index < index` ordering +
  `edge_has_shared_vert` exclusion. So at a vertex-vertex/vertex-edge coincidence **ppf has the same
  benign double-count we do**. The cubic barrier is *why* no mollifier is needed (the IPC mollifier
  fixed the **log**-barrier's gradient blowup at near-parallel edges; the cubic barrier + robust
  `_unclassified` closest-point + dynamics don't have that).
- **Verified.** Our 3D edge-edge is a faithful ppf port; the "dedup" is **not** worth building (it
  would diverge from ppf for a measure-zero gain; CCD keeps it penetration-free). Corrected the
  earlier docstring claim that "ppf dedups via classification" (it does not).

## End-to-end 3D distributed dynamics (two Hex8 blocks) — element + dimensionless design (2026-06-23)

**Q: Why didn't the two-block 3D dynamics converge, and what's the right way to set it up?**
- **Setup.** Two neo-Hookean Hex8 blocks, the top resting on the bottom under gravity, deformable
  barrier across the interface; `distributed_dynamics_3d_blocks`.
- **What went wrong, and the chain of fixes (all *physics*, not solver):**
  1. **Element.** The first kernel was a **`standard` (full-integration) Hex8** → volumetric LOCKING
     (artificially over-stiff). Switched to an **F-bar Hex8** (`generate_element(..., element='hex8',
     formulation='fbar_mechanics')`, FD-tangent-verified to 7e-10, vendored
     `neo_hookean_hex8_fbar.for`). F-bar is the correct choice — but it did **not** fix convergence
     alone.
  2. **Dimensionless gravity (the real cause).** The gravitational strain `εg = ρ g L / G` was **2**
     (`GRAV=2, L=ρ=G=1`) → the soft block was *crushed* (≈37% strain). **Even serial `solve_dynamics`
     with a line search FAILED at εg=2, for BOTH κ=2000 and κ=50** (`|R|=0.45`, 1000s of iters) — so it
     was never the solver or κ; a non-physical deformation just has no converged equilibrium. Setting
     `εg = GRAV ≲ 0.2–0.4` → converges in **2–3 Newton iters** (`|R|~1e-15`), `max|U|~0.05`.
  3. **Barrier/bulk stiffness `κ̃ = κ/G`.** `κ=2000` (κ̃=2000, your "200×") is over-stiff: it floats a
     gentle block out of the band and ill-conditions the linear system. Match it: `κ ~ K_bulk ~ 10²`.
  4. **Adaptive stiffness.** `s = κ + M/gap²` **over-repels** at a small starting gap (`M/gap²` blows
     up) → flings the block out. For a *gentle rest* use a **fixed κ** (ample capacity `κ·d̂²·n ≫`
     weight); reserve adaptive `s` for hard, high-speed impacts.
- **Dimensionless design (the takeaway).** With `L=ρ=G=1`: pick `GRAV = εg ≲ 0.2–0.4` (resting strain),
  `κ ~ K_bulk` (conditioning + a gentle load engages — CCD owns non-penetration, not κ), start near
  the equilibrium gap `g_eq` from `κ(d̂−g)²·n ≈ ρgL³`, fixed κ. The mass is `M_node = ρ·(tributary
  volume)`; total `ρL³`. The non-convergence was **physics (a non-physical load), not the element
  tangent or the solver** — check the dimensionless numbers before reaching for PSD-projection /
  line-search machinery.

## End-to-end 3D distributed dynamics + FRICTION (two Hex8 blocks sheared) (2026-06-24)

**Q: How do you make the frictional analog of the 3D collision capstone converge, and what sets the
friction strength vs. convergence tradeoff?** (`distributed_dynamics_3d_friction.py` — the last gap to
2D parity: bulk + barrier + *friction*, distributed, in 3D.)
- **Setup.** Two F-bar Hex8 blocks; modest gravity seats the top onto the bottom (sustained `λ_n`),
  the top face is dragged `+x` (ramped), `z,y` free; ppf smoothed friction on the vertex-face barrier.
  Vertex-face only (flat interface — the capstone's edge-edge perf/degeneracy lesson).
- **Chain of fixes (all measured, not guessed):**
  1. **A held-top + barrier overlap does NOT sustain a normal load.** A repulsive-only barrier with a
     held top just relaxes to `gap = d̂` where the force vanishes → `λ_n≈0`, no friction (`μ` and `μ=0`
     gave *identical* slip, ratio 1.00, `min_gap=0.046 > d̂`). Friction needs a **sustained downward
     load** → use gravity (as 2D does), `λ_n ≈ weight` at the contact.
  2. **εg again.** Gravity must stay modest (`εg = ρgL/G ≲ 0.4–0.5`) or the block has no converged
     equilibrium — *but* `μ·λ_n = μ·ρgV` must also beat the elastic shear drive `~G·γ·A` for friction
     to bite. `GRAV=0.5→0.4` threads both.
  3. **The friction convergence floor (the new lesson).** With `friction_eps=1e-4` (the default) and
     interface slip `ut ~ 1.5e-3 ≫ eps`, the solve **stalled** (60 iters, `|R|` floored ~1e-4). Cause:
     in the **slip plateau** (`ut ≫ eps`) the smoothed friction is `λ = μλ_n/ut` and the ppf
     Gauss-Newton tangent **drops `dλ`** — and there `dλ·(P·dx)` is *comparable* to the kept `λP` term,
     so Newton converges only linearly to a residual floor. In the **stick regime** (`ut < eps`),
     `λ = μλ_n/eps` is constant in `ut` → `dλ≈0` → the tangent is near-exact → tight convergence. Fix:
     set **`friction_eps` ABOVE the actual slip** (`2e-3 > 1.5e-3`) → `4` iters, `|R|=6e-9` (clean).
  4. **Friction strength has a convergence ceiling.** Larger eps softens friction (stick spring), so
     restore holding via `μ`: `μ=0.4→0.8` gives ratio 0.64 (held). `μ=1.0` makes the spring
     `μλ_n/eps` too stiff → the stall returns (60 iters) for *negligible* ratio gain (0.64→0.62 — the
     ratio has hit its **elastic floor**: even fully stuck, the secondary rides the *bottom block's*
     elastic shear). So `μ=0.8, eps=2e-3` is the sweet spot.
- **Result.** Converged (4 iters, `|R|=2.3e-9`), contact active (`min_gap=0.035 < d̂`), friction holds
  (μ=0.8 slip `1.53e-3` vs μ=0 `2.38e-3`, **ratio 0.64**), penetration-free. **Rank-independent to
  machine precision** at 1/2/4 ranks (`max|u2−u1|=5.4e-16`, `max|u4−u1|=6.3e-16` — identical
  convergence/slip every rank). `tests/test_mpi_distributed.py::
  test_distributed_dynamics_3d_friction_rank_independent`.
- **Takeaway.** For ppf/IPC **smoothed friction under a Newton solve, tune `friction_eps` to the slip
  scale** — not arbitrarily small. Too small → slip-plateau, the dropped `dλ` floors `|R|` (no tight
  convergence); too large → weak (over-creepy) friction. `eps ≳ ut` (near-stick) + `μ` for strength is
  the convergent operating point. (The 3D unit friction tests never saw this — they call the kernel
  directly, not a multi-iteration Newton solve where the tangent's *consistency* governs convergence.)

## Test methodology (oracles that have teeth)

**Q: Why did a "penetration" appear that the operator's own gap denied?**
- **Result.** The oracle compared a node's `y` to the **global-max** strip-top `y`; once the strip
  *bends* under load, a node resting on the dipped surface read as below the un-dipped corners — a
  **phantom** `−1.4` penetration. Fix: interpolate the **deformed** surface beneath each node.
- **Verified.** When both sides deform, the penetration oracle must be **surface-aware**; the
  operator's per-pair gap is the truth. A phantom failure usually indicts the oracle, not the code.
