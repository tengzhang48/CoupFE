# CoupFE — public API reference

The call surface. Everything is built on the operator contract (`docs/DESIGN.md`); this lists what
you actually import and call. Capability/maturity per item: `docs/capabilities.md`.

> Exact timings, crossover sizes, and speedups below are historical local
> measurements, not retained release benchmarks. Re-profile on the final
> revision and archive the environment/output before citation.

## Top-level (`import coupfe`)
- `Operator`, `Residual`, `Tangent`, `complex_step_tangent` — the operator contract.
- `ElementGroup`, `GroupState` — a batched element contribution. Fuses the residual/tangent
  kernel evaluation by default (`ElementGroup(..., fuse_rk=True)`; env `COUPFE_FUSE_RK=0` to
  disable): the compiled kernel returns R and K together (like the Abaqus `UEL`'s `RHS`+`AMATRX`),
  so caching the pair runs it once per Newton iterate instead of twice — bit-identical, ~1.5-1.8×.
  Turn OFF for schemes that mutate props/state between a residual and its paired tangent, or
  residual-only/matrix-free loops.
- `CompiledElement`, `build_element_kernel` — drive an f2py-compiled `.for` kernel (auto-detects the
  `native` / `abaqus_uel` backend; forces the f2py **meson** backend on all Python versions).
- `InertiaOperator` — lumped-mass backward-Euler inertia (the dynamics substrate).
- Drivers: `assemble_residual`, `assemble_tangent`, `newton_solve`, `solve_increments`,
  `solve_dynamics`, `solve_dynamics_adaptive`.
- Exact affine constraints: `ConstraintRelation`, `ConstraintTransform`,
  `compile_affine_constraints`.
- Mesh/geom: `KernelMeshView`, `Circle`, `Sphere`, `Plane`, `uniform_refine_quad`,
  `check_positive_jacobian`.
- Front door: `Model`, `Result`. Materials: `NeoHookean`.

## Drivers
| Call | Use |
|---|---|
| `newton_solve(operators, U, state, ndof, dirichlet, *, t, dt, constraints=None, …)` | one Newton solve (line search; consumes each operator's `max_step` CCD bound); optional exact affine relations are solved in reduced coordinates while operators receive full vectors |
| `solve_increments(operators, U0, ndof, dirichlet, *, n_steps, constraints=None, …)` | quasistatic load-stepped Newton; a static relation set ramps its affine offsets, while a callable owns a non-proportional relation schedule (for contact problems you often need to wrap fixed increments in adaptive stepping — see `examples/ring_compress/reproduce.py`) |
| `solve_dynamics(operators, U0, ndof, dirichlet, *, dt, n_steps, …)` | implicit backward-Euler; **CCD-bounds the inertial predictor** too. **Serial.** |
| `solve_dynamics_adaptive(operators, U0, ndof, dirichlet, *, t_end, dt_init, dt_min=None, dt_max=None, growth=1.2, cut=0.5, maxit=None, **newton_kw)` | adaptive backward-Euler: grows `dt` when Newton converges, cuts it when Newton exceeds `maxit` or produces a non-finite update. **Serial.** |
| `solve_distributed(ndof, my_gm, my_coords, dof_per_node, batch_fn, dirichlet_fn, n_steps, *, contact=None, deformable_contact=None, solver="superlu_dist", pc="gamg", ksp_type="gmres", …)` | **MPI** load-stepped Newton (petsc4py). Bulk = `batch_fn`; `contact=` wires **rigid penalty + Coulomb friction** node-local; `deformable_contact=` wires **cross-rank deformable** node-to-segment penalty/barrier (surface replication + off-process `ADD_VALUES` + global-CCD `Vec.min`). History-free, single-field, quasistatic. |
| `solve_dynamics_distributed(ndof, my_gm, my_coords, dof_per_node, batch_fn, dirichlet, mass, *, dt, n_steps, damping=0.0, force=None, deformable_contact=None, solver="superlu_dist", …)` | **MPI** implicit backward-Euler **dynamics** (petsc4py). Node-local inertia `M/dt²` + Rayleigh damping + `force` (gravity); ghosted bulk; cross-rank deformable barrier + **smoothed friction** (`deformable_contact={"kind":"barrier","mu":…, "mass":<per-node>, "freeze_pairing":True}` — `mass` enables the adaptive `s=κ+M/d²` capacity, `freeze_pairing` fixes the per-step closest-edge; both needed for fine-mesh convergence); **CCD-bounded predictor** + global-CCD step. The path for the penetration-free deformable barrier (which does not converge quasistatically). **2D** (node-to-segment) or **3D** (the spec carries `"vertices"/"faces"/"edges"` → vertex-face + edge-edge cross-rank, `_DistDeformableContact3D`). 1-vs-N to machine precision. |

## Generic affine constraints (`coupfe.constraints`)

`ConstraintRelation`, `ConstraintTransform`, and
`compile_affine_constraints(ndof, relations, *, dirichlet=None)` compile scalar
relations into `U = Pq + U0`. The transform exposes `lift`,
`project_increment`, `reduce_guess`, `restrict_residual`, `reduce_tangent`,
`reduce_linear_system`, and `constraint_error`. It rejects duplicate slaves,
cycles, invalid indices, and Dirichlet/MPC conflicts.

`newton_solve(..., constraints=relations)` and
`solve_increments(..., constraints=relations_or_schedule)` provide the
qualified serial quasistatic path. Assembly and state commit remain in full
space; reduced linear systems route through the same `linear_solve` policy as
the unconstrained driver, and contact operators receive the lifted full-space
increment through the existing time-aware `max_step` path. Fixed and adaptive
dynamics reject affine constraints explicitly because projected predictors,
velocity/history, and time-varying offsets have not been qualified.

The Core API is deliberately mesh-agnostic. Periodic boxes, mesh-node matching,
corner-equivalence graph construction, Gmsh/CAD semantics, and application
boundary policies belong in EDA, cardiac, or another consumer. Current MPI
drivers do not consume the transform.

### Dynamic relaxation — quasistatic states via `solve_dynamics`

`solve_dynamics` + `InertiaOperator(..., damping=alpha)` IS a correct quasistatic solver
when driven with the staged **ramp-hold-settle** protocol (`skills/preflight.md` has the
full pre-flight; `docs/lessons_learned.md` 2026-07-02 has the cautionary tale):

1. **Compute the structural timescale first** — `omega_1` from `eigsh(K, M)`; "slow"
   and "low damping" are meaningless without it.
2. **Ramp** the load/obstacle in stages (rate uncritical), **hold**, and relax under
   near-critical mass-proportional damping `alpha = 2*omega_1`.
3. **Sample only settled states**: `KE = 0.5*sum(M*v**2) < ~1e-8` against the
   strain-energy scale. At `v -> 0` the damping and inertia forces vanish identically,
   so the settled state solves `F_int + F_contact = 0` exactly and is dt-independent.

The release source contains a **RESEARCH** ring-compression workflow and
historical Abaqus comparison. Its proprietary deck is not redistributed, and
its solver-version, extraction, and retained-result provenance are incomplete.
Do not cite its recorded percent differences as release validation; see
`examples/REFERENCES.md`.

### Distributed rigid-penalty contact dict

`solve_distributed(..., contact=...)` accepts the node-local rigid `RigidContact`-style contact spec:

```python
contact = {
    "nodes": top_node_ids,
    "coords": nodes[top_node_ids],
    "obstacle": Sphere(...),      # or HalfSpace / Moving*
    "k": penalty,
    "mu": mu,
    "k_t": tangential_penalty,
    "comps": (0, 1, 2),
    # optional friction-state controls:
    "x_prev": step_start_contact_positions,
    "ft": step_start_tangential_forces,
    "return_state": True,
}
```

When `return_state=True`, `info["contact"]` contains global contact QoIs and the gathered
per-contact-node tangential state:

```python
{
    "P": ..., "Q_signed": ..., "Q": ..., "active_contact_nodes": ...,
    "ft": np.ndarray[(n_contact, cdim)],
}
```

This is for nodal penalty/return-map contact only. It is useful for two-phase Cattaneo-style
loading: run the normal phase frictionless, initialize `x_prev` from the accepted normal-state
positions and `ft=0`, then run the shear phase with `mu>0`.

## Linear solvers (`coupfe.assembly.factored`) — the ONE policy module

Doc contract (Teng, 2026-07-02): keep this SIMPLE, INFORMATIVE, and CORRECT — every claim
carries its measured bound, and that is enough for an agent to choose right. Full ladder +
traps: `skills/performance.md`, "The solver ladder".

| Call | Use |
|---|---|
| `linear_solve(K, b, *, prefer="auto")` | one-shot solve; scipy `spsolve` small systems, PETSc **MUMPS** above ~20k DOFs (SuperLU's 3D fill-in wall: 400 s → ~45 s at 96k). `newton_solve`/`solve_dynamics` route through it automatically |
| `factored_lu(K, *, prefer="auto")` | factor ONCE, solve many (vectors + dense RHS blocks; PETSc reuses the factorization per column); used by the condensed contact solvers. `prefer="scipy"` for 2D meshes at MEASURED sizes (≲130k DOFs; contact needs exact solves anyway), `"auto"` for 3D, `"petsc"` to force |
| `iterative_solve(K, b, *, ksp_type="gmres", pc_type=None, rtol=1e-8)` | **opt-in** Krylov+AMG for large WELL-CONDITIONED bulk. Default PC = **hypre, gamg fallback** (measured 2-4× gamg; beats direct on 2D SPD already at ~90k DOFs). NOT for contact-stiffened systems or validation work (those need exact solves); raises on non-convergence |
| `make_fieldsplit_solver(field_components, dof_per_node, *, split_type="additive", sub_pc="gamg")` | **coupled multifield** block preconditioning (ported from `abaqus_ufl.fe`, validated on u-c-phi; EDA runs the same on phi-T): FieldSplit by field, AMG per block. `field_components=[("u",[0,1]),("c",[2])]` node-major layout. Returns a pluggable `linear_solve(K,b)`; scale each equation to O(1) first |

Env: `COUPFE_LINEAR_SOLVER=scipy` forces scipy everywhere; `COUPFE_FACTORED_PETSC_MIN_N` moves the
threshold. PETSc failures fall back to scipy with a `RuntimeWarning` (a zero pivot at contact
engagement is usually a MODEL problem — degenerate/no-bulk config, see the module docstring). MPI is
separate (`solve_distributed(..., solver=, pc=)`; **parallel MUMPS is non-reproducible** → the
distributed default is `superlu_dist`, `skills/distributed.md`). Do not hand-roll PETSc KSP setups
in examples — extend this module instead (`skills/performance.md`, "Linear solvers").

## Contact operators (`coupfe.operators.contact`)
All implement `(residual, tangent, commit)`; `max_step(U, dU)` (optionally `max_step(U, dU, t, dt=None)` for time-aware CCD with moving obstacles) where penetration-free.

| Operator | What | Friction | Notes |
|---|---|---|---|
| `RigidContact(nodes_ref, contact_nodes, obstacle, *, dof_per_node, comps, k, mu, k_t)` | rigid penalty | return-map Coulomb (`mu>0`) | the only contact wired into `solve_distributed` (node-local); quasistatic compression problems need adaptive load stepping to avoid collapse, see `examples/ring_compress/reproduce.py` |
| `RigidBarrierContact(…, dhat, kappa, eta, mass, mu, friction_eps, ppf_norm=False, elastic_op=None, ndof=None)` | rigid cubic barrier + CCD | **ppf smoothed** (`mu>0`) | `mass` → adaptive `s=κ+M/d²` (dynamics); `ppf_norm=True` uses the geometry-normalized cubic shape `2/dhat`; `elastic_op` injects `nᵀK_elastn` into `s`; `mu=0` byte-identical |
| `DeformableContact2D(nodes_ref, secondary_nodes, primary_edges, *, dof_per_node, comps, k)` | node-to-segment penalty | — | |
| `DeformableBarrierContact2D(…, dhat, kappa, eta, mass, mu, friction_eps, friction_kt, friction_persistent, body_id, freeze_pairing)` | node-to-segment barrier + CCD | **smoothed** \| **return-map** (`friction_kt`) \| **persistent** (`friction_persistent`) | both bodies deform; broad-phase accelerated; `mu=0` byte-identical. **N-body use:** orient every body's edge loop consistently (outside-on-left, see skills/contact.md) + pass `body_id=` (per-node body id). **`mass=`** → adaptive `s=κ+M/d²` (capacity; cures the CCD-lock stall). **`freeze_pairing=True`** (opt-in) fixes each node's closest-edge per step (re-search at `commit`) → cures fine-mesh active-set chatter; the standard node-to-segment approach; default off = byte-identical |
| `DeformableBarrierContact3D(nodes_ref, vertices, faces, edges, *, …, mu, friction_eps, self_contact, friction_kt, friction_persistent)` | vertex-face + edge-edge barrier + ACCD (numba) | **smoothed** (numba) \| **return-map**/**persistent** (vertex-face, numpy) | `self_contact=True` → incident-exclusion (numba); distributed via dynamics |
| `SurfaceContact2D(nodes_ref, vertices, edges, *, dof_per_node, comps, k)` | vertex vs non-adjacent edge | — | multi-body + self-contact (penalty) |

Obstacles: `HalfSpace(point, normal, *, kinematic=False, thickness=None, constraint_tol=0.01)`, `Sphere(center, R, inside=False, *, …)`, `MovingHalfSpace(position, normal, *, …)`, `MovingSphere(center, R, inside=False, *, …)`. The `Moving*` variants accept a callable `position(t)` / `center(t)` so the CCD bound can sample the obstacle trajectory at intermediate times. `kinematic=True` floors the barrier gap at `constraint_tol*dhat` (ppf-style, prevents sticking); `thickness` makes the obstacle a pass-through shell.

### Friction modes (deformable barrier operators)
- **smoothed** (`friction_kt=None`, default) — ppf/IPC rate-form, distributed, the production path.
- **return-map** (`friction_kt=k_t`) — exact Coulomb cone, near-exact stick (micro-slip `~F/k_t`), 2D + 3D
  vertex-face (numpy).
- **persistent** (`+ friction_persistent=True`) — carries the committed tangential force per secondary/vertex
  (the εₚ analog) across steps, re-framed onto the current tangent plane at re-pairing ⇒ finite-sliding
  exact-stick. Strictly opt-in; default behaviour unchanged.

### Exact-stick / dual-multiplier (research-grade, `docs/dev/dual_multiplier_strategy.md`)
`coupfe.operators.contact_semismooth.SemismoothFrictionSolver(K, *, fixed_dofs, contact_tan_dofs,
  normal_dofs, mu, r=None)` + `.solve(f_ext, fixed_vals, p0=None)` — **Alart-Curnier semismooth Newton**:
  exact-stick friction (per-node partial slip, Schur-condensed; factor-once, amortized across a load path).
  Small-strain/small-sliding. The **relay** (frozen-active-set adjoint → differentiable friction-field ID) is
  demonstrated in `examples/friction_identifiability`; the distributed bulk-solve in
  `examples/mpi_smoke/distributed_dual_multiplier.py`.

**Friction params** (barrier operators): `mu` = Coulomb coefficient (0 ⇒ frictionless, stateless,
byte-identical); `friction_eps` = smoothed-friction slip tolerance `ε`. It's rate-form (bounded
sub-creep, not exact static stick). Theory: `docs/theory/contact_dynamics.md` §3b. **Tune `ε` to the
interface SLIP scale, NOT as small as possible** — smaller is *nearer* exact Coulomb but the ppf
Gauss-Newton tangent drops `dλ`, so in the slip plateau (`slip ≫ ε`) Newton only converges linearly
and floors at a moderate `|R|`. Pick `ε ≳ expected slip` (near-stick, near-exact tangent → tight
convergence) and set holding strength with `μ` (which has its own ceiling — too-stiff `μλ_n/ε`
re-stalls). See **Convergence notes** below + `docs/dev/contact_experiments.md` (2026-06-24).

### Convergence notes (contact / dynamics) — read before debugging a stall
- **Use `solve_dynamics*` for the barrier, not the quasistatic drivers.** The penetration-free
  deformable barrier does **not** converge quasistatically (a residual-norm line search stalls at the
  active-set/projection flip); inertia `M/dt²` (dynamics) or a true energy-Armijo line search is
  required. `solve_increments`/`solve_distributed` are for bulk + *penalty* contact, not the barrier.
- **Smoothed friction reproduces `f = μN` at all slip** (large *steady* slip is its *easy* regime —
  the tangent `~μλ_n/ut → 0`, friction → a near-constant force). Its narrow costs are **not** large slip:
  a convergence-sensitivity band at **moderate slip-per-step** (`ut` a few × `friction_eps`, tunable via
  `friction_eps`/sub-step/damping), a **regularized stick** (creep `~ε`, not exact lock), and an `O(dt)`
  direction lag for *turning* slip. Prefer the return-map friction (`RigidContact`, exact stick,
  non-symmetric) when you need **exact static stick / sharp stick-slip transitions** — not for large slip.
- **Solver is direct (`superlu_dist`).** The barrier tangent is *indefinite* (a consistent energy
  Hessian) and the rigid friction tangent *non-symmetric*; under dynamics `M/dt²` regularizes the
  barrier so direct LU works. No iterative preconditioner yet → large 3D contact is direct-bound.
  Use `superlu_dist`, **not MUMPS** (MUMPS is run-to-run non-reproducible on the near-null slip modes).
- **The active set is re-detected every Newton iteration** (brute-force closest feature) — robust to
  big predictor steps but can *chatter* when a feature switches (face↔edge↔vertex) mid-slide; if a
  solve oscillates without descending, suspect feature switching, not the linear solver.
- **`κ`, `ppf_norm`, `elastic_op`, and adaptive `s`.** `ppf_norm=True` matches the
  ppf-contact-solver geometry-normalized cubic (`force ∝ (2/dhat)·s·(dhat−d)²`), so `κ`
  is a true stiffness (force/length). `elastic_op` estimates `nᵀK_elastn` from the bulk
  tangent and adds it to `s`, completing the ppf `wᵀ(K+M/g²)w` recipe. Match `κ ~ K_bulk`
  (a 100×+ ratio ill-conditions); the adaptive `s=κ+M/gap²` *over-repels* at a small gap
  (use a fixed `κ` for a gentle rest, adaptive only for hard impact). A non-physical *load*
  (`εg=ρgL/G ≳ 0.5`) has no converged equilibrium — check it first.

## Contact broad-phase (`coupfe.operators.contact_search`)
- `candidate_pairs(positions, vertices, edges, dhat, *, exclude_incident=True, cell=None) -> {vertex_id: edge_indices}`
  — uniform spatial-hash; a **superset** of the true within-`dhat` set (no contact missed). O(N) for
  roughly-uniform features. Used inside `DeformableBarrierContact2D`; reusable standalone.

## Performance — which paths are accelerated, and why (match the tool to the structure)

The contact paths are accelerated with **different tools, by computational structure** — not numba
everywhere. The rule: **numba pays off only where numpy is forced into a Python per-element loop;
where numpy *vectorizes*, it is already near-optimal and numba adds nothing.**

| Path | Structure | Tool | Why |
|---|---|---|---|
| Return-map (rigid penalty) friction | node-local vs **one** obstacle; same ops for every node; non-smoothness = global **masks** | **vectorized numpy** | a few dense array ops, no per-node loop → numpy is near-optimal; numba would remove nothing |
| 2D smoothed / barrier narrow-phase | per-pair but expressible as masked array ops over active pairs | **vectorized numpy** | already loop-free |
| **3D** barrier narrow-phase (vertex-face + edge-edge) + ACCD | **per-pair, data-dependent branching** (closest-feature classification face/edge/vertex, degenerate fallbacks, edge-edge alternating projection) — each pair takes a different branch | **numba** `@njit` | numpy can't branch per element → the natural code is a Python per-pair loop (~80 µs/pair, ~99% overhead) → numba compiles loop+branches → **284×, bit-exact** |
| Broad-phase LBVH | tree build + stack query, branchy | **numba** | irregular control flow |
| Element kernels (`.for`) | hot dense array math + **must also run inside Abaqus** | **Fortran (f2py)** | dual-home; the in-Abaqus requirement is unique to *element* kernels (not contact) |

Discipline: **profile first; don't accelerate a non-bottleneck**; keep the **numpy version as the
bit-for-bit oracle** so the accelerated path is gated (numba ⇄ numpy must match to machine precision).
Rationale + the language decision: `docs/dev/contact.md`; capability/verification matrix:
`docs/capabilities.md`.

## Codegen (`coupfe.codegen`, build-time only — needs the `codegen` extra / sympy)
- `Material`, `WeakForm`, `VectorField`, `ScalarField`, `tensor` — define a form.
- `generators.uel_gen.generate_element(form, path, *, element, formulation, backend)` — emit a
  Fortran kernel; `backend='native'` (`coupfe_element_rk`) or `'abaqus_uel'` (`UEL`).
  `generate_uel(...)` = the Abaqus-UEL convenience wrapper.
- **`formulation=`** — `'standard'` (full integration) or **`'fbar_mechanics'`** (mean-dilatation /
  F-bar; **both backends**). This provides an anti-locking implementation seam
  for isochoric/near-incompressible studies. The private `j2_fefp_uel` bending
  study is withheld for source/port provenance, so its historical stiffness
  comparison is not first-release validation.
- **Spectral tensor ops** — `tensor.logm`/`expm`/`polar`/`eig` (→ `logm33z`/`expm33z`/`polar33z`/`eig33z`)
  for finite-strain plasticity (Hencky `E=½logm(C)`, exponential `Fp` update).
  The private finite-strain J2 example supplies internal implementation checks
  but is excluded from the public artifact pending provenance.
- `Material.state_schema` — declared named state (offsets generated for both backends; init;
  `field_history` vs stored `state_vars`). **Tensor state is stored column-major** — pack/unpack with
  `order='F'`; a non-symmetric tensor state (e.g. `Fp`) must round-trip through `reference_assembly`.
- `core.reference_assembly.assemble_element(...)` — the independent Python oracle for verification.

## Withheld diagnostic record (`cattaneo_3d`)

The private release-preparation tree contains a 3D rigid-sphere
Cattaneo-Mindlin/Abaqus diagnostic that exercised serial, PETSc/MUMPS, and
distributed node-local penalty paths. It is not part of the public artifact:
its reviewed end-to-end gate is a strict expected failure and its retained
solve says `converged=false`. Previous comparison values therefore are not
release evidence and must not be cited as validation.

The shipped Hertz, exact-stick, semismooth, finite-sliding, and 3D block
examples exercise the public contact APIs with passing gates. See
[`examples/REFERENCES.md`](../examples/REFERENCES.md) for the exact
READY/RESEARCH/WITHHELD boundary.

## Install extras
`pip install -e ".[runtime]"` (f2py compile: meson+ninja, + system gfortran) · `".[codegen]"`
(sympy, build-time) · `".[dev]"` (pytest). CI installs `".[dev,runtime,codegen]"`.
