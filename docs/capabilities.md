# CoupFE — capability status (honest)

A frank matrix of what works, what's serial-only, and what isn't built. The point is to stop
*implicit* assumptions of coverage (serial richness ≠ distributed coverage). Keep it current.

Legend: ✓ done + tested · ◐ partial/tracked · ✗ not built.

This is a development capability inventory, not an example-release allowlist.
The first public example boundary is authoritative in
[`examples/REFERENCES.md`](../examples/REFERENCES.md): `RESEARCH` examples
remain visible as unsupported capability demonstrations, while a capability
exercised only by a `WITHHELD` example is not part of the first public source.

> **Distributed evidence boundary (2026-07-30):** distributed checkmarks in
> this development inventory mean implementations and gates exist. Exact
> 1-vs-N/rank-invariance figures are historical observations; no retained
> final-revision MPI run is part of the current release audit. Rows below call
> out this pending qualification where it affects a public claim.

## Solve drivers
| Capability | Serial | Distributed (MPI) |
|---|---|---|
| Finite-strain bulk, hyperelastic, single-field | ✓ (vs Abaqus/feacheap) | ◐ implementation and historical 1-vs-N gates (`superlu_dist`; `gamg`/`gmres` iterative); retained final-revision rerun pending |
| Load-stepped Newton + line search | ✓ | ✓ |
| Implicit dynamics (backward-Euler) | ✓ core dynamics path gated; ring compression ships as a **RESEARCH workflow**, but its external comparison remains unqualified until authorized inputs and retained provenance are complete (`examples/REFERENCES.md`) | ◐ the deformable-contact stack runs on `solve_dynamics_distributed`; historical rank-invariance gates exist, final retained rerun pending |
| Path-dependent state / history (plasticity, Phase-9 schema) | ✓ (`CompiledElement.svars`) | ✗ distributed is history-free (state-commit tracked) |
| Coupled multi-field (u-T, u-φ, …) | ✓ | ◐ FieldSplit tracked, single-field today |
| Generic affine MPC reduction | ✓ mesh-agnostic compiler plus serial quasistatic `newton_solve` / `solve_increments` wiring with exact `U=Pq+U0`, full-space operator commit and `max_step`, established linear-solver routing, algebra/KKT/thermal controls, and explicit rejection in fixed/adaptive dynamics. Applications own node matching and relation construction. | ✗ current MPI drivers do not consume the transform; the legacy bulk-only replicated reference remains research history, not a shipped distributed capability |

## Contact
| Capability | Serial | Distributed |
|---|---|---|
| Rigid penalty + Coulomb friction (**return-map**, exact stick) | ✓ (2D/3D, vectorized numpy) | ✓ node-local; rank-independence gates exist, final retained MPI rerun pending |
| Rigid cubic barrier + CCD; adaptive `s = κ + M/d²` (dynamics) | ✓ | ✗ (distributed rigid contact = the penalty/return-map path above) |
| Deformable–deformable **barrier** (2D node-segment **+ all-primitive opt-in**; **3D vertex-face + edge-edge**) | ✓ + **2D ppf-style all-primitive** (`all_primitive=True`: point-edge over *all* pairs via `point_edge_coeff_unclassified` + `point_edge_toi` ACCD — no closest-edge to flip → no chatter; smoothed friction) | ◐ runs via dynamics with historical rank-invariance gates; pass `mass=` (adaptive `s=κ+M/d²` — fixes the CCD-lock) + `freeze_pairing` (fixes fine-mesh active-set chatter); final retained MPI rerun pending |
| **Smoothed friction** on the barrier | ✓ (rigid + deformable, 2D & 3D) | ✓ deformable, 2D & 3D with rank-independence gates (rigid-barrier-distributed not separately verified; final retained MPI rerun pending) |
| **Exact-stick friction (return-map)** | ✓ `friction_kt=` on deformable 2D and the 3D vertex-face NumPy path (exact cone, near-exact stick; gated) | ✗ not wired through `_DistDeformableContact`; distributed deformable friction is smoothed-only |
| **Exact-stick friction (dual multiplier, semismooth Newton)** | ✓ `coupfe.operators.contact_semismooth`: **per-node partial slip** (stick zone + slip zone coexist), exact `v_t=0` stick (machine precision, no `k_t`), Alart-Curnier semismooth + **Schur-condensed interface**; gated (`examples/semismooth_friction`). Linear bulk `K`, lagged normal | ◐ distributed bulk-solve example with a replicated dense interface; useful only for small `nc`, not the scale path (use smoothed / PERMON-FETI). Final retained MPI rerun pending. |
| **3D contact** (vertex-face + edge-edge + ACCD) | ✓ **numba, bit-exact vs numpy oracle** | ◐ cross-rank assembly + global CCD implemented; final retained MPI rerun pending |
| Implicit dynamics substrate (`solve_dynamics[_distributed]`) | ✓ | ✓ |
| Self-contact (penalty, `SurfaceContact2D`, frictionless) | ✓ | ✗ |
| **Self-contact (3D barrier + smoothed friction)** | ◐ `DeformableBarrierContact3D(self_contact=True)`: focused incident-exclusion, cross-layer, and active-operator gates pass, but the full hairpin subprocess exceeds its 600-second gate and its example is excluded from the first release | ✗ |
| **Broad-phase** (LBVH numba, ppf template; AABB-distance prune) | ✓ O(N), **all operators incl. `max_step`** (band = d̂+2·reach) | ✓ surface replication |
| **numba narrow-phase** (vertex-face + edge-edge + ACCD, bit-exact against NumPy) | ✓; historical local speedup is not release benchmark evidence | deterministic implementation; final retained MPI rerun pending |

## Codegen / elements
| Capability | Status |
|---|---|
| Form → Fortran generator (UEL), dual-home (UEL + native) | ✓ gated (byte-identical + cross-backend sign) |
| Material → Abaqus UMAT generator | ✓ finite-strain and small-strain paths are gated by four public material-point examples: stateless Neo-Hookean, spectral Ogden, stateful radial-return J2, and tensor-history standard-linear-solid viscoelasticity. Their independent oracles, deterministic regeneration, and compiler checks validate the stated material/codegen scope; no Abaqus structural run or standalone CoupFE UMAT host is implied. |
| Native element ABI (`coupfe_element_rk`) | ✓ (Quad4 2D; coupled scalar-diffusion) |
| State schema (named offsets, init, field_history vs stored) | ✓ Phase 9 (J2 plasticity proof) |
| Finite-strain J2 (Fe-Fp, Hencky + log-strain return, `Fp` state) | ◐ implemented in the private development tree with internal consistency checks; the `j2_fefp_uel` example is withheld from the first public artifact pending public formulation/port authority, so it is not release validation |
| Volumetric anti-locking — **F-bar** (`formulation='fbar_mechanics'`, centroid `J̄`) | ◐ implemented for both codegen backends. A private bending study suggested the expected anti-locking behavior but is withheld for provenance, and the current `(J̄/J)^(1/d)` path is unguarded when centroid `J̄≤0`; no first-release validation claim is made |
| Volumetric anti-locking — **mixed u-p** (`formulation='local_pressure'`, element-local condensed `p`) | ◐ `RESEARCH` implementation (native Quad4 + Hex8): `p=K·avg(lnJ)` volume-average/mean-dilatation. The examples ship to show the codegen path, not as a qualified formulation or release default; no general inf-sup or inversion-robustness claim is made. **Multiphysics still needs `J_e=J/J_inel`** (tracked gap). |
| Stabilized mixed **u-theta Tet4** | ◐ `RESEARCH` paper-form declaration with analytic `dS/dtheta` and homogeneous-state oracles, tangent verification, Tet4 generation, and compiler gate (`examples/stabilized_tet4`). The retained Abaqus block result is historical context, not a current Core solve. |
| Mixed **u-p-mu gel Quad8** | ◐ `RESEARCH` paper-form declaration with mixed-order field layout, reference assembly, native/UEL parity, and compiled execution (`examples/gel_chester_anand`). The externally seeded bilayer mesh and full contact solve are not Core assets. |
| Coupled **u-phi-c corrosion Quad8R** | ◐ `RESEARCH` paper-form declaration with J2/history state, dropped/full tangent variants, reference assembly, backend/state gates, and compile checks (`examples/phasefield_corrosion_cui`). The Cui-derived comparison mesh and Figure 3 run are not redistributed or claimed as current validation. |
| Pressure-gel **u-mu Hex8 + local p** | ◐ `RESEARCH` paper-form declaration with condensed-pressure verification and compiler gate (`examples/morphing_hex8`). The historical morphing result includes a mechanically active C3D8 companion mesh and exterior contact, so it is not a pure-UEL/CoupFE oracle. |
| **Anisotropic finite-strain** (Holzapfel–Ogden myocardium: fiber/sheet structural-tensor state, `I8fs` coupling, active stress, viscous) | ◐ implemented in `CoupFE-Cardiac` (separate repo): local material/operator gates exist, but the historical Case A backward-Euler comparison was not retained and does not validate the current Newmark default; Case B quantitative agreement and diagnosis remain open |
| Element zoo (~38 forms) | ◐ Phase 10, batched, in progress |
| Geometry — **compiled-element / UEL path** (f2py `.for`) | ✓ **Quad4 + Hex8** — 3D contact (`contact_3d_*`) and `hertz_contact` run F-bar / neo-Hookean **Hex8**; **local-pressure mixed u-p: Quad4 + Hex8**; Hex8/Hex20 shape+gauss templates exist (`shape_hex8`, `gauss_hex8`). |
| Geometry on the **native ABI** (`coupfe_element_rk`, standalone) | ◐ **Quad4-only**; Quad8/Hex8/Hex20 native = Batch 1 of Phase 10 (the compiled-element path above already covers Hex8) |

## Mesh and interoperability
| Capability | Status |
|---|---|
| Neutral in-memory bridge | ✓ `KernelMeshView` carries zero-based NumPy coordinates/connectivity, named node/element sets, and geometry classification; `Model.from_view` and `ElementGroup.from_view` consume it |
| Native mesh generation and refinement | ◐ structured rectangular Quad4 generation and uniform Quad4 refinement are core APIs; additional Hex8/annulus/disk generators are example-specific |
| Abaqus `.inp` | ◐ the tested UEL scaffold transforms flat CAE decks and preserves NSET/ELSET content, while the morphing example has a read-only, example-local extractor for labeled U3 and companion C3D8 blocks in a user-supplied pasta deck. Neither is a general runtime importer, and contact/material/step translation remains application-owned |
| Legacy VTK output | ◐ one example-specific ASCII Hex8 writer; no general core writer |
| Gmsh, `meshio`, VTU, XDMF, Exodus, CGNS | ✗ no core connector or dependency by design; application repositories own format/tool adapters |
| PETSc DMPlex / DMForest | ✗ no core adapter; current PETSc assembly uses raw global indices, and any future application integration must terminate at `KernelMeshView` |
| CAD, NURBS, STL geometry | ◐ `GeometryBackend` is the extension seam, but only analytic `Circle`, `Sphere`, and `Plane` implementations ship |
| Partitioning and tags | ◐ deterministic owned/ghost partitioning is tested for regular meshes; node/element sets exist, but facet sets, source-ID maps, mixed cell blocks, and graph partitioners do not |

**Ownership decision (2026-07-29): mesh-software adapters belong in the
applications, not CoupFE core.** EDA owns its Gmsh/CAD, physical-region,
periodic-face, and package-geometry translation; cardiac owns its ventricular
mesh, boundary, fiber/sheet, and material-region translation. Each adapter
produces the small `KernelMeshView` contract consumed by core. This keeps
Gmsh/`meshio`/CAD dependencies and domain-specific label semantics out of the
solver. If repeated code later justifies sharing, use a separate optional
interoperability package rather than expanding the core dependency surface.
CAD/discrete-geometry acquisition and adaptation likewise remain
application-owned because a volume-mesh file does not provide an authoritative
geometry model; core only calls the neutral `GeometryBackend` projection
contract.

Current adapter constraints matter: `KernelMeshView` holds one homogeneous,
fixed-width cell block and has no facet topology; `LocalMesh` does not preserve
sets or geometry metadata during partitioning; and
`check_positive_jacobian` currently skips non-2D meshes, so it is not a Hex8
quality gate.

## Honest limitations (read before claiming scale)
*(Updated for the release audit on 2026-07-30: distributed
deformable-contact implementations and development gates cover bulk, barrier,
and friction in 2D and 3D, but final-revision retained MPI qualification is
pending. The 3D narrow-phase remains locally gated as numba bit-exact.)*
- **Distributed deformable friction is smoothed-only** (ppf, rate-form: bounded sub-creep, not exact
  static stick). Serial deformable contact also has an opt-in **return-map** (`friction_kt=`) with the
  exact Coulomb cone and near-exact elastic stick in 2D and in the 3D vertex-face NumPy path; persistent
  mode carries the committed tangential force across steps. Distributed exact-stick is currently limited
  to the node-local rigid return-map path. The **dual-multiplier (Alart-Curnier / active-set) exact-stick** form —
  stick as a *constraint* `v_t=0` (Lagrange-multiplier friction force),
  no `k_t`/`k̃_t` conditioning knob — is **demonstrated for a deformable elastic block and gated**
  (`examples/exact_stick_friction`, **machine-zero stick**, Coulomb cap, analytic incipient onset). The
  **per-node *partial*-slip** form (a stick zone + a slip zone coexisting) is now built — the **Alart-
  Curnier semismooth Newton** in `coupfe.operators.contact_semismooth` with a **Schur-condensed
  interface** (`examples/semismooth_friction`). It resolves the active set consistently and does not
  cascade at the bonded stress concentration the naive switch-and-resolve method does. It runs on a
  **linear bulk `K`** with a **lagged/clamped normal** (no coupled Signorini / nonlinear outer Newton)
  and remains a standalone research solver, not the production dynamics/contact path.
- **Rate-and-state friction** (rate-dependent μ, stick-slip instability, large-slip physics) — **not
  built**; the future model when the physics is genuinely rate-dependent.
- **Self-contact**: 2D penalty (`SurfaceContact2D`, frictionless) plus a
  partially qualified 3D barrier/smoothed-friction operator
  (`DeformableBarrierContact3D(self_contact=True)`). Focused incident-exclusion
  and active-operator checks pass, but the bounded end-to-end example does not;
  that example is excluded from the public artifact. Gaps: serial only, 1-ring
  incident exclusion, no deeper 2-ring filter, and no exact-stick mode on
  self-pairs.
- **CPU only.** Acceleration is **numba** (`@njit`); GPU (`numba.cuda` / ppf's Karras LBVH) is deferred
  until scale demands it. The node layout/query are ppf-compatible so the GPU port is a direct escalation.
- **No retained large-scale benchmark or release-grade scaling record exists.**
  The plan and trigger conditions are in
  `docs/dev/contact_vs_abaqus_benchmark.md`. Do not quote historical speedups
  until a converged run, raw output, hardware/topology, MPI/PETSc versions, and
  locked environment are retained; a non-converged fixed-iteration run can
  produce a fake speedup.
- **2D smoothed friction is vectorized numpy, not numba** (already fast — it vectorizes; cosmetic parity).
- **Validation is case-specific** (feacheap/Abaqus + own oracles on particular cases), not a broad
  analytical-benchmark sweep across the regime space.
- Young, single-author + AI-assisted; no external users / published results yet.

## Roadmap priorities — ordered
**Done (2026-06):** broad-phase **LBVH (numba)**; **distributed dynamics + barrier + friction** (2D & 3D,
rank-independent); **cross-rank deformable contact** (assembly + global CCD); **3D contact** (vertex-face
+ edge-edge + ACCD) numba bit-exact. The original priorities 1–3 (broad-phase, distributed
dynamics+barrier/friction, cross-rank deformable contact) are complete. **Also (2026-06-26):**
distributed-barrier robustness — **CCD-lock fix** (per-node mass → adaptive `s=κ+M/d²`) and **pairing-freeze**
(fine-mesh active-set chatter); **2D all-primitive barrier port** (point-edge + smoothed friction + 2D ACCD).
The same development round added mixed u-p as a RESEARCH option. It is not a
release default, and historical scaling numbers are omitted pending retained
performance evidence.

**Next:**
1. **GPU** (`numba.cuda` / port ppf's Karras LBVH) — *when scale demands*; the per-pair kernels are
   embarrassingly parallel and the node layout/query are ppf-compatible for a direct escalation.
2. **Benchmark vs Abaqus** — `docs/dev/contact_vs_abaqus_benchmark.md` (trigger-based; the "kernel port
   lands" trigger is now met). Strong/weak scaling at ~1M DOF.
3. **Richer friction *when a problem needs it*:** deformable **exact-stick (return-map)** /
   **rate-and-state** (rate-dependent / stick-slip-instability).
4. **Self-contact** — make the existing 3D barrier/friction path complete a
   bounded end-to-end gate, then qualify deeper-neighborhood and distributed
   behavior.

Contact internals + the language-choice rationale: `docs/dev/contact.md`. Why numba (irregular per-pair)
vs vectorized-numpy (uniform/dense) vs Fortran (element kernels, dual-home): see api.md "Performance".
