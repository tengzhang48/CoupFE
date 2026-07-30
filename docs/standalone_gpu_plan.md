# Engineering Plan: UFL-to-PETSc Finite-Element Runtime with Contact and Geometry-Aware Mesh Generation

**Working origin:** `abaqus_ufl`  
**Target:** Python/F2PY-driven finite-element runtime with modern Fortran, PETSc, complex-step bulk tangents, contact mechanics, distributed refinement, MPI, and optional GPU backends  
**Plan date:** 2026-06-19  
**Status:** revision 3.0 — reality-aligned & strategically recalibrated against the existing `abaqus_ufl_lab` codebase (2026-06-20). The v2.2 body (sections 1–26) is retained as detailed reference; where it conflicts with the Revision 3.0 decisions below, **Revision 3.0 governs.**

> **Ownership update (2026-07-29):** this historical plan discusses mesh
> providers inside the runtime. The current architecture supersedes that
> placement: CoupFE core owns only `KernelMeshView` and reusable numerical
> algorithms. Gmsh, `meshio`, CAD, DMPlex/DMForest, geometry semantics, and
> format-specific I/O adapters belong to EDA, cardiac, or another application.
> If genuinely shared integration code emerges, it belongs in a separate
> optional package, not core.

### Revision 2.2 summary

This revision consolidates the standalone compiler plan with two application-level subsystems:

- a separate native contact operator using coarse-grained F2PY calls, PPF-inspired feature search, cubic barrier enforcement, CCD step limiting, explicit PSD and consistent tangent modes, PETSc integration, MPI, and GPU phases;
- a geometry-aware mesh path based on a coarse topological mesh plus authoritative analytical/CAD/NURBS/signed-distance geometry, parallel startup refinement, hidden DMPlex or forest backends, compact kernel views, and staged scaling to approximately ten million DOFs.

Python remains the primary model and solver wrapper. Modern Fortran remains the trusted CPU numerical core. Complex-step differentiation remains inside smooth bulk-element kernels; contact uses an algorithmic or generalized derivative.

---

## Revision 3.0 — reality alignment and strategic recalibration (2026-06-20)

This revision reconciles the v2.2 architecture with what the `abaqus_ufl_lab` codebase **already has**, and recalibrates priorities. The v2.2 body remains a valuable detailed reference; where it conflicts with the decisions here, **these decisions govern.**

### R3.1 What already exists (correcting the v2.2 Assumptions)

v2.2 §2 assumes "an initial PETSc connection." The lab is far further along — on a *different* axis than v2.2's standalone-Fortran target:

- **A working Python/PETSc runtime (`abaqus_ufl.fe`)**: distributed MPI assembly+solve (`solve_steps_mpi_local` + RCB partitioning + FieldSplit/Schur, rank-independent to 16 ranks), **compiled batched f2py assembly (~95×)**, AMG, staggered and monolithic coupled drivers, multi-material assembly, 3D Hex8, gmsh irregular meshing, VTU output. → v2.2 **Phase 4 (standalone PETSc CPU solver) and Phase 6 (MPI scalability) are substantially DONE**, in Python/PETSc.
- **Complex-step tangents in the generated `.for` kernels** (the `localp_eval_tangents` engine perturbs F/p/μ/∇μ). Only the matrix-free `Jv` is not yet built.
- **A backend-agnostic validation harness** (`abaqus_ufl/testing`): operator-level gates (block-definiteness, diffusive-flux, the coupled-scale gate), `RegimeManifest`, field-wise convergence, cross-backend tests. → much of v2.2 §14 exists.
- **Reference configuration (total Lagrangian, PK1)**: 481 `stress_pk1` vs 1 Cauchy — already the config v2.2 §4 recommends.
- **A multiphysics element zoo** (gel u-μ, MRE u-A, J2/rock plasticity, phase-field fracture, hydrogen u-c-φ, strain-gradient u-εₚ) — far beyond "smooth hyperelastic first."
- **Adjacent assets**: RetroMech (GPU FE — cuDSS, geometric multigrid, 64³; and a differentiable rigid-obstacle **SDF contact** = the plan's Stage-1), gmsh meshing, the CoupMechDeck preprocessor.

### R3.2 Product philosophy (the organizing principle)

Not a general PDE platform. The open FEA frameworks are large because they own meshing, BCs, time integration, and I/O — exactly the tedious per-problem glue that **AI now writes on demand**. The product is therefore:

> **A small, well-designed, heavily-tested scaffold + a concise model-setup pipeline + a validation harness that makes AI-generated glue trustworthy.**

The harness is the linchpin: AI writes the mesh/BC/loading/time-integration per problem; the operator gates + field-wise convergence + patch/energy tests **catch when it is wrong**. That trust layer — not breadth of features — is the moat.

### R3.3 The core abstraction (the spine)

**Everything is a residual-contributing operator with a pure `(R, tangent, state)` contract.** A bulk element group, a contact pair, a constraint, and a load all expose the same contract (residual contribution, optional assembled tangent, committed/trial state with no implicit commit). This single idea resolves the decisions below.

### R3.4 Decisions

1. **Configuration — reference / total-Lagrangian by default.** Already true (PK1). Abaqus does not force deformed config on the UEL path (only the UMAT path is Cauchy/current by convention). **Contact lives in current config regardless** — a current-config contact operator coexists with reference-config bulk; the bulk is never converted.
2. **Materials/parts — compositional groups, NOT the Abaqus parts/instances/sections/assembly hierarchy.** Follow the LAMMPS/CoupMPM model (groups by type/region + per-group operators) — which `assemble_groups` already does. Refine with per-group field/DOF masks (generalize the inert-DOF trick), per-group state layout, and interfaces (tie/cohesive/contact) as **separate operators between groups**.
3. **Runtime — keep Python/PETSc + batched f2py.** "No Python at runtime" (the v2.2 spine) is demoted to a later, optional productization. Batched f2py already gives compiled-kernel speed; the bottleneck is the solver/preconditioner, not Python; and v2.2 itself keeps Python/F2PY for contact (§25.2) and mesh (§26.13), so the standalone-Fortran target is only the bulk element loop — small marginal value, large cost (lost research velocity).
4. **Matrix-free `Jv` + GPU — DEFERRED, with a trigger.** The matrix-free payoff is gated on a *cheap-yet-effective preconditioner* `P`; for **coupled multiphysics**, finding `P` is a per-problem research project (cf. the phase-field-fracture preconditioning effort — real work for one system). Keep the **assembled tangent + direct (MUMPS) / FieldSplit-Schur** path for multiphysics. Revisit matrix-free/GPU only when a *specific* problem (a) outgrows assembled/direct memory AND (b) has a known good `P`.
5. **Contact — a separate `(R_contact, J_contact·v)` operator.** Stage-1 (rigid analytical SDF penalty) already exists in RetroMech. Adopt the *algorithm* from `ppf-contact-solver` (BVH broad-phase, point-triangle/edge-edge features, cubic barrier, additive CCD step bound, PSD rank-one Hessian) but **not its stack** (keep PETSc KSP/SNES; double precision; not its custom single-precision GPU CG).

### R3.5 Revised near-term roadmap (supersedes the v2.2 phase order)

The standalone-Fortran/GPU phases (v2.2 Phases 1–3, 5, 7–9) are deferred. Near-term workstreams, all sharing the operator contract:

- **B — Operator contract + compositional groups.** Formalize `(R, tangent, state)`; generalize `assemble_groups` with field/DOF masks, per-group state, interface operators. *The backbone.*
- **P — Model-setup pipeline.** The concise, AI-targetable front door: declare groups+materials, BCs, loading, time-integration, output. *The public face; what AI writes glue against.*
- **C — Contact operator.** RetroMech SDF (Stage-1) into the contract → deformable point-to-surface, learning from ppf-contact-solver.
- **D — Validation harness.** Gates for the operator contract, contact, and the assembled coupled solve (FieldSplit/Schur block-definiteness, the coupled-scale gate, patch/energy).

Sequence: **B + P first** (the contract + its front door, which C and D build against), then C and D in parallel. The solver stays on the proven assembled + FieldSplit/direct path — no dedicated solver workstream now.

### R3.6 Business context — embrace Abaqus, then extend

Abaqus is not going away; most target clients will keep using it. The strategy is therefore **embrace-and-extend, not replace** — and the architecture is already built for it.

**The core asset is one element definition with two homes.** A single UFL-like form → complex-step `.for` kernel runs **(i) inside Abaqus as a UEL/UMAT** and **(ii) standalone via f2py in `abaqus_ufl.fe`/CoupFE** — the *same validated kernel* (`drive_uel` / `CompiledAbaqusElement`). Write once; deploy to Abaqus or to CoupFE at scale.

- **Entry product — custom UEL/UMAT as a service.** Serve the large Abaqus installed base where it hurts most: writing and *validating* a custom element/material is painful. We deliver a validated `.for` they drop into their existing Abaqus workflow. **Low friction — they keep Abaqus.**
- **Extension — CoupFE.** The identical element runs standalone (open, MPI/PETSc-scalable, no license) — the migration path for clients who want to scale beyond, or eventually leave, Abaqus. Not a bet on people abandoning Abaqus (they won't); a bet on the constant need for *custom physics + trustworthy validation*.
- **The moat — the validation harness.** UEL/UMAT are notoriously hard to get right; the operator gates + field-wise convergence + cross-backend tests validate the element in **both Abaqus and CoupFE**. That cross-backend trust layer is the differentiator and is what makes bespoke client code deliverable with confidence.
- **Method breadth = upsell depth.** Beyond FE: CoupMPM/CoupLAM (particles/lattice), CoupLB/taichi-lbm-ibm (fluids/FSI), and **RetroMech** (differentiable inverse / calibration / design — a capability almost no consultancy has). Wedge: custom-FE entry; soft-matter/bio and inverse as the differentiated depth that wins the harder, higher-value engagements.

The "more scalable than Abaqus" claim is credible (open solvers/preconditioners, arbitrary ranks) but should be **benchmarked into evidence** (a strong/weak-scaling plot vs an Abaqus reference) before it becomes a sales claim.

### R3.7 Naming (decided)

`abaqus_ufl` cannot be the public name (`Abaqus` is a Dassault trademark; `UFL` is FEniCS's), but it remains the internal lineage for the **Abaqus-backend** path.

- **Engine / package: `CoupFE`** — fits the `Coup*` family (CoupLAM, CoupMPM, CoupLB) and reuses the **CoupMechDeck** preprocessor for meshing/BC glue. (Starting point is the existing `abaqus_ufl` + `CoupFE` together: the Abaqus UEL/UMAT backend and the standalone runtime, sharing one element core.)
- **Umbrella consulting brand: `CoupMech Lab`** — read as **"Coupled Mechanics Lab,"** the field, spanning every product line (CoupFE, CoupMPM/LAM/LB, taichi-lbm-ibm, RetroMech) regardless of substrate. "Lab" signals research-grade custom work — the consulting posture.
- **`Piola`** — optional sharp sub-brand for the CoupFE / solid-mechanics line (the PK1 stress; credible with the UEL/UMAT solids audience). Too narrow to be the umbrella (it means *solids*).

---

## 1. Executive recommendation

The existing `abaqus_ufl` package should be evolved rather than replaced. The important refactoring is to separate the package into:

1. a **UFL compiler core** that understands forms, tensors, quadrature, state variables, and complex-safe scalar operations;
2. multiple **code-generation backends**;
3. a **standalone runtime** that owns the mesh, degrees of freedom, state, loads, constraints, and PETSc solvers.

The recommended first production architecture is:

```text
Python/UFL model
      |
      v
UFL analysis and validation
      |
      v
Form IR  ---------- element/basis/quadrature metadata
      |
      v
Typed Kernel IR --- scalar expressions, loops, memory, state effects
      |
      +------------------+--------------------+-------------------+
      |                  |                    |                   |
      v                  v                    v                   v
Abaqus UEL         Modern Fortran CPU    C++/Kokkos GPU     diagnostic backend
backend             backend               backend             (JSON/text)
      |                  |                    |
      +------------------+--------------------+
                         |
                         v
              Standalone PETSc runtime
```

The key technical choices are:

- **Python remains a build-time/compiler dependency**, not a runtime dependency of the generated solver.
- **Modern Fortran is the primary CPU runtime and CPU-kernel language.** Use a conservative Fortran 2008/2018 subset for portability.
- **PETSc remains real-valued.** Complex arithmetic exists only in temporary element/kernel calculations used to obtain the local tangent or a Jacobian-vector product.
- Generate at least four related kernels from one residual definition:
  - real residual;
  - complex residual;
  - full local tangent by column-wise complex step;
  - matrix-free Jacobian-vector product by one complex directional evaluation.
- Make **matrix-free complex-step `Jv` the preferred high-order/GPU operator**, while retaining the exact assembled local tangent for verification, low-order problems, and preconditioning.
- Treat GPU support as a separate backend. The recommended portable production route is a **small C++/Kokkos device layer** called through `bind(C)`. A pure-Fortran GPU backend can be maintained as an NVIDIA-oriented option or experimental OpenMP-target backend.
- Preserve the **Abaqus backend as a regression oracle** during development.

This is a compiler project, not a general Python-to-Fortran translator. Only the UFL expression graph and explicitly supported model metadata should be translated.

---

## 2. Assumptions

> **Superseded by Revision 3.0 §R3.1.** This section understates the codebase: a full Python/PETSc MPI runtime, validation harness, reference-config (PK1) elements, and a multiphysics element zoo already exist. Read §R3.1 for the actual baseline.

This plan assumes that the current package already has the following capabilities:

- it accepts a useful subset of UFL;
- it generates working Abaqus UEL-related Fortran;
- it can evaluate an element residual;
- it computes the element tangent using complex-step perturbations inside the element;
- it has an initial PETSc connection;
- the existing Abaqus implementation can supply trusted reference results.

The plan does **not** assume a particular mesh library, element family, material library, or GPU vendor. Those should be isolated behind interfaces.

---

## 3. Product goals

### 3.1 Primary goals

The standalone package should be able to:

- compile a supported UFL form into self-contained numerical kernels;
- build an executable or library without requiring Python at runtime;
- solve nonlinear finite-element problems using PETSc SNES/KSP;
- run in serial and with MPI;
- use a real-valued PETSc build while retaining complex-step differentiation locally;
- support both assembled and matrix-free Jacobian operators;
- support history/state variables without corrupting them during Newton or complex-step evaluations;
- reproduce the existing Abaqus UEL residual, tangent, and solution for common test problems;
- add GPU execution without changing the UFL model;
- emit enough metadata and diagnostics to make generated code auditable.

### 3.2 First-release non-goals

Do not make the first standalone release responsible for all of the following:

- translating arbitrary Python;
- supporting every UFL operator;
- general contact;
- arbitrary remeshing;
- every Abaqus element/load convention;
- every nonlinear nonsmooth constitutive model;
- a complete pre/post-processing GUI;
- a new linear algebra package;
- replacing PETSc mesh and solver infrastructure wholesale.

A narrow, well-tested compiler is more valuable than a broad translator with unclear semantics.

---

## 4. Recommended first supported problem class

A practical version-0 scope is:

- quasi-static solid mechanics;
- total-Lagrangian/reference-configuration weak forms by default;
- displacement-based tetrahedral and/or hexahedral elements already supported by `abaqus_ufl`;
- cell and exterior-facet integrals;
- Dirichlet constraints, dead tractions, body forces, and optionally follower loads;
- smooth hyperelastic materials first;
- path-dependent material state added after the pure state protocol is proven;
- exact assembled complex-step tangent on CPU;
- matrix-free complex-step `Jv` on CPU;
- MPI distribution;
- one GPU proof of concept, followed by the portable backend.

The compiler should not force Cauchy stress. A reference formulation can use, for example,

\[
R_e(u_e)=\int_{\Omega_{0e}} B_F^T P\,dV - f_{e,\mathrm{ext}},
\]

and differentiate the element residual directly. Cauchy stress can be generated only for output, a spatial load, or an Abaqus-specific adapter.

---

## 5. Design principles

### 5.1 One residual is the source of truth

Do not maintain separate hand-written implementations for residual, tangent, and `Jv`. The mathematical source of truth should be the residual form. The other kernels are generated transformations of that form.

### 5.2 Element evaluation is functionally pure

Conceptually, an element call should behave as:

\[
(R_e, s_{\mathrm{trial}}, d_e)
= \mathcal{E}(X_e,u_e,s_n,p,t),
\]

where:

- `X_e` is reference geometry;
- `u_e` is the current element unknown;
- `s_n` is committed state from the previous accepted load/time step;
- `p` is immutable parameter data;
- `t` contains load/time data;
- `R_e` is the residual contribution;
- `s_trial` is a candidate state, never implicitly committed;
- `d_e` contains optional diagnostics.

Repeated calls with the same inputs must produce the same outputs. This is essential for complex-step columns, line searches, matrix-free Krylov products, threading, and GPU execution.

### 5.3 Complex safety is part of the language semantics

The IR must distinguish an ordinary real tensor contraction from a Hermitian complex inner product. It must not let backend language defaults silently introduce conjugation or discard imaginary parts.

### 5.4 Stable interfaces, replaceable implementations

The runtime should call generated kernels through a stable contract. The Fortran CPU, Abaqus, Kokkos, CUDA Fortran, and diagnostic backends may implement that contract differently.

### 5.5 Correctness before device optimization

Each GPU kernel must first match the CPU reference. Performance tuning begins only after residual, tangent, `Jv`, state, and Newton convergence tests pass.

---

## 6. Compiler architecture

UFL itself is a domain-specific language for finite-element forms, and FFCx uses staged analysis, IR construction, and code generation. The new compiler should follow that separation rather than directly printing Fortran while walking UFL nodes.

### 6.1 Stage A: model ingestion

Input may be a Python module or a serialized build description containing:

- UFL forms;
- element and quadrature choices;
- coefficient declarations;
- state-variable declarations;
- material parameters;
- backend options;
- generated-kernel names;
- optional output quantities.

Output is a normalized model object. Arbitrary Python execution is outside the compiler contract after this stage.

### 6.2 Stage B: UFL analysis and validation

Responsibilities:

- identify arguments, coefficients, geometry, and measures;
- split cell, exterior-facet, and interior-facet integrals;
- infer tensor shapes and free-index structure;
- validate the supported UFL subset;
- detect nonsmooth or complex-unsafe operations;
- classify coefficients as constant, element, quadrature, nodal, or global;
- record required basis values and derivatives;
- identify state reads and writes;
- produce useful source-level diagnostics.

Unsupported constructs should fail at compile time with an explanation and the UFL expression path. They should never silently fall back to an incorrect interpretation.

### 6.3 Stage C: Form IR

The Form IR describes finite-element meaning rather than target-language syntax. It should contain:

- integral type and domain;
- cell topology and geometric dimension;
- trial/test field layouts;
- basis tables and derivative tables;
- quadrature points and weights;
- geometry mapping requirements;
- coefficient access rules;
- state layout;
- residual output layout;
- load and boundary metadata;
- reference/current configuration requirements.

This IR is the correct place to decide what can be precomputed, what is constant per element, and what varies per quadrature point.

### 6.4 Stage D: typed Kernel IR

The Kernel IR should be low-level enough for both Fortran and C++/GPU code generation. Recommended node information:

- scalar kind: real, complex, integer, logical;
- precision: initially binary64;
- tensor shape and storage layout;
- explicit loop bounds;
- explicit reductions and contractions;
- memory space or storage class;
- const/read/write effects;
- branch/control-flow nodes;
- source-expression provenance;
- estimated operation count;
- reusable temporary lifetime.

Recommended operations include:

- arithmetic and fused multiply-add where safe;
- transcendental functions with defined complex behavior;
- determinant, inverse, trace, transpose, and tensor contraction;
- basis interpolation and gradient evaluation;
- explicit gather/scatter slots;
- state load/store;
- diagnostics and validity checks.

Do **not** represent a contraction by a backend-specific `dot_product` call. Represent it as a semantic contraction and let each backend emit a safe implementation.

### 6.5 Stage E: optimization

Start with conservative transformations:

- constant folding;
- common-subexpression elimination;
- loop-invariant code motion;
- quadrature-constant hoisting;
- dead-code elimination;
- temporary reuse;
- algebraic simplification proven valid for both real and complex arithmetic;
- optional unrolling for fixed, small tensor dimensions.

Avoid aggressive transformations that assume values are real, reassociate floating-point expressions unexpectedly, or erase tiny imaginary terms. Complex-step validation must be run after every optimization pass is introduced.

### 6.6 Stage F: backend code generation

Initial backends:

1. `abaqus_fortran` — preserves the current UEL path.
2. `fortran_cpu` — standalone real and complex kernels.
3. `kokkos` — portable device kernel and launcher.
4. `ir_dump` — human-readable and JSON output for debugging/tests.

Later backends may include OpenMP-target Fortran, CUDA Fortran, HIP, SYCL, or libCEED-compatible pointwise physics.

### 6.7 Generated manifest

Every generated kernel set should include a machine-readable manifest containing:

- compiler version and git commit;
- UFL version;
- source-form hash;
- element and quadrature metadata;
- supported scalar modes;
- state layout;
- kernel ABI version;
- backend and compiler flags;
- list of complex-safety warnings;
- estimated workspace sizes;
- generated source filenames.

This makes generated results reproducible and allows the runtime to reject incompatible kernels.

---

## 7. Kernel contract

### 7.1 Required generated operations

For each integral/kernel family, generate:

```text
cell_residual_real
cell_residual_complex
cell_tangent_cs
cell_jvp_cs
cell_state_real
cell_output_real
```

Not every function must be a separate source body. The wrappers may call a shared generated implementation, but their observable behavior should be distinct.

### 7.2 Real residual

Inputs:

- reference coordinates;
- element solution;
- committed state;
- coefficients and parameters;
- load/time data;
- optional branch/control cache.

Outputs:

- real element residual;
- real trial state;
- diagnostics/status.

### 7.3 Complex residual

Inputs are analogous, except the perturbed solution and any differentiable coefficients are complex. Committed state remains real, but temporary state generated from the perturbation may be complex.

Outputs:

- complex element residual;
- optional temporary complex trial state, normally discarded;
- diagnostics/status.

### 7.4 Full local tangent

For `n_e` local degrees of freedom,

\[
K_e(:,j)=\frac{\operatorname{Im} R_e(u_e+i h_j e_j)}{h_j}.
\]

The wrapper must restart every column from the same committed state and the same real control/branch information.

### 7.5 Matrix-free local Jacobian action

For a local direction `v_e`,

\[
J_e(u_e)v_e
=\frac{\operatorname{Im}R_e(u_e+i h v_e)}{h}.
\]

This requires one complex residual evaluation per element and Krylov `MatMult`, rather than one evaluation per local tangent column.

### 7.6 Suggested Fortran module API

The direct Fortran API can use explicit-shape or assumed-shape arrays internally:

```fortran
module generated_element_kernel
  use, intrinsic :: iso_fortran_env, only : real64
  implicit none
  private
  public :: cell_residual_real, cell_residual_complex
  public :: cell_tangent_cs, cell_jvp_cs

contains

  pure subroutine cell_residual_real(x, u, state_n, params, load, r, state_trial, status)
    real(real64), intent(in)  :: x(:), u(:), state_n(:), params(:), load(:)
    real(real64), intent(out) :: r(:), state_trial(:)
    integer,      intent(out) :: status
  end subroutine

  pure subroutine cell_residual_complex(x, uc, state_n, params, load, rc, status)
    real(real64),    intent(in)  :: x(:), state_n(:), params(:), load(:)
    complex(real64), intent(in)  :: uc(:)
    complex(real64), intent(out) :: rc(:)
    integer,         intent(out) :: status
  end subroutine

end module
```

`pure` is desirable where possible, but should not be forced if compiler/runtime diagnostics require controlled side effects. Production kernels must never allocate, perform I/O, or modify committed global state.

### 7.7 Stable mixed-language ABI

For interoperability, expose flat arrays with `iso_c_binding`:

```fortran
subroutine kernel_cell_jvp_cs(ndof, nx, nstate, nparam, nload, &
                              x, u, v, state_n, params, load, h, jv, status) &
                              bind(C, name="kernel_cell_jvp_cs")
  use, intrinsic :: iso_c_binding
  integer(c_int), value :: ndof, nx, nstate, nparam, nload
  real(c_double), intent(in)  :: x(*), u(*), v(*), state_n(*), params(*), load(*)
  real(c_double), value       :: h
  real(c_double), intent(out) :: jv(*)
  integer(c_int), intent(out) :: status
end subroutine
```

Use explicit dimensions in arguments and flat contiguous storage. Do not expose compiler-specific Fortran descriptors across the language boundary.

---

## 8. Complex-step implementation policy

### 8.1 PETSc scalar type

Use a **real PETSc configuration**. The global unknown, residual, matrices, communication buffers, and Krylov vectors remain real. Complex values are temporary element-local values.

This gives two execution modes:

```text
assembled tangent:
  real global u -> local complex seed per column -> real Ke -> real PETSc matrix

matrix-free Jv:
  real global u,v -> local complex u+i h v -> real local Jv -> real PETSc vector
```

### 8.2 Step-size policy

Do not hard-code an extremely small value without tests. Start with a configurable dimensionless base such as:

```text
h0 = 1.0e-30
```

For a tangent column, use a scaled step such as:

\[
h_j=h_0\max(1,|u_j|/u_{\mathrm{scale},j})u_{\mathrm{scale},j},
\]

or equivalently perturb a nondimensionalized local variable. For `Jv`, normalize the direction or select `h` so that `h v` remains in a safe range.

The test suite should sweep several values, for example `1e-16` through `1e-100`, to identify platform/compiler sensitivity. Avoid compiler modes that flush or remove small imaginary components.

### 8.3 Complex-safe operation rules

The compiler must reject or specially lower the following operations:

- conjugation introduced by a language intrinsic;
- `abs(z)` when the intended real operation was an analytic continuation;
- `real(z)` in the differentiable arithmetic path;
- comparison of a complex perturbation;
- `max`, `min`, `sign`, clipping, or saturation on perturbed values;
- Hermitian norm or dot product;
- eigensolvers, SVD, or polar decomposition implementations that use conjugation;
- branch logic whose active branch changes under the perturbation.

A crucial Fortran example is:

```fortran
! Unsafe for analytic complex-step continuation:
s = dot_product(a, b)

! Correct for an ordinary tensor contraction:
s = sum(a*b)
```

For complex arguments, Fortran `dot_product` conjugates the first vector. The IR should emit an explicit contraction so this mistake cannot occur.

For a Euclidean norm originally written as

\[
\sqrt{x\cdot x},
\]

the analytic continuation is based on `sum(z*z)`, not `sum(conjg(z)*z)`.

### 8.4 Branches and active sets

For smooth hyperelasticity, this issue is small. For plasticity, damage, contact-like penalties, and other switching models:

1. evaluate the real base state;
2. record branch/active-set decisions in a small control cache;
3. replay the same branch during all complex evaluations at that Newton point;
4. report when the base point is close to a switching surface;
5. accept that a classical tangent may not exist exactly at the switch.

This gives the derivative of the active smooth branch. A later semismooth or generalized-derivative mode may be added for genuinely nonsmooth models.

### 8.5 State-variable protocol

Use three state concepts:

- `state_committed`: accepted at the end of the previous load/time step;
- `state_trial`: computed from the current real Newton iterate;
- `state_cs`: temporary complex state inside a derivative evaluation.

Rules:

- every residual and derivative evaluation starts from `state_committed`;
- no complex-step column may see the state produced by a previous column;
- line-search trial residuals never commit state;
- after SNES accepts the converged solution, run one final real state evaluation and commit that result;
- rollback requires only retaining `state_committed`.

### 8.6 Residual completeness

The differentiated residual must include every displacement-dependent term whose derivative is required:

- internal forces;
- follower pressure or traction terms;
- stabilization terms;
- penalty terms;
- inertia or damping terms in a transient extension;
- constraint contributions implemented inside the operator.

If a load or boundary contribution is assembled by a separate kernel, it must also supply its residual and `Jv`/tangent contribution.

### 8.7 Verification against other derivatives

For every kernel, support optional checks:

- complex-step tangent versus an available analytic tangent;
- complex-step `Jv` versus `K_e v`;
- complex-step `Jv` versus a carefully scaled central difference;
- tangent symmetry where the mathematical model predicts symmetry;
- Newton convergence rate.

Complex-step is the production derivative method, but independent checks are still necessary to detect nonanalytic code paths and state errors.

---

## 9. Standalone modern-Fortran runtime

### 9.1 Role of modern Fortran

Modern Fortran should own:

- model setup and command-line/PETSc options;
- CPU mesh and element-batch data;
- load-step/time-step driver;
- state management;
- CPU assembly and matrix-free callbacks;
- PETSc SNES/KSP integration;
- file I/O and diagnostics;
- calling generated CPU kernels;
- calling the C/C++ GPU bridge through `bind(C)`.

Use modern language features for organization, not dynamic abstraction inside hot loops.

### 9.2 Recommended language subset

Use:

- modules and submodules;
- derived types outside hot kernels;
- `iso_fortran_env` and `iso_c_binding`;
- allocatable arrays with clear ownership;
- explicit interfaces;
- `error stop` only at top-level fatal boundaries;
- OpenMP for CPU parallel loops;
- `do concurrent` where semantically useful, without assuming it guarantees GPU execution.

Avoid in generated hot kernels:

- allocation/deallocation;
- polymorphic dispatch;
- type-bound virtual calls;
- procedure pointers;
- I/O;
- hidden array temporaries;
- compiler-specific descriptors at ABI boundaries;
- recursion unless proven safe and beneficial.

### 9.3 Runtime module layout

Suggested modules:

```text
runtime/
  precision_m.F90
  errors_m.F90
  options_m.F90
  mesh_api_m.F90
  mesh_plex_m.F90             # optional PETSc DMPlex provider
  element_batches_m.F90
  dofmap_m.F90
  constraints_m.F90
  quadrature_state_m.F90
  load_step_m.F90
  kernel_registry_m.F90
  residual_assembly_m.F90
  jacobian_assembly_m.F90
  matrix_free_m.F90
  petsc_snes_m.F90
  output_m.F90
  gpu_bridge_m.F90
```

### 9.4 Mesh strategy

Long term, PETSc DMPlex is a reasonable provider of topology, partitioning, labels, and distributed ownership. For performance, the runtime should extract fixed **element batches** at setup:

- same cell type;
- same polynomial order;
- same quadrature rule;
- same material/kernel;
- same state layout.

Generated kernels should operate on these flat batches rather than traversing a general mesh object inside the hot loop.

If the current PETSc connection already has a custom mesh representation, keep it behind `mesh_api_m` and add DMPlex later.

### 9.5 Build system

Use CMake for the runtime because the final system may combine:

- Fortran;
- C/C++;
- PETSc;
- MPI;
- Kokkos or vendor GPU toolchains;
- generated source files.

The Python package should provide a command such as:

```bash
abaqus-ufl compile model.py \
  --backend fortran-cpu \
  --emit-complex \
  --output build/generated
```

CMake then consumes the manifest and generated sources. The generated executable should not import Python.

---

## 10. PETSc integration

### 10.1 Nonlinear residual callback

The SNES function callback should:

1. update ghost values of the global solution;
2. gather element unknowns;
3. evaluate real element/facet residual kernels;
4. write element-local residual values into a preallocated COO value array;
5. call PETSc vector COO assembly;
6. apply constraints consistently;
7. leave committed material state unchanged.

PETSc documents `VecSetPreallocationCOO`/`VecSetValuesCOO` as an efficient GPU-oriented assembly path that sums repeated entries and performs assembly without a separate begin/end call.

### 10.2 Exact assembled Jacobian mode

At setup:

- generate row/column COO indices from element DOF maps;
- preallocate the PETSc matrix once;
- preserve the same ordering for every nonlinear iteration.

At a Jacobian update:

1. evaluate each local tangent by complex-step columns;
2. flatten local matrices into the matching COO value order;
3. call `MatSetValuesCOO`;
4. apply constraint rows/columns consistently.

This mode is the initial correctness reference and a useful choice for small/low-order elements.

### 10.3 Matrix-free exact `Jv` mode

Create a PETSc shell matrix whose `MATOP_MULT` performs:

1. ghost update of `v`;
2. gather `u_e` and `v_e`;
3. evaluate `R_e(u_e+i h v_e)` once per element;
4. extract `imag(R_e)/h`;
5. assemble into the output vector;
6. enforce constrained rows consistently.

The current nonlinear solution `u` and committed state belong in the shell context. When the Newton linearization point changes, update that context and increment the shell matrix state as required by PETSc.

### 10.4 Operator/preconditioner split

Use PETSc’s separate Jacobian operator and preconditioning matrix:

```text
J: exact matrix-free complex-step action
P: assembled approximate or lagged matrix
```

Initial preconditioner choices:

- exact assembled complex-step tangent, updated every Newton step;
- exact tangent updated less frequently;
- lower-order or reduced-quadrature elasticity operator;
- block Jacobi/ASM using local tangents;
- PETSc GAMG on an approximate assembled operator;
- problem-specific field split for mixed formulations later.

A matrix-free GPU operator without an effective preconditioner is not a complete solver strategy. Preconditioning must be treated as a first-class workstream.

### 10.5 Constraints

Prefer elimination/lifting when practical. If identity-row enforcement is used, matrix-free `Jv` must implement the same algebra:

- zero constrained components before element application when appropriate;
- set constrained output rows to the corresponding input value for an identity row;
- ensure the residual uses the same convention;
- ensure the preconditioner uses identical constrained rows.

### 10.6 MPI communication

Separate communication from kernel execution:

```text
begin ghost update
perform work that uses owned-only data, if available
end ghost update
launch element batches needing ghosts
assemble/scatter contributions
```

Start with a correct blocking implementation. Add overlap only after profiling.

### 10.7 PETSc device access and the mixed-language bridge

Current PETSc documentation marks `VecGetArrayAndMemType` and `VecGetKokkosView` as having no Fortran support. Therefore, a pure-Fortran runtime should not depend on receiving PETSc device pointers directly through the Fortran API.

Use a small C++ bridge that:

- receives PETSc `Vec`/`Mat` handles from Fortran;
- obtains raw device pointers or Kokkos views;
- launches generated GPU kernels;
- calls PETSc COO assembly where appropriate;
- returns only PETSc error codes and compact status information.

The rest of the solver remains Fortran.

---

## 11. CPU backend plan

### 11.1 Generated CPU kernels

Generate separate, specialized real and complex procedures rather than depending on run-time scalar polymorphism. This gives the compiler fixed types and avoids dynamic dispatch.

Recommended kernel properties:

- fixed loop bounds when the element is fixed;
- stack or caller-provided workspace;
- no allocation;
- explicit small tensor contractions;
- quadrature loop outer or inner ordering selected by profiling;
- optional generated operation counters;
- clear `status` return for invalid Jacobians, failed local solves, or NaNs.

### 11.2 CPU parallelism

Start with OpenMP over element batches. Assembly choices:

- COO values: each element writes to a unique segment, so the element loop is race free;
- PETSc sums duplicate global entries afterward;
- avoid atomics inside the generated kernel;
- maintain one committed-state owner per element/quadrature point.

### 11.3 Compiler matrix

Continuously test at least:

- GNU Fortran;
- one additional production compiler such as Intel `ifx`;
- NVIDIA `nvfortran` when the NVIDIA backend is enabled.

Use debug builds with bounds checking, floating-point exceptions where practical, and uninitialized-value detection. Use optimized builds without unsafe fast-math until complex-step tests prove a particular flag safe.

---

## 12. GPU architecture

### 12.1 Recommended backend order

1. **CPU Fortran reference** — mandatory.
2. **C++/Kokkos portable GPU backend** — recommended production device path.
3. **NVFORTRAN OpenMP-target or CUDA Fortran backend** — useful NVIDIA-specific path and comparison.
4. **libCEED integration** — optional after evaluating how to represent complex-step derivatives.

Kokkos currently exposes CUDA, HIP, SYCL, OpenMP, and other execution backends, and provides `Kokkos::complex` as a device-compatible replacement for `std::complex`. That aligns well with a generated complex residual kernel.

### 12.2 Why not make pure Fortran the only GPU path

Modern Fortran is suitable for the standalone CPU runtime, and NVIDIA’s compilers support GPU programming models. However:

- compiler support and feature behavior differ across vendors;
- PETSc’s relevant device-view interfaces currently do not have direct Fortran support;
- a portable CUDA/HIP/SYCL path is easier through Kokkos;
- mixed-language interoperability can be kept very small and stable.

A Fortran GPU backend remains valuable, but the compiler IR should not be designed around one vendor’s directives.

### 12.3 Device-resident data

Keep the following resident on the GPU throughout nonlinear iterations:

- coordinates;
- element connectivity/DOF maps;
- basis/quadrature tables;
- material parameters;
- committed state;
- current solution and direction vectors;
- element residual COO values;
- tangent COO values when assembled;
- branch/control caches;
- diagnostics counters.

Do not copy the full solution, state, or element output to the host on each residual or `Jv` call.

### 12.4 Kernel launch models

#### Real residual

Recommended initial mapping:

```text
one team/block per element
threads cooperate over quadrature points and/or residual entries
```

For very small low-order elements, one thread per element may be competitive. Support both through a backend policy.

#### Matrix-free complex-step `Jv`

```text
one team/block per element
construct local uc = u + i*h*v
one complex residual evaluation
write real imag(rc)/h into element COO values
```

This is the preferred first GPU derivative kernel.

#### Full local tangent

Two reasonable mappings are:

```text
A. one work item per (element, tangent column)
B. one block per element, threads cover seed columns
```

The second allows shared reuse of geometry and real branch information but can create high register/shared-memory demand. Implement the simple `(element,column)` mapping first and optimize only after measurement.

### 12.5 Residual and matrix assembly

Precompute COO indices once. Generated device kernels write values in exactly that ordering. Then use PETSc’s COO vector/matrix paths, which are designed to handle repeated entries and have CUDA/HIP/Kokkos matrix implementations.

This avoids fine-grained global atomics from the physics kernel and makes CPU/GPU assembly share the same logical ordering.

### 12.6 Complex representation

For the Kokkos backend, begin with `Kokkos::complex<double>`. Keep the IR independent of that type so a later split-complex representation can be tested:

```cpp
struct cs_complex {
  double re;
  double im;
};
```

A custom pair can reduce ABI ambiguity and permit targeted code generation, but should not be introduced until the standard Kokkos complex implementation has been benchmarked.

### 12.7 Register pressure and workspace

Complex arithmetic roughly doubles scalar storage, and local state/material algorithms may add more. The backend should report estimated per-element workspace and allow:

- recomputation versus temporary storage;
- quadrature tiling;
- seed-column tiling for full tangents;
- element batching by kernel complexity;
- optional local-memory workspace;
- fallback to CPU for kernels that exceed configured limits.

### 12.8 libCEED assessment

libCEED is attractive because its operator model separates element restriction, basis action, and quadrature physics, and it provides CPU/GPU backends plus PETSc examples. It is especially compelling for high-order matrix-free real operators.

However, the current documented `CeedScalar` choices are real single or double precision, not complex. Therefore, preserving the present complex-step method would require one of:

- representing the complex value as two real fields and generating complex arithmetic manually inside a QFunction;
- using libCEED for the real residual while providing `Jv` through a separate backend;
- replacing complex step with an analytic or AD-generated derivative for the libCEED path.

For that reason, libCEED should be an evaluated secondary path, not the first GPU milestone.

---

## 13. State and nonlinear-solver lifecycle

### 13.1 Load/time-step sequence

Recommended sequence:

```text
start step
  copy/retain committed state from previous accepted step
  initialize SNES solution
  while SNES evaluates residuals/Jacobians/line searches:
      compute all trial responses from committed state
      never commit
  after SNES convergence:
      recompute final real trial state at accepted solution
      validate it
      commit atomically
  on step failure:
      retain old committed state
end step
```

### 13.2 Local material iteration

If a quadrature-point constitutive update has a local Newton solve:

- the real and complex kernels must execute equivalent algorithms;
- stopping conditions should normally be based on the real base evaluation or a frozen iteration count/control path;
- complex perturbations must not change the number of local iterations unpredictably;
- local failure must propagate through the element `status` and into PETSc’s nonlinear failure handling.

For the first path-dependent model, choose one with a well-understood smooth consistent update and an analytic reference tangent.

### 13.3 Determinism

For identical inputs and partitioning, kernel outputs should be deterministic. Global reductions may vary slightly with MPI/GPU ordering, so regression tolerances should distinguish local kernel determinism from globally reduced floating-point order.

---

## 14. Validation strategy

### 14.1 Compiler unit tests

Test each supported IR operation with real and complex inputs:

- arithmetic;
- powers and transcendental functions;
- contractions;
- determinant and inverse;
- tensor indexing/permutation;
- basis interpolation;
- geometry mapping;
- conditional lowering;
- state access;
- common-subexpression elimination.

For each operation, compare:

- generated Fortran real result;
- generated Fortran complex result;
- Python/NumPy or high-precision reference;
- Kokkos result when enabled.

### 14.2 Complex-safety negative tests

The compiler should intentionally reject or warn on models containing:

- `abs` in a differentiable path;
- real extraction before the residual is complete;
- conjugating contractions;
- unsupported comparisons;
- unsupported spectral operations;
- state mutation outside the trial-state object.

These tests are as important as successful compilation tests.

### 14.3 Element tests

For every element/kernel combination:

- zero-displacement residual;
- rigid translation and rotation where appropriate;
- constant-strain/patch tests;
- energy-gradient check;
- local tangent versus `Jv`;
- local tangent versus analytic reference if available;
- local tangent versus finite difference over a safe step range;
- permutation/orientation tests;
- distorted geometry tests;
- invalid/inverted element diagnostics.

### 14.4 Global nonlinear tests

Include:

- single-element homogeneous deformation;
- multi-element patch;
- cantilever or block benchmark;
- finite-deformation hyperelastic benchmark;
- follower-load case when supported;
- path-dependent loading/unloading after state support is added.

Verify:

- residual norms;
- solution fields;
- reactions;
- energy;
- number and quality of Newton iterations;
- expected quadratic convergence near the solution for smooth problems;
- equality of assembled `K v` and matrix-free `Jv`.

### 14.5 Abaqus parity tests

Use the existing Abaqus backend as an external reference:

- same mesh and material parameters;
- same reference configuration and loading;
- compare element residual/tangent in a small harness where possible;
- compare global force-displacement curves;
- compare integration-point outputs;
- document any convention differences rather than hiding them.

### 14.6 CPU/GPU parity

For each GPU milestone, compare:

- real element residual;
- complex element residual;
- `Jv`;
- full local tangent where implemented;
- global residual;
- SNES solution and convergence history;
- committed state after each accepted step.

Use both regular and distorted meshes and several complex-step sizes.

### 14.7 Continuous-integration matrix

Suggested CI levels:

```text
Level 1: Python compiler unit tests
Level 2: generated Fortran compile/run tests
Level 3: PETSc serial integration tests
Level 4: MPI tests
Level 5: Kokkos host backend tests
Level 6: CUDA/HIP/SYCL tests on available dedicated runners
Level 7: periodic Abaqus comparison outside public CI if licensing requires it
```

---

## 15. Performance engineering plan

### 15.1 Establish baselines

Record separately:

- UFL analysis/code-generation time;
- generated source size and compile time;
- element residual throughput;
- complex residual throughput;
- full tangent throughput;
- matrix-free `Jv` throughput;
- residual/Jacobian assembly time;
- MPI communication time;
- KSP iterations and preconditioner setup/application time;
- full Newton step and solve time;
- host-device transfer volume.

### 15.2 PETSc profiling

Wrap major operations in PETSc log events:

```text
KernelResidual
KernelJv
KernelTangent
ResidualCOOAssembly
JacobianCOOAssembly
GhostUpdate
StateCommit
GPUDataTransfer
```

This makes performance visible in PETSc logging without a separate profiler.

### 15.3 Optimization order

Optimize in this order:

1. remove unnecessary host-device transfers;
2. reduce repeated geometry and basis work;
3. improve memory layout and batch homogeneity;
4. tune element/team mapping;
5. reduce complex temporary storage;
6. improve preconditioning and reduce KSP iterations;
7. overlap MPI communication;
8. test selective compiler fast-math options only with derivative regression tests.

A faster element kernel is not useful if the solver performs many more Krylov iterations or copies vectors to the host.

### 15.4 Expected decision point: assembled versus matrix free

Use benchmark evidence rather than a universal rule:

- low-order, small `n_e`: assembled tangent may be competitive;
- high-order or large state: matrix-free `Jv` should become more attractive;
- difficult nonlinear models may need a strong assembled/physics-based preconditioner even when `J` is matrix free.

The runtime should support both under one command-line interface.

---

## 16. Phased implementation roadmap

### Phase 0 — Freeze the working reference

### Work

- tag the current working `abaqus_ufl` state;
- select a small set of trusted Abaqus models;
- archive generated Fortran, element outputs, global results, and solver logs;
- document the current complex-step algorithm and step size;
- identify all current UFL operations and element types in use;
- create a minimal standalone element-driver harness around generated code.

### Deliverables

- reproducible reference repository/tag;
- baseline test-data bundle;
- current architecture note;
- list of supported and accidentally supported features.

### Exit criteria

- reference tests reproduce on a clean environment;
- element residual and tangent can be called outside Abaqus in a small test harness.

---

### Phase 1 — Extract the compiler core

### Work

- separate UFL analysis from Fortran text emission;
- introduce Form IR and typed Kernel IR;
- add expression provenance and diagnostics;
- implement `ir_dump` backend;
- move Abaqus-specific naming, arrays, and conventions into the Abaqus backend;
- add deterministic source hashes and manifests.

### Deliverables

- compiler package with staged API;
- documented IR schemas;
- Abaqus backend generated from the new IR;
- IR snapshot tests.

### Exit criteria

- existing Abaqus reference models still pass;
- no Abaqus convention appears in the common Kernel IR unless it is truly generic.

---

### Phase 2 — Generate modern-Fortran standalone kernels

### Work

- implement `fortran_cpu` real backend;
- implement complex specialization from the same IR;
- add flat C ABI wrappers;
- enforce complex-safe contractions and intrinsics;
- add generated workspace metadata;
- build a standalone one-element driver.

### Deliverables

- real and complex generated Fortran modules;
- C-interoperable wrappers;
- one-element executable;
- complex-operation test suite.

### Exit criteria

- real residual matches Abaqus backend/reference;
- complex residual produces stable derivatives over a step-size sweep;
- generated code compiles with at least two Fortran compilers.

---

### Phase 3 — Productionize element complex-step derivatives

### Work

- implement local full tangent wrapper;
- implement local `Jv` wrapper;
- implement branch/control cache;
- implement committed/trial/complex state protocol;
- add `K_e v` versus `Jv` tests;
- add analytic-tangent comparisons for selected models;
- add diagnostics for nonanalytic operations.

### Deliverables

- exact local tangent API;
- exact local `Jv` API;
- state-safe derivative execution;
- derivative validation report.

### Exit criteria

- tangent columns are independent of evaluation order;
- `K_e v` and direct complex-step `Jv` agree within the chosen tolerance;
- smooth benchmark Newton iterations show the expected local convergence behavior.

---

### Phase 4 — Standalone PETSc CPU solver

### Work

- create mesh/DOF/constraint interfaces;
- implement element batching;
- implement SNES residual callback;
- implement vector COO residual assembly;
- implement matrix COO tangent assembly;
- implement load-step driver and state commit;
- add PETSc options and logging;
- implement outputs needed for comparison.

### Deliverables

- serial standalone nonlinear solver;
- assembled exact Jacobian mode;
- regression examples and command lines.

### Exit criteria

- standalone global solutions match Abaqus reference cases;
- no Python is required to execute a previously built solver;
- state is committed only after accepted convergence.

---

### Phase 5 — Matrix-free PETSc operator and preconditioning

### Work

- implement `MatShell` `MATOP_MULT` for complex-step `Jv`;
- maintain current linearization point in shell context;
- implement constraint-consistent `Jv`;
- separate `J` and `P` in SNES;
- implement exact assembled and lagged preconditioning options;
- compare `MatShell v` to assembled `K v` globally;
- add preconditioner-update policies.

### Deliverables

- matrix-free exact operator mode;
- configurable assembled/lagged preconditioner;
- performance and convergence comparison.

### Exit criteria

- matrix-free and assembled solutions agree;
- global `Jv` tests pass for random directions;
- a useful preconditioner is available for the target benchmarks.

---

### Phase 6 — MPI scalability

### Work

- distribute mesh and DOFs;
- implement ghost updates and local-to-global maps;
- use local COO preallocation where appropriate;
- verify off-process residual/matrix contributions;
- add MPI state ownership and output;
- profile communication and load balance.

### Deliverables

- multi-rank assembled and matrix-free modes;
- partition-independent regression tests;
- scaling baseline.

### Exit criteria

- solutions match serial results within reduction-order tolerance;
- state ownership has no duplicates or omissions;
- no rank-dependent element behavior is observed.

---

### Phase 7 — GPU bridge and real residual

### Work

- introduce C++ bridge with PETSc/Kokkos handles;
- generate Kokkos real kernel;
- allocate persistent device mesh, basis, parameters, and state;
- generate element residual COO values on device;
- assemble with PETSc GPU-compatible COO path;
- add CPU/Kokkos-host/Kokkos-device parity tests.

### Deliverables

- GPU real residual callback;
- device-resident data manager;
- transfer and kernel profiling.

### Exit criteria

- no full-vector host round trip occurs per residual evaluation;
- GPU residual matches CPU;
- SNES can execute residual evaluations on device.

---

### Phase 8 — GPU complex-step `Jv`

### Work

- generate `Kokkos::complex<double>` residual kernel;
- implement one complex directional evaluation per element;
- return real element `Jv` COO values;
- connect to `MatShell` through the bridge;
- test several complex-step sizes and compiler options;
- profile register use and occupancy;
- optimize geometry/control-cache reuse.

### Deliverables

- GPU exact matrix-free `Jv`;
- CPU/GPU `Jv` validation report;
- matrix-free GPU nonlinear solve.

### Exit criteria

- GPU `Jv` matches assembled CPU `K v`;
- global nonlinear solution and convergence agree with CPU;
- complex imaginary components survive the complete GPU toolchain reliably.

---

### Phase 9 — GPU preconditioner and optional full tangent

### Work

- generate full tangent COO values on GPU for low-order elements, or generate an approximate operator;
- implement preconditioner update frequency controls;
- evaluate AIJ Kokkos/CUDA/HIP matrix paths;
- compare exact, lagged, reduced-order, and block preconditioners;
- tune element/column launch layout.

### Deliverables

- practical all-device solver path;
- documented preconditioner recommendations by problem class;
- assembled GPU tangent option where beneficial.

### Exit criteria

- production benchmark solves do not depend on per-iteration host assembly;
- total solve performance improves over the CPU reference on suitable problem sizes.

---

### Phase 10 — Generalization and release engineering

### Work

- add additional elements/integrals/materials incrementally;
- stabilize public compiler and kernel ABI;
- add semantic versioning and deprecation policy;
- package compiler and runtime;
- document custom material/state extensions;
- add examples and troubleshooting guides;
- complete dependency/license review.

### Deliverables

- standalone release candidate;
- user and developer documentation;
- supported-feature matrix;
- reproducible containers or build recipes.

### Exit criteria

- a new model can be compiled, built, and run from documented steps;
- generated manifests permit reproducible builds;
- unsupported constructs fail clearly at compile time.

---

## 17. Suggested repository structure

```text
project/
  pyproject.toml
  CMakeLists.txt
  cmake/

  python/
    abaqus_ufl/
      frontend/
      analysis/
      form_ir/
      kernel_ir/
      transforms/
      backends/
        abaqus_fortran/
        fortran_cpu/
        kokkos/
        ir_dump/
      diagnostics/
      cli/

  runtime/
    fortran/
    cpp/
      petsc_device_bridge/
      kokkos_runtime/

  generated_examples/

  tests/
    compiler/
    kernel_real/
    kernel_complex/
    element/
    petsc_serial/
    petsc_mpi/
    gpu/
    abaqus_reference/

  examples/
    hyperelastic_block/
    cantilever/
    follower_pressure/
    path_dependent_demo/

  docs/
    architecture/
    kernel_abi/
    complex_step/
    gpu/
    developer_guide/
```

Consider renaming the common compiler package only after the architecture is separated. The existing `abaqus_ufl` name can remain as a compatibility package or backend entry point.

---

## 18. Initial issue/work-package backlog

### Compiler core

- Define supported UFL subset.
- Specify Form IR schema.
- Specify typed Kernel IR schema.
- Add provenance-aware diagnostics.
- Implement deterministic IR serialization.
- Extract Abaqus backend.
- Implement real Fortran backend.
- Implement complex Fortran backend.
- Implement complex-safety verifier.
- Add generated manifest and ABI version.

### Runtime

- Define element-kernel registry.
- Define committed/trial state manager.
- Implement element batches.
- Implement constraints.
- Implement PETSc residual COO path.
- Implement PETSc matrix COO path.
- Implement `MatShell` context and `Jv`.
- Implement load-step commit/rollback.
- Add PETSc logging and diagnostics.

### GPU

- Define C ABI for the device bridge.
- Create Kokkos host-backend prototype.
- Add PETSc device vector access in C++.
- Generate real Kokkos element kernel.
- Generate complex Kokkos element kernel.
- Implement device residual COO values.
- Implement device `Jv` COO values.
- Add device state and branch-cache layouts.
- Add GPU tangent/preconditioner experiment.

### Quality

- Build Abaqus parity dataset.
- Add analytic hyperelastic tangent reference.
- Add tangent/Jv random-direction checker.
- Add complex-step-size sweep.
- Add compiler cross-check matrix.
- Add CPU/GPU deterministic kernel tests.
- Add MPI partition tests.

---

## 19. Risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| Direct UFL-to-Fortran printing remains entangled with Abaqus | Every new backend duplicates logic | Introduce Form IR and Kernel IR before adding GPU code |
| Nonanalytic operation in a material/residual | Complex-step tangent is silently wrong | Compile-time complex-safety analysis, negative tests, independent `Jv` checks |
| State is mutated during a tangent column or line search | Order-dependent tangent and failed Newton convergence | Pure committed/trial protocol and final recompute-before-commit |
| Branch path changes under complex perturbation | Unstable or meaningless derivative | Real branch cache, active-branch replay, switching diagnostics |
| GPU toolchain removes tiny imaginary parts | Zero or inaccurate `Jv` | Step-size sweep, safe compiler flags, device derivative unit tests |
| Full tangent has excessive GPU register/workspace demand | Low occupancy or compilation failure | Matrix-free primary path, seed tiling, per-kernel fallback |
| PETSc device pointers are awkward from Fortran | Hidden device-host copies | Small C++ bridge using PETSc device/Kokkos APIs |
| Matrix-free operator lacks strong preconditioner | High KSP iteration count | Develop `P` alongside `J`; exact/lagged/low-order options |
| GPU residual assembly uses heavy atomics | Poor scaling | Element-local COO value arrays and PETSc duplicate summation |
| Different backends interpret contractions differently | CPU/GPU mismatch | Semantic contraction IR and backend conformance tests |
| Unsupported UFL silently changes meaning | Incorrect simulation | Explicit supported subset and hard errors with source provenance |
| Compiler optimizations alter complex semantics | Derivative regression | Optimization pass isolation and complex tests after each pass |
| Mixed compiler ABI instability | Build/runtime failures | Flat C ABI, no Fortran descriptors, versioned kernel manifest |
| Scope grows into a full multiphysics platform too early | Delayed usable release | Freeze version-0 problem class and phase gates |

---

## 20. Acceptance criteria for a credible first standalone release

A release should not be labeled production-ready until all of these are true:

- a supported UFL model compiles into a documented standalone build;
- the executable runs without Python;
- the same residual definition generates real, complex, tangent, and `Jv` kernels;
- PETSc uses real scalars;
- assembled `K v` and matrix-free complex-step `Jv` agree globally;
- tangent columns are independent of evaluation order;
- line-search residual calls do not commit history;
- smooth benchmark Newton convergence is consistent with a correct tangent;
- serial and MPI results agree;
- Abaqus reference cases agree within documented tolerances;
- unsupported/nonanalytic constructs produce clear diagnostics;
- generated code and ABI are versioned in a manifest;
- CPU reference tests pass under at least two Fortran compilers;
- enabled GPU backend passes residual and `Jv` parity tests without per-call host vector copies.

---

## 21. Immediate implementation sequence

The most useful next actions are:

1. **Freeze three reference problems:** one single-element hyperelastic case, one multi-element nonlinear case, and one stateful case if already available.
2. **Write the kernel contract** independently of Abaqus and PETSc.
3. **Extract one residual through a minimal typed IR** and regenerate the existing Abaqus code from it.
4. **Generate real and complex standalone Fortran** from the same IR.
5. **Build the one-element tangent/Jv verifier** and run a complex-step-size sweep.
6. **Implement committed/trial state discipline** before adding more solver callbacks.
7. **Implement PETSc real residual and exact assembled COO tangent** on CPU.
8. **Add a shell matrix whose action is direct local complex-step `Jv`.**
9. **Compare global shell `Jv` against assembled `K v` for random directions.**
10. **Introduce the C ABI and C++ bridge** while everything still runs on the host.
11. **Run the Kokkos host backend** using the same generated GPU source structure.
12. **Move the real residual and then `Jv` to one GPU**, keeping the CPU path as the oracle.
13. **Only then decide** whether the first production GPU backend should remain Kokkos, add CUDA Fortran, or incorporate libCEED for selected real operators.

This ordering minimizes simultaneous unknowns. It proves compiler semantics, derivative correctness, solver integration, and device execution one layer at a time.

---

## 22. Decisions to record early

Create short architecture decision records for:

- the public project/package name;
- initial supported element and integral types;
- whether DMPlex is the first mesh provider or an adapter added later;
- kernel state layout and ABI;
- default complex-step scaling policy;
- behavior at nonsmooth branches;
- primary GPU target vendors;
- Kokkos versus vendor-specific first production backend;
- exact versus approximate preconditioner strategy;
- output/checkpoint format;
- dependency and generated-code licensing.

These are decisions, not reasons to delay the core refactoring. Sensible defaults can be selected and revised behind stable interfaces.

---

## 23. Reference material

1. [UFL form language documentation](https://docs.fenicsproject.org/ufl/main/manual/form_language.html)
2. [FFCx compiler stages and intermediate representation](https://docs.fenicsproject.org/ffcx/main/_modules/ffcx/compiler.html)
3. [PETSc SNES matrix-free methods](https://petsc.org/release/manual/snes/)
4. [PETSc `MatCreateShell`](https://petsc.org/release/manualpages/Mat/MatCreateShell/)
5. [PETSc vector COO assembly](https://petsc.org/release/manualpages/Vec/VecSetValuesCOO/)
6. [PETSc matrix COO assembly](https://petsc.org/release/manualpages/Mat/MatSetValuesCOO/)
7. [PETSc `VecGetArrayAndMemType`](https://petsc.org/release/manualpages/Vec/VecGetArrayAndMemType/)
8. [PETSc `VecGetKokkosView`](https://petsc.org/release/manualpages/Vec/VecGetKokkosView/)
9. [Kokkos programming-model and backend overview](https://kokkos.org/kokkos-core-wiki/)
10. [Kokkos complex-number API](https://kokkos.org/kokkos-core-wiki/API/core/numerics/complex.html)
11. [NVIDIA HPC Compilers User Guide](https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-user-guide/index.html)
12. [libCEED operator model](https://libceed.org/en/latest/api/CeedOperator/)
13. [libCEED GPU development notes](https://libceed.org/en/latest/gpu/)
14. [libCEED floating-point scalar support](https://libceed.org/en/latest/precision/)
15. [Martins, Sturdza, and Alonso, “The Complex-Step Derivative Approximation”](https://dl.acm.org/doi/10.1145/838250.838251)

---

## 24. Final technical position

The recommended standalone design is not “rewrite the package in modern Fortran.” It is:

> retain Python/UFL as the model compiler, introduce a target-neutral typed IR, generate modern Fortran as the trusted CPU backend, keep PETSc real, generate complex element kernels for exact local tangents and matrix-free `Jv`, and add GPU execution through a backend that shares the same IR and kernel semantics.

That structure preserves what already works, makes the complex-step method a deliberate compiler feature, and prevents Abaqus, PETSc, Fortran, or one GPU vendor from becoming the architecture of the whole project.

---

## 25. Contact-mechanics subsystem and PPF-inspired barrier/CCD backend

### 25.1 Architectural boundary

Contact mechanics is a separate nonlinear global operator. It is not another fixed-connectivity UFL volume element and should not be embedded in the ordinary element translator.

```text
Python model and solver wrapper
        |
        +----------------------+-----------------------+
        |                      |                       |
        v                      v                       v
UFL bulk compiler       contact configuration     PETSc orchestration
        |                      |                       |
        v                      v                       v
generated Fortran       native contact manager    residual/Jacobian solve
bulk kernels            search / law / state
        |                      |
        +----------+-----------+
                   v
        global residual and Jacobian/Jv
```

The global equations are

\[
\mathbf R(\mathbf u)
=
\mathbf R_{\mathrm{bulk}}(\mathbf u)
+
\mathbf R_{\mathrm{contact}}(\mathbf u)
-
\mathbf F_{\mathrm{external}}(\mathbf u),
\]

and

\[
\mathbf J(\mathbf u)
=
\mathbf J_{\mathrm{bulk}}(\mathbf u)
+
\mathbf J_{\mathrm{contact}}(\mathbf u)
-
\mathbf J_{\mathrm{external}}(\mathbf u).
\]

The bulk may remain total Lagrangian and reference based. Contact search, gap, normals, closest projections, and sliding geometry are evaluated in the current configuration. There is no requirement to convert the bulk formulation to a Cauchy-stress updated-Lagrangian formulation.

### 25.2 Preserve the F2PY architecture

Keep F2PY as the supported CPU binding. The Python layer should configure contact and make coarse-grained native calls; it should not iterate over facets, contact points, or candidate pairs.

A representative Python API is:

```python
contact = ContactProblem(
    secondary=surface_a,
    primary=surface_b,
    method="cubic_barrier_ccd",
    tangent="psd",
    friction=None,
)

contact.initialize(mesh)
```

The Newton callback then performs operations at operator granularity:

```python
bulk.assemble_residual(u, r)
contact.update_search(u)
contact.assemble_residual(u, r)
```

or, in matrix-free mode:

```python
bulk.apply_jacobian(u, v, y)
contact.apply_jacobian(u, v, y)
```

Use flat arrays and opaque integer handles across F2PY. Do not expose complicated Fortran derived types to Python.

Suggested native API:

```fortran
subroutine contact_create(surface_nodes, triangles, edges,       &
                          parameters, contact_id)

subroutine contact_begin_step(contact_id)

subroutine contact_update_search(contact_id, current_coordinates)

subroutine contact_assemble_residual(contact_id, current_coordinates, &
                                     residual, diagnostics)

subroutine contact_assemble_coo(contact_id, current_coordinates,      &
                                rows, columns, values, nnz_used)

subroutine contact_apply_jacobian(contact_id, current_coordinates,    &
                                  direction, result)

subroutine contact_max_step(contact_id, current_coordinates,          &
                            direction, alpha_max)

subroutine contact_commit(contact_id)
subroutine contact_rollback(contact_id)
subroutine contact_destroy(contact_id)
```

For the first implementation, a module-level singleton is acceptable. Move to a handle registry before supporting several simultaneous contact systems.

### 25.3 Contact subsystem modules

A practical modern-Fortran layout is:

```text
contact/
  contact_kinds_m.F90
  contact_types_m.F90
  contact_surface_m.F90
  contact_feature_ids_m.F90
  contact_bvh_m.F90
  contact_broad_phase_m.F90
  contact_projection_m.F90
  contact_distance_m.F90
  contact_barrier_m.F90
  contact_penalty_m.F90
  contact_augmented_lagrangian_m.F90
  contact_friction_m.F90
  contact_ccd_m.F90
  contact_state_m.F90
  contact_assembly_m.F90
  contact_matrix_free_m.F90
  contact_mpi_m.F90
  contact_f2py_api_m.F90
```

The first production implementation should support one surface representation and one enforcement law well before generalizing the module tree.

### 25.4 Why contact is not an ordinary element

Bulk element connectivity is fixed:

```text
cell e -> fixed node and DOF list
```

Contact connectivity changes with the current geometry:

```text
secondary feature 17 -> primary feature 54
secondary feature 17 -> primary feature 61
secondary feature 17 -> no active contact
```

The contact manager therefore builds a temporary interaction graph. A contact record should contain enough data to evaluate and linearize one feature interaction:

```fortran
type :: contact_record_t
  integer :: record_id
  integer :: feature_type
  integer :: secondary_feature
  integer :: primary_feature
  integer :: owner_rank
  integer :: state_index
  real(dp) :: primary_parametric_coordinates(2)
  real(dp) :: distance
  real(dp) :: gap
  real(dp) :: normal(3)
  real(dp) :: coefficients(max_contact_nodes)
  real(dp) :: quadrature_weight
end type
```

Search records are trial data and may change during a Newton iteration. Committed frictional or multiplier history must have an independent lifecycle.

### 25.5 Initial implementation sequence

Implement contact in increasing order of geometric and nonlinear difficulty.

#### Stage 1 — Rigid analytical obstacle

Support plane, sphere, and cylinder obstacles with frictionless penalty or barrier enforcement.

For an analytical signed distance \(g_n(\mathbf x)\), use

\[
\mathbf n
=
\frac{\nabla g_n}{\|\nabla g_n\|}.
\]

This stage validates:

- normal and gap conventions;
- equal and opposite force signs where applicable;
- residual and tangent/Jv;
- active-set or barrier activation logic;
- line search and step limiting;
- load-step commit/rollback.

It avoids deformable surface search and remote facet exchange.

#### Stage 2 — Deformable point-to-surface contact

Add linear triangular primary facets and secondary quadrature points or vertices. For one projected secondary point,

\[
g_n
=
\left(\mathbf x_s-\mathbf x_m(\boldsymbol\xi_m)\right)
\cdot\mathbf n_m.
\]

Initially support interior triangle projections and explicit edge/vertex fallbacks. Add robust degeneracy detection before self-contact.

#### Stage 3 — Full discrete feature contact

Add the feature pairs needed for large sliding and self-contact:

```text
point-triangle
edge-edge
point-edge
point-point
```

Use stable global feature IDs and deterministic pair ownership.

#### Stage 4 — Frictionless barrier with CCD

Add the PPF-inspired cubic barrier and continuous collision detection described below.

#### Stage 5 — Friction

Provide a simple regularized incremental model first, followed by a history-based engineering Coulomb formulation with stick/slip state and return mapping.

#### Stage 6 — Advanced discretizations

Treat mortar, dual multipliers, curved-feature CCD, shell thickness, and high-order contact as separate later backends.

### 25.6 PPF-inspired feature search

The `ppf-contact-solver` repository is a valuable algorithm source for large-sliding and self-contact. Adopt its general decomposition rather than its complete solver stack:

```text
surface points, edges, and triangles
       |
       v
AABB/BVH broad phase
       |
       v
point-triangle and edge-edge narrow phase
       |
       v
barrier force and contact Hessian/Jv
       |
       v
continuous collision detection step bound
```

The project should maintain a clear distinction between:

- a broad-phase candidate set;
- a narrow-phase proximity set;
- an active barrier/contact set;
- committed contact history.

Candidate buffers should be reusable and grow geometrically rather than allocate on every Newton call.

### 25.7 Cubic barrier law

For a positive clearance \(d\) and activation distance \(\hat d\), one useful cubic barrier is

\[
\phi(d)
=
\begin{cases}
\dfrac{2}{3\hat d}(\hat d-d)^3,
&0<d<\hat d,\\[1ex]
0,&d\geq\hat d.
\end{cases}
\]

Its derivatives in the active interval are

\[
\phi'(d)
=-\frac{2}{\hat d}(\hat d-d)^2,
\]

and

\[
\phi''(d)
=4\left(1-\frac{d}{\hat d}\right).
\]

The implementation should separate:

```text
barrier shape phi(d)
contact stiffness scale k_c
feature kinematics d(x)
assembly coefficients
```

This allows the same search and kinematics to support fixed, elasticity-scaled, or dynamic stiffness policies.

### 25.8 Contact stiffness policy

Do not copy a dynamics-specific stiffness formula unchanged into every application.

Provide at least:

```python
stiffness="fixed"
stiffness="elasticity_scaled"
stiffness="effective_dynamic"
```

#### Fixed stiffness

Use for initial verification and parameter studies. Report the nondimensional ratio between contact stiffness and representative bulk stiffness.

#### Elasticity-scaled quasi-static stiffness

Use a positive stiffness proxy:

\[
k_c
=
\widehat{\mathbf w}^{T}
\mathbf K_{\mathrm{proxy}}
\widehat{\mathbf w},
\qquad
\mathbf K_{\mathrm{proxy}}\succeq0.
\]

Do not directly use an indefinite exact bulk tangent as the barrier scale. Construct `K_proxy` from a material-only tangent, positive block projection, or another deliberately positive approximation.

#### Effective dynamic stiffness

For implicit dynamics, derive the scale from the positive part of the effective incremental operator, for example a combination of stiffness, mass, and damping consistent with the chosen time integrator.

Apply configurable lower and upper bounds to prevent vanishing or extreme barrier scales.

### 25.9 Continuous collision detection

A barrier method assumes a valid nonintersecting initial state and should prevent a Newton step from crossing the contact surface.

Given the current coordinates \(\mathbf x\) and search direction \(\Delta\mathbf x\), compute a maximum admissible step

\[
\alpha_{\mathrm{CCD}}
\]

over all relevant point-triangle and edge-edge trajectories. Apply

\[
\alpha
\leq
\eta\,\alpha_{\mathrm{CCD}},
\qquad 0<\eta<1,
\]

before or within PETSc line search globalization.

The contact interface should expose `contact_max_step`. In an MPI solve, each rank computes a local bound and the global bound is an MPI minimum reduction.

Initial penetration is a separate preprocessing problem. Provide:

```text
intersection detection
clear diagnostic output
optional geometry correction for selected cases
hard failure when a valid correction is unavailable
```

Do not hide penetrations by applying an arbitrary large penalty force.

### 25.10 Tangent modes

The PPF-style contact Hessian is intentionally positive semidefinite and is best regarded as a projected or Gauss-Newton approximation.

For a radial potential \(\phi(r)\), the full frozen-coefficient Hessian contains

\[
\mathbf H_{\mathrm{exact}}
=
\phi''(r)\,\mathbf n\otimes\mathbf n
+
\frac{\phi'(r)}{r}
\left(\mathbf I-\mathbf n\otimes\mathbf n\right).
\]

A PSD approximation retains the normal rank-one part:

\[
\mathbf H_{\mathrm{psd}}
=
\phi''(r)\,\mathbf n\otimes\mathbf n.
\]

Finite-sliding consistency also requires derivatives of closest-point parameters, normals, and interpolation coefficients. These terms may make the exact tangent indefinite or nonsymmetric.

Support two explicit modes:

```text
tangent="psd"
    robust positive-semidefinite approximation
    suitable for descent-oriented solves and CG when the complete operator is SPD

tangent="consistent"
    includes projection, normal, and geometric derivatives
    better local Newton behavior
    may require MINRES or GMRES depending on symmetry
```

The solver manifest must record which mode was used. Do not label the PSD mode an exact Newton tangent.

### 25.11 Complex-step policy for contact

Retain complex-step differentiation inside smooth fixed-connectivity bulk elements:

\[
\mathbf J_{\mathrm{bulk}}\mathbf v
=
\frac{\operatorname{Im}
\mathbf R_{\mathrm{bulk}}(\mathbf u+i h\mathbf v)}{h}.
\]

Do not blindly complex-step the complete contact algorithm because it includes nonsmooth or discrete operations:

```text
candidate-pair creation
closest-feature changes
open/contact activation
stick/slip transitions
max, min, clipping, and absolute value
CCD root selection
```

The global action is

\[
\mathbf J\mathbf v
=
\mathbf J_{\mathrm{bulk}}^{\mathrm{CS}}\mathbf v
+
\mathbf J_{\mathrm{contact}}^{\mathrm{alg}}\mathbf v.
\]

Complex-step may still be used to verify a smooth contact subkernel when the real iterate determines and freezes:

- the candidate pair;
- the active branch;
- the closest-feature classification;
- stick or slip state.

At switching points, use an active-set, semismooth, or generalized derivative and report the transition.

### 25.12 Friction strategy

Provide two friction models with different goals.

#### Regularized incremental friction

Use a smooth or piecewise-smooth tangential model as an early robust implementation. It is useful for testing GPU execution and large self-contact, but its approximate tangent and lack of conventional stick/slip history must be documented.

#### Engineering Coulomb friction

Add a history-based model with:

```text
committed tangential slip
trial tangential traction
stick/slip active set
Coulomb return mapping
algorithmic tangent
history transfer when the primary feature changes
```

The state update should resemble a local constitutive return mapping. Commit only after the global increment converges.

### 25.13 Contact state lifecycle

Use the same transactional discipline as path-dependent bulk materials:

```text
committed state at accepted increment
        |
        v
trial search and trial constitutive contact state
        |
        +-- rejected line search/Newton step -> discard
        |
        +-- accepted increment -> commit
```

Possible state fields include:

```text
active/open state
stick/slip state
normal multiplier or barrier diagnostics
tangential traction
accumulated tangential slip
previous primary feature
previous closest-point parameters
previous contact normal
contact record lineage
```

Search structures may be refit during Newton iterations, but permanent frictional state must not depend on the order of residual, Jacobian, or line-search evaluations.

### 25.14 PETSc integration

For displacement-only penalty, barrier, or augmented-Lagrangian contact, ordinary SNES residual and Jacobian interfaces are sufficient.

#### Assembled mode

Keep fixed bulk and changing contact sparsity separate:

```text
K_bulk:
    fixed pattern

K_contact:
    dynamic COO records generated from active/candidate pairs
```

Either assemble into separate PETSc matrices and combine them, or rebuild only the contact COO pattern when required. Avoid preallocating all possible surface pairs.

#### Matrix-free mode

Apply each active contact record directly:

\[
\mathbf y_c
=
\mathbf J_c\mathbf v
\]

without constructing a global contact matrix. This maps naturally to GPU execution.

Recommended split:

```text
operator J:
    matrix-free complex-step bulk Jv
    + matrix-free contact Jv

preconditioner P:
    assembled bulk proxy
    + contact block diagonal or selected local couplings
```

#### Line-search safeguard

Connect `contact_max_step` to the PETSc search-direction preprocessing or line-search layer so that the direction is CCD-limited before ordinary residual/energy globalization.

#### Multiplier formulations

If global Lagrange multiplier DOFs are added later, formulate the complementarity system explicitly. Variable bounds alone are not a complete general finite-sliding contact formulation.

### 25.15 Dynamic contact sparsity

The contact graph can introduce couplings absent from the volume-mesh adjacency graph. Maintain:

```text
fixed bulk adjacency
changing contact adjacency
```

For assembled CPU or GPU matrices, produce contact COO triplets:

```text
contact_rows
contact_columns
contact_values
```

Use deterministic ordering for reproducibility. Rebuild the row/column pattern only when the candidate graph changes materially; update values every required Jacobian evaluation.

For matrix-free operation, keep compact contact records sorted or bucketed by feature type to improve SIMD/GPU execution.

### 25.16 Distributed contact

Distributed deformable contact requires a spatial communication layer separate from the ordinary bulk ghost halo.

A staged algorithm is:

```text
1. Extract local contact facets and feature IDs.
2. Build/refit a local BVH.
3. Exchange coarse surface bounding regions.
4. Determine potentially interacting rank pairs.
5. Exchange only required remote feature geometry.
6. Build local candidate and active records.
7. Assign each record to exactly one deterministic owner.
8. Assemble local and remote nodal contributions.
9. Reduce contributions to DOF owners.
10. Apply a global minimum reduction to the CCD step.
```

Start with rigid obstacles, then two deformable bodies, and only then self-contact.

Bulk cell count is not a sufficient load-balancing weight when contact is localized. Allow contact records to be scheduled independently or include recent contact work in repartitioning weights.

### 25.17 GPU path

The contact backend should be designed around compact structure-of-arrays buffers:

```text
feature IDs and types
node indices
closest-point coefficients
gaps and normals
barrier parameters
state indices
output scatter indices
```

GPU work is naturally partitioned by candidate or active contact record.

Initial GPU sequence:

```text
1. CPU search + GPU contact residual/Jv
2. GPU BVH refit
3. GPU broad and narrow phase
4. GPU CCD
5. distributed GPU contact
```

Keep the Python/F2PY API for CPU. Use the same C/C++ PETSc device bridge planned for bulk GPU execution rather than moving device-resident vectors through NumPy.

### 25.18 Contact geometry and high-order surfaces

The initial PPF-style representation is piecewise linear. For high-order bulk elements, separate search geometry from final contact evaluation.

Support these modes:

```text
linear_contact_geometry
    triangulated search and force evaluation

proxy_search_curved_evaluation
    triangulated search proxy
    final projection, gap, and normal on the curved FE face

curved_contact_geometry
    curved projection and eventually curved CCD
```

For rigid analytical obstacles, use the analytical geometry directly.

A geometry-resolution criterion must compare surface approximation error with:

- barrier activation distance;
- allowed penetration/clearance tolerance;
- required normal accuracy;
- expected contact patch scale.

### 25.19 Adopt, adapt, and do not copy directly

| PPF-related component | Decision | Project treatment |
|---|---|---|
| Point-triangle and edge-edge feature pairs | Adopt | Core large-sliding/self-contact representation |
| AABB/BVH broad phase | Adopt | Implement behind a search backend interface |
| Cubic barrier activation | Adopt | One contact enforcement option |
| Additive CCD step limit | Adopt | Mandatory safeguard for barrier mode |
| Dynamic contact sparsity | Adopt | Separate contact COO or matrix-free record graph |
| PSD rank-one contact Hessian | Adapt | Expose explicitly as `tangent="psd"` |
| Exact/consistent geometric tangent | Add | Engineering mode for Newton studies and selected applications |
| Elasticity-inclusive stiffness scale | Adapt | Use a positive proxy and distinguish static/dynamic policies |
| Custom GPU CG solver | Do not copy | Retain PETSc KSP/SNES and project-specific preconditioning |
| Single precision assumptions | Do not copy | Use double precision as the engineering default |
| Simplified friction | Adapt | Early regularized option, followed by stateful Coulomb friction |
| Linear contact geometry only | Adapt | Add geometry tolerance and curved-evaluation roadmap |
| Initial-penetration behavior | Strengthen | Explicit detection and correction/failure policy |

Track the reference algorithm version or repository commit in the generated solver manifest because the upstream main branch may evolve.

### 25.20 Validation suite

Minimum contact tests include:

```text
single node/point against rigid plane
sphere against plane with analytical symmetry
two-block compression
sliding block with frictionless contact
stick/slip transition under Coulomb friction
point-triangle and edge-edge distance derivatives
CCD time-of-impact regression
self-contact of a folded strip or shell-like surface
partition-invariance test
matrix-free Jv versus assembled contact tangent
energy/work consistency over load steps
```

For smooth fixed active sets, compare the algorithmic contact tangent against a centered finite-difference or carefully frozen-branch complex-step reference. At transitions, test directional behavior and nonlinear convergence rather than assuming a classical derivative exists.

Record:

```text
minimum clearance
number of candidates and active records
CCD-limited steps
barrier energy and contact work
normal and tangential reaction balance
stick/slip counts
Newton and KSP iterations
search, residual, and Jacobian timings
```

### 25.21 Contact implementation work packages

#### Contact-1 — Native interface and rigid obstacle

- Define surface, contact record, state, and handle APIs.
- Implement plane/sphere/cylinder geometry.
- Add frictionless penalty and fixed-stiffness barrier modes.
- Add residual and tangent/Jv tests.

**Exit criterion:** rigid-contact results match analytical or independent references and survive line-search rollback.

#### Contact-2 — Linear deformable surfaces

- Extract boundary points, edges, and triangles.
- Add BVH broad phase and robust narrow-phase distances.
- Add point-triangle and edge-edge records.
- Add equal-and-opposite assembly.

**Exit criterion:** two-body contact is invariant under surface ordering and mesh partition in serial.

#### Contact-3 — Cubic barrier and CCD

- Add activation-distance policy.
- Add ACCD/CCD maximum-step computation.
- Integrate the step bound with PETSc globalization.
- Detect initial intersections.

**Exit criterion:** benchmark trajectories remain intersection free within tolerance.

#### Contact-4 — Tangent and preconditioning modes

- Implement PSD contact Jv and COO tangent.
- Implement a consistent tangent for selected feature cases.
- Add contact block-diagonal preconditioning.
- Verify global bulk-plus-contact `Jv`.

**Exit criterion:** documented convergence and robustness comparisons exist for PSD and consistent modes.

#### Contact-5 — Friction

- Implement regularized incremental friction.
- Add committed/trial tangential state.
- Implement Coulomb return mapping and stick/slip tangent.

**Exit criterion:** stick/slip and cyclic sliding tests reproduce reference behavior without state-order dependence.

#### Contact-6 — MPI and self-contact

- Add rank-level bounding exchange and remote feature ghosts.
- Add deterministic pair ownership and remote residual reduction.
- Add global CCD minimum reduction.
- Add contact-work diagnostics and balancing.

**Exit criterion:** results are invariant under rank count and no pair is lost or double counted.

#### Contact-7 — GPU

- Port contact-record residual and Jv.
- Port BVH refit, search, and CCD in stages.
- Keep data device resident through Newton/Krylov loops.

**Exit criterion:** GPU results match CPU and show no hot-path host staging.

### 25.22 Contact risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Contact is embedded in the bulk UFL element abstraction | Changing connectivity becomes unmanageable | Keep a separate global contact operator and record graph |
| Complex-step is applied through active-set/search logic | Invalid or zero derivatives | Use algorithmic/generalized contact tangents and frozen-branch verification only |
| Raw indefinite bulk tangent sets barrier stiffness | Negative or unstable contact scale | Use a separate positive stiffness proxy |
| CCD is omitted | Newton step can cross surfaces despite a barrier | Mandatory maximum-step calculation for barrier mode |
| Initial intersections are ignored | Barrier assumptions are violated | Explicit intersection check and correction/failure policy |
| Contact pattern is forced into fixed bulk sparsity | Excessive preallocation or missing couplings | Separate dynamic COO or matrix-free contact graph |
| Friction state is updated during trial evaluations | Path and call-order dependence | Strict committed/trial lifecycle |
| Piecewise-linear contact under-resolves curved geometry | Incorrect gap, normal, and pressure | Geometry tolerance and curved-evaluation path |
| Pair ownership is nondeterministic in MPI | Double counting or lost reactions | Stable global feature IDs and deterministic ownership |
| PSD Hessian is described as exact | Misleading convergence expectations | Explicit tangent mode and solver manifest |

### 25.23 Recommended contact baseline

The first serious contact baseline is:

> Native modern-Fortran contact manager called through coarse-grained F2PY; linear surface points/edges/triangles; BVH broad phase; point-triangle and edge-edge narrow phase; cubic barrier; CCD-limited Newton steps; PSD contact Jv; PETSc matrix-free operator; fixed or elasticity-scaled positive contact stiffness; double precision; no friction in the first self-contact milestone.

Retain penalty and augmented-Lagrangian rigid-contact modes as simpler verification backends. Add consistent geometric tangents, stateful Coulomb friction, curved contact evaluation, MPI self-contact, and GPU search in later controlled phases.

---

## 26. Geometry-aware distributed mesh generation and refinement

### 26.1 Architectural decision

The production mesh path should not require reading a fully refined mesh containing approximately ten million degrees of freedom. The preferred workflow is:

```text
authoritative geometry + coarse topological mesh
                |
                v
     semantic classification and labels
                |
                v
      bootstrap refinement if required
                |
                v
       coarse-mesh MPI distribution
                |
                v
       parallel nested refinement
                |
                v
 geometry re-embedding and quality correction
                |
                v
 DOF ownership, ghosts, hierarchy, and kernel views
                |
                v
      bulk and contact operator setup
```

The crucial qualification is that the coarse mesh is **not** the authoritative description of a curved boundary. It supplies topology, region connectivity, and an initial parameterization. Curvature and sharp features must come from a separate geometry source.

The input contract is therefore:

```text
coarse mesh topology
+ material and boundary labels
+ geometric-entity classification
+ authoritative geometry representation
```

not merely:

```text
coarse nodal coordinates
```

Refining a faceted but geometrically incorrect boundary only creates smaller facets on the same incorrect surface. Projection can repair geometric discretization error when the topology and entity classification are correct; it cannot reliably recover a missing hole, a merged face, an unresolved thin region, or an incorrectly classified sharp edge.

### 26.2 Scale target

Approximately ten million displacement DOFs is a realistic engineering target for the proposed PETSc-based runtime, provided that:

- the fine mesh is distributed before it becomes large;
- each MPI rank stores only its owned and ghost portion;
- the solver uses matrix-free `Jv` or carefully controlled assembled storage;
- the preconditioner is designed for distributed memory;
- mesh, state, and contact data are not duplicated unnecessarily across PETSc, NumPy, and Fortran;
- input, checkpoint, and output paths do not gather the fine mesh onto one rank.

For orientation, a low-order three-dimensional displacement mesh with roughly 3.3 million vertices has roughly 10 million displacement DOFs. Two or three nested uniform refinement levels can generate that scale from a relatively modest coarse mesh, depending on element type and initial resolution.

The target should be validated in stages rather than treated as the first integration test:

```text
100k DOFs   -> correctness and serial/parallel equivalence
1M DOFs     -> ownership, ghosting, memory, and solver profiling
10M DOFs    -> production-scale bulk benchmark
10M + rigid contact
10M + distributed deformable contact
```

### 26.3 Hide the general mesh framework

DMPlex is powerful but should not be exposed throughout the application. The runtime should define a narrow, stable mesh contract and place DMPlex, DMForest, or a restricted native mesh implementation behind it.

User-facing code should resemble:

```python
mesh = MeshPlan(
    topology="coarse_mesh.msh",
    geometry="part.step",
    refinement={"mode": "uniform", "levels": 2},
    partitioner="automatic",
)

mesh.build()
view = mesh.kernel_view()
```

The bulk and contact kernels should consume `view`, not DMPlex objects. A kernel view contains compact rank-local data such as:

```text
owned-cell connectivity
reference coordinates
owned and ghost DOF maps
cell and facet labels
constraint/gather operators
quadrature-state offsets
contact-surface feature lists
owner-to-ghost communication plans
```

The minimum backend contract should provide operations equivalent to:

```text
read_coarse_topology()
attach_geometry()
refine()
distribute()
rebalance()
create_fields()
update_ghosts()
reduce_residual()
extract_kernel_view()
write_checkpoint()
```

No generated constitutive or element kernel should know about cones, supports, closures, sections, stars, or other general-topology implementation details.

### 26.4 Two mesh backends

Support two complementary backends rather than forcing every application through the most general one.

#### Restricted hierarchical backend

Target initially:

- one or two low-order element families, such as Hex8 and Tet4;
- uniform startup refinement;
- optional predefined-region refinement before the solve;
- fixed topology during an increment;
- simple ownership and halo construction;
- nested levels suitable for geometric multigrid.

For structured or tree-derived quad/hex applications, a Morton-ordered forest or equivalent hierarchical representation may be substantially simpler than exposing a fully general unstructured mesh API.

#### General PETSc backend

Use a DMPlex/DMForest adapter for:

- imported unstructured meshes;
- mixed topology;
- general labels and field layouts;
- arbitrary partitioners;
- advanced nonconforming refinement;
- interoperability with PETSc mesh and multilevel facilities.

Both backends must produce the same kernel-view contract and pass the same ownership, ghosting, assembly, and partition-invariance tests.

### 26.5 Geometry backend

Add a geometry subsystem independent of the mesh backend. Its interface should support:

```python
class GeometryBackend:
    def evaluate(self, entity_id, parametric_coordinates): ...
    def project(self, entity_id, x, initial_guess=None): ...
    def derivatives(self, entity_id, parametric_coordinates): ...
    def normal(self, entity_id, parametric_coordinates): ...
    def contains(self, entity_id, parametric_coordinates): ...
```

Initial geometry implementations may include:

```text
AnalyticalGeometry
CADGeometry
NURBSGeometry
SignedDistanceGeometry
DiscreteSurfaceGeometry
```

Every coarse boundary entity must carry:

```text
geometric dimension: vertex, curve, or surface
geometric entity ID
parametric coordinate when available
feature classification: smooth face, sharp edge, or corner
```

This classification must survive distribution and refinement.

For a new node on a curve, use the curve parameterization:

\[
\mathbf X_{\mathrm{new}}=\mathbf C(t_{\mathrm{new}}).
\]

For a new node on a surface, use the surface parameterization:

\[
\mathbf X_{\mathrm{new}}=\mathbf S(u_{\mathrm{new}},v_{\mathrm{new}}).
\]

A Cartesian midpoint is only an initial guess; it is not generally the final curved-boundary coordinate.

Projection rules must preserve feature classification:

```text
corner node -> remains on its geometry vertex
edge node   -> moves only along its classified geometry curve
face node   -> moves on its classified geometry face
```

This prevents a closest-point routine from rounding a sharp corner or moving an edge node onto an adjacent face.

### 26.6 Refinement and distribution sequence

Use the following default sequence.

#### Case A: the coarse mesh has enough cells for the MPI job

```text
1. Read coarse topology and labels.
2. Attach geometry entity IDs and parametric coordinates.
3. Distribute the coarse mesh.
4. Refine independently but consistently on all ranks.
5. Re-embed new boundary nodes in the geometry.
6. Rebalance if the refined workload is uneven.
7. Construct fields, DOF ownership, and ghost maps.
8. Extract compact kernel arrays.
```

#### Case B: the starting mesh is too small to occupy the MPI job

```text
1. Read the very coarse mesh.
2. Perform one or more bootstrap refinements.
3. Distribute once there are sufficiently many cells.
4. Continue refinement in parallel.
5. Re-embed, rebalance, and construct fields.
```

Do not refine the complete model to the target scale on rank zero and distribute only afterward.

The refinement policy must be deterministic under changes in MPI rank count. Global entity IDs, parent-child relations, and label propagation should not depend on local traversal order.

### 26.7 Curved finite-element geometry

The compiler and runtime must distinguish three independent choices:

```text
solution basis and polynomial order
geometry basis and polynomial order
quadrature rule and order
```

For a curved reference element,

\[
\mathbf X_h(\boldsymbol\xi)
=\sum_a N_a^g(\boldsymbol\xi)\mathbf X_a,
\]

and

\[
\mathbf F
=
\frac{\partial\mathbf x}{\partial\boldsymbol\xi}
\left(
\frac{\partial\mathbf X_h}{\partial\boldsymbol\xi}
\right)^{-1}.
\]

The geometry basis may be isoparametric, subparametric, or superparametric. It must not be implicitly assumed to be affine merely because the displacement element is known.

For linear volume elements, repeated refinement plus boundary projection gives a converging piecewise-linear approximation when a valid geometry source exists. For higher-order elements, edge, face, and interior geometry nodes must be generated and curved consistently.

Complex-step differentiation remains local to the solution variables. Reference geometry coordinates, geometry classification, and real projection state are not complex-perturbed during an element tangent evaluation.

### 26.8 Mesh-quality correction

Boundary projection can distort or invert adjacent cells. After new boundary points are re-embedded, adjust interior geometry nodes using an available method such as:

- transfinite interpolation;
- Laplacian smoothing;
- linear-elastic mesh motion;
- nonlinear Jacobian or shape-quality optimization.

Every refined level intended for analysis must pass:

\[
\det\left(\frac{\partial\mathbf X_h}{\partial\boldsymbol\xi}\right)>0
\]

at quadrature points and at additional geometry-check points.

Record at least:

```text
minimum reference Jacobian determinant
maximum boundary-position error
maximum boundary-normal error
minimum scaled cell quality
number of repaired and rejected cells
```

A refinement level that fails geometry or Jacobian checks must not proceed to solver setup.

### 26.9 Preserve the hierarchy for preconditioning

Nested refinement creates a natural hierarchy:

```text
level 0: authoritative coarse topology
level 1: refined distributed mesh
level 2: fine analysis mesh
```

Retain parent-child relations and transfer operators when they are used for geometric multigrid. A recommended first scalable operator split is:

```text
fine nonlinear operator:
    real bulk residual + contact residual

fine Jacobian action:
    matrix-free complex-step bulk Jv
    + algorithmic contact Jv

multigrid/preconditioner hierarchy:
    cheaper real bulk operators
    + optional simplified contact contribution on selected levels
```

Contact can initially exist only on the fine level while the coarse hierarchy represents bulk elasticity. More sophisticated coarse contact models can be added after the bulk multigrid path is stable.

If hierarchy data are not used, destroy unnecessary intermediate mesh objects after constructing the final partition to avoid retaining multiple complete mesh levels in memory.

### 26.10 Contact-surface construction on a refined curved mesh

Construct deformable contact features only after final startup refinement and geometry correction:

```text
volume refinement
-> geometry re-embedding
-> quality validation
-> boundary-facet extraction
-> stable contact feature IDs
-> BVH/search structure
-> contact history allocation
```

For rigid planes, spheres, cylinders, or implicit obstacles, evaluate gap and normal directly from the analytical or signed-distance geometry when practical. Avoid meshing a rigid obstacle merely to support search.

The first PPF-inspired backend uses point-triangle and edge-edge primitives and is therefore piecewise linear. Provide three geometry modes:

```text
linear_contact_geometry:
    search and final gap evaluation on triangulated facets

proxy_search_curved_evaluation:
    linear triangles for broad/narrow search candidates
    curved FE face for final projection, gap, and normal

curved_contact_geometry:
    native curved projection and eventually curved CCD
```

The first release may use `linear_contact_geometry`, but it must enforce a geometry-resolution criterion. The contact-surface approximation error should be substantially smaller than both the barrier activation distance and the required clearance accuracy. Otherwise the contact response is governed by faceting error.

### 26.11 Hanging nodes and constraints

Uniform conforming startup refinement should be implemented first because it avoids most hanging-node complications.

When local tree refinement is introduced, dependent DOFs satisfy a constraint of the form

\[
\mathbf u_d=\mathbf C\mathbf u_i.
\]

The element operations must then use:

\[
\mathbf R_i=\mathbf C^T\mathbf R_e,
\qquad
\mathbf K_i=\mathbf C^T\mathbf K_e\mathbf C,
\]

or, for matrix-free application,

\[
\mathbf v_e=\mathbf C\mathbf v_i,
\qquad
\mathbf y_i=\mathbf C^T\mathbf J_e\mathbf v_e.
\]

The compact kernel view must therefore be able to represent a gather/scatter constraint operator, not only a flat global-DOF list.

### 26.12 Runtime adaptation is a later phase

Separate three capabilities clearly.

#### Startup uniform refinement

First priority:

- performed before constitutive history or contact history exists;
- nested and reproducible;
- easy to validate;
- naturally supports multigrid.

#### Startup region-based refinement

Second priority:

- refine known contact zones, notches, thin regions, or anticipated localization regions;
- still performed before the nonlinear solve;
- may introduce hanging-node constraints.

#### Refinement during simulation

Later priority because it requires reliable transfer of:

```text
displacement, velocity, and acceleration
pressure and multiplier fields
quadrature-point constitutive history
damage and plastic strain
contact multipliers and accumulated tangential slip
active contact records and feature ownership
time-integration and load-step state
```

For path-dependent mechanics, naive interpolation of history may alter dissipation and material response. Runtime adaptation should not be included in the first ten-million-DOF milestone.

### 26.13 Memory ownership and F2PY boundary

The intended ownership model is:

```text
mesh backend/PETSc:
    topology, numbering, hierarchy, ownership, and communication

NumPy/F2PY boundary:
    compact rank-local contiguous buffers and opaque handles

Fortran kernels:
    operate on passed buffers; do not duplicate the full mesh
```

On each MPI rank, make only coarse-grained F2PY calls over complete element batches or contact-record batches. Keep arrays in the expected precision, integer kind, and Fortran-contiguous layout before entering F2PY.

Avoid simultaneously retaining:

```text
a complete PETSc mesh representation
+ a second complete Python mesh
+ a third complete Fortran-owned mesh
+ unused copies of all refinement levels
```

The code should expose memory accounting by category so a 10-million-DOF run reports at least:

```text
topology and connectivity
coordinates and geometry data
solution and work vectors
quadrature state
bulk operator/preconditioner storage
contact search and history storage
mesh hierarchy storage
```

### 26.14 Distributed contact implications

Bulk communication follows mesh adjacency; deformable contact communication follows spatial proximity and may connect remote partitions. The distributed contact layer must therefore be designed separately from the bulk halo.

A staged algorithm is:

```text
1. Build/refit one local surface BVH per rank.
2. Exchange coarse surface bounding regions.
3. Identify potentially interacting rank pairs.
4. Exchange only required remote surface features.
5. Build point-triangle and edge-edge candidate records locally.
6. Assign each contact record to one deterministic owner.
7. Assemble local and remote nodal contributions.
8. Reduce remote contributions to their DOF owners.
9. Compute the global CCD step as an MPI minimum reduction.
```

Start distributed contact development with rigid analytical obstacles because they require no remote surface-facet exchange. Follow with two-body deformable contact, then self-contact and contact-work rebalancing.

Bulk partition balance does not imply contact-work balance. Keep the possibility of distributing temporary contact records independently of the volume partition.

### 26.15 AI-assisted mesh compilation

AI can substantially reduce the human difficulty of DMPlex and geometry APIs, but it must operate above a fixed contract.

The AI/compiler layer may generate:

- a `MeshPlan` from the application description;
- geometry-entity attachment and label-propagation code;
- backend-specific DMPlex, DMForest, or native refinement calls;
- extraction of compact kernel views;
- partition and communication diagnostics;
- geometry projection and quality-check drivers;
- application-specific mesh and contact regression tests.

The AI layer must not silently infer missing topology or invent curvature not supported by the authoritative geometry. Every generated mesh plan should emit a manifest containing:

```text
input mesh and geometry hashes
geometry entity map
refinement policy and levels
partitioner and rank count
field layout
constraint policy
geometry and quality tolerances
contact-surface construction mode
backend and compiler versions
```

Generated adapters are accepted only after invariant tests pass.

### 26.16 Required invariants and automated tests

For every backend and partition count, verify:

- every cell has exactly one owner;
- every required element DOF is owned locally or available as a ghost;
- owner-to-ghost broadcast and ghost-to-owner reduction are adjoint within tolerance;
- boundary and material labels are preserved after refinement and redistribution;
- geometry vertex/curve/surface classification is preserved;
- refined boundary points satisfy the geometry tolerance;
- all reference element Jacobians are positive at required check points;
- residual and `Jv` are invariant, within numerical tolerance, under MPI rank count, partitioner, cell ordering, and ghost ordering;
- contact feature IDs and pair ownership are deterministic;
- restart reproduces solution, committed state, mesh hierarchy, and contact state;
- no production input/output path gathers the complete fine mesh onto one rank.

A useful algebraic communication test is:

\[
\langle G\mathbf u,\mathbf v\rangle
=
\langle\mathbf u,G^T\mathbf v\rangle,
\]

where `G` is the owner-to-ghost gather operation.

### 26.17 Implementation work packages

#### Mesh-1 — Define contracts

- Define `MeshPlan`, `GeometryBackend`, and `KernelMeshView` schemas.
- Define global IDs, ownership, ghost, label, and constraint semantics.
- Add a serialized mesh-build manifest.

**Exit criterion:** a serial mesh can be converted into the kernel view without exposing backend-specific topology calls to the kernels.

#### Mesh-2 — Uniform refinement with analytical geometry

- Implement one low-order element family.
- Add uniform startup refinement.
- Support planes, cylinders, and spheres as geometry sources.
- Propagate boundary labels and project new boundary nodes.
- Add Jacobian and boundary-error validation.

**Exit criterion:** refined curved-boundary convergence is demonstrated against an analytical geometry.

#### Mesh-3 — MPI distribution and compact views

- Distribute the coarse mesh before large refinement.
- Construct owned/ghost DOFs and communication plans.
- Extract batch-oriented local arrays for F2PY.
- Verify serial/MPI residual and `Jv` equivalence.

**Exit criterion:** a one-million-DOF bulk problem runs without global mesh replication.

#### Mesh-4 — Ten-million-DOF bulk benchmark

- Add scalable input/checkpoint/output.
- Add hierarchy-based or other scalable preconditioning.
- Profile mesh build, refinement, communication, residual, `Jv`, and memory.
- Establish strong- and weak-scaling baselines.

**Exit criterion:** the target problem fits the planned hardware envelope and meets documented solver and memory criteria.

#### Mesh-5 — CAD or NURBS geometry backend

- Import or connect an authoritative curved geometry.
- Preserve vertex/curve/surface classification and parameters.
- Add high-order coordinate nodes and quality optimization.
- Validate normals and boundary location independently.

**Exit criterion:** high-order curved bulk benchmarks converge at the expected geometric and solution rates.

#### Mesh-6 — Refined contact surfaces

- Extract contact surfaces after final refinement.
- Build stable feature IDs and BVHs.
- Add rigid analytical contact first.
- Add distributed deformable contact and remote feature exchange.

**Exit criterion:** contact results are invariant under partition changes and satisfy geometry-resolution tolerances.

#### Mesh-7 — Local startup refinement and constraints

- Add region-based refinement.
- Add hanging-node constraint operators to gather/scatter and `Jv`.
- Extend multilevel transfers.

**Exit criterion:** constrained residual and Jacobian actions match an independently assembled reference.

#### Mesh-8 — Runtime adaptivity research track

- Specify state-transfer semantics.
- Implement conservative or model-aware history transfer for selected materials.
- Transfer contact history and ownership.
- Add refinement/coarsening acceptance tests.

**Exit criterion:** path-dependent work and dissipation remain within documented transfer-error tolerances.

### 26.18 Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Coarse boundary is treated as exact geometry | Refined mesh preserves the wrong shape | Require an authoritative geometry source or explicitly declare the discrete surface as the physical geometry |
| Geometry topology is wrong | Projection cannot repair missing or merged features | Reject the mesh during classification/validation; fix the coarse topology |
| Boundary projection inverts cells | Solver failure or invalid integration | Interior smoothing/optimization and mandatory Jacobian checks |
| DMPlex concepts leak into generated kernels | Compiler and runtime become hard to maintain | Enforce the narrow `KernelMeshView` contract |
| Fine mesh is built on rank zero | Startup memory and time do not scale | Bootstrap only when necessary, then distribute and refine in parallel |
| Mesh exists in several full copies | Ten-million-DOF memory target is lost | Explicit ownership model and memory-category reporting |
| High-order solution uses affine geometry | Curved-boundary convergence is degraded | Separate geometry basis/order in the IR and manifest |
| PPF contact operates on under-resolved facets | Gap and contact pressure follow faceting error | Geometry-resolution criterion and curved-evaluation mode |
| Local refinement introduces ignored constraints | Incorrect residual and `Jv` | Constraint-aware gather/scatter operators |
| AI-generated adapter is plausible but wrong | Silent ownership or geometry errors | Mandatory invariant, partition, geometry, and adjoint communication tests |
| Runtime adaptation corrupts history | Nonphysical path-dependent response | Defer to a separate research phase with model-aware transfer |

### 26.19 Revised near-term recommendation

The near-term mesh strategy is:

> Read a small, topologically correct coarse mesh together with an authoritative analytical, CAD, NURBS, signed-distance, or explicitly discrete geometry; distribute before the mesh becomes large; refine and re-embed it in parallel; validate geometry and element quality; then expose only compact rank-local arrays to the existing F2PY/Fortran bulk and contact kernels.

DMPlex should be one hidden implementation of this strategy, not the public programming model. AI should synthesize and maintain application-specific mesh plans and adapters, while ownership, communication, geometry, and verification contracts remain small, stable, and trusted.
