# CoupFE status and plan

> **Historical planning snapshot.** This document preserves the migration
> decisions and sequencing at the time they were made; several “remaining”
> items below have since landed. For the current, claim-bounded inventory use
> [`capabilities.md`](capabilities.md).

Honest status: the **spine** is built and one real compiled element runs through
it. The bulk of a complete engine is **porting** proven code from the research lab
(`abaqus_ufl`); two capabilities are **genuinely new** (contact, distributed mesh).

> For the current **serial-vs-distributed capability matrix** (what runs under MPI, what's
> serial-only, what isn't built), see [`docs/capabilities.md`](capabilities.md). Public call
> surface: [`docs/api.md`](api.md).

## Three buckets

### 1. Done (the spine)
- The operator contract (`coupfe/operators/base.py`) + a composing driver with
  Newton, line search, and load incrementation (`coupfe/assembly`).
- One compiled element end-to-end: a neo-Hookean Quad4 via `CompiledElement` +
  `ElementGroup` (`coupfe/runtime`, `examples/neo_hookean_block`), 9 tests.
- The validation registry seed (`validation/`) and three skill docs
  (`skills/SKILL.md`, `pitfalls.md`, `testing.md`).

### 2. Port-remaining — exists in `abaqus_ufl`, low-risk but real work + verification
Everything here is *repackaging proven code*, not inventing. What's in the lab:

| area | lab module(s) | brings to CoupFE |
|---|---|---|
| distributed solver | `fe/petsc_backend.py`, `fe/petsc_mpi.py` (`solve_steps_mpi_local`, `..._staggered`, `partition_rcb/unstructured`) | PETSc/MPI scale (CoupFE's seed driver is scipy) |
| coupled drivers | `fe/driver.py` (`solve_steps_staggered`, `solve_steps_multimaterial`) | staggered + multi-material solves |
| harness | `testing/` (operators, invariants, finite_strain, objectivity, manifest, …) | the trust layer (incl. `assert_coupled_field_scale_balance`) |
| meshing / output | `fe/mesh.py`, `fe/vtk.py` | gmsh in, VTU out |
| **element zoo** | **38 examples / 71 `.for` kernels** (see `porting.md`) | the "test broadly" validation backlog |

The neo-Hookean is **history-free**; porting a *stateful/coupled* element (gel u-µ,
J2, phase-field) will exercise the per-operator `svars` commit + state protocol the
seed driver does not yet fully carry. That generalization is the heart of remaining
workstream **B**.

### The compiler is separate (decided)
The form→`.for` **generator** (`abaqus_ufl/generators`: `uel_gen`, `umat_gen`,
`uel_fbar_coupled`, `uel_local_pressure`, `uel_magneto`, `element_config`, plus the
Abaqus emitter) is a **compiler — a build-time tool, not part of the runtime.** It
turns a form into a `.for` that then runs Python-free in *both* Abaqus and CoupFE; it
is what makes "one definition, two homes" real. Decision:

- **Now (A):** keep it as `abaqus_ufl` and import it *at build time* to emit kernels;
  CoupFE the runtime consumes the `.for` (today it vendors pre-generated ones).
- **Later (B):** re-home it as a clean `coupfe-gen` package — a *mechanical* copy +
  rename (not a rewrite), porting-track work.
- **Never (C):** merge it into the `coupfe` runtime package — it is the most
  pitfall-dense code in the lab and re-touching it is a regression risk; build-time
  machinery does not belong in the runtime.

```
coupfe-gen (compiler, build-time):  form.py ─▶ element.for ─┬─▶ Abaqus (UEL)
                                                            └─▶ coupfe (runtime, f2py+PETSc)
```

### 3. Genuinely new — NOT in `abaqus_ufl`, real R&D (developed in CoupFE)
- **Contact** — `docs/dev/contact.md`. Lab has none for the *solver* (only
  `uinter_gen`, the Abaqus contact-interface ABI). RetroMech has rigid-SDF Stage-1.
  Deformable / self-contact (ppf-style barrier + CCD) is new.
- **Distributed mesh generation + parallel refinement to ~10M DOF** —
  `docs/dev/distributed_mesh.md`. Lab distributes an *existing* mesh and does basic
  gmsh; the coarse→refine-in-parallel-without-gathering pipeline is new.
  **M1 ✓, M2 ✓, M3 ✓:** the `GeometryBackend`/`KernelMeshView` contracts, uniform
  refinement with curved-boundary re-embedding + the Jacobian gate (`tests/test_mesh.py`),
  curved-annulus convergence at ~h² (`examples/curved_annulus`,
  `tests/test_curved_convergence.py`), and mesh distribution — owned/ghost `LocalMesh`
  partition with the *sum-of-parts == serial* invariant (`tests/test_distribute.py`) and a
  real-MPI smokes (serial == N-rank to ~1e-15 at 1/2/4 ranks). **M3b ✓ (core):** a
  memory-local distributed *solve* — each rank builds only its block (no global mesh),
  PETSc KSP, serial == N-rank with a **rank-independent** CG count
  (`examples/mpi_smoke/distributed_solve.py`, gated in `tests/test_mpi_distributed.py`).
  **Remaining:** wire the operator contract + f2py element + nonlinear Newton onto this
  pattern (the production distributed solve, reusing the lab's `solve_steps_mpi_local`).
- (matrix-free `Jv` + GPU — deferred until a problem needs the scale *and* has a
  known preconditioner.)

## Division of work
- **Porting:** bucket 2 — pour the 38-element zoo into the `validation/`
  registry, and port the distributed solver / codegen / harness into the clean
  structure. Recipe + backlog: `docs/porting.md`. Each port lands with an
  independent oracle + a broken control (`skills/testing.md`).
- **New algorithms:** bucket 3 — contact and distributed mesh, designed
  against the operator contract. Specs in `docs/dev/`.

These run in parallel: the porting fills out a complete standalone+Abaqus engine
(the product); the new algorithms add the capabilities `abaqus_ufl` never had.

## Sequencing of the new work
**Distributed mesh** is the foundational, fully-greenfield piece (it underwrites the
"~10M DOF / more scalable than Abaqus" claim) — a natural first new-algorithm.
**Contact** proceeds in parallel: its rigid-SDF Stage-1 is a *port* (from RetroMech,
bucket-2 work), freeing the new-algorithm effort for the deformable/self-contact
core. Both are application-driven — build to the depth a real engagement needs.
