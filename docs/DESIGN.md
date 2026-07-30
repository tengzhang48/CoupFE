# CoupFE design

## What CoupFE is (and is not)

CoupFE is a **scaffold**, not a general PDE platform. It owns the
correctness-critical numerical core and a trust layer; it does **not** try to
own mesh-software integration, domain-specific boundary/loading definitions, or
format-specific I/O — those are thin, per-problem glue that is now fast to
write (often AI-assisted) and that the harness validates.

The product bet has three parts, and the second and third are the moat:

1. **A small, tested core** — the operator contract + the complex-step kernels.
2. **A validation harness** — operator-level gates that catch a wrong element
   from its tangent alone, before any solve (this is what makes written/generated
   glue safe).
3. **Codified development knowledge** — `skills/` ships the AI skills, pitfalls,
   and lessons *as part of the package*, so the next custom operator is built
   correctly. No other FE package ships this.

### Core boundary

CoupFE core contains only reusable numerical contracts and algorithms: operator
composition, assembly and solve drivers, element/runtime contracts, generic
constraint algebra, contact kernels, validation machinery, and the small
`KernelMeshView` array interface required by those algorithms.

Application and tool integration stays outside core. In particular, core does
not own Gmsh, `meshio`, CAD, DMPlex, or vendor-format adapters; EDA package
geometry and periodic-mesh provenance; cardiac chamber surfaces and
fiber/sheet fields; application boundary/material naming; or format-specific
I/O. EDA and cardiac translate their own data into the core contracts. Shared
integration code, if it ever becomes substantial, belongs in a separate
optional package rather than in core.

## Positioning: lightweight + tailorable, with a *robust core* — and inclusive

CoupFE does **not** set out to beat Abaqus, PERMON, deal.II, or IPC at their own
game. Those are traditions to **learn from**, not rivals to displace. The aim is a
**lightweight, tailorable** framework for **method-builders** (researchers /
specialists) — not a general-purpose product for engineers running CAD parts.

**Flexibility over *general* robustness — but a robust core.** General robustness
("works on any messy geometry without tuning") is decades of edge-case engineering
plus a product layer (CAD, meshing, GUI, support) — miles a small team won't and
needn't accumulate. It also isn't the right target: a method *tailored* to a
problem's structure (smoothness, symmetry, the specific regime) beats a general
method on that problem, and reaches problems a general tool can't express at all.
So robustness is **relocated, not abandoned**:

- **The core is robust** — the small, correctness-critical set: the operator
  contract, the complex-step / AD tangents, the contact guarantees (non-penetration,
  PSD descent), reproducibility, and the validation harness. This is held to a high
  bar; it is the thing we *do* make rock-solid.
- **The glue is flexible** — meshing, domain BC/load schedules, problem
  composition, and output: thin, per-problem, fast to (re)write, and **made
  safe by the core's guarantees surviving composition**.

A robust core is exactly what makes aggressive tailoring trustworthy. Flexibility
without invariants is a footgun; flexibility + composition-preserving guarantees is a
power tool — and it delivers *problem-specific* robustness (fit the method to the
problem) without chasing the *general* robustness we'd never win.

**Inclusive — take the best technique from each tradition (techniques, not territory):**

| Tradition | What we learn / borrow |
|---|---|
| Abaqus | engineering maturity; sensible defaults where cheap; the breadth we deliberately *don't* chase |
| IPC / ppf | **robustness by construction** (barrier + CCD + PSD projection) — adopted as the contact substrate |
| mortar (dual basis) / FETI / PERMON | **local multiplier condensation + matrix-free** scaling — adopt *if / when* scale demands it |
| FEniCS / deal.II / PETSc | composable-substrate design — own the core, not the glue |
| JAX / Warp / differentiable sim | autodiff + differentiability — **our edge**, the thing the others can't express |

Being *complementary* — reaching problems the general tools don't, carrying
guarantees they don't — is a stronger and more honest position than being marginally
"better" at what they already do well.

**The scarce asset is the validation methodology, not the flexibility.** A framework
that can express anything has an unbounded test surface, so the moat is the discipline
that keeps tailoring trustworthy: operator-level gates, the verification harness, and
curated validated exemplars (`skills/`, the material-point harness). The robust core
plus that discipline are what turn flexibility from a liability into the product.

## One element definition, two homes

A single residual definition compiles to a complex-step kernel that runs **inside
Abaqus as a UEL** and **standalone here** (f2py → PETSc/MPI) — the same kernel
in two ABIs. Deliver a custom element to a client on Abaqus; run the identical
element standalone at scale when they outgrow it. Cross-backend parity is a
code-generation/ABI regression check, not an independent physics oracle; each
model needs its own physical evidence.

## The spine: everything is an Operator

```
Operator:  residual(U, state, t, dt) -> Residual
           tangent (U, state, t, dt) -> Tangent      # COO triplets
           commit  (U, state, t, dt) -> new_state
```

A bulk element group, a contact set, and a load all implement this one
contract. The driver composes them and knows nothing about elements, materials,
or contact. Exact affine constraints are the one deliberate exception: they
transform the global solution space as `U=Pq+U0`, while every physical operator
still assembles and commits in full space. Mesh matching and domain semantics
stay in consuming packages. Materials/parts are **compositional groups**
(LAMMPS/CoupMPM style: per-group operators), not an Abaqus
parts/instances/sections hierarchy.

Two disciplines carry most of the weight:

- **One residual is the source of truth.** The tangent is *derived* from it by
  complex step — exact, no hand-coded stiffness (`complex_step_tangent`).
- **Operators are pure; state is transactional.** `state_committed` →
  `state_trial` (never committed implicitly) → recompute-and-commit only after the
  global solve accepts the step. This is what makes complex-step columns, line
  searches, and matrix-free products correct.

## Configuration

**Reference / total-Lagrangian (PK1) by default** — fixed reference geometry,
quadrature precomputed once, and complex-step differentiates the reference
residual cleanly. Contact lives in the **current** configuration regardless; a
current-config contact operator coexists with reference-config bulk (the bulk is
never converted).

## Materials — and why there is no UMAT *runtime*

In CoupFE a material is **just a function called inside an element operator's
residual**. Its tangent comes from complex-stepping the whole element residual, so
the UMAT's reason to exist (the hand-coded `DDSDDE` spatial tangent, the Jaumann
rate, the Abaqus state-var ABI) is moot. There is therefore **no UMAT runtime
path** in CoupFE — no `STRESS`/`DDSDDE` interface to host.

UMAT survives only as an **Abaqus-backend output**: if a client wants a custom
*material* inside their existing Abaqus model built from standard elements, the
same material function can be *emitted* as a UMAT. (Optionally, later, a legacy
compiled UMAT could be *hosted* via f2py inside an element operator — a "run your
Abaqus material standalone" convenience — but that is not core.) The native path
is materials-as-functions.

## Solver

Assembled tangent + direct (MUMPS) / FieldSplit-Schur for coupled multiphysics —
robust and problem-agnostic. **Field-wise convergence** (gate each field on its
own characteristic scale, as Abaqus does by default) is mandatory for coupled
problems; a single global ‖R‖ is dominated by the strong field and silently
under-resolves the weak one. Matrix-free `Jv` and GPU are a **later** lever, gated
on a problem that both outgrows assembled/direct memory *and* has a known good
preconditioner (for coupled physics, finding `P` is a per-problem research project).
