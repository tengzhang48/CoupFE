# CoupFE model-setup pipeline (P) skill

Read this before touching `coupfe/model.py` or `coupfe/materials.py`, or before adding a
problem-setup convenience. `Model` is the **declarative front door** — and the AI-glue
target (the core philosophy): the concise surface a person or an agent writes a problem
against. Keep it thin.

## What `Model` is (and is not)

`Model` is a **thin layer over the operator contract** with **no physics**:
- `material(name, mat, elements, comps)` → an `ElementGroup` (via the material spec);
- `contact(selector, obstacle, k, comps)` → a `RigidContact` operator;
- `fix` / `prescribe` → entries in the Dirichlet dict (`gdof → value`);
- `solve(steps)` → `solve_increments(operators, …)` → a `Result`.

Every behaviour is already in an operator/material. `Model` only *collects operators + builds
the Dirichlet dict + drives the solver*. If a `Model` method does something no operator could,
the abstraction has leaked — stop and move it into an operator/material.

## The one rule for extending it

**Add the capability as an operator/material first; then expose setup convenience on `Model`.
Never the reverse.**
- New physics (a coupled field, a constraint, a load) → a new operator (see `SKILL.md`):
  residual is the source of truth, tangent by complex step, state explicit.
- New material → a spec in `coupfe/materials.py` with an `element_group(view, elem_set, comps)`
  method that builds its (cached) compiled kernel and returns an `ElementGroup`. Import the
  f2py runtime **lazily** inside the method so importing the package needs no Fortran toolchain.
- Only then add the `Model` sugar (a `.material(...)` that accepts it, a selector, a schedule).

## Selectors — resolve a node target flexibly

`_resolve(selector)` accepts, in order: a **named node set** (in `view.node_sets`); a **bbox
face** (`"left"`/`"right"`/`"top"`/`"bottom"`/`"front"`/`"back"`, auto-detected from coordinate
extremes); a **predicate** `f(coords)->bool`; or an **index array**. Prefer bbox faces and
predicates for setup — they remove the node-set boilerplate. Caveat: a predicate only matches
nodes that **exist** on the mesh (e.g. `y==0.5` finds nothing on a grid whose y-nodes are
`{0, ⅓, ⅔, 1}`) — match the selector to the actual node coordinates.

## How to test a pipeline change

**Gate `Model` against the hand-wired path.** For any problem `Model` can express, there
should be a hand-wired test (`tests/test_pipeline.py`-style: view → element → contact →
`solve_increments`) and a `Model` test (`tests/test_model.py`-style) that solves the *same*
problem and asserts the *same* result (converged, physical, deformed). The pairing proves the
front door is a faithful re-expression that changes nothing about the solve. Build the
hand-wired one first.

## What still belongs in the driver/example layer (not `Model` yet)

Multi-material / mixed-DOF regions (per-region `comps`), loading schedules richer than a linear
ramp, time integration, and output (VTU — not yet ported) are demand-driven increments. Until a
real problem needs them, write them in the example/driver and let the harness validate them —
the same "keep glue out of the core" discipline, one level up.
