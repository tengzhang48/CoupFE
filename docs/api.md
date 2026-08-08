# CoupFE public API

CoupFE is alpha software. This page describes the calls available in the
current public source; maturity and unsupported combinations are tracked in
[`capabilities.md`](capabilities.md).

The base installation requires NumPy and SciPy. Compiled elements require a
Fortran toolchain, code generation requires the `codegen` extra, PETSc/MPI
paths require a consistent PETSc, `petsc4py`, and MPI environment, and
accelerated 3-D contact can use the `performance` extra.

## Top-level imports

The following names are exported by `import coupfe`:

| Area | Names |
|---|---|
| Operator contract | `Operator`, `Residual`, `Tangent`, `complex_step_tangent` |
| Element runtime | `ElementGroup`, `GroupState`, `CompiledElement`, `build_element_kernel` |
| Assembly and drivers | `assemble_residual`, `assemble_tangent`, `newton_solve`, `solve_increments`, `solve_dynamics`, `solve_dynamics_adaptive`, `InertiaOperator` |
| Constraints | `ConstraintRelation`, `ConstraintTransform`, `compile_affine_constraints` |
| Mesh and geometry | `KernelMeshView`, `Circle`, `Sphere`, `Plane`, `uniform_refine_quad`, `check_positive_jacobian` |
| Declarative setup | `Model`, `Result`, `NeoHookean` |

`CompiledElement` and `build_element_kernel` are set to `None` if their optional
runtime import cannot be loaded. Check them before using the compiled path in a
base-only environment.

## Operator contract

An `Operator` contributes a residual, tangent, and optional committed state.
Assembly collects local values into the global system:

```python
R, trial = assemble_residual(operators, U, state, t, dt, ndof)
K = assemble_tangent(operators, U, state, t, dt, ndof)
```

`complex_step_tangent(residual_fn, ue, h=1e-30)` differentiates a smooth local
residual. Discrete search, active-set, or branch decisions must be treated
separately rather than differentiated as though they were smooth.

`ElementGroup(element, nodes, elems, dof_per_node, comps=None, *,
fuse_rk=None, evaluation_mode="joint")` wraps a batched compiled element as an
operator. ``evaluation_mode="joint"`` preserves the default cached R/K call.
``"split"`` uses a native residual-only entry for residual callbacks and the
joint entry for tangents; it raises if the kernel predates that entry.
The modes are numerically gated, but the faster choice depends on callback
order and the number of residual-only trials in the consuming solver.

## Serial drivers

| Call | Purpose |
|---|---|
| `newton_solve(operators, U0, state, ndof, dirichlet, *, t=1, dt=1, rtol=..., maxit=..., constraints=None)` | One Newton solve with line search and operator step bounds. |
| `solve_increments(operators, U0, ndof, dirichlet, *, n_steps=4, constraints=None, **newton_kw)` | Quasistatic load increments. Constraint offsets are ramped for a fixed relation set; a callable may provide a non-proportional schedule. |
| `solve_dynamics(operators, U0, ndof, dirichlet, *, dt, n_steps, constraints=None, **newton_kw)` | Fixed-step implicit backward-Euler dynamics. |
| `solve_dynamics_adaptive(operators, U0, ndof, dirichlet, *, t_end, dt_init, dt_min=None, dt_max=None, growth=1.2, cut=0.5, maxit=None, constraints=None, **newton_kw)` | Adaptive backward-Euler dynamics with step growth and retry. |
| `InertiaOperator(mass, ndof, *, u0=None, v0=None, damping=0)` | Lumped-mass inertia and optional mass-proportional damping. |

`newton_solve` returns an iteration count rather than a convergence flag and
calls each operator's `commit` method after the Newton loop. Callers using
path-dependent state must independently establish convergence before treating
that state as accepted. `solve_increments` passes `state=None` at every
increment and is intended for history-free operators.

Dynamic relaxation can be used to approach a quasistatic equilibrium when the
loading protocol includes a hold/settle stage and the final state satisfies
appropriate residual, kinetic-energy, and time-step checks. It does not by
itself guarantee that a desired equilibrium branch was reached.

The serial dynamics functions currently reject nonempty affine constraints.

## Exact affine constraints

One scalar relation has the form

```text
U[slave] = sum(coefficients[j] * U[masters[j]]) + offset
```

Construct it with:

```python
relation = ConstraintRelation(
    slave,
    masters=(master_a, master_b),
    coefficients=(1.0, -1.0),
    offset=0.0,
    label="optional description",
)
transform = compile_affine_constraints(ndof, [relation], dirichlet=dirichlet)
```

`ConstraintTransform` represents `U = P @ q + offset` and provides `lift`,
`project_increment`, `reduce_guess`, `restrict_residual`, `reduce_tangent`,
`reduce_linear_system`, `relation_matrix`, and `constraint_error`. Compilation
rejects duplicate slaves, cycles, invalid indices, and Dirichlet/relation
conflicts.

`newton_solve(..., constraints=relations)` and
`solve_increments(..., constraints=relations_or_schedule)` are the qualified
driver integrations. Applications construct mesh matching, periodic graphs,
and domain-specific relations; current MPI drivers do not apply this transform.

## Declarative `Model`

`Model` is a convenience layer over the operator contract for regular 2-D
examples:

```python
from coupfe import Model, NeoHookean

model = Model.structured(8, 4, Lx=2.0, Ly=1.0)
model.material("block", NeoHookean(G=1.0, K=10.0))
model.fix("left", x=0.0, y=0.0)
model.prescribe("right", x=0.1)
result = model.solve(steps=4)
```

For `NeoHookean`, `G` is shear modulus and `K` is the physical small-strain
bulk modulus. The convenience material converts these to the retained raw
core-kernel ABI `(G, lambda)`, where `lambda = K - 2G/3`. Direct users of the
vendored kernels under `coupfe/runtime/elements` can call
`neo_hookean_kernel_props(G, K)` to perform the same conversion. Code-generated
kernels instead retain the property convention declared by their source model.

Construction and setup calls are:

- `Model.structured(nx, ny, Lx=1, Ly=1)`;
- `Model.from_view(view)`;
- `refine(levels=1)`;
- `material(name, material, elements="all", comps=(0, 1))`;
- `fix(selector, **components)` and `prescribe(selector, **components)`;
- `contact(selector, obstacle, k=..., comps=(0, 1), mu=0, k_t=None)`; and
- `solve(steps=1, *, rtol=..., **newton_kw)`.

Selectors may be named node sets, bounding-box names such as `left` and `top`,
callables, or explicit node arrays. `Result` exposes `U`, `converged`, `iters`,
`displacement()`, and `position(selector=None)`. This convenience layer does not
replace application-specific model review.

## Mesh contracts

`KernelMeshView(nodes, elems, dof_per_node=2, node_sets=..., elem_sets=...,
node_geometry=..., geometries=...)` is the neutral in-memory bridge.

- `uniform_refine_quad(view, reembed=True)` uniformly refines a Quad4 view and
  optionally projects new boundary nodes through registered geometry objects.
- `check_positive_jacobian(view)` checks the currently supported 2-D element
  geometry.
- `Circle`, `Sphere`, and `Plane` provide analytic projection backends.

General file import, CAD association, mixed-cell topology, and application
labels are outside this API.

## Compiled-element runtime

```python
module = build_element_kernel(for_path, module_name, workdir=None, backend=None)
element = CompiledElement(
    module,
    props,
    dof_per_node,
    n_svars=0,
    mcrd=2,
    n_elem=None,
    dt=1.0,
    backend=None,
    state_schema=None,
)
```

The runtime recognizes native and Abaqus-UEL-style generated element entry
points. It compiles a supplied Fortran source once through f2py; a compatible
Fortran compiler, Meson, and Ninja must be available for that build step.

The two backends are parallel. Native calls use CoupFE's own ABI and never
receive Abaqus ``LFLAGS``. The UEL route is a normal-static joint-call
compatibility adapter for focused parity checks and selected research examples,
not a general Abaqus procedure simulator; Abaqus supplies the real call context
when it runs an exported UEL.

The generated UELs in this release are scoped to their documented normal
implicit/static use. Mass-, damping-, perturbation-, and general dynamic-request
handling is not qualified by this package; those Abaqus procedures require
dedicated request and state-sequencing work before they can be claimed.

Current standard and F-bar native generation emits both
``coupfe_element_rk`` and a ``coupfe_element_r`` twin. ``CompiledElement`` exposes
``element_rk``/``element_rk_batch`` and ``element_r``/``element_r_batch``;
``has_element_r`` and ``has_element_r_batch`` report the optional entries.
For older native sources and Abaqus UELs, the residual calls fall back to the
joint path without changing signs or state semantics.

The generated ``coupfe_element_rk`` signature is unchanged. The low-level
f2py ``drive_native*`` wrapper now takes only the seven native inputs
``(svars, coords, u, du, props, time, dtime)``; rebuild cached compiled modules
created with an earlier wrapper.

## Linear solvers

The following calls live in `coupfe.assembly.factored`:

| Call | Purpose |
|---|---|
| `linear_solve(K, b, *, prefer="auto")` | One sparse linear solve with the available backend policy. |
| `factored_lu(K, *, prefer="auto")` | Factor once and solve one or more right-hand sides. |
| `iterative_solve(K, b, *, ksp_type="gmres", pc_type=None, rtol=..., max_it=...)` | Opt-in PETSc Krylov solve; raises on non-convergence. |
| `make_fieldsplit_solver(field_components, dof_per_node, *, ksp_type="gmres", split_type="additive", sub_pc="gamg", ...)` | Build a callable PETSc FieldSplit solver for node-major coupled fields. |

Backend availability depends on the PETSc installation. Profile and verify a
representative problem before overriding the default policy; no universal size
crossover is claimed.

## Distributed drivers

The optional PETSc/MPI calls live in `coupfe.assembly.distributed`:

- `element_partition(view, rank, size, comps=None)`;
- `solve_distributed(...)` for history-free, single-field quasistatic Newton;
  and
- `solve_dynamics_distributed(...)` for implicit dynamics with node-local
  inertia and optional deformable contact.

`solve_distributed` accepts optional node-local rigid penalty contact and
cross-rank deformable-contact specifications. `solve_dynamics_distributed`
accepts optional force, Robin, pressure, and deformable-contact specifications.
See the function signatures and `examples/mpi_smoke/` for the complete argument
shape.

For quasistatic Newton, pass
``residual_batch_fn=element.element_r_batch`` with
``evaluation_mode="split"`` to use residual-only assembly for convergence and
line-search callbacks. ``evaluation_mode="joint"`` remains the default.

These implementations ship, but the current public release does not include a
retained final-revision multi-rank qualification record. Distributed stateful
element commit, generic affine constraints, and every serial contact mode are
not supported.

## Contact APIs

Rigid obstacles and 2-D contact live in `coupfe.operators.contact`:

- `HalfSpace`, `Sphere`, `MovingHalfSpace`, `MovingSphere`;
- `RigidContact` for penalty contact with optional return-map friction;
- `RigidBarrierContact` for cubic-barrier contact and collision step bounds;
- `SurfaceContact2D` and `DeformableContact2D` for penalty contact; and
- `DeformableBarrierContact2D` for deformable barrier contact with optional
  smoothed, return-map, or persistent friction modes.

`DeformableBarrierContact3D` lives in `coupfe.operators.contact3d` and supports
vertex-face and edge-edge contact. Optional numba acceleration has a NumPy
fallback. `candidate_pairs(...)` in `coupfe.operators.contact_search` provides
the 2-D broad-phase candidate map.

Small-scale exact-stick research calls live in
`coupfe.operators.contact_semismooth`:

- `solve_friction_semismooth(...)`; and
- `SemismoothFrictionSolver(...)`.

They use a linear bulk matrix and lagged normal data; they are not the general
nonlinear or distributed contact driver.

### Hertz example evidence record

The retained Hertz benchmark has a source-tree helper for reproducible evidence;
it is an example interface, not a name exported by `coupfe`:

```python
from examples.hertz_contact.run import run_hertz

evidence = run_hertz(deltas=(0.03, 0.05, 0.07))
```

`run_hertz(...)` compiles and solves the finite Hex8 model, then returns:

- `configuration`: material parameters, block and mesh dimensions, penalty,
  and problem size;
- `deltas`, `force_fe`, `force_hertz`, `force_ratios`, and `fit_slope`;
- `cases`: Newton iterations, free-DOF residual, base/contact force-balance
  error, penetration, and active-node diagnostics for every load; and
- `snapshot`: reference/deformed nodes, displacement, connectivity, boundary
  node IDs, gaps, and discrete nodal reactions for the final load.

`solve_hertz(...)` retains the earlier compact
`(deltas, force_fe, active_node_radius)` return. The richer record is preferred
for tests and figures. `render_hertz_svg(evidence, output_path)` in the adjacent
`render.py` writes the solver-backed SVG without rerunning when an evidence
record is supplied.

The reaction arrays are defined on `snapshot["top_nodes"]`. They are discrete
penalty-node quantities, not a continuous pressure reconstruction, and
`active_node_radius` is a mesh diagnostic rather than a qualified contact-radius
extractor. The complete setup and evidence boundary are in
[`../examples/hertz_contact/README.md`](../examples/hertz_contact/README.md).

## Build-time code generation

Install the `codegen` extra and import `coupfe.codegen` for:

- declarations: `Material`, `SmallStrainMaterial`, `WeakForm`, `VectorField`,
  `ScalarField`, and `LocalScalar`;
- verification: `VerificationError` and each declaration's `verify()` method;
- UEL generation: `generate_uel` and `generate_uel_local_pressure`;
- UMAT generation: `generate_umat` and `generate_small_strain_umat`;
- Abaqus input scaffolding: `UELModelConfig`, `ScaffoldReport`,
  `generate_inp_scaffold`, `write_job_inp`, and
  `ensure_coupled_dummy_material`; and
- tensor, plasticity, soil, and Fortran-helper utilities exposed by the module.

Generated-kernel/reference-assembly or native/UEL agreement checks
implementation consistency. They are not independent validation of the source
equations or parameters.
