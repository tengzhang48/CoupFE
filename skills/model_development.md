# CoupFE model development

Use this skill when adding a new element, material, or coupled-field model to
CoupFE. It adapts the lab's battle-tested workflow to the CoupFE namespace and
the new native backend.

## First moves

1. **Check the worktree:**
   ```bash
   git status --short
   ```
   Do not overwrite collaborator edits or generated artifacts without understanding
   them. Stage explicit paths only; never `git add -A`.

2. **Read the live API and guides:**
   - `docs/API_USAGE.md`
   - `docs/quickstart.md`
   - `docs/dev/native_element_abi.md`
   - `skills/pitfalls.md`
   - `skills/testing.md`

3. **Pick the target deliberately:**
   - **Finite-strain material** returning `stress_PK1(F)` → use `coupfe.codegen.Material`.
   - **Small-strain path-dependent material** → use `coupfe.codegen.SmallStrainMaterial`.
   - **WeakForm / UEL** for any extra solved nodal field, gradient term, mixed
     interpolation, phase field, diffusion, transport, or coupled multi-field problem
     → subclass `coupfe.codegen.WeakForm`.
   - **Native CoupFE element** → `generate_element(..., backend='native')`.
   - **Abaqus export** → `generate_uel(...)`.

## Core rule

Do not start from Fortran. Start from the theory, write generator-friendly
Python, verify the Python model, generate Fortran, compile, then climb the
validation ladder.

## Validation ladder

Use the smallest layer that can expose the suspected bug. Do not call a model
production-ready because a generated `.for` file compiles.

1. **Python reference / material-point checks.** Run the constitutive law by
   hand on characteristic states: zero stress/zero flux at the reference state,
   elastic limit, monotonic state evolution, flux sign vs. the strong form.
2. **`problem.verify()` (tangent consistency).** Checks that the generated
   tangent blocks are the derivatives of the implemented residual. This is a
   consistency check, not a physical oracle.
3. **Generated Fortran compile.** `gfortran -c -ffixed-form -ffixed-line-length-none`
   on the generated `.for`. For finite-strain tensor returns, grep for
   impossible component-indexed matrix helpers such as `det33z(F(ii,jj))`.
4. **f2py single-element check.** Build the kernel with
   `coupfe.runtime.compiled_element.build_element_kernel` and drive it with
   `CompiledElement.element_rk`. Compare against
   `coupfe.codegen.core.reference_assembly.assemble_element` (the Python
   reference assembly) at the same state. `assemble_element` re-assembles the
   tangent from a **hand-enumerated** case table; it now covers a value-residual
   that depends on a **scalar field's gradient** (`grad_phi`/`grad_c` source/storage
   coupling), and **raises `NotImplementedError("unhandled … pattern …")`** if your
   element's coupling hits an unfilled cell — that means "extend the oracle," not
   "your element is wrong." For tangents the table can't cover (and for any
   **post-processed / hand-written** tangent, e.g. axisymmetric hoop blocks), use a
   **generic FD/CS consistency check `K ≈ ∂R/∂U`** instead — it is field-count- and
   coupling-agnostic. See `skills/testing.md` and
   `docs/dev/verification_harness_redesign.md`.
5. **Solver run.** Exercise the element through `newton_solve` or
   `solve_increments` on a simple mesh: patch test, uniaxial stretch, or a
   one-step transient.
6. **Abaqus-UEL export validation (when Abaqus is available).** Generate the
   same element with `generate_uel(...)` and compare the Abaqus run against the
   native result. This is the strongest external oracle for convention and sign.

## Generator-friendly Python for `coupfe.codegen`

Use:

- `coupfe.codegen.core.tensor` helpers: `eye`, `trace`, `det`, `inv`, `log`,
  `exp`, `sqrt`, `dev`, `eig/eigh`, `logm`, `sqrtm`, `expm`, `dyad`, `sym3`.
- 3x3 tensor form until the Fortran boundary.
- bounded `for k in range(N)` loops with optional `break`.
- explicit scalar/tensor state variables in `state_vars`.
- explicit three-item compare/swap logic instead of `np.argsort`.
- smooth regularization where complex-step derivatives matter.
- scalar temporaries for `det(F)` and other matrix-to-scalar calls in tensor
  returns (see `skills/pitfalls.md`).

Avoid in generated methods:

- `while`, dynamic `append`, list comprehensions, dataclasses as generated
  state carriers, `np.einsum`, hidden file I/O, general dynamic arrays,
  undocumented parameter guesses.
- `z**2` for a square; write `z*z`.
- `sym(A)` in generated code; write `0.5*(A + A.T)`.
- branching on the imaginary part of a complex-step perturbation; branch on
  `.real` instead.

If generation fails, prefer rewriting the model into supported 3x3 tensor
operations. Add generator support only when the pattern will recur.

## WeakForm / UEL patterns

### Canonical field and test-function names

The recognized material methods live in `coupfe.codegen.core.defs:METHOD_INFO`:
`stress_PK1`, `stress_update`, `pressure_resid`, `solvent_storage`,
`solvent_flux`, `phase_storage`, `phase_flux`, `pressure_storage`,
`pressure_flux`, `species_storage`, `species_flux`, `h_field`,
`volumetric_PK1`, etc. A method with a different name is silently ignored.

The recognized equation methods live in `coupfe.codegen.core.weakform:EQUATION_INFO`:
`momentum_equation`, `pressure_equation`, `transport_equation`,
`phase_equation`, `pressure_transport_equation`, `species_transport_equation`.

Test-function-to-field mapping (`core.weakform:TEST_TO_FIELD`):

| test function | field | typical equation |
|---|---|---|
| `v` | `u` (displacement) | `momentum_equation` |
| `q` | `p` (pressure) | `pressure_equation` |
| `w` | `mu` (chemical potential) | `transport_equation` |

When you declare a scalar field with a custom test name, the first argument of
the equation method is that test name; its purpose is documentation. The
framework resolves it to the declared field.

### Momentum vs pressure vs transport

- **Momentum** (`momentum_equation(v, F, ...) -> P`): returns a 3x3 PK1 stress.
  The weak residual is `−∫ P : Grad(N) dV`.
- **Pressure/constraint** (`pressure_equation(q, F, ...) -> r_p`): returns a
  scalar. The weak residual is `−∫ r_p N dV`. This is the natural home for an
  algebraic constraint.
- **Transport-like** (`transport_equation(w, F, c, grad_c, c_old, dt)`): returns
  `(storage, flux)`. The weak residual is
  `∫ storage N dV − ∫ flux · Grad(N) dV`.

### Scalar UEL sign rule

`return storage, flux` means the generated weak residual is:

```text
R += storage * test - flux . grad(test)
```

For ordinary diffusion with `storage = c_dot`, return the physical flux
`-D * grad_c`. For AT2 damage/fracture with
`storage = Gc/ell*d - 2*(1-d)*H`, return `-Gc*ell*grad_d` so the weak form has
`+Gc*ell*grad(d).grad(test)`. For phase-field models, do not reason from
diffusion flux examples. Write the positive weak-form gradient term first, then
return the adapter required by the generator convention.

Do not infer flux sign from field name; derive it from the exact `storage`
quantity returned by that equation.

### F-bar and local-pressure only when justified

Use the standard UEL path for coupled fields. Use F-bar/local-pressure
formulations only when the element theory justifies them; they are not default
stabilization switches for gels, diffusion, phase fields, or transport.

For the native backend, `formulation='local_pressure'` is supported as a
prototype path for `u,mu` gel-style Quad4/Hex8 elements with one condensed
element-local pressure. Document the `SVARS` layout before writing any deck.

## Independent-oracle requirement

Every new model needs a quantitative comparison to something the code did not
produce: an analytic limit, a digitized paper point, a hand computation, or a
cross-backend (Abaqus UEL) run. Consistency checks (`verify()`, CS-vs-FD,
compile) and code-vs-itself oracles (f2py comparing generated Fortran to the
Python reference) are not enough — they differentiate the wrong code faithfully.

A green suite with only consistency checks certifies almost nothing about
constitutive correctness. The one gate that breaks that is a quantitative
comparison to an independent source.

## Broken control

Every regression test must fail if the bug is reintroduced. Before calling a
new test done, deliberately reintroduce the bug (flip a flux sign, drop a
tangent block, use `V.T` instead of `inv(V)`, omit a state `_old` argument) and
confirm the test rejects it. A test that passes on the broken code is asserting
nothing.

## When to use native vs Abaqus backend

- **Native** (`generate_element(..., backend='native')`): use for standalone
  CoupFE. The emitted `coupfe_element_rk` returns the weak-form residual directly,
  `CompiledElement` consumes it by default, and the sign convention is the weak
  form.
- **Abaqus UEL** (`generate_uel(...)`): use for Abaqus export and validation.
  The wrapper applies the Abaqus `RHS = −R` / `AMATRX = −dRHS/dU` convention.

Keep the physics identical in both backends; only the entry-point wrapper
changes.

## Done definition

A new element/material is not done until it has:

- scoped equations and non-scope documented;
- properties, units, sign convention, and state layout documented;
- Python reference checks or material-point tests;
- generated Fortran compile coverage;
- f2py, solver, or Abaqus validation appropriate to the model;
- at least one independent, quantitative oracle;
- at least one deliberately broken control;
- a clear status label: "code implementation complete", "solver stabilization
  open", or "validation complete".
