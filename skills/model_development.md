# CoupFE model development

Use this skill when adding a new element, material, or coupled-field model to
CoupFE through the native or Abaqus-export backend.

## First moves

1. **Inspect repository state:**
   ```bash
   git status --short
   ```
   Do not overwrite collaborator edits or generated artifacts without understanding
   them. Stage explicit paths only; never `git add -A`.

2. **Read the live API and guides:**
   - `README.md`
   - `docs/api.md`
   - `docs/capabilities.md`
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
   **post-processed / hand-written** tangent), use a
   **generic FD/CS consistency check `K ≈ ∂R/∂U`** instead — it is field-count- and
   coupling-agnostic. See `skills/testing.md`.
5. **Solver run.** Exercise the element through `newton_solve` or
   `solve_increments` on a simple mesh: patch test, uniaxial stretch, or a
   one-step transient.
6. **Abaqus-UEL export validation (when Abaqus is available).** Generate the
   same element with `generate_uel(...)` and compare the Abaqus run against the
   native result. This is a useful convention and ABI comparison, but it is not
   an independent physics oracle when both backends share the declaration.

## Generator-friendly Python for `coupfe.codegen`

Use:

- generator-supported `coupfe.codegen.core.tensor` helpers such as `eye`,
  `trace`, `det`, `inv`, `log`, `exp`, `sqrt`, `dev`, `eig`, `logm`,
  `sqrtm`, `expm`, `dyad`, and `sym3`;
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
  The native weak residual is `+∫ P : Grad(N) dV`; the Abaqus wrapper
  returns its negative through `RHS`.
- **Pressure/constraint** (`pressure_equation(q, F, ...) -> r_p`): returns a
  scalar. The native weak residual is `+∫ r_p N dV`. This is the natural
  home for an algebraic constraint.
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

The scoped local-pressure path emits an Abaqus-style Quad4/Hex8 UEL with one
condensed element-local pressure; `CompiledElement` can drive that supported
UEL ABI in standalone tests. It is not a distinct native-kernel implementation.
Document the `SVARS` layout before writing any deck.

## Independent-oracle requirement for validation claims

A physical-validation claim needs a quantitative comparison to something the
code did not produce, such as an analytic limit, an authorized published datum,
or an independent hand computation. A cross-backend run can check
implementation agreement but is independent physical evidence only when its
formulation and data do not share the same source. Consistency checks
(`verify()`, CS-vs-FD, compile) and generated-code parity can faithfully compare
the wrong equations.

A code-generation demonstration may intentionally stop at consistency checks,
but it must be labeled as an implementation proof rather than validation.

## Broken control

For a high-value regression, deliberately reintroduce the representative bug
(for example, flip a flux sign, drop a tangent block, or omit a state argument)
and confirm the test rejects it. Keep a broken control when it materially
improves the public evidence; do not require one mechanically for every small
unit test.

## When to use native vs Abaqus backend

- **Native** (`generate_element(..., backend='native')`): use for standalone
  CoupFE. The emitted `coupfe_element_rk` returns the weak-form residual directly,
  `CompiledElement` consumes it by default, and the sign convention is the weak
  form.
- **Abaqus UEL** (`generate_uel(...)`): use for Abaqus export and validation.
  The wrapper applies the Abaqus `RHS = −R` / `AMATRX = −dRHS/dU` convention.

Keep the physics identical in both backends; only the entry-point wrapper
changes.

## Review checklist

A public element or material should have evidence appropriate to its stated
role:

- scoped equations and non-scope documented;
- properties, units, sign convention, and state layout documented;
- Python reference checks or material-point tests;
- generated Fortran compile coverage;
- f2py, solver, or Abaqus validation appropriate to the model;
- an independent quantitative oracle when making a validation claim;
- a deliberately broken control where it materially improves discrimination;
- a clear status label: "code implementation complete", "solver stabilization
  open", or "validation complete".
