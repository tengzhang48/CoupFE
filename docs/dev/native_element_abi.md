# Native element ABI — implemented

**Status: implemented.** The native element ABI is no longer a spec item; it is the default
backend for standalone CoupFE.

- `generate_element(..., backend='native')` is available and emits `SUBROUTINE coupfe_element_rk`.
- `coupfe/runtime/compiled_element.py` consumes native kernels by default (`CompiledElement`
  auto-detects `drive_native` vs `drive_uel`).
- The Abaqus-UEL backend remains via `generate_uel(...)` (an alias for
  `generate_element(..., backend='abaqus_uel')`) and is still used for Abaqus export and the
  dual-home regression.

**Goal.** Give CoupFE its *own* element-kernel interface so the standalone (and eventually
open-source) runtime no longer depends on the Abaqus UEL ABI. Abaqus-UEL stays a **backend**
(we remain UEL-compatible — that's our entry product), but it is demoted from *the foundation*
to *an export*. This also clears the flux-sign confusion (the native residual = the weak form;
no Abaqus `RHS = −R` / `AMATRX = −dRHS/dU` layer). Design rationale: `docs/dev/element_interface.md`.

**Why now (Teng):** the codegen is fresh, the runtime is two small files, and every element
added *before* this accumulates UEL-shaped baggage. Better now than later.

## The native kernel ABI

A clean Fortran subroutine — only what CoupFE needs, **none** of `LFLAGS`, `NRHS`, `RHS(:,2)`
(Modified Riks), `PNEWDT`, `KSTEP`/`KINC`, or the Abaqus arg order:
```
SUBROUTINE coupfe_element_rk(coords, u, du, props, svars_in,
                             R, K, svars_out)
   ! coords     (ndim, nnode)        reference nodal coordinates
   ! u, du      (ndofel)             total dof + increment (node-major)
   ! props      (nprops)             material parameters
   ! svars_in   (nsvars)             committed state
   ! R          (ndofel)        OUT  residual  = the WEAK-FORM contribution (see signs)
   ! K          (ndofel,ndofel) OUT  consistent tangent = ∂R/∂u (complex step inside)
   ! svars_out  (nsvars)        OUT  trial state
```
This is exactly what `CompiledElement.element_rk_batch(coords, U, DU) → (R, K)` already exposes
at the Python level — so the runtime change is small.

## Sign convention — follow the weak form, full stop

The native residual is the weak-form integral as written in `EQUATION_INFO`:
```
R = Σ_terms rhs_sign · ∫ (material_return · test-gradient-or-shape) · w·detJ
    momentum:  −∫ P : Grad(N)        storage: −∫ ċ N        flux: +∫ j·Grad(N)
```
Newton solves `K δu = −R`. **No `RHS = −R` and no `AMATRX = −dRHS/dU`.** Those exist only
because Abaqus defines RHS as the *negative* internal force and AMATRX as `−dRHS/dU`; they are
applied **only in the `abaqus_uel` backend**, never in the native kernel or the runtime. One
sign convention (the weak form) in the core; Abaqus's is a backend translation.

## Generator structure — shared physics, two wrappers

The material subroutines, the element-RK loop, and the complex-step tangent emit are **shared**.
Only the entry-point wrapper differs:
- **`native` backend** → `coupfe_element_rk` (the clean ABI above; weak-form signs).
- **`abaqus_uel` backend** → `SUBROUTINE UEL(RHS, AMATRX, ..., LFLAGS, NRHS, ...)` with the
  Abaqus conventions (RHS = −R, AMATRX = −dRHS/dU, the Riks `RHS(:,2)` guard, etc.). **Keep
  this** — it is the Abaqus-export path and the dual-home claim.

`generate_uel(...)` stays (the `abaqus_uel` backend). Add `generate_native(...)` (or
`generate_element(..., backend="native"|"abaqus_uel")`). They differ only in the wrapper.

## Runtime change

- `coupfe/runtime/compiled_element.py` targets the **native** kernel directly. The
  `drive_uel.f90` UEL-adapter (which currently re-shapes `UEL(...)` into the batched
  `(R, K)`) is thinned/retired for the standalone path — the native kernel already *is*
  `(R, K)`-shaped, so the wrapper becomes a trivial batch loop (or vanishes).
- Re-vendor `neo_hookean_q4` in **native** form for the runtime; keep the **UEL** form for the
  Abaqus-export path + the dual-home test.

## Iterative-solver friendliness (forward note)

The native kernel returns `(R, K-COO)` regardless of how the system is solved. Direct stays the
**verification** solver (the exact 1-vs-N invariant; `superlu_dist`); **iterative**
(`gamg`/CG for SPD elasticity, `gmres`+FieldSplit for coupled, matrix-free `Jv` for contact —
see `docs/dev/contact.md`) is the **production-scale** default. Nothing in the ABI blocks this;
keeping the kernel free of Abaqus state makes the matrix-free `Jv` path cleaner later.

## Migration steps
1. Lock this ABI (one review pass).
2. Add the `native` wrapper backend to the generator (shares all physics emit); `generate_uel`
   becomes the `abaqus_uel` backend.
3. Switch `compiled_element` to the native kernel; thin/retire `drive_uel` for the standalone path.
4. Re-vendor `neo_hookean_q4` native; re-validate the runtime (patch test + pipeline) on it.
5. `abaqus_uel` + the dual-home UEL test stay as the Abaqus-export backend (regression: regen
   UEL == vendored UEL).
6. Confirm import hygiene unchanged (runtime stays codegen/sympy-free).

## Source-line synchronization

The codegen originated in a separate research source line. Any behavior change
to the native backend, weak-form sign convention, or spectral handling must be
validated independently and recorded wherever that source lineage is still
maintained; do not allow active implementations to diverge silently.
