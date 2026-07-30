# Sustainable element interface — Abaqus-UEL as a *backend*, not the foundation

**The concern (Teng).** Today the dual-home works by generating an **Abaqus UEL `.for`** and
having the *standalone* runtime borrow it (the generated `UEL(...)` is wrapped by
`drive_uel`/`drive_uel_batch`, which `CompiledElement` calls). That ties CoupFE's standalone
runtime to the **Abaqus UEL ABI**. If the package grows, that coupling is a long-term risk:

- **Technical** — the UEL signature is a straitjacket (fixed args, fixed DOF/state layout). It
  has no room for matrix-free `Jv`, GPU memory layouts, extra outputs (energy/fluxes), or
  richer mixed/higher-order elements. We'd be designing every future capability to fit a
  lowest-common-denominator interface owned by someone else.
- **IP / commercial** — a CoupMech Lab *product* whose standalone engine *requires* a
  proprietary interface is tethered to Abaqus. Abaqus can also change the UEL ABI and we'd chase it.
- **Maintenance** — backend churn (Abaqus version N+1) would ripple into the runtime.

## The principle

**CoupFE must be complete and valuable WITHOUT Abaqus.** The *form* (the symbolic WeakForm/IR)
is the durable asset; the *runtime depends on a native element contract*; **Abaqus-UEL is ONE
export backend, not the master.** Dual-home stays a feature (emit a customer's element as an
Abaqus UEL — the embrace-and-extend entry product), but it must be a *value-add export*, never
the thing the standalone engine is built on.

## The architecture (the "different element interface")

```
            WeakForm / form  ──►  symbolic IR  (core/: tensor, weakform, symbolic_tangent)
                                   the asset; Abaqus-independent
                                          │  emit backends (share the IR)
        ┌─────────────────────────┬───────┴───────────┬───────────────────────────┐
        ▼                         ▼                    ▼                           ▼
  backend: native_f2py     backend: abaqus_uel   backend: (future) native_c   (future) cuda /
  CoupFE's own kernel       .for w/ UEL(...) ABI    / native_rust              matrix-free Jv
  signature  (the          for Abaqus customers
  standalone runtime)      (embrace-and-extend)
        │
        ▼
  CompiledElement / ElementGroup  ── depend ONLY on the NATIVE contract
```

## The native element contract (what the runtime owns)

The runtime already consumes a *native* batched kernel — `CompiledElement.element_rk_batch` +
`commit_group` — **not** the raw UEL. Make that the documented, primary ABI:

```
element_rk(coords, u, du, props, svars_old) -> (R, K, svars_new)     # batched over elements
  coords     (nelem, nne, ndim)        reference nodal coordinates
  u, du      (nelem, ndofel)           total dof + increment (node-major, the comps layout)
  props      (nprops,)                 material parameters
  svars_old  (nelem, nsvars)           committed state
  R          (nelem, ndofel)           residual (one source of truth)
  K          (nelem, ndofel, ndofel)   consistent tangent (complex-step inside the kernel)
  svars_new  (nelem, nsvars)           trial state (commit on accept)
```

It is **acceleration-agnostic** (the operator contract already is): the kernel may be emitted
as f2py-Fortran, C, Rust, or be plain numpy/numba — the runtime calls one contract. And it is
**extensible** in ways the UEL is not — optional outputs can be added without breaking callers:
`element_jv(coords,u,v,...) -> Jv` (matrix-free), energy/flux returns, per-field DOF blocks,
device/batched layouts.

**Where we are:** the IR (`core/`) is already backend-neutral, and the runtime contract
(`element_rk_batch`) is already native. The *only* coupling is that the generator currently
emits **abaqus_uel only**, and the standalone runtime borrows that `.for` via `drive_uel`. So
the architecture is ~80% there; what remains is to name the backends and add a native emit.

## The plan (defer-on-need; cheap now, build on need)

- **Phase A — now (cheap, mostly naming + docs).** Treat `uel_gen` as the **`abaqus_uel`
  backend** explicitly; document the native contract above as the runtime's element ABI. No
  behavior change. (This doc.) Keep emitting Abaqus-UEL; keep the runtime on it for now.
- **Phase B — on need (likely the first post-port codegen task).** Add a **`native_f2py`
  emit**: the same physics (from the shared IR) emitted as the `element_rk` subroutine
  directly, *without* the `UEL(...)` boilerplate + `drive_uel` indirection. Switch
  `CompiledElement` to consume it. Outcome: **the standalone path no longer contains the
  Abaqus UEL ABI** — Abaqus-UEL becomes purely an export. Small (the physics emit is shared;
  only the wrapper signature differs) and high-leverage (removes the coupling + a cleaner,
  leaner kernel).
- **Phase C — later (deferred, on concrete need).** Richer native backends as the existing
  deferred needs land: matrix-free `Jv`, GPU layout, Rust/C — none constrained by the UEL ABI.

## Sustainability guarantees this buys

- Standalone CoupFE **never requires** Abaqus-UEL → clean IP + technical independence.
- The **form/IR is the durable asset**; backends are swappable and isolated.
- Adding a backend (Abaqus vN+1, GPU, Rust) touches neither the IR nor the runtime contract.
- The **dual-home value is preserved** (`abaqus_uel` is a first-class backend) **without the
  dual-home coupling**.

## Decisions for Teng
1. Adopt this native-contract-first framing now (Phase A doc), Abaqus-UEL as a named backend? (rec: yes — it's just naming + docs.)
2. Schedule **Phase B (native_f2py emit)** as the first codegen task *after* the port + examples are green — or defer until a concrete UEL-can't-express need appears? (rec: do it soon; it's small and removes the coupling while the codegen is fresh.)
