# Verification-harness redesign — stop building a second assembler

**Status:** direction note (2026-06-22). Motivated by the Li 2026 5-field port. Scope: the codegen
verification harness (`coupfe/codegen/core/reference_assembly.py` and how elements are gated).
This is a deliberate codegen redesign, not a quick patch.

## What we found

Porting the Li 2026 element (5 fields: `u, phi, c, T, d`) surfaced an `IndexError` in
`reference_assembly`, originally labelled "indexing not generalized to 5 fields." That label is
**wrong** and the confusion is instructive:

- The **weak form** was fine, and the **generator** was fine — the generated Fortran produced
  **correct R and K** (verified to 1e-9 once the oracle was fixed). The generator differentiates
  generically (complex-step over DOFs), exactly like FFCx differentiates a UFL form.
- The gap was **only** in `reference_assembly` — our **verification oracle**, which is a *second,
  hand-written assembler*. It re-encodes every coupling pattern as a case table
  (`row_asm ∈ {value,grad}` × `col_asm ∈ {value,grad}` × test-field-kind × `wrt_kind ∈ {matrix,
  vector}`). The trigger was not a field *count* but a **weak-form term shape**: a value-assembled
  source term (the `T` equation's `pressure_storage`) that depends on the **gradient of another
  scalar field** (`grad_phi`, `grad_c`) — the `value × grad`, `wrt_kind='vector'` cell, which was
  never filled.

**Root cause:** we mixed physics into numerics *in the harness*. The oracle is a miniature FE
framework that must be taught every coupling by hand, so it falls behind real multiphysics weak
forms. FEniCSx has no such problem because it has **one generic assembler** (a term is
`∫ (op on test)·D·(op on trial)` with `op ∈ {value, grad, …}` applied uniformly to basis functions;
the Jacobian comes from symbolic differentiation). We have that genericity in the *generator*; we
accidentally lost it in the *harness*.

## Principle

**The harness must not be a second assembler.** Its job is to make the generated kernel
*trustworthy*, not to re-derive assembly per coupling pattern. Keep the genericity where it belongs
and gate by **independent generic checks + physical/mathematical properties**.

## Target architecture (separate the three concerns)

1. **Authoring** — the weak form is the single source of truth (user/AI writes it). Already true.
   The harness should *guide* authoring (next bullet), not re-implement it.

2. **Generation** — generic auto-diff translation to Fortran. Already mostly true. **Violation to
   fix:** the axisymmetric gel (3.4) hand-splices its hoop tangent into the generated Fortran after
   codegen — that re-introduces hand-written numerics and defeats the auto-tangent guarantee. Same
   disease, generation side.

3. **Verification** — split by *what* is checked; each piece generic or property-based, never a
   coupling table:
   - **Tangent (K) correctness → generic FD/CS consistency `K ≈ ∂R/∂U`.** Difference the *whole*
     element residual w.r.t. each DOF. Zero coupling knowledge, works for any number of fields, and
     it **subsumes the entire tangent-assembly table**. (It is also the right gate for the axisym
     hand-tangent — one generic check covers both.) Caveat: FD-consistency checks K *against R*, not
     R itself — so pair it with an R check below.
   - **Residual (R) / physics correctness → patch tests, method of manufactured solutions, known/
     analytical solutions, and/or a residual-only reference.** Residual assembly is *barely*
     combinatorial (`RHS += jR·dN` for fluxes, `RHS -= cdot·N` for storage) — it already worked at
     5 fields; only the tangent table exploded. So a residual-only reference is cheap and robust, and
     property tests (patch/MMS) catch translation errors the residual-ref can't.
   - **Well-posedness / authoring guidance → operator-level gates** (every field has an equation;
     expected coupling blocks are present; block definiteness where it should hold at `h ≲ l_eff`;
     unit/scale consistency). This is the harness "giving guidance on how to write the weak form,"
     and it is more discriminating than a consistency check — see
     `docs/lessons_learned` (operator-level gates beat consistency gates).

## Migration plan

1. Add a generic **`assert_tangent_consistency(kernel, state)`** helper (`K ≈ ∂R/∂U`, central
   difference, configurable tol). Wire it into every element test. *This is the single
   highest-value addition* and is element-agnostic.
2. Keep / factor a **residual-only reference** for R (drop the tangent half of `reference_assembly`).
3. Add **patch / MMS** gates for the representative elements (covers R-physics independently).
4. Once (1)+(2)+(3) cover an element, **retire that element's dependence on the tangent-assembly
   table**; when all elements are covered, delete the `{value,grad}²×kind×wrt_kind` if/elif tree.
5. **Interim tripwire (already in place):** `reference_assembly` now raises a clear
   `NotImplementedError("unhandled … pattern … use B-matrix assembly")` instead of silently skipping
   a block or cryptically `IndexError`-ing. This makes the table's incompleteness self-reporting
   until it is replaced.

## Why this is the right moat

CoupFE's value is "one weak form → UEL + standalone, **validated**." The validation must scale to
arbitrary coupled physics (the FEniCSx bar) and be *trustworthy*. A parallel assembler is neither —
it is a second framework with its own gaps. Generic FD-consistency + property gates + authoring
guidance scale to any number of fields and are easier to trust and maintain.

Related: the operator-level-gate, consistency-not-correctness, and
"representative coverage = least-verified coverage" lessons (2026-06-22).
