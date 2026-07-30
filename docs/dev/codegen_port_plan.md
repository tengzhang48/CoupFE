# UEL/UMAT generator port plan for CoupFE (option B)

> **STATUS 2026-06-21 — scope corrected + Phase 1 DONE on `main`.**
> - **Scope = UEL-focused.** "We only port UEL" canNOT mean deleting `umat_gen.py`:
>   `uel_gen` imports the shared Python→Fortran engine (`FortranTranslator`, `_tensorize_expr`,
>   `_get_source_body`) **from** `umat_gen.py`. So **`umat_gen.py` is KEPT** (the engine);
>   **`uinter_gen.py` is DROPPED** (UEL doesn't depend on it). Validate/release the **UEL** path;
>   the UMAT material zoo is out of scope for now.
> - **Phase 1 done:** mechanical copy + namespace rename, `uinter_gen` removed +
>   `__init__` trimmed, the 3 fixes applied (`material.py` verify-import; `umat_gen.py` two
>   `/mnt/project/...` fallbacks → `FileNotFoundError`; templates shipped), `pyproject` updated
>   (`codegen=["sympy"]`, template package-data, `OperatorSignWarning`→error). **`import
>   coupfe.codegen` works; import-hygiene holds (runtime stays codegen/sympy-free); the
>   generator SMOKE produces a valid neo-Hookean Quad4 UEL.** Tests:
>   `tests/test_codegen_import_hygiene.py`, `tests/test_codegen_smoke.py`.
> - **Remaining work (now unblocked — the package imports):** Phase 2 (port the UEL test
>   suite), **Phase 3 dual-home proof** (regenerate `neo_hookean_q4.for` == vendored → compile
>   via `build_element_kernel` → run in CoupFE), Phase 4 capability sweep, Phase 5 bug-audit.
> - **Workflow:** codegen and contact development used separate worktrees to avoid
>   shared-checkout collisions.

**Goal.** Re-home the form→`.for` generator from `abaqus_ufl_lab` into CoupFE as a
**separate, build-time-only** package `coupfe/codegen/`, preserving its validated behavior
**exactly**, and re-verifying the known codegen bugs + the dual-home proof along the way.

**This is a MECHANICAL port (decision B), NOT a rewrite.** The generator is the most
pitfall-dense code in the project and every line is validated against feacheap/Abaqus. Copy
verbatim → rename the namespace → fix imports → run the *existing* tests. **Do not refactor,
"clean up", or re-derive anything.** Any behavior change is a regression until proven
otherwise. If you think something is a bug, do NOT silently fix it — log it in the bug-audit
(Phase 5) with evidence and an independent oracle.

---

## Non-negotiable guardrails

1. **Runtime stays codegen-free.** `coupfe/{runtime,operators,assembly,mesh,model,materials}`
   must NEVER import `coupfe.codegen`. The runtime *consumes* a compiled `.for`; it never
   *generates* one. Enforce with an import-hygiene test (Phase 1 gate).
2. **Deps: numpy + sympy only.** The codegen core imports `numpy` (11×) and `sympy` (1×) and
   nothing else heavy. Add `sympy` as a **codegen-only** optional dependency
   (`[project.optional-dependencies] codegen = ["sympy"]`), NOT a runtime dep.
3. **feacheap is the ORACLE, not a dependency.** feacheap appears only in the *verification*
   layer (`testing/finite_strain.py`, `testing/element_convergence.py`) and one generator
   variant (`generators/uel_local_pressure.py`). Port the core WITHOUT feacheap; gate any
   feacheap-using check behind `pytest.importorskip`/try-import so it skips cleanly when
   feacheap is absent.
4. **Output must stay compatible with `coupfe/runtime/drive_uel.f90`** (`drive_uel` +
   `drive_uel_batch`). The vendored `coupfe/runtime/elements/neo_hookean_q4.for` was produced
   by *this* generator — regenerating it must reproduce a functionally identical file
   (Phase 3 gate).
5. **Verify by RUNNING, never by static review.** A ported file that imports is not done;
   a generated `.for` that *looks* right is not done — compile it and run it through CoupFE.

---

## Landing structure (DECIDED: subpackage `coupfe/codegen/`, Teng 2026-06-21)

```
coupfe/codegen/                 # build-time only; runtime never imports this
  __init__.py                   # exposes generate_uel, generate_umat, WeakForm, ...
  core/                         # <- abaqus_ufl/core/   (symbolic form, tangent, verify)
  generators/                   # <- abaqus_ufl/generators/ (the .for emitters + templates/)
  testing/                      # <- abaqus_ufl/testing/ (the validation harness)
```
**Decided:** a **subpackage** `coupfe/codegen/` (one repo, strict import isolation) — simpler,
and because the namespace is fully isolated it can be split out to a standalone `coupfe-gen`
repo later by a pure move if ever wanted. Codegen and contact work use separate worktrees
when developed in parallel to avoid shared-checkout collisions.

---

## Source inventory (from `abaqus_ufl/`, ~14.7k LOC)

**core/ (port all — the symbolic engine):** `tensor.py` (558), `weakform.py` (558),
`symbolic_tangent.py` (735), `material.py` (352), `fields.py` (107), `defs.py` (80),
`fortran_helper.py` (61), `tagent.py` (170), `verify.py` (425), `reference_assembly.py` (811,
the Python reference assembler — the *independent* in-process oracle), plus the small
material helpers (`soil.py`, `small_strain_plasticity.py`, `_cs_state.py`).

**generators/ (port all):** `uel_gen.py` (2462, **main** — `generate_uel`), `umat_gen.py`
(3227, `generate_umat`), `uel_fbar_coupled.py` (1112), `uel_local_pressure.py` (634, has a
feacheap path → isolate), `uel_magneto.py` (651), `uinter_gen.py` (1126), `inp_scaffold.py`
(1113), `element_config.py` (315, `ELEMENT_CONFIGS`), `_fortran_format.py` (88, fixed-format
line wrap), **and `generators/templates/`** (the `.for` templates read by `_read_template`).

**testing/ (port — the gates):** `finite_strain.py`, `element_convergence.py`, `invariants.py`,
`manifest.py`, `objectivity.py`, `operators.py`, `paths.py`, `_util.py`. (NOTE: CoupFE already
has a `validation/` seed and may already carry parts of this from the review program — *check
first* and reconcile, don't duplicate.)

**Entry point:** `generate_uel(weakform, output_path, element='Quad8', mat_prefix=None,
fbar=None, element_config=None, formulation=None)` → writes a `.for`.

---

## Phases (commit per phase; each has a GATE that must pass before the next)

### Phase 0 — Inventory & scaffold
- Reconcile with what CoupFE already has (`coupfe/validation`, `coupfe/runtime/elements`).
- Create `coupfe/codegen/{core,generators,testing}/` skeleton + the `codegen` optional dep.
- Produce a file-by-file port table (port / port-feacheap-optional / skip / already-present).
- **Gate:** the table is reviewed; nothing is silently dropped.

### Phase 1 — Mechanical copy + namespace rename
- Copy files verbatim; rename `abaqus_ufl.core`→`coupfe.codegen.core`,
  `abaqus_ufl.generators`→`coupfe.codegen.generators`, `abaqus_ufl.testing`→
  `coupfe.codegen.testing` (mechanical find/replace). Copy `generators/templates/` too.
- Fix only what the rename requires (imports, template path resolution). **Touch no logic.**
- Add the **import-hygiene test**: in a fresh process, `import coupfe` then assert
  `coupfe.codegen` (and `sympy`) are absent from `sys.modules`.
- **Gate:** `import coupfe.codegen` works; `import coupfe` does NOT pull in codegen/sympy.

### Phase 2 — Port the generator test suite (the real gate)
- Port the lab tests that exercise the generator and run them against the ported code:
  `test_uel_gen.py`, `test_weakform.py`, `test_symbolic_tangent.py`, `test_verify_raises.py`,
  `test_reference_assembly.py`, `test_uel_helper_codegen.py`, `test_generated_uel_compile.py`,
  `test_f2py_uel_smoke.py`, `test_umat_gen.py`, `test_generated_umat_compile.py`,
  `test_f2py_umat_smoke.py`, `test_quad8r_config.py`, `test_fbar_verification.py`,
  `test_inp_scaffold.py`, `test_uinter_gen.py`, `test_uel_magneto.py`,
  `test_multi_scalar_transport.py`. (f2py/feacheap ones skip cleanly when the toolchain/oracle
  is absent.)
- **Gate:** the ported generator passes the SAME tests it passes in the lab (run both, diff the
  pass/skip sets). A test that passes in the lab but fails/skips here is an unfinished port.

### Phase 3 — Dual-home proof (headline gate)
- Regenerate `neo_hookean_q4.for` with the ported `generate_uel` → **diff against the vendored
  `coupfe/runtime/elements/neo_hookean_q4.for`** (functional identity; ideally byte-identical).
- Compile it via `coupfe.runtime.build_element_kernel`, wrap in an `ElementGroup`, and confirm
  it reproduces the existing neo-Hookean element behavior (the `CompiledElement`/pipeline tests
  still pass against the *regenerated* kernel).
- Generate ONE more element end-to-end (a second from the zoo, e.g. a coupled u–scalar one),
  compile in CoupFE, and verify it with `reference_assembly` + the CS-vs-FD tangent.
- **Gate:** one definition → a `.for` that (a) matches the vendored element and (b) runs
  standalone in CoupFE. This is the "embrace-and-extend Abaqus" claim, made concrete.

### Phase 4 — Generator capability sweep (breadth, lower risk)
- Drive the ported generator over a slice of the 38-element zoo (`docs/porting.md`): for each,
  generate → compile → verify (reference_assembly / CS-vs-FD) → register in `coupfe/validation`.
- **Gate:** N elements generated + verified + registered (start small, e.g. 5, then scale).

### Phase 5 — Bug-audit ("double-check the bugs" — do NOT skip)
See the checklist below. For EACH known issue: locate the site in the ported code, **determine
its CURRENT status (fixed / open) by test, not by memory**, ensure the ported code carries the
fix, and add a regression test with an **independent oracle + a broken control** (a test that
fails if the bug returns). Report a status table.

### Phase 6 — Docs
- `coupfe/codegen/README.md` (build-time-only, the form→.for recipe, deps).
- Update `docs/porting.md`, `docs/DESIGN.md` (the compiler is separate), and
  the current release/capability documentation.
- A "write a new element" walkthrough (define form → `generate_uel` → compile → verify).

---

## Bug-audit checklist (Phase 5) — verify status, don't assume

These were found earlier; some are fixed, some may be open. **Test each; report fixed/open.**

| # | Issue | Where to look | How to verify (independent oracle / broken control) |
|---|-------|---------------|------------------------------------------------------|
| 1 | **helper-codegen NameError** (`self._helper()` in a UEL) | `uel_gen.py` helper emission; `test_uel_helper_codegen.py` | carry that test; assert a helper-using form compiles + runs |
| 2 | **phase_flux sign trap** (μ-def needs `+κ∇ξ`; 5 sites) | flux assembly in `uel_gen`/`weakform`; `OperatorSignWarning` | carry the warning→**error** promotion (lab `pyproject`); coupled-dispersion / block-definiteness gate at h≲l_eff |
| 3 | **`z**2` NaN-at-0** (power expr evaluated at 0 in codegen) | `symbolic_tangent.py` / complex-step emit | generate a form with `z**2` (z→0), compile, eval at 0 → finite (not NaN) |
| 4 | **stale-deck-props NaN** | `inp_scaffold.py` / prop wiring | regenerate a deck, run, assert no NaN from unset props |
| 5 | **Sylvester `V.T` vs `inv(V)`** (eigenprojection) | `tensor.py` spectral / `umat_gen` | quantitative oracle vs analytic eigendecomposition (NOT code-vs-itself) |
| 6 | **single-interval yield range / `P_ys` 10×** (zhang_soga) | `examples/zhang_soga_2025` codegen path | independent stress oracle over the FULL path (not elastic-only `verify()`) |
| 7 | **`eps_r` zero-trap** (zhang_soga) | same | path with `eps_r→0`, assert finite |
| 8 | **anand_rock 4 paper-deviations** (Fp_old push-forward; slip-rate numerator `τ−β·σ` vs `τ`; D_duct exponent 1 vs 2; ζ_d 1e8 vs 1e6) | `umat_gen` rock material | compare to the **paper** equations + a non-axisymmetric, non-qualitative path |
| 9 | **eig33z** Fortran port (H2) | `core/fortran_helper.py` / emitted eig | quantitative vs numpy eig on random SPD + repeated-eigenvalue cases |
| 10 | **translator idiom limits** (NOT bugs — constraints) | translator | document: `sym` unsupported (use `0.5*(F+F.T)`); a scalar field needing its own `(f−f_old)/dt` history must map to the `p` slot |

**Audit lens — the 6 generation failure modes** (apply when reading each emitter): pattern-matched
duals, coaxial algebra, constraint-as-definition, param cross-contamination, symbol reuse, path
drift. **And the 4 test-blindnesses**: consistency≠correctness (CS-vs-FD can't see a convention
error), code-vs-itself, qualitative-only, benign-path. → Every gate needs an **independent
quantitative oracle** (reference_assembly, analytic, or feacheap/Abaqus) **and a broken control**.

---

## Final acceptance gates (all must pass)
1. Runtime import hygiene: `import coupfe` pulls in neither `coupfe.codegen` nor `sympy`.
2. Ported generator passes the lab's generator test set (same pass/skip profile).
3. Dual-home: regenerated `neo_hookean_q4.for` == vendored (functional) + compiles + runs in CoupFE.
4. ≥1 *additional* element generated → compiled → verified standalone in CoupFE.
5. Bug-audit table complete: every item marked fixed/open, fixes carried, each with a regression test.

## Don'ts
- Don't refactor or "improve" the generator while porting (option B = copy+rename).
- Don't let the runtime import codegen or sympy.
- Don't fix a suspected bug silently — log it with an oracle in the audit.
- Don't claim a port done from a passing *import* or a *plausible-looking* `.for` — compile and run.
- Don't gate correctness on consistency checks (CS-vs-FD) alone — use an independent oracle.
