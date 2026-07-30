# Upstream assessment — CoupFE-EDA & CoupFE-cardiac vs the core (2026-06-28)

Decision record. Question: *which of the updates in the application repos should come into the
CoupFE core (`coupfe/`)?* Lens: **keep the core lightweight + robust** — only correctness/robustness
fixes or genuinely reusable, domain-agnostic capability go up; everything else stays in the app.

**Conclusion up front: almost nothing comes up. The real cleanup runs the *other* direction —
update the apps to depend on the lean core.**

## CoupFE-cardiac-elastodynamics → core: NOTHING to upstream
Cardiac carries a **forked `coupfe/` copy** that is **purely behind** main, not divergent:
- Only 4 files differ (`codegen/core/defs.py`, `material.py`, `generators/uel_gen.py`,
  `generators/uel_local_pressure.py`). For 3 of them cardiac adds **zero** lines (it only *lacks*
  main's changes). The shared commit SHAs are identical (`65eaf50`, `ef22493`, `a184a05`) → shared
  history, cardiac simply older.
- `uel_local_pressure.py` is **byte-identical to main's `65eaf50`** version. The large diff is
  main's later refactor (`5295b36` generalize-to-pure-mechanical-u-p, `d203694` J_inel hook), which
  cardiac never had. The inspected cardiac tree contains no HO-generator change.
- No cardiac `coupfe/` file is **ahead** of main anywhere.

⇒ **The earlier "cardiac's HO local-pressure overlaps my J_inel work" conflict worry is resolved:**
cardiac never touched the generator; there is no 3-way conflict.

**Downstream action:** **de-fork cardiac** — make it `pip install "coupfe @ …"` and delete its
`coupfe/` copy (the EDA pattern). That ends the codegen-divergence drift permanently. Cardiac's
genuine value is *app* code and stays in cardiac: the Holzapfel-Ogden u-p element, the Case-B
epicardial-spring fix (`A_EPI`), the cardiac benchmark, `AGENTS.md` (its dev-discipline doc could
optionally fold into the core `skills/`, but that's minor).

## CoupFE-EDA → core: nothing now (one idea to remember)
EDA is already the **correct pattern**: a standalone `coupfe-eda` package that **depends on** the
core (`pip install "coupfe @ …"`), no forked `coupfe/`.

Its `coupled_newton` (`eda_multiphysics/_coupled_solve.py`, 48 lines) is **redundant** with the
core's `newton_solve` (`coupfe/assembly/assemble.py`), which is strictly more capable:

| | `newton_solve` (core) | `coupled_newton` (EDA) |
|---|---|---|
| convergence | relative residual `‖R‖<rtol·‖R₀‖` | increment `‖dU‖∞<tol` |
| line search | ✅ CCD-bounded backtracking | ❌ |
| load-step / dynamics | ✅ `solve_increments`/`solve_dynamics` | ❌ |
| multi-field | ✅ composes *any* operators | ✅ groups summed |
| linear backend | **hard-coded** direct | **pluggable** `linsolve(K,R,drows)` |

- **Do NOT upstream `coupled_newton`** — a second, weaker Newton loop in the core is exactly the
  bloat to avoid.
- **The one reusable idea = the pluggable linear backend.** It is the enabler for the core's
  *tracked-not-built* coupled-distributed / FieldSplit gap. **When** that work is taken up, add an
  **optional `linsolve=` callback to the existing `newton_solve`** (default = current behavior) — a
  ~5-line generalization of the *better* driver, not a new one. Not before (YAGNI).
- **Tet4 is not implemented** in the inspected application tree — not a candidate.

**Downstream action:** eventually **consolidate EDA onto `newton_solve`** (retire `coupled_newton`)
to end the dedup; otherwise EDA stays as-is.

## The architecture principle (restated)
Applications are **separate repos that depend on the `coupfe` package**; they must **not fork
`coupfe/`**. The core owns the reusable substrate + the validation harness; the apps own their
domain physics/benchmarks. EDA already follows this; cardiac is the anti-pattern to fix. See
`docs/DESIGN.md` (Positioning).
