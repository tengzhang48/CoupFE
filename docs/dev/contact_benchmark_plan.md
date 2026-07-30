# Contact-benchmark validation plan — GetFEM tire + Kratos-family, with our edge

Draft for review (2026-06-28). Validate/showcase the consolidated 3D contact stack against
recognized open-source benchmarks (GetFEM, Kratos ContactStructuralMechanics). **Framing, stated
up front and honestly:** linear Hex8 (more stable for contact — no mid-side-node surface), **mixed
u-p, no F-bar**, validation-and-showcase **not** "beating GetFEM/Kratos"; analytic oracles where
they exist; and the **differentiable** angle as the thing the references structurally can't do.

**Current release boundary (2026-07-30):** this is a research plan, not a
completed validation record. Hertz is the passing public analytic example.
The Cattaneo driver is withheld after a strict-xfailed, non-converged result;
the tire ships as a RESEARCH workflow with only its mesh gate qualified. See
[`examples/REFERENCES.md`](../../examples/REFERENCES.md) for the authoritative
status and evidence gaps.

## Settled decisions
- **Element (near-incompressible rubber): Hex8 local-pressure (condensed-p) mixed u-p** (codegen,
  `uel_local_pressure` confirms Hex8; neo-Hookean / Mooney-Rivlin). Zero global pressure DOFs (u-only
  system). This is a RESEARCH option; current invariants do not establish
  general inf-sup, anti-locking, or inversion robustness.
- **NO F-bar in this research driver.** The current `(J̄/J)^{1/d}` path is
  undefined when centroid `J̄≤0`; the tire study selected local pressure rather
  than claiming either formulation is generally superior.
- **Linear Hex8, not Q2.** More stable in contact; the cost (a single linear layer can't bend) is
  paid with **a few layers through the tire wall** (GetFEM gets bending from one Q2 layer).
- **Small-strain rigid-contact benchmarks (Hertz, ν=0.3): plain Hex8** — no incompressibility, so no
  mixed u-p needed there.

## The ladder (analytic → reference → showcase → edge)
| # | Benchmark | Contact | Oracle | Status |
|---|---|---|---|---|
| 0 | Hex8 mixed-u-p implementation invariants | — | analytic uniform-state checks | internal development checks; formulation remains RESEARCH |
| 1 | **3D Hertz** sphere-on-Hex8-block | frictionless, small-strain | analytic `F(δ)=(4/3)E*√R·δ^{3/2}` (slope 3/2) | **DONE — `examples/hertz_contact`** (loose band; tighten to a convergence gate as polish) |
| 2 | **3D Cattaneo** sphere | friction partial-slip | analytic `c/a=(1−Q/μP)^{1/3}` (cube root) | **WITHHELD**; reviewed solve is non-converged |
| 3 | **GetFEM tire-inspired workflow** — rubber torus on ground, own weight | hyperelastic friction, 3D | no exact upstream source/result retained yet | RESEARCH; only mesh gate reviewed |
| 4 | **Differentiable sensitivity** of the tire | adjoint | grad-vs-FD `~1e-8`; **the edge** | honest negative; see `docs/dev/tire_buildlog.md` |
| 5 (stretch) | Hyperelastic **rubber-ring / tube** contact (Kratos family) | hyperelastic | self-consistency + Kratos ref if available | future |
| 6 (stretch) | **Self-contact** (folding/buckling) | self-contact | focused operator checks; future reference needed | partial operator implemented; end-to-end example withheld |

## Phases & parallelization
- **Phase 0 (foundation):** generate + verify the **Hex8 local-pressure mixed-u-p** element
  with uniform-state implementation invariants; general formulation
  qualification remains open. Everything rubber in this study depends on it.
- **Phase 1 (independent):** 3D Hertz (#1) **already exists** (`examples/hertz_contact`), so
  the only new piece was **3D Cattaneo (#2)** — the friction partial-slip cube-root stick radius —
  plus optionally **tightening `hertz_contact`'s loose band into a convergence gate**. Plain Hex8
  (ν=0.3, small strain), so it needs **nothing** from Phase 0.
- **Phase 2:** the **tire** — torus hex mesh → mixed-u-p rubber → rigid-ground barrier
  contact + friction + gravity on `solve_dynamics` (dynamic relaxation to quasistatic) → Von Mises
  recovery on the deformed tire. Mirror `examples/contact_3d_friction` (but rigid ground + torus +
  mixed-u-p, not F-bar blocks).
- **Phase 3:** the **differentiable** sensitivity (`d(peak-VonMises or contact-patch)/d(tire
  stiffness or μ)` via the adjoint) — the showcase the references can't match.
- **Phase 4 (stretch):** qualify the existing partial self-contact operator
  with a bounded end-to-end rubber-ring/folding case.

## 3D Cattaneo benchmark requirements (historical Phase 1)
- **Task:** `examples/cattaneo_3d/` — a sphere pressed (full normal load) then tangentially sheared,
  partial slip; gate the analytic **3D** stick radius `c/a=(1−Q/μP)^{1/3}` (cube root, *not* the 2D
  square root). Reuse the `examples/hertz_contact` Hex8 setup as the starting geometry. **3D Hertz
  (frictionless) already exists** — do not rebuild it; optionally tighten its loose prefactor band into
  a convergence gate (refine → prefactor approaches `(4/3)E*√R`) as a small polish.
- **Discipline (non-negotiable — `skills/testing.md`):** an **independent analytic oracle**, a
  **convergence gate** (error shrinks under refinement, not a loose band), **reject
  `converged=False`**, test a **mesh finer than the one you tuned to**, a **broken control**
  (`μ=0` ⇒ no stick zone). Extract the patch radius from the **pressure profile / contact set**, not
  `max|x|` (the brittle-extractor lesson). Report convergence before writing the assert.
- **Provenance:** no copyrighted reference files are committed; proprietary inputs remain
  user-supplied.

## Validation discipline (applies to every benchmark)
Analytic oracle first; **convergence** gates not bands; reject non-converged iterates; finer-than-
tuned mesh; **structural-zero** broken controls; separate the **rigorous** claim (self-consistent /
exact) from the **benchmark-match** claim (within a few % of a reference) — never call the second
exact; **compare like with like** (match the reference's BCs / material / assumptions).

## Honest caveats (state these in the writeups)
- Linear Hex8 differs from GetFEM's Q2/order-2 geometry; no reproduction claim
  is available without the exact upstream case and refinement evidence.
- Tire vs GetFEM remains an intended comparison until the upstream source,
  parameters, and reported results are retained.
- The target is a future validation and differentiable showcase, not a current
  release claim or a claim of superiority on forward contact
  (`docs/DESIGN.md`, Positioning).
