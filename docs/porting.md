# Porting guide — bringing `abaqus_ufl` models into CoupFE

This is a historical recipe for the porting workstream (bucket 2 in
`status.md`). Public release adds a stricter rule: do not copy source or
generated output from a private research repository unless the copyright owner,
license, adaptation lineage, and authority to redistribute are recorded.
Prefer an independent implementation from a citable public method source.
Passing implementation-consistency tests do not resolve provenance.

Maintainer source: the private `abaqus_ufl_lab` research repository. In internal
porting sessions, refer to its checkout as `$COUPFE_LAB_ROOT`; it is not a
runtime or public-install dependency. Names below are an internal inventory,
not a list of public-release examples; the authoritative per-example boundary
is [`examples/REFERENCES.md`](../examples/REFERENCES.md).

## Recipe: port one element into the validation registry

1. **Establish authority before touching code.** Record the public formulation
   source, original/generated-source ownership, applicable license, changes,
   and redistribution basis. Only then import an authorized self-contained
   kernel; otherwise reimplement from the public method description or keep the
   work private.
2. **Wrap it.** Build a `CompiledElement` and an `ElementGroup` (see
   `examples/neo_hookean_block/block.py`). Pass the element shape *explicitly*
   (`props`, `dof_per_node`, `n_svars`, `mcrd`) — no lab `WeakForm` introspection.
3. **State.** If the element is **history-free** (hyperelastic), `n_svars=0` and the
   seed driver is enough. If it is **stateful/coupled** (gel µ, plasticity `svars`,
   phase-field), it needs the per-operator committed-state path — coordinate with the
   B workstream before porting; do not fake it.
4. **Add a validation entry, with an INDEPENDENT oracle.** In
   `validation/models/<name>.py`, `@register` a function that builds a problem,
   solves through `coupfe.solve_increments`/`newton_solve`, and compares a measured
   quantity to a reference derived *independently* of the kernel (analytic solution,
   patch test, manufactured solution, or energy/work balance). Add a **broken
   control**. If two backends are generated from the same residual, test their
   ABI/result parity separately; that catches implementation drift but does not
   independently validate the physics.
   See `skills/testing.md` (non-negotiable) and `validation/models/neo_hookean.py`.
5. **Run it.** `PYTHONPATH=. pytest -q`. A static port that "looks right" is not
   done — the neo-Hookean port passed review but hit a runtime singular matrix.
6. **Curate.** Only promote a clean, illustrative few to public `examples/`; the rest
   stay in `validation/` (test broadly, release narrowly).

## Recipe: port a solver / harness module

- **Distributed solver** (`fe/petsc_backend.py`, `fe/petsc_mpi.py`): port
  `solve_steps_mpi_local` & friends as a PETSc-backed alternative to the seed
  scipy driver, behind the same operator-composition interface. Keep petsc4py-only
  (never import mpi4py). Validate with a 1-rank == N-rank differential test.
- **Harness** (`testing/`): the gates already operate on raw arrays, so they port
  almost verbatim — rename `abaqus_ufl.testing` → `coupfe.testing`, keep the broken
  controls in `tests/`.
- **Codegen** (`generators/`): the form→`.for` compiler is **separate, build-time,
  and NOT ported into the runtime** (see `status.md` → "The compiler is separate").
  For now use `abaqus_ufl` as the build-time generator and vendor its `.for` output
  into `coupfe/runtime/elements/`. The later **B** option — re-home it as a clean
  `coupfe-gen` package — is a *mechanical copy + rename* (not a rewrite): copy
  `abaqus_ufl/generators` + the `core` it depends on, rename `abaqus_ufl` →
  `coupfe_gen`, fix imports, green the existing test suite. Do **not** rewrite it.

## The element backlog (38 examples)

Group by what they exercise, so porting builds coverage deliberately:

- **Hyperelastic (history-free, easiest):** `neo_hookean_mixed`, `neo_hookean_umat`,
  `mooney_rivlin_umat`, `Fbar_uel`, `visco_hyperelastic_umat`.
- **Plasticity (stateful `svars`):** `J2_FeFp`, `small_strain_j2_umat`,
  `anand_2025_rock`/`Anand_2025`, `small_strain_crystal_plasticity_umat`,
  `small_strain_drucker_prager_umat`, `*_mcc_umat`, `small_strain_norsand_umat`,
  `zhang_soga_2025`, `strain_gradient_plasticity_msg`, `small_strain_smp_umat`.
- **Gels (coupled u-µ, stateful):** `gel_chester_anand`, `gel_three_field`,
  `simple_gel_quad4`.
- **Phase-field / damage (staggered):** `phasefield_fracture_uel`,
  `phasefield_corrosion_cui`.
- **Coupled multiphysics:** `MRE` (u-A magneto), `Hussein_2026` (u-c-φ hydrogen),
  `Wang_2026` (lymph node), `thermo_mechanics_quad8`, `Li_2026_battery`,
  `scalar_diffusion_uel`, `LCE`.
- **Other / scaffolds:** `Ukidwe_2023`, `Jiao_2026`, `Xue_2025`,
  `uel_scaffold_quad4`, `uel_scaffold_mixed_quad8`.

Suggested order: hyperelastic first (proves the path, no state), then one stateful
plasticity and one coupled gel (these force the state protocol + field-wise
convergence to be correct), then breadth. Each gel/coupled port should also wire the
`assert_coupled_field_scale_balance` gate (catches the scale disparity before a
multi-day under-resolution hunt — see `skills/pitfalls.md`).

### `*_umat` vs `*_uel` — CoupFE has no UMAT *host* yet (OK for now)
A `*_uel` example is a *full element* (geometry + quadrature + physics) — `ElementGroup`
drives it directly. A `*_umat` example is only a *material* (PK1/Cauchy + tangent at a
point). CoupFE drives full elements; it does **not** yet have a generic **UMAT host**
(a standard isoparametric element that loops Gauss points, calls a material, and
assembles). So a `*_umat` model is ported by either:
- **(a) generate it as a UEL** via the codegen (material baked into the element) — the
  path for now; or
- **(b)** wait for the planned generic **material-element operator** — a standard
  Quad4/Hex8 that takes a material *function* `F → PK1` and complex-steps the element
  (the "materials are functions in operators" design made concrete; no Abaqus UMAT ABI).

So the `*_umat` rows in the backlog (j2_umat, mcc, drucker_prager, norsand, …) wait on
(a) or the material-element operator; start the porting with the `*_uel` / coupled
elements (`Fbar_uel`, gels, phase-field, MRE, …).
