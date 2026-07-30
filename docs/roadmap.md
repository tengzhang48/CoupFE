# CoupFE roadmap

> **Historical roadmap.** Completed and superseded steps are intentionally
> retained as engineering provenance. See [`capabilities.md`](capabilities.md)
> for the current inventory and known limitations.

`docs/standalone_gpu_plan.md` is the full engineering plan (Revision 3.0 is the
current, reality-aligned direction). This file is the short near-term sequence.

## Now — seed (this commit)
- The operator contract (`coupfe/operators/base.py`) + the composing driver
  (`coupfe/assembly/assemble.py`).
- A working, tested vertical slice (`examples/linear_bar/`, `tests/`).
- Codified development knowledge (`skills/`) — first-class, evolves with the code.

## Near-term workstreams (each builds against the operator contract)
- **B — Operator contract + compositional groups.** Generalize beyond the bar:
  field/DOF masks per group, per-group state, interfaces (tie/cohesive/contact)
  as operators between groups. Migrate the batched f2py element runtime
  (`CompiledAbaqusElement`/`drive_uel`) into an `ElementGroup` operator.
  - *Done (first slice):* the batched f2py runtime is ported clean-room as
    `coupfe.runtime.CompiledElement` + `build_element_kernel` (no `abaqus_ufl`
    import), and `coupfe.operators.ElementGroup` wraps it on the contract with the
    uniform-DOF + per-group `comps` mask (multi-material composition).  One real
    compiled element runs end-to-end: a neo-Hookean Quad4 block
    (`examples/neo_hookean_block/`) solved through `newton_solve`.
- **P — Model-setup pipeline.** The concise, AI-targetable front door: declare
  groups+materials, BCs, loading, time integration, output. The thing AI writes
  glue against; the harness validates it.
- **C — Contact operator.** Rigid analytical SDF (Stage-1, port from RetroMech)
  into the contract → deformable point-to-surface (learn from ppf-contact-solver).
- **D — Validation harness.** Migrate the backend-agnostic gates (block-definite,
  diffusive-flux, coupled-scale balance, field-wise convergence) and the
  `RegimeManifest`; wire cross-backend (Abaqus) parity.
  - *Seeded:* `validation/` is the internal "test broadly, release narrowly"
    registry (separate from the curated `examples/`); each model pairs a problem
    with an independent oracle and runs under `tests/test_validation_models.py`.
    The neo-Hookean model is wired; hooks/notes for gel, plasticity, phase-field,
    and Abaqus-parity are in `validation/README.md`.

Migrate from the research lab deliberately (clean, tested) — do not fork it.

## Later (gated)
- The form→`.for` codegen + the Abaqus-UEL emitter (dual-backend from one
  definition).
- Matrix-free `Jv` + GPU — only when a problem outgrows assembled/direct memory
  *and* has a known good preconditioner.
- Geometry-aware distributed meshing to ~10M DOF.
