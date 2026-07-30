# CoupFE validation scaffold — test broadly, release narrowly

This directory is CoupFE's **internal confidence harness**. It is deliberately
**not** the public support boundary.

- `examples/` is the broader development inventory. The first-release decision
  is recorded in `examples/REFERENCES.md`: 21 directories have a limited
  `READY` role, 14 ship as explicitly unsupported `RESEARCH` capability demos,
  and 12 are `WITHHELD` pending source authority, licensing, or repair of a
  known release-gate failure.
- `validation/` holds implementation and numerical confidence checks. Their
  evidence can be analytic, published, cross-backend, or self-consistency based;
  those categories do not carry the same scientific weight.

This separation is the point. A package earns trust by being tested against far more
than it advertises; it stays clean by advertising only the parts that are polished.

The first-release sdist retains the capability-rich READY and RESEARCH example
set plus an explicitly reviewed 30-file public test partition. It omits the
remaining executable `validation/` harness and tests tied to WITHHELD,
private-input, expected-failure, MPI, or unqualified codegen paths. The artifact
guard's `PUBLIC_TEST_FILES` set is the authoritative allowlist.

The base public tier has no skips or expected failures:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_affine_constraints.py \
  tests/test_operator_contract.py \
  tests/test_mesh.py tests/test_distribute.py tests/test_dynamics.py \
  tests/test_all_primitive_barrier_2d.py \
  tests/test_contact_3d_persistent_friction.py \
  tests/test_contact_finite_sliding_repairing.py \
  tests/test_contact_persistent_friction.py \
  tests/test_contact_return_map_friction.py \
  tests/test_contact_search.py tests/test_multibody_contact.py \
  tests/test_tire_mesh.py
```

Fresh release-worktree result: **57 passed, 0 skipped, 0 xfailed**. Seventeen
additional allowlisted files gate Hertz, exact-stick, semismooth and
finite-sliding friction, the CoupFE side of the ppf interoperability recipe,
both 3-D contact examples, curved convergence, declarative model setup, the
compiled contact pipeline, four paper forms, four UMAT material/codegen
examples, and the example-local pasta mesh extractor. In the current
base-dependency environment, the exact 30-file partition reports **95 passed,
5 skipped**: the five module-level skips all state that SymPy is not installed.
Run the complete optional tier with its codegen dependencies and `gfortran` in
the final release environment before recording an all-green partition result.
The strict-xfailed Cattaneo and timed-out self-contact files are deliberately
absent; the full pasta-deck inventory is a separate opt-in check because the
large companion inputs are not bundled.

## How it works

Each model is a `ValidationModel`: a name, a one-line description, and a
zero-argument `check()` that builds a problem, solves it through
`coupfe.newton_solve`, compares a measured quantity with its stated oracle, and
returns a `ValidationResult`. Generated-kernel/reference-assembly and
native/UEL comparisons are implementation checks when both sides share the
same weak form; they are not independent physics validation.

Models self-register via `@register(...)` (see `validation/registry.py`). The registry
is enumerated by `tests/test_validation_models.py`, so every registered model is a
pytest case automatically.

```python
from validation import run_all, run_one, registered_models
run_all()                       # run every model's classified evidence check
run_one("neo_hookean_uniaxial") # run one
```

## Adding a model (the template)

1. Drop a module in `validation/models/` (copy `neo_hookean.py`).
2. Build the problem in the **example layer** (`examples/...`) and import it here —
   validation consumes per-problem glue, it does not duplicate meshing/BCs.
3. State the oracle category honestly and compare to it with a meaningful
   tolerance. Use an independent analytic or published reference when making a
   physics-validation claim.
4. `@register(...)` the `check()` and import the module in
   `validation/models/__init__.py`.

## Hooks / notes for the models still to be ported

The neo-Hookean entry is stateless and monolithic. The lab has harder models; the
scaffold is ready for them, with these seams already present:

- **Stateful elements (gel, J2 plasticity).** `CompiledElement` carries `svars` and
  `ElementGroup.commit()` calls `commit_group` after an accepted step (the
  transactional state protocol). A stateful model registers the same way but its
  oracle is usually a *path* quantity (a stress–strain curve, a swell ratio), so its
  `check()` will loop `newton_solve` over load steps, threading `state` between them.
- **Coupled / multi-field (u-µ, u-φ).** Use the `dof_per_node` + per-group `comps`
  pattern (`ElementGroup(..., comps=(0,1,2))`); add the coupled-field block-scale gate
  (`skills/pitfalls.md`) to the model's `check()` before the transient.
- **Staggered solves (phase-field fracture).** The monolithic `newton_solve` is the
  baseline; a staggered driver is per-problem glue that composes the same operators.
- **Cross-backend parity (Abaqus UEL).** Running the same generated `.for` in
  two hosts is a valuable ABI, sign, ordering, and state-transfer check. It is
  not an independent formulation oracle unless the external result was
  produced from an independently sourced model and its provenance is retained.

Each new model must ship its **broken control** (the test fails when a known bug is
reintroduced) — consistency (CS-vs-FD, compile, smoke) is not correctness.
