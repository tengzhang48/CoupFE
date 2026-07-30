# CoupFE

A small, well-tested **finite-element scaffold** — not a general PDE platform.

The open FE frameworks are large because they own meshing, boundary conditions,
time integration, and I/O — exactly the tedious, per-problem glue that is now
quick to (re)write. CoupFE owns only the **correctness-critical core** and a
**validation harness** that makes the generated/written glue trustworthy.

## The idea

> **One element definition → two homes.** A single weak-form/residual definition
> compiles to a complex-step kernel that runs **(i) inside Abaqus as a UEL** and
> **(ii) standalone here** on PETSc/MPI — the same kernel in two ABIs.

So you can deliver a custom element to a client who keeps using Abaqus, and run
the identical element standalone at scale when they want to grow beyond it.
Backend parity catches code-generation/ABI drift; confidence in the model still
depends on its independent physical evidence.

## The spine

Every physical contribution to the global nonlinear system — a bulk element
group, a contact set, or a load — is an **`Operator`** with one contract:
`residual`, `tangent`, `commit` (see `coupfe/operators/base.py`). The driver
(`coupfe/assembly/assemble.py`) composes operators and knows nothing about
elements, materials, or contact. Exact affine constraints use a separate,
mesh-agnostic `U=Pq+U0` transform; mesh matching and application policy stay in
the consuming package.

Two principles do a lot of work:

- **One residual is the source of truth.** The tangent is *derived* from the
  residual by **complex step** (exact, no hand-coded stiffness). See
  `complex_step_tangent`.
- **Operators are pure and never commit state implicitly** — what makes
  complex-step columns, line searches, and matrix-free products safe.

## Quick look

> **Environments / PETSc:** use one consistent conda-forge PETSc/MPI stack for
> PETSc/MPI/f2py work; do not mix pip-built PETSc with conda MPI libraries.
> See the portable setup in [`docs/install.md`](docs/install.md).

```bash
pip install -e ".[dev,runtime,codegen,performance]"
python -m pytest -q -m "not slow"          # required serial/codegen/compiled/mesh gates
python examples/linear_bar/run.py          # a nonlinear bar through the contract (pure Python)
python examples/neo_hookean_block/run.py   # a compiled neo-Hookean element through the contract
python examples/curved_annulus/run.py      # curved-boundary convergence (~h²) on a refined mesh
python examples/hertz_contact/run.py       # quantitative normal-contact gate
```

The public source also keeps four scoped **RESEARCH** forms from the
`abaqus_ufl` paper instead of reducing the release to toy elements:
phase-field corrosion (Quad8R), mixed `u-p-mu` gel (Quad8), stabilized
`u-theta` elasticity (Tet4), and pressure-gel morphing (local-pressure Hex8).
Each directory regenerates a UEL from `coupfe.codegen` and states the precise
boundary between current implementation gates and historical Abaqus/paper
evidence. A separate four-example material gallery exercises the finite-strain
and small-strain UMAT backends with Neo-Hookean, Ogden, J2 plasticity, and
standard-linear-solid viscoelasticity declarations. Those examples use
independent material-point oracles plus regeneration and compiler gates; they
do not claim standalone structural solves.

No mesh-software dependency or installed mesh adapter was added to Core. The
morphing directory contains only a read-only, example-local extractor for the
U3 and companion C3D8 blocks in a user-supplied pasta deck; it deliberately
does not translate Abaqus contact or analysis-step semantics. Full evidence
bundles remain in the pinned public companion repository.

The source archive carries a curated 30-file public test partition. Its
base-dependency tier currently passes 57 tests without skips. In the current
base environment, the entire partition reports 95 passed and five module-level
skips because SymPy is not installed; the complete optional-dependency tier
must be rerun in the final release environment. The larger development checkout
has additional private, expected-failure, MPI, and provenance-blocked gates; see
[`validation/README.md`](validation/README.md) for the exact boundary.

The MPI smoke tests additionally require the matched conda-forge PETSc/MPI
environment described in [`docs/install.md`](docs/install.md):

```bash
mpirun -n 4 python examples/mpi_smoke/distributed_solve.py
```

Opt-in long reproductions use `python -m pytest -q -ra -m slow`. The
self-contact hairpin E2E case is excluded from the first public example set
because its accelerated subprocess exceeds the 600-second gate; smaller
incident-exclusion and contact-operator tests remain useful development
evidence.

The bar (`examples/linear_bar/`) is the smallest operator (one residual,
complex-step tangent); the neo-Hookean block runs a real compiled element; the
curved annulus shows refinement + geometry re-embedding converging at the
Quad4 rate. The MPI smoke is a rerunnable harness for a memory-local
distributed solve and serial-vs-rank comparison; its final-revision retained
MPI rerun is still pending.
The contact examples intentionally remain prominent: Hertz normal contact,
finite-sliding and capstan friction, exact-stick and semismooth friction, and
the passing 3-D collision/friction drivers show distinct parts of the contact
stack rather than one showcase script standing in for the whole capability.

The [`examples/README.md`](examples/README.md) index gives the correct entry
point and test status for every example family. Check the accompanying
[`examples/REFERENCES.md`](examples/REFERENCES.md) ledger before treating a
research study, backend-parity check, or external-solver comparison as an
independently validated result.

Development knowledge ships with the code in `skills/` (SKILL, pitfalls, testing,
distributed) — the codified "how to build this correctly" that, with the harness, makes
AI-assisted custom development trustworthy.

## Status

The operator spine, build-time form→Fortran compiler, compiled-element runtime,
validation harness, regular-mesh path, PETSc/MPI assembly, and serial and
distributed contact lines are present and tested. Development remains active;
the most precise support/limitation inventory is
[`docs/capabilities.md`](docs/capabilities.md). The dated planning and research
record remains in `docs/status.md`, `docs/roadmap.md`, `docs/lessons_learned.md`,
and `docs/dev/`.

Part of **CoupMech Lab** (Coupled Mechanics Lab).

## License

Source code, generated source, examples, and configuration are licensed under
the [Apache License 2.0](LICENSE), except for the identified paper-example
ports retained under their
[MIT license](LICENSE-ABAQUS-UFL-EXAMPLES). Documentation prose and figures
are licensed under
[Creative Commons Attribution 4.0 International](LICENSE-DOCS.md); code
snippets embedded in the documentation may also be used under Apache-2.0. See
[NOTICE](NOTICE) for attribution, third-party provenance, and trademark notes.

Supported capabilities and known limitations are maintained in
[`docs/capabilities.md`](docs/capabilities.md).
