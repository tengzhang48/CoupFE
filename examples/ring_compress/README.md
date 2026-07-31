# Ring-compression research workflow

**First-release status: RESEARCH workflow; not validation evidence pending
reference-result provenance.**

This directory contains two manual CoupFE workflows for a two-dimensional
plane-strain neo-Hookean ring compressed between rigid plates:

- `reproduce.py` — quasistatic adaptive load stepping with penalty contact and
  return-map friction.
- `reproduce_dynamics.py` — staged ramp/hold/settle dynamic relaxation.

The scripts require a user-supplied `ring_compress.inp`. The proprietary input
deck is not distributed. Point the scripts and tests to a deck you are entitled
to use:

```bash
export COUPFE_RING_INP=/path/to/ring_compress.inp
PYTHONPATH=. python examples/ring_compress/reproduce.py
PYTHONPATH=. python examples/ring_compress/reproduce_dynamics.py penalty
```

An optional external-solver comparison must also be supplied by the user:

```bash
export COUPFE_RING_REFERENCE_CSV=/path/to/authorized/reaction_table.csv
```

The reaction table is deliberately not distributed because its per-file
redistribution authority is unresolved. Without that variable, the dynamic
workflow runs and reports no external comparison.

No retained release run establishes the external comparison. Historical
summary values are deliberately not shipped because their raw runs and exact
environments were not retained. Treat the scripts as manual RESEARCH workflows
and retain the authorized input, environment, and raw output for any new
comparison.

## What is and is not established

The drivers document the model choices and the important
Abaqus-time-versus-plate-displacement interpretation for an optional
user-supplied table, but the public record does not yet identify:

- the exact source or vendor identifier for `ring_compress.inp`;
- the Abaqus product release and platform;
- the input-deck hash and relevant solver/job settings;
- the output extraction command or script, run date, and raw output hash; or
- the per-file basis for redistributing the derived reaction table.

The workflow may ship to expose the contact/dynamic-relaxation path, but until
those fields are recorded, do not present an external comparison as an
independently reproducible public benchmark. The general policy that
proprietary decks are not redistributed is in [`NOTICE`](../../NOTICE), and the
cross-example release decision is in
[`examples/REFERENCES.md`](../REFERENCES.md). The dynamic-relaxation protocol and
its limitations are discussed in
[`docs/lessons_learned.md`](../../docs/lessons_learned.md).
