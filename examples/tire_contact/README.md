# Tire contact and qualitative GetFEM comparison

**First-release status: RESEARCH workflow; not validation evidence pending an
exact external reference and a bounded full-solve gate.**

This directory is a layered research study:

- `mesh.py` builds and checks a structured hollow-torus Hex8 mesh.
- `run.py` solves a gravity-loaded, mixed-`u-p` tire on a rigid floor through
  smoothed barrier contact.
- `vonmises.py` independently reconstructs element von Mises stress.
- `analyze.py` writes a deformed legacy-VTK file and checks that stress
  concentrates near the contact region.
- `sensitivity.py` intentionally demonstrates that a static adjoint is invalid
  for the non-equilibrated dynamic-resting state; it is a negative diagnostic,
  not a validated sensitivity method.

From the repository root:

```bash
PYTHONPATH=. python examples/tire_contact/mesh.py
PYTHONPATH=. python examples/tire_contact/run.py
PYTHONPATH=. python examples/tire_contact/analyze.py
PYTHONPATH=. python examples/tire_contact/sensitivity.py
```

The element/contact runs require the compiled-element toolchain and can be
expensive. At present, `tests/test_tire_mesh.py` gates the mesh dimensions and
element validity. The full contact solve, von-Mises analysis, and negative
sensitivity diagnostic self-report their conclusions but do not have a bounded
pytest end-to-end gate.

## External-reference limit

The current code and build log call this a qualitative reproduction of the
GetFEM “tire under its own weight” example and identify it only by the name
“Khenous.” The repository does not yet contain a stable URL or full publication
entry, an upstream GetFEM version/commit, a precise reference parameter table, a
redistributable result record, or a retained comparison image/hash. No GetFEM
mesh, screenshot, or other third-party asset is distributed.

Consequently, the current checks establish internal mesh validity and expose
the layered CoupFE workflow. A completed self-reported run and qualitative
stress pattern do not establish a reproducible GetFEM parity benchmark.
Resolve the reference, retained result, and bounded-gate fields in
[`examples/REFERENCES.md`](../REFERENCES.md) before promoting this example from
research to the supported set. The full development record and honest negative
result are in
[`docs/dev/tire_buildlog.md`](../../docs/dev/tire_buildlog.md).
