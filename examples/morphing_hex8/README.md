# Pressure-gel morphing Hex8 — RESEARCH

This example exposes CoupFE's three-dimensional coupled-field code-generation
path: a Hex8 element with global displacement `u` and chemical potential
`mu`, plus one element-local pressure `p` stored in `SVARS(1)` and statically
condensed. Run it from the repository root:

```bash
PYTHONPATH=. python examples/morphing_hex8/build.py
```

That command verifies the material response and regenerates
`pressuregel_local_pressure_hex8.for`. It does not run Abaqus or reproduce the
paper's boundary-value problem.

## Optional Abaqus mesh extraction

`abaqus_mesh.py` is a narrow, example-local reader for a user-supplied copy of
the public pasta `.inp` deck:

```bash
PYTHONPATH=. python examples/morphing_hex8/abaqus_mesh.py \
  /path/to/Pasta_W15_G5_H20_B10_L127_T50_M10_V_abaqus_ufl_hex8.inp
```

It preserves Abaqus node and element labels, reads the U3 analysis
connectivity, resolves the bare C3D8 companion-connectivity include, and reads
numeric named NSET/ELSET blocks (inline or `GENERATE`). U3 and C3D8 mappings
are exposed separately as `mesh.uel_elements` and
`mesh.companion_elements`; no CoupFE renumbering is implied.

This is intentionally not a general Abaqus parser or an installed CoupFE API.
Part/instance/assembly scoping, transformed coordinates, parameterized mesh
records, set-name composition, element continuations, and element families
outside U3/C3D8/C3D8T fail explicitly. Material-property includes, solver
steps, contact, and output requests are not interpreted. The large deck and
connectivity include remain in the separately versioned public `abaqus_ufl`
repository and are not copied here.

The synthetic parser tests run without an external deck. A read-only inventory
check of the full public deck is opt-in:

```bash
COUPFE_MORPHING_ABAQUS_INP=/path/to/main-pasta-deck.inp \
  PYTHONPATH=. pytest -q tests/test_morphing_hex8_inp.py
```

## Scientific and source lineage

The pressure-based gel formulation, reparameterization, grooved-sheet problem,
and solvent exposure follow Ye Tao et al., “Morphing pasta and beyond,”
*Science Advances* 7 (2021), eabf4098,
[doi:10.1126/sciadv.abf4098](https://doi.org/10.1126/sciadv.abf4098).
The broader coupled diffusion-deformation gel lineage is Shawn A. Chester,
Claudio V. Di Leo, and Lallit Anand,
[doi:10.1016/j.ijsolstr.2014.08.015](https://doi.org/10.1016/j.ijsolstr.2014.08.015).
This generated pressure-based element is a later project implementation; it is
not a redistribution of the Chester–Di Leo–Anand supplemental UEL.

`build.py` is adapted from Teng Zhang's project-authored
[`pipeline_hex8.py`](https://github.com/tengzhang48/abaqus_ufl/blob/0f525339db1aad70e9f8f4825a02c1164f0da7a0/paper_examples/morphing_hex8/pipeline_hex8.py)
at public `abaqus_ufl` commit
`0f525339db1aad70e9f8f4825a02c1164f0da7a0`. The original declaration is
Copyright (c) 2026 Teng Zhang and is licensed under the
[MIT License](https://github.com/tengzhang48/abaqus_ufl/blob/0f525339db1aad70e9f8f4825a02c1164f0da7a0/LICENSE).
This CoupFE port changes the package namespace and adds the reusable
verification/build helpers.

## Evidence and limits

The focused CoupFE test checks the constitutive equations against an
independently written NumPy oracle, including the dissipative flux direction
and a reversed-flux broken control. It also runs `problem.verify()`, regenerates
the local-pressure Hex8 UEL in a temporary directory, and object-compiles it
with `gfortran` when that compiler is available. These gates establish the
current declaration/code-generation path; they are not an independent
boundary-value validation.

`reference_result.json` records, but does not claim to reproduce, the
historical Abaqus/Standard 2022 paper run. That result used an older archived
generated source together with a node-sharing, mechanically active C3D8
companion mesh (initial shear modulus 200 versus 800 for the UEL) and an
Abaqus contact setup. The companion mesh therefore affected equilibrium, and
its stiffness-sensitivity audit remains pending. The exact submitted source
also differs from the current generator output.

No Abaqus deck, archived source, mesh, NPZ data, figure, or contact adapter is
copied into CoupFE. Consequently, neither the recorded `12.5329 mm` displacement
nor Figure 6 is a current CoupFE reproduction claim. Promotion from
**RESEARCH** requires a clean current-source boundary-value rerun, a
companion-mesh sensitivity decision (or removal), and retained quantitative
acceptance evidence.
