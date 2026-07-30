# Phase-field stress-corrosion element

This research example is the CoupFE form of the three-field
displacement–phase–concentration element used for Section 3.1 / Figure 3 of
the `abaqus_ufl` paper. It combines a reduced-integration Quad8 element,
small-strain J2 plasticity, damage, repassivation history, phase evolution,
and species transport.

From the repository root:

```bash
PYTHONPATH=. python examples/phasefield_corrosion_cui/build.py
```

The command verifies the fully coupled declaration and generates
`phasefield_corrosion_cui_full_uel.for`. The retained source intentionally
omits the Abaqus-only visualization bridge. Focused tests exercise material
tangents, independent element assembly, state transfer, native/UEL parity,
generated-source compilation, and a pathological-initialization broken
control.

## Paper and source lineage

The reference formulation and comparison implementation are:

Chuanjie Cui, Rujin Ma, and Emilio Martínez-Pañeda, “A phase field formulation
for dissolution-driven stress corrosion cracking,” *Journal of the Mechanics
and Physics of Solids* 147 (2021), 104254.
[doi:10.1016/j.jmps.2020.104254](https://doi.org/10.1016/j.jmps.2020.104254)

The CoupFE declaration is project-authored and was published under the MIT
License in
[`abaqus_ufl` at commit `0f52533`](https://github.com/tengzhang48/abaqus_ufl/tree/0f525339db1aad70e9f8f4825a02c1164f0da7a0/paper_examples/phasefield_corrosion).
The complete attribution and source hashes are in that repository’s
[`CREDITS.md`](https://github.com/tengzhang48/abaqus_ufl/blob/0f525339db1aad70e9f8f4825a02c1164f0da7a0/CREDITS.md).

The original Cui UEL and deck are not copied here. The comparison mesh derives
from the reference distribution, and its exact BSD notice and redistribution
status still need to be attached before that artifact can move into CoupFE.
Use the pinned companion repository for the current paper evidence bundle.

## Evidence boundary

This directory is **RESEARCH**, not an independent reproduction claim. Its
tests establish the CoupFE declaration, tangent, assembly, state, and code
generation paths. They do not rerun Abaqus or independently establish the
published Figure 3 response. The current generated source also need not be
byte-identical to the archived paper-run source because the generator and
comparison-tangent choices evolved.
