# Stabilized mixed u-theta Tet4

**Status: RESEARCH.** This example contains the constitutive declaration,
independent Python checks, framework tangent checks, and a currently generated
four-point Tet4 Abaqus UEL. It is not a current CoupFE solver reproduction of
the paper benchmark.

## Scope

`build.py` implements the compressible neo-Hookean mixed formulation used for
the block-compression benchmark in Section 6.3, Figure 14 of:

Guglielmo Scovazzi, Rubén Zorrilla, and Riccardo Rossi, “A kinematically
stabilized linear tetrahedral finite element for compressible and nearly
incompressible finite elasticity,” *Computer Methods in Applied Mechanics and
Engineering* 412 (2023), 116076.
[doi:10.1016/j.cma.2023.116076](https://doi.org/10.1016/j.cma.2023.116076)

The equal-order element has four displacement/Jacobian-discrepancy DOFs per
node (`u1`, `u2`, `u3`, `thetat`) and 16 element DOFs. Here
`theta = 1 + thetat`, so the reference value of every nodal unknown is zero.
The scalar residual includes the VMS gradient term; `phase_flux` carries the
minus sign required by CoupFE's `storage*N - flux.Grad(N)` convention.

`PROPS` are ordered as:

1. `mu = 80.194 N/mm2`
2. `lam = 400889.806 N/mm2`
3. `c_tau_u = 2.0`
4. `c_tau_theta = 0.1`
5. `h_elem = 0.0625 mm`

No Gmsh pipeline, Abaqus deck, solver adapter, ODB result, or figure asset is
included here. In the historical n=16 run, the element-size property was
assigned in volume-based bins; the single `h_elem` above is the declaration's
default/target value and is not a replacement for that deck setup.

## Build and checks

From the CoupFE repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python examples/stabilized_tet4/build.py

PYTHONDONTWRITEBYTECODE=1 \
  pytest -p no:cacheprovider tests/test_stabilized_tet4.py
```

The focused test checks:

- an analytic `dS/dtheta` against an independently evaluated complex-step
  derivative, including a deliberately wrong-sign control;
- the exact homogeneous limit `theta = J`;
- material and weak-form tangent verification at distorted states;
- Tet4 generation and `gfortran` object compilation when the compiler exists.

These checks validate the implemented residual and current generated source.
They do not exercise boundary conditions, follower pressure, mesh sensitivity,
nonlinear solution controls, or reproduce the published curve.

## Historical evidence and provenance

`reference_result.json` records reduced metadata from a completed historical
Abaqus run: final center displacement `u3 = -0.6962435841560364 mm`, compared
with the approximately `0.7 mm` published n=16 value. That file is comparison
metadata, not output from the present CoupFE tree.

The declaration was adapted from project-authored MIT code in
[`tengzhang48/abaqus_ufl`](https://github.com/tengzhang48/abaqus_ufl), commit
`0f525339db1aad70e9f8f4825a02c1164f0da7a0`, principally
`paper_examples/stabilized_tet4/scovazzi_block.py` and its shared u-theta
declaration. Copyright © 2026 Teng Zhang. The DOI above is
formulation/benchmark attribution; this example does not claim reuse of the
paper authors' source code.

The adapted declaration remains under the MIT license, as recorded by the SPDX
header in `build.py`. Generated Fortran is mechanically derived from that
declaration and CoupFE's generator.
