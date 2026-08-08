# Compressible neo-Hookean UMAT

This example declares a stateless finite-strain material and generates a
self-contained, three-dimensional Abaqus/Standard UMAT:

```text
lambda = K - 2 G / 3
P = G (F - F^-T) + lambda ln(det F) F^-T
PROPS = (G, K)
```

Here `K` is the physical small-strain bulk modulus; the generated constitutive
law computes the first Lamé coefficient used by the `ln(J)` term.

The generated wrapper converts first Piola stress and its tangent to Cauchy
stress and the Jaumann-rate `DDSDDE` used by Abaqus. It requires
`NDI=3`, `NSHR=3`, and `NTENS=6`; plane-stress, plane-strain, and
axisymmetric use are not supported. It does not maintain Abaqus energy
outputs or state variables.

Run from the repository root:

```bash
python examples/neo_hookean_umat/build.py
pytest tests/test_abaqus_ufl_hyperelastic_umats.py
```

The focused tests compare uniaxial and simple-shear stresses with independent
closed forms, require a deliberately K-free broken control to fail, run
CoupFE tangent verification, require byte-identical regeneration, and compile
the generated fixed-form source with `gfortran` when available. No Abaqus
execution result is claimed.

## Provenance

The declaration is a namespace-only port of the project-authored MIT example
released at
[`tengzhang48/abaqus_ufl@0f52533`](https://github.com/tengzhang48/abaqus_ufl/tree/0f525339db1aad70e9f8f4825a02c1164f0da7a0/examples/neo_hookean_umat).
Original declaration copyright © 2026 Teng Zhang. See the
[upstream MIT license](https://github.com/tengzhang48/abaqus_ufl/blob/0f525339db1aad70e9f8f4825a02c1164f0da7a0/LICENSE).

The parameters are illustrative and use any consistent unit system. For large
`K/G`, choose an appropriate mixed or stabilized formulation instead of
assuming this displacement material avoids volumetric locking.
