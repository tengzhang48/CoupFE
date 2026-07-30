# Small-strain J2 plasticity UMAT

This code-generation example implements compression-positive, small-strain J2
plasticity with linear isotropic hardening and a radial return:

```text
sigma_trial = sigma_old + 2 G dstrain + lam tr(dstrain) I
f = q(sigma_trial) - (sigma_y + H ep_old)
dgamma = f / (3 G + H)                         when f > 0
sigma = sigma_trial - 2 G dgamma n
ep = ep_old + dgamma
```

The Abaqus `*User Material` property order is `G, lam, sigma_y, H`.
`STATEV(1)` stores the equivalent plastic strain. The Python model is
compression-positive; the generated UMAT converts to Abaqus tension-positive
stress and engineering-shear conventions at the interface.

The independent oracle uses the exact monotonic pure-shear solution:

```text
ep(e) = (2 sqrt(3) G e - sigma_y) / (3 G + H)
tau(e) = (sigma_y + H ep) / sqrt(3)
```

It exercises both elastic and plastic branches. The focused test also confirms
that dropping `H` from the return denominator is rejected.

From the CoupFE repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python examples/small_strain_j2_umat/build.py

PYTHONDONTWRITEBYTECODE=1 \
  pytest -p no:cacheprovider tests/test_abaqus_ufl_inelastic_umats.py
```

Current evidence covers the independent Python oracle, framework tangent
verification, byte-identical regeneration, and `gfortran` object compilation.
No Abaqus deck or current Abaqus execution result is included. The UMAT is
strictly three-dimensional (`NDI=3`, `NSHR=3`, `NTENS=6`); it does not implement
kinematic hardening, rate dependence, thermal effects, finite rotation, or
Abaqus energy outputs.

The declaration was ported from project-authored MIT code in
[`tengzhang48/abaqus_ufl`](https://github.com/tengzhang48/abaqus_ufl), commit
`0f525339db1aad70e9f8f4825a02c1164f0da7a0`,
`examples/small_strain_j2_umat/build.py`. Copyright © 2026 Teng Zhang. The
adapted declaration remains MIT-licensed, as recorded by its SPDX header.
