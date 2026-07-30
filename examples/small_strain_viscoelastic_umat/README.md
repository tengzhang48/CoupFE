# Small-strain viscoelastic SLS UMAT

This code-generation example implements a Standard Linear Solid with one
deviatoric Maxwell branch in parallel with an equilibrium spring:

```text
eps_v_new = (eps_v_old + (dt/tau) dev(eps)) / (1 + dt/tau)
sigma = K tr(eps) I + 2 G_inf dev(eps) + 2 G_v (dev(eps) - eps_v_new)
```

The Abaqus `*User Material` property order is `K, G_inf, G_v, tau`.
`STATEV(1..9)` stores the full viscous strain tensor in column-major order.
The Python model is compression-positive; the generated UMAT converts to
Abaqus tension-positive stress and engineering-shear conventions at the
interface.

For a held tensor-shear step `e` and `r = dt/tau`, the independent discrete
backward-Euler oracle is:

```text
tau_n = 2 e [G_inf + G_v/(1+r)^n]
eps_v_xy(n) = e [1 - 1/(1+r)^n]
```

The focused test compares every increment to these formulas, checks the
instantaneous/equilibrium bounds and continuous-time limit, and confirms that
an explicit-dashpot update is rejected.

From the CoupFE repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python examples/small_strain_viscoelastic_umat/build.py

PYTHONDONTWRITEBYTECODE=1 \
  pytest -p no:cacheprovider tests/test_abaqus_ufl_inelastic_umats.py
```

Current evidence covers the independent Python oracle, framework tangent
verification, byte-identical regeneration, and `gfortran` object compilation.
No Abaqus deck or current Abaqus execution result is included. The UMAT is
strictly three-dimensional (`NDI=3`, `NSHR=3`, `NTENS=6`); it omits multiple
Maxwell branches, volumetric relaxation, temperature dependence, finite-strain
kinematics, and Abaqus energy outputs.

The declaration was ported from project-authored MIT code in
[`tengzhang48/abaqus_ufl`](https://github.com/tengzhang48/abaqus_ufl), commit
`0f525339db1aad70e9f8f4825a02c1164f0da7a0`,
`examples/small_strain_viscoelastic_umat/build.py`. Copyright © 2026 Teng
Zhang. The adapted declaration remains MIT-licensed, as recorded by its SPDX
header.
