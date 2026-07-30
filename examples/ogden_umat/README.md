# One-term compressible Ogden UMAT

This example declares a stateless spectral hyperelastic material and generates
a self-contained, three-dimensional Abaqus/Standard UMAT. With
`lb_i = J^(-1/3) lambda_i`, it uses

```text
W = (2 mu / alpha^2) sum_i (lb_i^alpha - 1) + K/2 ln(J)^2
PROPS = (mu, alpha, K)
```

The implementation reconstructs stress invariantly from the eigenspaces of
`C = F^T F`. Pair- and triple-repeated principal stretches are included in
the independent value checks. Derivatives of individual eigenvectors are not
part of the supported contract.

Run from the repository root:

```bash
python examples/ogden_umat/build.py
pytest tests/test_abaqus_ufl_hyperelastic_umats.py
```

The focused tests use eig-free closed forms for dilation, isochoric uniaxial
stretch, and the `alpha=2` neo-Hookean degeneracy. They also require a broken
model without the isochoric split to fail, run CoupFE tangent verification,
require byte-identical regeneration, and compile the fixed-form source with
`gfortran` when available.

The generated UMAT requires `NDI=3`, `NSHR=3`, and `NTENS=6`. It has no state
variables and does not maintain Abaqus energy outputs. No Abaqus execution
result is claimed.

## Provenance

The declaration is a namespace-only port of the project-authored MIT example
released at
[`tengzhang48/abaqus_ufl@0f52533`](https://github.com/tengzhang48/abaqus_ufl/tree/0f525339db1aad70e9f8f4825a02c1164f0da7a0/examples/ogden_umat).
Original declaration copyright © 2026 Teng Zhang. See the
[upstream MIT license](https://github.com/tengzhang48/abaqus_ufl/blob/0f525339db1aad70e9f8f4825a02c1164f0da7a0/LICENSE).

The single-term parameters are illustrative and use any consistent unit
system. Multiple Ogden terms and inelastic effects are outside this example.
