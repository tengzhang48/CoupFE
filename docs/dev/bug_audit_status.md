# Phase 5 bug-audit status

This document records the outcome of the UEL-focused Phase 5 bug audit for
the CoupFE codegen port.  Each known issue was checked by an independent
oracle plus a broken control; status is **fixed**, **open**, or **deferred**.

| # | Issue | Status | Test file | Notes |
|---|-------|--------|-----------|-------|
| 1 | helper-codegen NameError (`self._helper()` in a UEL) | fixed | `tests/test_uel_helper_codegen.py` | Already ported; compiles and the helper is emitted exactly once. |
| 2 | phase_flux sign trap | fixed | `tests/test_phase_flux_sign_trap.py` | `OperatorSignWarning` is promoted to error in `pyproject.toml`. Correct flux gives a definite block; flipped flux gives an indefinite block and raises the warning. |
| 3 | `z**2` NaN-at-0 | fixed | `tests/test_z2_nan_at_zero.py` | Generated/compiled UEL evaluates `z**2` at `z = 0` finitely; a non-zero analytic oracle (`z = 2`) confirms the residual. |
| 4 | stale-deck-props NaN | fixed | `tests/test_inp_scaffold_props.py` | `UELModelConfig.from_weakform` derives `n_properties` from the material; the generated scaffold advertises the same count. |
| 5 | Sylvester `V.T` vs `inv(V)` | fixed for SPD / diagonal repeated; **open** for non-diagonal repeated eigenvalues | `tests/test_tensor_eig_reconstruction.py` | Random SPD and diagonal repeated matrices reconstruct correctly with `inv(V)` and `V.T` fails as broken control. Non-diagonal repeated eigenvalues return degenerate/near-zero eigenvectors (`test_eig_reconstruction_nondiagonal_repeated_eigenvalues` is `xfail`). |
| 6 | single-interval yield range / `P_ys` 10× (zhang_soga) | deferred | — | Depends on the lab `examples/zhang_soga_2025` research example, which is out of scope for the UEL port. |
| 7 | `eps_r` zero-trap (zhang_soga) | deferred | — | Same as item 6: requires the zhang_soga lab example / UMAT path. |
| 8 | anand_rock 4 paper-deviations | deferred | — | Depends on the lab anand_rock UMAT material, which is out of scope for the UEL port. |
| 9 | eig33z Fortran port (H2) | fixed for SPD / diagonal repeated; **open** for non-diagonal repeated eigenvalues | `tests/test_eig33z_fortran.py` | `eig33z` matches `numpy.linalg.eigh` on random SPD and diagonal repeated matrices; `V.T` reconstruction fails as broken control. Non-diagonal repeated eigenvalues are not handled correctly (`test_eig33z_nondiagonal_repeated_eigenvalue` is `xfail`). |
| 10 | translator idiom limits | constraint (not a bug) | — | Documented as constraints: `sym` is unsupported (use `0.5*(F+F.T)`); a scalar field needing its own `(f - f_old)/dt` history must map to the `p` slot. |

## Update 2026-06-21 — items 5 & 9 (eig) reclassified: resolved, not "open"

The "open for non-diagonal repeated eigenvalues" status was reconsidered against the
constitutive limitation recorded during the source-port audit.
The eig degeneracy is **not a code bug to fix**, in two parts:
1. **Matrix functions** (`logm`/`sqrtm`/`expm`/`polar`) default to the **iterative backend**
   (Denman–Beavers / inverse-scaling-squaring + Taylor — already the CoupFE default,
   `matrix_backend='iterative'`). They never call `eig`/`eig33z`, so they are robust to repeated
   eigenvalues and complex-step-safe. (The anand_rock "bug" was the `eig33z` near-diagonal guard
   *leaking* complex-step perturbations — fixed by this backend, already ported.)
2. **Explicit eigenvectors** at a repeated eigenvalue (a model needing principal *directions*)
   are a **constitutive ambiguity** — any orthonormal basis of the degenerate eigenspace is
   valid → different response. Fix at the model level (a flow rule invariant under eigenvector
   rotation), **not** in `eig`.
So the `test_*_nondiagonal_repeated_eigenvalue` xfails are **intentional limitations**, not
unfinished fixes — reach for the iterative matrix backend instead. (Reframe the xfail reasons
accordingly; see `skills/pitfalls.md`.)

## Deferred / out-of-scope items

Items 6, 7, and 8 involve the lab research examples `zhang_soga_2025` and
`anand_rock`, which depend on UMAT material definitions and example files
that were not ported into CoupFE because this phase is **UEL-focused**.
Likewise, `uinter` generation was dropped from the port (the UEL path does
not depend on it).  These items are explicitly deferred pending a future
UMAT/material-zoo port, not silently ignored.

## Translator constraints (item 10)

The Python-to-Fortran translator intentionally does not support every NumPy
idiom.  Two constraints matter in practice:

1. `sym(A)` is not translated; write `0.5 * (A + A.T)` explicitly.
2. If a scalar field needs a history term of the form
   `(f - f_old) / dt`, it must be declared as the pressure-like field `p`
   so that the generator's history plumbing treats it correctly.
