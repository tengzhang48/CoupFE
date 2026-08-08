# Example review response — 2026-08-08

This record documents the response to a repository-wide review of the public
examples on `agent/hertz-contact-evidence` at `88811a2`.  It is an engineering
audit record, not an additional validation claim.

## Method

The review findings were handled in four stages:

1. Preserve the reviewed branch and reproduce each reported value without
   modifying the source.
2. Derive the relevant invariant independently and distinguish core behavior
   from example logic, evidence extraction, and documentation.
3. Correct confirmed defects with direct invariant tests and deliberately
   broken controls. Preserve established raw ABIs where silently changing them
   could break direct callers.
4. Run focused material, contact, generation, compilation, and example tests,
   followed by the default CI-equivalent suite and repository checks.

The reusable version of this workflow is in [`README.md`](README.md).

## Finding disposition

### J2 plastic return

**Confirmed; example physics corrected.** The example combined the standard
`3G + H` return denominator with a nonstandard `s/q` direction and scaled
hardening increment. At its retained plastic state, the old update returned
`q = 819.14` against an updated yield radius of `265.86`.

The declaration now uses the conventional associative direction
`3s/(2q)`, `alpha += delta_gamma`, and `delta_gamma = phi/(3G + H)`. Tests
cover the elastic branch, active yield-surface closure, deviatoric state,
stress reconstructed from committed state, generated-kernel trial/commit
behavior, and the former update as a broken control.

### Neo-Hookean material parameters

**Confirmed; public/raw boundary corrected without changing the retained core
kernel ABI.** The public API accepts physical shear and bulk moduli `(G, K)`,
while the core kernels under `coupfe/runtime/elements` and their legacy raw
copies consume `(G, lambda)` because the coefficient of `ln(J)` has the
infinitesimal role of the first Lamé parameter.

The public boundary now performs the named conversion
`lambda = K - 2G/3`. Direct callers of those raw kernels use the same converter,
and raw generator fixtures name the property `lame_lambda`. Code-generated
examples whose declared ABI is `(G,K)` instead perform the conversion inside
their constitutive declaration and continue accepting physical `(G,K)` without
a caller-side conversion. Independent centered shear and hydrostatic
perturbations recover the requested `G` and `K`; a raw
`K`-in-the-`lambda`-slot control must fail the bulk-modulus gate. The Hertz
benchmark is routed through the converter and retains its existing physical
parameters and result.

### Three-dimensional contact gap

**Confirmed as an evidence defect; no core collision defect established.** The
old example extractor used unsigned closest-point distance, which becomes
positive again after a vertex crosses a face. The examples now select the
closest triangle, orient its deformed normal consistently with the reference
facet, and track signed normal separation over the accepted trajectory. A
collapsed facet fails closed. A synthetic through-face control verifies that a
tunneled secondary surface has negative gap.

The core `DeformableBarrierContact3D.max_step` ACCD path was not changed: this
finding concerned the example's evidence gate, not a reproduced failure of the
collision-bound implementation.

### Exact-stick Coulomb cap

**Confirmed; example load distribution and evidence corrected.** The old slip
branch returned the literal expected value `mu*P`, and its unnormalized nodal
weights could apply a different global resultant. The slip load now scales the
all-stick tangential-multiplier pattern to one normalized global resultant,
which also preserves a continuous onset. The reported friction force is
recovered independently from the constrained top reaction. Half-cap and
reversed-direction broken controls demonstrate that the public gates observe
both an incorrect magnitude and friction doing positive work.

### Stabilized Tet4 tangent

**The dead flag was confirmed; the claimed major tangent failure was not.** At
all retained states, the fixed-scalar material tangent is major-symmetric to
roundoff and its derivative check passes well below the normal tolerance. The
misleading unused `symmetric_tangent = False` flag and comment were removed,
the verification tolerance was restored from `5e-5` to `1e-6`, and an explicit
symmetry regression was added.

### Minor findings

- The thermo-mechanical example now documents `alpha` as a log-volumetric
  coefficient and `T` as a change from an implicit zero reference.
- The viscoelastic continuous-limit diagnostic uses the material's actual
  `tau`, with a non-default-`tau` regression.
- The J2 equation stub accepts and forwards its state/time arguments through
  the instantiated material.
- The 3-D contact comment no longer equates barrier scale `kappa` with bulk
  modulus; they have different roles and units.

## Verification record

The corrected code was pinned at
`a4a7c3843cb9c4864b6306ad0f57647a6fe981f2` before refreshing public numerical
evidence.

- `python -m pytest -q -m 'not slow'` completed with `170 passed`, `1 skipped`,
  and `1` opt-in slow test deselected.
- Focused material/code-generation suites included `12` physical-modulus
  perturbation tests, raw-kernel regeneration/hash checks, generated-source
  compilation, J2 trial/commit checks, and the contact/friction broken controls.
- Every modified retained UEL/UMAT source was regenerated; the corresponding
  declarations also generated and compiled in temporary directories.
- The pinned `neo_hookean_block`, `curved_annulus`, `hertz_contact`,
  `contact_3d_blocks`, and `contact_3d_friction` simulations were rerun. Their
  refreshed values and source hashes are in [`../site/evidence.json`](../site/evidence.json).
- `python .github/scripts/check_site.py` passed after the evidence and displayed
  values were refreshed.

## Claim boundary

These changes strengthen the stated example-level checks. They do not turn the
research examples into real-device validation, establish general self-contact
or distributed-contact qualification, or make a small-strain J2 proof element
a finite-strain plasticity model. The final executable results and exact
commands are recorded in the commit/CI history for the corrected revision.
