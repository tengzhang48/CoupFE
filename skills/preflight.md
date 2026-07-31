# CoupFE pre-flight guide — dimensionless analysis and a dry run

Use this before a substantial new simulation or a strong conclusion from one.

AI assistance can make a model runnable quickly, but boundary conditions,
loading protocols, units, and material scales still require independent review.
Two inexpensive disciplines catch many setup defects before a long run:
dimensionless analysis and a dry run that tests the setup itself.

## 1. Dimensionless analysis

Before choosing ramp times, damping, `dt`, penalties, or barrier bands:

- **Identify the scales**: stiffness (`G`, `K` → wave speed `c = sqrt(G/rho)`),
  geometry (`R`, thickness `h`), load/displacement scale, and the QOI's governing
  dimensionless group (for example, `Q/(mu P)` for partial slip). Check whether
  the selected inputs reach the regime being studied.
- **Estimate structural timescales from the assembled model.** For a symmetric
  constrained stiffness and positive mass matrix, one possible reduced-system
  calculation is:

  ```python
  # K_free and M_free have constrained DOFs eliminated; choose k < system size.
  k = min(8, K_free.shape[0] - 1)
  vals = spla.eigsh(K_free, k=k, M=M_free, sigma=0.0,
                    which="LM", return_eigenvectors=False)
  scale = max(np.max(np.abs(vals)), 1.0)
  positive = np.sort(vals[vals > 1.0e-10 * scale])
  omega = np.sqrt(positive)
  ```

  Verify symmetry, positive mass, constraints, and residual rigid modes before
  interpreting the eigenvalues. Interpret "slow ramp" and "low damping"
  relative to `omega_1`: a quasistatic study normally needs a ramp time well above
  `T_1 = 2*pi/omega_1` (or a ramp-hold-settle protocol), and mass-proportional
  damping has `zeta_i = alpha/(2*omega_i)` — pick `alpha ~ 2*omega_1` for near-critical
  relaxation of the slowest mode. Check sensitivity rather than treating one
  nominally slow ramp as quasistatic.
- **Derive the numerics from the groups**: `dt` vs the relaxation time you actually
  need to resolve; penalty `k` vs the elastic scale (penetration `~F/k` below mesh
  tolerance, but not so stiff it destroys conditioning); barrier `dhat` vs local element
  size; time-step travel of any moving obstacle vs `dhat`/gap.
- **Write the group table down** in the script documentation or retained run
  notes so contributors and reviewers can check the chosen scales.

## 2. Dry-run the setup

A cheap run (coarse mesh, few steps, small load) with explicit checks before a
long run should verify:

- **Boundary conditions.** Count the near-zero eigenvalues — exactly the expected
  rigid-body modes and no more (a missing constraint shows up as an extra ~0 mode; an
  accidental over-constraint removes one). Reactions appear at the intended DOFs and
  sum to the applied load. A symmetry BC preserves the symmetry (mirror the field and
  diff). Where feasible, use a broken control: deliberately drop or flip one
  BC and confirm the checks catch it.
- **Loading protocol.** Log the driver (plate position, load factor, prescribed value)
  vs `t` and inspect it before coupling to physics. Wire a
  kinetic-energy monitor from step one; for any relaxation/quasistatic-via-dynamics run,
  sample only at a settled state under a kinetic-energy threshold derived from
  the stated strain-energy or work scale. Do not use the end of a ramp as the
  criterion by itself. Decide the sample points in advance.
- **Material properties.** One-element/patch sanity: recover the small-strain modulus
  from a uniaxial patch; check total mass = `rho * V`; recompute the step-1 group table
  from the assembled model (what the code actually built), not from the intended inputs
  — unit-system slips live in exactly that gap.
- **Contact.** Trace the gap through first touch; penetration below tolerance; the
  active-set size evolves smoothly (chatter in the count is a protocol or scale
  problem, not noise).

Keep the dry run as a committed script or test where feasible — it doubles as the
broken-control harness for the setup, and the KE/driver logs become the evidence a
validation claim rests on.

## 3. The gate this imposes on conclusions

Before a validation or method-limitation claim, retain the dimensionless table
and dry-run log. A surprising result, especially one that changes under gentler
loading, should trigger a setup and protocol audit alongside the formulation
and solver investigation.
