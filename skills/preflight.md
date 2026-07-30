# CoupFE pre-flight skill — dimensionless analysis + a dry run BEFORE any production simulation

Read this before setting up ANY new simulation (and before concluding anything from one).

**The principle (Teng, 2026-07-02):** in the AI era, generating a runnable simulation is
cheap — the dominant failure mode is a WRONG SETUP that runs fine. An agent then debugs
the physics or the solver when the boundary conditions, loading protocol, or material
scales were wrong from the start, and closes the problem with a false "method
limitation". Two cheap disciplines catch most of this before the expensive run:
(1) a dimensionless analysis, (2) a dry run that tests the setup itself.

Both documented mis-closures in this repo were exactly this: the ring-compression
"solve_dynamics cannot reach the quasistatic branch" verdict (a 40 s "slow" ramp against
a 126 s fundamental period — nobody computed the timescale; `docs/lessons_learned.md`
2026-07-02) and the earlier absolute-parameter transfer failures. The pre-flight costs
minutes; the mis-closure costs days and pollutes the record for the next agent.

## 1. Dimensionless analysis FIRST — derive every number, never pick one

Before choosing ramp times, damping, `dt`, penalties, or barrier bands:

- **Identify the scales**: stiffness (`G`, `K` → wave speed `c = sqrt(G/rho)`),
  geometry (`R`, thickness `h`), load/displacement scale, and the QOI's governing
  dimensionless group (e.g. `Q/(mu P)` for partial slip; a phenomenon has a THRESHOLD
  value of its group — check your inputs actually reach it before running).
- **Compute the structural timescales NUMERICALLY — do not guess.** The lowest
  eigenmode of the assembled model is one call:

  ```python
  vals = spla.eigsh(K, k=8, M=sp.diags(M_lumped), sigma=0.0,
                    which="LM", return_eigenvectors=False)
  omega = np.sqrt(np.sort(np.abs(vals)))     # rigid-body ~0s first, then omega_1
  ```

  "Slow ramp" and "low damping" are MEANINGLESS without `omega_1`: quasistatic means
  ramp time >> `T_1 = 2*pi/omega_1` (or use ramp-hold-settle), and mass-proportional
  damping has `zeta_i = alpha/(2*omega_i)` — pick `alpha ~ 2*omega_1` for near-critical
  relaxation of the slowest mode. A thin soft ring had `T_1 = 126 s`; the "slow" 40 s
  ramp was impulsive.
- **Derive the numerics from the groups**: `dt` vs the relaxation time you actually
  need to resolve; penalty `k` vs the elastic scale (penetration `~F/k` below mesh
  tolerance, but not so stiff it destroys conditioning); barrier `dhat` vs local element
  size; time-step travel of any moving obstacle vs `dhat`/gap.
- **Write the group table down** (in the script docstring or run notes). It is the
  first thing the next agent needs and the first thing a reviewer checks.

## 2. The dry run — test the SETUP, not the physics

A cheap run (coarse mesh, few steps, small load) with explicit checks, BEFORE the
production run. What it must verify:

- **Boundary conditions.** Count the near-zero eigenvalues — exactly the expected
  rigid-body modes and no more (a missing constraint shows up as an extra ~0 mode; an
  accidental over-constraint removes one). Reactions appear at the intended DOFs and
  sum to the applied load. A symmetry BC preserves the symmetry (mirror the field and
  diff). Where feasible, run a BROKEN CONTROL of the setup: deliberately drop/flip one
  BC and confirm the checks catch it.
- **Loading protocol.** Log the driver (plate position, load factor, prescribed value)
  vs `t` and LOOK at it — verify it matches intent before coupling to physics. Wire a
  kinetic-energy monitor from step one; for any relaxation/quasistatic-via-dynamics run,
  sampling is legal ONLY at a settled state (`KE < ~1e-8` against the strain-energy
  scale) — never at the end of a ramp. Decide the sample points in advance.
- **Material properties.** One-element/patch sanity: recover the small-strain modulus
  from a uniaxial patch; check total mass = `rho * V`; recompute the step-1 group table
  from the ASSEMBLED model (what the code actually built), not from the intended inputs
  — unit-system slips live in exactly that gap.
- **Contact.** Trace the gap through first touch; penetration below tolerance; the
  active-set size evolves smoothly (chatter in the count is a protocol or scale
  problem, not noise).

Keep the dry run as a committed script or test where feasible — it doubles as the
broken-control harness for the setup, and the KE/driver logs become the evidence a
validation claim rests on.

## 3. The gate this imposes on conclusions

No production run, no validation claim, and above all **no "method/solver limitation"
conclusion** until the pre-flight artifacts exist: the dimensionless table and the
dry-run log. A surprising result — especially one where GENTLER loading looks worse —
is a setup bug until the pre-flight proves otherwise (see `docs/lessons_learned.md`
2026-07-02 and the corrected 2026-06-30 entry for the canonical example).
