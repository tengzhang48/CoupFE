# PERMON (PermonQP / PermonFLLOP) — assessment for CoupFE contact

Standing note (2026-06-24). https://permon.vsb.cz/permonqp.htm — VSB-TU Ostrava / IT4Innovations.
Companion to `docs/dev/dual_multiplier_strategy.md`. Question it settles: is PERMON a shortcut to the
"efficient + MPI dual-multiplier contact" we want, and where does it fit?

## What PERMON is

A collection built **on top of PETSc**:
- **PermonQP** — parallel **quadratic-programming** solver. MPRGP (bound-constrained, for non-penetration),
  SMALBE/SMALSE (augmented Lagrangian for equality constraints / gluing).
- **PermonFLLOP** — **(T)FETI / H-TFETI** domain decomposition on PETSc.
- **Flagship application: contact mechanics as a QP.** Signorini non-penetration + Coulomb/Tresca friction
  expressed as constraints; solved in the **dual** (the contact forces *are* the Lagrange multipliers),
  condensed to the interface, **scalable to thousands of cores**.

So PERMON is essentially the **productized, HPC-scale, dual-multiplier contact solver** — and it **proves
the direction is real**: multiplier contact-as-QP, interface-condensed, scalable on the same PETSc CoupFE
already uses. It de-risks "can the dual-multiplier be efficient + MPI?" — demonstrably yes, at scale.

## Where it helps us, and where it doesn't

| | PERMON | Our condensed solver (`SemismoothFrictionSolver`) |
|---|---|---|
| Scalable MPI forward contact-QP | ✅ mature, TFETI, 1000s cores | ◐ serial now; PETSc FieldSplit = the planned MPI path |
| Niche fit (small-sliding / small-strain) | ✅ (FETI contact is node/mortar = small-sliding) | ✅ same niche |
| **Differentiable** (our only real edge) | ❌ forward C solver | ✅ JAX-portable (the moat) |
| Architecture | **FETI / domain-decomposition** — a real commitment | global assembled + KSP, matches CoupFE |
| Dependency | C library on PETSc; **Python binding to verify** | in-Python, in CoupFE |

## Honest read / decision

1. **Massive-scale *forward* dual-multiplier contact → use PERMON, don't rebuild FETI+QP.** It's
   battle-tested and on our PETSc. Worth it only if we actually need that scale.
2. **CoupFE's niche + the differentiable angle → PERMON doesn't serve it** (forward-only, like Abaqus).
   Our lightweight condensed solver + a PETSc-FieldSplit MPI version fits CoupFE and can go differentiable
   — which is the thing nobody else has and the only part worth *our* time.
3. **Regardless, PERMON is the blueprint** — MPRGP / TFETI / SMALBE is exactly how the scalable QP contact
   is done; study it to inform our MPI FieldSplit version.

Two things to verify before leaning on it: a usable **Python/petsc4py binding** (it's primarily C), and the
**license** (believed BSD-ish — confirm).

**Bottom line:** PERMON changes the calculus on *forward scale* (adopt it if we ever need HPC-scale forward
contact), but not the strategy — put our effort on **differentiable** dual-multiplier contact (the moat),
and borrow PERMON's algorithms as the recipe for our own MPI version if/when scale demands it.

## Current CoupFE parallel status (context)

- **Already parallel (verified):** the bulk (finite-strain elements) + **ppf smoothed contact** (barrier +
  friction, 2D & 3D) run distributed via `coupfe/assembly/distributed.py` (PETSc KSP, VecScatter ghosting,
  ADD_VALUES assembly; MPICH/petsc4py-only). 1-vs-N rank-independent to machine precision, gated
  (`tests/test_mpi_distributed.py`), demonstrated to 16 ranks (project record).
- **NOT parallel yet:** the **dual-multiplier / semismooth** solver (`contact_semismooth.py`) is serial
  scipy — the just-added `SemismoothFrictionSolver` amortizes the factorization (7× on a load path) but is
  still serial. MPI = wire it onto the existing PETSc FieldSplit/Schur path (the planned step).
