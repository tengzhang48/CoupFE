# Confined compression of cylinders — RESEARCH many-body-contact workflow

This CoupFE workflow is inspired by the Abaqus/Explicit **“Compression of
cylinders with general contact”** example
(`xpl_2dgencont_compression`). The main driver exercises serial many-body
deformable contact; a companion MPI smoke program exercises the narrower
distributed-contact path. This is not a benchmark reproduction: the copyrighted
input deck is not distributed, its exact release/hash and raw CoupFE result are
not retained, and the mesh/material/solver are deliberately adapted.

> **First-release status: RESEARCH.** Passing primitive and smaller contact
> gates establish reusable code paths, not this full-pack workflow or an Abaqus
> comparison.

---

## 1. Historical source-case summary

The deck (`xpl_2dgencont_compression.inp`, fetched with `abaqus fetch` — it is
SIMULIA-copyrighted and not redistributed here) contains, as parsed by
`parse_inp.py`:

| | |
|---|---|
| **Bodies** | **58 deformable cylinders** — 49 RUBBER + 9 STEEL, radii 1.5–8.0 (mixed sizes) |
| **Mesh** | 4315 nodes, 3528 quads (CPS4R) + 180 tris (CPS3), **plane stress** |
| **RUBBER** | Mooney-Rivlin N=1: C10 = C01 = 4.48632 MPa, D1 = 0.02229 MPa⁻¹ |
| **STEEL** | linear elastic E = 70 000 MPa, ν = 0.3 (≈3 orders stiffer) |
| **Container** | rigid U-box `x∈[0,90], y∈[0,95]` (floor + 2 walls), ENCASTRE |
| **Lid** | rigid horizontal line at y = 72.5, displaced **−54 mm** over a 0→0.4 s ramp |
| **Contact** | **general contact** (auto all-exterior) — self/mutual + container + lid; Coulomb friction **μ = 0.1** |
| **Loads** | gravity 9801 mm/s² (−y) + the lid displacement |
| **Solver** | Abaqus/**Explicit**, nlgeom, 2 steps |

## 2. Why not consume the raw mesh directly

57 of the 58 cylinders are **mixed quad+tri** (each disk mesh has a few CPS3
core elements), and the deck is **plane stress**. CoupFE has a clean **Quad4**
(and no Tri3 / plane-stress element). So rather than build a Tri3+plane-stress
element just to read the mesh, we keep **A's geometry as the comparison basis**
and regenerate each cylinder as a clean all-quad disk:

- `parse_inp.py` — a minimal Abaqus `.inp` parser (nodes, elements, elsets,
  instances) + connected-components to identify the 58 separate cylinders.
- `build_model.py` — per cylinder, extract centroid + radius + material from A,
  then `disk_mesh()` regenerates it as an all-quad disk via the
  **Fernández-Guasti squircle map** of a square grid (no singular center; clean
  circular boundary for contact).

The adaptation is intended to preserve the historically parsed count,
positions, radii, material labels, container, lid, and loading while replacing
the per-cylinder mesh. Re-establish that mapping from a licensed, hashed input
before making an exact-source claim.

## 3. Why F-bar

The rubber is **near-incompressible**: Mooney-Rivlin C10=C01=4.486, D1=0.0223 maps
to a neo-Hookean G = 2(C10+C01) = 17.95, K = 2/D1 = 89.7, i.e. **K/G ≈ 5, ν ≈ 0.40**.
A fully integrated Quad4 is susceptible to volumetric locking under large
compaction strains, so this RESEARCH workflow selects the **F-bar
(mean-dilatation) Quad4** kernel (`neo_fbar_q4.for`,
`formulation='fbar_mechanics'`). The workflow does not establish a public
quantitative or general “no locking” claim.

> Mooney-Rivlin itself is approximated by the equivalent neo-Hookean here; a true
> Mooney-Rivlin codegen element is a later refinement if a faithful material
> response is wanted.

## 4. The CoupFE model

- **Mesh:** the command defaults to the eight lowest disks as a bounded
  demonstration; `n_cyl` may request a larger subset, up to the 58 parsed
  bodies. Rubber and steel quads use separate `ElementGroup`s.
- **Container:** 3 rigid `HalfSpace`s (floor y=0, left x=0, right x=90).
- **Lid:** a rigid `HalfSpace` (normal −y) lowered incrementally — the compaction.
- **Contact:** one `DeformableBarrierContact2D` over the **union** of all disk
  boundary nodes/edges. Per-node body identifiers exclude same-cylinder edges;
  broad-phase and incident-feature rules handle the remaining candidate set.
  A `RigidBarrierContact` represents each wall and the lid. The study uses
  ppf-inspired smoothed friction with μ = 0.1.
- **Driver:** implicit `solve_dynamics` (backward Euler) as a research workflow;
  this does not establish that a quasistatic formulation cannot converge.

Supply a lawfully obtained deck, then run:

```bash
export COUPFE_CYLINDERS_INP=/path/to/xpl_2dgencont_compression.inp.txt
PYTHONPATH=. python examples/compression_cylinders/run.py [n_cyl]
```

The scripts fail with a direct setup message when neither the environment
variable nor the backward-compatible repository-root path resolves a file. The
deck is never bundled in a wheel or source distribution.

## 5. Verification requirements

A bounded qualification run must retain convergence, mutual- and rigid-contact
minimum gaps, reactions, and environment records. Use consistent units to
derive mass and gravity. Any prescribed obstacle motion must participate in
CCD; changing an obstacle position outside the time-aware contact update can
bypass the step bound. The current script recreates the lid at discrete
positions and therefore remains a demonstration rather than that qualified
motion-aware workflow. See `skills/contact.md` for the current contract.

## 6. Serial vs distributed intent

The full pack is intended to exercise the distributed path; this repository
does not retain a release-grade performance record for it.

`examples/mpi_smoke/distributed_cylinders.py` exercises
`solve_dynamics_distributed` with cross-rank deformable-barrier assembly and a
global CCD minimum. Generic public tests cover distributed assembly/contact
primitives. Re-run this workflow at multiple ranks with hashed output
before quoting correctness or scaling for this pack. The walls/lid remain
outside that distributed path, and self-contact is not part of this distinct
convex-body setup.

## 7. Honest comparison to Abaqus

This is not yet an Abaqus validation, qualitative match, or bit-for-bit
reproduction. Differences include implicit versus explicit time integration,
regenerated Quad4 versus quad/tri plane-stress mesh, and neo-Hookean versus
Mooney-Rivlin rubber. The open-source value today is the inspectable many-body
contact workflow; a comparison claim requires the exact source deck/version,
authorized inputs, retained runs, and stated acceptance criteria.
