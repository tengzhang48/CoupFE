# Confined compression of cylinders — RESEARCH many-body-contact workflow

This CoupFE workflow is inspired by the Abaqus/Explicit **“Compression of
cylinders with general contact”** example
(`xpl_2dgencont_compression`). It exercises many-body deformable contact and
the distributed-contact path, but is not a benchmark reproduction: the
copyrighted input deck is not distributed, its exact release/hash and raw
CoupFE result are not retained, and the mesh/material/solver are deliberately
adapted.

> **First-release status: RESEARCH.** Passing primitive and smaller contact
> gates establish the reusable code paths. The subset/full-pack observations
> below are historical development notes, not retained release evidence or an
> Abaqus comparison.

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
`formulation='fbar_mechanics'`). A private development study observed the
expected stiffness reduction, but that example is withheld for provenance and
does not establish a public quantitative or general “no locking” claim.

> Mooney-Rivlin itself is approximated by the equivalent neo-Hookean here; a true
> Mooney-Rivlin codegen element is a later refinement if a faithful material
> response is wanted.

## 4. The CoupFE model

- **Mesh:** 58 F-bar Quad4 disks; two `ElementGroup`s (rubber props, steel props).
- **Container:** 3 rigid `HalfSpace`s (floor y=0, left x=0, right x=90).
- **Lid:** a rigid `HalfSpace` (normal −y) lowered incrementally — the compaction.
- **Contact:** one `DeformableBarrierContact2D` over the **union** of all disk
  boundary nodes/edges (broad-phase + 1-ring incident exclusion; convex disks ⇒
  a node never contacts its own disk) for **mutual** cylinder contact, plus a
  `RigidBarrierContact` per wall + the lid. ppf-smoothed friction μ = 0.1.
- **Driver:** implicit `solve_dynamics` (backward-Euler; inertia regularizes the
  non-smooth contact — the barrier does not converge quasistatically).

Supply a lawfully obtained deck, then run:

```bash
export COUPFE_CYLINDERS_INP=/path/to/xpl_2dgencont_compression.inp.txt
PYTHONPATH=. python examples/compression_cylinders/run.py [n_cyl]
```

The scripts fail with a direct setup message when the environment variable is
unset and the private developer-only repository-root deck is absent. The deck
is never bundled in a wheel or source distribution.

## 5. Historical serial-subset observation

A development run of a six-disk subset reported the following console
summary. The raw output and environment were not retained, so these values are
illustrative debugging context, not release evidence:

```
seated: pack top y = 10.18
lid 10.43 → 7.61   pack top 9.90 → 7.21   (floor/wall gaps stay +)
compaction: 10.18 → 7.21  = 29% of pack height
penetration-free throughout: True (min gap +0.22)   -> OK
```

The intended workflow compacts a mixed rubber/steel pack while the bodies
deform and slide. A future bounded gate must retain convergence, minimum-gap,
reaction, and environment records before reporting those outcomes. Two issues
surfaced during development (see
`docs/lessons_learned.md`, 2026-06-25):

- **Gravity = a setup/units mistake (not engine physics).** Using an arbitrary
  nodal mass + a hand-tuned body force "crushed" the soft rubber. The fix is
  consistent units (real density → lumped mass → `force = mass·g`; with
  `εg = ρgL/G ≈ 6.5e-5`, real gravity is gentle and just seats the pack).
- **Lid tunneling = missing physics (a real capability gap).** Non-penetration is
  guaranteed by **CCD bounding the Newton step**, not by the barrier (which is
  zero outside `[0,dhat)`). The lid is a *moving rigid obstacle* moved externally
  (recreating its `HalfSpace`), so its motion **bypasses the CCD**; jump `>dhat`
  and it lands already-penetrated in the barrier's dead zone → tunnels. The
  current "steps `< dhat`" is a **breadcrumb**, not a guarantee. The proper fix —
  give the obstacle a velocity and fold its motion into the gap + `max_step` so a
  displacement-controlled platen is CCD-bounded at any step — is the next contact
  item (displacement-controlled rigid tooling is ubiquitous: indentation,
  compaction, forming; the Hertz indenter uses the same workaround).

## 6. Serial vs distributed intent

Serial multi-body contact is **iteration-heavy** (~50 Newton iters/step for the
barrier on a dense pack), so the full 58-cylinder pack is slow serially — which
is exactly why this example targets the **distributed** path.

`examples/mpi_smoke/distributed_cylinders.py` exercises
`solve_dynamics_distributed` with cross-rank deformable-barrier assembly and a
global CCD minimum. Generic public tests cover distributed assembly/contact
primitives, but the historical eight-disk equality/residual run is not retained
as first-release evidence. Re-run it at multiple ranks with hashed output
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
