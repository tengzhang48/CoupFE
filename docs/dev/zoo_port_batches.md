# Element zoo port batches (Phase 10)

Inventory of in-scope UEL examples in `abaqus_ufl_lab/examples/` and proposed port
batches into CoupFE codegen. Out of scope: `*_umat` directories, UMAT-only
examples (`Anand_2025`, `anand_2025_rock`, `J2_FeFp`, `Jiao_2026`,
`zhang_soga_2025`), magneto-UEL (`MRE`), and non-UEL utilities.

**Release note (2026-07-30):** this is a historical private-source inventory,
not a shipped-capability list. Several named paths are withheld from the first
public artifact. A port is publishable only after its method citation,
source/generated-source license, adaptation lineage, and redistribution
authority are recorded in
[`examples/REFERENCES.md`](../../examples/REFERENCES.md).

## Already ported (3)

| Lab directory | CoupFE directory | Element | Fields | State | Notes |
|---------------|--------------------------|---------|--------|-------|-------|
| `Fbar_uel` | `examples/Fbar_uel` | Quad4 | `u` | — | F-bar neo-Hookean |
| `scalar_diffusion_uel` | `examples/scalar_diffusion_uel` | Quad4 | `u`, `T` | — | Coupled thermo-mechanical |
| `uel_scaffold_quad4` | `examples/uel_scaffold_quad4` | Quad4 | `u` | — | Neo-Hookean scaffold |

## Proposed batches

### Batch 1 — geometry / dimension de-risk (~4–5 elements)

Goal: prove the native ABI + state schema generalize across element geometry
and 3D **before** research multiphysics rides on it.  All have simple/no stored
state so failures point to geometry/DOF-interleaving, not state.

| # | Lab directory / file | Element | Fields | State | Why in Batch 1 |
|---|----------------------|---------|--------|-------|----------------|
| 1.1 | `thermo_mechanics_quad8/thermo_mechanics_quad8.py` | Quad8 | `u`, `T` | — | First Quad8; coupled DOF interleaving |
| 1.2 | `neo_hookean_mixed/quad8_mixed_patch.py` | Quad8 | `u`, `p` | — | Quad8 mixed formulation |
| 1.3 | `simple_gel_quad4/simple_gel_quad4.py` | Quad4 | `u`, `mu` | — | Coupled two-field gel; H1/H3 bug-site candidate |
| 1.4 | `gel_three_field/hex20_gel.py` | Hex20 | `u`, `p`, `mu` | — | First 3D / Hex20 geometry |
| 1.5 | `gel_chester_anand/u_p_mu_quad8/build.py` | Quad8 | `u`, `p`, `mu` | — | Mixed three-field Quad8 (no local-pressure condensation) |

**Batch 1 gates:** generate → compile → native↔UEL sign check →
`reference_assembly` match → register in `validation/models/`.

### Batch 2 — coupled + stored-state workhorses (~10–12 elements)

Goal: exercise stored state + commit round-trip and known bug-sites now that
geometry is proven.

| # | Lab directory / file | Element | Fields | Stored state | Notes |
|---|----------------------|---------|--------|--------------|-------|
| 2.1 | `phasefield_fracture_uel/build.py` | Quad4 | `u`, `d` | — | Simple coupled damage; no state |
| 2.2 | `LCE/build.py` | Quad4 | `u`, `theta`, `S` | `Fv11/22/33/12/21` | LCE director/order-parameter |
| 2.3 | `Xue_2025/build.py` (`Xue2025ReducedProblem`) | Quad4 | `u`, `volt`, `xi_phase`, `d`, `grain` | `H` | Electro-chemo-mechanical + damage; H1/H2/H3 |
| 2.4 | `Xue_2025/build.py` (`Xue2025MixedXiProblem`) | Quad4 | `u`, `volt`, `xi_phase`, `d`, `grain`, `mu_xi` | `H` | Mixed Cahn-Hilliard variant |
| 2.5 | `Hussein_2026/build.py` | Quad4 | `u`, `d` | `ep`, `epsp` (3×3), `wp`, `Hdr` | Ductile PFF; H2/H3/M22 |
| 2.6 | `Hussein_2026/build_mediavilla.py` | Quad4 | `u`, `d` | `ep`, `epsp` (3×3), `wp`, `Hdr` | Power-law hardening variant |
| 2.7 | `Li_2026_battery/build.py` | Quad4 | `u`, `phi`, `c`, `T`, `d` | `H` | Electro-chemo-thermo-mechanical fracture; H1/H3 |
| 2.8 | `Ukidwe_2023/build.py` (`Ukidwe2023ReducedUEL`) | Quad8 | `u`, `T`, `cw`, `pg` | — | 4-field drying; Quad8 + H1/H3 |
| 2.9 | `strain_gradient_plasticity_msg/build.py` | Quad4 | `u`, `p` (=`e_p`) | — | Higher-order gradient plasticity; H3 |
| 2.10 | `gel_chester_anand/chester_gel.py` (`ChesterAnandLocalPressureQuad4`) | Quad4 | `u`, `mu` + local `p` | `p` (scalar) | Local-pressure condensation; M10/M16/H3 |
| 2.11 | `phasefield_corrosion_cui/phasefield_corrosion_cui.py` (`CuiJ2Corrosion`) | Quad8R | `u`, `phi`, `c` | `ep`, `epsp`, `deqpl`, `hydro`, `xL`, `ti`, `ei` | Corrosion + J2 plasticity; H1/H3/M22 |
| 2.12 | `phasefield_corrosion_cui/phasefield_corrosion_cui.py` (`CuiJ2CorrosionDiag`) | Quad8R | `u`, `phi`, `c` | same | Block-diagonal variant |

**Batch 2 gates:** all Batch 1 gates + stored-state native↔UEL commit
round-trip (per-GP) + bug-audit regression for known-bug sites.

### Batch 3 — heavy multiphysics long tail (~6–8 elements)

Goal: port the most coupled / multi-block / specialized elements on the fully
proven path.

| # | Lab directory / file | Element | Fields | Stored state | Notes |
|---|----------------------|---------|--------|--------------|-------|
| 3.1 | `Hussein_2026/build_hydrogen.py` | Quad4 | `u`, `p` (=c), `d` | `ep`, `epsp`, `wp`, `Hdr`, `hydro` | Hydrogen-assisted PFF stage 2 |
| 3.2 | `Hussein_2026/build_hydrogen_drift.py` | Quad4 | `u`, `p`, `mu`, `rho`, `d` | `ep`, `epsp`, `wp`, `Hdr`, `hydro`, `rho_bar` | Full drift-flux hydrogen embrittlement |
| 3.3 | `Wang_2026/wang_lymph_node.py` | Quad8 | `u`, `p`, `mu` | — | Mechano-immunological poroelastic; M24 |
| 3.4 | `gel_three_field/axisymmetric_gel.py` | Quad8 (axisym) | `u`, `p`, `mu` | — | Axisymmetric Flory-Huggins gel |
| 3.5 | `gel_chester_anand/chester_gel.py` (`ChesterAnandLocalPressureHex8`) | Hex8 | `u`, `mu` + local `p` | `p` | 3D local-pressure gel |
| 3.6 | `gel_chester_anand/bilayer_multimaterial.py` | Quad4 + Quad4 | gel `u`,`mu`+local `p`; rubber `u` | gel `p` | Multi-material bilayer |
| 3.7 | `Ukidwe_2023/build.py` (`Ukidwe2023GasSplitUEL`) | Quad8 | `u`, `T`, `cw`, `cv`, `pg` | — | 5-field gas-split drying |

**Batch 3 gates:** all Batch 2 gates; register each; log any skipped
sub-elements or known limitations.

## Excluded from Phase 10

- `*_umat` directories (13+) — UMAT path, deferred.
- `Anand_2025`, `anand_2025_rock`, `J2_FeFp`, `Jiao_2026`, `zhang_soga_2025` —
  UMAT/UMATHT, deferred.
- `MRE` — magneto-UEL uses a separate generator, deferred.
- `fe_solver` — non-UEL utility.

## Totals

- Already ported: 3
- Batch 1: 5
- Batch 2: 12
- Batch 3: 7
- **Total in-scope UEL variants: ~27**
