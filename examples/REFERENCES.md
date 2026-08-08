# Example evidence and references

This ledger covers all and only the 35 top-level example directories shipped in
the public tree. It records the role each example can support and the boundary
between implementation evidence and physical validation.

- **READY** means supported for the limited role stated here. It does not imply
  production readiness or broad validation of a method family.
- **RESEARCH** means the example demonstrates an implemented path but is not a
  supported validation or production claim.

Evidence categories must remain distinct. Analytic solutions, independently
implemented invariants, and convergence studies can support scoped physical or
numerical claims. Tangent consistency, generated-kernel/reference-assembly
agreement, and native/UEL parity primarily check implementation. Agreement
between backends generated from the same form is not an independent oracle for
the equations or parameters.

## Shared software and method references

[`../CREDITS.md`](../CREDITS.md) is the project-level citation record. In
particular, it identifies:

- the submitted `abaqus_ufl` manuscript for the UEL declaration approach and
  paper-form cases, plus the public software and pinned source revision for the
  UEL and UMAT ports;
- the Abaqus UEL/UMAT interface references, while keeping Abaqus distinct from
  CoupFE's parallel standalone runtime;
- UFL/FEniCSx as conceptual context, not a CoupFE dependency or compatibility
  claim; and
- the `ppf-contact-solver` software and cubic-barrier paper for the adapted
  contact paths.

The inventory below adds the formulation, benchmark, data, and evidence source
specific to each example.

## Complete shipped inventory

| Example | Status | Evidence and reference boundary |
|---|---|---|
| `compression_cylinders` | **RESEARCH** | Qualitative many-body contact workflow based on geometry from the Abaqus/Explicit `xpl_2dgencont_compression` example. The copyrighted deck is not redistributed and no bundled quantitative Abaqus comparison is claimed. |
| `contact_3d_blocks` | **RESEARCH** | Self-contained 3-D deformable collision path covered by `tests/test_examples_contact_3d.py`. [`NOTICE`](../NOTICE) records the Apache-2.0 ppf-contact-solver lineage used by contact primitives. |
| `contact_3d_friction` | **RESEARCH** | Self-contained 3-D smoothed-friction path covered by `tests/test_examples_contact_3d.py`; no external result data. ppf-contact-solver attribution is in [`NOTICE`](../NOTICE). |
| `contact_vs_ppf` | **RESEARCH** | Interoperability recipe against ppf-contact-solver commit `8b7740b032131aeeb46f51d882c96e09b171acc8`. No retained external run supports a quantitative performance or accuracy comparison. |
| `curved_annulus` | **READY** | Convergence against the exact axisymmetric Lamé field implemented locally and gated by `tests/test_curved_convergence.py`; no external data. |
| `exact_stick_friction` | **READY — study** | Small-scale global Coulomb exact-stick and analytic onset checks. Method context: P. Alart and A. Curnier, *Computer Methods in Applied Mechanics and Engineering* 92(3), 353–375 (1991), DOI `10.1016/0045-7825(91)90022-X`. |
| `finite_sliding_capstan` | **READY — study** | Rotating-frame friction state transfer checked against the capstan equation stated in the driver; no external data. |
| `finite_sliding_friction` | **READY — study** | Self-contained re-pairing, state-transfer, Coulomb-force, dissipation, and reversal-memory checks in `tests/test_friction_relay_finite_sliding.py`. |
| `friction_identifiability` | **READY — study** | Frozen-active-set derivative checked against finite differences and slip-support invariants in `tests/test_friction_relay_finite_sliding.py`. This is a small-scale differentiability study. |
| `gel_chester_anand` | **RESEARCH — paper form** | Project-authored mixed `u-p-mu` Quad8 declaration with formulation, reference-assembly, backend, and compiled-execution checks. Source port: MIT-licensed `abaqus_ufl` commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`, Sec. 3.3/Fig. 5. Model reference: S. A. Chester, C. V. Di Leo, and L. Anand, *International Journal of Solids and Structures* 52 (2015), 1–18, DOI `10.1016/j.ijsolstr.2014.08.015`. Supplemental mesh/contact data are not redistributed, so this is not a full paper reproduction. |
| `hertz_contact` | **READY** | Rigid-sphere finite-block normal-force check against the stated Hertz law, gated by `tests/test_hertz_contact.py`, including an opt-in bulk-modulus broken control. The retained force differs by `+1.3%` to `+6.2%` and the fitted exponent is `1.533` versus `1.500`; this is not a mesh/domain-convergence or contact-pressure claim. Reference: K. L. Johnson, *Contact Mechanics*, Cambridge University Press (1985), ISBN `978-0-521-34796-9`. |
| `j2_plasticity_uel` | **READY — codegen proof** | Deliberately small-strain state-schema example with locally stated synthetic/textbook scope. It is not evidence for finite-strain plasticity or a named structural benchmark. |
| `linear_bar` | **READY** | Minimal nonlinear operator and solve path; the linear limit and analytic tangent are gated by `tests/test_operator_contract.py`. No external data. |
| `model_pipeline` | **READY** | Declarative model-construction path covered by `tests/test_model.py` and `tests/test_pipeline.py`. The example is an API demonstration, not a separate physical benchmark. |
| `morphing_hex8` | **RESEARCH — paper form** | Project-authored pressure-gel `u-mu` Hex8 declaration with condensed-pressure and generated-source checks. Source port: MIT-licensed `abaqus_ufl` commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`, Sec. 3.4/Fig. 6. Geometry/model context: Y. Tao et al., *Science Advances* (2021), DOI `10.1126/sciadv.abf4098`; gel lineage: Chester et al., DOI `10.1016/j.ijsolstr.2014.08.015`. The narrow reader accepts user-supplied mesh blocks but does not translate contact, materials, loads, or step semantics, so no full paper reproduction is claimed. |
| `mpi_smoke` | **READY — smoke collection** | Rerunnable bulk, contact, dynamics, friction, and serial-versus-rank programs. They qualify an installation rather than a physical model; no retained final-revision multi-rank or scaling result is published. |
| `neo_hookean_block` | **READY** | Compiled finite-strain block solve checked against the locally derived traction-free lateral stretch; no external data. |
| `neo_hookean_inelastic_local_pressure_quad4` | **RESEARCH** | Project-authored inelastic-volume/local-pressure declaration and build script. No independent pressure oracle or boundary-value solve is claimed. |
| `neo_hookean_local_pressure_hex8` | **RESEARCH** | Project-authored element-local condensed-pressure Hex8 declaration and build script. No independent pressure oracle, inf-sup result, locking result, or inversion-robustness claim is made. |
| `neo_hookean_local_pressure_quad4` | **RESEARCH** | Project-authored element-local condensed-pressure Quad4 declaration and build script. No independent pressure oracle, inf-sup result, locking result, or inversion-robustness claim is made. |
| `neo_hookean_mixed` | **READY — codegen proof** | Mixed Quad8 displacement/Quad4-corner pressure declaration with a material verification state and generated UEL. No boundary-value patch solve, general no-locking result, or inf-sup result is claimed. |
| `neo_hookean_umat` | **READY — codegen proof** | Stateless finite-strain Abaqus UMAT generation checked against independently written uniaxial and simple-shear material-point oracles, deterministic regeneration, and compilation. Source port: MIT-licensed `abaqus_ufl` commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`. No Abaqus structural run or standalone UMAT host is implied. |
| `ogden_umat` | **READY — codegen proof** | One-term compressible spectral Ogden UMAT generation checked against independent principal-stretch states, repeated-eigenvalue cases, deterministic regeneration, and compilation. Source port: MIT-licensed `abaqus_ufl` commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`. No fitted rubber model or structural benchmark is claimed. |
| `phasefield_corrosion_cui` | **RESEARCH — paper form** | Project-authored `u-phi-c` Quad8R declaration with reference-assembly, backend, state, tangent, compile, and broken-initialization checks. Source port: MIT-licensed `abaqus_ufl` commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`, Sec. 3.1/Fig. 3. Formulation reference: C. Cui, R. Ma, and E. Martínez-Pañeda, *Journal of the Mechanics and Physics of Solids* 147 (2021), 104254, DOI `10.1016/j.jmps.2020.104254`. The original UEL/deck and comparison mesh/data are not redistributed; current checks do not reproduce Figure 3. |
| `phasefield_fracture_uel` | **RESEARCH** | Simplified AT2-style coupled displacement-damage codegen demonstration. It is not presented as a complete fracture model or a reproduced published benchmark. |
| `ring_compress` | **RESEARCH** | Manual penalty-contact and dynamic-relaxation workflows requiring a user-supplied input deck. The deck and external reaction table are not redistributed; an authorized table may be supplied through `COUPFE_RING_REFERENCE_CSV`. No bundled Abaqus reproduction is claimed. |
| `scalar_diffusion_uel` | **READY — codegen proof** | Synthetic coupled thermo-mechanical/Fourier generator with locally stated equations and implementation checks. It is not a reproduction of an external model. |
| `semismooth_friction` | **READY — study** | Small-scale per-node partial-slip, Schur-condensed semismooth example gated by `tests/test_contact_semismooth_friction.py`. Reference: P. Alart and A. Curnier, *Computer Methods in Applied Mechanics and Engineering* 92(3), 353–375 (1991), DOI `10.1016/0045-7825(91)90022-X`. The linear-bulk, lagged-normal limitations remain part of the claim. |
| `simple_gel_quad4` | **READY — codegen proof** | Explicitly simplified synthetic displacement/chemical-potential example with locally stated equations. It is not a faithful reproduction of a named gel benchmark. |
| `small_strain_j2_umat` | **READY — codegen proof** | Small-strain radial-return J2 UMAT generation with elastic/plastic path, state-update, broken-control, deterministic-regeneration, and compiler checks. Source port: MIT-licensed `abaqus_ufl` commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`. This is a material-point/codegen proof, not finite-strain plasticity or a structural benchmark. |
| `small_strain_viscoelastic_umat` | **READY — codegen proof** | Standard-linear-solid UMAT generation with tensor-history, relaxation/path, broken-control, deterministic-regeneration, and compiler checks. Source port: MIT-licensed `abaqus_ufl` commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`. This is not a fitted material or structural benchmark. |
| `stabilized_tet4` | **RESEARCH — paper form** | Project-authored stabilized mixed `u-theta` Tet4 declaration with analytic derivative/homogeneous-state, tangent, generation, and compiler checks. Source port: MIT-licensed `abaqus_ufl` commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`. Formulation reference: G. Scovazzi, R. Zorrilla, and R. Rossi, *Computer Methods in Applied Mechanics and Engineering* 412 (2023), 116076, DOI `10.1016/j.cma.2023.116076`. No current Abaqus block reproduction is claimed. |
| `thermo_mechanics_quad8` | **READY — codegen proof** | Synthetic coupled displacement-temperature Quad8 declaration with locally stated equations and scoped implementation checks. It is not a named external-model reproduction. |
| `tire_contact` | **RESEARCH** | Layered mesh, solve, analysis, and sensitivity study. Public tests cover the torus mesh and selected primitives, not a complete tire validation or device prediction. |
| `uel_scaffold_quad4` | **READY — codegen proof** | Minimal F-bar Quad4 declaration that verifies and generates a UEL, with optional f2py compilation. It does not exercise the input-scaffold APIs or validate a boundary-value problem. |

## Source, license, and external-data boundaries

The four paper-form directories (`gel_chester_anand`, `morphing_hex8`,
`phasefield_corrosion_cui`, and `stabilized_tet4`) and four UMAT examples
(`neo_hookean_umat`, `ogden_umat`, `small_strain_j2_umat`, and
`small_strain_viscoelastic_umat`) contain project-authored source ported from the
MIT-licensed `abaqus_ufl` revision identified above. See
[`../LICENSE-ABAQUS-UFL-EXAMPLES`](../LICENSE-ABAQUS-UFL-EXAMPLES) and
[`../NOTICE`](../NOTICE).

References to published equations or benchmarks provide attribution and model
context; they do not imply reuse of paper text, figures, proprietary decks, or
supplemental data. The repository does not redistribute Abaqus input decks or
results for the external workflows. Users must obtain and supply any such files
under terms that permit their use.
