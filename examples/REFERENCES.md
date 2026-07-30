# Example references, evidence, and first-release scope

This ledger covers every tracked top-level directory under `examples/`. It records
what each example is evidence for, where its oracle or external comparison comes
from, and whether it belongs in the supported first-release example set.

The status labels are deliberately conservative:

- **READY** — sufficiently documented for its stated, limited role in the first
  release. `READY` does not turn a research study into a production capability.
- **RESEARCH** — shipped to expose an implemented capability, but explicitly
  unsupported as a release validation claim. Its stated citation, provenance,
  performance, or end-to-end qualification gap must remain visible.
- **WITHHELD** — preserved in this private release-preparation working tree, but
  **not approved for the first public snapshot** until the stated citation,
  source-authority, licensing, or known release-gate failure is closed.
  Presence in this checkout is not release approval.

The capability-rich first-release target is 21 `READY` examples plus 14
explicitly labeled `RESEARCH` demonstrations. The remaining 12 `WITHHELD`
directories must either have their source/rights gap resolved or be excluded
from the public tree and artifact. Do not copy
third-party papers, presentations, proprietary input decks, or screenshots into
the repository to close a reference gap; use a stable bibliographic record and
link, plus an independently redistributable result record where needed.

## What the evidence labels mean

An analytic solution or independently implemented physical invariant can validate
a model quantity. A serial-versus-MPI comparison can validate distribution
invariance. In contrast, complex-step-versus-finite-difference tangents,
generated-kernel-versus-reference-assembly checks, and native-versus-Abaqus-UEL
backend comparisons primarily validate implementation, assembly, signs, state
transfer, or backend parity.

**Implementation/backend parity is not independent physics validation.** When
both sides are generated from the same weak form or use the same constitutive
definition, agreement does not independently establish that the source equations,
parameters, or benchmark interpretation match the literature. The validation
harness makes the same distinction in
[`validation/README.md`](../validation/README.md).

## Complete inventory

| Example | First release | Stated scope and present evidence | Reference/provenance state |
|---|---|---|---|
| `Fbar_uel` | **WITHHELD** | Build-time F-bar Quad4 generator; compiled kernel is compared with a separately assembled Python F-bar element in `validation/models/fbar_uel.py`. | Add the full F-bar method citation and record the public authority/provenance for the ported generator and generated source. Current checks establish implementation consistency, not literature fidelity. |
| `cattaneo_3d` | **WITHHELD** | Closed 3D Cattaneo-Mindlin diagnostic. Two setup/oracle checks pass, but the reviewed end-to-end acceptance gate is a strict expected failure and the retained result says `converged=false`. | Exclude until the solve converges and passes the stated half-space radius, resolved stick/slip, and mesh-refinement criteria. Then add the complete source and external-run provenance before citing comparison values. |
| `compression_cylinders` | **RESEARCH** | Qualitative many-body contact study derived from the Abaqus/Explicit `xpl_2dgencont_compression` example; the copyrighted deck is not redistributed. | Ships as a qualitative capability demo. Record the exact Abaqus release/manual edition, deck identifier, full Fernández-Guasti citation, and reproducible result before making a comparison claim. |
| `contact_3d_blocks` | **RESEARCH** | Self-contained 3D deformable collision example, gated by `tests/test_examples_contact_3d.py`; no external data. | Ships to demonstrate the 3D contact path. [`NOTICE`](../NOTICE) identifies the Apache-2.0 lineage and reproducible upstream snapshot `8b7740b032131aeeb46f51d882c96e09b171acc8`; retained evidence does not prove that snapshot was the exact historical checkout. |
| `contact_3d_friction` | **RESEARCH** | Self-contained 3D smoothed-friction example, gated by `tests/test_examples_contact_3d.py`; no external data. | Ships to demonstrate the friction path. [`NOTICE`](../NOTICE) identifies the Apache-2.0 lineage and reproducible upstream snapshot `8b7740b032131aeeb46f51d882c96e09b171acc8`; retained evidence does not prove that snapshot was the exact historical checkout. |
| `contact_vs_ppf` | **RESEARCH** | Qualitative comparison with the Apache-2.0 ppf-contact-solver. The local README explains non-comparable magnitudes and `ppf_reference.py` supplies the external-side script. | Ships as an interoperability recipe, not quantitative evidence. Use upstream snapshot `8b7740b032131aeeb46f51d882c96e09b171acc8` for a reproducible source baseline; retain the external environment and output before citing a comparison result. |
| `curved_annulus` | **READY** | Curved-boundary convergence study against the exact axisymmetric Lamé field implemented in `annulus.py`; gated by `tests/test_curved_convergence.py`. | The oracle is fully specified in the example and uses no external data. |
| `exact_stick_friction` | **READY** *(study)* | Small-scale global exact-stick Coulomb example with an analytic onset check; gated by `tests/test_exact_stick_friction.py`. | Method lineage and limitations are documented in `docs/dev/dual_multiplier_strategy.md`. Semismooth contact reference: P. Alart and A. Curnier, *Computer Methods in Applied Mechanics and Engineering* 92(3), 353–375 (1991), DOI `10.1016/0045-7825(91)90022-X`. |
| `finite_sliding_capstan` | **READY** *(study)* | Rotating-frame friction state transfer checked against the fully stated capstan equation; gated by `tests/test_friction_relay_finite_sliding.py`. | Analytic oracle is stated in the driver; no external data. |
| `finite_sliding_friction` | **READY** *(study)* | Flat-surface re-pairing/state-transfer study with Coulomb-force, dissipation, and reversal-memory invariants; gated by `tests/test_friction_relay_finite_sliding.py`. | Self-contained physical invariants; no external data. |
| `friction_identifiability` | **READY** *(study)* | Frozen-active-set adjoint checked against finite differences and slip-support invariants; gated by `tests/test_friction_relay_finite_sliding.py`. | Self-contained derivative oracle; small-scale scope and limitations are stated in the driver and dual-multiplier strategy note. |
| `gel_axisymmetric_quad8` | **WITHHELD** | Axisymmetric mixed gel element with codegen and focused implementation tests. | Add full sources for the Flory-Huggins gel law and the axisymmetric/mixed formulation, and record port authority. Current gates do not independently validate a published gel benchmark. |
| `gel_chester_anand` | **RESEARCH** | Project-authored mixed `u-p-mu` Quad8 paper-form declaration with material verification, native/UEL parity, independent reference assembly, compiled execution, and a reduced historical companion-run record. | Source port: MIT-licensed `abaqus_ufl` commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`, Sec. 3.3/Fig. 5. Theory/benchmark: S. A. Chester, C. V. Di Leo, and L. Anand, *IJSS* 52 (2015), 1–18, DOI `10.1016/j.ijsolstr.2014.08.015`. The external supplemental mesh seed is not redistributed; the full Abaqus/contact run is not a Core validation gate. |
| `gel_chester_anand_local_pressure_quad4` | **WITHHELD** | Local-pressure-condensed Chester-Anand-style Quad4 and focused implementation tests. | Add the complete Chester-Anand source, identify any modified equations, and record port/generated-source authority. |
| `gel_three_field_hex20` | **WITHHELD** | Three-field Hex20 gel codegen exercise with focused backend/assembly checks. | Add complete constitutive and mixed-formulation citations plus parameter and port provenance. |
| `hertz_contact` | **READY** | Quantitative rigid-sphere normal-contact gate against the fully stated Hertz force law; gated by `tests/test_hertz_contact.py`. The driver clearly documents the coarse finite-block bias. | No external data. Reference: K. L. Johnson, *Contact Mechanics*, Cambridge University Press (1985), ISBN `978-0-521-34796-9`, Hertz normal contact. |
| `hussein_2026_ductile_pff` | **WITHHELD** | Ductile phase-field-fracture Quad4 with native/UEL/state checks. | “Hussein et al. 2026” and an equation number are not a complete source. Add title, full authors, venue/DOI, exact adapted equations and parameters, and port authority. Backend parity is not published-case validation. |
| `hussein_2026_mediavilla_pff` | **WITHHELD** | Power-law ductile PFF variant with native/UEL/state checks. | Add complete Hussein source and the original Mediavilla benchmark citation, adapted equations/parameters, and port authority. |
| `j2_fefp_uel` | **WITHHELD** | Finite-strain Hencky/log-return J2 study with non-coaxial, backend, and locking checks. | The driver says the physics was ported from a private research source and review. Add a public formulation citation and explicit source/port authority before release promotion. |
| `j2_plasticity_uel` | **READY** *(codegen proof)* | Deliberately small-strain, non-research-grade state-schema element; internal optional toolchain tests cover implementation/state behavior but are not in the reviewed public test partition and were skipped in the current environment. | Its limited synthetic/textbook role is explicit in `build.py`; it is not advertised as validation of a named published model. |
| `lce_quad4` | **WITHHELD** | Reduced liquid-crystal-elastomer codegen port with implementation checks. | “Jiang 2026” is not a complete citation. Add the exact source, identify reductions from the published model, record parameter provenance, and confirm port authority. |
| `li_2026_battery` | **WITHHELD** | Reduced five-field battery codegen stress test with native/UEL/state parity. | “Li 2026” is not a complete citation. Add the exact paper, adapted equations/parameters, reduction statement, and port authority. Current parity is not independent battery-model validation. |
| `linear_bar` | **READY** | Minimal nonlinear operator/solve example. The linear limit and analytic tangent are gated by `tests/test_operator_contract.py`. | Self-contained analytic oracle; no external data. |
| `model_pipeline` | **READY** | Declarative model-construction demonstration. Model and pipeline behavior are covered by `tests/test_model.py` and `tests/test_pipeline.py`, although the exact script is not a subprocess gate. | Self-contained API example; no external data. |
| `morphing_hex8` | **RESEARCH** | Project-authored pressure-gel `u-mu` Hex8 declaration with one condensed element-local pressure; focused material, tangent, local-pressure generation, broken-control, and compile gates. Its example-local reader extracts labeled U3 and companion C3D8 mesh blocks and named sets from a user-supplied pasta deck; it is not a general Abaqus importer. | Source port: MIT-licensed `abaqus_ufl` commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`, Sec. 3.4/Fig. 6. Model/geometry reference: Y. Tao et al., *Science Advances* (2021), DOI `10.1126/sciadv.abf4098`; gel lineage is recorded in the local README. The retained paper result combines the UEL with a mechanically active C3D8 companion mesh and exterior contact, so it is historical context rather than a pure CoupFE oracle. The reader does not translate contact, materials, loads, or coupled-step semantics and therefore does not change that evidence boundary. |
| `mpi_smoke` | **READY** *(smoke collection)* | Ships rerunnable serial-versus-N-rank residual, solve, contact, dynamics, and exact-stick harnesses. The environment-only `tests/test_mpi_distributed.py` is an internal development gate and is not in the reviewed public test partition; a retained final-revision MPI rerun is pending. | These are distribution/assembly checks, not external physics benchmarks. `distributed_cylinders.py` remains a diagnostic companion rather than a direct MPI pytest case. |
| `neo_hookean_block` | **READY** | Finite-strain block solve checked against the explicitly derived traction-free lateral stretch; also supplies compiled-element integration coverage. | Self-contained analytic oracle; no external data. |
| `neo_hookean_umat` | **READY** *(codegen proof)* | Stateless finite-strain Abaqus UMAT generation, checked against independently written uniaxial and simple-shear closed forms plus deterministic-generation and object-compile gates. This does not claim an Abaqus run or a standalone CoupFE boundary-value solve. | Project-authored source ported from the MIT-licensed `abaqus_ufl` example at commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`; exact equations, conventions, tested states, and limitations are stated locally. |
| `neo_hookean_inelastic_local_pressure_quad4` | **RESEARCH** | Isotropic inelastic-volume/local-pressure generator with focused invariants. | Ships as a codegen capability demo, not a qualified formulation. Add a public constitutive/formulation reference or explicit original-derivation statement plus generated-source provenance before promotion. |
| `neo_hookean_local_pressure_hex8` | **RESEARCH** | Element-local condensed-pressure Hex8 with analytic pressure invariants. | Ships as a research formulation demo. Add a mixed/mean-dilatation citation or original-derivation record and scoped locking/inversion evidence before promotion. |
| `neo_hookean_local_pressure_quad4` | **RESEARCH** | Element-local condensed-pressure Quad4 with analytic pressure checks. | Ships as a research formulation demo. Add a mixed/mean-dilatation citation or original-derivation record, generated-source provenance, and scoped evidence before promotion. |
| `neo_hookean_mixed` | **READY** *(codegen proof)* | Mixed Quad8 displacement/Quad4-corner pressure patch generator with internal optional codegen/assembly tests; those tests are not in the reviewed public partition and were skipped in the current environment. | Treat as a mixed-element implementation example; do not infer a general no-locking or inf-sup validation beyond the tested patch. |
| `ogden_umat` | **READY** *(codegen proof)* | One-term compressible spectral Ogden Abaqus UMAT generation, checked against independently written principal-stretch states, repeated-eigenvalue behavior, deterministic generation, and object compilation. This does not establish a broad rubber calibration or Abaqus boundary-value result. | Project-authored source ported from the MIT-licensed `abaqus_ufl` example at commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`; the exact compressible one-term specialization and evidence limits are stated locally. |
| `phasefield_corrosion_cui` | **RESEARCH** | Project-authored Cui-style corrosion/phase-field/J2 Quad8R paper-form declaration with reference-assembly, backend, state, tangent, compile, and broken-initialization checks. | Source port: MIT-licensed `abaqus_ufl` commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`, Sec. 3.1/Fig. 3. Comparison formulation: C. Cui, R. Ma, and E. Martínez-Pañeda, *JMPS* 147 (2021), 104254, DOI `10.1016/j.jmps.2020.104254`. The original UEL/deck and Cui-derived comparison mesh/data are not redistributed here; current gates establish implementation consistency, not Figure 3 reproduction. |
| `phasefield_fracture_uel` | **RESEARCH** | Minimal AT2-style coupled displacement-damage codegen example. | Ships as a simplified codegen demo, not fracture validation. Add the complete AT2/model source and make clear which simplifications remove history/irreversibility before promotion. |
| `ring_compress` | **RESEARCH** | Ring-compression contact/dynamics scripts and a project-authored historical CoupFE summary; execution requires a user-supplied proprietary input deck. | Ships to show the workflow, not an Abaqus reproduction. The external reaction table is excluded because redistribution authority is unresolved and may be supplied only through `COUPFE_RING_REFERENCE_CSV`. Record the exact deck/source, Abaqus version, hashes, run/extraction metadata, and authority before promotion. |
| `scalar_diffusion_uel` | **READY** *(codegen proof)* | Synthetic coupled thermo-mechanical/Fourier diffusion generator with internal optional reference-assembly/backend checks; those checks are not shipped as public release tests and were skipped in the current environment. | The equations are stated locally and are not presented as a reproduction of a named external model. |
| `self_contact_friction` | **WITHHELD** | Hairpin self-contact demonstration; the smaller incident-exclusion, cross-layer, and active-operator tests pass. | The full subprocess gate is a strict expected failure because it exceeds 600 seconds. Exclude the example until a bounded end-to-end gate passes; do not hide that failure behind the passing primitive tests. |
| `semismooth_friction` | **READY** *(study)* | Small-scale per-node partial-slip, Schur-condensed semismooth example, gated by `tests/test_contact_semismooth_friction.py`. | Reference: P. Alart and A. Curnier, “A mixed formulation for frictional contact problems prone to Newton like solution methods,” *Computer Methods in Applied Mechanics and Engineering* 92(3), 353–375 (1991), DOI `10.1016/0045-7825(91)90022-X`. Limitations are recorded in `docs/dev/dual_multiplier_strategy.md`; no external data. |
| `simple_gel_quad4` | **READY** *(codegen proof)* | Explicitly simplified synthetic displacement/chemical-potential example with internal optional implementation checks; those checks are not shipped as public release tests and were skipped in the current environment. | It is not presented as a faithful published gel model; no external data. |
| `small_strain_j2_umat` | **READY** *(codegen proof)* | Small-strain radial-return J2 Abaqus UMAT generation with elastic/plastic path and state-update oracles, a broken control, deterministic regeneration, and object compilation. It is a material-point/codegen proof, not finite-strain plasticity or a structural benchmark. | Project-authored source ported from the MIT-licensed `abaqus_ufl` example at commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`; the synthetic constitutive law, property order, state schema, and sign convention are fully stated locally. |
| `small_strain_viscoelastic_umat` | **READY** *(codegen proof)* | Standard-linear-solid Abaqus UMAT generation with tensor-history and relaxation/path oracles, a broken control, deterministic regeneration, and object compilation. It is a material-point/codegen proof, not a fitted material or structural benchmark. | Project-authored source ported from the MIT-licensed `abaqus_ufl` example at commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`; the synthetic constitutive law, time integration, property order, and state layout are fully stated locally. |
| `stabilized_tet4` | **RESEARCH** | Project-authored stabilized mixed `u-theta` Tet4 declaration for the paper Sec. 3.2/Fig. 4 block case, with analytic derivative/homogeneous-state oracles, material/tangent verification, generated-source compilation, and a reduced historical result record. | Source port: MIT-licensed `abaqus_ufl` commit `0f525339db1aad70e9f8f4825a02c1164f0da7a0`. Formulation/benchmark attribution (no source-code reuse): G. Scovazzi, R. Zorrilla, and R. Rossi, *CMAME* 412 (2023), 116076, DOI `10.1016/j.cma.2023.116076`. The historical `u3=-0.696244 mm` Abaqus result is not a current Core solve gate. |
| `strain_gradient_plasticity_msg` | **WITHHELD** | Reduced higher-order gradient-plasticity codegen example with implementation checks. | “MSG/Taylor” is not a complete method citation. Add exact source equations, reduction/regularization statement, parameters, and port authority. |
| `thermo_mechanics_quad8` | **READY** *(codegen proof)* | Synthetic coupled displacement-temperature Quad8 with internal optional scalar-field/backend checks; those checks are not shipped as public release tests and were skipped in the current environment. | Equations and limited purpose are stated locally; no external data. |
| `tire_contact` | **RESEARCH** | Qualitative tire/contact/von-Mises study and an intentionally negative static-adjoint diagnostic. Only the torus mesh currently has a pytest gate. | Ships to show the layered mesh/solve/analysis/sensitivity workflow, not as validation. Add the exact GetFEM/Khenous source, upstream revision, reference parameters, retained comparison evidence, and a bounded full-solve gate before promotion. |
| `uel_scaffold_quad4` | **READY** *(codegen proof)* | Minimal generated Quad4 scaffold. Its separate Python-assembly check lives in the internal, unshipped validation registry and was not rerun in the current public audit. | Release only as an implementation example; the check does not independently validate F-bar literature fidelity. |

## External data boundary

A prior private history contains the external-solver ring table, and the
withheld `cattaneo_3d` directory contains two diagnostic files. None is present
in the public artifact:

- `cattaneo_3d/benchmark.json` — a small analytic-oracle record.
- `cattaneo_3d/coupfe_petsc_penalty200_shear6.json` — a diagnostic result whose
  README records `converged=false` for the capped normal increment.
- `ring_compress/abaqus_ring_compress_reactions.csv` — external-solver reference
  values with incomplete per-run provenance and unresolved redistribution
  authority.

The public RESEARCH workflow accepts an authorized external table only through
`COUPFE_RING_REFERENCE_CSV`. One project-authored local summary remains:

- `ring_compress/coupfe_ring_compress_summary.csv` — a summarized CoupFE result
  record without retained raw logs.

The summary ships only as historical context and is not release-validation
evidence. No proprietary `.inp` deck, third-party mesh, external-solver table,
paper PDF, presentation, or reference screenshot is intentionally distributed.
The four paper-form directories deliberately carry only project-authored model
source, generated kernels, reduced metadata, and scoped gates. Full Abaqus
decks, archived run sources, output databases, Gmsh adapters, meshes, NPZ
snapshots, and figure assets remain in the pinned companion repository or are
withheld when their artifact-level provenance is unresolved.
The four generic UMAT directories likewise carry only project-authored
declarations, generated source, documentation, and material-point/codegen
gates; no Abaqus model deck or solver output is bundled or claimed.
The repository-wide policy and third-party attribution are in
[`NOTICE`](../NOTICE).
