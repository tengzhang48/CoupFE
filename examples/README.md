# CoupFE examples — index

The example tree contains several kinds of material: end-to-end `run.py`
drivers, build-time element generators, analytic benchmarks, external-solver
reproductions, MPI smoke scripts, and research diagnostics. They do not all have
the same entry point or evidence level.

- Run commands below assume the repository root and `PYTHONPATH=.`.
- A script that prints `OK` / `FAIL` is not necessarily a pytest gate.
- Codegen examples generally use `build.py`; many generate a kernel rather than
  solve a boundary-value problem.
- Compiled examples need `gfortran`. MPI examples need a matched `mpirun` /
  `petsc4py` stack, normally with `OMP_NUM_THREADS=1`.
- External reproductions can require licensed, user-supplied inputs and may skip
  in the test suite when those inputs are absent.

**[ready]** means supported in the first-release example set for the limited
role stated here. **[research]** means the example ships to demonstrate an
implemented capability but is not a supported validation claim.
**[withheld]** means its source/authority, licensing record, or known release
gate must be resolved before it enters the first public snapshot. See the complete
[`REFERENCES.md`](REFERENCES.md) ledger before citing an example as validation.

Implementation checks also need careful language: complex-step versus finite
difference, generated kernel versus reference assembly, and native versus
Abaqus-UEL backend parity can verify implementation consistency. When the two
sides share the same weak form, they are **not independent physics validation**.

## Core FE

| Example | Release | Entry point and evidence |
|---|---|---|
| `linear_bar` | **[ready]** | `run.py`; analytic linear limit and tangent are gated in `tests/test_operator_contract.py`. |
| `neo_hookean_block` | **[ready]** | `run.py`; traction-free lateral stretch and compiled-element paths are gated. |
| `neo_hookean_mixed` | **[ready, codegen proof]** | `build.py`; mixed Quad8/pressure patch implementation tests. This is not a general no-locking claim. |
| `curved_annulus` | **[ready]** | `run.py`; exact Lamé-field convergence is gated in `tests/test_curved_convergence.py`. |
| `model_pipeline` | **[ready]** | `run.py`; the exact script is not a subprocess test, but `tests/test_model.py` and `tests/test_pipeline.py` gate the demonstrated API path. |

## Element / codegen examples

The usual entry point is `PYTHONPATH=. python examples/<name>/build.py`.
`gel_chester_anand` uses
`examples/gel_chester_anand/u_p_mu_quad8/build.py`. These scripts generally
generate or compile an element; they do not all self-report a solved benchmark.

| Example | Release | Scope |
|---|---|---|
| `Fbar_uel` | **[withheld]** | F-bar citation and public port/generated-source provenance remain open. |
| `simple_gel_quad4` | **[ready, codegen proof]** | Explicitly simplified synthetic coupled-field example. |
| `gel_axisymmetric_quad8` | **[withheld]** | Constitutive/formulation citations and port provenance remain open. |
| `gel_chester_anand` | **[research, paper form]** | Mixed-order `u-p-mu` Quad8 declaration from paper Sec. 3.3/Fig. 5; focused formulation/backend gates pass. The external supplemental mesh seed and full Abaqus reproduction stay outside Core. |
| `gel_chester_anand_local_pressure_quad4` | **[withheld]** | Complete source, modifications, and port provenance remain open. |
| `gel_three_field_hex20` | **[withheld]** | Constitutive/mixed-form citations and provenance remain open. |
| `neo_hookean_umat` | **[ready, codegen proof]** | Stateless finite-strain UMAT with closed-form uniaxial and simple-shear oracles, deterministic regeneration, and a compiler gate. |
| `ogden_umat` | **[ready, codegen proof]** | One-term compressible spectral Ogden UMAT with principal-stretch oracles, repeated-eigenvalue coverage, deterministic regeneration, and a compiler gate. |
| `small_strain_j2_umat` | **[ready, codegen proof]** | Stateful radial-return J2 UMAT with elastic/plastic path oracles, state-update checks, deterministic regeneration, and a compiler gate. |
| `small_strain_viscoelastic_umat` | **[ready, codegen proof]** | Standard-linear-solid UMAT with tensor history, relaxation/path oracles, deterministic regeneration, and a compiler gate. |
| `j2_plasticity_uel` | **[ready, codegen proof]** | Deliberately small-strain state-schema proof, not a research-grade finite-strain model. |
| `j2_fefp_uel` | **[withheld]** | Public formulation citation and private-source port authority remain open; `run.py` is a separate study driver. |
| `scalar_diffusion_uel` | **[ready, codegen proof]** | Synthetic thermo-mechanical/Fourier coupling example. |
| `thermo_mechanics_quad8` | **[ready, codegen proof]** | Synthetic generic scalar-field/codegen example. |
| `lce_quad4` | **[withheld]** | Exact Jiang source, reduction statement, parameters, and port authority remain open. |
| `li_2026_battery` | **[withheld]** | Exact Li source, reduction statement, parameters, and port authority remain open. |
| `phasefield_fracture_uel` | **[research]** | Simplified AT2-style codegen demo; complete source and simplification statement remain open. |
| `morphing_hex8` | **[research, paper form]** | Pressure-gel `u-mu` Hex8 with condensed local pressure from paper Sec. 3.4/Fig. 6; generated-UEL gates plus an example-local, read-only extractor for user-supplied pasta decks. The extractor separates U3 and companion C3D8 blocks but does not translate Abaqus contact or step semantics. |
| `phasefield_corrosion_cui` | **[research, paper form]** | Project-authored `u-phi-c` Quad8R declaration from paper Sec. 3.1/Fig. 3, with focused implementation gates. The Cui-derived comparison mesh/deck remains excluded pending artifact-level provenance. |
| `stabilized_tet4` | **[research, paper form]** | Scovazzi-style stabilized `u-theta` Tet4 declaration from paper Sec. 3.2/Fig. 4; analytic, tangent, generation, and compile gates. No Gmsh or Abaqus dependency is added to Core. |
| `strain_gradient_plasticity_msg` | **[withheld]** | Complete MSG source, reduction/regularization record, and provenance remain open. |
| `hussein_2026_ductile_pff` | **[withheld]** | Complete paper, equations/parameters, and port authority remain open. |
| `hussein_2026_mediavilla_pff` | **[withheld]** | Complete Hussein and Mediavilla sources and port authority remain open. |
| `uel_scaffold_quad4` | **[ready, codegen proof]** | Minimal generator/scaffold and backend implementation check. |
| `neo_hookean_local_pressure_quad4` | **[research]** | Mixed/mean-dilatation codegen demo; source/original-derivation record and scoped evidence remain open. |
| `neo_hookean_local_pressure_hex8` | **[research]** | 3D mixed/mean-dilatation codegen demo; source/original-derivation record and scoped evidence remain open. |
| `neo_hookean_inelastic_local_pressure_quad4` | **[research]** | Inelastic-volume codegen demo; formulation/original-derivation record remains open. |

## Contact — smoothed production path

| Example | Release | Entry point and evidence |
|---|---|---|
| `contact_3d_blocks` | **[research]** | `run.py`; two F-bar Hex8 blocks collide through the 3-D contact path, with optional numba/LBVH acceleration, and the end-to-end driver is gated by `tests/test_examples_contact_3d.py`; `NOTICE` pins a reproducible corresponding upstream snapshot without overstating the unknown historical checkout. |
| `contact_3d_friction` | **[research]** | `run.py`; gravity-seated blocks and smoothed friction, with optional numba/LBVH acceleration, and the end-to-end driver is gated by `tests/test_examples_contact_3d.py`; `NOTICE` pins a reproducible corresponding upstream snapshot without overstating the unknown historical checkout. |
| `self_contact_friction` | **[withheld]** | `run.py`; smaller self-contact gates pass, but the end-to-end subprocess gate is a strict expected timeout failure at 600 seconds, so this directory is excluded from the first release. |

## Contact — analytic and external-reference cases

| Example | Release | Entry point and evidence |
|---|---|---|
| `hertz_contact` | **[ready]** | `run.py`; quantitative force-law exponent and a loose coarse-mesh prefactor band are gated in `tests/test_hertz_contact.py`. |
| `ring_compress` | **[research]** | `reproduce.py` / `reproduce_dynamics.py`; ships to show the workflow. The deck is licensed/user-supplied, and the external reaction table is excluded because redistribution authority is unresolved; an authorized table may be supplied through `COUPFE_RING_REFERENCE_CSV`. |
| `contact_vs_ppf` | **[research]** | CoupFE and GPU-side scripts document an interoperability recipe against the reproducible upstream snapshot named in `NOTICE`; no retained external run log supports a quantitative comparison claim. |
| `compression_cylinders` | **[research]** | `run.py`; qualitative serial-subset/many-body capability demo requiring a user-fetched Abaqus deck for source geometry. |
| `cattaneo_3d` | **[withheld]** | Setup/oracle checks pass, but the reviewed end-to-end Cattaneo acceptance gate is a strict expected failure and the retained solve is non-converged. |
| `tire_contact` | **[research]** | Layered `mesh.py` / `run.py` / `analyze.py` / `sensitivity.py` capability study. Only the torus mesh currently has a pytest gate. |

## Contact — exact-stick / dual-multiplier studies

These are small-scale studies, not the smoothed production path. Method lineage
and limitations are in `docs/dev/dual_multiplier_strategy.md`.

| Example | Release | Entry point and evidence |
|---|---|---|
| `exact_stick_friction` | **[ready, study]** | `run.py`; global Coulomb, machine-zero stick, and analytic onset gate. |
| `semismooth_friction` | **[ready, study]** | `run.py`; per-node partial slip and Schur-condensed interface gate. |
| `friction_identifiability` | **[ready, study]** | `run.py`; frozen-active-set adjoint versus finite differences and slip-support gate. |
| `finite_sliding_friction` | **[ready, study]** | `run.py`; flat re-pairing, state-transfer, dissipation, and reversal-memory gates. |
| `finite_sliding_capstan` | **[ready, study]** | `run.py`; rotating friction frame checked against `T/T₀ = exp(μθ)`. |

## Distributed (MPI) — `mpi_smoke`

Run an individual script with, for example:

```bash
OMP_NUM_THREADS=1 mpirun -n 4 \
  python examples/mpi_smoke/distributed_solve.py
```

The collection is **[ready]** as MPI smoke and diagnostic material, not as an
external physics benchmark. `tests/test_mpi_distributed.py` invokes the
serial-versus-N-rank gates for:

- bulk/load paths: `distributed_residual`, `distributed_solve`,
  `distributed_neohookean`, `distributed_robin_pressure`;
- contact primitives and solves: `distributed_broadphase`,
  `distributed_deformable_residual`, `distributed_deformable_barrier`,
  `distributed_deformable_solve`, `distributed_3d_primitives`;
- dynamics/friction: `distributed_friction`,
  `distributed_dynamics_barrier`, `distributed_dynamics_friction`,
  `distributed_dynamics_3d_blocks`, `distributed_dynamics_3d_friction`; and
- exact-stick bulk recovery: `distributed_dual_multiplier`.

`distributed_lid_walls.py` has a focused gate in
`tests/test_lid_walls_contact.py`. `distributed_cylinders.py` is a many-body
diagnostic companion and is not itself a direct MPI pytest case.
