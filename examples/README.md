# CoupFE examples

This index covers all and only the 35 example directories shipped in the public
tree. Examples have different purposes: runnable boundary-value problems,
build-time generators, analytic checks, MPI smoke programs, and research
workflows.

- Commands assume the repository root and `PYTHONPATH=.`.
- Code-generation examples usually use `build.py`; generation or compilation
  is not the same as solving a benchmark.
- Compiled examples require a compatible Fortran toolchain.
- MPI programs require a consistent PETSc, `petsc4py`, and `mpirun` stack.
- Some research workflows require licensed or user-supplied inputs that are not
  redistributed.

Two labels are used:

- **READY** — supported for the limited role stated in the table.
- **RESEARCH** — shipped to demonstrate an implemented path, without a general
  validation or production-support claim.

Generated-kernel/reference-assembly, complex-step/finite-difference, and
native/UEL parity checks establish implementation consistency when both sides
share a formulation. They are not independent physical validation. See
[`REFERENCES.md`](REFERENCES.md) for evidence and provenance details.

## Core finite-element examples

| Example | Status | Entry point and scope |
|---|---|---|
| `linear_bar` | **READY** | `run.py`; minimal nonlinear operator and analytic linear-limit checks. |
| `neo_hookean_block` | **READY** | `run.py`; compiled finite-strain block with a traction-free lateral-stretch oracle. |
| `neo_hookean_mixed` | **READY — codegen proof** | `build.py`; mixed Quad8/pressure declaration, verification-state tangent check, and generated UEL; no boundary-value patch solve. |
| `curved_annulus` | **READY** | `run.py`; curved-boundary convergence against the Lamé field. |
| `model_pipeline` | **READY** | `run.py`; declarative `Model` construction and solve path. |

## Element and material code generation

| Example | Status | Entry point and scope |
|---|---|---|
| `simple_gel_quad4` | **READY — codegen proof** | `build.py`; explicitly simplified synthetic displacement/chemical-potential form. |
| `gel_chester_anand` | **RESEARCH — paper form** | `u_p_mu_quad8/build.py`; mixed `u-p-mu` Quad8 declaration with focused implementation checks. |
| `neo_hookean_umat` | **READY — codegen proof** | `build.py`; finite-strain UMAT with independent material-point checks. |
| `ogden_umat` | **READY — codegen proof** | `build.py`; one-term compressible spectral Ogden UMAT with principal-stretch checks. |
| `small_strain_j2_umat` | **READY — codegen proof** | `build.py`; radial-return J2 UMAT with elastic/plastic path and state checks. |
| `small_strain_viscoelastic_umat` | **READY — codegen proof** | `build.py`; standard-linear-solid UMAT with tensor-history and relaxation checks. |
| `j2_plasticity_uel` | **READY — codegen proof** | `build.py`; deliberately small-strain state-schema proof. |
| `scalar_diffusion_uel` | **READY — codegen proof** | `build.py`; synthetic thermo-mechanical/Fourier coupling form. |
| `thermo_mechanics_quad8` | **READY — codegen proof** | `build.py`; synthetic displacement-temperature form. |
| `phasefield_fracture_uel` | **RESEARCH** | `build.py`; simplified AT2-style codegen demonstration, not fracture validation. |
| `morphing_hex8` | **RESEARCH — paper form** | `build.py`; pressure-gel `u-mu` Hex8 and a narrow reader for user-supplied mesh blocks. |
| `phasefield_corrosion_cui` | **RESEARCH — paper form** | `build.py`; `u-phi-c` Quad8R declaration with formulation-specific implementation checks. |
| `stabilized_tet4` | **RESEARCH — paper form** | `build.py`; stabilized `u-theta` Tet4 declaration with analytic and compiler checks. |
| `uel_scaffold_quad4` | **READY — codegen proof** | `build.py`; minimal generated Quad4 scaffold. |
| `neo_hookean_local_pressure_quad4` | **RESEARCH** | `build.py`; element-local condensed-pressure Quad4 demonstration. |
| `neo_hookean_local_pressure_hex8` | **RESEARCH** | `build.py`; element-local condensed-pressure Hex8 demonstration. |
| `neo_hookean_inelastic_local_pressure_quad4` | **RESEARCH** | `build.py`; inelastic-volume/local-pressure codegen demonstration. |

## Contact and friction

| Example | Status | Entry point and scope |
|---|---|---|
| `contact_3d_blocks` | **RESEARCH** | `run.py`; tested 3-D deformable collision path with optional acceleration. |
| `contact_3d_friction` | **RESEARCH** | `run.py`; tested 3-D smoothed-friction path with optional acceleration. |
| `hertz_contact` | **READY** | `run.py`; finite-block normal-contact force check against the stated Hertz law. `render.py` regenerates the solver-backed field/force figure; the README states the mesh and contact-radius limitations. |
| `ring_compress` | **RESEARCH** | `reproduce.py` / `reproduce_dynamics.py`; manual workflow requiring a user-supplied input deck. |
| `contact_vs_ppf` | **RESEARCH** | `ppf_reference.py` plus local comparison scripts; interoperability recipe, not retained quantitative evidence. |
| `compression_cylinders` | **RESEARCH** | `run.py`; qualitative many-body workflow requiring a user-obtained source deck. |
| `tire_contact` | **RESEARCH** | `mesh.py`, `run.py`, `analyze.py`, and `sensitivity.py`; layered capability study with only scoped public gates. |
| `exact_stick_friction` | **READY — study** | `run.py`; small-scale global Coulomb exact-stick and onset checks. |
| `semismooth_friction` | **READY — study** | `run.py`; small-scale per-node partial-slip semismooth solve. |
| `friction_identifiability` | **READY — study** | `run.py`; frozen-active-set derivative and slip-support checks. |
| `finite_sliding_friction` | **READY — study** | `run.py`; re-pairing, state transfer, dissipation, and reversal memory. |
| `finite_sliding_capstan` | **READY — study** | `run.py`; rotating friction frame checked against the capstan equation. |

## Distributed smoke programs

| Example | Status | Entry point and scope |
|---|---|---|
| `mpi_smoke` | **READY — smoke collection** | Rerunnable bulk, contact, dynamics, friction, and serial-versus-rank diagnostics; not an external physics benchmark or a published scaling record. |

For example:

```bash
OMP_NUM_THREADS=1 mpirun -n 4 \
  python examples/mpi_smoke/distributed_solve.py
```

The directory also contains bulk, contact, dynamics, friction, and
serial-versus-rank diagnostic programs. Qualify them in the exact PETSc/MPI
environment you intend to use; the current public release does not include a
retained final-revision multi-rank result.
