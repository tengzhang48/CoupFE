# Citation, lineage, and credits

This page distinguishes the citation for CoupFE itself from the software,
methods, and interfaces that shaped particular parts of the project. A citation
does not imply that CoupFE contains another project's source, implements its
full interface, or has been endorsed by its authors.

Legal notices and license scope remain authoritative in [`NOTICE`](NOTICE),
[`LICENSE`](LICENSE), and the identified exception licenses. Scientific and
software references do not replace those records.

## Cite CoupFE itself

Use [`CITATION.cff`](CITATION.cff) for the primary software citation and report
the exact release or commit used. CoupFE does not currently have a dedicated
paper, and users should not cite a related paper as though it documented every
CoupFE subsystem. An archival DOI can be added to the CFF record after a
versioned software deposit is made.

Suggested text for the current public version:

> Teng Zhang, *CoupFE: a compact finite-element scaffold for custom operators
> and generated elements*, version 0.0.1, software,
> <https://github.com/tengzhang48/CoupFE>.

## Generated-element lineage: `abaqus_ufl`, Abaqus, and UFL

CoupFE's build-time `coupfe.codegen` subsystem was mechanically ported from the
maintainer's [`abaqus_ufl`](https://github.com/tengzhang48/abaqus_ufl) project
and subsequently extended for native CoupFE execution. Eight shipped examples
also retain a pinned `abaqus_ufl` source record in [`NOTICE`](NOTICE) and
[`examples/REFERENCES.md`](examples/REFERENCES.md).

For work that materially uses this declaration/code-generation path or the
ported paper examples, cite CoupFE and also cite:

- Teng Zhang, *Making coupled-field Abaqus user elements simple: automatic
  generation with complex-step tangents*, manuscript submitted for
  publication, 2026. This record should be replaced with the journal or
  preprint citation when a public identifier is available.
- Teng Zhang, *abaqus_ufl: coupled-field Abaqus user elements from declared
  weak forms*, version 0.1.0, software,
  <https://github.com/tengzhang48/abaqus_ufl>.

The declaration style is conceptually inspired by the Unified Form Language
(UFL), but CoupFE does not depend on, implement, or claim compatibility with
UFL or FEniCSx. For that intellectual context:

- Martin S. Alnæs, Anders Logg, Kristian B. Ølgaard, Marie E. Rognes, and
  Garth N. Wells, "Unified Form Language," *ACM Transactions on Mathematical
  Software* 40 (2014), DOI
  [`10.1145/2566630`](https://doi.org/10.1145/2566630).
- Igor A. Baratta, Joseph P. Dean, Jørgen S. Dokken, Michal Habera, Jack S.
  Hale, Chris N. Richardson, Marie E. Rognes, Matthew W. Scroggs, Nathan Sime,
  and Garth N. Wells, *DOLFINx: The next generation FEniCS problem solving
  environment* (2023), DOI
  [`10.5281/zenodo.10447666`](https://doi.org/10.5281/zenodo.10447666).

Abaqus is an external execution target and comparison environment, not a
CoupFE dependency. CoupFE does not distribute Abaqus itself, proprietary Abaqus
source, or SIMULIA example decks. When reporting an Abaqus-backed result, cite
the exact Abaqus release used and the applicable interface reference, for
example these Abaqus 2024 UEL/UMAT reference pages:

- Dassault Systèmes, *Abaqus 2024 User Subroutines Guide: UEL*,
  <https://docs.software.vt.edu/abaqusv2024/English/SIMACAESUBRefMap/simasub-c-uel.htm>.
- Dassault Systèmes, *Abaqus 2024 User Subroutines Guide: UMAT*,
  <https://docs.software.vt.edu/abaqusv2024/English/SIMACAESUBRefMap/simasub-c-umat.htm>.

These documentation links identify the interfaces; they do not claim that all
public CoupFE examples were qualified with Abaqus 2024.

## Contact lineage: `ppf-contact-solver`

Portions of CoupFE's contact geometry, collision detection, and smoothed
friction implementation are modified adaptations of Ryoichi Ando's
Apache-2.0-licensed
[`ppf-contact-solver`](https://github.com/st-tech/ppf-contact-solver). The exact
file roles, copyright notice, and reproducible upstream commit are recorded in
[`NOTICE`](NOTICE).

For work that materially uses those contact paths, cite CoupFE and also cite:

- Ryoichi Ando, *ZOZO's Contact Solver*, software, released 2024-11-01,
  <https://github.com/st-tech/ppf-contact-solver>. CoupFE records upstream
  commit `8b7740b032131aeeb46f51d882c96e09b171acc8` for reproducible inspection.
- Ryoichi Ando, "A Cubic Barrier with Elasticity-Inclusive Dynamic
  Stiffness," *ACM Transactions on Graphics* 43(6) (2024), 1–13, DOI
  [`10.1145/3687908`](https://doi.org/10.1145/3687908).
- Minchen Li, Zachary Ferguson, Teseo Schneider, Timothy Langlois, Denis
  Zorin, Daniele Panozzo, Chenfanfu Jiang, and Danny M. Kaufman, "Incremental
  Potential Contact: Intersection- and Inversion-free, Large-Deformation
  Dynamics," *ACM Transactions on Graphics* 39(4) (2020), DOI
  [`10.1145/3386569.3392425`](https://doi.org/10.1145/3386569.3392425).

These papers provide method context. CoupFE adapts selected algorithms and does
not claim numerical identity with the upstream solver or the full IPC method.

## Numerical infrastructure

CoupFE directly depends on NumPy and SciPy and has optional PETSc/petsc4py and
MPI execution paths. Publications that rely materially on those packages
should follow each project's current citation guidance. Common references are:

- Charles R. Harris et al., "Array programming with NumPy," *Nature* 585
  (2020), 357–362, DOI
  [`10.1038/s41586-020-2649-2`](https://doi.org/10.1038/s41586-020-2649-2).
- Pauli Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific
  computing in Python," *Nature Methods* 17 (2020), 261–272, DOI
  [`10.1038/s41592-019-0686-2`](https://doi.org/10.1038/s41592-019-0686-2).
- PETSc, [current citation guidance](https://petsc.org/release/#citing-petsc).
  PETSc-enabled runs can also print configuration-specific references with the
  PETSc `-citations` option.

The per-example formulation, benchmark, data, and evidence references remain
in [`examples/REFERENCES.md`](examples/REFERENCES.md).
