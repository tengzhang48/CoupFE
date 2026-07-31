# CoupFE — installation and PETSc/MPI environments

The base package needs Python, NumPy, and SciPy. Code generation additionally
needs SymPy. Install a compatible Fortran compiler before running compiled
element tests; compiled elements also need Meson and Ninja. The recommended
development install is:

```bash
git clone https://github.com/tengzhang48/CoupFE.git
cd CoupFE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,runtime,codegen,performance]"
python -m pytest -q -m "not slow"
```

For PETSc/MPI, use one conda-forge stack throughout the environment; mixing a
pip-built PETSc with conda MPI libraries is unsupported. One portable
conda-forge setup is:

```bash
mamba create -n coupfe -c conda-forge \
  python=3.11 numpy scipy sympy pytest meson ninja compilers \
  petsc petsc4py mpich numba
mamba activate coupfe
python -m pip install -e ".[dev,runtime,codegen,performance]"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  mpirun -n 4 python examples/mpi_smoke/distributed_solve.py
```

Use the activated conda environment for both Python and `mpirun`; this keeps
PETSc, petsc4py, and MPI on one compatible stack. Do not install the `[petsc]`
pip extra into that environment, because conda-forge already owns those
packages. The extra exists for non-conda installations.

Pin `OMP_NUM_THREADS=1` under MPI, and normally pin the BLAS thread variables as
shown above as well. Otherwise ranks multiply the underlying thread count and a
healthy solve can appear to scale poorly.

## PETSc / MPI practical notes

A few choices avoid common environment problems:

- Keep PETSc, petsc4py, and the MPI launcher from one compatible
  distribution. Optional solvers such as SuperLU_DIST, MUMPS, and hypre depend
  on how PETSc was built. Use `PETSc.Sys.hasExternalPackage(...)` and the PETSc
  configuration output instead of assuming every distribution enables the same
  packages; record `PETSc.Sys.getVersionInfo()` separately for the version.
- Choose a solver from the options actually available in the environment and
  retain the configuration when reporting performance or reproducibility.
  Near-null or poorly conditioned contact modes can affect any factorization or
  iterative method and should be diagnosed as a model/solver issue, not hidden
  by changing packages.
- The example environment above explicitly installs MPICH. Use flags supported
  by that launcher and start with `mpirun -n N` where `N` does not exceed the
  allocated cores.
- Set `OMP_NUM_THREADS=1` under `mpirun` (and normally the BLAS thread variables
  too) so each MPI rank does not create another full set of worker threads.
- f2py's meson backend invokes `meson` and `ninja` as commands, so their
  executables must be on `PATH`. Code generation also uses
  `inspect.getsource`; put weak-form definitions in a real Python module rather
  than a REPL or heredoc.

## Application environments

CoupFE-EDA, CoupFE-Cardiac, and other consuming applications document their own
dependencies and setup paths. The common rule is to keep each environment on
one compatible PETSc/MPI stack and record that stack with any distributed
result. An application's optional container recipe is not a Core requirement.
