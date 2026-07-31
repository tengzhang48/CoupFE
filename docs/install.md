# CoupFE — installation and PETSc/MPI environments

The base package needs Python, NumPy, and SciPy. Code generation additionally
needs SymPy; compiled elements need a Fortran compiler, meson, and ninja. The
recommended development install is:

```bash
git clone https://github.com/tengzhang48/CoupFE.git
cd CoupFE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,runtime,codegen,performance]"
python -m pytest -q -m "not slow"
```

Install a system `gfortran` first when using compiled elements. For PETSc/MPI,
use one conda-forge stack throughout the environment; mixing a pip-built PETSc
with conda MPI libraries is unsupported. One portable conda-forge setup is:

```bash
mamba create -n coupfe -c conda-forge \
  python=3.11 numpy scipy sympy pytest meson ninja compilers \
  petsc petsc4py mpi4py mpich numba
mamba activate coupfe
python -m pip install -e ".[dev,runtime,codegen,performance]"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  mpirun -n 4 python examples/mpi_smoke/distributed_solve.py
```

Use the activated conda environment for both Python and `mpirun`; this keeps
PETSc, petsc4py, mpi4py, and MPI on one compatible stack. Do not install the
`[petsc]` pip extra into that environment, because conda-forge already owns
those packages. The extra exists for non-conda installations.

Pin `OMP_NUM_THREADS=1` under MPI, and normally pin the BLAS thread variables as
shown above as well. Otherwise ranks multiply the underlying thread count and a
healthy solve can appear to scale poorly.

## PETSc / MPI best practices (do this; avoid that)
A few choices remove most of the friction. They are general — not specific to any one model.

- **Prefer conda-forge `petsc4py` over the pip wheel.** The conda-forge build bundles **`superlu_dist`**
  (the reproducible parallel *direct* solver — see below) and **`hypre`** (BoomerAMG). The pip wheel
  has **neither**: it is iterative-only (no `superlu_dist`) and `PCSetType('hypre')` errors. If you only
  have the pip wheel, **GAMG + the rigid-body near-null-space** gives the same elasticity scalability as
  hypre (see `skills/distributed.md`), so it's livable — but conda-forge is the path of least surprise.
- **Distributed direct solver = `superlu_dist`, never MUMPS.** MUMPS's parallel pivoting is
  run-to-run non-reproducible; an ill-conditioned/near-null contact mode amplifies it. `superlu_dist`
  is reproducible to machine precision (and conda-forge-only).
- **Know your MPI flavor — conda-forge pulls MPICH, not OpenMPI.** Practical consequences: MPICH
  **oversubscribes by default** and **rejects** `--oversubscribe` / `--allow-run-as-root` — so just use
  plain `mpirun -n N` with `N ≤ physical cores`. A launch that produces **silent empty output** is
  usually a rejected flag swallowing stderr — re-run without flag-suppression to see the error.
- **`OMP_NUM_THREADS=1` under `mpirun`** (and `OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`) — else
  ranks × BLAS-threads oversubscribe the cores and a healthy solve looks like "doesn't scale."
- **f2py's meson backend is invoked as a *command*.** `pip install meson ninja` **and** put the env's
  `bin/` on `PATH` — an importable `mesonbuild` is *not* enough (f2py shells out to `meson`). Also: the
  codegen reads weak-form bodies via `inspect.getsource`, so a weak form must live in a real `.py`
  file, **not** a heredoc / REPL string.

## Sibling app repos define their own env (same rule)
The applications are separate repos that depend on the CoupFE core:
- **`CoupFE-EDA`** (`coupfe-eda` package) ships `environment.yml` (conda-forge, python 3.11,
  conda-forge `petsc4py` for the distributed PDN) + a `Dockerfile`. Its `setup.sh` clones CoupFE
  over public HTTPS, verifies that the configured public branch contains the pinned commit, checks
  out that commit detached, and installs the resulting local checkout.
- **`CoupFE-Cardiac`** uses the same single-stack conda-forge approach.

**Principle for the whole CoupFE family:** every repo uses **conda-forge `petsc4py`**
(superlu_dist); pip-PETSc is only for the JAX side; **never mix the two in one env.**
