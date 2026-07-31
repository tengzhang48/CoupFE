# CoupFE distributed-development guide

Use this guide when changing the PETSc/MPI paths. Read it with
[`docs/capabilities.md`](../docs/capabilities.md),
[`docs/install.md`](../docs/install.md), and
[`skills/testing.md`](testing.md). The capability table is authoritative: this
guide explains how to extend and qualify the implementation, not what has
already been qualified on every environment.

## Scope

CoupFE's distributed code targets compact array meshes and regular or
block-structured partitions. Core provides deterministic partition helpers and
memory-local patterns for structured cases. It does not provide general graph
partitioning, parallel adaptive refinement, hanging-node constraints, or a
universal DMPlex/Gmsh adapter.

Applications with general geometry should translate their mesh and labels into
`KernelMeshView` or an equally small array contract. Keep CAD and domain
semantics with the application.

## Qualification gates

Treat serial-versus-rank agreement as an implementation check, not a physics
oracle. Qualify a new distributed path in layers:

1. Partition in process and verify ownership, ghost data, and reduction against
   the serial assembly. `tests/test_distribute.py` exercises the shipped helper
   layer.
2. Run the relevant program in `examples/mpi_smoke/` at more than one rank
   count in the target PETSc/MPI environment.
3. Compare residuals, tangents, solution fields, and iteration behavior using
   tolerances appropriate to the selected linear solver.
4. Record the exact revision, PETSc configuration, MPI implementation, solver,
   tolerances, rank/thread counts, command, and raw output before publishing a
   qualification or performance claim.

The repository ships rerunnable MPI programs but does not currently publish a
retained final-revision multi-rank qualification record.

## PETSc assembly rules

- Use global indices with `ADD_VALUES`, then assemble the PETSc matrix and
  vector. PETSc owns off-process accumulation.
- Apply symmetric Dirichlet elimination when the Krylov method requires a
  symmetric operator.
- Invoke collectives on every rank, including ranks with no local contact
  entities. Guard only the local data insertion.
- Keep PETSc, petsc4py, and the launcher in one compatible environment.
- Default to one OpenMP/BLAS thread per MPI rank unless a retained profile
  supports another setting.
- Query optional packages from the active PETSc build rather than assuming a
  particular direct solver or preconditioner is present.

## Distributed bulk path

The serial `Operator` loop consumes global vectors and is not itself the
distributed unit. `solve_distributed(...)` instead consumes a batched element
evaluator and each rank's element coordinates and global DOF map. This keeps
element evaluation local while PETSc owns the global sparse system.

Two mesh sources are useful for different purposes:

- build then partition, which preserves the serial global numbering and is
  convenient for direct comparisons; and
- memory-local structured generation, which avoids allocating the full mesh on
  every rank.

A new history-free element can reuse the bulk path when it satisfies the same
batch interface. Path-dependent state, coupled-field orchestration, and affine
constraints need additional distributed ownership and commit designs; do not
infer those capabilities from a passing hyperelastic run.

## Contact paths

Rigid analytical contact is node-local: assign each contact node to one rank,
assemble its contribution once, and keep any supported node state with that
owner.

Deformable contact couples surface entities that may live on different ranks.
The current implementation replicates the contact-surface coordinates, assigns
each candidate pair to one owner, assembles with global indices, and reduces
the collision step bound globally. This can be appropriate when the surface is
small relative to the bulk, but it is not a general large-interface scaling
claim.

Three-dimensional vertex-face and edge-edge paths, friction, and distributed
dynamics have runnable smoke programs. Their support differs by formulation;
consult the contact matrix in `docs/capabilities.md` before describing a mode
as distributed.

The dense-Schur semismooth friction study is suitable only for a small
interface. Replicating a dense interface operator is not the project's general
parallel contact strategy.

## Preconditioning

Correct assembly and scalable convergence are separate problems. For a new
preconditioner:

- scale equations before interpreting residuals or Krylov convergence;
- use FieldSplit only with verified field index sets and appropriate block
  solvers;
- provide rigid-body near-null-space information for elasticity blocks when
  the chosen multigrid method needs it;
- test iteration trends across mesh sizes and rank counts; and
- retain the matrix/problem definition with any performance result.

An exact or tightly solved direct system can be useful for a small
serial-versus-rank gate. Larger runs usually need iterative methods, in which
case agreement is limited by the solver tolerances and conditioning. Backend
behavior is environment-specific; rerun the gate with the actual PETSc build.

## Debugging sequence

When a distributed result changes unexpectedly, vary one factor at a time:

1. rank count;
2. linear solver or preconditioner;
3. thread count;
4. contact or state mode; and
5. build-then-partition versus memory-local generation.

First compare assembled values and ownership, then the linear solve, then the
nonlinear trajectory. A plausible explanation is not evidence until a targeted
check distinguishes it from the alternatives.
