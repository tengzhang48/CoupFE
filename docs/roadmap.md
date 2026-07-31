# CoupFE roadmap

This roadmap describes direction, not shipped capability. The current support
boundary is authoritative in [`capabilities.md`](capabilities.md).

## Near term

1. **Keep the public surface internally consistent.** Maintain API,
   capability, example, provenance, and packaging checks together as behavior
   changes.
2. **Retain reproducible distributed evidence.** Qualify representative MPI
   bulk, dynamics, and contact paths on the exact release revision, with the
   environment and raw output recorded.
3. **Strengthen example evidence.** Add independent oracles, convergence checks,
   and broken controls where a shipped research demonstration is promoted to a
   stronger claim.
4. **Harden application boundaries.** Keep mesh and domain adapters outside
   core while improving the small `KernelMeshView`, geometry, and constraint
   contracts they consume.

## Subsequent priorities

- distributed state commit for path-dependent elements;
- broader coupled-field preconditioning and distributed qualification;
- deeper self-contact and friction qualification, including bounded
  end-to-end tests;
- additional native element geometries where public examples require them;
- clearer interoperability packages when independently maintained applications
  demonstrate repeated adapter needs; and
- benchmark artifacts that report convergence, hardware, software versions,
  memory, and solver settings together.

## Gated work

GPU kernels, matrix-free Jacobian actions, very-large-mesh workflows, and more
general geometry backends remain gated by a demonstrated application need and a
credible solver or preconditioning plan. They should not be presented as
current capabilities until an implementation and reproducible evidence ship.

AI-assisted development may speed implementation and review, but does not
change the acceptance standard: domain assumptions, provenance, independent
validation, and executable tests remain required.
