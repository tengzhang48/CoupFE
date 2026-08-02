# Extending CoupFE from existing formulations

CoupFE's build-time `coupfe.codegen` subsystem was mechanically ported from the
maintainer's [`abaqus_ufl`](https://github.com/tengzhang48/abaqus_ufl) project
and subsequently extended for native CoupFE execution. The declaration style
is conceptually inspired by UFL, but CoupFE does not depend on, implement, or
claim compatibility with UFL or FEniCSx. [`../CREDITS.md`](../CREDITS.md)
records the exact citation roles. This page records the reusable porting method
without publishing an inventory of unreleased directories or implying that
every historical experiment is a supported CoupFE capability.

The authoritative public boundary is:

- [`../examples/README.md`](../examples/README.md) for runnable entry points;
- [`../examples/REFERENCES.md`](../examples/REFERENCES.md) for attribution and
  evidence; and
- [`capabilities.md`](capabilities.md) for implemented support and limitations.

## What the history suggests

The research lineage has explored more formulations than the curated examples
currently show, including finite-strain and inelastic materials, mixed and
pressure-based elements, gels and swelling, phase-field models, diffusion and
electrochemical coupling, and thermo- and magneto-mechanical systems. That
breadth is useful evidence that the architecture can be extended; it is not a
claim that those models are all shipped, validated, or ready for use.

An additional model belongs in the public tree only when its formulation,
source authority, license, adaptation history, tests, and limitations can be
stated clearly. Detailed experimental inventories remain research history.

## How AI agents help

An AI coding agent can accelerate source review, API adaptation, generator
scaffolding, test construction, documentation, and repetitive cross-backend
checks. It does not supply redistribution rights, choose a physically correct
formulation, or turn implementation parity into validation. A useful agent
workflow therefore makes provenance and independent evidence explicit before
optimizing the code.

## Porting an element or material

1. **Establish authority and provenance.** Record the formulation source,
   copyright and license, permission to redistribute original or generated
   source, modifications, and any external data requirements. If that record is
   incomplete, reimplement from an authorized public method description or keep
   the experiment outside the public tree.
2. **Classify the interface.** Decide whether the source is a full element, a
   material-point law, a coupled form, or only a solver/input workflow. These
   map to different CoupFE seams and should not be presented as equivalent.
3. **Express one residual.** For a supported form, use `coupfe.codegen` to emit
   a native element kernel and, where useful, an Abaqus UEL from the same
   definition. Keep application-specific mesh, loading, and output code outside
   the kernel.
4. **Declare state explicitly.** History-free models may use no state variables.
   Path-dependent models must distinguish committed and trial state and update
   state only after an accepted increment. Verify the chosen driver explicitly:
   generic `solve_increments` is history-free, and `newton_solve` does not
   expose a convergence flag before calling commit.
5. **Wrap the element.** Load a compiled kernel with `CompiledElement`, passing
   explicit `props`, `dof_per_node`, `n_svars`, `mcrd`, and `n_elem`, plus the
   generated `state_schema` when applicable. Then place it in an
   `ElementGroup`, using `comps` when the element fields occupy only part of the
   global node-major layout.
6. **Add independent evidence.** Prefer an analytic solution, manufactured
   solution, patch/objectivity test, energy or work balance, or a published
   dataset with clear provenance. Native-versus-UEL or generated-versus-reference
   parity catches implementation drift but does not independently validate the
   model.
7. **Add a broken control.** Deliberately change a sign, state transition,
   ordering, or boundary condition and confirm that the proposed test fails.
8. **Test the distributed and packaged paths that you claim.** A source-tree
   import, a one-rank run, and an installed wheel are different environments.
   Check only the ones needed for the stated capability, but state the boundary.
9. **Document a narrow claim.** Say whether the result is an implementation
   proof, analytic validation, research demonstration, or external
   reproduction. Avoid turning a successful compile into a physics claim.

See [`../skills/model_development.md`](../skills/model_development.md) and
[`../skills/testing.md`](../skills/testing.md) for the detailed development and
verification checklists.

## Code generation and runtime boundary

`coupfe.codegen` is a build-time subsystem. Public entry points include
`Material`, `SmallStrainMaterial`, `WeakForm`, the field declarations,
`coupfe.codegen.generators.uel_gen.generate_element(..., backend="native")`,
`generate_uel(...)`,
`generate_umat(...)`, and `generate_small_strain_umat(...)`. The runtime
consumes generated kernels without importing SymPy.

CoupFE does not currently provide a generic host that takes an arbitrary
compiled Abaqus UMAT and turns it into a standalone structural element. The
material examples exercise declarations, generation, state updates, and
material-point oracles. A structural use still needs an element operator that
owns geometry, quadrature, assembly, and state layout.

The current distributed bulk path is a separate implementation and does not
provide generic committed-history or coupled-field support. Passing a serial
element test therefore does not establish distributed support.

Similarly, Core does not absorb every mesh or file-format adapter used during a
port. Application packages translate their own meshes, boundary semantics, and
tool outputs into the small Core contracts. Reusable integration code can later
become an optional package if multiple applications genuinely share it.
