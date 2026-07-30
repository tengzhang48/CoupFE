# Mixed-order gel bilayer element

This research example contains the CoupFE form of the mixed
displacement–pressure–chemical-potential Quad8 element used for Section 3.3 /
Figure 5 of the `abaqus_ufl` paper. Displacement and chemical potential are
quadratic on all eight nodes; pressure is bilinear on the four corner nodes,
giving 28 element DOFs.

Generate the UEL from the repository root:

```bash
PYTHONPATH=. python examples/gel_chester_anand/u_p_mu_quad8/build.py
```

The generated source is
`u_p_mu_quad8/chester_anand_upmu_quad8_uel.for`. Focused tests cover material
verification, native/UEL parity, independent Python reference assembly, and
compiled execution.

## Paper and source lineage

The coupled gel theory and swell-induced-bending benchmark follow:

Shawn A. Chester, Claudio V. Di Leo, and Lallit Anand, “A finite element
implementation of a coupled diffusion-deformation theory for elastomeric
gels,” *International Journal of Solids and Structures* 52 (2015), 1–18.
[doi:10.1016/j.ijsolstr.2014.08.015](https://doi.org/10.1016/j.ijsolstr.2014.08.015)

The pressure-based CoupFE declaration is project-authored and was published
under the MIT License in
[`abaqus_ufl` at commit `0f52533`](https://github.com/tengzhang48/abaqus_ufl/tree/0f525339db1aad70e9f8f4825a02c1164f0da7a0/paper_examples/gel_bilayer).
The complete distinction between project source, benchmark attribution, and
the separately obtained supplemental mesh seed is in the companion
repository’s
[`CREDITS.md`](https://github.com/tengzhang48/abaqus_ufl/blob/0f525339db1aad70e9f8f4825a02c1164f0da7a0/CREDITS.md).

The accepted bilayer deck in the companion repository was written by this
project. Its mesh discretization follows the Chester–Di Leo–Anand
supplemental example. Their original, unlicensed supplemental files are not
redistributed, and the external mesh seed is not copied into CoupFE.

## Evidence boundary

This directory is **RESEARCH**. Its tests establish formulation and backend
implementation consistency; they do not rerun the 2160-increment Abaqus
bilayer analysis or independently validate Figure 5. The paper deck also uses
a slider MPC and frictionless hard contact. A future native CoupFE
gel-plus-contact reproduction belongs in an application workflow after a
fresh full-solve gate, while mesh adaptation remains outside Core.
