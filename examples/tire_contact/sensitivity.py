"""Diagnose a static-adjoint assumption after a barrier-dynamics tire solve.

The script forms a static bulk/contact tangent at the final dynamic state,
computes a compliance derivative, and compares it with finite differences of
the complete re-solve. A static adjoint assumes that the sampled state satisfies
the static residual equation. The reported ``|R_static(U*)|`` checks that
precondition before the gradient is interpreted.

When that residual remains nonzero, disagreement is the expected diagnostic:
the static adjoint is not a derivative of the time-stepping problem. A dynamic
adjoint would need to differentiate the complete trajectory. The separate
``friction_identifiability`` example studies a small frozen-active-set static
system. Neither study is a validated tire sensitivity prediction; see
``examples/REFERENCES.md``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from coupfe.assembly.assemble import assemble_residual, assemble_tangent, _apply_dirichlet
from coupfe.mesh import KernelMeshView
from coupfe.operators.element_group import ElementGroup
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

from examples.tire_contact.run import solve_tire, K_BULK, _UP_HEX8_FOR

# A small mesh keeps the three solves tractable. These settings define the
# diagnostic; they are not a mesh-convergence or quasistatic certificate.
DAMP_SETTLE, STEPS_SETTLE = 2.0, 110
N_PHI_S, N_THETA_S, N_RHO_S = 24, 10, 2


def _compliance(U, fext):
    return float(fext @ U)


def _bulk_residual(nodes, elems, U, G, ndof):
    view = KernelMeshView(nodes, elems, dof_per_node=3)
    elem = CompiledElement(build_element_kernel(_UP_HEX8_FOR, "tire_up_sens"),
                           props=(G, K_BULK), dof_per_node=3, n_svars=1, mcrd=3, n_elem=len(elems))
    grp = ElementGroup.from_view(view, elem, comps=(0, 1, 2))
    R, _ = assemble_residual([grp], U, None, 1.0, 1.0, ndof)
    return R


def adjoint_dC_dG(base, G, dG_rel=1e-3):
    U, nodes, elems = base["U"], base["nodes"], base["elems"]
    ndof, ops, con, fext = base["ndof"], base["static_ops"], np.array(base["rim_con"], int), base["fext"]

    R_static, _ = assemble_residual(ops, U, None, 1.0, 1.0, ndof)
    K = assemble_tangent(ops, U, None, 1.0, 1.0, ndof)
    K_bc, _ = _apply_dirichlet(K.copy(), np.zeros(ndof), con)

    dC_dU = fext.copy()
    dC_dU[con] = 0.0
    lam = spla.spsolve(K_bc.T.tocsc(), dC_dU)

    dG = dG_rel * G
    dR_dG = (_bulk_residual(nodes, elems, U, G + dG, ndof)
             - _bulk_residual(nodes, elems, U, G - dG, ndof)) / (2.0 * dG)
    dR_dG[con] = 0.0
    return float(-lam @ dR_dG), float(np.linalg.norm(R_static))


def _solve_settled(G=None, n_steps=STEPS_SETTLE):
    import examples.tire_contact.run as run
    G_old, damp_old = run.G, run.DAMP
    run.DAMP = DAMP_SETTLE
    if G is not None:
        run.G = G
    try:
        return solve_tire(n_steps=n_steps, n_phi=N_PHI_S, n_theta=N_THETA_S, n_rho=N_RHO_S,
                          verbose=False)
    finally:
        run.G, run.DAMP = G_old, damp_old


def fd_dC_dG(G, dG_rel=0.05):
    dG = dG_rel * G
    rp, rm = _solve_settled(G + dG), _solve_settled(G - dG)
    return (_compliance(rp["U"], rp["fext"]) - _compliance(rm["U"], rm["fext"])) / (2.0 * dG)


def main():
    """Return whether the run exhibits the intended dynamics/static mismatch."""
    import examples.tire_contact.run as run
    G = run.G
    base = _solve_settled()
    dadj, r_static = adjoint_dC_dG(base, G)
    dfd = fd_dC_dG(G)
    rel = abs(dadj - dfd) / max(abs(dfd), 1e-30)
    print(f"  |R_static(U*)| = {r_static:.3e}   (a true static equilibrium would be ~0)")
    print(f"  dC/dG  static-adjoint = {dadj:+.5f}   re-solve FD = {dfd:+.5f}   rel.diff = {rel:.1%}")
    gap_exhibited = r_static > 1e-2 and rel > 0.10
    print("DIAGNOSTIC: the sampled dynamic state does not satisfy the static equilibrium "
          f"criterion (|R_static|={r_static:.2f}), so this static-adjoint result should not "
          "be interpreted as a derivative of the dynamic solve. See examples/REFERENCES.md.")
    return gap_exhibited


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
