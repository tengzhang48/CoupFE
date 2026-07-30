"""Phase 3 — DIAGNOSTIC: why a naive static adjoint fails to differentiate the barrier-DYNAMICS tire.

We tried the cheap route to `d(compliance)/dG`: a **static adjoint at the resting state** — assemble the
bulk+contact tangent `K=∂R_static/∂U` (no inertia), solve `Kᵀλ=f_ext`, `dC/dG=−λᵀ∂R/∂G` — and validate
against a full re-solve FD. QOI = compliance `C=f_ext·U` (gravity work; cleanly G-sensitive, `∂C/∂U=f_ext`).

**Finding (the point of this script): it does NOT validate, and `|R_static(U*)| ≈ 0.2` is why.** The
forward needs *dynamics* (the barrier tangent is indefinite → static Newton stalls); but then the
**resting state is not a static equilibrium** — even after hard settling (`DAMP=2`, 110 steps),
`R_static ≉ 0`. The static adjoint *assumes* `R_static=0`, so its gradient (−0.52) is far from the true
FD gradient (−0.05).

**The correct paths (the lesson):** differentiating a barrier-*dynamics* solve needs the **dynamic
adjoint** (through the time-stepping, carrying the inertia/damping), OR a contact formulation that
converges to a **clean static KKT** — the **dual-multiplier**, whose frozen-active-set adjoint *is*
validated (`examples/friction_identifiability`, grad-vs-FD ~1e-10). So differentiability and the
smoothed-barrier's forward robustness are in tension: the robust forward path has no cheap static
adjoint. This is a real, reusable result (`docs/dev/tire_buildlog.md`), not a tuning failure.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from coupfe.assembly.assemble import assemble_residual, assemble_tangent, _apply_dirichlet
from coupfe.mesh import KernelMeshView
from coupfe.operators.element_group import ElementGroup
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

from examples.tire_contact.run import solve_tire, K_BULK, _UP_HEX8_FOR

# Coarse mesh + hard settling: the adjoint-vs-FD check is a METHODOLOGY proof (mesh-independent), so a
# small fast mesh keeps the 3 re-solves tractable while DAMP/STEPS drive R_static(U*) -> 0.
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
    """Demonstrate the dynamics-vs-static gap: the static adjoint is INVALID here, and |R_static| shows
    why. The diagnostic 'passes' by correctly exhibiting the gap (R_static ≉ 0 ⇒ adjoint ≠ FD)."""
    import examples.tire_contact.run as run
    G = run.G
    base = _solve_settled()
    dadj, r_static = adjoint_dC_dG(base, G)
    dfd = fd_dC_dG(G)
    rel = abs(dadj - dfd) / max(abs(dfd), 1e-30)
    print(f"  |R_static(U*)| = {r_static:.3e}   (a true static equilibrium would be ~0)")
    print(f"  dC/dG  static-adjoint = {dadj:+.5f}   re-solve FD = {dfd:+.5f}   rel.diff = {rel:.1%}")
    gap_exhibited = r_static > 1e-2 and rel > 0.10
    print("FINDING: the dynamics resting state is NOT a static equilibrium "
          f"(|R_static|={r_static:.2f}); the naive static adjoint is INVALID for this barrier-DYNAMICS "
          "solve. Use the DYNAMIC adjoint, or the dual-multiplier (clean static KKT, differentiable — "
          "examples/friction_identifiability). See docs/dev/tire_buildlog.md.")
    return gap_exhibited


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
