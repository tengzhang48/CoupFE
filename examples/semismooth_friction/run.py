"""Per-node PARTIAL-SLIP friction via the dual-multiplier semismooth Newton. ``python run.py``.

The companion to ``examples/exact_stick_friction`` (which uses the *global* Coulomb criterion — the
interface sticks/slides as a unit). Here the friction force at *each* contact node is its own Lagrange
multiplier, so a **stick zone and a slip zone coexist**: some nodes stick with exact ``v_t=0`` while
others sit on the Coulomb cone ``|p|=μN`` and slide. The Alart-Curnier semismooth Newton resolves the
active set consistently — it does NOT cascade at the bonded stress concentration the way a naive
switch-and-resolve does.

An elastic block (CoupFE ``NeoHookean`` tangent at ``u=0``) is pressed by ``P`` onto a rigid frictional
floor (bottom-``y`` fixed → ``N`` = floor reaction) and sheared by a prescribed top displacement ``δ``.
Two regimes:
  * small ``δ`` — **press-dominated**: the Poisson bulge slips the *edges* outward (symmetric), centre sticks;
  * large ``δ`` — **shear-dominated**: the stick zone shifts toward the leading side (Cattaneo-like).

Self-check (prints ``OK`` / ``FAIL``):
  * a genuine partial-slip state (≥1 stick AND ≥1 slip node) at both regimes;
  * every stick node has ``v_t = 0`` to machine precision (exact stick — a constraint, not creep);
  * every slip node sits exactly on the cone (``|p|/μN = 1``);
  * the **Schur-condensed** interface solve returns the **same** ``(U, p)`` as the direct augmented solve.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from coupfe.materials import NeoHookean, _build
from coupfe.mesh import KernelMeshView
from coupfe.model import _structured_quad_mesh
from coupfe.operators.contact_semismooth import solve_friction_semismooth
from coupfe.operators.element_group import ElementGroup
from coupfe.runtime.compiled_element import CompiledElement

NX, NY = 6, 4
G, K_BULK, MU, P = 1.0, 10.0, 0.4, 1.0


def build():
    nodes, elems = _structured_quad_mesh(NX, NY, 1.0, 1.0)
    view = KernelMeshView(nodes, elems, dof_per_node=2)
    mat = NeoHookean(G, K_BULK)
    elem = CompiledElement(_build(mat._for, mat._module), props=mat.props,
                           dof_per_node=2, n_svars=0, mcrd=2, n_elem=len(elems))
    grp = ElementGroup.from_view(view, elem, comps=(0, 1))
    t = grp.tangent(np.zeros(view.ndof), None, 0.0, 0.0)
    K = sp.coo_matrix((t.values, (t.rows, t.cols)), shape=(view.ndof, view.ndof)).tocsr()
    return nodes, view.ndof, K


def solve(nodes, ndof, K, delta, schur=False):
    ytol = 1e-9
    bot = np.nonzero(np.abs(nodes[:, 1] - nodes[:, 1].min()) < ytol)[0]
    top = np.nonzero(np.abs(nodes[:, 1] - nodes[:, 1].max()) < ytol)[0]
    bx, by, tx, ty = bot * 2, bot * 2 + 1, top * 2, top * 2 + 1
    f = np.zeros(ndof)
    f[ty] = -P / len(top)                                     # press the block down → normal reaction N
    fixed = np.concatenate([tx, by])                          # prescribe top shear δ; floor (bottom y)
    vals = np.concatenate([np.full(len(tx), delta), np.zeros(len(by))])
    return solve_friction_semismooth(K, f, fixed_dofs=fixed, fixed_vals=vals,
                                     contact_tan_dofs=bx, normal_dofs=by, mu=MU, schur=schur)


def main():
    nodes, ndof, K = build()
    bot = np.nonzero(np.abs(nodes[:, 1] - nodes[:, 1].min()) < 1e-9)[0]
    bot = bot[np.argsort(nodes[bot, 0])]
    bx = bot * 2
    ok = True
    print(f"{'δ':>7} {'regime':>15} {'iters':>6} {'res':>9} {'#stick':>7} {'#slip':>6} "
          f"{'max v_t|stick':>13} {'Schur==direct':>13}")
    for delta, label in ((0.02, "press-dominated"), (0.8, "shear-dominated")):
        res = solve(nodes, ndof, K, delta, schur=False)
        res_s = solve(nodes, ndof, K, delta, schur=True)
        vt_stick = np.max(np.abs(res.U[bx][res.stick])) if res.stick.any() else 0.0
        # slip nodes sit exactly on the cone: |p| = μN (absolute — handles unloaded nodes where N→0)
        slip_p, slip_cap = np.abs(res.p[~res.stick]), MU * res.N[~res.stick]
        schur_match = np.allclose(res.U, res_s.U, atol=1e-7) and np.allclose(res.p, res_s.p, atol=1e-7)
        partial = res.stick.any() and (~res.stick).any()
        cone_ok = slip_p.size == 0 or np.allclose(slip_p, slip_cap, atol=1e-6)
        print(f"{delta:>7.3f} {label:>15} {res.iters:>6d} {res.residual:>9.1e} "
              f"{int(res.stick.sum()):>7d} {int((~res.stick).sum()):>6d} {vt_stick:>13.2e} "
              f"{str(schur_match):>13}")
        ok &= res.converged and partial and vt_stick < 1e-9 and cone_ok and schur_match
    print("OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
