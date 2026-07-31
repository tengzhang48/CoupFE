"""Differentiable dual-multiplier friction — the RELAY: friction-field identifiability. ``python run.py``.

The semismooth forward solve changes active sets. This scoped study freezes the
converged active set, leaving a linearized system whose adjoint gives derivatives
with respect to a per-node friction field ``μ_i`` within that fixed set.

The frozen-active-set Jacobian on the condensed multipliers ``p`` (interface compliance ``G = S Kff⁻¹ Sᵀ``,
normal-coupling ``H = ∂N/∂p``):
    stick rows: ``r·G``                    (the constraint ``v_t = 0``)
    slip  rows: ``I − diag(s μ)·H``        (``p = s μ N(p)`` on the cone)
and crucially ``∂C/∂μ`` is **nonzero ONLY at slip nodes** — so ``dJ/dμ_i = λ_i s_i N_i`` is zero wherever a
node stuck. Thus the local sensitivity in this example is supported only on slip
nodes. This is not a general inverse-problem identifiability proof.

Self-check (prints ``OK`` / ``FAIL``):
  * the relay gradient ``dp/dμ`` matches finite differences to ~machine precision (where the active set is
    stable under the perturbation);
  * the slip support holds: ``dJ/dμ`` is exactly zero at every stick node, nonzero at slip nodes;
  * one Gauss-Newton step using the relay gradient reduces a μ-recovery loss.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from coupfe.materials import NeoHookean, _build
from coupfe.mesh import KernelMeshView
from coupfe.model import _structured_quad_mesh
from coupfe.operators.element_group import ElementGroup
from coupfe.runtime.compiled_element import CompiledElement

NX, NY, P, DELTA = 8, 4, 1.0, 0.08


def build():
    nodes, elems = _structured_quad_mesh(NX, NY, 1.5, 1.0)
    view = KernelMeshView(nodes, elems, dof_per_node=2)
    mat = NeoHookean(1.0, 10.0)
    elem = CompiledElement(_build(mat._for, mat._module), props=mat.props,
                           dof_per_node=2, n_svars=0, mcrd=2, n_elem=len(elems))
    grp = ElementGroup.from_view(view, elem, comps=(0, 1))
    t = grp.tangent(np.zeros(view.ndof), None, 0.0, 0.0)
    K = sp.coo_matrix((t.values, (t.rows, t.cols)), shape=(view.ndof, view.ndof)).tocsr()
    return nodes, view.ndof, K


class Relay:
    """Forward dual-multiplier friction with a per-node μ field + the frozen-active-set adjoint."""

    def __init__(self, nodes, ndof, K):
        bot = np.nonzero(np.abs(nodes[:, 1] - nodes[:, 1].min()) < 1e-9)[0]
        bot = bot[np.argsort(nodes[bot, 0])]
        top = np.nonzero(np.abs(nodes[:, 1] - nodes[:, 1].max()) < 1e-9)[0]
        bx, by, tx, ty = bot * 2, bot * 2 + 1, top * 2, top * 2 + 1
        self.nc = len(bot)
        fixed = np.concatenate([tx, by]); free = np.setdiff1d(np.arange(ndof), fixed)
        pos = {d: i for i, d in enumerate(free)}; ci = np.array([pos[d] for d in bx])
        S = sp.csr_matrix((np.ones(self.nc), (np.arange(self.nc), ci)), shape=(self.nc, len(free)))
        Kff = K[np.ix_(free, free)].tocsc(); Kfc = K[np.ix_(free, fixed)]; lu = spla.splu(Kff)
        KinvST = lu.solve(S.T.toarray()); self.G = S @ KinvST
        Knf = np.asarray(K[np.ix_(by, free)].todense()); Knc = np.asarray(K[np.ix_(by, fixed)].todense())
        self.H = Knf @ KinvST                                          # ∂N/∂p (normal–tangential coupling)
        self.r = float(Kff.diagonal().mean())
        uc = np.concatenate([np.full(len(tx), DELTA), np.zeros(len(by))])
        f = np.zeros(ndof); f[ty] = -P / len(top)
        ufree0 = lu.solve(f[free] - Kfc @ uc)
        self.vt0 = S @ ufree0; self.N0 = Knf @ ufree0 + Knc @ uc - f[by]
        self.eye = np.eye(self.nc)

    def _state(self, p, mu):
        N = np.maximum(self.N0 + self.H @ p, 0.0); vt = self.vt0 + self.G @ p
        y = p - self.r * vt; cap = mu * N
        stick = np.abs(y) <= cap; sgn = np.sign(y)
        C = p - np.where(stick, y, sgn * cap)
        return C, stick, N, sgn, vt

    def forward(self, mu):                                            # damped semismooth Newton → converged p
        p = np.zeros(self.nc)
        for _ in range(120):
            C, stick, N, sgn, vt = self._state(p, mu); res = float(np.sqrt(C @ C))
            if res < 1e-13:
                break
            Jp = np.where(stick[:, None], self.r * self.G, self.eye)
            dp = -np.linalg.solve(Jp, C); a = 1.0
            for _ in range(40):
                if np.sqrt(self._state(p + a * dp, mu)[0] @ self._state(p + a * dp, mu)[0]) < res:
                    break
                a *= 0.5
            p = p + a * dp
        return p, stick, N, sgn, vt

    def _Jp_exact(self, stick, N, sgn, mu):                          # exact frozen-active-set Jacobian
        Jp = np.where(stick[:, None], self.r * self.G, self.eye); slip = ~stick
        Jp[slip] = self.eye[slip] - (sgn * mu * (N > 0))[slip, None] * self.H[slip]
        return Jp

    def dp_dmu(self, mu):
        """The relay: exact dp/dμ via the adjoint Jacobian; columns are zero at stick nodes (slip support)."""
        p, stick, N, sgn, vt = self.forward(mu)
        Jp = self._Jp_exact(stick, N, sgn, mu)
        rhs = np.diag(sgn * N * (~stick))                            # ∂C/∂μ — ONLY at slip nodes
        return np.linalg.solve(Jp, rhs), p, stick, N, sgn, vt        # (nc×nc) dp/dμ

    def grad(self, mu, dJ_dvt):
        """dJ/dμ for a QOI whose sensitivity to the slip is dJ_dvt (J depends on μ only through v_t)."""
        dpdmu, p, stick, N, sgn, vt = self.dp_dmu(mu)
        return (dJ_dvt @ self.G) @ dpdmu, stick, vt                  # dJ/dμ = (dJ/dp)·dp/dμ, dJ/dp = G dJ_dvt


def main():
    nodes, ndof, K = build()
    relay = Relay(nodes, ndof, K)
    nc = relay.nc
    mu0 = np.full(nc, 0.4)

    # (1) gradient check: dp/dμ vs FD where the active set is stable
    dpdmu, p, stick, N, sgn, vt = relay.dp_dmu(mu0)
    h = 1e-7; fd = np.zeros((nc, nc)); stable = np.ones(nc, bool)
    for i in range(nc):
        mp = mu0.copy(); mp[i] += h; mm = mu0.copy(); mm[i] -= h
        pp, sp_, *_ = relay.forward(mp); pm, sm, *_ = relay.forward(mm)
        fd[:, i] = (pp - pm) / (2 * h)
        stable[i] = np.array_equal(sp_, stick) and np.array_equal(sm, stick)
    grad_err = np.max(np.abs(dpdmu[:, stable] - fd[:, stable])) / (np.max(np.abs(fd[:, stable])) + 1e-30)

    # (2) slip support: dJ/dμ == 0 at stick, != 0 at slip, for J = sum of interface slip
    g, _, _ = relay.grad(mu0, np.ones(nc))
    slip = ~stick
    support_ok = np.allclose(g[stick], 0.0) and np.all(np.abs(g[slip]) > 1e-12)

    # (3) one Gauss-Newton step recovers slip-node μ from observed slip (the inverse)
    mu_true = mu0.copy(); mu_true[slip] *= 1.3                       # perturb μ where it will slip
    vt_obs = relay.forward(mu_true)[4]
    loss0 = 0.5 * float(np.sum((vt - vt_obs) ** 2))
    gloss, _, vt_now = relay.grad(mu0, (vt - vt_obs))               # dJ/dμ of 0.5||v_t - v_t_obs||²
    mu1 = mu0 - 0.5 * gloss
    loss1 = 0.5 * float(np.sum((relay.forward(mu1)[4] - vt_obs) ** 2))

    ok = grad_err < 1e-6 and support_ok and loss1 < loss0
    print(f"nc={nc}  stick={int(stick.sum())} slip={int(slip.sum())}")
    print(f"(1) relay dp/dμ vs FD (stable {int(stable.sum())}/{nc}): max rel err = {grad_err:.2e}")
    print(f"(2) slip support: dJ/dμ==0 at stick & !=0 at slip -> {support_ok}  (stick nodes unidentifiable)")
    print(f"(3) μ-recovery loss: {loss0:.3e} -> {loss1:.3e} after one relay Gauss-Newton step  -> {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
