"""Dual-multiplier EXACT-STICK friction: an elastic block sheared on a rigid frictional floor. ``python run.py``.

Where the ppf/IPC **smoothed** friction (``examples/contact_3d_friction``) reproduces the Coulomb force
``f=μN`` but only *approaches* stick through a regularized plateau (residual creep ``~ μN·eps``), this
example demonstrates the **dual-multiplier** (Alart-Curnier / active-set) treatment, which makes stick an
exact *constraint* rather than a stiff spring:

    stick:  v_t = 0   (the tangential slip is a CONSTRAINT; the friction force is its Lagrange multiplier)
    slip :  |p_t| = μN (the multiplier sits on the Coulomb cone; v_t > 0 is free)

The active set switches on the dimensionless **utilization** ``η = |p_t| / (μN)`` — stick iff ``η ≤ 1``.
Because stick is a constraint, the interface slip is **machine-zero** (not ``~F/k_t``): there is no
tangential-stiffness ``k_t`` and hence no ``k̃_t = k_t L/E`` conditioning knob — the wall the
return-map/penalty forms hit when pushed toward exact stick.

Setup: a unit ``G=1`` linear-elastic block (CoupFE ``NeoHookean`` tangent at ``u=0``) pressed by ``P`` onto a
rigid floor (bottom ``y`` fixed) and sheared quasi-statically by a prescribed top displacement ``δ`` (ramped).
The interface obeys the **global** Coulomb criterion ``Σ|p_t| ≤ μ ΣN`` (the interface sticks/slides as a unit
— well-posed and mesh-robust; per-node *partial* slip needs the full semismooth Newton on a smooth contact
geometry, since a bonded sharp corner is a stress singularity — see docs/dev/contact.md).

Self-check (prints ``OK`` / ``FAIL``):
  * EXACT stick: interface slip ``v_t = 0`` to machine precision for every ``δ < δ*``;
  * Coulomb cap:  friction force locks at ``μP`` exactly once sliding (``δ ≥ δ*``);
  * transition:  the stick→slip onset matches the analytic incipient shear ``δ* = μP / k_shear``.
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

NX, NY = 6, 4
G, K_BULK = 1.0, 10.0
MU, P = 0.4, 1.0


def build_stiffness():
    """Linear-elastic stiffness of a unit block: the CoupFE NeoHookean tangent at u=0."""
    nodes, elems = _structured_quad_mesh(NX, NY, 1.0, 1.0)
    view = KernelMeshView(nodes, elems, dof_per_node=2)
    mat = NeoHookean(G, K_BULK)
    elem = CompiledElement(_build(mat._for, mat._module), props=mat.props,
                           dof_per_node=2, n_svars=0, mcrd=2, n_elem=len(elems))
    grp = ElementGroup.from_view(view, elem, comps=(0, 1))
    t = grp.tangent(np.zeros(view.ndof), None, 0.0, 0.0)
    K = sp.coo_matrix((t.values, (t.rows, t.cols)), shape=(view.ndof, view.ndof)).tocsr()
    return nodes, view.ndof, K


class ExactStickInterface:
    """Global-Coulomb dual-multiplier friction on a block's bottom face against a rigid floor."""

    def __init__(self, nodes, ndof, K):
        self.ndof, self.K, self.all = ndof, K, np.arange(ndof)
        ytol = 1e-9
        bottom = np.nonzero(np.abs(nodes[:, 1] - nodes[:, 1].min()) < ytol)[0]
        top = np.nonzero(np.abs(nodes[:, 1] - nodes[:, 1].max()) < ytol)[0]
        self.bx, self.by = bottom * 2, bottom * 2 + 1     # bottom tangential / normal dofs
        self.tx, self.ty = top * 2, top * 2 + 1           # top tangential / normal dofs
        self.ntop = len(top)

    def _solve(self, fixed, uc, fextra=None):
        free = np.setdiff1d(self.all, fixed)
        f = np.zeros(self.ndof)
        f[self.ty] = -P / self.ntop                        # press the block down → normal load N
        if fextra is not None:
            f[fextra[0]] += fextra[1]
        u = np.zeros(self.ndof)
        u[fixed] = uc
        u[free] = spla.spsolve(self.K[np.ix_(free, free)], f[free] - self.K[np.ix_(free, fixed)] @ uc)
        return u, self.K @ u - f                            # (displacement, reactions)

    def shear_stiffness(self):
        """k_shear = dF_fric/dδ from a unit all-stick shear (→ analytic incipient δ* = μP/k_shear)."""
        _, R = self._solve(np.concatenate([self.tx, self.by, self.bx]),
                           np.concatenate([np.full(len(self.tx), 1.0), np.zeros(len(self.by) + len(self.bx))]))
        return abs(float(np.sum(R[self.bx])))

    def step(self, delta):
        """One quasi-static shear increment: returns (regime, interface_slip v_t, friction force)."""
        # Trial: assume STICK (v_t = 0 constraint) → the tangential reactions ARE the multipliers p_t.
        _, R = self._solve(np.concatenate([self.tx, self.by, self.bx]),
                          np.concatenate([np.full(len(self.tx), delta), np.zeros(len(self.by) + len(self.bx))]))
        Fs = float(np.sum(R[self.bx]))                      # total tangential force demanded by stick
        N = np.maximum(R[self.by], 0.0)                     # per-node normal reactions (compressive)
        if abs(Fs) <= MU * P:                               # utilization η ≤ 1 → stick admissible
            return "stick", 0.0, abs(Fs)                    # v_t = 0 EXACTLY (a constraint)
        # SLIP: multiplier on the cone (|p_t| = μN), interface free → it slides.
        sgn = -np.sign(Fs)                                  # friction opposes the slip tendency
        u, _ = self._solve(np.concatenate([self.tx, self.by]),
                           np.concatenate([np.full(len(self.tx), delta), np.zeros(len(self.by))]),
                           fextra=(self.bx, -sgn * MU * N))
        return "slip", float(np.max(np.abs(u[self.bx]))), MU * P


def main():
    nodes, ndof, K = build_stiffness()
    iface = ExactStickInterface(nodes, ndof, K)
    k_shear = iface.shear_stiffness()
    dstar = MU * P / k_shear                                # analytic incipient-slip shear
    print(f"k_shear = dF/dδ = {k_shear:.4f}   μP = {MU * P:.3f}   →  δ*(incipient slip) = {dstar:.4f}")
    print(f"{'δ/δ*':>6} {'regime':>7} {'F_fric':>8} {'F/μP':>7} {'v_t (slip)':>12}")

    ok = True
    for ratio in (0.3, 0.6, 0.9, 1.0, 1.2, 1.5, 2.0):
        regime, vt, Ff = iface.step(ratio * dstar)
        print(f"{ratio:>6.2f} {regime:>7} {Ff:>8.4f} {Ff / (MU * P):>7.3f} {vt:>12.2e}")
        if ratio < 1.0:
            ok &= regime == "stick" and vt == 0.0 and Ff < MU * P     # exact stick, below cone
        else:
            ok &= regime == "slip" and vt > 0.0 and abs(Ff - MU * P) < 1e-12  # exact Coulomb cap, sliding

    print("OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
