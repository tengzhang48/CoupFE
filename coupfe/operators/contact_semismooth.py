"""Semismooth-Newton DUAL-MULTIPLIER Coulomb friction (Alart-Curnier), with a Schur-condensed interface.

This is the *exact-stick* friction solver: the tangential friction force at each contact node is a
**Lagrange multiplier** ``p`` and stick is the **constraint** ``v_t = 0`` (not a stiff spring), so stick
is exactly zero-slip — no ``k_t`` and hence no ``k̃_t = k_t L/E`` conditioning knob (the wall the
return-map/penalty forms hit). Unlike the *global* Coulomb demo in ``examples/exact_stick_friction``, this
resolves **per-node partial slip** (a stick zone and a slip zone coexisting), which a naive
switch-and-resolve cannot do at a bonded stress-concentration without cascading.

Formulation (friction multiplier; the NORMAL force ``N`` is supplied/lagged — e.g. from the barrier or the
floor reaction). Unknowns ``(u, p)``:

    equilibrium :  K u = f_ext + Sᵀ p          (S u = v_t, the per-node tangential slip; p its conjugate force)
    Alart-Curnier:  C_i = p_i − proj_[−μN_i, μN_i]( p_i − r v_t,i ) = 0

The generalized (semismooth) Jacobian per node — ``y = p − r v_t``:

    stick  (|y| ≤ μN):  ∂C/∂u =  r S,   ∂C/∂p = 0     →  the constraint  v_t = 0,  p free (the reaction)
    slip   (|y| >  μN):  ∂C/∂u =  0,     ∂C/∂p = 1     →  p = ±μN (on the cone),  v_t free

so the Newton step solves the saddle-point ``[[K, −Sᵀ], [A_cu, A_cp]] (du, dp) = −(R_u, C)``. The
augmentation ``r`` affects the convergence rate only, not the converged answer.

``schur=True`` condenses the non-contact (interior) DOFs out of ``K`` first, leaving a small interface
system in ``(contact tangential DOFs, multipliers)`` — the Schur-complement form for scale; it returns the
same solution as the direct augmented solve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class FrictionResult:
    U: np.ndarray            # full displacement vector
    p: np.ndarray            # per-contact-node tangential friction multiplier (signed force)
    stick: np.ndarray        # bool mask, True where the node sticks (v_t = 0)
    N: np.ndarray            # per-node normal force used (lagged reaction)
    iters: int
    residual: float
    converged: bool


def solve_friction_semismooth(K, f_ext, *, fixed_dofs, fixed_vals, contact_tan_dofs,
                              normal_dofs, mu, r=None, schur=False, maxit=80, tol=1e-10):
    """Solve a linear-elastic body in frictional contact via the dual-multiplier semismooth Newton.

    K, f_ext           : assembled bulk stiffness (csr) and external force.
    fixed_dofs/vals    : Dirichlet DOFs and their prescribed values.
    contact_tan_dofs   : the DOFs whose displacement is the per-node tangential slip v_t (friction acts here).
    normal_dofs        : the DOFs whose reaction gives the per-node normal force N (e.g. the floor-y DOFs);
                         N is lagged (recomputed each iterate, frozen inside the Jacobian).
    mu, r              : friction coefficient; AC augmentation (default ~ mean free-block stiffness).
    schur              : condense the interior DOFs to a small interface system (same answer).
    """
    ndof = K.shape[0]
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    ctan = np.asarray(contact_tan_dofs, dtype=int)
    ndofn = np.asarray(normal_dofs, dtype=int)
    nc = len(ctan)
    free = np.setdiff1d(np.arange(ndof), fixed_dofs)
    pos = {d: i for i, d in enumerate(free)}
    # S (nc, n_free): picks the contact tangential DOFs out of the free vector
    S = sp.csr_matrix((np.ones(nc), (np.arange(nc), [pos[d] for d in ctan])), shape=(nc, len(free)))
    Kff = K[np.ix_(free, free)].tocsc()
    Kfc = K[np.ix_(free, fixed_dofs)]
    if r is None:
        r = float(Kff.diagonal().mean())

    U = np.zeros(ndof)
    U[fixed_dofs] = fixed_vals
    p = np.zeros(nc)

    # static-condensation partition (Schur): interface = contact tangential DOFs, interior = the rest
    if schur:
        Iidx = np.array([pos[d] for d in ctan])            # interface positions in `free`
        Oidx = np.setdiff1d(np.arange(len(free)), Iidx)    # interior positions
        Koo = Kff[np.ix_(Oidx, Oidx)].tocsc()
        Koi = Kff[np.ix_(Oidx, Iidx)]
        Kio = Kff[np.ix_(Iidx, Oidx)]
        Kii = Kff[np.ix_(Iidx, Iidx)].toarray()
        lu_oo = spla.splu(Koo)
        Kcond = Kii - (Kio @ lu_oo.solve(Koi.toarray()))   # Schur complement on the interface

    for it in range(maxit):
        R = K @ U - f_ext
        N = np.maximum(R[ndofn], 0.0)                       # lagged compressive normal reaction
        vt = U[ctan]
        y = p - r * vt
        cap = mu * N
        stick = np.abs(y) <= cap
        proj = np.where(stick, y, np.sign(y) * cap)
        C = p - proj
        Ru = R[free] - S.T @ p
        res = float(np.sqrt(Ru @ Ru + C @ C))
        if res < tol:
            return FrictionResult(U.copy(), p.copy(), stick, N, it, res, True)

        d_stick = np.where(stick, r, 0.0)
        Acp = sp.diags(np.where(stick, 0.0, 1.0))
        if not schur:
            Acu = sp.diags(d_stick) @ S
            J = sp.bmat([[Kff, -S.T], [Acu, Acp]], format="csc")
            delta = spla.spsolve(J, -np.concatenate([Ru, C]))
            U[free] += delta[:len(free)]
            p += delta[len(free):]
        else:
            # condense the interior residual onto the interface, solve (du_I, dp), recover du_O
            Ru_I, Ru_O = Ru[Iidx], Ru[Oidx]
            rhs_I = -(Ru_I - Kio @ lu_oo.solve(Ru_O))
            Acp_d = np.where(stick, 0.0, 1.0)
            # interface system: [[Kcond, -I],[diag(d_stick), diag(Acp)]] (du_I, dp) = (rhs_I, -C)
            top = np.hstack([Kcond, -np.eye(nc)])
            bot = np.hstack([np.diag(d_stick), np.diag(Acp_d)])
            duI_dp = np.linalg.solve(np.vstack([top, bot]), np.concatenate([rhs_I, -C]))
            duI, dp = duI_dp[:nc], duI_dp[nc:]
            duO = lu_oo.solve(-Ru_O - Koi @ duI)
            U[free[Iidx]] += duI
            U[free[Oidx]] += duO
            p += dp

    return FrictionResult(U.copy(), p.copy(), stick, N, maxit, res, False)


class SemismoothFrictionSolver:
    """Efficient, reusable dual-multiplier friction solver for SMALL-SLIDING / SMALL-STRAIN contact.

    The bulk stiffness ``K`` is constant (small strain), so factor it ONCE and pre-build the interface
    compliance ``G = S Kff⁻¹ Sᵀ`` (``nc×nc``, ``nc`` = contact tangential dofs) + the recovery map
    ``Kff⁻¹ Sᵀ``. Then the contact is fully **condensed to the interface**: with ``u_free = ufree0 +
    (Kff⁻¹Sᵀ) p`` the equilibrium is satisfied *by construction*, the slip is ``v_t = v_t0 + G p``, and
    the semismooth Newton runs on the multipliers ``p`` ALONE — every inner iteration is solve-free
    (``O(nc²)`` dense), no bulk back-substitution. The expensive factorization is paid once in
    ``__init__`` and **amortized across all load steps** (warm-start ``p`` along the friction path), which
    is the realistic use (path-dependent friction). Same formulation as :func:`solve_friction_semismooth`
    (lagged normal, semismooth AC Jacobian: stick rows ``r·G``, slip rows identity); a damped step
    (backtracking on ``‖C‖``) adds robustness. This maps directly onto a PETSc FieldSplit/Schur solve for
    the distributed case — the factorization here is the serial stand-in for the amortized bulk solve.
    """

    def __init__(self, K, *, fixed_dofs, contact_tan_dofs, normal_dofs, mu, r=None):
        self.ndof = K.shape[0]
        self.fixed = np.asarray(fixed_dofs, dtype=int)
        self.ctan = np.asarray(contact_tan_dofs, dtype=int)
        self.ndofn = np.asarray(normal_dofs, dtype=int)
        self.mu = float(mu)
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)
        pos = {d: i for i, d in enumerate(self.free)}
        ci = np.array([pos[d] for d in self.ctan])
        self.nc = len(ci)
        Kff = K[np.ix_(self.free, self.free)].tocsc()
        self.r = float(Kff.diagonal().mean()) if r is None else float(r)
        self.lu = spla.splu(Kff)                                       # bulk factorization — ONCE
        self.Kfc = K[np.ix_(self.free, self.fixed)]
        self.S = sp.csr_matrix((np.ones(self.nc), (np.arange(self.nc), ci)),
                               shape=(self.nc, len(self.free)))
        self.Kinv_ST = self.lu.solve(self.S.T.toarray())              # nfree×nc recovery map (nc backsolves)
        self.G = self.S @ self.Kinv_ST                                # nc×nc interface compliance
        self.Knf = K[np.ix_(self.ndofn, self.free)]                   # normal-reaction blocks (for N)
        self.Knc = K[np.ix_(self.ndofn, self.fixed)]
        self._eye = np.eye(self.nc)

    def solve(self, f_ext, fixed_vals, *, p0=None, maxit=80, tol=1e-10, damping=True):
        """One load step. Reuses the factorization; warm-start ``p0`` from the previous step."""
        Ufix = np.asarray(fixed_vals, dtype=float)
        f_ext = np.asarray(f_ext, dtype=float)
        ufree0 = self.lu.solve(f_ext[self.free] - self.Kfc @ Ufix)    # the only per-step bulk solve
        vt0 = self.S @ ufree0
        fn = f_ext[self.ndofn]
        p = np.zeros(self.nc) if p0 is None else np.array(p0, dtype=float)

        def aux(p):                                                   # solve-free: dense matvecs only
            ufree = ufree0 + self.Kinv_ST @ p
            N = np.maximum(self.Knf @ ufree + self.Knc @ Ufix - fn, 0.0)
            vt = vt0 + self.G @ p
            y = p - self.r * vt
            cap = self.mu * N
            stick = np.abs(y) <= cap
            C = p - np.where(stick, y, np.sign(y) * cap)
            return C, stick, N

        res, it, converged = np.inf, 0, False
        for it in range(maxit):
            C, stick, N = aux(p)
            res = float(np.sqrt(C @ C))
            if res < tol:
                converged = True
                break
            Jp = np.where(stick[:, None], self.r * self.G, self._eye)  # stick: r·G ; slip: identity
            dp = np.linalg.solve(Jp, -C)
            if damping:                                               # backtrack on ‖C‖ (robustness)
                a = 1.0
                for _ in range(25):
                    Cn, _, _ = aux(p + a * dp)
                    if np.sqrt(Cn @ Cn) < res:
                        break
                    a *= 0.5
                p = p + a * dp
            else:
                p = p + dp

        U = np.zeros(self.ndof)
        U[self.fixed] = Ufix
        U[self.free] = ufree0 + self.Kinv_ST @ p
        _, stick, N = aux(p)
        return FrictionResult(U, p, stick, N, it, res, converged)
