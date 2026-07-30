"""Rigid-obstacle contact as a CoupFE operator (Stage 1) — vectorized.

Contact is a **separate** `(residual, tangent, commit)` operator, not part of any
element — so the bulk stays reference/total-Lagrangian while contact lives in the
**current** configuration (gap/normal are spatial). A rigid analytical obstacle exposes
a signed `gap(x)` (>0 separated, <0 penetration) and an outward `normal(x)` (out of the
obstacle, into the body). Penalty contact adds the energy `Π_c = ½ k g²` for `g < 0`; its
residual is the gradient `∂Π_c/∂u = k g n` (signs unambiguous because they come from the
energy). The tangent is the **complex-step** of that gradient with the **active set frozen
from the real iterate** (the `g < 0` switch is the only non-smooth part), so the curved-
obstacle geometric term `g ∂n/∂u` is captured automatically.

**No Python node-loop:** gap/normal/force and the complex-step tangent are evaluated for
all contact nodes at once (the obstacle ops use `axis=-1`, so they take a single point or
an `(n, dim)` batch). The tangent costs `dim` vectorized complex evaluations, not one per
node. For regular contact that is fast; the irregular *search* (BVH / narrow-phase) is the
part that later moves to numba or a native manager (see `docs/dev/contact.md`).

The all-primitive barrier, closest-feature geometry, adaptive barrier scaling,
and smoothed-friction portions include modified NumPy adaptations of concepts
and routines from Apache-2.0-licensed ``ppf-contact-solver``. Upstream
provenance and licensing are recorded in the repository ``NOTICE`` file.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from coupfe.operators.base import Residual, Tangent
from coupfe.operators.contact_search import candidate_pairs


class HalfSpace:
    """Rigid half-space through ``point`` with outward unit ``normal`` (body on the
    +normal side). ``gap = (x - point)·n``; flat, so the tangent is ``k n⊗n``.

    ``half_width`` (2D only) turns the infinite half-space into a finite strip centered
    at ``point`` and extending ``±half_width`` along the in-plane tangent. Nodes whose
    projection falls outside the strip return ``+inf`` gap and are ignored. This matches
    Abaqus-style analytical rigid surfaces of finite extent.

    ``kinematic=True`` marks a prescribed-motion obstacle. The barrier evaluates the gap
    with a ppf-style floor ``max(gap, constraint_tol*dhat)`` so the moving wall cannot
    generate an unbounded repulsive force as it pushes into a node.

    ``thickness > 0`` enables pass-through: nodes that have penetrated deeper than
    ``thickness`` are ignored (the obstacle is treated as a thin shell).
    """

    def __init__(self, point, normal, *, half_width=None, kinematic=False,
                 thickness=None, constraint_tol=0.01):
        self.p = np.asarray(point, dtype=float)
        n = np.asarray(normal, dtype=float)
        self.n = n / np.linalg.norm(n)
        self.half_width = (None if half_width is None else float(half_width))
        self.kinematic = bool(kinematic)
        self.thickness = (None if thickness is None else float(thickness))
        self.constraint_tol = float(constraint_tol)
        if self.half_width is not None:
            # 2D tangent perpendicular to the normal (either sign works for projection).
            self._t = np.array([-self.n[1], self.n[0]], dtype=float)

    def _in_strip(self, x, p):
        if self.half_width is None:
            return None
        proj = np.tensordot(np.asarray(x) - p, self._t, axes=([-1], [0]))
        return np.abs(proj) <= self.half_width

    def gap(self, x, t=None):
        # t is accepted so time-dependent obstacles can share the interface; ignored here.
        d = np.tensordot(np.asarray(x) - self.p, self.n, axes=([-1], [0]))
        in_strip = self._in_strip(x, self.p)
        if in_strip is not None:
            d = np.where(in_strip, d, np.inf)
        return d

    def normal(self, x, t=None):
        return np.broadcast_to(self.n, np.shape(x))


class Sphere:
    """Rigid sphere of radius ``R`` about ``center``. ``inside=False``: body outside a
    solid sphere (``gap = |x-c| - R``). ``inside=True``: body inside a spherical cavity.

    ``kinematic=True`` and ``thickness`` follow the same convention as :class:`HalfSpace`.
    """

    def __init__(self, center, R, inside=False, *, kinematic=False, thickness=None,
                 constraint_tol=0.01):
        self.c = np.asarray(center, dtype=float)
        self.R = float(R)
        self.inside = bool(inside)
        self.kinematic = bool(kinematic)
        self.thickness = (None if thickness is None else float(thickness))
        self.constraint_tol = float(constraint_tol)

    def _dr(self, x):
        d = np.asarray(x) - self.c
        r = np.sqrt(np.sum(d * d, axis=-1, keepdims=True))   # complex-safe
        return d, r

    def gap(self, x, t=None):
        # t accepted for a uniform time-aware obstacle interface; ignored here.
        _, r = self._dr(x)
        r = r[..., 0]
        return (self.R - r) if self.inside else (r - self.R)

    def normal(self, x, t=None):
        d, r = self._dr(x)
        nrm = d / r
        return -nrm if self.inside else nrm


class MovingHalfSpace(HalfSpace):
    """Half-space whose reference point moves in time.

    ``position(t)`` returns a point on the plane at physical time ``t``; the normal is
    fixed. This lets the CCD step bound in :class:`RigidBarrierContact` sample the
    obstacle trajectory at intermediate times during a predictor jump, so a node cannot
    tunnel through a rapidly moving plate. For operators that do not pass ``t`` (e.g. the
    penalty :class:`RigidContact`), the plane falls back to the stored ``self.p``.

    ``half_width`` is forwarded to :class:`HalfSpace` to model a finite-width moving plate.
    """

    def __init__(self, position, normal, *, half_width=None, kinematic=False,
                 thickness=None, constraint_tol=0.01):
        self._position_fn = position
        p0 = np.asarray(position(0.0), dtype=float)
        n = np.asarray(normal, dtype=float)
        self.p = p0
        self.n = n / np.linalg.norm(n)
        self.half_width = (None if half_width is None else float(half_width))
        self.kinematic = bool(kinematic)
        self.thickness = (None if thickness is None else float(thickness))
        self.constraint_tol = float(constraint_tol)
        if self.half_width is not None:
            self._t = np.array([-self.n[1], self.n[0]], dtype=float)

    def position(self, t):
        return np.asarray(self._position_fn(t), dtype=float)

    def gap(self, x, t=None):
        p = self.p if t is None else self.position(t)
        d = np.tensordot(np.asarray(x) - p, self.n, axes=([-1], [0]))
        in_strip = self._in_strip(x, p)
        if in_strip is not None:
            d = np.where(in_strip, d, np.inf)
        return d

    def normal(self, x, t=None):
        return np.broadcast_to(self.n, np.shape(x))


class MovingSphere(Sphere):
    """Sphere whose center moves in time.

    ``center(t)`` returns the sphere center at physical time ``t``. The radius and
    ``inside`` flag are fixed. Like :class:`MovingHalfSpace`, this is consumed by the
    time-aware CCD in :class:`RigidBarrierContact`.
    """

    def __init__(self, center, R, inside=False, *, kinematic=False, thickness=None,
                 constraint_tol=0.01):
        self._center_fn = center
        c0 = np.asarray(center(0.0), dtype=float)
        super().__init__(c0, R, inside=inside, kinematic=kinematic,
                         thickness=thickness, constraint_tol=constraint_tol)

    def center_at(self, t):
        return np.asarray(self._center_fn(t), dtype=float)

    def _dr(self, x, t=None):
        c = self.c if t is None else self.center_at(t)
        d = np.asarray(x) - c
        r = np.sqrt(np.sum(d * d, axis=-1, keepdims=True))
        return d, r

    def gap(self, x, t=None):
        _, r = self._dr(x, t)
        r = r[..., 0]
        return (self.R - r) if self.inside else (r - self.R)

    def normal(self, x, t=None):
        d, r = self._dr(x, t)
        nrm = d / r
        return -nrm if self.inside else nrm


def rigid_penalty_eval(x, obstacle, *, k, mu=0.0, k_t=None, x_prev, ft):
    """The frictional rigid-penalty contact kernel — the contact analog of
    ``CompiledElement.element_rk_batch``: one call over all nodes returns everything.

    Given current positions ``x`` (n, dim), the obstacle, the penalty/friction params, and
    the committed friction state (``x_prev`` stick anchors, ``ft`` tangential forces),
    returns ``(R, K, ft_new, active)``:
      - ``R`` (n, dim) — per-node residual force (``k g n`` + friction), 0 where inactive;
      - ``K`` (n, dim, dim) — per-node complex-step tangent (consistent, possibly non-sym);
      - ``ft_new`` (n, dim) — the new committed tangential force (for ``commit``);
      - ``active`` (n,) bool — the frozen active set (``g < 0``).
    Active set + stick/slip set are frozen from the **real** ``x`` (the only non-smooth
    switches); within each branch the force is analytic so complex step is exact. Reused by
    both the serial ``RigidContact`` operator and the distributed solver (node-local).
    """
    x = np.asarray(x, dtype=float)
    n_cn, dim = x.shape
    kt = float(k if k_t is None else k_t)
    g = np.real(obstacle.gap(x))
    active = g < 0.0
    R = np.zeros((n_cn, dim))
    K = np.zeros((n_cn, dim, dim))
    ft_new = np.zeros((n_cn, dim))
    if not active.any():
        return R, K, ft_new, active
    xa, xpa, fpa = x[active], x_prev[active], ft[active]

    # stick/slip frozen from the real active positions
    if mu > 0.0:
        n = obstacle.normal(xa)
        ds = xa - xpa
        ds = ds - np.sum(ds * n, axis=-1)[..., None] * n
        ftp = fpa - np.sum(fpa * n, axis=-1)[..., None] * n
        ft_trial = ftp + kt * ds
        cap0 = mu * k * (-g[active])
        slip = np.sqrt(np.sum(ft_trial ** 2, axis=-1)) > cap0
    else:
        slip = np.zeros(len(xa), dtype=bool)

    def nt(xx):                                       # (na,dim) -> (f_normal, f_tangential)
        gg = obstacle.gap(xx)
        nn = obstacle.normal(xx)
        fn = k * gg[..., None] * nn
        if mu <= 0.0:
            return fn, np.zeros_like(fn)
        d = xx - xpa
        d = d - np.sum(d * nn, axis=-1)[..., None] * nn           # P_t (x - x_prev)
        ftp_ = fpa - np.sum(fpa * nn, axis=-1)[..., None] * nn    # committed ft → tangent plane
        ftr = ftp_ + kt * d
        cap = mu * k * (-gg)                          # μ|f_n|
        nrm = np.sqrt(np.sum(ftr ** 2, axis=-1))                 # complex-step-safe |·|
        fts = ftr.copy()
        if slip.any():                               # slip: return to the Coulomb cone
            fts[slip] = (cap[slip] / nrm[slip])[..., None] * ftr[slip]
        return fn, fts

    fn, fts = nt(xa)
    R[active] = fn + fts
    ft_new[active] = fts
    Ka = np.empty((len(xa), dim, dim))
    h = 1e-30
    for j in range(dim):                              # dim complex evals, all active nodes
        xp = xa.astype(complex)
        xp[:, j] += 1j * h
        fnp, ftsp = nt(xp)
        Ka[:, :, j] = (fnp + ftsp).imag / h
    K[active] = Ka
    return R, K, ft_new, active


class RigidContact:
    """Penalty contact of a set of nodes against a rigid obstacle.

    Frictionless by default (``mu=0``); set ``mu>0`` for **incremental penalty-Coulomb
    friction**. The normal residual is the penalty gradient ``k g n`` (active where ``g<0``,
    frozen from the real iterate). Friction adds a tangential force from an elastic *stick*
    spring anchored at the contact point, return-mapped to the Coulomb cone ``|f_t| ≤ μ|f_n|``
    — the friction analogue of plastic return-mapping, so it is **stateful** (the committed
    tangential force is per-node state, updated in ``commit``).

    Complex-step gives the consistent tangent with **both** the active set *and* the
    stick/slip set frozen from the real iterate; friction's normal-tangential coupling makes
    that tangent non-symmetric, which the solvers handle. The state is held internally (like
    ``CompiledElement.svars``) so it survives load steps and distributes node-locally (each
    rank owns its contact nodes' state). The math lives in :func:`rigid_penalty_eval`.
    """

    def __init__(self, nodes_ref, contact_nodes, obstacle, *, dof_per_node,
                 comps=None, k=1.0e3, mu=0.0, k_t=None):
        self.X = np.asarray(nodes_ref, dtype=float)
        self.cn = np.asarray(contact_nodes, dtype=int)
        self.obs = obstacle
        self.dpn = int(dof_per_node)
        self.comps = (np.arange(self.X.shape[1]) if comps is None
                      else np.asarray(comps, dtype=int))
        self.k = float(k)
        self.mu = float(mu)
        self.kt = float(k if k_t is None else k_t)
        self.dim = len(self.comps)
        # (n_cn, dim) global-DOF map for the contact nodes' displacement components
        self.gd = self.cn[:, None] * self.dpn + self.comps[None, :]
        # friction state (committed): stick anchor positions + tangential force per node
        self._x_prev = self.X[self.cn].copy()
        self._ft = np.zeros((len(self.cn), self.dim))

    def _positions(self, U):
        return self.X[self.cn] + np.asarray(U, dtype=float)[self.gd]   # (n_cn, dim)

    def _eval(self, x):
        return rigid_penalty_eval(x, self.obs, k=self.k, mu=self.mu, k_t=self.kt,
                                  x_prev=self._x_prev, ft=self._ft)

    def residual(self, U, state, t, dt) -> Residual:
        R, _K, _ft, active = self._eval(self._positions(U))
        if not active.any():
            return Residual(np.array([], dtype=int), np.array([]))
        return Residual(self.gd[active].ravel(), R[active].ravel())

    def tangent(self, U, state, t, dt) -> Tangent:
        _R, K, _ft, active = self._eval(self._positions(U))
        if not active.any():
            return Tangent(np.array([], dtype=int), np.array([], dtype=int), np.array([]))
        gda, Ka = self.gd[active], K[active]          # (na, dim), (na, dim, dim)
        na = gda.shape[0]
        rows = np.broadcast_to(gda[:, :, None], (na, self.dim, self.dim))
        cols = np.broadcast_to(gda[:, None, :], (na, self.dim, self.dim))
        return Tangent(rows.ravel(), cols.ravel(), Ka.ravel())

    def commit(self, U, state, t, dt):
        if self.mu <= 0.0:
            return state                              # frictionless: stateless
        x = self._positions(U)
        _R, _K, ft_new, _active = self._eval(x)
        self._ft = ft_new                             # inactive nodes reset (separation)
        self._x_prev = x                              # advance the stick anchor
        return state


def rigid_barrier_eval(x, obstacle, *, dhat, kappa, mass=None, mu=0.0, eps=1.0e-4,
                       x0=None, ppf_norm=False, stiff_k=None, kinematic=False,
                       constraint_tol=0.01, thickness=None):
    """Cubic-barrier rigid contact kernel (Stage 4) — penetration-free, well-conditioned.

    Barrier energy per node ``B(d) = (s/3)(d̂-d)³`` on the *active* band ``d < d̂`` (``d`` =
    signed gap, >0 separated), 0 beyond ``d̂``, with **stiffness** ``s``. Cubic (not the IPC
    log-barrier) on purpose: bounded stiffness, C² (C¹ force), polynomial → **no NaN** even at
    ``d ≤ 0``. Non-penetration is enforced by the CCD step bound (see ``max_step``), not by the
    energy → ∞; the barrier supplies a smooth force ``B'(d) n = -s(d̂-d)² n``.

    **Adaptive stiffness (``mass`` given — ppf-style, requires dynamics).** Fixed ``s = κ`` has
    a capacity bound (max force ``κd̂²``) — too soft and the gap collapses; too stiff and the
    Newton step ill-conditions. The ppf fix scales the barrier by ``s = κ + M/d²``: the inertial
    term ``M/d²`` → ∞ as the gap closes, giving **gap-dependent capacity** (no hand-tuned κ),
    while ``κ`` sets the well-conditioned baseline away from contact. (This is the
    ``wᵀ(K_elast + M/g²)w`` recipe with ``K_elast→κ``; ``M`` = the node's lumped mass, so it
    only makes sense under the dynamic driver.)

    **ppf-normalized barrier (``ppf_norm=True``).** The reference ppf-contact-solver separates
    barrier *shape* from contact *stiffness*: the cubic shape is normalized by ``2/d̂``, so the
    activation distance no longer sets the force magnitude. The force becomes
    ``- (2/d̂)·s·(d̂-d)² n`` and the barrier energy ``(2·s/(3·d̂))·(d̂-d)³``. This makes ``κ`` a
    material/dynamic stiffness (force/length) rather than a shape coefficient, which is the
    form used in the ppf stiffness recipe ``s = wᵀ(K+M/g²)w``.

    **Pre-computed stiffness (``stiff_k``).** If a caller supplies a per-contact-node ``stiff_k``
    array, it overrides the ``κ + M/d²`` recipe. This lets :class:`RigidBarrierContact` inject
    the ppf ``wᵀ K_elast w`` normal-direction elastic stiffness on top of the inertial term.

    **Kinematic obstacles (``kinematic=True``).** For prescribed-motion walls (the CoupFE
    equivalent of ppf's ``kinematic`` floor/sphere flag), the effective gap used in the barrier
    is floored at ``constraint_tol * dhat``. This keeps the repulsive force bounded when the
    obstacle pushes into a node and prevents the solve from sticking to a moving wall.

    **Pass-through (``thickness``).** A finite thickness turns the obstacle into a shell: nodes
    that have penetrated deeper than ``thickness`` are ignored, allowing the body to pass
    through rather than being trapped by a deep-penetration barrier force.

    **Friction (``mu > 0``, ``x0`` given — ppf/IPC smoothed, semi-implicit).** The barrier already
    supplies the normal-force magnitude ``λ_n = s(d̂-d)²``. With ``P = I - n⊗n`` (tangent projection)
    and ``dx = x - x0`` (tangential slip since the step start ``x0``), the friction adds residual
    ``R_t = λ (P·dx)`` and tangent ``K_t = λ P`` (symmetric PSD) with ``λ = μ λ_n / max(ε, ‖P·dx‖)``:
    one smooth expression that is the whole stick/slip law (slip ``‖P·dx‖≥ε`` → ``|R_t|=μλ_n``;
    stick ``<ε`` → a linear spring), always ``|R_t| ≤ μλ_n``. ``λ_n`` and ``n`` are **frozen** here
    (so ``K_t`` is the PSD/Gauss-Newton projection, not the exact Jacobian — exact only in the stick
    band where ``λ`` is constant); friction is therefore semi-implicit (the outer step updates them).
    Analytic — no complex step. See ``docs/theory/contact_dynamics.md`` §3b. ``x0`` is the step-start
    contact-node positions (the operator advances it in ``commit``); requires the dynamic driver.

    Returns ``(R, K, active)``: residual, complex-step tangent (the ``M/d²`` g-dependence enters
    it — the auto-stiffening), frozen active mask.
    """
    x = np.asarray(x, dtype=float)
    n_cn, dim = x.shape
    d = np.real(obstacle.gap(x))
    active = d < dhat                                  # incl. penetration: cubic stays finite
    if thickness is not None:
        active = active & (d > -float(thickness))      # shell pass-through
    R = np.zeros((n_cn, dim))
    K = np.zeros((n_cn, dim, dim))
    if not active.any():
        return R, K, active
    xa = x[active]
    mass_a = (None if mass is None
              else np.broadcast_to(np.asarray(mass, dtype=float), (n_cn,))[active])
    shape = (2.0 / dhat) if ppf_norm else 1.0          # ppf geometry-normalized cubic
    stiff_a = (None if stiff_k is None
               else np.asarray(stiff_k, dtype=float)[active])

    # Effective gap for force/stiffness: kinematic obstacles are floored to avoid sticking.
    d_eff = d[active].copy()
    if kinematic:
        d_eff = np.maximum(d_eff, float(constraint_tol) * dhat)

    def force(xx):                                    # B'(d) n = -shape·s·(d̂-d)² n, analytic
        dd = obstacle.gap(xx)
        nn = obstacle.normal(xx)
        if kinematic:
            # Freeze the kinematic floor from the real iterate; keeps complex step smooth.
            dd = np.where(np.real(dd) < float(constraint_tol) * dhat,
                          float(constraint_tol) * dhat, dd)
        g = dhat - dd                                 # > 0 in the active band
        if stiff_a is not None:
            s = stiff_a
        elif mass_a is None:
            s = kappa
        else:
            # Floor the squared gap so M/d² never overflows as the CCD bound
            # lets d approach machine precision; keep the barrier C¹ everywhere.
            dd2 = np.real(dd) ** 2
            s = kappa + mass_a / np.maximum(dd2, 1.0e-24)
        return (-shape * s * g * g)[..., None] * nn

    R[active] = force(xa)
    Ka = np.empty((len(xa), dim, dim))
    h = 1e-30
    for j in range(dim):
        xp = xa.astype(complex)
        xp[:, j] += 1j * h
        Ka[:, :, j] = force(xp).imag / h

    if mu > 0.0 and x0 is not None:                   # ppf/IPC smoothed friction (analytic, PSD)
        na = np.real(obstacle.normal(xa))             # frozen normal
        ga = dhat - d_eff
        if stiff_a is not None:
            sa = stiff_a
        elif mass_a is None:
            sa = kappa
        else:
            sa = kappa + mass_a / np.maximum(d_eff ** 2, 1.0e-24)
        lam_n = shape * sa * ga * ga                  # |f_n| from the barrier (frozen)
        dxa = xa - np.asarray(x0, dtype=float)[active]            # slip since step start
        pdx = dxa - np.sum(dxa * na, axis=-1)[..., None] * na     # P·dx (tangential)
        ut = np.sqrt(np.sum(pdx * pdx, axis=-1))                  # ‖P·dx‖
        lam = mu * lam_n / np.maximum(eps, ut)
        R[active] += lam[..., None] * pdx                          # R_t = λ P·dx
        eye = np.eye(dim)
        P = eye[None, ...] - na[:, :, None] * na[:, None, :]       # I - n⊗n
        Ka += lam[:, None, None] * P                               # K_t = λ P (SPD)

    K[active] = Ka
    return R, K, active


class RigidBarrierContact:
    """Penetration-free penalty-free contact against a rigid obstacle (IPC-style, Stage 4).

    Uses the cubic barrier :func:`rigid_barrier_eval` (force → ∞ as the gap → 0) plus a
    **CCD step bound** (:meth:`max_step`) so the Newton line search never steps a node
    through the obstacle. Together they *guarantee* every iterate stays strictly separated
    (gap > 0) — no tunable penetration like the penalty `RigidContact`. Frictionless and
    stateless; the obstacle must be initially separated (gap > 0 at all contact nodes).

    ``dhat`` = barrier activation distance (forces appear within it); ``kappa`` = barrier
    stiffness; ``eta`` = the fraction of the current gap a step may close (CCD safety).

    ``mass`` (per contact node, requires the dynamic driver) enables the **ppf-style adaptive
    stiffness** ``s = κ + M/d²``: the inertial ``M/d²`` term gives gap-dependent capacity
    (→∞ as the gap closes) so a fixed κ no longer has to be hand-sized to the load — it fixes
    the hard-impact fragility of the fixed-κ barrier. ``M`` is the node's lumped mass (e.g.
    ``lumped_mass(...)`` at the contact nodes).

    ``ppf_norm=True`` switches to the geometry-normalized cubic barrier used by the reference
    ppf-contact-solver: the barrier shape carries a ``2/dhat`` factor, so ``kappa`` becomes a
    true contact stiffness (force/length) and the activation distance no longer sets the force
    magnitude. This is the form that pairs naturally with the ppf ``wᵀ(K+M/g²)w`` stiffness
    recipe.

    ``elastic_op`` (operator or list, optional) takes the ppf stiffness recipe one step further:
    it estimates the normal-direction elastic stiffness ``wᵀ K_elast w`` at each active contact
    node from the bulk tangent(s) and adds it to ``s`` on top of ``κ + M/d²``. Pass the same
    bulk operator(s) you give to the driver (e.g. an :class:`~coupfe.operators.element_group.ElementGroup`).
    ``ndof`` is the global system size; if omitted it is inferred from the contact-node DOF map.
    """

    def __init__(self, nodes_ref, contact_nodes, obstacle, *, dof_per_node, comps=None,
                 dhat=0.05, kappa=1.0e2, eta=0.9, mass=None, mu=0.0, friction_eps=1.0e-4,
                 ppf_norm=False, elastic_op=None, ndof=None):
        self.X = np.asarray(nodes_ref, dtype=float)
        self.cn = np.asarray(contact_nodes, dtype=int)
        self.obs = obstacle
        self.dpn = int(dof_per_node)
        self.comps = (np.arange(self.X.shape[1]) if comps is None
                      else np.asarray(comps, dtype=int))
        self.dhat = float(dhat)
        self.kappa = float(kappa)
        self.eta = float(eta)
        self.mass = (None if mass is None else np.asarray(mass, dtype=float))
        self.mu = float(mu)
        self.friction_eps = float(friction_eps)
        self.ppf_norm = bool(ppf_norm)
        self.elastic_op = elastic_op
        self.dim = len(self.comps)
        self.gd = self.cn[:, None] * self.dpn + self.comps[None, :]
        self.ndof = (ndof if ndof is not None
                     else int(self.gd.max()) + 1)
        # smoothed friction reference = the contact-node positions at the step start
        # (advanced in commit). mu=0 → unused (stateless, frictionless backward-compat).
        self._x0 = self.X[self.cn].copy()

    def _positions(self, U):
        return self.X[self.cn] + np.asarray(U, dtype=float)[self.gd]

    def _elastic_normal_stiffness(self, U, state, t, dt, active, normals):
        """Per-active-node estimate of ``nᵀ K_elast n`` from the supplied bulk operator(s)."""
        ops = (self.elastic_op if isinstance(self.elastic_op, (list, tuple))
               else [self.elastic_op])
        rows, cols, vals = [], [], []
        for op in ops:
            T = op.tangent(U, state, t, dt)
            if len(T.values) == 0:
                continue
            rows.append(np.asarray(T.rows, dtype=int))
            cols.append(np.asarray(T.cols, dtype=int))
            vals.append(np.asarray(T.values, dtype=float))
        if not rows:
            return np.zeros(active.sum())
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        n = int(max(rows.max(), cols.max(), self.ndof - 1)) + 1
        K = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        gd_a = self.gd[active]
        na = gd_a.shape[0]
        K_dir = np.zeros(na)
        for a in range(na):
            gd = gd_a[a]
            block = K[gd[:, None], gd].toarray()
            K_dir[a] = float(normals[a] @ block @ normals[a])
        # Positive part only — an indefinite bulk tangent should not destabilize contact.
        return np.maximum(K_dir, 0.0)

    def _eval(self, x, U=None, state=None, t=None, dt=None):
        x0 = self._x0 if self.mu > 0.0 else None
        stiff_k = None
        if self.elastic_op is not None and U is not None:
            d = np.real(self.obs.gap(x))
            active = d < self.dhat
            if active.any():
                normals = np.real(self.obs.normal(x[active]))
                K_el = self._elastic_normal_stiffness(U, state, t, dt, active, normals)
                base = np.full(active.sum(), self.kappa)
                if self.mass is not None:
                    mass_a = np.broadcast_to(self.mass, (len(x),))[active]
                    base = base + mass_a / np.maximum(d[active] ** 2, 1.0e-24)
                stiff_k = np.zeros(len(x))
                stiff_k[active] = base + K_el
        return rigid_barrier_eval(
            x, self.obs, dhat=self.dhat, kappa=self.kappa,
            mass=self.mass, mu=self.mu, eps=self.friction_eps,
            x0=x0, ppf_norm=self.ppf_norm, stiff_k=stiff_k,
            kinematic=getattr(self.obs, 'kinematic', False),
            constraint_tol=getattr(self.obs, 'constraint_tol', 0.01),
            thickness=getattr(self.obs, 'thickness', None))

    def residual(self, U, state, t, dt) -> Residual:
        R, _K, active = self._eval(self._positions(U), U, state, t, dt)
        if not active.any():
            return Residual(np.array([], dtype=int), np.array([]))
        return Residual(self.gd[active].ravel(), R[active].ravel())

    def tangent(self, U, state, t, dt) -> Tangent:
        _R, K, active = self._eval(self._positions(U), U, state, t, dt)
        if not active.any():
            return Tangent(np.array([], dtype=int), np.array([], dtype=int), np.array([]))
        gda, Ka = self.gd[active], K[active]
        na = gda.shape[0]
        rows = np.broadcast_to(gda[:, :, None], (na, self.dim, self.dim))
        cols = np.broadcast_to(gda[:, None, :], (na, self.dim, self.dim))
        return Tangent(rows.ravel(), cols.ravel(), Ka.ravel())

    def commit(self, U, state, t, dt):
        if self.mu > 0.0:                             # advance the friction slip reference
            self._x0 = self._positions(U)
        return state                                  # otherwise stateless

    def max_step(self, U, dU, t=None, dt=None):
        """CCD: the largest α∈(0,1] keeping every contact node's gap above a tiny floor.

        Linear estimate ``gap(α) ≈ gap + α (Δx·n)`` (exact for a flat obstacle) capped so no
        node closes more than ``eta`` of its *remaining margin* above ``gap_floor``, then a
        bisection safety net that re-checks the *actual* gap. If ``dt`` is provided, the
        safety check evaluates the obstacle at the time corresponding to ``α`` along the step
        ``[t-dt, t]``; this keeps the predictor jump penetration-free for moving obstacles
        without requiring the obstacle to move at every Newton correction.
        """
        x = self._positions(U)
        dx = np.asarray(dU, dtype=float)[self.gd]         # (n_cn, dim) contact-node increment

        def _gap_at(xq, time):
            try:
                return self.obs.gap(xq, time)
            except TypeError:
                return self.obs.gap(xq)

        def _normal_at(xq, time):
            try:
                return self.obs.normal(xq, time)
            except TypeError:
                return self.obs.normal(xq)

        d0 = _gap_at(x, t)
        s = np.sum(dx * _normal_at(x, t), axis=-1)        # d(gap)/dα, linear estimate
        # A small absolute floor, plus a fraction of dhat, keeps the CCD bound well-scaled.
        gap_floor = max(1.0e-4, 1.0e-3 * self.dhat)
        if getattr(self.obs, 'kinematic', False):
            gap_floor = max(gap_floor,
                            getattr(self.obs, 'constraint_tol', 0.01) * self.dhat)
        alpha = 1.0
        approaching = s < 0.0
        if approaching.any():
            margin = np.maximum(d0[approaching] - gap_floor, 0.0)
            cand = -self.eta * margin / s[approaching]
            alpha = min(alpha, float(cand.min()))
        for _ in range(30):                               # safety: keep the TRUE gap above floor
            time_a = t
            if dt is not None and t is not None:
                time_a = t - dt + alpha * dt              # fraction α into the predictor step
            if np.all(_gap_at(x + alpha * dx, time_a) > gap_floor):
                break
            alpha *= 0.5
        return max(alpha, 1.0e-12)


class SurfaceContact2D:
    """General 2D surface contact (Stage 3): every surface vertex vs its closest
    NON-ADJACENT surface edge — so it handles **multi-body** *and* **self-contact** (a
    body's surface against itself). A vertex's own edges are excluded (it never
    self-forces); the closest penetrated edge applies the node-to-segment penalty (the
    Stage-2 force/tangent: rank-1 PSD ``k d⊗d``, small-sliding).

    The candidate search is brute-force here (every vertex × every edge); the broad-phase
    (spatial hash / BVH) and the consistent large-sliding tangent are the refinements
    (vectorized numpy broad-phase, then numba/Rust — `docs/dev/contact.md`). Adjacency
    exclusion is the simple "vertex not an endpoint of the edge" rule; k-ring exclusion for
    tight folds is a later refinement. Edge normal = +90° of node0→node1.
    """

    def __init__(self, nodes_ref, vertices, edges, *, dof_per_node, comps=None, k=1.0e3,
                 search_band=None):
        self.X = np.asarray(nodes_ref, dtype=float)
        self.vertices = np.asarray(vertices, dtype=int)
        self.edges = np.asarray(edges, dtype=int)
        self.dpn = int(dof_per_node)
        self.comps = (np.arange(2) if comps is None else np.asarray(comps, dtype=int))
        self.k = float(k)
        # broad-phase band: candidates within this distance of a vertex. Penalty has no fixed d̂,
        # so the band must cover the (penetration depth + reach); None ⇒ max current edge length
        # (generous — catches up to ~one element of penetration). Identical to brute force as long
        # as penetrations stay within the band.
        self.search_band = search_band

    def _x(self, U, n):
        return self.X[n] + np.asarray(U, dtype=float)[n * self.dpn + self.comps]

    def _positions_all(self, U):
        U = np.asarray(U, dtype=float)
        return self.X + U.reshape(len(self.X), self.dpn)[:, self.comps]

    def _candidates(self, U):
        pos = self._positions_all(U)
        if self.search_band is None:
            xe = pos[self.edges]
            band = float(np.sqrt(((xe[:, 1] - xe[:, 0]) ** 2).sum(-1)).max())
        else:
            band = self.search_band
        return candidate_pairs(pos, self.vertices, self.edges, band, exclude_incident=True)

    def _closest_nonadjacent(self, U, v, xs, edge_idxs=None):
        best = None
        idxs = range(len(self.edges)) if edge_idxs is None else edge_idxs
        for ei in idxs:
            a, b = int(self.edges[ei][0]), int(self.edges[ei][1])
            if v == a or v == b:                       # exclude the vertex's own edges
                continue
            x0, x1 = self._x(U, a), self._x(U, b)
            e = x1 - x0
            L2 = float(e @ e)
            xi = min(1.0, max(0.0, float((xs - x0) @ e) / L2))
            xm = x0 + xi * e
            nrm = np.array([-e[1], e[0]]) / np.sqrt(L2)
            g = float((xs - xm) @ nrm)
            dist = abs(g) if 0.0 < xi < 1.0 else float(np.linalg.norm(xs - xm))
            if best is None or dist < best[0]:
                best = (dist, a, b, xi, g, nrm)
        return None if best is None else best[1:]

    def _active(self, U):
        cands = self._candidates(U)                    # broad phase (O(N))
        out = []
        for v in self.vertices:
            ce = cands.get(int(v))
            if ce is None:
                continue
            c = self._closest_nonadjacent(U, int(v), self._x(U, v), edge_idxs=ce)
            if c is not None and c[3] < 0.0:           # g < 0: penetration
                out.append((v,) + c)
        return out

    def residual(self, U, state, t, dt) -> Residual:
        rows, vals = [], []
        for v, a, b, xi, g, nrm in self._active(U):
            rows += [v * self.dpn + self.comps, a * self.dpn + self.comps,
                     b * self.dpn + self.comps]
            vals += [self.k * g * nrm, -self.k * g * (1 - xi) * nrm,
                     -self.k * g * xi * nrm]
        if not rows:
            return Residual(np.array([], dtype=int), np.array([]))
        return Residual(np.concatenate(rows), np.concatenate(vals))

    def tangent(self, U, state, t, dt) -> Tangent:
        R, C, V = [], [], []
        for v, a, b, xi, g, nrm in self._active(U):
            gd = np.concatenate([v * self.dpn + self.comps, a * self.dpn + self.comps,
                                 b * self.dpn + self.comps])
            d = np.concatenate([nrm, -(1 - xi) * nrm, -xi * nrm])
            Ke = self.k * np.outer(d, d)
            for i, gi in enumerate(gd):
                for j, gj in enumerate(gd):
                    R.append(gi); C.append(gj); V.append(Ke[i, j])
        return Tangent(np.array(R, dtype=int), np.array(C, dtype=int), np.array(V))

    def commit(self, U, state, t, dt):
        return state


class DeformableContact2D:
    """Node-to-segment penalty contact in 2D — secondary nodes vs primary linear edges.

    Each secondary node projects to its closest primary edge; if the signed gap ``g < 0``
    (penetration) a penalty force ``k|g|`` along the edge normal pushes the secondary out,
    distributed **equal-and-opposite** to the two edge nodes by the shape functions
    ``(1-ξ, ξ)``. The primary surface DEFORMS — the edge nodes are DOFs, so the contact
    couples the 3 nodes (6 DOFs). Small-sliding (normal & projection frozen *per iteration*,
    re-evaluated each Newton step): the ``(residual, tangent)`` pair is consistent for the
    frozen energy — ``R = k g d``, ``K = k d⊗d`` (rank-1 PSD), ``d = [n, -(1-ξ)n, -ξn]``.

    Scope: the large-sliding **consistent** tangent (differentiating ``n``/``ξ``) is a later
    refinement; the closest-edge **search** is brute-force here (numba/Rust at scale —
    `docs/dev/contact.md`). Edge orientation: the normal is the +90° rotation of
    ``node0→node1``, so the body lies on the −n side (a secondary on the +n side separates).
    """

    def __init__(self, nodes_ref, secondary_nodes, primary_edges, *, dof_per_node,
                 comps=None, k=1.0e3, search_band=None):
        self.X = np.asarray(nodes_ref, dtype=float)
        self.sec = np.asarray(secondary_nodes, dtype=int)
        self.edges = np.asarray(primary_edges, dtype=int)
        self.dpn = int(dof_per_node)
        self.comps = (np.arange(2) if comps is None else np.asarray(comps, dtype=int))
        self.k = float(k)
        self.search_band = search_band                 # broad-phase band (None ⇒ max edge length)

    def _x(self, U, n):
        return self.X[n] + np.asarray(U, dtype=float)[n * self.dpn + self.comps]

    def _positions_all(self, U):
        U = np.asarray(U, dtype=float)
        return self.X + U.reshape(len(self.X), self.dpn)[:, self.comps]

    def _candidates(self, U):
        pos = self._positions_all(U)
        if self.search_band is None:
            xe = pos[self.edges]
            band = float(np.sqrt(((xe[:, 1] - xe[:, 0]) ** 2).sum(-1)).max())
        else:
            band = self.search_band
        return candidate_pairs(pos, self.sec, self.edges, band, exclude_incident=False)

    def _closest(self, U, xs, edge_idxs=None):
        """Closest primary edge to secondary point ``xs`` → (a, b, ξ, gap, normal). ``edge_idxs``
        (broad-phase candidates) restricts the scan; ``None`` scans all edges."""
        best = None
        idxs = range(len(self.edges)) if edge_idxs is None else edge_idxs
        for ei in idxs:
            a, b = int(self.edges[ei][0]), int(self.edges[ei][1])
            x0, x1 = self._x(U, a), self._x(U, b)
            e = x1 - x0
            L2 = float(e @ e)
            xi = min(1.0, max(0.0, float((xs - x0) @ e) / L2))
            xm = x0 + xi * e
            nrm = np.array([-e[1], e[0]]) / np.sqrt(L2)        # +90° of node0→node1
            g = float((xs - xm) @ nrm)
            dist = abs(g) if 0.0 < xi < 1.0 else float(np.linalg.norm(xs - xm))
            if best is None or dist < best[0]:
                best = (dist, a, b, xi, g, nrm)
        return None if best is None else best[1:]

    def _active(self, U):
        """Active (penetrating) secondary→edge pairs, via the broad phase (O(N))."""
        cands = self._candidates(U)
        out = []
        for s in self.sec:
            ce = cands.get(int(s))
            if ce is None:
                continue
            c = self._closest(U, self._x(U, s), edge_idxs=ce)
            if c is not None and c[3] < 0.0:           # g < 0: penetration (c = a,b,xi,g,nrm)
                out.append((int(s),) + c)
        return out

    def residual(self, U, state, t, dt) -> Residual:
        rows, vals = [], []
        for s, a, b, xi, g, nrm in self._active(U):
            rows += [s * self.dpn + self.comps, a * self.dpn + self.comps,
                     b * self.dpn + self.comps]
            vals += [self.k * g * nrm, -self.k * g * (1 - xi) * nrm,
                     -self.k * g * xi * nrm]
        if not rows:
            return Residual(np.array([], dtype=int), np.array([]))
        return Residual(np.concatenate(rows), np.concatenate(vals))

    def tangent(self, U, state, t, dt) -> Tangent:
        R, C, V = [], [], []
        for s, a, b, xi, g, nrm in self._active(U):
            gd = np.concatenate([s * self.dpn + self.comps, a * self.dpn + self.comps,
                                 b * self.dpn + self.comps])
            d = np.concatenate([nrm, -(1 - xi) * nrm, -xi * nrm])   # ∂g/∂u (frozen n,ξ)
            Ke = self.k * np.outer(d, d)                            # rank-1 PSD
            for i, gi in enumerate(gd):
                for j, gj in enumerate(gd):
                    R.append(gi); C.append(gj); V.append(Ke[i, j])
        return Tangent(np.array(R, dtype=int), np.array(C, dtype=int), np.array(V))

    def commit(self, U, state, t, dt):
        return state


# ---------------------------------------------------------------------------
# Deformable–deformable barrier contact (2D, node-to-segment) — the modern,
# penetration-free version of DeformableContact2D. Both sides are DOFs: the
# secondary node *and* the two primary-edge nodes get equal-and-opposite barrier
# forces, so two deformable bodies (or a body against itself) cannot interpenetrate.
# ---------------------------------------------------------------------------

def _seg_kinematics(xs, xa, xb):
    """Node-to-line kinematics in 2D, **complex-safe** (no abs/min/max on the value path).

    Returns the *signed* gap ``d`` of point ``xs`` to the infinite line through ``xa→xb``
    (``d>0`` on the +n side, where ``n`` = +90° of ``xa→xb``; the primary body lies on −n,
    matching :class:`DeformableContact2D`), the foot parameter ``xi`` (``∈[0,1]`` ⇔ the foot
    is on the segment), and the unit normal ``n``. Note ``d`` is the distance to the *line*
    and is independent of ``xi`` — the perpendicular distance to a line does not depend on
    where along it the foot falls; ``xi`` only gates the active region and the reaction split.
    """
    e = xb - xa
    r = xs - xa
    L2 = e[..., 0] ** 2 + e[..., 1] ** 2
    L = np.sqrt(L2)
    cross = e[..., 0] * r[..., 1] - e[..., 1] * r[..., 0]     # e × r
    d = cross / L
    xi = (r[..., 0] * e[..., 0] + r[..., 1] * e[..., 1]) / L2
    n = np.stack([-e[..., 1], e[..., 0]], axis=-1) / L[..., None]
    return d, xi, n, e, r, L, L2


def _seg_grad_d(xs, xa, xb):
    """Analytic ``∂d/∂[xs, xa, xb]`` (shape ``(..., 6)``), **complex-safe**.

    Derived from ``d = (eₓ r_y − e_y rₓ)/L`` (``e = xb−xa``, ``r = xs−xa``). The three
    nodal gradients sum to **zero** (translation invariance — a rigid shift leaves the gap
    unchanged), which is what makes the assembled barrier force action-reaction by
    construction. ``∂d/∂xs = n`` exactly, so the secondary force reduces to the rigid-obstacle
    kernel; the ``xa``/``xb`` terms carry the normal-rotation (consistent) geometry.
    """
    d, _xi, n, e, r, L, L2 = _seg_kinematics(xs, xa, xb)
    ex, ey = e[..., 0], e[..., 1]
    rx, ry = r[..., 0], r[..., 1]
    ddxs = n                                                  # [-ey, ex]/L
    ddxa = np.stack([ey - ry, rx - ex], axis=-1) / L[..., None] + (d / L2)[..., None] * e
    ddxb = np.stack([ry, -rx], axis=-1) / L[..., None] - (d / L2)[..., None] * e
    return np.concatenate([ddxs, ddxa, ddxb], axis=-1)       # (..., 6)


def deformable_barrier_eval(Xs, Xa, Xb, *, dhat, kappa, mass=None,
                            mu=0.0, eps=1.0e-4, Xs0=None, Xa0=None, Xb0=None, friction_kt=None,
                            ft_prev=None, return_ft=False):
    """Node-to-segment **cubic barrier** kernel (deformable primary + secondary), vectorized.

    Mirrors :func:`rigid_barrier_eval` but the "obstacle" is a *deformable* linear edge, so the
    barrier energy ``B(d) = (s/3)(d̂−d)³`` (``d`` = signed gap to the edge line, >0 separated)
    couples all **three** nodes. The residual is the energy gradient ``∂B/∂X = −s(d̂−d)² ∂d/∂X``
    over the 6 DOFs ``X = [xs, xa, xb]`` — built from the analytic, action-reaction
    :func:`_seg_grad_d`. The 6×6 tangent is the **complex-step** of that force (the *consistent*
    tangent: it includes the normal-rotation geometry, not just the rank-1 ``n⊗n`` term), with
    the feature pair and the active set **frozen** from the real iterate (the only non-smooth
    parts). Adaptive ``s = κ + M/d²`` (``mass`` = secondary node mass, dynamics only) gives the
    same gap-dependent capacity as the rigid kernel.

    Active band: ``d < d̂`` **and** the foot is on the segment (``0 ≤ ξ ≤ 1``) — a node off the
    end of the edge is a point-point case, not handled here (skip; the closest *other* edge or
    the search catches it). Penetration (``d < 0``) stays active and finite (cubic), giving a
    strong recovery force; non-penetration is the job of the CCD step bound, not this energy.

    **Friction (``mu > 0``, ``Xs0/Xa0/Xb0`` given — ppf/IPC smoothed, semi-implicit).** Same model
    as :func:`rigid_barrier_eval`, but the slip is the **relative** tangential motion of the
    secondary vs the (moving) foot point on the deforming edge. With the foot weights ``(1−ξ, ξ)``
    (ξ frozen), the relative slip since the step start is ``dx = (xs − xs0) − [F − F0]``, where
    ``F = xa + ξ(xb−xa)`` and ``F0`` uses the same ξ on the step-start positions. With ``P = I−n⊗n``
    and ``λ = μ λ_n / max(ε, ‖P·dx‖)`` (``λ_n = s(d̂−d)²``), the friction force in the relative
    coordinate is ``λ(P·dx)``, mapped to the 6 DOFs by ``J = [I, −(1−ξ)I, −ξI]`` → residual
    ``Jᵀ λ(P·dx)`` (action-reaction, ``Σ = 0``) and **symmetric PSD** tangent ``λ JᵀP J`` (``λ_n``,
    ``n``, ξ frozen). Analytic; theory §3b. ``X*0`` are the step-start positions (operator-tracked).

    Args: ``Xs, Xa, Xb`` shape ``(n_pair, 2)`` (secondary, edge node 0, edge node 1);
    ``mass`` shape ``(n_pair,)`` or None. Returns ``(R, K, active)`` with ``R`` ``(n_pair, 6)``,
    ``K`` ``(n_pair, 6, 6)``, ``active`` ``(n_pair,)``.
    """
    Xs = np.asarray(Xs, dtype=float)
    Xa = np.asarray(Xa, dtype=float)
    Xb = np.asarray(Xb, dtype=float)
    n_pair = Xs.shape[0]
    R = np.zeros((n_pair, 6))
    K = np.zeros((n_pair, 6, 6))
    d, xi, n_all, _e, _r, _L, _L2 = _seg_kinematics(Xs, Xa, Xb)
    active = (d < dhat) & (xi >= 0.0) & (xi <= 1.0)
    if not active.any():
        return R, K, active
    X6 = np.concatenate([Xs, Xa, Xb], axis=-1)[active]        # (n_act, 6)
    mass_a = (None if mass is None
              else np.broadcast_to(np.asarray(mass, dtype=float), (n_pair,))[active])

    def force(X):                                             # ∂B/∂X = −s(d̂−d)² ∂d/∂X
        xs, xa, xb = X[..., 0:2], X[..., 2:4], X[..., 4:6]
        dd, _xi, _nn, _ee, _rr, _LL, _LL2 = _seg_kinematics(xs, xa, xb)
        gd = _seg_grad_d(xs, xa, xb)
        g = dhat - dd
        s = kappa if mass_a is None else (kappa + mass_a / (dd * dd))
        return (-(s * g * g))[..., None] * gd

    R[active] = force(X6)
    Ka = np.empty((X6.shape[0], 6, 6))
    h = 1e-30
    for j in range(6):
        Xp = X6.astype(complex)
        Xp[:, j] += 1j * h
        Ka[:, :, j] = force(Xp).imag / h

    ft_commit = (np.zeros((np.asarray(Xs).shape[0], 2)) if return_ft else None)   # committed friction force
    if mu > 0.0 and Xs0 is not None:                          # ppf/IPC smoothed friction (analytic)
        na = np.real(n_all[active])                          # frozen edge normal
        xia = xi[active]
        ga = dhat - d[active]
        sa = kappa if mass_a is None else (kappa + mass_a / (d[active] ** 2))
        lam_n = sa * ga * ga                                 # |f_n| from the barrier (frozen)
        xsa, xaa, xba = Xs[active], Xa[active], Xb[active]
        xs0, xa0, xb0 = (np.asarray(Xs0)[active], np.asarray(Xa0)[active],
                         np.asarray(Xb0)[active])
        foot = xaa + xia[:, None] * (xba - xaa)             # current contact point on the edge
        foot0 = xa0 + xia[:, None] * (xb0 - xa0)            # same ξ on step-start positions
        dxr = (xsa - xs0) - (foot - foot0)                  # relative slip since the step start
        pdx = dxr - np.sum(dxr * na, axis=-1)[:, None] * na    # P·dx (tangential)
        ut = np.sqrt(np.sum(pdx * pdx, axis=-1))
        # J = [I, -(1-ξ)I, -ξI] (n_act, 2, 6); residual Jᵀ ffric; P = I - n⊗n
        n_act = xsa.shape[0]
        eye = np.eye(2)
        I_b = np.broadcast_to(eye, (n_act, 2, 2))
        J = np.concatenate([I_b, -(1.0 - xia)[:, None, None] * I_b,
                            -xia[:, None, None] * I_b], axis=2)    # (n_act, 2, 6)
        P = eye[None, ...] - na[:, :, None] * na[:, None, :]       # I - n⊗n
        if friction_kt is None:
            # ppf/IPC SMOOTHED: soft spring λ = μλ_n/max(ε,ut) → bounded creep; PSD tangent λ JᵀPJ
            lam = mu * lam_n / np.maximum(eps, ut)
            R[active] += np.einsum('pki,pk->pi', J, lam[:, None] * pdx)
            Ka += lam[:, None, None] * np.einsum('pki,pkl,plj->pij', J, P, J)
        else:
            # EXACT-STICK return-map (radial return): a stiff stick spring k_t projected onto the EXACT
            # Coulomb cone. The carried tangential force ``ft_prev`` (the friction STATE = ε_p analog) is
            # projected onto the CURRENT tangent plane (handling re-pairing/normal drift) and the step's
            # slip ``k_t·pdx`` is accumulated onto it: ``f = returnmap(P·ft_prev + k_t·pdx, μλ_n)``. With
            # ``ft_prev=None`` (the per-step mode) this reduces exactly to ``f = returnmap(k_t·pdx)``.
            # Slip tangent keeps the −t̂⊗t̂ term the smoothed model drops → no plateau floor (λ_n frozen).
            cap = mu * lam_n                                  # Coulomb cap μ|f_n|
            if ft_prev is not None:                           # PERSISTENT: carry + re-frame the friction force
                fp = np.asarray(ft_prev)[active]
                fp = fp - np.sum(fp * na, axis=-1)[:, None] * na   # project onto the current tangent plane
                ft_trial = fp + friction_kt * pdx             # accumulate this step's slip
            else:
                ft_trial = friction_kt * pdx                  # per-step (zero anchor) — the original mode
            ftm = np.sqrt(np.sum(ft_trial * ft_trial, axis=-1))
            ftms = np.maximum(ftm, 1e-300)
            stick = ftm <= cap
            ft = np.where(stick[:, None], ft_trial, (cap / ftms)[:, None] * ft_trial)   # return-map to cone
            R[active] += np.einsum('pki,pk->pi', J, ft)
            t_hat = ft_trial / ftms[:, None]
            ttT = t_hat[:, :, None] * t_hat[:, None, :]        # t̂⊗t̂
            Pmod = np.where(stick[:, None, None], P, P - ttT)  # P (stick) | P−t̂⊗t̂ (slip, exact cap)
            coef = np.where(stick, friction_kt, cap * friction_kt / ftms)
            Ka += coef[:, None, None] * np.einsum('pki,pkl,plj->pij', J, Pmod, J)
            if return_ft:
                ft_commit[active] = ft                        # the new committed friction force

    K[active] = Ka
    if return_ft:
        return R, K, active, ft_commit
    return R, K, active


def _cross2(u, v):
    """2D scalar cross product ``uₓ v_y − u_y vₓ`` (broadcasts on the last axis)."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def edge_barrier_eval(P, A, B, *, dhat, kappa, mass=None, mu=0.0, eps=1.0e-4, X0=None):
    """Point-edge **cubic barrier** (vertex ``P`` vs deformable edge ``A-B``) — the 2D ALL-PRIMITIVE
    ppf port, the 2D analogue of :func:`coupfe.operators.contact3d.tri_barrier_eval`.

    Gap = **unclassified closest-point distance** ``|P − (w·edge)|`` via the ported
    ``point_edge_coeff_unclassified`` weights ``w`` — which clamp to an endpoint, so a single barrier
    covers BOTH edge-interior (point-edge) and vertex (point-point) contact, exactly as ppf does (no
    type-classification, no dedup). Unlike the node-to-segment :func:`deformable_barrier_eval` (a
    *single closest edge* with a signed line gap that flips at a vertex → chatter → needs the freeze),
    here every nearby (vertex, edge) pair contributes a smooth term, so there is **no closest-edge
    choice to flip** — the all-primitive way to robustness.

    Residual ``−s(d̂−gap)²·[n, −w₀n, −w₁n]`` over ``X=[P,A,B]`` (action-reaction by ``Σ weights = 0``);
    PSD tangent ``2s(d̂−gap)·gradᵀgrad`` (``n,w`` frozen); adaptive ``s = κ + M/gap²``. Frictionless
    (the 2D smoothed-friction port follows). ``P,A,B`` ``(n_pair,2)``; returns ``(R (n,6), K (n,6,6),
    active (n,))``. Penetration-free is the CCD's job, not a signed gap.
    """
    from coupfe.operators.contact3d import point_edge_coeff_unclassified
    P = np.asarray(P, dtype=float); A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    n_pair = P.shape[0]
    R = np.zeros((n_pair, 6)); K = np.zeros((n_pair, 6, 6)); active = np.zeros(n_pair, dtype=bool)
    for i in range(n_pair):
        w = point_edge_coeff_unclassified(P[i], A[i], B[i])
        closest = w[0] * A[i] + w[1] * B[i]
        gap_vec = P[i] - closest
        gap = float(np.sqrt(gap_vec @ gap_vec))
        if gap >= dhat or gap == 0.0:
            continue
        active[i] = True
        n = gap_vec / gap                                  # closest→P (separating) direction
        g = dhat - gap
        # ppf's adaptive capacity M/gap² → ∞ as gap → 0 (the point); ppf never overflows because ACCD
        # guarantees gap > 0. Until ACCD is ported, floor gap in this term so a transient near-zero gap
        # gives a huge-but-FINITE capacity (no FP overflow → no NaN); K scales with s so the Newton step
        # stays bounded and the pair recovers. A safety net, not a physics change.
        gd2 = max(gap, 1.0e-6 * dhat)
        s = kappa if mass is None else (kappa + float(np.asarray(mass)[i]) / (gd2 * gd2))
        grad = np.concatenate([n, -w[0] * n, -w[1] * n])   # ∂gap/∂X (frozen w, n)
        R[i] = -(s * g * g) * grad
        K[i] = (2.0 * s * g) * np.outer(grad, grad)        # PSD rank-1
        if mu > 0.0 and X0 is not None:                    # ppf/IPC smoothed friction (tangent line)
            c = np.array([1.0, -w[0], -w[1]])              # signed weights (Σc=0 ⇒ Σf=0)
            cur = np.array([P[i], A[i], B[i]])             # (3,2) current stencil
            dxr = c @ (cur - X0[i])                        # (2,) relative slip since step start
            pdx = dxr - (dxr @ n) * n                      # tangential component (P·dx)
            ut = float(np.sqrt(pdx @ pdx))
            lam = mu * (s * g * g) / max(eps, ut)          # λ = μ λ_n / max(ε, ‖slip‖)
            Bm = np.zeros((2, 6))
            for j in range(3):
                Bm[:, 2 * j:2 * j + 2] = c[j] * np.eye(2)
            R[i] += Bm.T @ (lam * pdx)
            K[i] += lam * (Bm.T @ (np.eye(2) - np.outer(n, n)) @ Bm)   # symmetric PSD Gauss-Newton
    return R, K, active


class DeformableBarrierContact2D:
    """**Node-to-segment** cubic-barrier contact between **deformable** bodies (2D, Stage 4).

    ACCURATE NAME (do not call this "ppf"): this is the *node-to-segment* contact model — each
    secondary node pairs with its **single closest edge**. It borrows ppf/IPC's **barrier energy**
    (cubic, :func:`deformable_barrier_eval`), **adaptive `s = κ + M/d²` stiffness**, and **smoothed
    friction**, but the *pairing* is the classical node-to-segment scheme, NOT ppf's all-primitive
    barrier. The single closest-edge choice is the discrete thing that flips at a vertex → use
    ``freeze_pairing=True`` (fix the pairing per step) for fine-mesh robustness.

    The **faithful ppf port** — an *all-primitive* barrier (point-edge summed over all nearby pairs,
    the unclassified distance covering point-point at vertices) — is available **opt-in via
    ``all_primitive=True``** (:func:`edge_barrier_eval`), the 2D analogue of
    :class:`~coupfe.operators.contact3d.DeformableBarrierContact3D` (ppf's vertex-face + edge-edge),
    reusing the ported ``point_edge_coeff_unclassified``. It removes the closest-edge fragility at the
    source — **no freeze needed**. Currently **frictionless** (the 2D smoothed-friction port + numba
    narrow-phase + ACCD follow); the default node-to-segment model (+ freeze for fine mesh) stays the
    friction-capable path until then.

    The deformable analogue of :class:`RigidBarrierContact`: secondary nodes vs primary linear
    edges, the cubic barrier on the signed gap, plus a **point-edge CCD** step bound
    (:meth:`max_step`). Both bodies deform — the edge nodes are DOFs and feel the reaction. For two
    bodies, compose two of these (A-nodes vs B-edges *and* B-nodes vs A-edges); for self-contact use
    the search-based sibling. Consistent (complex-step) tangent.

    ``dhat`` barrier band, ``kappa`` stiffness, ``eta`` CCD safety fraction, ``mass`` (per secondary
    node, dynamics only) → adaptive ``s = κ + M/d²``. The closest-edge **search** is brute force here
    (numba at scale — `docs/dev/contact.md`).
    """

    def __init__(self, nodes_ref, secondary_nodes, primary_edges, *, dof_per_node,
                 comps=None, dhat=0.05, kappa=1.0e2, eta=0.9, mass=None,
                 mu=0.0, friction_eps=1.0e-4, friction_kt=None, friction_persistent=False,
                 body_id=None, freeze_pairing=False, all_primitive=False):
        self.X = np.asarray(nodes_ref, dtype=float)
        self.sec = np.asarray(secondary_nodes, dtype=int)
        self.edges = np.asarray(primary_edges, dtype=int)
        # Per-node body id for MULTI-BODY / self-contact use: a node must not contact
        # any edge of its OWN body, not just its incident (1-ring) edges. Without it,
        # the closest non-incident edge can be a NON-ADJACENT edge of the same body,
        # whose signed line gap is negative (the node is on the body's interior side)
        # ⇒ the barrier reads a deep penetration ⇒ a spurious force at the rest state.
        # None ⇒ two-body mode (only incident exclusion), byte-identical to before.
        self.body = (None if body_id is None else np.asarray(body_id, dtype=int))
        self.dpn = int(dof_per_node)
        self.comps = (np.arange(2) if comps is None else np.asarray(comps, dtype=int))
        self.dhat = float(dhat)
        self.kappa = float(kappa)
        self.eta = float(eta)
        self.mass = (None if mass is None else np.asarray(mass, dtype=float))
        self.mu = float(mu)
        self.friction_eps = float(friction_eps)
        # exact-stick (return-map) friction mode: tangential stick stiffness k_t. None ⇒ ppf smoothed.
        self.friction_kt = (None if friction_kt is None else float(friction_kt))
        # finite-sliding persistent friction: carry the committed tangential force per secondary across
        # steps (the ε_p analog), re-framed onto the new edge at re-pairing. Requires friction_kt.
        self.friction_persistent = bool(friction_persistent)
        # PAIRING FREEZE (opt-in): fix each secondary's closest-edge assignment for the whole
        # Newton solve of a step, re-pairing only at commit(). Re-pairing every iteration lets a
        # node at a vertex between two edges flip its closest edge each iterate → the residual
        # oscillates and Newton stalls (the fine-mesh active-set chatter, 2026-06-26). Freezing
        # the *topology* (positions still update with U; the `active` gap-mask still filters) gives
        # a well-posed per-step Newton; the small per-step pairing lag is the standard trade-off.
        self.freeze_pairing = bool(freeze_pairing)
        self._frozen_topo = None
        # ALL-PRIMITIVE mode (opt-in): the faithful 2D ppf port. Each secondary node pairs with EVERY
        # nearby edge (not just the single closest), each via the unclassified-distance cubic barrier
        # (edge_barrier_eval). No closest-edge choice → no flip → no freeze needed. Default off keeps
        # the node-to-segment model byte-identical.
        self.all_primitive = bool(all_primitive)
        self._ft = np.zeros((len(self.sec), 2))
        # smoothed-friction step-start reference: positions of every node that can take part
        # (secondary or edge endpoint), advanced in commit. mu=0 → unused (stateless).
        self._x0 = self.X.copy()
        self._involved = np.unique(np.concatenate([self.sec, self.edges.ravel()]))

    def _x(self, U, n):
        return self.X[n] + np.asarray(U, dtype=float)[n * self.dpn + self.comps]

    def _closest_edge(self, U, xs, v, edge_idxs=None):
        """Closest primary edge (by point-*segment* distance) to secondary ``xs`` (node id ``v``),
        excluding edges incident to ``v``. ``edge_idxs`` (broad-phase candidates) restricts the scan
        to those edge indices; ``None`` scans all edges (brute force)."""
        best = None
        idxs = range(len(self.edges)) if edge_idxs is None else edge_idxs
        same_body = (None if self.body is None else int(self.body[v]))
        for ei in idxs:
            a, b = self.edges[ei]
            if v == a or v == b:
                continue
            if same_body is not None and int(self.body[a]) == same_body:
                continue                                      # same-body: never self-contact
            x0, x1 = self._x(U, a), self._x(U, b)
            e = x1 - x0
            L2 = float(e @ e)
            xi = min(1.0, max(0.0, float((xs - x0) @ e) / L2))
            xm = x0 + xi * e
            dist = float(np.linalg.norm(xs - xm))
            if best is None or dist < best[0]:
                best = (dist, int(a), int(b))
        return None if best is None else (best[1], best[2])

    def _positions_all(self, U):
        """Spatial positions of every node (n_node, dim) at displacement ``U``."""
        U = np.asarray(U, dtype=float)
        return self.X + U.reshape(len(self.X), self.dpn)[:, self.comps]

    def _pair_topology(self, U):
        """The active node→closest-edge assignment as a list of ``(sec_index, v, a, b)``.

        Recomputed from the broad phase + closest-edge scan each call — UNLESS
        ``freeze_pairing`` and a cached assignment from this step exists, in which case the
        cached topology is reused (only the geometry, in :meth:`_pairs`, updates with ``U``).
        ``commit`` clears the cache so the next step re-pairs."""
        if self.freeze_pairing and self._frozen_topo is not None:
            return self._frozen_topo
        cands = candidate_pairs(self._positions_all(U), self.sec, self.edges, self.dhat)
        topo = []
        for i, v in enumerate(self.sec):
            v = int(v)
            ce = cands.get(v)
            if ce is None:                                    # no edge within d̂ → inactive
                continue
            if self.all_primitive:                            # ALL nearby edges (ppf-style), not the closest
                same_body = (None if self.body is None else int(self.body[v]))
                for ei in ce:
                    a, b = int(self.edges[ei][0]), int(self.edges[ei][1])
                    if v == a or v == b:                      # incident (1-ring) exclusion
                        continue
                    if same_body is not None and int(self.body[a]) == same_body:
                        continue
                    topo.append((i, v, a, b))
                continue
            c = self._closest_edge(U, self._x(U, v), v, edge_idxs=ce)
            if c is None:
                continue
            topo.append((i, v, int(c[0]), int(c[1])))
        if self.freeze_pairing:
            self._frozen_topo = topo
        return topo

    def _pairs(self, U):
        """Per secondary node → (gdofs, Xs, Xa, Xb, mass_s, Xs0, Xa0, Xb0) for its closest edge.
        The ``*0`` arrays are the step-start positions (friction reference).

        **Broad phase (band = d̂):** only edges within d̂ of a vertex can be active (point-segment
        distance < d̂ ⇔ line gap < d̂ with the foot on the segment), and a vertex with no edge in
        the band contributes no force; so restricting the closest-edge scan to the spatial-hash
        candidates is *identical* to the brute-force active set — just O(N) instead of O(N²)."""
        topo = self._pair_topology(U)
        gd, Xs, Xa, Xb, ms, Xs0, Xa0, Xb0, seci = [], [], [], [], [], [], [], [], []
        for (i, v, a, b) in topo:
            gd.append(np.concatenate([v * self.dpn + self.comps,
                                      a * self.dpn + self.comps,
                                      b * self.dpn + self.comps]))
            Xs.append(self._x(U, v)); Xa.append(self._x(U, a)); Xb.append(self._x(U, b))
            Xs0.append(self._x0[v]); Xa0.append(self._x0[a]); Xb0.append(self._x0[b])
            ms.append(np.nan if self.mass is None else float(self.mass[i]))
            seci.append(i)                                    # the secondary index (for persistent friction)
        if not gd:
            return None
        mass = None if self.mass is None else np.asarray(ms)
        return (np.array(gd, dtype=int), np.array(Xs), np.array(Xa), np.array(Xb), mass,
                np.array(Xs0), np.array(Xa0), np.array(Xb0), np.array(seci, dtype=int))

    def _eval(self, p, want_ft=False):
        gd, Xs, Xa, Xb, mass, Xs0, Xa0, Xb0, seci = p
        if self.all_primitive:                                # 2D ppf all-primitive barrier + smoothed friction
            fr = self.mu > 0.0
            X0 = (np.stack([Xs0, Xa0, Xb0], axis=1) if fr else None)   # (n_pair,3,2) step-start stencil
            R, K, active = edge_barrier_eval(Xs, Xa, Xb, dhat=self.dhat, kappa=self.kappa, mass=mass,
                                             mu=self.mu, eps=self.friction_eps, X0=X0)
            if want_ft:                                       # smoothed friction is stateless → no committed ft
                return R, K, active, np.zeros((len(active), 2))
            return R, K, active
        fr = self.mu > 0.0
        ft_prev = (self._ft[seci] if (self.friction_persistent and fr) else None)
        return deformable_barrier_eval(
            Xs, Xa, Xb, dhat=self.dhat, kappa=self.kappa, mass=mass, ft_prev=ft_prev, return_ft=want_ft,
            mu=self.mu, eps=self.friction_eps, friction_kt=self.friction_kt,
            Xs0=(Xs0 if fr else None), Xa0=(Xa0 if fr else None), Xb0=(Xb0 if fr else None))

    def residual(self, U, state, t, dt) -> Residual:
        p = self._pairs(U)
        if p is None:
            return Residual(np.array([], dtype=int), np.array([]))
        R, _K, active = self._eval(p)
        if not active.any():
            return Residual(np.array([], dtype=int), np.array([]))
        return Residual(p[0][active].ravel(), R[active].ravel())

    def tangent(self, U, state, t, dt) -> Tangent:
        p = self._pairs(U)
        if p is None:
            return Tangent(np.array([], dtype=int), np.array([], dtype=int), np.array([]))
        _R, K, active = self._eval(p)
        if not active.any():
            return Tangent(np.array([], dtype=int), np.array([], dtype=int), np.array([]))
        gda, Ka = p[0][active], K[active]
        na = gda.shape[0]
        rows = np.broadcast_to(gda[:, :, None], (na, 6, 6))
        cols = np.broadcast_to(gda[:, None, :], (na, 6, 6))
        return Tangent(rows.ravel(), cols.ravel(), Ka.ravel())

    def commit(self, U, state, t, dt):
        if self.mu > 0.0:                                     # advance the friction state
            if self.friction_persistent:                     # carry the committed friction force per secondary
                p = self._pairs(U)
                new_ft = np.zeros_like(self._ft)             # secondaries out of contact → 0 (lose the stick)
                if p is not None:
                    *_unused, ft_commit = self._eval(p, want_ft=True)
                    new_ft[p[8]] = ft_commit                 # p[8] = per-pair secondary index
                self._ft = new_ft
            U = np.asarray(U, dtype=float)                    # advance the per-step slip reference
            dofs = self._involved[:, None] * self.dpn + self.comps[None, :]
            self._x0[self._involved] = self.X[self._involved] + U[dofs]
        self._frozen_topo = None                              # re-pair at the start of the next step
        return state

    def max_step(self, U, dU):
        """Point-edge CCD: largest ``α∈(0,1]`` so no secondary crosses *any* nearby edge along
        ``U+α dU``. The secondary is colinear with the (moving) edge when ``e(α) × r(α) = 0``,
        a quadratic ``Aα²+Bα+C`` (``e``, ``r`` linear in α). Take the smallest root in ``(0,1]``
        whose foot is on the segment, cap by ``eta``; then a bisection safety net re-checks the
        *true* signed gap stays positive (covers the rotating-normal geometry).

        **Broad phase, band = d̂ + 2·reach.** Unlike the active-set search (band d̂), CCD must catch
        an edge the secondary could *cross this step*, so the candidate band includes the per-step
        reach (``reach`` = max nodal displacement; relative approach ≤ 2·reach). Each secondary is
        CCD-checked against *all* its candidate edges (not just the closest — a correctness upgrade),
        and the safety net reuses the same candidate set (valid since nodes move ≤ reach for α≤1)."""
        dU = np.asarray(dU, dtype=float)
        if len(self._involved):
            du_inv = dU[self._involved[:, None] * self.dpn + self.comps[None, :]]
            reach = float(np.max(np.sqrt(np.sum(du_inv * du_inv, axis=-1))))
        else:
            reach = 0.0
        cands = candidate_pairs(self._positions_all(U), self.sec, self.edges,
                                self.dhat + 2.0 * reach)
        alpha = 1.0
        for v in self.sec:
            v = int(v)
            ce = cands.get(v)
            if ce is None:
                continue
            xs = self._x(U, v)
            dxs = dU[v * self.dpn + self.comps]
            for ei in ce:
                a, b = int(self.edges[ei][0]), int(self.edges[ei][1])
                xa, xb = self._x(U, a), self._x(U, b)
                dxa = dU[a * self.dpn + self.comps]
                dxb = dU[b * self.dpn + self.comps]
                e0, de = xb - xa, dxb - dxa
                r0, dr = xs - xa, dxs - dxa
                root = self._first_crossing(_cross2(de, dr),
                                            _cross2(e0, dr) + _cross2(de, r0),
                                            _cross2(e0, r0), e0, de, r0, dr)
                if root is not None:
                    alpha = min(alpha, self.eta * root)
        for _ in range(30):                          # safety net: keep the TRUE gap > 0
            if not self._any_penetration(U + alpha * dU, cands):
                break
            alpha *= 0.5
        return max(alpha, 0.0)

    @staticmethod
    def _first_crossing(A, B, C, e0, de, r0, dr):
        """Smallest ``α∈(0,1]`` root of ``Aα²+Bα+C=0`` whose foot parameter ξ(α)∈[0,1]."""
        roots = []
        if abs(A) < 1e-14:
            if abs(B) > 1e-14:
                roots = [-C / B]
        else:
            disc = B * B - 4 * A * C
            if disc >= 0.0:
                sq = np.sqrt(disc)
                roots = [(-B - sq) / (2 * A), (-B + sq) / (2 * A)]
        best = None
        for a in roots:
            if a <= 1e-12 or a > 1.0:
                continue
            e = e0 + a * de
            r = r0 + a * dr
            L2 = float(e @ e)
            if L2 <= 0.0:
                continue
            xi = float(r @ e) / L2
            if -0.05 <= xi <= 1.05 and (best is None or a < best):   # small ξ margin
                best = a
        return best

    def _any_penetration(self, U, cands):
        """True if any secondary penetrates its closest candidate edge (d ≤ 0, foot on segment).
        Reuses the broad-phase candidate set ``cands`` (the safety-net check, O(N))."""
        for v in self.sec:
            v = int(v)
            ce = cands.get(v)
            if ce is None:
                continue
            c = self._closest_edge(U, self._x(U, v), v, edge_idxs=ce)
            if c is None:
                continue
            a, b = c
            d, xi, *_ = _seg_kinematics(self._x(U, v), self._x(U, a), self._x(U, b))
            if 0.0 <= xi <= 1.0 and d <= 0.0:
                return True
        return False
