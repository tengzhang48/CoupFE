"""3D deformable–deformable contact primitives (Stage 5) — PORTED from ppf-contact-solver.

3D needs two primitives where 2D needed one: **point-triangle** (vertex vs face) and **edge-edge**
(two skew segments — new in 3D). The hard, error-prone part is the *robust closest-feature geometry*
(a point-triangle contact degenerates to point-edge or point-vertex when the foot leaves the face;
edge-edge to point-edge), and the **CCD**. Rather than re-derive these, we **port the battle-tested
implementations from `ppf-contact-solver`** (`contact/distance.hpp`, `contact/accd.hpp`, Apache-2.0)
— a faithful numpy translation, verified against analytic cases + our gates. Adopt the algorithm,
not the stack.

These portions were translated and modified for NumPy/CoupFE. The upstream
provenance and license are recorded in the repository ``NOTICE`` file.

The contact gap is the **closest-point distance** `gap = |p_a − p_b|` where `p_a, p_b` are the
closest points on the two features, written via the ported distance *coefficients* (closest-point
weights `w`). The barrier then mirrors 2D: residual `−s(d̂−gap)²·∂gap/∂X`, PSD tangent
`2s(d̂−gap)·∂gapᵀ∂gap`, with `∂gap/∂X = [n·(weights)]` (the weights are `∂p/∂node`); `n` and the
weights frozen (the `psd` mode). Penetration-free comes from CCD (ported next), not a signed gap —
exactly as in ppf. This file: point-edge + point-triangle coeffs + the point-triangle barrier.
Edge-edge coeffs + ACCD are the next port.
"""

from __future__ import annotations

import numpy as np

try:                                                     # numba LBVH broad-phase (grid = fallback)
    from coupfe.operators.bvh_numba import build_bvh, query_csr
    _HAS_BVH = True
except Exception:
    _HAS_BVH = False

try:                                                     # numba self-contact incident-exclusion + ee pairs
    from coupfe.operators.contact3d_numba import vf_drop_incident_nb, ee_pairs_nb
    _HAS_INCIDENT = True
except Exception:
    _HAS_INCIDENT = False


# ---------------------------------------------------------------------------- distance coefficients
# Faithful ports of ppf-contact-solver/crates/.../contact/distance.hpp (Apache-2.0).

def point_edge_coeff(p, e0, e1):
    """Closest-point weights (w0, w1) of ``p`` on segment ``e0-e1`` (unclamped t → clamp by the
    caller's *_unclassified). Port of ``point_edge_distance_coeff``."""
    r = e1 - e0
    d = float(r @ r)
    if d > 0.0:
        t = float(r @ (p - e0)) / d
        return np.array([1.0 - t, t])
    return np.array([0.5, 0.5])


def point_edge_coeff_unclassified(p, e0, e1):
    c = point_edge_coeff(p, e0, e1)
    if 0.0 <= c[0] <= 1.0:
        return c
    return np.array([1.0, 0.0]) if c[0] > 1.0 else np.array([0.0, 1.0])


def _solve2(a, b):
    """``x = adj(a)·b`` (NOT divided by det) + det, for a 2×2 ``a`` — ppf's ``solve``."""
    det = a[0, 0] * a[1, 1] - a[1, 0] * a[0, 1]
    adj = np.array([[a[1, 1], -a[0, 1]], [-a[1, 0], a[0, 0]]])
    return adj @ b, float(det)


def point_triangle_coeff(p, t0, t1, t2):
    """Barycentric weights of the closest point of ``p`` on the triangle plane; degenerate-triangle
    fallback to the longest edge. Port of ``point_triangle_distance_coeff``."""
    r0 = t1 - t0
    r1 = t2 - t0
    a = np.column_stack([r0, r1])              # 3×2
    at = a.T
    c, det = _solve2(at @ a, at @ (p - t0))
    if det != 0.0:
        u, v = c[0] / det, c[1] / det
        return np.array([1.0 - u - v, u, v])
    e0, e1, e2 = t1 - t0, t2 - t1, t0 - t2     # degenerate → longest edge
    l0, l1, l2 = float(e0 @ e0), float(e1 @ e1), float(e2 @ e2)
    if l0 >= l1 and l0 >= l2:
        w = point_edge_coeff(p, t0, t1); return np.array([w[0], w[1], 0.0])
    if l1 >= l2:
        w = point_edge_coeff(p, t1, t2); return np.array([0.0, w[0], w[1]])
    w = point_edge_coeff(p, t2, t0); return np.array([w[1], 0.0, w[0]])


def point_triangle_coeff_unclassified(p, t0, t1, t2):
    """Robust barycentric weights — if the foot leaves the triangle, fall back to the closest edge
    or vertex. Port of ``point_triangle_distance_coeff_unclassified``."""
    c = point_triangle_coeff(p, t0, t1, t2)
    if c.min() >= 0.0 and c.max() <= 1.0:
        return c
    if c[0] < 0.0:
        w = point_edge_coeff(p, t1, t2)
        if 0.0 <= w[0] <= 1.0:
            return np.array([0.0, w[0], w[1]])
        return np.array([0.0, 1.0, 0.0]) if w[0] > 1.0 else np.array([0.0, 0.0, 1.0])
    if c[1] < 0.0:
        w = point_edge_coeff(p, t0, t2)
        if 0.0 <= w[0] <= 1.0:
            return np.array([w[0], 0.0, w[1]])
        return np.array([1.0, 0.0, 0.0]) if w[0] > 1.0 else np.array([0.0, 0.0, 1.0])
    w = point_edge_coeff(p, t0, t1)
    if 0.0 <= w[0] <= 1.0:
        return np.array([w[0], w[1], 0.0])
    return np.array([1.0, 0.0, 0.0]) if w[0] > 1.0 else np.array([0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------- point-triangle barrier
def _smoothed_friction_3d(c, nodes_cur, nodes0, n, lam_n, mu, eps):
    """ppf/IPC smoothed friction for ONE 3D contact pair (vertex-face or edge-edge).

    The 3D generalization of the 2D node-to-segment friction (theory §3b). ``c`` (k,) are the
    SIGNED kinematic weights so the contact gap-vector is ``Σ cᵢ Xᵢ`` — vertex-face
    ``[1,−w₀,−w₁,−w₂]``, edge-edge ``[a₀,a₁,−b₀,−b₁]`` — i.e. exactly the weights the barrier uses
    (``grad = Bᵀn`` with block ``Bᵢ = cᵢ I₃``). Because ``Σ cᵢ = 0`` (barycentric / edge weights),
    a rigid co-translation produces zero relative slip and the nodal forces sum to zero
    (action-reaction) BY CONSTRUCTION. ``nodes_cur``/``nodes0`` (k,3) are the stencil's current /
    step-start positions, ``n`` the frozen barrier normal, ``lam_n = s(d̂−gap)²`` the barrier normal
    force. Relative tangential slip since the step start is ``dx = P·Σ cᵢ(Xᵢ−X0ᵢ)``, ``P = I−n⊗n``;
    the friction force ``λ(P·dx)`` (``λ = μ λ_n / max(ε,‖dx‖)``) maps to the 3k DOFs by ``Bᵀ`` →
    residual ``Bᵀ λ(P·dx)`` and **symmetric-PSD** Gauss-Newton tangent ``λ Bᵀ P B`` (``λ_n``, ``n``,
    weights frozen — the same three ppf approximations as 2D). Returns ``(Rf (3k,), Kf (3k,3k))``.
    """
    c = np.asarray(c, dtype=float)
    k = c.shape[0]
    dxr = c @ (np.asarray(nodes_cur, float) - np.asarray(nodes0, float))   # (3,) relative slip
    pdx = dxr - (dxr @ n) * n                                              # tangential (P·dx)
    ut = float(np.sqrt(pdx @ pdx))
    lam = mu * lam_n / max(eps, ut)
    B = np.zeros((3, 3 * k))
    for i in range(k):
        B[:, 3 * i:3 * i + 3] = c[i] * np.eye(3)
    Rf = B.T @ (lam * pdx)                                                 # (3k,)
    Pp = np.eye(3) - np.outer(n, n)
    Kf = lam * (B.T @ Pp @ B)                                             # (3k,3k) symmetric PSD
    return Rf, Kf


def _returnmap_friction_3d(c, nodes_cur, nodes0, n, lam_n, mu, friction_kt, ft_prev):
    """3D EXACT-STICK return-map friction for ONE pair (the 3D analog of the 2D ``friction_kt`` mode).

    Same kinematics as :func:`_smoothed_friction_3d` (signed weights ``c``, ``B_i = c_i I₃``, tangent plane
    ``P = I − n⊗n``, per-step slip ``pdx = P·Σ cᵢ(Xᵢ−X0ᵢ)``), but the force is a stiff stick spring ``k_t``
    return-mapped onto the EXACT Coulomb cone, carrying the committed tangential force ``ft_prev`` (the ε_p
    analog) across steps: ``f = returnmap(P·ft_prev + k_t·pdx, μλ_n)``. ``P·ft_prev`` re-frames the carried
    force onto the current tangent plane (handling re-pairing / normal drift). ``ft_prev=None`` ⇒ the
    per-step return-map. Returns ``(Rf (3k,), Kf (3k,3k), ft_new (3,))``."""
    c = np.asarray(c, dtype=float)
    k = c.shape[0]
    dxr = c @ (np.asarray(nodes_cur, float) - np.asarray(nodes0, float))
    Pp = np.eye(3) - np.outer(n, n)
    pdx = Pp @ dxr                                                     # tangential slip this step
    cap = mu * lam_n
    if ft_prev is not None:
        ft_trial = Pp @ np.asarray(ft_prev, float) + friction_kt * pdx   # carried force, re-framed + slip
    else:
        ft_trial = friction_kt * pdx
    ftm = float(np.sqrt(ft_trial @ ft_trial)); ftms = max(ftm, 1e-300)
    stick = ftm <= cap
    ft = ft_trial if stick else (cap / ftms) * ft_trial               # return-map onto the cone
    Bm = np.zeros((3, 3 * k))
    for i in range(k):
        Bm[:, 3 * i:3 * i + 3] = c[i] * np.eye(3)
    Rf = Bm.T @ ft
    if stick:
        Kf = friction_kt * (Bm.T @ Pp @ Bm)
    else:
        that = ft_trial / ftms
        Kf = (cap * friction_kt / ftms) * (Bm.T @ (Pp - np.outer(that, that)) @ Bm)
    return Rf, Kf, ft


def tri_barrier_eval(P, A, B, C, *, dhat, kappa, mass=None, mu=0.0, eps=1.0e-4, X0=None,
                     friction_kt=None, ft_prev=None, return_ft=False):
    """Point-triangle **cubic barrier** (vertex P vs deformable face A,B,C), per-pair over ``n_pair``.

    Gap = closest-point distance ``|P − (w·triangle)|`` via the *robust* ported weights ``w`` (handles
    face/edge/vertex contact). Active where ``gap < d̂``. Residual = ``−s(d̂−gap)²·[n, −w₀n, −w₁n, −w₂n]``
    over ``X=[P,A,B,C]`` (action-reaction: Σforce=0, Στorque=0 since ``P−closest ∥ n``); PSD tangent
    ``2s(d̂−gap)·gradᵀgrad`` (``n``, ``w`` frozen). Adaptive ``s = κ + M/gap²`` (``mass`` per P).
    Penetration-free is the CCD's job (ported next), not a signed gap.

    Args ``P,A,B,C`` ``(n_pair,3)``; ``mass`` ``(n_pair,)`` or None. Returns ``(R, K, active)`` with
    ``R`` ``(n_pair,12)``, ``K`` ``(n_pair,12,12)``, ``active`` ``(n_pair,)``.
    """
    P = np.asarray(P, dtype=float); A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float); C = np.asarray(C, dtype=float)
    n_pair = P.shape[0]
    R = np.zeros((n_pair, 12))
    K = np.zeros((n_pair, 12, 12))
    active = np.zeros(n_pair, dtype=bool)
    ft_commit = np.zeros((n_pair, 3)) if return_ft else None     # committed friction force per pair
    for i in range(n_pair):
        w = point_triangle_coeff_unclassified(P[i], A[i], B[i], C[i])
        closest = w[0] * A[i] + w[1] * B[i] + w[2] * C[i]
        gap_vec = P[i] - closest
        gap = float(np.sqrt(gap_vec @ gap_vec))
        if gap >= dhat or gap == 0.0:
            continue
        active[i] = True
        n = gap_vec / gap                                  # closest→P (separating) direction
        g = dhat - gap
        s = kappa if mass is None else (kappa + float(np.asarray(mass)[i]) / (gap * gap))
        grad = np.concatenate([n, -w[0] * n, -w[1] * n, -w[2] * n])    # ∂gap/∂X (frozen w, n)
        R[i] = -(s * g * g) * grad
        K[i] = (2.0 * s * g) * np.outer(grad, grad)        # PSD rank-1
        if mu > 0.0 and X0 is not None:                    # friction (tangent plane)
            c = np.array([1.0, -w[0], -w[1], -w[2]])
            cur = np.array([P[i], A[i], B[i], C[i]])
            if friction_kt is None:                        # ppf smoothed (default)
                Rf, Kf = _smoothed_friction_3d(c, cur, X0[i], n, lam_n=s * g * g, mu=mu, eps=eps)
            else:                                          # EXACT-STICK return-map (persistent if ft_prev)
                fp = None if ft_prev is None else ft_prev[i]
                Rf, Kf, ftc = _returnmap_friction_3d(c, cur, X0[i], n, s * g * g, mu, friction_kt, fp)
                if return_ft:
                    ft_commit[i] = ftc
            R[i] += Rf
            K[i] += Kf
    if return_ft:
        return R, K, active, ft_commit
    return R, K, active


# ---------------------------------------------------------------------------- edge-edge coefficients
def edge_edge_coeff(ea0, ea1, eb0, eb1):
    """Closest-point weights (a0,a1,b0,b1) of segments ea0-ea1 and eb0-eb1 (pa=a0·ea0+a1·ea1,
    pb=b0·eb0+b1·eb1). Direct 2×2 solve, fall back to ea0-vs-edge-b, then a 4-step alternating
    projection refinement. Port of ``edge_edge_distance_coeff``."""
    r0 = ea1 - ea0
    r1 = eb1 - eb0
    a = np.column_stack([r0, -r1])                          # 3×2
    x, det = _solve2(a.T @ a, a.T @ (eb0 - ea0))
    c = point_edge_coeff(ea0, eb0, eb1)
    result = np.array([1.0, 0.0, c[0], c[1]])
    pb = c[0] * eb0 + c[1] * eb1
    min_dist = float((ea0 - pb) @ (ea0 - pb))
    if det != 0.0:
        xx = x / det
        direct = np.array([1.0 - xx[0], xx[0], 1.0 - xx[1], xx[1]])
        pa_d = direct[0] * ea0 + direct[1] * ea1
        pb_d = direct[2] * eb0 + direct[3] * eb1
        if float((pa_d - pb_d) @ (pa_d - pb_d)) < min_dist:
            result = direct
    x0, x1 = result[1], result[3]
    q0, q1 = eb0 - ea0, eb1 - ea0
    p0, p1 = ea0 - eb0, ea1 - eb0
    for _ in range(4):                                     # alternating projection refinement
        x0 = point_edge_coeff(x1 * r1, p0, p1)[1]
        x1 = point_edge_coeff(x0 * r0, q0, q1)[1]
    return np.array([1.0 - x0, x0, 1.0 - x1, x1])


def edge_edge_coeff_unclassified(ea0, ea1, eb0, eb1):
    """Robust edge-edge weights — if the interior solution leaves [0,1], pick the best of the four
    endpoint-vs-edge cases (COG-centered for stability). Port of
    ``edge_edge_distance_coeff_unclassified``."""
    c = edge_edge_coeff(ea0, ea1, eb0, eb1)
    if c.min() >= 0.0 and c.max() <= 1.0:
        return c

    def _clamp(cc):
        if cc[0] < 0.0:
            return np.array([0.0, 1.0])
        if cc[0] > 1.0:
            return np.array([1.0, 0.0])
        return cc

    c1 = _clamp(point_edge_coeff(ea0, eb0, eb1))
    c2 = _clamp(point_edge_coeff(ea1, eb0, eb1))
    c3 = _clamp(point_edge_coeff(eb0, ea0, ea1))
    c4 = _clamp(point_edge_coeff(eb1, ea0, ea1))
    types = [np.array([1.0, 0.0, c1[0], c1[1]]), np.array([0.0, 1.0, c2[0], c2[1]]),
             np.array([c3[0], c3[1], 1.0, 0.0]), np.array([c4[0], c4[1], 0.0, 1.0])]
    cog = 0.25 * (ea0 + ea1 + eb0 + eb1)
    pts = [ea0 - cog, ea1 - cog, eb0 - cog, eb1 - cog]
    best, bd = types[0], np.inf
    for t in types:
        xa = t[0] * pts[0] + t[1] * pts[1]
        xb = t[2] * pts[2] + t[3] * pts[3]
        d = float((xb - xa) @ (xb - xa))
        if d < bd:
            bd, best = d, t
    return best


def edge_edge_barrier_eval(P0, P1, Q0, Q1, *, dhat, kappa, mass=None, mu=0.0, eps=1.0e-4, X0=None):
    """Edge-edge cubic barrier (segment P0-P1 vs Q0-Q1), per-pair. Gap = closest-point distance via
    the robust ported weights; residual ``−s(d̂−gap)²·[a0 n, a1 n, −b0 n, −b1 n]`` over
    ``X=[P0,P1,Q0,Q1]`` (action-reaction), PSD tangent. ``mass`` per pair (on P0/P1's edge)."""
    P0 = np.asarray(P0, float); P1 = np.asarray(P1, float)
    Q0 = np.asarray(Q0, float); Q1 = np.asarray(Q1, float)
    n_pair = P0.shape[0]
    R = np.zeros((n_pair, 12))
    K = np.zeros((n_pair, 12, 12))
    active = np.zeros(n_pair, dtype=bool)
    for i in range(n_pair):
        a0, a1, b0, b1 = edge_edge_coeff_unclassified(P0[i], P1[i], Q0[i], Q1[i])
        gap_vec = (a0 * P0[i] + a1 * P1[i]) - (b0 * Q0[i] + b1 * Q1[i])
        gap = float(np.sqrt(gap_vec @ gap_vec))
        if gap >= dhat or gap == 0.0:
            continue
        active[i] = True
        n = gap_vec / gap
        g = dhat - gap
        s = kappa if mass is None else (kappa + float(np.asarray(mass)[i]) / (gap * gap))
        grad = np.concatenate([a0 * n, a1 * n, -b0 * n, -b1 * n])
        R[i] = -(s * g * g) * grad
        K[i] = (2.0 * s * g) * np.outer(grad, grad)
        if mu > 0.0 and X0 is not None:                    # ppf smoothed friction (tangent plane)
            c = np.array([a0, a1, -b0, -b1])
            cur = np.array([P0[i], P1[i], Q0[i], Q1[i]])
            Rf, Kf = _smoothed_friction_3d(c, cur, X0[i], n, lam_n=s * g * g, mu=mu, eps=eps)
            R[i] += Rf
            K[i] += Kf
    return R, K, active


# ---------------------------------------------------------------------------- ACCD (additive CCD)
# Port of ppf-contact-solver/crates/.../contact/accd.hpp (Apache-2.0). Conservative advancement:
# returns the time-of-impact (max safe fraction of the step) that keeps features apart.

def _centerize(pts):
    """Subtract the centroid (translation-invariance / float precision); pts (C, 3)."""
    return pts - pts.mean(axis=0)


def _max_relative_u(u):
    """Max pairwise relative speed ‖u_i − u_j‖ over the C points; u (C, 3)."""
    m = 0.0
    C = len(u)
    for i in range(C):
        for j in range(i + 1, C):
            d = u[i] - u[j]
            m = max(m, float(d @ d))
    return float(np.sqrt(m))


def accd_toi(x0, dx, sq_dist, *, offset=0.0, max_t=1.0, reduction=0.2, max_iter=100):
    """Additive-CCD time-of-impact for points ``x0`` (C,3) moving by ``dx`` over [0,max_t], keeping
    ``sqrt(sq_dist) > offset``. ``sq_dist(x)`` = squared closest-feature distance. Port of
    ``accd::ccd_helper`` (conservative advancement; returns ``max_t`` if never within reach)."""
    u_max = _max_relative_u(dx)
    if u_max == 0.0:
        return max_t
    toi = 0.0
    eps = reduction * (np.sqrt(sq_dist(x0)) - offset)
    target = eps + offset
    eps_sqr = eps * eps
    inv = 1.0 / u_max
    for _ in range(max_iter):
        d2 = sq_dist(x0 + toi * dx)
        d_minus_target = (d2 - target * target) / (np.sqrt(d2) + target)
        if (max_t - toi) * u_max < d_minus_target - eps:
            return max_t
        if toi > 0.0 and d_minus_target * d_minus_target < eps_sqr:
            break
        toi_next = toi + d_minus_target * inv
        if toi_next == toi:
            break
        toi = toi_next
        if toi > max_t:
            return max_t
    return toi


def _pt_sq(x):                                             # x (4,3) = [P, t0, t1, t2]
    P, t0, t1, t2 = x
    c = point_triangle_coeff_unclassified(P, t0, t1, t2)
    y = c[0] * (t0 - P) + c[1] * (t1 - P) + c[2] * (t2 - P)
    return float(y @ y)


def _ee_sq(x):                                             # x (4,3) = [p0, p1, q0, q1]
    p0, p1, q0, q1 = x
    c = edge_edge_coeff_unclassified(p0, p1, q0, q1)
    d = (c[0] * p0 + c[1] * p1) - (c[2] * q0 + c[3] * q1)
    return float(d @ d)


def point_triangle_toi(p0, p1, t00, t01, t02, t10, t11, t12, **kw):
    """ACCD time-of-impact for vertex p (p0→p1) vs triangle (t0*,t1*,t2* : 0→1 positions)."""
    x0 = _centerize(np.array([p0, t00, t01, t02], dtype=float))
    dx = _centerize(np.array([p1 - p0, t10 - t00, t11 - t01, t12 - t02], dtype=float))
    return accd_toi(x0, dx, _pt_sq, **kw)


def edge_edge_toi(a00, a01, b00, b01, a10, a11, b10, b11, **kw):
    """ACCD time-of-impact for edge (a00-a01)→(a10-a11) vs edge (b00-b01)→(b10-b11)."""
    x0 = _centerize(np.array([a00, a01, b00, b01], dtype=float))
    dx = _centerize(np.array([a10 - a00, a11 - a01, b10 - b00, b11 - b01], dtype=float))
    return accd_toi(x0, dx, _ee_sq, **kw)


def _pe_sq(x):                                             # x (3,2) = [P, e0, e1]  — 2D point-edge
    P, e0, e1 = x
    c = point_edge_coeff_unclassified(P, e0, e1)
    d = P - (c[0] * e0 + c[1] * e1)
    return float(d @ d)


def point_edge_toi(p0, p1, a0, a1, b0, b1, **kw):
    """**2D** ACCD time-of-impact for vertex ``p`` (p0→p1) vs edge ``A``(a0→a1)–``B``(b0→b1).

    The 2D analogue of :func:`point_triangle_toi`, reusing the generic :func:`accd_toi` with the 2D
    point-edge squared distance. Guarantees the secondary stays > 0 from the moving edge (the gap→0
    feasibility ppf needs so the M/d² barrier never overflows). Returns the max safe step fraction."""
    x0 = _centerize(np.array([p0, a0, b0], dtype=float))
    dx = _centerize(np.array([p1 - p0, a1 - a0, b1 - b0], dtype=float))
    return accd_toi(x0, dx, _pe_sq, **kw)


# ---------------------------------------------------------------------------- 3D broad-phase
# AABB-with-margin candidate search (the AABB primitive is ppf's `aabb.hpp`; we use a uniform 3D
# grid for the structure — BVH is ppf's extreme-scale escalation, `docs/dev/contact.md`). Two pair
# types: vertex-face (point-triangle) and edge-edge. Each is a SUPERSET of the true within-d̂ set
# (a feature's d̂-expanded AABB contains everything within d̂ of it), so no contact is missed; the
# narrow phase (`tri_barrier_eval` / `edge_edge_barrier_eval`) computes the real gap + active set.

def _grid_bin(los, his, cell):
    """Bin axis-aligned boxes ``[los, his]`` (each ``(n, 3)``) into integer 3D cells of size
    ``cell``; returns ``{(i,j,k): [box indices]}``."""
    grid = {}
    cl = np.floor(los / cell).astype(np.int64)
    ch = np.floor(his / cell).astype(np.int64)
    for e in range(len(los)):
        for i in range(int(cl[e, 0]), int(ch[e, 0]) + 1):
            for j in range(int(cl[e, 1]), int(ch[e, 1]) + 1):
                for k in range(int(cl[e, 2]), int(ch[e, 2]) + 1):
                    grid.setdefault((i, j, k), []).append(e)
    return grid


def vertex_face_candidates(positions, vertices, faces, dhat, *, cell=None, self_contact=False):
    """Per vertex → candidate face indices (superset of faces within ``d̂``). ``faces`` ``(n_f, 3)``
    node ids. Each face's (AABB + d̂) is binned; a vertex queries its own cell, then a **point-to-AABB
    distance prune** drops cell co-occupants whose AABB is provably > ``d̂`` away (tight superset: since
    ``dist(v, AABB) ≤ dist(v, face) < d̂`` for any face truly within ``d̂``, the prune never drops a real
    contact — it only removes the grid's coarse-cell false positives).

    ``self_contact=True`` (one body's surface against itself): drop candidates where the vertex is a node
    of the face (incident-exclusion — gap ≡ 0; numba kernel). Faces in the vertex's 1-ring that do NOT
    contain it sit ≥ one element-height away, so for ``d̂`` < mesh size this is the complete exclusion."""
    pos = np.asarray(positions, dtype=float)
    faces = np.asarray(faces, dtype=int)
    vertices = np.asarray(vertices, dtype=int)
    if len(faces) == 0 or len(vertices) == 0:
        return {}
    xf = pos[faces]                                        # (n_f, 3, 3)
    fmin = xf.min(1); fmax = xf.max(1)                     # raw face AABBs (n_f, 3)
    pv = pos[vertices]
    d2 = dhat * dhat
    out = {}
    if _HAS_BVH:                                           # LBVH query (non-uniform-robust) + AABB prune
        bv = build_bvh(fmin, fmax, 8)
        cand, ptr = query_csr(pv - dhat, pv + dhat, fmin, fmax, *bv)
        if not cand.size:
            return out
        # VECTORIZED point-to-AABB prune over ALL candidates at once (no per-vertex Python loop):
        vrep = np.repeat(np.arange(len(vertices)), np.diff(ptr))     # owning vertex per candidate
        vp = pv[vrep]
        clamped = np.minimum(np.maximum(vp, fmin[cand]), fmax[cand])
        keep = np.sum((vp - clamped) ** 2, axis=1) < d2             # dist(v, faceAABB)² < d̂²
        kv = vrep[keep]; kf = cand[keep]
        if self_contact and kv.size:                                # drop incident (vertex ∈ face) pairs
            vid = vertices[kv].astype(np.int64)
            if _HAS_INCIDENT:
                ok = vf_drop_incident_nb(vid, kf.astype(np.int64), faces)
            else:
                ok = ~(faces[kf] == vid[:, None]).any(axis=1)
            kv = kv[ok]; kf = kf[ok]
        if kv.size:                                                 # group surviving faces by vertex
            uniq, starts = np.unique(kv, return_index=True)
            ends = np.append(starts[1:], kv.size)
            for u, s, e in zip(uniq, starts, ends):
                out[int(vertices[u])] = np.sort(kf[s:e])
        return out
    if cell is None:                                       # uniform-grid fallback (no numba)
        cell = float(max((fmax - fmin).max(), dhat))
    cell = max(cell, 1e-30)
    grid = _grid_bin(fmin - dhat, fmax + dhat, cell)
    vc = np.floor(pv / cell).astype(np.int64)
    for vi in range(len(vertices)):
        c = grid.get((int(vc[vi, 0]), int(vc[vi, 1]), int(vc[vi, 2])))
        if not c:
            continue
        c = np.fromiter(set(c), dtype=np.int64)
        clamped = np.minimum(np.maximum(pv[vi], fmin[c]), fmax[c])
        keep = np.sum((pv[vi] - clamped) ** 2, axis=1) < d2
        c = c[keep]
        if self_contact and len(c):                                 # drop incident (vertex ∈ face) faces
            c = c[~(faces[c] == int(vertices[vi])).any(axis=1)]
        if len(c):
            out[int(vertices[vi])] = np.sort(c)
    return out


def edge_edge_candidates(positions, edges, dhat, *, cell=None, exclude_shared=True):
    """Candidate edge-index pairs ``(i, j)`` with ``i < j`` (superset of edge pairs within ``d̂``),
    excluding edges that share a node. ``edges`` ``(n_e, 2)`` node ids."""
    pos = np.asarray(positions, dtype=float)
    edges = np.asarray(edges, dtype=int)
    if len(edges) == 0:
        return []
    xe = pos[edges]                                        # (n_e, 2, 3)
    emin = xe.min(1); emax = xe.max(1)                     # raw edge AABBs (n_e, 3)
    d2 = dhat * dhat
    pairs = set()

    def _consider(i, j):                                   # i<j, shared-vertex exclusion, AABB prune
        if exclude_shared and (set(edges[i].tolist()) & set(edges[j].tolist())):
            return
        gap = np.maximum(0.0, np.maximum(emin[i] - emax[j], emin[j] - emax[i]))
        if float(gap @ gap) < d2:                          # AABB-to-AABB dist ≤ edge-edge dist → superset
            pairs.add((i, j))

    if _HAS_BVH:                                           # LBVH query per edge (its d̂-expanded box)
        bv = build_bvh(emin, emax, 8)
        cand, ptr = query_csr(emin - dhat, emax + dhat, emin, emax, *bv)
        if _HAS_INCIDENT:                                   # numba pair filter (the Python loop is the bottleneck)
            oi, oj = ee_pairs_nb(cand.astype(np.int64), ptr.astype(np.int64), edges,
                                 emin, emax, dhat, bool(exclude_shared))
            return list(zip(oi.tolist(), oj.tolist()))
        for i in range(len(edges)):                         # numpy/python oracle (numba absent)
            for k in range(ptr[i], ptr[i + 1]):
                j = int(cand[k])
                if j > i:
                    _consider(i, j)
        return sorted(pairs)
    if cell is None:                                       # uniform-grid fallback (no numba)
        cell = float(max((emax - emin).max(), dhat))
    cell = max(cell, 1e-30)
    grid = _grid_bin(emin - dhat, emax + dhat, cell)
    for c in grid.values():
        cs = sorted(set(c))
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                _consider(cs[a], cs[b])
    return sorted(pairs)


# ---------------------------------------------------------------------------- 3D operator
from coupfe.operators.base import Residual, Tangent      # noqa: E402

try:                                                     # numba production kernels (numpy = oracle)
    from coupfe.operators.contact3d_numba import (tri_barrier_eval_nb, closest_faces_nb,
                                                  edge_edge_barrier_eval_nb, min_toi_vf_nb, min_toi_ee_nb)
    _HAS_NUMBA = True
except Exception:                                        # numba absent → numpy per-pair fallback
    _HAS_NUMBA = False


class DeformableBarrierContact3D:
    """Penetration-free deformable–deformable contact in 3D (Stage 5) — the analog of
    :class:`~coupfe.operators.contact.DeformableBarrierContact2D`. Composes the two ported 3D
    primitives over the broad phase: **vertex-face** (closest face per surface vertex) +
    **edge-edge** (all active non-adjacent edge pairs), each a cubic barrier (PSD tangent), and a
    **CCD** ``max_step`` via ACCD (the min time-of-impact over all candidate pairs). ``mass`` (per
    surface vertex) → adaptive ``s=κ+M/gap²`` on the vertex-face term, dynamics only.

    **Friction (``mu > 0`` — ppf/IPC smoothed, semi-implicit):** the 2D smoothed model generalized
    to the 3D tangent plane ``P = I − n⊗n``. Each active vertex-face / edge-edge pair adds a friction
    force ``λ(P·dx)`` (``λ = μ λ_n / max(ε, ‖P·dx‖)``, ``λ_n`` = the barrier normal force) opposing
    the **relative** tangential slip ``dx`` of the contact-point pair since the step start, mapped to
    the 12 stencil DOFs by the kinematic Jacobian (barycentric weights for vertex-face, edge weights
    for edge-edge); symmetric-PSD tangent ``λ BᵀP B``. Stateful: the step-start positions ``_x0``
    are advanced in :meth:`commit`. ``mu=0`` is byte-identical/stateless (frictionless).

    At exact vertex-vertex / vertex-edge coincidences the vertex-face and edge-edge terms can mildly
    double-count — but **ppf does the same** (its barrier uses the unclassified closest distance per
    candidate pair, with no type-classification/dedup and no edge-edge mollifier; the cubic barrier
    avoids the IPC log-barrier's parallel-edge gradient blowup). It is benign (measure-zero
    coincidences; the barrier stays penetration-free, CCD-guaranteed), so we match ppf and do not
    dedup. Edge-edge candidates use the same `i<j` + shared-vertex exclusion ppf does. Faces/edges
    are the contact surface; broad phase is the grid.
    """

    def __init__(self, nodes_ref, vertices, faces, edges, *, dof_per_node=3, comps=None,
                 dhat=0.05, kappa=1.0e2, mass=None, mu=0.0, friction_eps=1.0e-4,
                 owns_edge_pair=None, self_contact=False, friction_kt=None, friction_persistent=False):
        self.X = np.asarray(nodes_ref, dtype=float)
        self.vertices = np.asarray(vertices, dtype=int)
        self.faces = np.asarray(faces, dtype=int)
        self.edges = np.asarray(edges, dtype=int)
        # self_contact: one body's surface against ITSELF — exclude incident vertex-face pairs (the
        # vertex's own faces, gap≡0) and shared-vertex edge-edge pairs. Friction rides unchanged.
        self.self_contact = bool(self_contact)
        self.dpn = int(dof_per_node)
        self.comps = (np.arange(3) if comps is None else np.asarray(comps, dtype=int))
        self.dhat = float(dhat)
        self.kappa = float(kappa)
        self.mu = float(mu)
        self.friction_eps = float(friction_eps)
        # exact-stick return-map friction (vertex-face, numpy path); friction_persistent carries the
        # committed tangential force per vertex across steps + re-pairing (the ε_p analog, finite sliding).
        self.friction_kt = (None if friction_kt is None else float(friction_kt))
        self.friction_persistent = bool(friction_persistent)
        self._ft = np.zeros((len(self.vertices), 3))         # committed friction force per surface vertex
        self._vpos = {int(v): i for i, v in enumerate(self.vertices)}   # vertex id → index into _ft
        # Distributed edge-edge ownership: an optional predicate `node_id -> bool` on the FIRST node
        # of the first edge of each pair, so a pair is processed by exactly one rank (vertex-face is
        # partitioned by passing owned `vertices`; edge-edge needs this since it has no "secondary").
        # Default None = process every pair (serial / single-rank). Applied in _contributions AND
        # max_step so the CCD bound's per-rank min reduces to the correct global min.
        self._owns_edge_pair = owns_edge_pair
        self.massmap = (None if mass is None
                        else {int(v): float(m) for v, m in zip(self.vertices, np.asarray(mass))})
        # smoothed-friction step-start positions (advanced in commit; mu=0 → unused/stateless)
        self._x0 = self.X.copy()

    def _positions_all(self, U):
        U = np.asarray(U, dtype=float)
        return self.X + U.reshape(len(self.X), self.dpn)[:, self.comps]

    def _gd(self, nodes):
        return (np.asarray(nodes, dtype=int)[:, None] * self.dpn + self.comps).ravel()

    def _closest_face(self, pos, v, fis):
        best = None
        for fi in fis:
            f = self.faces[fi]
            w = point_triangle_coeff_unclassified(pos[v], pos[f[0]], pos[f[1]], pos[f[2]])
            gap = float(np.linalg.norm(pos[v] - (w[0]*pos[f[0]] + w[1]*pos[f[1]] + w[2]*pos[f[2]])))
            if best is None or gap < best[0]:
                best = (gap, fi)
        return best                                       # (gap, face_idx) or None

    def _contributions(self, U, want_ft=False):
        """Yield (global dofs (12,), R (12,), K (12,12)) for every active vertex-face + edge-edge.
        Vertex-face runs on the **numba** narrow-phase (broad-phase → batched closest-face + barrier);
        the numpy per-pair path is the bit-identical fallback (and the oracle the numba is gated against),
        and ALSO the path for the EXACT-STICK return-map friction (``friction_kt`` set). ``want_ft`` (commit
        only) additionally returns the per-vertex committed friction force for the persistent state.
        Edge-edge stays numpy (flat block-on-block uses vertex-face only; numba edge-edge is a follow-up)."""
        pos = self._positions_all(U)
        fr = self.mu > 0.0
        out = []
        new_ft = (np.zeros_like(self._ft) if want_ft else None)   # out-of-contact vertices → 0
        cand = vertex_face_candidates(pos, self.vertices, self.faces, self.dhat,
                                      self_contact=self.self_contact)
        if cand and _HAS_NUMBA and self.friction_kt is None:      # numba only for the smoothed friction
            verts = np.fromiter(cand.keys(), dtype=np.int64, count=len(cand))
            cand_ptr = np.zeros(len(verts) + 1, dtype=np.int64)
            flats = []
            for idx in range(len(verts)):
                fl = cand[int(verts[idx])]
                cand_ptr[idx + 1] = cand_ptr[idx] + len(fl)
                flats.append(fl)
            cand_flat = np.concatenate(flats).astype(np.int64)
            best_f, best_g = closest_faces_nb(pos, verts, self.faces, cand_flat, cand_ptr, self.dhat)
            act = best_g < self.dhat
            if np.any(act):
                av = verts[act]
                f0 = self.faces[best_f[act], 0]; f1 = self.faces[best_f[act], 1]; f2 = self.faces[best_f[act], 2]
                if self.massmap is not None:
                    mass = np.array([self.massmap.get(int(v), 0.0) for v in av]); has_mass = True
                else:
                    mass = np.zeros(len(av)); has_mass = False
                if fr:
                    X0 = np.stack([self._x0[av], self._x0[f0], self._x0[f1], self._x0[f2]], axis=1)
                    has_fric = True
                else:
                    X0 = np.zeros((len(av), 4, 3)); has_fric = False
                R, K, active = tri_barrier_eval_nb(pos[av], pos[f0], pos[f1], pos[f2], self.dhat,
                                                   self.kappa, mass, self.mu, self.friction_eps,
                                                   X0, has_mass, has_fric)
                for m in range(len(av)):
                    if active[m]:
                        out.append((self._gd([int(av[m]), int(f0[m]), int(f1[m]), int(f2[m])]), R[m], K[m]))
        elif cand:                                            # numpy path (no numba, OR friction_kt return-map)
            rm = self.friction_kt is not None
            for v, fis in cand.items():
                bf = self._closest_face(pos, v, fis)
                if bf is None or bf[0] >= self.dhat:
                    continue
                f = self.faces[bf[1]]
                m = None if self.massmap is None else np.array([self.massmap.get(v, 0.0)])
                X0 = (self._x0[[v, f[0], f[1], f[2]]][None] if fr else None)
                ftp = (self._ft[self._vpos[int(v)]][None] if (rm and fr and self.friction_persistent)
                       else None)                             # the carried friction force for this vertex
                res = tri_barrier_eval(pos[v][None], pos[f[0]][None], pos[f[1]][None], pos[f[2]][None],
                                       dhat=self.dhat, kappa=self.kappa, mass=m, mu=self.mu,
                                       eps=self.friction_eps, X0=X0, friction_kt=self.friction_kt,
                                       ft_prev=ftp, return_ft=want_ft)
                R, K, a = res[0], res[1], res[2]
                if a[0]:
                    out.append((self._gd([v, f[0], f[1], f[2]]), R[0], K[0]))
                    if want_ft and rm:
                        new_ft[self._vpos[int(v)]] = res[3][0]
        ee = [(i, j) for i, j in edge_edge_candidates(pos, self.edges, self.dhat)
              if self._owns_edge_pair is None or self._owns_edge_pair(int(self.edges[i][0]))]
        use_ee_numba = _HAS_NUMBA and 'edge_edge_barrier_eval_nb' in globals()
        if ee and use_ee_numba:                            # batched numba edge-edge barrier + friction
            ei = self.edges[[i for i, _ in ee]]; ej = self.edges[[j for _, j in ee]]
            P0 = pos[ei[:, 0]]; P1 = pos[ei[:, 1]]; Q0 = pos[ej[:, 0]]; Q1 = pos[ej[:, 1]]
            mass = np.zeros(len(ee)); has_mass = False     # edge-edge has no adaptive mass term
            if fr:
                X0 = np.stack([self._x0[ei[:, 0]], self._x0[ei[:, 1]],
                               self._x0[ej[:, 0]], self._x0[ej[:, 1]]], axis=1)
                has_fric = True
            else:
                X0 = np.zeros((len(ee), 4, 3)); has_fric = False
            R, K, act = edge_edge_barrier_eval_nb(P0, P1, Q0, Q1, self.dhat, self.kappa, mass,
                                                  self.mu, self.friction_eps, X0, has_mass, has_fric)
            for m in range(len(ee)):
                if act[m]:
                    out.append((self._gd([int(ei[m, 0]), int(ei[m, 1]), int(ej[m, 0]), int(ej[m, 1])]),
                                R[m], K[m]))
        elif ee:                                           # numpy fallback (per pair)
            for i, j in ee:
                ei, ej = self.edges[i], self.edges[j]
                X0 = (self._x0[[ei[0], ei[1], ej[0], ej[1]]][None] if fr else None)
                R, K, act = edge_edge_barrier_eval(pos[ei[0]][None], pos[ei[1]][None], pos[ej[0]][None],
                                                   pos[ej[1]][None], dhat=self.dhat, kappa=self.kappa,
                                                   mu=self.mu, eps=self.friction_eps, X0=X0)
                if act[0]:
                    out.append((self._gd([ei[0], ei[1], ej[0], ej[1]]), R[0], K[0]))
        if want_ft:
            return out, new_ft
        return out

    def residual(self, U, state, t, dt) -> Residual:
        rows, vals = [], []
        for gd, R, _K in self._contributions(U):
            rows.append(gd); vals.append(R)
        if not rows:
            return Residual(np.array([], dtype=int), np.array([]))
        return Residual(np.concatenate(rows), np.concatenate(vals))

    def tangent(self, U, state, t, dt) -> Tangent:
        Ri, Ci, Vi = [], [], []
        for gd, _R, K in self._contributions(U):
            Ri.append(np.repeat(gd, 12)); Ci.append(np.tile(gd, 12)); Vi.append(K.ravel())
        if not Ri:
            return Tangent(np.array([], dtype=int), np.array([], dtype=int), np.array([]))
        return Tangent(np.concatenate(Ri), np.concatenate(Ci), np.concatenate(Vi))

    def commit(self, U, state, t, dt):
        if self.mu > 0.0:                                 # advance the friction state
            if self.friction_kt is not None and self.friction_persistent:
                _out, self._ft = self._contributions(U, want_ft=True)   # carry the committed friction force
            self._x0 = self._positions_all(U)             # advance the per-step slip reference
        return state

    def max_step(self, U, dU):
        """CCD: min ACCD time-of-impact over all candidate vertex-face + edge-edge pairs.

        The broad-phase band is **``dhat + 2·reach``** (``reach`` = max per-node displacement over the
        step), NOT ``dhat`` — CCD must catch a face a node could *sweep* through, even if it is farther
        than ``dhat`` at the current config (each of the two primitives can close up to ``reach``). The
        barrier broad-phase uses ``dhat`` (current-config); only this swept query needs the larger band.
        (A loose grid used to mask this; the tightened AABB prune makes the band explicit.)"""
        pos = self._positions_all(U)
        dU = np.asarray(dU, dtype=float)
        disp = dU.reshape(len(self.X), self.dpn)[:, self.comps]
        pos1 = pos + disp
        reach = float(np.sqrt((disp * disp).sum(1)).max()) if len(disp) else 0.0
        band = self.dhat + 2.0 * reach
        toi = 1.0
        vf_v, vf_f = [], []                                # (vertex, face) candidate pairs
        for v, fis in vertex_face_candidates(pos, self.vertices, self.faces, band,
                                             self_contact=self.self_contact).items():
            for fi in fis:
                vf_v.append(v); vf_f.append(fi)
        ee = [(i, j) for i, j in edge_edge_candidates(pos, self.edges, band)
              if self._owns_edge_pair is None or self._owns_edge_pair(int(self.edges[i][0]))]
        if _HAS_NUMBA:                                      # batched numba ACCD over the candidate pairs
            if vf_v:
                vv = np.asarray(vf_v, dtype=np.int64); ff = self.faces[np.asarray(vf_f, dtype=np.int64)]
                toi = min(toi, float(min_toi_vf_nb(pos[vv], pos1[vv], pos[ff[:, 0]], pos1[ff[:, 0]],
                                                   pos[ff[:, 1]], pos1[ff[:, 1]], pos[ff[:, 2]], pos1[ff[:, 2]],
                                                   0.0, 1.0, 0.2, 100)))
            if ee:
                ei = self.edges[[i for i, _ in ee]]; ej = self.edges[[j for _, j in ee]]
                toi = min(toi, float(min_toi_ee_nb(pos[ei[:, 0]], pos1[ei[:, 0]], pos[ei[:, 1]], pos1[ei[:, 1]],
                                                   pos[ej[:, 0]], pos1[ej[:, 0]], pos[ej[:, 1]], pos1[ej[:, 1]],
                                                   0.0, 1.0, 0.2, 100)))
        else:                                              # numpy fallback (per pair)
            for v, fi in zip(vf_v, vf_f):
                f = self.faces[fi]
                toi = min(toi, point_triangle_toi(pos[v], pos1[v], pos[f[0]], pos[f[1]], pos[f[2]],
                                                  pos1[f[0]], pos1[f[1]], pos1[f[2]]))
            for i, j in ee:
                ei, ej = self.edges[i], self.edges[j]
                toi = min(toi, edge_edge_toi(pos[ei[0]], pos[ei[1]], pos[ej[0]], pos[ej[1]],
                                             pos1[ei[0]], pos1[ei[1]], pos1[ej[0]], pos1[ej[1]]))
        return max(toi, 0.0)
