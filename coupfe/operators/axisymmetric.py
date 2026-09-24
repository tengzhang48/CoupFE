"""Axisymmetric (r, z) cavity, contact and boundary helpers as CoupFE operators.

Coordinates are ``(r, z)``: component 0 is the radius ``r >= 0`` and component 1
the axial coordinate; fields are independent of the angle. Every integral
carries the ring factor ``2*pi*r``, so residuals and reactions are total
three-dimensional forces. Axisymmetric *elements* come from ``coupfe.codegen``
with the ``*_axi`` element configurations (native generated kernels run by
:class:`~coupfe.operators.element_group.ElementGroup`); this module supplies the
boundary operators that act on those meshes:

* :class:`AxisymmetricCavity` -- the follower pressure of a closed cavity,
  driven by one global pressure unknown that is either prescribed (pressure
  control) or closed by a :class:`FluidLaw` (volume control);
* :class:`AxisymmetricContact` -- frictionless node-to-segment contact with a
  ring-area-weighted penalty and optional augmented-Lagrangian multipliers;
* :func:`boundary_edges` and :func:`ring_areas` -- oriented boundary edges and
  consistent ring areas of Quad4/Quad8/Tri3/Tri6 meshes.

As elsewhere in CoupFE, the residual is the source of truth: tangents are
complex-step derivatives of it with every discrete choice (active contact set,
paired segment, clamped projection) frozen at the real iterate. The residual
follows the CoupFE sign convention ``R = f_internal - f_external``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coupfe.operators.base import Residual, Tangent
from coupfe.operators.contact_search import candidate_pairs

TWO_PI = 2.0 * np.pi
_H = 1.0e-30  # complex-step size, as in coupfe.operators.base.complex_step_tangent

# Local edges run counterclockwise around a counterclockwise element as
# (start, end) or (start, end, mid); the body lies to the left of each edge.
# Node order follows Gmsh (corners first, then mid-side nodes).
_EDGES = {
    "quad4": ((0, 1), (1, 2), (2, 3), (3, 0)),
    "quad8": ((0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)),
    "tri3": ((0, 1), (1, 2), (2, 0)),
    "tri6": ((0, 1, 3), (1, 2, 4), (2, 0, 5)),
}


def boundary_edges(elems, element):
    """Boundary edges of a mesh block, oriented counterclockwise around it.

    Returns ``(n_edges, 2)`` or ``(n_edges, 3)`` node arrays ordered as
    ``(start, end[, mid])``. For counterclockwise elements the body lies to the
    left of each returned edge, so the outward normal is the edge direction
    rotated by -90 degrees -- the convention used by :class:`AxisymmetricContact`
    and :class:`AxisymmetricCavity`.
    """
    elems = np.asarray(elems, dtype=int)
    local = _EDGES[element]
    count, first = {}, {}
    for cell in elems:
        for loc in local:
            nodes = tuple(int(cell[i]) for i in loc)
            key = (min(nodes[0], nodes[1]), max(nodes[0], nodes[1]))
            count[key] = count.get(key, 0) + 1
            first.setdefault(key, nodes)
    out = [first[k] for k, c in count.items() if c == 1]
    return np.asarray(out, dtype=int).reshape(-1, len(local[0]))


def _edge_rule(n_edge_nodes):
    g, w = np.polynomial.legendre.leggauss(3 if n_edge_nodes == 3 else 2)
    if n_edge_nodes == 2:
        N = np.column_stack([0.5 * (1 - g), 0.5 * (1 + g)])
        dN = np.column_stack([-0.5 * np.ones_like(g), 0.5 * np.ones_like(g)])
    elif n_edge_nodes == 3:  # (start, end, mid)
        N = np.column_stack([0.5 * g * (g - 1), 0.5 * g * (g + 1), 1 - g * g])
        dN = np.column_stack([g - 0.5, g + 0.5, -2 * g])
    else:
        raise ValueError("edges must have 2 or 3 nodes")
    return N, dN, w


def ring_areas(nodes, edges, n_nodes=None):
    """Consistent ring areas ``A_i = sum over edges of int 2*pi*r N_i ds``.

    ``edges`` are Line2 ``(start, end)`` or Line3 ``(start, end, mid)`` boundary
    edges in reference coordinates. The result, indexed by node, is the
    axisymmetric analogue of a tributary length; an axis node receives the
    small positive area of its adjacent ring segment.
    """
    X = np.asarray(nodes, dtype=float)
    edges = np.asarray(edges, dtype=int)
    N, dN, w = _edge_rule(edges.shape[1])
    Xe = X[edges]                                         # (ne, nen, 2)
    r = np.einsum("ga,ea->eg", N, Xe[..., 0])
    dxds = np.einsum("ga,eai->egi", dN, Xe)
    ds = np.sqrt((dxds ** 2).sum(-1))
    contrib = np.einsum("g,eg,eg,ga->ea", w, TWO_PI * r, ds, N)
    out = np.zeros(len(X) if n_nodes is None else int(n_nodes))
    np.add.at(out, edges, contrib)
    return out


# ---------------------------------------------------------------------------
# Pressure / fluid cavity
# ---------------------------------------------------------------------------

@dataclass
class FluidLaw:
    """Barotropic fill of a closed cavity: ``V(p) = V_ref exp(-beta (p - p_ref))``.

    ``beta`` is the compressibility ``(1/rho) d rho/dp``; ``beta = 0`` is an
    incompressible fill. The law conserves the fluid mass present at the
    reference state ``(V_ref, p_ref)``.
    """

    reference_volume: float
    reference_pressure: float
    compressibility: float = 0.0

    def volume(self, p):
        return self.reference_volume * np.exp(-self.compressibility * (p - self.reference_pressure))

    def dvolume(self, p):
        return -self.compressibility * self.volume(p)


class AxisymmetricCavity:
    """Follower pressure of a closed axisymmetric cavity with a pressure unknown.

    ``edges`` are the cavity's wall edges (Line2 or Line3, ``(start, end[, mid])``)
    ordered counterclockwise around the *cavity* region; together with the
    symmetry axis they close it. The cavity volume is

        V = sum over wall edges of int pi r**2 dz        (Green's theorem),

    where axis segments contribute nothing. The global unknown
    ``U[pressure_dof]`` is the cavity pressure. The operator adds the potential
    ``-p V(u)`` (so the walls receive the consistent follower load ``p dV/du``)
    and, when a :class:`FluidLaw` is set, the mass-conservation equation
    ``scale * (V_fluid(p) - V(u)) = 0`` on the pressure row. Without a fluid law
    the pressure must be prescribed by the caller (pressure control).

    ``scale`` converts the volume equation into force-like units for residual
    norms (for example ``1e6`` N/m^3 for kPa-level pressures on mm-scale
    cavities). It does not change the solution.
    """

    def __init__(self, nodes, edges, pressure_dof, *, dof_per_node=2, comps=(0, 1),
                 fluid=None, scale=1.0):
        self.X = np.asarray(nodes, dtype=float)
        self.edges = np.asarray(edges, dtype=int)
        if self.edges.ndim != 2 or self.edges.shape[1] not in (2, 3):
            raise ValueError("cavity edges must be (n, 2) or (n, 3) node arrays")
        self.pdof = int(pressure_dof)
        self.dpn = int(dof_per_node)
        self.comps = np.asarray(comps, dtype=int)
        self.fluid = fluid
        self.scale = float(scale)
        self._N, self._dN, self._w = _edge_rule(self.edges.shape[1])
        self.gd = (self.edges[:, :, None] * self.dpn + self.comps[None, None, :]).reshape(len(self.edges), -1)
        V0 = self.volume(np.zeros(int(self.gd.max()) + 1))
        if not V0 > 0.0:
            raise ValueError("cavity volume is not positive; order the wall edges "
                             "counterclockwise around the cavity")
        self.reference_volume = V0

    def _positions(self, U):
        U = np.asarray(U)
        u = U[self.gd].reshape(len(self.edges), -1, 2)
        return self.X[self.edges] + u

    def _edge_volume(self, x):                     # x: (ne, nen, 2), real or complex
        r = np.einsum("ga,ea->eg", self._N, x[..., 0])
        dz = np.einsum("ga,ea->eg", self._dN, x[..., 1])
        return np.einsum("g,eg->e", self._w, np.pi * r * r * dz)

    def _edge_gradient(self, x):                   # dV_e/dx_e, (ne, 2*nen)
        r = np.einsum("ga,ea->eg", self._N, x[..., 0])
        dz = np.einsum("ga,ea->eg", self._dN, x[..., 1])
        g_r = np.einsum("g,eg,ga->ea", self._w, TWO_PI * r * dz, self._N)
        g_z = np.einsum("g,eg,ga->ea", self._w, np.pi * r * r, self._dN)
        return np.stack([g_r, g_z], axis=-1).reshape(len(x), -1)

    def volume(self, U):
        return float(self._edge_volume(self._positions(U)).sum())

    def volume_gradient(self, U):
        """``(gdofs, dV/du)`` in COO form."""
        return self.gd.ravel(), self._edge_gradient(self._positions(U)).ravel()

    def set_fluid(self, fluid):
        """Close the cavity with ``fluid`` (a :class:`FluidLaw`), or open it with ``None``."""
        self.fluid = fluid

    def residual(self, U, state, t, dt) -> Residual:
        p = float(np.asarray(U)[self.pdof])
        rows, g = self.volume_gradient(U)
        rows, vals = [rows], [-p * g]
        if self.fluid is not None:
            rows.append(np.array([self.pdof]))
            vals.append(np.array([self.scale * (self.fluid.volume(p) - self.volume(U))]))
        return Residual(np.concatenate(rows), np.concatenate(vals))

    def tangent(self, U, state, t, dt) -> Tangent:
        p = float(np.asarray(U)[self.pdof])
        x = self._positions(U)
        n = x.shape[1] * 2
        H = np.empty((len(x), n, n))
        for j in range(n):
            xc = x.astype(complex)
            xc[:, j // 2, j % 2] += 1j * _H
            H[:, :, j] = self._edge_gradient(xc).imag / _H
        g = self._edge_gradient(x)
        gd = self.gd
        rows = [np.broadcast_to(gd[:, :, None], H.shape).ravel(), gd.ravel()]
        cols = [np.broadcast_to(gd[:, None, :], H.shape).ravel(), np.full(gd.size, self.pdof)]
        vals = [(-p * H).ravel(), -g.ravel()]
        if self.fluid is not None:
            rows += [np.full(gd.size, self.pdof), np.array([self.pdof])]
            cols += [gd.ravel(), np.array([self.pdof])]
            vals += [-self.scale * g.ravel(), np.array([self.scale * self.fluid.dvolume(p)])]
        return Tangent(np.concatenate(rows), np.concatenate(cols), np.concatenate(vals))

    def commit(self, U, state, t, dt):
        return state


# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------

def _segments(edges):
    """Straight contact segments (start, end) from Line2/Line3 boundary edges."""
    edges = np.asarray(edges, dtype=int)
    if edges.shape[1] == 2:
        return edges.copy()
    return np.concatenate([edges[:, [0, 2]], edges[:, [2, 1]]])


class AxisymmetricContact:
    """Frictionless axisymmetric node-to-segment contact.

    Secondary nodes are the nodes of ``secondary_edges``; each carries its ring
    area ``A_s`` (:func:`ring_areas`). The primary surface is the polyline through
    the nodes of ``primary_edges`` (quadratic edges are split at their mid
    nodes), oriented counterclockwise around the primary body so the outward
    normal is the segment direction rotated by -90 degrees. For each secondary
    node the closest segment, the clamped projection ``xi`` and the active state
    are chosen at the real iterate; the signed normal gap is
    ``g = (x_s - x_m) . n`` (negative = penetration).

    The contact force on an active node is ``f_s = A_s (lambda_s - eps g)`` along
    ``n`` (an augmented-Lagrangian contact pressure ``lambda_s`` plus the penalty
    ``eps`` in Pa/m), shared equal-and-opposite with the two primary nodes by
    ``(1 - xi, xi)``. With ``lambda = 0`` this is the energy
    ``sum 1/2 eps A_s g**2`` of a ring-weighted penalty. :meth:`augment` performs
    the Uzawa update ``lambda <- max(0, lambda - eps g)``; the driver decides
    when to call it and accepts it like committed state.

    A rigid primary surface is modeled by primary nodes whose displacements are
    all prescribed. Two passes (A on B and B on A) give symmetric treatment of
    two deformable bodies.

    Pairing: by default the closest segment and the clamped projection are
    re-selected at every real iterate. On a deformed polyline a node near a
    primary vertex can then alternate between the two adjacent segments, whose
    normals differ, and Newton cycles. :meth:`freeze` fixes each node's segment
    for the following solves; frozen nodes project onto the segment line
    without clamping, which keeps the residual smooth. Calling :meth:`freeze`
    again re-pairs with hysteresis: a node keeps its segment while its
    projection stays within ``switch_margin`` (a fraction of the segment
    length) of it, and the return value counts the nodes whose segment
    changed, so a driver can re-solve until the pairing is stable.
    :meth:`release` restores per-iterate pairing.

    Smoothing: with ``smoothing = delta > 0`` the contact pressure is
    ``eps * r(lambda/eps - g)`` with the C1 ramp ``r(x) = 0`` for
    ``x <= -delta``, ``(x + delta)**2 / (4 delta)`` for ``|x| < delta`` and ``x``
    for ``x >= delta``. It equals the unsmoothed pressure once a node
    penetrates by more than ``delta`` and replaces the jump in stiffness at the
    edge of the contact zone by a transition of width ``2 delta``, where
    semismooth Newton otherwise cycles between active sets. ``delta = 0`` (the
    default) is the unsmoothed law.
    """

    def __init__(self, nodes_ref, secondary_edges, primary_edges, *, penalty,
                 dof_per_node=2, comps=(0, 1), search_band=None, n_nodes=None,
                 switch_margin=0.05, smoothing=0.0):
        self.X = np.asarray(nodes_ref, dtype=float)
        sec_edges = np.asarray(secondary_edges, dtype=int)
        self.secondary = np.unique(sec_edges)
        areas = ring_areas(self.X, sec_edges, n_nodes=len(self.X))
        self.area = areas[self.secondary]
        self.segments = _segments(primary_edges)
        self.eps = float(penalty)
        if self.eps <= 0.0:
            raise ValueError("penalty must be positive")
        self.dpn = int(dof_per_node)
        self.comps = np.asarray(comps, dtype=int)
        self.search_band = search_band
        self.lam = np.zeros(len(self.secondary))
        self.switch_margin = float(switch_margin)
        self.smoothing = float(smoothing)
        if self.smoothing < 0.0:
            raise ValueError("smoothing must be nonnegative")
        self.frozen = None                  # segment per secondary node (-1: unpaired)

    def _ramp(self, x):
        """The pressure ramp in length units (complex-safe; branch by the real part)."""
        d = self.smoothing
        if d == 0.0:
            return np.where(np.real(x) > 0.0, x, 0.0 * x)
        xr = np.real(x)
        return np.where(xr >= d, x, np.where(xr > -d, (x + d) ** 2 / (4.0 * d), 0.0 * x))

    def _positions_all(self, U):
        U = np.asarray(U, dtype=float)
        n = len(self.X)
        return self.X + U[: n * self.dpn].reshape(n, self.dpn)[:, self.comps]

    def _pairs(self, U):
        """Pairing at the real iterate: (secondary index, segment, xi, gap, clamp)."""
        if self.frozen is None:
            return self._closest_pairs(U)
        k = np.nonzero(self.frozen >= 0)[0]
        sid = self.frozen[k]
        pos = self._positions_all(U)
        xa, xb = pos[self.segments[sid, 0]], pos[self.segments[sid, 1]]
        xs = pos[self.secondary[k]]
        e = xb - xa
        L2 = (e * e).sum(1)
        xi = ((xs - xa) * e).sum(1) / L2
        n = np.stack([e[:, 1], -e[:, 0]], axis=1) / np.sqrt(L2)[:, None]
        g = ((xs - xa - xi[:, None] * e) * n).sum(1)
        return k, sid, xi, g, np.zeros(len(k), dtype=int)

    def freeze(self, U):
        """Fix the segment of every secondary node at ``U``; returns how many changed.

        Starting from the closest segments, a node that was already frozen keeps
        its previous segment while its unclamped projection stays within
        ``[-switch_margin, 1 + switch_margin]``.
        """
        idx, sid, _, _, _ = self._closest_pairs(U)
        seg = np.full(len(self.secondary), -1)
        seg[idx] = sid
        previous = self.frozen
        if previous is not None:
            keep = np.nonzero((previous >= 0) & (seg >= 0))[0]
            if len(keep):
                pos = self._positions_all(U)
                p = previous[keep]
                xa, xb = pos[self.segments[p, 0]], pos[self.segments[p, 1]]
                e = xb - xa
                xi = ((pos[self.secondary[keep]] - xa) * e).sum(1) / (e * e).sum(1)
                m = self.switch_margin
                stay = (xi >= -m) & (xi <= 1.0 + m)
                seg[keep[stay]] = p[stay]
        self.frozen = seg
        return int(len(seg)) if previous is None else int((seg != previous).sum())

    def release(self):
        """Return to per-iterate closest-segment pairing."""
        self.frozen = None

    def _closest_pairs(self, U):
        """Closest segment and clamped projection for every nearby secondary node."""
        pos = self._positions_all(U)
        seg = self.segments
        xa, xb = pos[seg[:, 0]], pos[seg[:, 1]]
        if self.search_band is None:
            band = float(np.sqrt(((xb - xa) ** 2).sum(1)).max())
        else:
            band = float(self.search_band)
        cands = candidate_pairs(pos, self.secondary, seg, band, exclude_incident=True)
        idx, sid, xis, gaps, clamp = [], [], [], [], []
        for k, s in enumerate(self.secondary):
            ce = cands.get(int(s))
            if ce is None:
                continue
            xs = pos[s]
            e = xb[ce] - xa[ce]
            L2 = (e * e).sum(1)
            xi = ((xs - xa[ce]) * e).sum(1) / L2
            xic = np.clip(xi, 0.0, 1.0)
            xm = xa[ce] + xic[:, None] * e
            dist = np.sqrt(((xs - xm) ** 2).sum(1))
            j = int(np.argmin(dist))
            n = np.array([e[j, 1], -e[j, 0]]) / np.sqrt(L2[j])
            idx.append(k)
            sid.append(int(ce[j]))
            xis.append(float(xic[j]))
            gaps.append(float((xs - xm[j]) @ n))
            clamp.append(0 if 0.0 < xi[j] < 1.0 else (1 if xi[j] <= 0.0 else 2))
        return (np.asarray(idx, dtype=int), np.asarray(sid, dtype=int),
                np.asarray(xis), np.asarray(gaps), np.asarray(clamp, dtype=int))

    def _pair_force(self, x6, lam, area, clamp):
        """Residual on (s, a, b) coordinates for active pairs (complex-safe)."""
        xs, xa, xb = x6[:, 0:2], x6[:, 2:4], x6[:, 4:6]
        e = xb - xa
        L = np.sqrt((e * e).sum(1))
        n = np.stack([e[:, 1], -e[:, 0]], axis=1) / L[:, None]
        xi = ((xs - xa) * e).sum(1) / (L * L)
        xi = np.where(clamp == 1, 0.0, np.where(clamp == 2, 1.0, xi))
        g = ((xs - xa - xi[:, None] * e) * n).sum(1)
        f = area * self.eps * self._ramp(lam / self.eps - g)
        return -f[:, None] * np.concatenate([n, -(1 - xi)[:, None] * n, -xi[:, None] * n], axis=1)

    def _active_data(self, U):
        idx, sid, xi, g, clamp = self._pairs(U)
        if len(idx) == 0:
            return None
        lam, area = self.lam[idx], self.area[idx]
        active = lam / self.eps - g > -self.smoothing if self.smoothing else lam - self.eps * g > 0.0
        if not active.any():
            return None
        idx, sid, clamp, lam, area = idx[active], sid[active], clamp[active], lam[active], area[active]
        nodes = np.stack([self.secondary[idx], self.segments[sid, 0], self.segments[sid, 1]], axis=1)
        gd = (nodes[:, :, None] * self.dpn + self.comps[None, None, :]).reshape(len(idx), 6)
        x6 = self._positions_all(U)[nodes].reshape(len(idx), 6)
        return gd, x6, lam, area, clamp

    def residual(self, U, state, t, dt) -> Residual:
        data = self._active_data(U)
        if data is None:
            return Residual(np.array([], dtype=int), np.array([]))
        gd, x6, lam, area, clamp = data
        return Residual(gd.ravel(), self._pair_force(x6, lam, area, clamp).ravel())

    def tangent(self, U, state, t, dt) -> Tangent:
        data = self._active_data(U)
        if data is None:
            return Tangent(np.array([], dtype=int), np.array([], dtype=int), np.array([]))
        gd, x6, lam, area, clamp = data
        K = np.empty((len(gd), 6, 6))
        for j in range(6):
            xc = x6.astype(complex)
            xc[:, j] += 1j * _H
            K[:, :, j] = self._pair_force(xc, lam, area, clamp).imag / _H
        rows = np.broadcast_to(gd[:, :, None], K.shape)
        cols = np.broadcast_to(gd[:, None, :], K.shape)
        return Tangent(rows.ravel(), cols.ravel(), K.ravel())

    def commit(self, U, state, t, dt):
        return state  # multipliers change only through augment()

    def gaps(self, U):
        """Signed gaps of the paired secondary nodes: (secondary node ids, gaps)."""
        idx, _, _, g, _ = self._pairs(U)
        return self.secondary[idx], g

    def contact_pressure(self, U):
        """Normal contact pressure ``eps r(lambda/eps - g)`` per paired secondary node."""
        idx, _, _, g, _ = self._pairs(U)
        return self.secondary[idx], self.eps * self._ramp(self.lam[idx] / self.eps - g)

    def augment(self, U):
        """Uzawa multiplier update; returns the largest penetration before it."""
        idx, _, _, g, _ = self._pairs(U)
        lam = np.zeros_like(self.lam)
        if len(idx):
            lam[idx] = self.eps * self._ramp(self.lam[idx] / self.eps - g)
        self.lam = lam
        return float(max(0.0, -g.min())) if len(g) else 0.0
