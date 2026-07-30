"""The model-setup pipeline — CoupFE's declarative front door.

A thin layer over the operator contract: collect element groups (materials over mesh
regions), contact operators, and constraints; build the Dirichlet dict from named/selected
node sets; hand the operator list to the composing driver (`solve_increments`). No new
physics — just the concise, AI-targetable surface that turns a problem statement into a
solve. This module supplies only neutral convenience operations; domain geometry,
format adapters, BC/load definitions, and application metadata live in the
application repositories.

    m = Model.structured(4, 4, 1.0, 1.0)
    m.refine(levels=1)
    m.material("block", NeoHookean(G=1, K=10))
    m.fix("left", x=0)
    m.prescribe("top", y=-0.05)
    m.contact("bottom", HalfSpace([0, -0.01], [0, 1]), k=1e4)
    res = m.solve(steps=4)
    res.converged, res.U, res.position("bottom")
"""

from __future__ import annotations

import numpy as np

from coupfe.assembly.assemble import assemble_residual, solve_increments
from coupfe.mesh.view import KernelMeshView
from coupfe.mesh.refine import uniform_refine_quad
from coupfe.operators.contact import RigidContact

_AXIS = {"x": 0, "y": 1, "z": 2}
_BBOX = {"left": (0, "min"), "right": (0, "max"),
         "bottom": (1, "min"), "top": (1, "max"),
         "front": (2, "min"), "back": (2, "max")}


def _structured_quad_mesh(nx, ny, Lx=1.0, Ly=1.0):
    """Nodes (row-major, lexicographic) + CCW Quad4 connectivity for an nx×ny grid."""
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs], dtype=float)
    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elems.append([n0, n0 + 1, n0 + 1 + (nx + 1), n0 + (nx + 1)])  # CCW
    return nodes, np.array(elems, dtype=int)


class Result:
    """Outcome of a solve: the DOF vector plus convenience accessors."""

    def __init__(self, U, converged, iters, model):
        self.U = U
        self.converged = bool(converged)
        self.iters = int(iters)
        self._m = model

    def displacement(self):
        """(n_node, dof_per_node) reshaped DOFs."""
        return self.U.reshape(-1, self._m.view.dof_per_node)

    def position(self, selector=None):
        """Deformed coordinates (reference + spatial displacement), optionally at a set."""
        ndim = self._m.view.nodes.shape[1]
        pos = self._m.view.nodes + self.displacement()[:, :ndim]
        return pos if selector is None else pos[self._m._resolve(selector)]


class Model:
    """Declarative problem: mesh + materials + BCs + contact → solve."""

    def __init__(self, view: KernelMeshView):
        self.view = view
        self._materials = []          # (name, material, elements, comps)
        self._contacts = []           # (selector, obstacle, k, comps)
        self._dirichlet = {}          # gdof -> prescribed value

    # --- construction -------------------------------------------------------
    @classmethod
    def structured(cls, nx, ny, Lx=1.0, Ly=1.0):
        nodes, elems = _structured_quad_mesh(nx, ny, Lx, Ly)
        return cls(KernelMeshView(nodes, elems, dof_per_node=2))

    @classmethod
    def from_view(cls, view):
        return cls(view)

    def refine(self, levels=1):
        for _ in range(int(levels)):
            self.view = uniform_refine_quad(self.view)
        return self

    # --- problem definition -------------------------------------------------
    def material(self, name, material, elements="all", comps=(0, 1)):
        self._materials.append((name, material, elements, tuple(comps)))
        return self

    def fix(self, selector, **comps):
        """Hold DOFs at a value (default 0) on a node selection, e.g. ``fix("left", x=0)``."""
        self._set_bc(selector, comps)
        return self

    def prescribe(self, selector, **comps):
        """Drive DOFs to a (ramped) value, e.g. ``prescribe("top", y=-0.05)``."""
        self._set_bc(selector, comps)
        return self

    def contact(self, selector, obstacle, k=1.0e4, comps=(0, 1), mu=0.0, k_t=None):
        """Rigid-obstacle penalty contact on a node selection; ``mu>0`` adds Coulomb friction."""
        self._contacts.append((selector, obstacle, float(k), tuple(comps), float(mu), k_t))
        return self

    # --- resolution helpers -------------------------------------------------
    def _resolve(self, selector):
        """Selector → node-index array. Name in node_sets, bbox name, predicate, or array."""
        if isinstance(selector, str):
            if selector in self.view.node_sets:
                return np.asarray(self.view.node_sets[selector], dtype=int)
            return self._bbox_nodes(selector)
        if callable(selector):
            X = self.view.nodes
            return np.array([i for i in range(len(X)) if selector(X[i])], dtype=int)
        return np.asarray(selector, dtype=int)

    def _bbox_nodes(self, name, tol=1e-9):
        if name not in _BBOX:
            raise KeyError(f"unknown selector {name!r} (not a node set or bbox face)")
        axis, side = _BBOX[name]
        X = self.view.nodes
        if axis >= X.shape[1]:
            raise KeyError(f"selector {name!r} needs a {axis + 1}D mesh")
        target = X[:, axis].min() if side == "min" else X[:, axis].max()
        return np.nonzero(np.abs(X[:, axis] - target) < tol)[0]

    def _set_bc(self, selector, comps):
        dpn = self.view.dof_per_node
        nodes = self._resolve(selector)
        for axis_name, value in comps.items():
            c = _AXIS[axis_name]
            for n in nodes:
                self._dirichlet[int(n) * dpn + c] = float(value)

    # --- assembly + solve ---------------------------------------------------
    def _operators(self):
        ops = []
        for _name, mat, elements, comps in self._materials:
            elem_set = None if elements == "all" else elements
            ops.append(mat.element_group(self.view, elem_set, comps))
        for selector, obstacle, k, comps, mu, k_t in self._contacts:
            ops.append(RigidContact(self.view.nodes, self._resolve(selector), obstacle,
                                    dof_per_node=self.view.dof_per_node, comps=comps,
                                    k=k, mu=mu, k_t=k_t))
        return ops

    def solve(self, steps=1, *, rtol=1e-5, **newton_kw):
        if not self._materials:
            raise ValueError("Model has no material — add one with .material(...)")
        ops = self._operators()
        ndof = self.view.ndof
        U, nit = solve_increments(ops, np.zeros(ndof), ndof, dict(self._dirichlet),
                                  n_steps=steps, **newton_kw)
        R, _ = assemble_residual(ops, U, None, 1.0, 1.0, ndof)
        free = np.ones(ndof, dtype=bool)
        if self._dirichlet:
            free[np.array(sorted(self._dirichlet), dtype=int)] = False
        converged = float(np.linalg.norm(R[free])) < rtol
        return Result(U, converged, nit, self)
