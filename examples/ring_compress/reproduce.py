"""Reproduce Abaqus ring_compress.inp in CoupFE.

The Abaqus model is a 2D plane-strain neo-Hookean ring (inner radius 8, outer
radius 10) compressed between two rigid flat plates, then the top plate is slid
horizontally.

CoupFE model choices:
- 2D F-bar Quad4 element (``neo_hookean_q4.for``) for the ring.
- ``RigidContact`` with ``HalfSpace`` obstacles for the rigid plates.  A penalty
  stiffness ``k=1e4`` is used: small enough that the return-map friction has a
  meaningful cap, large enough that penetration stays below ~5e-6.
- Return-map Coulomb friction (``mu=0.5``, ``k_t=1e4``) on both plates.
- **Adaptive load stepping** that mimics Abaqus's automatic time incrementation:
  the step size is cut when Newton does not converge and grown when it does.
  This is the key to avoiding the asymmetric collapse mode that a fixed large
  increment follows.

The sliding step is reproduced by prescribing the Abaqus top-plate horizontal
shift (2 units) to the ring nodes that are in contact with the top plate at the
end of compression.  CoupFE's ``RigidContact`` friction tracks stick anchors in
the spatial frame, so a moving rigid obstacle does not by itself generate
tangential friction; prescribing the displacement is the stick-limited
equivalent.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from coupfe import solve_increments
from coupfe.mesh import KernelMeshView
from coupfe.operators.contact import RigidContact, HalfSpace
from coupfe.operators.element_group import ElementGroup
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INP = Path(os.environ.get("COUPFE_RING_INP", _REPO_ROOT / "ring_compress.inp")).expanduser()
_Q4_FOR = "coupfe/runtime/elements/neo_hookean_q4.for"

# Abaqus neo-Hookean: C10=1.0, D1=0.1  =>  G=mu=2*C10=2,  K=2/D1=20
G_RING = 2.0
K_RING = 20.0

# Penalty contact parameters.
PENALTY = 1.0e4
MU = 0.5
FRICTION_KT = 1.0e4

# Plate geometry from Abaqus (analytical surfaces span [-14, 14]).
PLATE_Y0_TOP = 10.0
PLATE_Y0_BOT = -10.0
PLATE_HALF_WIDTH = 14.0

# Adaptive stepping controls.
INITIAL_STEP = 0.05          # initial |plate-y| / |slide-x| increment
STEP_GROWTH = 1.2            # multiplier on successful steps
STEP_CUT = 0.5               # multiplier on failed steps
MAX_NEWTON = 20              # convergence threshold for step acceptance
MIN_STEP = 1.0e-6            # abort if step shrinks below this

# Tolerance used to identify nodes that are in contact with the top plate at the
# end of compression (for the sliding-step Dirichlet set).
TOP_CONTACT_TOL = 1.0e-4


def _parse_ring(inp_path: str):
    """Extract ring nodes and Quad4 connectivity from the Abaqus .inp file."""
    text = Path(inp_path).read_text()
    lines = text.splitlines()
    nodes = []
    elems = []
    in_ring_nodes = False
    in_ring_elems = False
    for line in lines:
        s = line.strip()
        if s.startswith("*Part, name=Ring"):
            continue
        if s.startswith("*Node") and len(nodes) == 0:
            in_ring_nodes = True
            in_ring_elems = False
            continue
        if s.startswith("*Element") and len(elems) == 0:
            in_ring_nodes = False
            in_ring_elems = True
            continue
        if s.startswith("*") and (in_ring_nodes or in_ring_elems):
            in_ring_nodes = False
            in_ring_elems = False
            continue
        if in_ring_nodes:
            parts = [p.strip() for p in s.split(",")]
            if len(parts) >= 3:
                nodes.append([float(parts[1]), float(parts[2])])
        elif in_ring_elems:
            parts = [p.strip() for p in s.split(",")]
            if len(parts) >= 5:
                elems.append([int(p) for p in parts[1:5]])
    return np.array(nodes, float), np.array(elems, int) - 1  # 1-based -> 0-based


def build_model():
    ring_nodes, ring_elems = _parse_ring(_INP)

    # Rigid plates as analytical half-spaces.  The body is on the +normal side.
    # Top plate: normal points downward; ring is below.
    top_plate = HalfSpace([0.0, PLATE_Y0_TOP], [0.0, -1.0])
    # Bottom plate: normal points upward; ring is above.
    bot_plate = HalfSpace([0.0, PLATE_Y0_BOT], [0.0, 1.0])

    view = KernelMeshView(ring_nodes, ring_elems, dof_per_node=2)
    ndof = view.ndof

    elem = CompiledElement(
        build_element_kernel(_Q4_FOR, "ring_compress_q4"),
        props=(G_RING, K_RING),
        dof_per_node=2,
        n_svars=0,
        mcrd=2,
        n_elem=len(ring_elems),
    )
    grp = ElementGroup.from_view(view, elem, comps=(0, 1))

    # Use only ring outer-surface nodes as contact candidates against the plates.
    r_node = np.hypot(ring_nodes[:, 0], ring_nodes[:, 1])
    ring_nodes_idx = np.nonzero(np.abs(r_node - 10.0) < 1e-6)[0]

    contact_top = RigidContact(
        ring_nodes, ring_nodes_idx, top_plate,
        dof_per_node=2, comps=(0, 1), k=PENALTY, mu=MU, k_t=FRICTION_KT,
    )
    contact_bot = RigidContact(
        ring_nodes, ring_nodes_idx, bot_plate,
        dof_per_node=2, comps=(0, 1), k=PENALTY, mu=MU, k_t=FRICTION_KT,
    )

    operators = [grp, contact_bot, contact_top]

    # Dirichlet: anchor ring centerline nodes in x (Abaqus ANCHOR = {1,6}).
    anchor = np.array([0, 5], dtype=int)
    base_bc = {
        int(anchor[0]) * 2 + 0: 0.0,
        int(anchor[1]) * 2 + 0: 0.0,
    }

    return {
        "nodes": ring_nodes,
        "ring_elems": ring_elems,
        "ndof": ndof,
        "operators": operators,
        "base_bc": base_bc,
        "top_plate": top_plate,
        "bot_plate": bot_plate,
        "ring_nodes_idx": ring_nodes_idx,
        "equator": 6,  # Abaqus node 7 -> index 6
    }


def _top_contact_nodes(contact_top, U):
    """Outer-surface nodes whose signed gap to the top HalfSpace is < tol."""
    x = contact_top._positions(U)
    gap = contact_top.obs.gap(x)
    return contact_top.cn[gap < TOP_CONTACT_TOL]


def _adaptive_solve(ops, U0, ndof, param0, param1, *, make_bc, apply_param=None,
                    label="step", verbose=True):
    """Advance a scalar load parameter from param0 to param1 with Abaqus-like
    automatic incrementation: cut on non-convergence, grow on convergence.

    ``make_bc(param)`` returns the Dirichlet dict for the attempted parameter
    value.  ``apply_param(param, ops)`` is an optional callback to update
    operators (e.g. move a HalfSpace).  Returns ``(U, param, n_steps, n_cuts)``.
    """
    U = np.asarray(U0, dtype=float).copy()
    direction = 1.0 if param1 > param0 else -1.0
    dp = INITIAL_STEP * direction
    p = float(param0)
    n_steps = 0
    n_cuts = 0
    printed = 0
    while (p - param1) * direction < -1e-12:
        dp_try = dp
        if (p + dp_try - param1) * direction > 0.0:
            dp_try = param1 - p
        p_try = p + dp_try
        if apply_param is not None:
            apply_param(p_try, ops)
        bc = make_bc(p_try)
        try:
            U_new, nit = solve_increments(ops, U, ndof, bc, n_steps=1,
                                          maxit=MAX_NEWTON)
        except Exception:
            nit = MAX_NEWTON + 1
        if nit > MAX_NEWTON:
            dp *= STEP_CUT
            n_cuts += 1
            if abs(dp) < MIN_STEP:
                raise RuntimeError(
                    f"{label}: adaptive step collapsed below {MIN_STEP} at "
                    f"param={p:.6f}"
                )
            continue
        U = U_new
        p = p_try
        n_steps += 1
        # Grow step but do not exceed the initial step magnitude.
        dp_grown = dp_try * STEP_GROWTH
        if abs(dp_grown) > INITIAL_STEP:
            dp_grown = INITIAL_STEP * direction
        dp = dp_grown
        if verbose and (n_steps == 1 or n_steps % 5 == 0 or
                        abs(p - param1) < 1e-12):
            print(f"  {label} {n_steps}: param={p:.4f}, inc={dp_try:.4f}, "
                  f"iters={nit}")
            printed = n_steps
    if verbose and printed != n_steps and n_steps > 0:
        print(f"  {label} {n_steps}: param={p:.4f}, inc={dp_try:.4f}, "
              f"iters={nit}")
    return U, p, n_steps, n_cuts


def run_ring_compress(*, verbose=True):
    m = build_model()
    ndof = m["ndof"]
    ops = m["operators"]
    base_bc = m["base_bc"]
    top_plate = m["top_plate"]
    equator = m["equator"]

    U = np.zeros(ndof)

    # ---- Compression: move top plate down by 4 units --------------------------------
    if verbose:
        print("Compression step ...")

    def _apply_top_y(y, ops_):
        ops_[2].obs.p[1] = y

    def _bc_comp(_y):
        return base_bc

    U, _, n_comp, n_comp_cuts = _adaptive_solve(
        ops, U, ndof, PLATE_Y0_TOP, PLATE_Y0_TOP - 4.0,
        make_bc=_bc_comp, apply_param=_apply_top_y, label="comp",
        verbose=verbose,
    )

    # Compression reaction: sum the y-component of the top contact residual.
    # RigidContact returns R = k*g*n (g<0), n=(0,-1), so R_y is positive
    # (upward on the ring).  Abaqus RF2 is the equal/opposite reaction on the
    # plate, also upward, so RF2 = sum(R_y).
    contact_top = ops[2]
    cR_comp = contact_top.residual(U, None, 1.0, 1.0)
    rf2_comp = float(cR_comp.values[1::2].sum())
    u1_equator_comp = float(U[equator * 2 + 0])

    # ---- Sliding: shear the ring top contact nodes horizontally by 2 units ----------
    top_contact = _top_contact_nodes(contact_top, U)
    if verbose:
        print(f"Sliding step ({len(top_contact)} top-contact nodes) ...")

    def _bc_slide(q):
        bc = dict(base_bc)
        for v in top_contact:
            bc[int(v) * 2 + 0] = q
        return bc

    U, _, n_slide, n_slide_cuts = _adaptive_solve(
        ops, U, ndof, 0.0, 2.0, make_bc=_bc_slide, label="slide",
        verbose=verbose,
    )

    # Final reaction and equator displacement.
    cR = contact_top.residual(U, None, 1.0, 1.0)
    rf2_total = float(cR.values[1::2].sum())
    u1_equator = float(U[equator * 2 + 0])

    result = {
        "top_plate_y": float(top_plate.p[1]),
        "active_top_nodes_final": int(cR.values[1::2].size),
        "RF2_comp": rf2_comp,
        "RF2_total": rf2_total,
        "U1_equator_comp": u1_equator_comp,
        "U1_equator": u1_equator,
        "n_comp_steps": n_comp,
        "n_comp_cuts": n_comp_cuts,
        "n_slide_steps": n_slide,
        "n_slide_cuts": n_slide_cuts,
    }
    if verbose:
        print("Compression end:")
        print(f"  Top plate y          : {PLATE_Y0_TOP - 4.0:.4f}")
        print(f"  Adaptive steps/cuts  : {n_comp}/{n_comp_cuts}")
        print(f"  Active top nodes     : {cR_comp.values[1::2].size}")
        print(f"  RF2                  : {result['RF2_comp']:.4f}")
        print(f"  U1 equator           : {result['U1_equator_comp']:.4f}")
        print("Final after sliding:")
        print(f"  Top plate final y    : {result['top_plate_y']:.4f}")
        print(f"  Adaptive steps/cuts  : {n_slide}/{n_slide_cuts}")
        print(f"  Active top nodes     : {result['active_top_nodes_final']}")
        print(f"  RF2                  : {result['RF2_total']:.4f}")
        print(f"  U1 equator           : {result['U1_equator']:.4f}")
    return result, U


if __name__ == "__main__":
    run_ring_compress()
