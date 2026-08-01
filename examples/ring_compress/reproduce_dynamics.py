"""RESEARCH dynamic-relaxation ring-compression workflow.

This solves the ring-between-rigid-plates problem through the implicit dynamics
solver using a staged ramp-hold-settle protocol. No authoritative external raw
run, extraction record, or locked environment is retained, so the workflow is
not release validation.

Recompute or justify ramp and settling durations for the supplied mesh and
parameters; sampling reaction at the end of an underdamped ramp can capture a
transient rather than a quasistatic state. The defaults below are research
settings, not a retained modal qualification.

The implemented protocol (skills/contact.md "dynamic relaxation" note):

1. ramp the plate one displacement unit over ``T_RAMP`` and check sensitivity
   to the chosen rate before treating the result as quasistatic;
2. HOLD the plate and let the ring relax under ``alpha ~ 2*omega_1`` (near
   critical for the fundamental mode).  The damping force ``alpha*M*v``
   contributes while the ring moves; as ``v -> 0`` that term vanishes, but the
   residual must still be checked against the stated tolerance;
3. sample RF2 only when the kinetic energy has collapsed below the stated
   threshold.

When ``COUPFE_RING_REFERENCE_CSV`` names an authorized table, the script reports
an external column for research inspection but does not turn its difference
into a public pass/fail gate. Without that variable it runs without a
comparison. A qualified comparison requires a newly retained, licensed input
and raw external/CoupFE result record.

Run from the repo root::

    PYTHONPATH=. python examples/ring_compress/reproduce_dynamics.py            # penalty
    PYTHONPATH=. python examples/ring_compress/reproduce_dynamics.py barrier    # ppf barrier
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from coupfe import InertiaOperator
from coupfe.assembly.assemble import newton_solve, _call_max_step
from coupfe.mesh import KernelMeshView
from coupfe.operators.contact import (
    HalfSpace,
    MovingHalfSpace,
    RigidBarrierContact,
    RigidContact,
)
from coupfe.operators.element_group import ElementGroup
from coupfe.operators.inertia import lumped_mass
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INP = Path(os.environ.get("COUPFE_RING_INP", _REPO_ROOT / "ring_compress.inp")).expanduser()
_Q4_FOR = "coupfe/runtime/elements/neo_hookean_q4_fbar_native.for"

G_RING = 2.0
K_RING = 20.0
DENSITY = 1.0

# Historical research setting used to choose mass-proportional damping. A
# qualified run must recompute the spectrum for its supplied mesh and units.
OMEGA_1 = 0.0498
DAMPING = 2.0 * OMEGA_1                 # ~0.1; zeta_1 ~ 1, zeta_2 ~ 0.37

PENALTY = 1.0e4
MU = 0.5
FRICTION_KT = 1.0e4

DHAT = 0.20
KAPPA = 5.0e1
FRICTION_EPS = 1.0e-1

PLATE_Y0_TOP = 10.0
PLATE_Y0_BOT = -10.0
PLATE_HALF_WIDTH = 14.0           # matches Abaqus TOP_SURF/BOT_SURF segment length 28
COMPRESSION = 4.0

# Staged ramp-hold-settle protocol (see module docstring).
T_RAMP_PER_UNIT = 30.0            # plate travel time per displacement unit
T_HOLD_MAX = 150.0                # hold cap; the KE gate usually exits sooner
KE_TOL = 1.0e-8                   # settled when KE < KE_TOL (strain energy ~0.26)


class _MovingObstacleContact:
    """Wrap a RigidContact/RigidBarrierContact and move its obstacle in time.

    The wrapped operator's ``obs.p`` is updated from ``motion(t)`` before every
    residual/tangent/commit call.  ``max_step`` is forwarded unchanged.
    """

    def __init__(self, contact, motion):
        self.contact = contact
        self.motion = motion

    def residual(self, U, state, t, dt):
        self.contact.obs.p = np.asarray(self.motion(t), dtype=float)
        return self.contact.residual(U, state, t, dt)

    def tangent(self, U, state, t, dt):
        self.contact.obs.p = np.asarray(self.motion(t), dtype=float)
        return self.contact.tangent(U, state, t, dt)

    def commit(self, U, state, t, dt):
        self.contact.obs.p = np.asarray(self.motion(t), dtype=float)
        return self.contact.commit(U, state, t, dt)

    def max_step(self, U, dU, t=0.0, dt=None):
        # The wrapped obstacle must be at the *current* time before the CCD
        # bound is evaluated; otherwise the predictor/Newton step may be
        # allowed to jump through the moving plate.
        self.contact.obs.p = np.asarray(self.motion(t), dtype=float)
        ms = getattr(self.contact, "max_step", None)
        if ms is None:
            return 1.0
        try:
            return float(ms(U, dU, t, dt))
        except TypeError:
            return float(ms(U, dU, t))


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
    return np.array(nodes, float), np.array(elems, int) - 1


def build_model(contact_kind: str = "penalty"):
    """Build the ring model with the requested contact formulation.

    ``contact_kind`` is ``"penalty"`` or ``"barrier"``.  The returned
    ``moving_top.motion`` is a placeholder (stationary plate); the driver
    installs the staged schedule.
    """
    ring_nodes, ring_elems = _parse_ring(_INP)
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

    M = lumped_mass(ring_nodes, ring_elems, density=DENSITY,
                    dof_per_node=2, comps=(0, 1))
    inertia = InertiaOperator(M, ndof, damping=DAMPING)

    r_node = np.hypot(ring_nodes[:, 0], ring_nodes[:, 1])
    ring_nodes_idx = np.nonzero(np.abs(r_node - 10.0) < 1e-6)[0]

    if contact_kind == "penalty":
        top_y0 = PLATE_Y0_TOP
        top_plate = MovingHalfSpace(
            lambda t: np.array([0.0, top_y0]), [0.0, -1.0],
            half_width=PLATE_HALF_WIDTH,
        )
        contact_top = RigidContact(
            ring_nodes, ring_nodes_idx, top_plate,
            dof_per_node=2, comps=(0, 1), k=PENALTY, mu=MU, k_t=FRICTION_KT,
        )
        contact_bot = RigidContact(
            ring_nodes, ring_nodes_idx,
            HalfSpace([0.0, PLATE_Y0_BOT], [0.0, 1.0],
                      half_width=PLATE_HALF_WIDTH),
            dof_per_node=2, comps=(0, 1), k=PENALTY, mu=MU, k_t=FRICTION_KT,
        )
    elif contact_kind == "barrier":
        # Both plates must be initially separated from the contact nodes (gap > 0),
        # otherwise the barrier's CCD bound collapses to zero on the first step.
        top_y0 = PLATE_Y0_TOP + DHAT
        bot_y0 = PLATE_Y0_BOT - DHAT
        top_plate = MovingHalfSpace(
            lambda t: np.array([0.0, top_y0]), [0.0, -1.0],
            half_width=PLATE_HALF_WIDTH,
        )
        bot_plate = HalfSpace(
            [0.0, bot_y0], [0.0, 1.0], half_width=PLATE_HALF_WIDTH
        )
        nodal_mass = M[ring_nodes_idx * 2]
        contact_top = RigidBarrierContact(
            ring_nodes, ring_nodes_idx, top_plate,
            dof_per_node=2, comps=(0, 1), dhat=DHAT, kappa=KAPPA,
            mass=nodal_mass, mu=MU, friction_eps=FRICTION_EPS,
        )
        contact_bot = RigidBarrierContact(
            ring_nodes, ring_nodes_idx, bot_plate,
            dof_per_node=2, comps=(0, 1), dhat=DHAT, kappa=KAPPA,
            mass=nodal_mass, mu=MU, friction_eps=FRICTION_EPS,
        )
    else:
        raise ValueError("contact_kind must be 'penalty' or 'barrier'")

    moving_top = _MovingObstacleContact(
        contact_top, lambda t: np.array([0.0, top_y0]))

    anchor = np.array([0, 5], dtype=int)
    base_bc = {
        int(anchor[0]) * 2 + 0: 0.0,
        int(anchor[1]) * 2 + 0: 0.0,
    }

    return {
        "nodes": ring_nodes,
        "ndof": ndof,
        "operators": [grp, inertia, contact_bot, moving_top],
        "base_bc": base_bc,
        "contact_top": contact_top,
        "moving_top": moving_top,
        "inertia": inertia,
        "ring_nodes_idx": ring_nodes_idx,
        "equator": 6,
        "top_y0": top_y0,
    }


def abaqus_rf2_at(disp: float):
    """Interpolate ``|RF2|`` from an optional user-supplied external table.

    IMPORTANT: the CSV column ``U`` is SIMULATION TIME (step 1 spans 0..1 while
    the plate travels the full COMPRESSION), so displacement = COMPRESSION * U.
    This conversion is useful for research inspection; the table is not
    release validation evidence. Return ``None`` when no table is configured.
    """
    configured = os.environ.get("COUPFE_RING_REFERENCE_CSV")
    if not configured:
        return None
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "COUPFE_RING_REFERENCE_CSV does not name a readable file: "
            f"{path}"
        )
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.dtype.names is None or not {"U", "RF2"}.issubset(data.dtype.names):
        raise ValueError(
            "COUPFE_RING_REFERENCE_CSV must contain U and RF2 columns"
        )
    t = np.asarray(data["U"], dtype=float)
    rf2 = np.abs(np.asarray(data["RF2"], dtype=float))
    comp = t <= 1.0 + 1e-9
    d = COMPRESSION * t[comp]
    order = np.argsort(d)
    return float(np.interp(disp, d[order], rf2[comp][order]))


def run_staged(contact_kind: str = "penalty", *, stages=(1.0, 2.0, 3.0, 4.0),
               dt: float = 1.0, t_ramp_per_unit: float = T_RAMP_PER_UNIT,
               t_hold_max: float = T_HOLD_MAX, ke_tol: float = KE_TOL,
               rtol: float = 1e-8, verbose: bool = True):
    """Staged ramp-hold-settle dynamic relaxation; returns one record per stage.

    Each record holds the SETTLED (KE-gated) reaction, equator displacement and
    kinetic energy at the stage's plate displacement.

    The kinetic-energy gate checks the intended settled-state protocol. A
    release-quality equilibrium claim also needs a retained residual gate and
    time-step study.
    """
    m = build_model(contact_kind)
    ndof, ops = m["ndof"], m["operators"]
    inertia, moving_top, eq = m["inertia"], m["moving_top"], m["equator"]
    base_bc = m["base_bc"]
    top_y0 = m["top_y0"]

    # Piecewise-linear plate schedule, built incrementally as holds complete so
    # the motion is a genuine function of t (the CCD bound sees the plate at
    # the correct position throughout each ramp).
    segments = []                     # (t0, y0, t1, y1)

    def plate_y(t):
        if not segments:
            return top_y0
        for t0, y0, t1, y1 in segments:
            if t <= t1:
                if t <= t0:
                    return y0
                return y0 + (y1 - y0) * (t - t0) / (t1 - t0)
        return segments[-1][3]

    moving_top.motion = lambda t: np.array([0.0, plate_y(t)])

    U = np.zeros(ndof)
    t = 0.0
    records = []

    def step():
        nonlocal U, t
        t = t + dt
        U_prev = U
        U_pred = inertia.predictor(dt)
        dU_pred = U_pred - U_prev
        a = 1.0
        for op in ops:
            a = min(a, _call_max_step(op, U_prev, dU_pred, t, dt))
        U = U_prev + a * dU_pred
        U, _, _ = newton_solve(ops, U, None, ndof, dict(base_bc),
                               t=t, dt=dt, rtol=rtol, maxit=40)

    def ke():
        return float(0.5 * np.sum(inertia.M * inertia.v_prev ** 2))

    y_cur = top_y0
    if verbose:
        print(f"staged dynamic relaxation ({contact_kind}): "
              f"alpha={DAMPING:.3f} (=2*omega_1), dt={dt}")
        print(f"{'U':>5} {'RF2_dyn':>9} {'RF2_ref':>8} {'diff':>6} "
              f"{'KE':>10} {'equatorU1':>10} {'t':>7}")

    for target in stages:
        y_target = top_y0 - target
        t_ramp = t_ramp_per_unit * (y_cur - y_target)
        segments.append((t, y_cur, t + t_ramp, y_target))
        t_ramp_end = t + t_ramp
        while t < t_ramp_end - 1e-9:
            step()
        y_cur = y_target
        # Hold: relax until the kinetic energy collapses.
        t_hold_end = t + t_hold_max
        while t < t_hold_end - 1e-9:
            step()
            if ke() < ke_tol:
                break
        rf2 = float(moving_top.residual(U, None, t, dt).values[1::2].sum())
        reference = abaqus_rf2_at(target)
        rec = {
            "U": target,
            "rf2": rf2,
            "abq": reference,
            "ke": ke(),
            "equator_u1": float(U[eq * 2]),
            "t": t,
            "settled": ke() < ke_tol,
        }
        records.append(rec)
        if verbose:
            if reference is None:
                ref_text, err_text = "n/a", "n/a"
            else:
                ref_text = f"{reference:.4f}"
                err_text = f"{abs(rf2 - reference) / reference:.1%}"
            print(f"{target:5.1f} {rf2:9.4f} {ref_text:>8} "
                  f"{err_text:>6} {rec['ke']:10.2e} "
                  f"{rec['equator_u1']:10.4f} {t:7.1f}", flush=True)

    return records


def main():
    contact_kind = sys.argv[1] if len(sys.argv) > 1 else "penalty"
    print("Ring compression via STAGED dynamic relaxation "
          "(ramp-hold-settle, KE-gated sampling)")
    recs = run_staged(contact_kind)
    settled = all(r["settled"] for r in recs)
    print("SETTLED_RESEARCH_RUN" if settled else "UNSETTLED")


if __name__ == "__main__":
    main()
