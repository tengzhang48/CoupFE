"""1-vs-N gate for the ``robin=`` / ``pressure=`` hooks of ``solve_dynamics_distributed``.

A small Hex8 block: one face Dirichlet-fixed, a **Robin** spring-dashpot on a second face, a
linear **follower-style pressure** on a third, under implicit dynamics. The distributed solve
(gathered ``U``) is compared against the SERIAL ``solve_dynamics`` composing the SAME operators as
a list — so this gates BOTH the 1-vs-N invariant AND that the owned-row hooks assemble the
operators correctly (an independent serial oracle, not self-consistency).

Run: ``OMP_NUM_THREADS=1 mpiexec -n {1,2,4} python examples/mpi_smoke/distributed_robin_pressure.py``
(optional trailing arg = path to save the gathered U for a 1-vs-N diff). Prints ``OK`` / ``FAIL``.
"""
from __future__ import annotations

import sys

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

from coupfe import InertiaOperator, neo_hookean_kernel_props, solve_dynamics
from coupfe.assembly.distributed import element_partition, solve_dynamics_distributed
from coupfe.mesh import KernelMeshView
from coupfe.operators.base import Residual, Tangent
from coupfe.operators.element_group import ElementGroup
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

_HEX8_FOR = "coupfe/runtime/elements/neo_hookean_hex8_fbar.for"
G, K_BULK, DENSITY = 1.0, 10.0, 1.0
NE = 2                          # 2x2x2 hex block
DT, N_STEPS, DAMP = 0.1, 4, 0.5
K_SPRING, C_DASH = 5.0, 0.5     # Robin
P0, K_PRESS = 0.5, 0.5          # follower load: constant drive P0 + deformation-dependent K_press·u


def _block(ne):
    xs = np.linspace(0.0, 1.0, ne + 1)
    nodes = np.array([[x, y, z] for z in xs for y in xs for x in xs], float)
    nn = ne + 1

    def nid(i, j, k):
        return k * nn * nn + j * nn + i

    elems = [[nid(i, j, k), nid(i+1, j, k), nid(i+1, j+1, k), nid(i, j+1, k),
              nid(i, j, k+1), nid(i+1, j, k+1), nid(i+1, j+1, k+1), nid(i, j+1, k+1)]
             for k in range(ne) for j in range(ne) for i in range(ne)]
    return nodes, np.array(elems, int)


def _lumped_mass(nodes, elems, ndof):
    M = np.zeros(ndof)
    nodal = np.zeros(len(nodes))
    for e in elems:
        vol = float(np.prod(nodes[e].max(0) - nodes[e].min(0)))
        np.add.at(nodal, e, DENSITY * vol / 8.0)
    for c in range(3):
        M[c::3] = nodal
    return M


class RobinOp:
    """Constant spring-dashpot on ``dofs``: residual K·u + C·(u−u_prev)/dt, tangent K + C/dt.

    Dual interface: ``.residual``/``.tangent``/``.commit`` for the serial operator list, and
    ``.Kmat``/``.Cmat``/``.dofs``/``.u_prev`` for the distributed ``robin=`` hook."""

    def __init__(self, dofs, ndof):
        self.dofs = np.asarray(dofs, int)
        d = np.zeros(ndof)
        d[self.dofs] = 1.0
        self.Kmat = sp.diags(K_SPRING * d).tocsr()
        self.Cmat = sp.diags(C_DASH * d).tocsr()
        self.u_prev = np.zeros(ndof)

    def residual(self, U, state, t, dt):
        r = self.Kmat @ U + self.Cmat @ (U - self.u_prev) / dt
        return Residual(gdofs=self.dofs, values=r[self.dofs],
                        state_trial=None)

    def tangent(self, U, state, t, dt):
        val = K_SPRING + C_DASH / dt
        return Tangent(rows=self.dofs, cols=self.dofs,
                       values=np.full(len(self.dofs), val))

    def commit(self, U, state, t, dt):
        self.u_prev = np.asarray(U, float).copy()
        return None


class PressureOp:
    """Linear follower-style load on ``dofs``: residual −K_press·u, tangent −K_press.

    Implements the ``Operator`` contract (used both in the serial list and the distributed
    ``pressure=`` hook)."""

    def __init__(self, dofs):
        self.dofs = np.asarray(dofs, int)

    def residual(self, U, state, t, dt):
        # constant drive P0 (pushes the face) + deformation-dependent K_press·u (real tangent)
        vals = -(P0 + K_PRESS * np.asarray(U)[self.dofs])
        return Residual(gdofs=self.dofs, values=vals, state_trial=None)

    def tangent(self, U, state, t, dt):
        return Tangent(rows=self.dofs, cols=self.dofs,
                       values=np.full(len(self.dofs), -K_PRESS))

    def commit(self, U, state, t, dt):
        return None


def _serial_reference(nodes, elems, ndof, M, robin_dofs, press_dofs, base):
    """Serial solve_dynamics with the SAME operators as a list — the 1-vs-N oracle."""
    view = KernelMeshView(nodes, elems, dof_per_node=3)
    elem = CompiledElement(build_element_kernel(_HEX8_FOR, "rp_hex8_serial"),
                           props=neo_hookean_kernel_props(G, K_BULK),
                           dof_per_node=3, n_svars=0, mcrd=3,
                           n_elem=len(elems))
    grp = ElementGroup.from_view(view, elem, comps=(0, 1, 2))
    inertia = InertiaOperator(M, ndof, damping=DAMP)
    ops = [grp, inertia, RobinOp(robin_dofs, ndof), PressureOp(press_dofs)]
    bc = {int(g): 0.0 for g in base}
    U, _ = solve_dynamics(ops, np.zeros(ndof), ndof, bc, dt=DT, n_steps=N_STEPS,
                          rtol=1e-10, maxit=40)
    return U


def main():
    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    nodes, elems = _block(NE)
    view = KernelMeshView(nodes, elems, dof_per_node=3)
    ndof = view.ndof
    M = _lumped_mass(nodes, elems, ndof)
    nn = NE + 1

    zmin = np.nonzero(np.abs(nodes[:, 2]) < 1e-9)[0]        # fixed face
    zmax = np.nonzero(np.abs(nodes[:, 2] - 1.0) < 1e-9)[0]  # Robin face
    xmax = np.nonzero(np.abs(nodes[:, 0] - 1.0) < 1e-9)[0]  # pressure face
    base = np.concatenate([zmin * 3, zmin * 3 + 1, zmin * 3 + 2])
    robin_dofs = np.concatenate([zmax * 3, zmax * 3 + 1, zmax * 3 + 2])
    press_dofs = xmax * 3                                   # x-load on the +x face

    my_gm, my_coords, _ = element_partition(view, rank, size)
    elem = CompiledElement(build_element_kernel(_HEX8_FOR, "rp_hex8_dist"),
                           props=neo_hookean_kernel_props(G, K_BULK),
                           dof_per_node=3, n_svars=0, mcrd=3,
                           n_elem=max(len(my_gm), 1))
    dirich = {int(g): 0.0 for g in base}

    U_par, info = solve_dynamics_distributed(
        ndof, my_gm, my_coords, 3, elem.element_rk_batch, dirich, M,
        dt=DT, n_steps=N_STEPS, damping=DAMP, pc="lu", solver="superlu_dist",
        tol=1e-9, rtol=1e-12,
        robin=RobinOp(robin_dofs, ndof), pressure=PressureOp(press_dofs))

    if rank == 0:
        U_ser = _serial_reference(nodes, elems, ndof, M, robin_dofs, press_dofs, base)
        err = float(np.max(np.abs(U_par - U_ser)))
        moved = float(np.max(np.abs(U_par))) > 1e-6          # the load actually did something
        ok = err < 1e-9 and moved and info["rnorm"] < 1e-6
        if len(sys.argv) > 1:
            np.save(sys.argv[1], U_par)
        print(f"[size={size}] ndof={ndof} newton(last)={info['n_newton']} |R|={info['rnorm']:.2e} "
              f"max|U|={float(np.max(np.abs(U_par))):.4f}  max|U_dist - U_serial|={err:.2e} "
              f"-> {'OK' if ok else 'FAIL'}", flush=True)
    comm.barrier()


if __name__ == "__main__":
    main()
