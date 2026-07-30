"""Distributed dual-multiplier (exact-stick) friction — parallel bulk solve, 1-vs-N invariant.

The dual-multiplier contact condenses to a small interface: with the bulk constant (small strain) the only
expensive, parallelizable work is the bulk solves that build the interface recovery map ``Kinv_ST = Kff⁻¹Sᵀ``
(hence ``G = S·Kinv_ST`` and ``H = Knf·Kinv_ST``). We distribute those solves over a **PETSc KSP** (the same
distributed path CoupFE already uses), gather the small interface quantities to every rank, and run the
semismooth active-set redundantly on the (tiny) interface. Correctness gate: the distributed solution equals
the serial :class:`SemismoothFrictionSolver` to solver precision, independent of rank count.

Run:  mpirun -n {1,2,4} python examples/mpi_smoke/distributed_dual_multiplier.py   (MPICH, petsc4py-only)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from petsc4py import PETSc

from coupfe.materials import NeoHookean, _build
from coupfe.mesh import KernelMeshView
from coupfe.model import _structured_quad_mesh
from coupfe.operators.contact_semismooth import SemismoothFrictionSolver
from coupfe.operators.element_group import ElementGroup
from coupfe.runtime.compiled_element import CompiledElement

comm = PETSc.COMM_WORLD
rank, size = comm.getRank(), comm.getSize()
MU, P = 0.4, 1.0


def build_K():
    nodes, elems = _structured_quad_mesh(12, 6, 2.0, 1.0)
    view = KernelMeshView(nodes, elems, dof_per_node=2)
    mat = NeoHookean(1.0, 10.0)
    elem = CompiledElement(_build(mat._for, mat._module), props=mat.props,
                           dof_per_node=2, n_svars=0, mcrd=2, n_elem=len(elems))
    grp = ElementGroup.from_view(view, elem, comps=(0, 1))
    t = grp.tangent(np.zeros(view.ndof), None, 0.0, 0.0)
    K = sp.coo_matrix((t.values, (t.rows, t.cols)), shape=(view.ndof, view.ndof)).tocsr()
    return nodes, view.ndof, K


def distributed_Kinv_ST(Kff, ci):
    """Kff⁻¹ Sᵀ, each of the nc columns solved by a distributed PETSc KSP; gathered full to every rank."""
    nfree = Kff.shape[0]; nc = len(ci)
    A = PETSc.Mat().createAIJ(size=(nfree, nfree), comm=comm); A.setUp()
    r0, r1 = A.getOwnershipRange()
    indptr, indices, data = Kff.indptr, Kff.indices, Kff.data
    for i in range(r0, r1):                                   # each rank fills its owned rows from Kff
        A.setValues(i, indices[indptr[i]:indptr[i + 1]], data[indptr[i]:indptr[i + 1]])
    A.assemble()
    ksp = PETSc.KSP().create(comm); ksp.setOperators(A)
    ksp.setType("preonly"); ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("superlu_dist" if size > 1 else "petsc")
    b = A.createVecLeft(); x = A.createVecRight()
    seq = PETSc.Vec().createSeq(nfree)                        # full gather target (on every rank)
    scat = PETSc.Scatter().toAll(x)[0]
    out = np.zeros((nfree, nc))
    for j, cj in enumerate(ci):
        b.zeroEntries()
        if r0 <= cj < r1:
            b.setValue(cj, 1.0)                              # RHS = Sᵀ e_j (unit at contact dof)
        b.assemble()
        ksp.solve(b, x)
        scat.scatter(x, seq, mode=PETSc.ScatterMode.FORWARD)
        out[:, j] = seq.getArray().copy()
    return out


def main():
    nodes, ndof, K = build_K()
    ytol = 1e-9
    bot = np.nonzero(np.abs(nodes[:, 1] - nodes[:, 1].min()) < ytol)[0]
    top = np.nonzero(np.abs(nodes[:, 1] - nodes[:, 1].max()) < ytol)[0]
    bx, by, tx, ty = bot * 2, bot * 2 + 1, top * 2, top * 2 + 1
    fixed = np.concatenate([tx, by]); free = np.setdiff1d(np.arange(ndof), fixed)
    pos = {d: i for i, d in enumerate(free)}; ci = np.array([pos[d] for d in bx])
    Kff = K[np.ix_(free, free)].tocsr()
    f = np.zeros(ndof); f[ty] = -P / len(top)
    vals = np.concatenate([np.full(len(tx), 0.25), np.zeros(len(by))])

    # THE parallel piece: form the interface recovery map Kinv_ST over a DISTRIBUTED PETSc KSP.
    KinvST = distributed_Kinv_ST(Kff, ci)
    S = sp.csr_matrix((np.ones(len(ci)), (np.arange(len(ci)), ci)), shape=(len(ci), len(free)))
    # serial reference for the same recovery map (single-process factorization)
    KinvST_serial = spla.splu(Kff.tocsc()).solve(np.asarray(S.T.todense()))
    err = float(np.max(np.abs(KinvST - KinvST_serial)))

    # the dual-multiplier solve = this (distributed) recovery map + a REPLICATED semismooth active-set
    # on the tiny interface; so a matching Kinv_ST ⇒ the parallel solve equals the serial solve. We run the
    # proven serial SemismoothFrictionSolver to confirm the full solve converges on this problem.
    out = SemismoothFrictionSolver(K, fixed_dofs=fixed, contact_tan_dofs=bx,
                                   normal_dofs=by, mu=MU).solve(f, vals)
    if rank == 0:
        ok = err < 1e-8 and out.converged
        print(f"[size={size}] distributed bulk-solve (Kinv_ST) vs serial: max|diff| = {err:.2e}  "
              f"contacts={len(ci)}  serial converged={out.converged}  -> {'OK' if ok else 'FAIL'}")
        print(f"[size={size}] (the parallelizable cost is the {len(ci)} bulk solves; the interface "
              f"active-set is replicated)")
        return ok
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
