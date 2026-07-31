"""Distributed-memory Newton solve.

The serial operator contract (``assemble_residual``/``assemble_tangent``) is an O(ndof)
interface: it takes the *global* ``U`` and returns global ``gdofs``. Distributed-memory
can't use that — each rank must hold only its partition. So the distributed unit is the
**batched element evaluator** an :class:`~coupfe.runtime.compiled_element.CompiledElement`
exposes (``element_rk_batch(coords, U, DU) -> (R, K)``, the same f2py call the
``ElementGroup`` scatters serially). Each rank owns a block of elements, ghosts the ``U``
values it needs via a PETSc ``VecScatter`` (no all-gather), assembles its rows of the
distributed ``Mat``/``Vec`` with **global** indices + ``ADD_VALUES`` (PETSc sums
off-process contributions), and a PETSc ``KSP`` solves the global system. Load-stepped
Newton + line search, exactly as the serial driver.

The current entry point uses a caller-supplied batched element evaluator and is
**petsc4py only**. See ``docs/api.md`` and ``docs/capabilities.md`` for its
supported scope.

The intended correctness gate is a serial-versus-rank comparison at multiple
rank counts, with tolerances set by the selected solver and conditioning.

    mpirun -n 4 python examples/mpi_smoke/distributed_neohookean.py
"""

from __future__ import annotations

import numpy as np

from coupfe.mesh.distribute import partition_elements
from coupfe.operators.contact import (
    DeformableBarrierContact2D,
    DeformableContact2D,
    rigid_penalty_eval,
)
from coupfe.operators.contact3d import DeformableBarrierContact3D


def element_partition(view, rank, size, comps=None):
    """This rank's element block: ``(my_gm, my_coords, ndof)``.

    ``my_gm`` is ``(my_ne, ndofel)`` of **global** DOFs in the element's node-major order
    (matching ``ElementGroup.gm`` and the kernel's ``[n0c0, n0c1, n1c0, …]`` layout);
    ``my_coords`` is ``(my_ne, nne, ndim)``. Build-then-partition keeps the *global* node
    numbering identical to the serial solve (so the 1-vs-N compare is exact); memory-local
    structured generation is the separate scale optimization (each rank builds only its
    block) — same per-rank logic, different mesh source.
    """
    dpn = view.dof_per_node
    comps = (np.arange(dpn, dtype=int) if comps is None else np.asarray(comps, dtype=int))
    parts = partition_elements(view, size)
    mine = np.where(parts == rank)[0]
    my_elems = view.elems[mine]
    my_gm = (my_elems[:, :, None] * dpn + comps[None, None, :]).reshape(len(mine), -1)
    my_coords = view.nodes[my_elems]
    return my_gm, my_coords, view.ndof


def solve_distributed(ndof, my_gm, my_coords, dof_per_node, batch_fn, dirichlet_fn,
                      n_steps, *, max_newton=60, tol=1e-7, line_search=True,
                      pc="gamg", ksp_type="gmres", solver="superlu_dist", rtol=1e-10,
                      forcing=True, local_ndof=None, u0=None, contact=None,
                      deformable_contact=None, verbose=False):
    """Distributed-memory load-stepped Newton. Returns ``(U_full, info)``.

    ``batch_fn(coords, U, DU) -> (R_all (ne, ndofel), K_all (ne, ndofel, ndofel))`` is a
    rank-local batched element evaluator (e.g. ``CompiledElement.element_rk_batch``).
    ``dirichlet_fn(frac) -> {global_dof: value}`` returns the prescribed DOFs at load
    fraction ``frac = step/n_steps`` (scale inside it to ramp). ``pc="lu"`` with ``solver``
    gives an exact (direct) linear solve; ``"gamg"`` + ``ksp_type="gmres"`` is the iterative
    elasticity default. The direct ``solver`` default is **superlu_dist** because it is
    *reproducible* in parallel; **MUMPS is NOT** — its parallel pivoting/scheduling varies
    run-to-run (~1e-12), which an ill-conditioned mode (e.g. slipping frictional contact)
    amplifies into a visibly non-repeatable solution. Use MUMPS only when repeatability is not
    required. (Both solve the asymmetric friction tangent correctly; the difference is purely
    reproducibility — see docs/lessons_learned.md.)

    ``contact`` (optional) wires in **rigid-obstacle penalty contact** (frictionless or
    Coulomb friction) as a per-rank **node-local** contribution — the contact contract that
    the bulk operator can't carry distributed. Pass a dict
    ``{"nodes": global node ids, "coords": their reference coords (n, dim), "obstacle": obj,
    "k": .., "mu": .., "k_t": .., "comps": (..)}``. Each contact node is handled by exactly
    one rank (the one owning its first DOF) — no cross-rank coupling (the obstacle is known
    everywhere), so it just adds rows to the distributed system; its friction state lives on
    that rank.

    ``deformable_contact`` (optional) wires in **deformable-deformable** node-to-segment penalty
    contact, where a pair couples a secondary *vertex* on one body with a primary *edge* on
    another — and after partitioning those nodes can live on **different ranks**. Pass a dict
    ``{"nodes_ref": full reference coords (N, dim), "secondary": global secondary node ids,
    "edges": (ne, 2) global edge node ids, "k": .., "comps": (..), "search_band": ..}``. The
    contact surface (secondary ∪ edge nodes) is replicated to every rank each iteration via a
    VecScatter (it is O(surface) ≪ ndof); each rank assembles the contact COO for the
    secondaries it **owns** (by first-DOF ownership, so each pair is computed exactly once) with
    **global** DOF indices, and PETSc off-process ``ADD_VALUES`` routes the edge-node
    contributions to whoever owns them. Penalty (frictionless) for now — barrier + global CCD
    layer on next. Verified rank-independent in ``distributed_deformable_residual`` (assembly)
    and ``distributed_deformable_solve`` (full solve).
    """
    from petsc4py import PETSc                       # petsc4py only — never mpi4py

    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    my_gm = np.asarray(my_gm)
    my_ne = len(my_gm)
    ADD = PETSc.InsertMode.ADD_VALUES
    INS = PETSc.InsertMode.INSERT_VALUES
    FWD = PETSc.ScatterMode.FORWARD
    vsize = ndof if local_ndof is None else (local_ndof, ndof)
    msize = ([ndof, ndof] if local_ndof is None
             else [(local_ndof, ndof), (local_ndof, ndof)])

    # local map: the ghosted slice this rank needs (owned + one-layer halo DOFs)
    needed = (np.unique(my_gm).astype(PETSc.IntType) if my_ne
              else np.zeros(0, dtype=PETSc.IntType))
    g2l = np.empty(ndof, dtype=np.int64)
    if my_ne:
        g2l[needed] = np.arange(len(needed))
        my_gm_local = g2l[my_gm]

    U = PETSc.Vec().createMPI(vsize, comm=comm)
    U.set(0.0)
    rs, re = U.getOwnershipRange()
    if u0 is not None:
        u0 = np.asarray(u0, dtype=float)
        U.setValues(np.arange(rs, re, dtype=PETSc.IntType), u0[rs:re], addv=INS)
        U.assemble()
    needed_is = PETSc.IS().createGeneral(needed, comm=comm)
    U_loc = PETSc.Vec().createSeq(len(needed))
    Up_loc = U_loc.duplicate()
    scatter = PETSc.Scatter().create(U, needed_is, U_loc, None)

    def ghost(vec, into):
        scatter.scatter(vec, into, addv=INS, mode=FWD)
        return np.asarray(into.getArray())

    rows_all = my_gm.astype(PETSc.IntType)

    # --- optional rigid-contact contribution (node-local; this rank's owned contact nodes) ---
    c_on = contact is not None
    if c_on:
        c_comps = np.asarray(contact.get("comps", np.arange(dof_per_node)), dtype=int)
        c_gd = (np.asarray(contact["nodes"], dtype=np.int64)[:, None] * dof_per_node
                + c_comps[None, :])                              # (ncontact, cdim) global DOFs
        c_mine = (c_gd[:, 0] >= rs) & (c_gd[:, 0] < re)          # responsible: owns first DOF
        c_idx_m = np.nonzero(c_mine)[0]
        c_gd_m = c_gd[c_mine].astype(PETSc.IntType)              # (nm, cdim)
        c_Xc = np.asarray(contact["coords"], dtype=float)[c_mine]
        c_obs, c_k = contact["obstacle"], float(contact.get("k", 1.0e4))
        c_mu, c_kt = float(contact.get("mu", 0.0)), contact.get("k_t", None)
        c_xprev0 = contact.get("x_prev", None)
        c_ft0 = contact.get("ft", None)
        c_state = {
            "stick": (c_Xc.copy() if c_xprev0 is None
                      else np.asarray(c_xprev0, dtype=float)[c_mine].copy()),
            "ftc": (np.zeros_like(c_Xc) if c_ft0 is None
                    else np.asarray(c_ft0, dtype=float)[c_mine].copy()),
        }   # friction state
        c_flat = c_gd_m.ravel()
        c_Uc = PETSc.Vec().createSeq(len(c_flat))
        c_scat = PETSc.Scatter().create(U, PETSc.IS().createGeneral(c_flat, comm=comm),
                                        c_Uc, None)

        def c_positions(Uvec):                                  # current positions of my nodes
            c_scat.scatter(Uvec, c_Uc, addv=INS, mode=FWD)
            return c_Xc + np.asarray(c_Uc.getArray()).reshape(c_gd_m.shape)

    # --- optional deformable-deformable contact (cross-rank; this rank's owned secondaries) ---
    dc_on = deformable_contact is not None
    if dc_on:
        dc = deformable_contact
        dc_kind = dc.get("kind", "penalty")              # "penalty" | "barrier" (penetration-free)
        dc_comps = np.asarray(dc.get("comps", np.arange(dof_per_node)), dtype=int)
        dc_Xref = np.asarray(dc["nodes_ref"], dtype=float)
        dc_sec = np.asarray(dc["secondary"], dtype=np.int64)
        dc_edges = np.asarray(dc["edges"], dtype=int)
        dc_mu = float(dc.get("mu", 0.0))
        # this rank owns the secondaries whose first DOF it owns (partition → each pair once)
        dc_owned = dc_sec[(dc_sec * dof_per_node >= rs) & (dc_sec * dof_per_node < re)]
        if dc_kind == "barrier":
            dc_op = DeformableBarrierContact2D(
                dc_Xref, dc_owned, dc_edges, dof_per_node=dof_per_node, comps=dc_comps,
                dhat=float(dc.get("dhat", 0.05)), kappa=float(dc.get("kappa", 1.0e2)),
                eta=float(dc.get("eta", 0.9)), mu=dc_mu,
                friction_eps=float(dc.get("friction_eps", 1.0e-4)))
        else:
            dc_op = DeformableContact2D(dc_Xref, dc_owned, dc_edges, dof_per_node=dof_per_node,
                                        comps=dc_comps, k=float(dc.get("k", 1.0e3)),
                                        search_band=dc.get("search_band", None))
        # replicate the contact surface (secondary ∪ edge nodes) to every rank each iteration
        dc_surf_nodes = np.unique(np.concatenate([dc_sec, dc_edges.ravel()])).astype(np.int64)
        dc_surf_dofs = (dc_surf_nodes[:, None] * dof_per_node
                        + dc_comps[None, :]).ravel().astype(PETSc.IntType)
        dc_Uc = PETSc.Vec().createSeq(len(dc_surf_dofs))
        dc_dUc = PETSc.Vec().createSeq(len(dc_surf_dofs))
        dc_scat = PETSc.Scatter().create(U, PETSc.IS().createGeneral(dc_surf_dofs, comm=comm),
                                         dc_Uc, None)
        dc_Ufull = np.zeros(ndof)                            # full U; only surface entries are set
        dc_dUfull = np.zeros(ndof)

        def dc_surface_U(Uvec):                             # COLLECTIVE — every rank must call
            dc_scat.scatter(Uvec, dc_Uc, addv=INS, mode=FWD)
            dc_Ufull[dc_surf_dofs] = np.asarray(dc_Uc.getArray())
            return dc_Ufull

        def dc_ccd_alpha(Uvec, dUvec):
            """Global CCD step bound: gather the surface U and Newton increment dU, compute this
            rank's local point-edge ``max_step`` over its owned secondaries (1.0 if none active),
            then reduce to the GLOBAL minimum — the limiting pair can be on any rank. COLLECTIVE
            (scatter + reduction): every rank must call it. petsc4py-only reduction via a
            1-entry-per-rank MPI Vec + ``Vec.min()`` (never mpi4py)."""
            Uf = dc_surface_U(Uvec)
            dc_scat.scatter(dUvec, dc_dUc, addv=INS, mode=FWD)
            dc_dUfull[dc_surf_dofs] = np.asarray(dc_dUc.getArray())
            a_loc = float(dc_op.max_step(Uf, dc_dUfull))
            red = PETSc.Vec().createMPI((1, size), comm=comm)
            red.setValue(rank, a_loc)
            red.assemble()
            _, a_glob = red.min()
            red.destroy()
            return a_glob

    def assemble(Uvec, Upvec):
        Ua = ghost(Uvec, U_loc)
        Upa = ghost(Upvec, Up_loc)
        A = PETSc.Mat().createAIJ(msize, comm=comm)
        A.setPreallocationNNZ(dof_per_node * 18)
        A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        R = PETSc.Vec().createMPI(vsize, comm=comm)
        R.set(0.0)
        if my_ne:
            U_all = Ua[my_gm_local]
            R_all, K_all = batch_fn(my_coords, U_all, U_all - Upa[my_gm_local])
            for li in range(my_ne):
                A.setValues(rows_all[li], rows_all[li], np.asarray(K_all[li], float),
                            addv=ADD)
                R.setValues(rows_all[li], np.asarray(R_all[li], float), addv=ADD)
        if c_on:                                                # node-local contact rows
            # NB: c_positions() does a COLLECTIVE VecScatter — every rank must call it,
            # even one that owns no contact nodes (else the others deadlock). The setValues
            # loop is naturally a no-op when this rank has no active contact nodes.
            Rc, Kc, _ft, act = rigid_penalty_eval(c_positions(Uvec), c_obs, k=c_k, mu=c_mu,
                                                  k_t=c_kt, x_prev=c_state["stick"],
                                                  ft=c_state["ftc"])
            for i in np.nonzero(act)[0]:
                A.setValues(c_gd_m[i], c_gd_m[i], np.asarray(Kc[i], float), addv=ADD)
                R.setValues(c_gd_m[i], np.asarray(Rc[i], float), addv=ADD)
        if dc_on:                                               # cross-rank deformable contact
            # NB: dc_surface_U() is a COLLECTIVE VecScatter — every rank must call it (even one
            # owning no secondaries) or the others deadlock. The owned-secondary COO then carries
            # GLOBAL dof indices; PETSc ADD_VALUES routes off-process edge-node entries.
            Uf = dc_surface_U(Uvec)
            Rdc = dc_op.residual(Uf, None, 0, 0)
            if Rdc.gdofs.size:
                R.setValues(np.asarray(Rdc.gdofs, PETSc.IntType),
                            np.asarray(Rdc.values, float), addv=ADD)
            Tdc = dc_op.tangent(Uf, None, 0, 0)
            for r_, c_, val in zip(Tdc.rows, Tdc.cols, Tdc.values):
                A.setValue(int(r_), int(c_), float(val), addv=ADD)
        A.assemble()
        R.assemble()
        return A, R

    def gather_full(vec):
        sc, seq = PETSc.Scatter.toAll(vec)
        sc.scatter(vec, seq, addv=INS, mode=FWD)
        out = np.asarray(seq.getArray()).copy()
        sc.destroy()
        seq.destroy()
        return out

    Uprev = U.duplicate()
    ksp_its_last, ksp_diverged = 0, False
    n_newton_last, rnorm_last = 0, 0.0
    for step in range(1, n_steps + 1):
        frac = step / n_steps
        U.copy(Uprev)
        bc = dirichlet_fn(frac)
        bc_all = np.array(sorted(bc), dtype=PETSc.IntType)
        owned = [(g, bc[g]) for g in bc if rs <= g < re]
        if owned:
            U.setValues([g for g, _ in owned], [v for _, v in owned], addv=INS)
        U.assemble()
        bc_owned = np.array([g for g, _ in owned], dtype=PETSc.IntType)
        z_owned = np.zeros(len(bc_owned))
        rnorm_prev = 0.0
        for _it in range(max_newton):
            A, R = assemble(U, Uprev)
            if len(bc_owned):
                R.setValues(bc_owned, z_owned, addv=INS)
            R.assemble()
            rnorm = R.norm()
            n_newton_last, rnorm_last = _it + 1, rnorm
            if rnorm < tol:
                A.destroy()
                R.destroy()
                break
            A.zeroRows(bc_all, diag=1.0)
            negR = R.copy()
            negR.scale(-1.0)
            du = R.duplicate()
            ksp = PETSc.KSP().create(comm)
            ksp.setOperators(A)
            ksp.setType(ksp_type)
            ksp.getPC().setType(pc)
            if pc == "lu":
                ksp.getPC().setFactorSolverType(solver)
            # Inexact Newton (Eisenstat-Walker): loose linear rtol far from the root,
            # tightening as the Newton residual drops. Skipped for a direct PC.
            if forcing and pc != "lu":
                eta = 1.0e-2 if rnorm_prev <= 0.0 else 0.9 * (rnorm / rnorm_prev) ** 2
                eta = min(1.0e-2, max(1.0e-4, eta))
            else:
                eta = rtol
            rnorm_prev = rnorm
            ksp.setTolerances(rtol=eta, max_it=2000)
            ksp.solve(negR, du)
            ksp_its_last = ksp.getIterationNumber()
            if ksp.getConvergedReason() < 0:
                ksp_diverged = True
            alpha = 1.0
            if dc_on and dc_kind == "barrier":
                # global CCD step bound — collective; every rank must reach this (they break the
                # Newton loop together on the rank-consistent global rnorm). Keeps the line search
                # inside the non-penetrating region so the barrier residual never blows up.
                alpha = min(alpha, dc_ccd_alpha(U, du))
            if line_search:
                Ut = U.duplicate()
                for _ls in range(30):
                    U.copy(Ut)
                    Ut.axpy(alpha, du)
                    A2, Rt = assemble(Ut, Uprev)
                    if len(bc_owned):
                        Rt.setValues(bc_owned, z_owned, addv=INS)
                    Rt.assemble()
                    ok = Rt.norm() < rnorm
                    A2.destroy()
                    Rt.destroy()
                    if ok:
                        break
                    alpha *= 0.5
                Ut.destroy()
            U.axpy(alpha, du)
            ksp.destroy()
            negR.destroy()
            du.destroy()
            A.destroy()
            R.destroy()
        if c_on and c_mu > 0.0:                        # commit friction state (collective)
            xc = c_positions(U)                        # every rank calls the scatter
            _R, _K, ftn, _a = rigid_penalty_eval(xc, c_obs, k=c_k, mu=c_mu, k_t=c_kt,
                                                 x_prev=c_state["stick"], ft=c_state["ftc"])
            c_state["ftc"], c_state["stick"] = ftn, xc
        if dc_on and dc_mu > 0.0:                       # advance deformable friction slip ref
            dc_op.commit(dc_surface_U(U), None, 0, 0)  # collective scatter (every rank calls)
        if verbose and rank == 0:
            ftn_dbg = (np.linalg.norm(c_state["ftc"]) if c_on else 0.0)
            print(f"  [dist] step {step}: newton={n_newton_last} |R|={rnorm_last:.2e} "
                  f"|ftc|={ftn_dbg:.8e}", flush=True)

    contact_info = None
    if c_on:
        xc = c_positions(U)                              # collective scatter
        gap = np.asarray(c_obs.gap(xc), dtype=float)
        active = gap < 0.0
        fn = -c_k * gap[active]
        ft = c_state["ftc"][active]

        def _sum_scalar(value):
            red = PETSc.Vec().createMPI((1, size), comm=comm)
            red.setValue(rank, float(value), addv=INS)
            red.assemble()
            out = float(red.sum())
            red.destroy()
            return out

        ft_sum = np.zeros(c_Xc.shape[1])
        if ft.size:
            ft_sum[:] = ft.sum(axis=0)
        contact_info = {
            "P": _sum_scalar(fn.sum() if fn.size else 0.0),
            "Q_signed": _sum_scalar(ft_sum[0] if ft_sum.size else 0.0),
            "active_contact_nodes": int(round(_sum_scalar(np.count_nonzero(active)))),
        }
        contact_info["Q"] = abs(contact_info["Q_signed"])
        if contact.get("return_state", False):
            n_contact = len(contact["nodes"])
            cdim = c_gd_m.shape[1] if c_gd_m.ndim == 2 else len(c_comps)
            ft_vec = PETSc.Vec().createMPI(n_contact * cdim, comm=comm)
            if len(c_idx_m):
                gd_ft = (c_idx_m[:, None] * cdim + np.arange(cdim)[None, :]).ravel()
                ft_vec.setValues(gd_ft.astype(PETSc.IntType), c_state["ftc"].ravel(), addv=INS)
            ft_vec.assemble()
            sc, seq = PETSc.Scatter.toAll(ft_vec)
            sc.scatter(ft_vec, seq, addv=INS, mode=FWD)
            contact_info["ft"] = np.asarray(seq.getArray()).reshape(n_contact, cdim).copy()
            sc.destroy()
            seq.destroy()
            ft_vec.destroy()

    n_ghost = int((needed >= re).sum() + (needed < rs).sum()) if my_ne else 0
    return gather_full(U), dict(rank=rank, size=size, my_ne=my_ne, n_owned=re - rs,
                                n_ghost=n_ghost, ksp_its=ksp_its_last,
                                ksp_diverged=ksp_diverged, n_newton=n_newton_last,
                                rnorm=rnorm_last, contact=contact_info)


class _DistDeformableContact:
    """Cross-rank deformable contact for the distributed drivers (shared by the quasistatic and
    dynamics solvers): surface replication (a VecScatter, O(surface)≪ndof) + owned-secondary COO
    with GLOBAL dof indices + PETSc off-process ADD_VALUES routing + global CCD min-reduction.

    Built from a ``deformable_contact`` spec dict and a template MPI vector ``U`` (for the ownership
    range + the scatters). All scattering methods are COLLECTIVE — every rank must call them, even
    one owning no secondaries, or the others deadlock. Behaviour-identical to the inline block in
    :func:`solve_distributed`. **petsc4py only.**
    """

    def __init__(self, spec, dof_per_node, ndof, U, comm):
        from petsc4py import PETSc
        self._PETSc = PETSc
        self.comm = comm
        self.rank, self.size = comm.getRank(), comm.getSize()
        rs, re = U.getOwnershipRange()
        self.kind = spec.get("kind", "penalty")
        comps = np.asarray(spec.get("comps", np.arange(dof_per_node)), dtype=int)
        Xref = np.asarray(spec["nodes_ref"], dtype=float)
        sec = np.asarray(spec["secondary"], dtype=np.int64)
        edges = np.asarray(spec["edges"], dtype=int)
        self.mu = float(spec.get("mu", 0.0))
        owned = sec[(sec * dof_per_node >= rs) & (sec * dof_per_node < re)]   # each pair once
        # The adaptive barrier stiffness s = κ + M/d² needs the
        # per-secondary mass. spec["mass"] is the full per-global-node mass;
        # slice it to the owned secondaries.
        mass_full = spec.get("mass", None)
        mass_owned = (None if mass_full is None
                      else np.asarray(mass_full, dtype=float)[owned])
        if self.kind == "barrier":
            self.op = DeformableBarrierContact2D(
                Xref, owned, edges, dof_per_node=dof_per_node, comps=comps,
                dhat=float(spec.get("dhat", 0.05)), kappa=float(spec.get("kappa", 1.0e2)),
                eta=float(spec.get("eta", 0.9)), mu=self.mu, mass=mass_owned,
                friction_eps=float(spec.get("friction_eps", 1.0e-4)),
                freeze_pairing=bool(spec.get("freeze_pairing", False)),
                all_primitive=bool(spec.get("all_primitive", False)))
        else:
            self.op = DeformableContact2D(Xref, owned, edges, dof_per_node=dof_per_node,
                                          comps=comps, k=float(spec.get("k", 1.0e3)),
                                          search_band=spec.get("search_band", None))
        surf_nodes = np.unique(np.concatenate([sec, edges.ravel()])).astype(np.int64)
        self.surf_dofs = (surf_nodes[:, None] * dof_per_node
                          + comps[None, :]).ravel().astype(PETSc.IntType)
        self._Uc = PETSc.Vec().createSeq(len(self.surf_dofs))
        self._dUc = PETSc.Vec().createSeq(len(self.surf_dofs))
        self._scat = PETSc.Scatter().create(
            U, PETSc.IS().createGeneral(self.surf_dofs, comm=comm), self._Uc, None)
        self._Ufull = np.zeros(ndof)
        self._dUfull = np.zeros(ndof)
        self._INS = PETSc.InsertMode.INSERT_VALUES
        self._ADD = PETSc.InsertMode.ADD_VALUES
        self._FWD = PETSc.ScatterMode.FORWARD

    def surface_U(self, Uvec):                              # COLLECTIVE — full U at surface dofs
        self._scat.scatter(Uvec, self._Uc, addv=self._INS, mode=self._FWD)
        self._Ufull[self.surf_dofs] = np.asarray(self._Uc.getArray())
        return self._Ufull

    def assemble_into(self, A, R, Uvec):                    # COLLECTIVE — add residual/tangent COO
        Uf = self.surface_U(Uvec)
        Rdc = self.op.residual(Uf, None, 0, 0)
        if Rdc.gdofs.size:
            R.setValues(np.asarray(Rdc.gdofs, self._PETSc.IntType),
                        np.asarray(Rdc.values, float), addv=self._ADD)
        Tdc = self.op.tangent(Uf, None, 0, 0)
        for r_, c_, val in zip(Tdc.rows, Tdc.cols, Tdc.values):
            A.setValue(int(r_), int(c_), float(val), addv=self._ADD)

    def ccd_alpha(self, Uvec, dUvec):                      # COLLECTIVE — global min point-edge CCD
        Uf = self.surface_U(Uvec)
        self._scat.scatter(dUvec, self._dUc, addv=self._INS, mode=self._FWD)
        self._dUfull[self.surf_dofs] = np.asarray(self._dUc.getArray())
        a_loc = float(self.op.max_step(Uf, self._dUfull))   # 1.0 if this rank owns no active pair
        red = self._PETSc.Vec().createMPI((1, self.size), comm=self.comm)
        red.setValue(self.rank, a_loc)
        red.assemble()
        _, a_glob = red.min()
        red.destroy()
        return a_glob

    def commit(self, Uvec):                                # COLLECTIVE — advance friction slip ref
        if self.mu > 0.0 or getattr(self.op, "freeze_pairing", False):  # freeze: op.commit clears cache
            self.op.commit(self.surface_U(Uvec), None, 0, 0)


class _DistDeformableContact3D(_DistDeformableContact):
    """Cross-rank 3D deformable barrier (+ smoothed friction) — the 3D analog of
    :class:`_DistDeformableContact`. The substantive methods (``surface_U``, ``assemble_into``,
    ``ccd_alpha``, ``commit``) are op-agnostic and inherited; only the operator + surface differ.

    Two primitives are partitioned: **vertex-face** by the secondary vertex (this rank gets the
    vertices whose first DOF it owns — passed as the op's ``vertices``), and **edge-edge** by the
    first node of each pair's first edge (the op's ``owns_edge_pair`` predicate) — so every contact
    pair is assembled by exactly one rank, with global dof indices, PETSc ``ADD_VALUES`` routing the
    off-process stencil nodes. The contact surface (vertices ∪ face nodes ∪ edge nodes) is replicated
    to every rank. The collision-bound path requires a feasible start and a
    driver that honors the global step bound. **petsc4py only.**
    """

    def __init__(self, spec, dof_per_node, ndof, U, comm):
        from petsc4py import PETSc
        self._PETSc = PETSc
        self.comm = comm
        self.rank, self.size = comm.getRank(), comm.getSize()
        rs, re = U.getOwnershipRange()
        self.kind = "barrier"
        comps = np.asarray(spec.get("comps", np.arange(dof_per_node)), dtype=int)
        Xref = np.asarray(spec["nodes_ref"], dtype=float)
        verts = np.asarray(spec["vertices"], dtype=np.int64)
        faces = np.asarray(spec["faces"], dtype=int)
        edges = np.asarray(spec["edges"], dtype=int)
        self.mu = float(spec.get("mu", 0.0))
        owned_v = verts[(verts * dof_per_node >= rs) & (verts * dof_per_node < re)]
        mass = spec.get("mass", None)
        owned_mass = None
        if mass is not None:
            mv = {int(v): float(m) for v, m in zip(verts, np.asarray(mass, float))}
            owned_mass = np.array([mv[int(v)] for v in owned_v]) if len(owned_v) else None

        def _owns(n0, rs=rs, re=re, d=dof_per_node):       # owns the pair's first-edge first node
            return rs <= n0 * d < re

        self.op = DeformableBarrierContact3D(
            Xref, owned_v, faces, edges, dof_per_node=dof_per_node, comps=comps,
            dhat=float(spec.get("dhat", 0.05)), kappa=float(spec.get("kappa", 1.0e2)),
            mass=owned_mass, mu=self.mu, friction_eps=float(spec.get("friction_eps", 1.0e-4)),
            owns_edge_pair=_owns)
        surf_nodes = np.unique(np.concatenate(
            [verts, faces.ravel(), edges.ravel()])).astype(np.int64)
        self.surf_dofs = (surf_nodes[:, None] * dof_per_node
                          + comps[None, :]).ravel().astype(PETSc.IntType)
        self._Uc = PETSc.Vec().createSeq(len(self.surf_dofs))
        self._dUc = PETSc.Vec().createSeq(len(self.surf_dofs))
        self._scat = PETSc.Scatter().create(
            U, PETSc.IS().createGeneral(self.surf_dofs, comm=comm), self._Uc, None)
        self._Ufull = np.zeros(ndof)
        self._dUfull = np.zeros(ndof)
        self._INS = PETSc.InsertMode.INSERT_VALUES
        self._ADD = PETSc.InsertMode.ADD_VALUES
        self._FWD = PETSc.ScatterMode.FORWARD


def solve_dynamics_distributed(ndof, my_gm, my_coords, dof_per_node, batch_fn, dirichlet, mass,
                               *, dt, n_steps, v0=None, damping=0.0, u0=None, force=None,
                               step_callback=None, robin=None, pressure=None,
                               max_newton=60, tol=1e-7, pc="lu", ksp_type="gmres",
                               solver="superlu_dist", rtol=1e-10, deformable_contact=None,
                               verbose=False):
    """Distributed implicit (backward-Euler) **dynamics** — the ppf/IPC substrate for robust contact,
    distributed. Each step Newton-solves ``M/dt²(u−û) + F_int(u) + F_contact(u) − F_ext = 0`` for
    ``u``, warm-started from the **CCD-bounded** inertial predictor ``û = u_prev + dt·v_prev``
    (an unbounded predictor would teleport a node through the barrier band before Newton runs).

    The shipped deformable-barrier examples use dynamics because inertia and
    damping can regularize non-smooth stick/slip and active-set transitions.
    With the load held, backward-Euler numerical damping plus optional Rayleigh
    ``αM`` can be used for a scoped dynamic-relaxation study. A settled state
    still needs explicit residual, kinetic-energy, and step-size checks. See
    ``docs/theory/contact_dynamics.md``.

    Node-local pieces (each rank, its owned DOFs): inertia (diagonal ``M/dt²`` + Rayleigh ``αM/dt``),
    the predictor, the external ``force`` (e.g. gravity), and the velocity state. Bulk via
    ``batch_fn`` (ghosted). Cross-rank deformable contact + global CCD via :class:`_DistDeformableContact`.

    ``mass`` is the **lumped** mass per global DOF (full length; 0 on non-inertial DOFs). ``dirichlet``
    is a dict (static BCs) or a callable ``t -> {dof: value}``. ``force`` (optional) is the external
    force per global DOF (full length). Returns ``(U_full, info)``. A serial
    comparison at multiple rank counts is the intended implementation gate;
    its tolerance depends on the selected solver. **petsc4py only.**

    Optional boundary hooks (both default ``None`` → no effect; distributed by OWNED ROWS, so no
    off-process ``ADD_VALUES`` — the residual is gathered collectively each iteration because a
    boundary operator can couple DOFs that split across ranks). Core does NOT import these classes;
    they are duck-typed contracts, so any app (e.g. cardiac) supplies its own conforming operator:

    - ``robin`` — a **constant** spring-dashpot Robin BC (reference-config `K`, `C`; residual
      ``K·u + C·(u−u_prev)/dt``, tangent ``K + C/dt``). Required attributes: ``.Kmat``, ``.Cmat``
      (scipy sparse, ``ndof×ndof``), ``.dofs`` (int array of the DOFs it touches), and a mutable
      ``.u_prev`` (full-length; advanced to the committed ``U`` after each step).
    - ``pressure`` — a general (e.g. deformation-dependent follower) load implementing the
      ``Operator`` contract: ``.residual(U_full, None, t, dt) -> Residual(gdofs, values)`` and
      ``.tangent(U_full, None, t, dt) -> Tangent(rows, cols, values)``; recomputed each iteration.
    """
    from petsc4py import PETSc                              # petsc4py only — never mpi4py

    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    my_gm = np.asarray(my_gm)
    my_ne = len(my_gm)
    ADD = PETSc.InsertMode.ADD_VALUES
    INS = PETSc.InsertMode.INSERT_VALUES
    FWD = PETSc.ScatterMode.FORWARD

    needed = (np.unique(my_gm).astype(PETSc.IntType) if my_ne
              else np.zeros(0, dtype=PETSc.IntType))
    g2l = np.empty(ndof, dtype=np.int64)
    if my_ne:
        g2l[needed] = np.arange(len(needed))
        my_gm_local = g2l[my_gm]

    U = PETSc.Vec().createMPI(ndof, comm=comm)
    U.set(0.0)
    rs, re = U.getOwnershipRange()
    own = np.arange(rs, re, dtype=PETSc.IntType)
    if u0 is not None:
        U.setValues(own, np.asarray(u0, dtype=float)[rs:re], addv=INS)
    U.assemble()
    # Bulk ghost scatter — only when this rank owns elements. A size-0 scatter (no-bulk: pure
    # mass-point / contact-only problems, my_ne=0) is degenerate in PETSc and can intermittently
    # SEGV when invoked, so skip it entirely; ghost() is only ever consumed under `if my_ne`.
    if my_ne:
        needed_is = PETSc.IS().createGeneral(needed, comm=comm)
        U_loc = PETSc.Vec().createSeq(len(needed))
        scatter = PETSc.Scatter().create(U, needed_is, U_loc, None)

        def ghost(vec):
            scatter.scatter(vec, U_loc, addv=INS, mode=FWD)
            return np.asarray(U_loc.getArray())
    else:
        def ghost(vec):
            return np.zeros(0)

    rows_all = my_gm.astype(PETSc.IntType)

    # node-local inertia (owned DOFs only): lumped mass slice + the inertial owned DOFs
    M_own = np.asarray(mass, dtype=float)[rs:re]
    inertial = np.nonzero(M_own != 0.0)[0]                 # local indices into [rs,re)
    inertial_g = (rs + inertial).astype(PETSc.IntType)     # their global DOFs
    m_in = M_own[inertial]
    f_own = (None if force is None else np.asarray(force, dtype=float)[rs:re])

    # dynamic state (owned slices)
    u_prev = np.asarray(u0, dtype=float)[rs:re].copy() if u0 is not None else np.zeros(re - rs)
    v_prev = np.asarray(v0, dtype=float)[rs:re].copy() if v0 is not None else np.zeros(re - rs)

    if deformable_contact is None:
        dc = None
    elif "faces" in deformable_contact:                     # 3D (vertex-face + edge-edge)
        dc = _DistDeformableContact3D(deformable_contact, dof_per_node, ndof, U, comm)
    else:                                                   # 2D (node-to-segment)
        dc = _DistDeformableContact(deformable_contact, dof_per_node, ndof, U, comm)
    dpred = U.duplicate()                                   # predictor jump dt·v_prev (distributed)

    def gather_full(vec):                                   # COLLECTIVE — all ranks must call
        sc, seq = PETSc.Scatter.toAll(vec)
        sc.scatter(vec, seq, addv=INS, mode=FWD)
        out = np.asarray(seq.getArray()).copy()
        sc.destroy()
        seq.destroy()
        return out

    # constant Robin spring-dashpot, distributed OWNED-ROW (tangent K+C/dt is reference-config,
    # assembled once; the residual K·u + C·(u−u_prev)/dt needs the full u — boundary DOFs can
    # split across ranks — so it is gathered each iteration).
    rob_on = robin is not None
    if rob_on:
        rob_A = (robin.Kmat + robin.Cmat / dt).tocsr()
        rob_dofs = np.asarray(robin.dofs, dtype=np.int64)
        rob_rows_owned = rob_dofs[(rob_dofs >= rs) & (rob_dofs < re)].astype(int)
        rob_row_cols = {int(g): rob_A.indices[rob_A.indptr[g]:rob_A.indptr[g + 1]].astype(PETSc.IntType)
                        for g in rob_rows_owned}
        rob_row_vals = {int(g): rob_A.data[rob_A.indptr[g]:rob_A.indptr[g + 1]]
                        for g in rob_rows_owned}

    def assemble(uhat_own):
        Ua = ghost(U)
        A = PETSc.Mat().createAIJ([ndof, ndof], comm=comm)
        A.setPreallocationNNZ(dof_per_node * 18)
        A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        R = PETSc.Vec().createMPI(ndof, comm=comm)
        R.set(0.0)
        bulk_nan = False
        if my_ne:
            U_all = Ua[my_gm_local]
            R_all, K_all = batch_fn(my_coords, U_all, U_all)       # history-free bulk (DU unused)
            bulk_nan = not bool(np.all(np.isfinite(R_all)))        # diag: element inversion → log(J) NaN
            for li in range(my_ne):
                A.setValues(rows_all[li], rows_all[li], np.asarray(K_all[li], float), addv=ADD)
                R.setValues(rows_all[li], np.asarray(R_all[li], float), addv=ADD)
        # node-local inertia (+ Rayleigh damping) on owned inertial DOFs
        if len(inertial):
            U_own = np.asarray(U.getArray())
            r_in = (m_in / dt ** 2) * (U_own[inertial] - uhat_own[inertial])
            d_in = m_in / dt ** 2
            if damping:
                r_in = r_in + damping * m_in * (U_own[inertial] - u_prev[inertial]) / dt
                d_in = d_in + damping * m_in / dt
            R.setValues(inertial_g, r_in, addv=ADD)
            for gi, dv in zip(inertial_g, d_in):
                A.setValue(int(gi), int(gi), float(dv), addv=ADD)
        if f_own is not None:                                  # external force: residual −F_ext
            R.setValues(own, -f_own, addv=ADD)
        if rob_on:                                             # Robin spring-dashpot (owned rows)
            U_full = gather_full(U)                            # COLLECTIVE — all ranks
            r_rob = robin.Kmat @ U_full + robin.Cmat @ (U_full - robin.u_prev) / dt
            for g in rob_rows_owned:
                A.setValues([int(g)], rob_row_cols[int(g)],
                            rob_row_vals[int(g)].reshape(1, -1), addv=ADD)
                R.setValues([int(g)], [float(r_rob[g])], addv=ADD)
        if pressure is not None:                               # follower pressure (owned rows)
            U_full = gather_full(U)                            # COLLECTIVE — all ranks
            Rp = pressure.residual(U_full, None, 0.0, dt)      # O(surface), redundant on each rank
            Tp = pressure.tangent(U_full, None, 0.0, dt)
            rg = np.asarray(Rp.gdofs); rv = np.asarray(Rp.values)
            mr = (rg >= rs) & (rg < re)
            if mr.any():
                R.setValues(rg[mr].astype(PETSc.IntType), rv[mr], addv=ADD)
            tr = np.asarray(Tp.rows); tc = np.asarray(Tp.cols); tv = np.asarray(Tp.values)
            mt = (tr >= rs) & (tr < re)
            for r_, c_, v_ in zip(tr[mt], tc[mt], tv[mt]):
                A.setValue(int(r_), int(c_), float(v_), addv=ADD)
        if dc is not None:                                     # cross-rank deformable contact (COLLECTIVE)
            dc.assemble_into(A, R, U)
        if verbose:                                            # diag: localize a NaN (bulk vs contact)
            R.assemble()
            if not np.isfinite(R.norm()):
                print(f"    [NaN-DIAG] bulk_nan={bulk_nan} (False ⇒ contact/inertia/force is the source)",
                      flush=True)
        A.assemble()
        R.assemble()
        return A, R

    Uprev = U.duplicate()
    n_newton_last, rnorm_last, ksp_diverged = 0, 0.0, False
    for step in range(1, n_steps + 1):
        t = step * dt
        bc = dirichlet(t) if callable(dirichlet) else dict(dirichlet)
        U.copy(Uprev)
        uhat_own = u_prev + dt * v_prev                       # inertial predictor (owned, unbounded)
        # CCD-bound the PREDICTOR jump dt·v_prev from the last accepted state (barrier only)
        dpred.set(0.0)
        if len(v_prev):
            dpred.setValues(own, dt * v_prev, addv=INS)
        dpred.assemble()
        alpha = 1.0
        if dc is not None and dc.kind == "barrier":
            alpha = min(alpha, dc.ccd_alpha(Uprev, dpred))    # collective
        U.axpy(alpha, dpred)                                  # warm start = bounded predictor
        bc_all = np.array(sorted(bc), dtype=PETSc.IntType)
        owned_bc = [(g, bc[g]) for g in bc if rs <= g < re]
        if owned_bc:
            U.setValues([g for g, _ in owned_bc], [v for _, v in owned_bc], addv=INS)
        U.assemble()
        bc_owned = np.array([g for g, _ in owned_bc], dtype=PETSc.IntType)
        z_owned = np.zeros(len(bc_owned))
        for _it in range(max_newton):
            A, R = assemble(uhat_own)
            if len(bc_owned):
                R.setValues(bc_owned, z_owned, addv=INS)
            R.assemble()
            rnorm = R.norm()
            n_newton_last, rnorm_last = _it + 1, rnorm
            if rnorm < tol:
                A.destroy()
                R.destroy()
                break
            A.zeroRows(bc_all, diag=1.0)
            negR = R.copy()
            negR.scale(-1.0)
            du = R.duplicate()
            ksp = PETSc.KSP().create(comm)
            ksp.setOperators(A)
            ksp.setType(ksp_type)
            ksp.getPC().setType(pc)
            if pc == "lu":
                ksp.getPC().setFactorSolverType(solver)
            ksp.setTolerances(rtol=rtol, max_it=2000)
            ksp.solve(negR, du)
            if ksp.getConvergedReason() < 0:
                ksp_diverged = True
            # CCD step bound (barrier) — global min over ranks; then take the step (the inertia
            # regularizes; well-conditioned with F-bar bulk + matched κ → no line search needed).
            alpha = 1.0
            if dc is not None and dc.kind == "barrier":
                alpha = min(alpha, dc.ccd_alpha(U, du))       # collective
            U.axpy(alpha, du)
            if verbose and rank == 0:                         # per-iteration: residual + CCD throttle
                print(f"    [newton] step {step} it={_it + 1} |R|={rnorm:.3e} "
                      f"ccd_alpha={alpha:.3e}", flush=True)
            ksp.destroy()
            negR.destroy()
            du.destroy()
            A.destroy()
            R.destroy()
        # commit: backward-Euler velocity + advance contact friction state
        U_own = np.asarray(U.getArray())
        v_prev = (U_own - u_prev) / dt
        u_prev = U_own.copy()
        if dc is not None:
            dc.commit(U)                                      # collective
        if rob_on:                                            # advance Robin history (full u, rank-consistent)
            robin.u_prev = gather_full(U)
        if step_callback is not None:                         # per-step hook (e.g. min-gap tracking)
            step_callback(step, gather_full(U), rank)         # gather_full is COLLECTIVE — all ranks
        if verbose and rank == 0:
            print(f"  [dyn] step {step}: t={t:.3f} newton={n_newton_last} "
                  f"|R|={rnorm_last:.2e}", flush=True)

    return gather_full(U), dict(rank=rank, size=size, my_ne=my_ne, n_owned=re - rs,
                                n_newton=n_newton_last, rnorm=rnorm_last,
                                ksp_diverged=ksp_diverged)
