"""Distributed contact broad-phase via **surface replication** — run under mpirun.

Distributed deformable contact couples surfaces by *spatial proximity*, so a contacting vertex and
edge can live on different ranks. The contact **surface** is tiny next to the bulk, so the robust,
standard approach at moderate scale is: **replicate the surface positions to every rank** (PETSc
`Scatter.toAll`), then each rank runs the serial spatial-hash broad phase (`candidate_pairs`) for the
**vertices it owns** against *all* edges — cross-rank candidate pairs fall out for free.

Gate (rank independence): on every rank the replicated surface positions match the truth to machine
precision, and each owned vertex's candidate edges equal the **serial** broad phase on the full
surface. So the union over ranks == the serial result, independent of rank count.

    OMP_NUM_THREADS=1 mpirun -n 4 python examples/mpi_smoke/distributed_broadphase.py

(For the real solver you replicate only the surface DOFs, not the whole vector; here the whole mesh
*is* the surface, so `toAll` is the surface replication.)
"""

from __future__ import annotations

import numpy as np

from coupfe.operators.contact_search import candidate_pairs


def main():
    from petsc4py import PETSc                       # petsc4py only — never mpi4py

    comm = PETSc.COMM_WORLD
    rank, size = comm.getRank(), comm.getSize()
    dim, band = 2, 0.05

    # Two parallel rows 0.03 apart (within `band`): top-row vertices contact bottom-row edges
    # (and vice versa) — many candidate pairs that span ranks once the nodes are distributed.
    nx = 16
    xs = np.linspace(0.0, 1.0, nx)
    X = np.vstack([np.stack([xs, np.zeros(nx)], 1),
                   np.stack([xs, 0.03 * np.ones(nx)], 1)])      # (2*nx, 2)
    N = len(X)
    edges = np.array([(i, i + 1) for i in range(nx - 1)]
                     + [(nx + i, nx + i + 1) for i in range(nx - 1)], dtype=int)
    ndof = N * dim

    serial = candidate_pairs(X, np.arange(N), edges, band)      # ground truth (full surface)

    # distributed position vector U (here U holds the positions directly)
    U = PETSc.Vec().createMPI(ndof, comm=comm)
    rs, re = U.getOwnershipRange()
    flat = X.ravel()
    for g in range(rs, re):
        U.setValue(g, float(flat[g]), PETSc.InsertMode.INSERT_VALUES)
    U.assemble()

    # replicate ALL surface DOFs to every rank → full positions on each rank
    sc, seq = PETSc.Scatter.toAll(U)
    sc.scatter(U, seq, addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
    pos_rep = seq.getArray().reshape(N, dim)

    ok_rep = bool(np.allclose(pos_rep, X, atol=1e-14))         # gate 1: replication exact

    # this rank's owned vertices (node owned by the rank owning its first DOF) — a partition
    owned = np.array([n for n in range(N) if rs <= n * dim < re], dtype=int)
    local = candidate_pairs(pos_rep, owned, edges, band)        # broad phase on replicated surface

    ok_match = True                                             # gate 2: owned candidates == serial
    for v in owned:
        got = set(local.get(int(v), np.array([], dtype=int)).tolist())
        truth = set(serial.get(int(v), np.array([], dtype=int)).tolist())
        if got != truth:
            ok_match = False
            break

    # completeness: total owned across ranks == N (PETSc reduction, no mpi4py)
    cnt = PETSc.Vec().createMPI((1, size), comm=comm)
    cnt.setValue(rank, float(len(owned)), PETSc.InsertMode.INSERT_VALUES)
    cnt.assemble()
    total_owned = int(round(cnt.sum()))
    ok_complete = (total_owned == N)

    ok = ok_rep and ok_match and ok_complete
    if rank == 0:
        n_cand = sum(len(c) for c in serial.values())
        print(f"[size={size}] replicate={ok_rep} owned_match={ok_match} "
              f"complete={ok_complete} (owned {total_owned}/{N}, {n_cand} serial pairs) "
              f"-> {'OK' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
