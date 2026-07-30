"""Von Mises recovery for the RESEARCH tire workflow.

Reconstructs the Cauchy stress and its von Mises invariant **independently in Python** from the
input displacement (the element kernel computes stress internally but returns
only R/K). This supplies a separate post-processing implementation of the same
mixed-u-p law. It is not an independent formulation oracle or a GetFEM
comparison: `P = G(F−F⁻ᵀ) + p F⁻ᵀ`, `p = K·avg(lnJ)`
(element-constant), `σ = J⁻¹ P Fᵀ`, `σ_vm = √(3/2 s:s)`.
"""
from __future__ import annotations

import numpy as np

# Hex8 reference corners (±1), matching the mesh's bottom-face-then-top ordering, and the 8 Gauss points.
_XI = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
_GP = _XI / np.sqrt(3.0)


def _shape_grads(X, xi):
    """Return dN/dx (8,3) and detJ at natural coordinate xi for element node coords X (8,3)."""
    dNdxi = np.empty((8, 3))
    for a in range(8):
        for i in range(3):
            d = _XI[a, i]
            for j in range(3):
                if j != i:
                    d *= (1.0 + _XI[a, j] * xi[j])
            dNdxi[a, i] = 0.125 * d
    J = dNdxi.T @ X
    return dNdxi @ np.linalg.inv(J).T, np.linalg.det(J)


def element_vonmises(nodes, elems, U, G, K):
    """Per-element von Mises (averaged over Gauss points) on the deformed configuration."""
    u = U.reshape(len(nodes), 3)
    vm = np.zeros(len(elems))
    for ei, e in enumerate(elems):
        X, ue = nodes[e], u[e]
        Fs, dJs = [], []
        for gp in _GP:
            dNdx, detJ = _shape_grads(X, gp)
            Fs.append(np.eye(3) + ue.T @ dNdx)     # F_ij = δ_ij + Σ_a u_a,i dN_a/dx_j
            dJs.append(detJ)
        dJs = np.asarray(dJs)
        lnJ = np.array([np.log(np.linalg.det(F)) for F in Fs])
        p = K * np.sum(lnJ * dJs) / np.sum(dJs)    # element-constant condensed pressure
        vmg = []
        for F in Fs:
            Jdet = np.linalg.det(F)
            Finv = np.linalg.inv(F)
            P = G * (F - Finv.T) + p * Finv.T
            sigma = (P @ F.T) / Jdet               # Cauchy
            s = sigma - np.trace(sigma) / 3.0 * np.eye(3)
            vmg.append(np.sqrt(1.5 * np.sum(s * s)))
        vm[ei] = np.mean(vmg)
    return vm


def write_vtk(path, nodes, elems, U, cell_scalar, name="vonmises"):
    """Minimal legacy-VTK (ASCII) of the DEFORMED mesh + a per-cell scalar (for ParaView)."""
    pos = nodes + U.reshape(len(nodes), 3)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\ntire von mises\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(pos)} float\n")
        for x, y, z in pos:
            f.write(f"{x:.6e} {y:.6e} {z:.6e}\n")
        f.write(f"CELLS {len(elems)} {len(elems) * 9}\n")
        for e in elems:
            f.write("8 " + " ".join(map(str, e)) + "\n")
        f.write(f"CELL_TYPES {len(elems)}\n" + "12\n" * len(elems))
        f.write(f"CELL_DATA {len(elems)}\nSCALARS {name} float 1\nLOOKUP_TABLE default\n")
        for v in cell_scalar:
            f.write(f"{v:.6e}\n")
