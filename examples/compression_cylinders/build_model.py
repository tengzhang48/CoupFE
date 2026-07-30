"""Build a CoupFE compaction model from the Abaqus deck's GEOMETRY.

The raw Abaqus mesh is mixed quad+tri plane-stress; CoupFE has clean Quad4.
So we take A's *geometry* (per-cylinder centroid, radius, material; container;
lid; loading) and regenerate each cylinder as a clean all-quad disk (squircle
map of a square grid -> no singular centre).  Same pack, comparable physics.
"""
from __future__ import annotations

import os

import numpy as np

from parse_inp import connected_components, parse_inp


def disk_mesh(center, R, n=4):
    """All-quad disk of radius R at center via the FG-squircle map of [-1,1]^2."""
    cx, cy = center
    us = np.linspace(-1, 1, n + 1)
    nodes = []
    for v in us:
        for u in us:
            x = u * np.sqrt(max(1 - 0.5 * v * v, 0.0))
            y = v * np.sqrt(max(1 - 0.5 * u * u, 0.0))
            nodes.append([cx + R * x, cy + R * y])
    nodes = np.array(nodes)
    nn = n + 1
    quads = []
    for j in range(n):
        for i in range(n):
            a = j * nn + i
            quads.append([a, a + 1, a + nn + 1, a + nn])
    return nodes, np.array(quads, int)


def boundary_edges(quads, nodes):
    """Free (boundary) edges of a body, **oriented so the body's OUTSIDE is on the
    LEFT of a→b**.  The node-to-segment barrier's signed gap is ``d=(e×r)/L`` (>0 when
    the secondary is on the left of the edge), so for N-body mutual contact every
    body's loop must wind consistently — otherwise a node sees a *negative* gap to a
    wrongly-wound edge and the barrier reads a spurious deep penetration (this was the
    16-disk non-convergence). ``nodes`` is the body's node coords (for its centroid)."""
    from collections import Counter
    cnt = Counter()
    for q in quads:
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            cnt[(min(a, b), max(a, b))] += 1
    nodes = np.asarray(nodes)
    C = nodes.mean(0)
    out = []
    for (a, b), k in cnt.items():
        if k != 1:
            continue
        m = 0.5 * (nodes[a] + nodes[b])
        d = m - C
        e = nodes[b] - nodes[a]
        if e[0] * d[1] - e[1] * d[0] < 0.0:          # outside not on left → swap
            a, b = b, a
        out.append([a, b])
    return np.array(out, int)


def extract_geometry(inp_path=None):
    """Return ``(cylinders, container_box, params)`` from a supplied deck.

    ``parse_inp`` resolves ``COUPFE_CYLINDERS_INP`` when ``inp_path`` is
    omitted. Cylinders are dictionaries with ``center``, ``R``, and
    ``material``; ``container_box`` is ``(xL, xR, yB)``.
    """
    P = parse_inp(inp_path)
    p1 = P["Part-1-1"]
    alle = {**p1["elems"].get("CPS4R", {}), **p1["elems"].get("CPS3", {})}
    rubber = set(p1["elsets"].get("Set-1", []))
    comps = connected_components(alle)
    cyls = []
    for g in comps:
        ns = set()
        for e in g:
            ns.update(alle[e])
        c = np.array([p1["nodes"][k] for k in ns])
        ctr = c.mean(0)
        R = np.linalg.norm(c - ctr, axis=1).max()
        mat = "RUBBER" if g[0] in rubber else "STEEL"
        cyls.append(dict(center=ctr, R=float(R), material=mat))
    cont = np.array(list(P["Part-2-1"]["nodes"].values()))
    lid = np.array(list(P["Part-3-1"]["nodes"].values()))
    box = (float(cont[:, 0].min()), float(cont[:, 0].max()), float(cont[:, 1].min()))
    params = dict(
        # Mooney-Rivlin C10=C01=4.48632, D1=0.02229 -> neo-Hookean approx:
        #   small-strain shear G = 2(C10+C01); K = 2/D1
        G_rubber=2.0 * (4.48632 + 4.48632), K_rubber=2.0 / 0.02229,
        E_steel=70000.0, nu_steel=0.3,
        mu_fric=0.1, grav=9801.0, lid_disp=-54.0, lid_y0=float(lid[:, 1].mean()),
    )
    return cyls, box, params


if __name__ == "__main__":
    cyls, box, params = extract_geometry()
    print(f"cylinders: {len(cyls)} (RUBBER {sum(c['material']=='RUBBER' for c in cyls)}, "
          f"STEEL {sum(c['material']=='STEEL' for c in cyls)})")
    print(f"container box: xL={box[0]} xR={box[1]} yB={box[2]};  lid y0={params['lid_y0']}")
    print(f"neo-Hookean rubber: G={params['G_rubber']:.2f} K={params['K_rubber']:.1f}; "
          f"steel E={params['E_steel']}; mu={params['mu_fric']}")
    # sanity: mesh the median-radius disk, check valid + boundary
    R = float(np.median([c["R"] for c in cyls]))
    nodes, quads = disk_mesh((0, 0), R, n=4)
    be = boundary_edges(quads, nodes)
    print(f"sample disk R={R:.2f}: {len(nodes)} nodes, {len(quads)} quads, {len(be)} boundary edges")
    # crude positive-area check on the quads
    def area(q):
        p = nodes[q]
        return 0.5 * abs(np.cross(p[2] - p[0], p[3] - p[1]))
    areas = np.array([area(q) for q in quads])
    print(f"  quad areas: min {areas.min():.3f} max {areas.max():.3f} (all>0: {areas.min()>0})")
