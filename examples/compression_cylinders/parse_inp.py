"""Minimal Abaqus .inp parser for the compression-of-cylinders deck.

Extracts, per instance: node coords, elements (by type), and elsets.  Enough to
rebuild the deformable cylinder pack + the rigid container/lid in CoupFE.
"""
from __future__ import annotations

import os

import numpy as np


_DECK_BASENAME = "xpl_2dgencont_compression.inp.txt"


def resolve_input_deck(path=None):
    """Resolve a user-supplied, lawfully obtained Abaqus input deck.

    The copyrighted deck is not part of the public distribution.  An explicit
    argument takes precedence over ``COUPFE_CYLINDERS_INP``. For backward
    compatibility, the repository-root basename is checked last.
    """

    candidate = path or os.environ.get("COUPFE_CYLINDERS_INP")
    if candidate is None:
        candidate = os.path.join(
            os.path.dirname(__file__), "..", "..", _DECK_BASENAME
        )
    resolved = os.path.abspath(os.path.expanduser(os.fspath(candidate)))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            "The compression-cylinders Abaqus deck is not distributed. "
            "Set COUPFE_CYLINDERS_INP to a lawfully obtained "
            f"{_DECK_BASENAME!r} file (looked for {resolved!r})."
        )
    return resolved


def parse_inp(path=None):
    path = resolve_input_deck(path)
    parts = {}                       # instance_name -> dict(nodes, elems, elsets)
    cur = None
    mode = None
    etype = None
    eset_name = None
    eset_gen = False
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            s = line.strip()
            if not s or s.startswith("**"):
                continue
            if s.startswith("*"):
                kw = s.split(",")[0].lower().strip()
                opts = {kv.split("=")[0].strip().lower(): kv.split("=")[1].strip()
                        for kv in s.split(",")[1:] if "=" in kv}
                mode = None
                if kw == "*instance":
                    cur = opts.get("name")
                    parts[cur] = dict(nodes={}, elems={}, elsets={})
                elif kw == "*end instance":
                    cur = None
                elif kw == "*node" and cur:
                    mode = "node"
                elif kw == "*element" and cur:
                    mode = "elem"; etype = opts.get("type")
                    parts[cur]["elems"].setdefault(etype, {})
                elif kw == "*elset" and cur:
                    mode = "elset"; eset_name = opts.get("elset")
                    eset_gen = ("generate" in s.lower())
                    parts[cur]["elsets"].setdefault(eset_name, [])
                continue
            if mode == "node" and cur:
                p = [x.strip() for x in s.split(",")]
                parts[cur]["nodes"][int(p[0])] = (float(p[1]), float(p[2]))
            elif mode == "elem" and cur:
                p = [int(x) for x in s.split(",") if x.strip()]
                parts[cur]["elems"][etype][p[0]] = p[1:]
            elif mode == "elset" and cur:
                vals = [int(x) for x in s.split(",") if x.strip()]
                if eset_gen and len(vals) == 3:
                    a, b, st = vals
                    parts[cur]["elsets"][eset_name].extend(range(a, b + 1, st))
                else:
                    parts[cur]["elsets"][eset_name].extend(vals)
    return parts


def connected_components(elems):
    """Union-find over elements sharing a node -> list of element-id groups."""
    parent = {}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        parent.setdefault(a, a); parent.setdefault(b, b)
        parent[find(a)] = find(b)
    node2elem = {}
    for eid, conn in elems.items():
        parent.setdefault(eid, eid)
        for n in conn:
            node2elem.setdefault(n, []).append(eid)
    for els in node2elem.values():
        for e in els[1:]:
            union(els[0], e)
    groups = {}
    for eid in elems:
        groups.setdefault(find(eid), []).append(eid)
    return list(groups.values())


if __name__ == "__main__":
    P = parse_inp()
    print("instances:", list(P.keys()))
    p1 = P["Part-1-1"]
    allelems = {}
    for et, d in p1["elems"].items():
        print(f"  Part-1 {et}: {len(d)} elements")
        allelems.update(d)
    print(f"  Part-1 nodes: {len(p1['nodes'])}")
    rubber = set(p1["elsets"].get("Set-1", []))
    steel = set(p1["elsets"].get("Set-2", []))
    print(f"  RUBBER elems: {len(rubber)}   STEEL elems: {len(steel)}")
    comps = connected_components(allelems)
    print(f"  CYLINDERS (connected components): {len(comps)}")
    xy = np.array(list(p1["nodes"].values()))
    print(f"  Part-1 bbox: x[{xy[:,0].min():.1f},{xy[:,0].max():.1f}] y[{xy[:,1].min():.1f},{xy[:,1].max():.1f}]")
    # per-cylinder radius estimate + material
    rads = []
    for g in comps:
        ns = set()
        for e in g:
            ns.update(allelems[e])
        c = np.array([p1["nodes"][n] for n in ns])
        ctr = c.mean(0); r = np.linalg.norm(c - ctr, axis=1).max()
        mat = "RUBBER" if g[0] in rubber else ("STEEL" if g[0] in steel else "?")
        rads.append((r, mat, len(g)))
    rads_arr = np.array([r[0] for r in rads])
    print(f"  cylinder radii: min {rads_arr.min():.2f}  max {rads_arr.max():.2f}  mean {rads_arr.mean():.2f}")
    nrub = sum(1 for _, m, _ in rads if m == "RUBBER")
    nste = sum(1 for _, m, _ in rads if m == "STEEL")
    print(f"  cylinders by material: RUBBER {nrub}, STEEL {nste}")
    for inst in ("Part-2-1", "Part-3-1"):
        nn = np.array(list(P[inst]["nodes"].values()))
        print(f"  {inst} (rigid): {len(nn)} nodes  bbox x[{nn[:,0].min():.1f},{nn[:,0].max():.1f}] y[{nn[:,1].min():.1f},{nn[:,1].max():.1f}]")
