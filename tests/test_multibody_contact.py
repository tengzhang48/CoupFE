"""Multi-body mutual contact: no spurious rest force (the 16-disk-divergence bug).

`DeformableBarrierContact2D` is a node-to-segment barrier whose signed gap
`d=(e×r)/L` is `>0` when the secondary is on the LEFT of the edge `a→b`.  For
N-body mutual contact (one operator over the union of all body boundaries), every
body's loop must be **consistently oriented** (outside-on-left) — otherwise a node
sees a *negative* gap to a wrongly-wound edge and the barrier reads a spurious deep
penetration **at the undeformed rest state**, which makes the solve diverge.

`build_model.boundary_edges` now orients each body's edges.  This test gates that
separated bodies produce **zero** contact force at rest, with a broken control
(flipped winding) that must produce a large spurious force.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from coupfe.operators.contact import DeformableBarrierContact2D

_DIR = Path(__file__).parent.parent / "examples" / "compression_cylinders"


def _bm():
    sys.path.insert(0, str(_DIR))          # build_model does `from parse_inp import ...`
    spec = importlib.util.spec_from_file_location("cc_build_model", _DIR / "build_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _disks(centers, R=2.0):
    bm = _bm()
    an, be, body = [], [], []
    off = 0
    for bi, c in enumerate(centers):
        nd, q = bm.disk_mesh(c, R, n=3)
        e = bm.boundary_edges(q, nd) + off
        an.append(nd); be += e.tolist(); body += [bi] * len(nd); off += len(nd)
    nodes = np.vstack(an)
    be = np.array(be, int)
    bn = np.array(sorted(set(be.ravel().tolist())), int)
    return nodes, bn, be, np.array(body)


def _force(nodes, bn, be, dhat=0.5, **kw):
    ct = DeformableBarrierContact2D(nodes, bn, be, dof_per_node=2, comps=(0, 1),
                                    dhat=dhat, kappa=100.0, **kw)
    R = ct.residual(np.zeros(len(nodes) * 2), None, 1.0, 1.0)
    r = np.zeros(len(nodes) * 2)
    np.add.at(r, R.gdofs, R.values)
    return float(np.max(np.abs(r)))


def test_separated_no_spurious_rest_force():
    """Separated disks (gaps > dhat, oriented edges) ⇒ zero contact force at rest."""
    nodes, bn, be, _ = _disks([(0.0, 0.0), (6.0, 0.0), (3.0, 5.0)])   # gaps ~1.8–2.0
    assert _force(nodes, bn, be) < 1e-9


def test_orientation_is_load_bearing():
    """Two disks in contact (gap ≈ 0.3 < dhat): correct winding gives a moderate
    REPULSIVE force; flipping one disk's winding flips the signed gap (+0.3 → −0.3)
    so the barrier reads penetration and the force balloons — the bug, and proof the
    orientation in `boundary_edges` is load-bearing (a non-vacuous broken control)."""
    centers = [(0.0, 0.0), (4.3, 0.0)]                   # R=2 each → gap ≈ 0.3
    nodes, bn, be, body = _disks(centers)
    f_ok = _force(nodes, bn, be)
    be_bad = be.copy()
    be_bad[body[be_bad[:, 0]] == 1] = be_bad[body[be_bad[:, 0]] == 1][:, ::-1]  # flip disk 1
    f_bad = _force(nodes, bn, be_bad)
    assert f_ok > 1e-6, "expected real (repulsive) contact at gap<dhat"
    assert f_bad > 3.0 * f_ok, f"flipped winding should inflate the force: {f_bad} vs {f_ok}"


def test_body_id_excludes_self_contact():
    """The defensive same-body exclusion also yields zero rest force when separated."""
    nodes, bn, be, body = _disks([(0.0, 0.0), (6.0, 0.0), (3.0, 5.0)])
    assert _force(nodes, bn, be, body_id=body) < 1e-9
