"""Exact point-triangle closest point, including obtuse triangles (numpy and numba).

The closest-feature search used to project onto the edge opposite the first negative
plane weight, which is not always the closest edge on an obtuse triangle.
"""

import itertools

import numpy as np
import pytest

from coupfe.operators import contact3d as c3


def triangle_reference(p, tri):
    """Independent constrained least-squares reference over all simplex faces.

    Enumerate vertices, segments and the face; SVD solves each affine problem,
    rather than reusing the production projection or feature classifier.
    """
    candidates = []
    for size in (1, 2, 3):
        for ids in itertools.combinations(range(3), size):
            base = tri[ids[0]]
            if size == 1:
                candidates.append(base)
                continue
            basis = (tri[list(ids[1:])] - base).T
            weights, _, rank, _ = np.linalg.lstsq(basis, p - base, rcond=None)
            if rank < size - 1:
                continue
            bary = np.r_[1.0 - weights.sum(), weights]
            if bary.min() >= 0.0:
                candidates.append(base + basis @ weights)
    return min(candidates, key=lambda q: np.linalg.norm(p - q))


def triangle_cases():
    rng = np.random.default_rng(290_1026)
    yield np.array([.5, -1., 0.]), np.array([[0., 0., 0.], [1., 0., 0.], [2., .1, 0.]])
    for i in range(450):
        tri = rng.normal(size=(3, 3))
        if i % 3 == 0:                   # deliberately obtuse / near-degenerate
            tri[2] = tri[0] + 2.0 * (tri[1] - tri[0]) + 10.**(-1 - i % 12) * rng.normal(size=3)
        if i % 31 == 0:
            tri[2] = tri[1]              # repeated vertices
        if i % 73 == 0:
            tri[:] = tri[0]             # collapsed triangle
        yield rng.normal(size=3), tri
    # Interior projections onto very thin triangles (including below Gram precision).
    for height in (1.e-4, 1.e-8, 1.e-12, 0.):
        tri = np.array([[0., 0., 0.], [1., 0., 0.], [2., height, 0.]])
        yield np.array([.7, .2 * height, .3]), tri


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_point_triangle_exact_randomized(backend):
    coeff = c3.point_triangle_coeff_unclassified
    if backend == 'numba':
        pytest.importorskip('numba')
        from coupfe.operators.contact3d_numba import _pt_coeff_unclassified
        coeff = _pt_coeff_unclassified
    for p, tri in triangle_cases():
        w = np.asarray(coeff(p, *tri))
        assert np.isfinite(w).all() and w.min() >= 0.
        np.testing.assert_allclose(w.sum(), 1., atol=2.e-15)
        ref = triangle_reference(p, tri)
        np.testing.assert_allclose(np.linalg.norm(p - w @ tri), np.linalg.norm(p - ref),
                                   rtol=2.e-11, atol=2.e-12)


def test_obtuse_counterexample():
    p, tri = next(triangle_cases())
    w = c3.point_triangle_coeff_unclassified(p, *tri)
    np.testing.assert_array_equal(w, [.5, .5, 0.])
    assert np.linalg.norm(p - w @ tri) == 1.


def test_corrected_closest_face_and_ccd_cpu():
    p, tri = next(triangle_cases())
    other = np.array([[-2., -2., 1.05], [3., -2., 1.05], [.5, 3., 1.05]])
    nodes = np.vstack([p, tri, other])
    faces = np.array([[1, 2, 3], [4, 5, 6]])
    op = c3.DeformableBarrierContact3D(nodes, [0], faces, np.empty((0, 2), dtype=int), dhat=2.)
    assert op._closest_face(nodes, 0, [1, 0]) == (1., 0)
    # The swept point reaches AB at t=.5; ACCD must stop safely before it.
    p1 = p + np.array([0., 2., 0.])
    assert 0. < c3.point_triangle_toi(p, p1, *tri, *tri) < .5

