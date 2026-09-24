"""Generated axisymmetric elements and the axisymmetric cavity/contact operators.

Elements are ``coupfe.codegen`` declarations generated for the native ABI with
the ``*_axi`` configurations (``examples/axisymmetric_locking/kernels.py``).

Independent oracles: homogeneous (patch) states with closed-form stresses, the
uniaxial response from a one-dimensional root solve, and inflation of a thick
spherical shell against a spherically symmetric boundary-value solve
(``scipy.integrate.solve_bvp``) or the incompressible closed form. Consistency
checks: generated tangents against central finite differences. Broken
control: a plane-strain kernel fails the axisymmetric patch test.
"""

import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pytest
from scipy.optimize import brentq

pytest.importorskip("sympy")

from coupfe import ElementGroup, assemble_tangent, newton_solve  # noqa: E402
from coupfe.operators.axisymmetric import (  # noqa: E402
    _EDGES,
    AxisymmetricCavity,
    AxisymmetricContact,
    FluidLaw,
    boundary_edges,
    ring_areas,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "axisymmetric_locking"))
import kernels  # noqa: E402
from shell import (  # noqa: E402
    bvp_inner_radius,
    edge_family,
    incompressible_pressure,
    mapped_block,
    principal_pk1,
    shell_mesh,
    solve_inflation,
)

CASES = [("quad4", "standard"), ("quad4", "fbar"), ("quad8", "standard"), ("quad8r", "standard"),
         ("quad8", "mixed"), ("tri3", "standard"), ("tri6", "standard"), ("tri6", "mixed")]
PROPS = (0.5, 0.05, 0.01, 50.0)

try:  # a compiled kernel is required; skip cleanly without a Fortran toolchain
    kernels.kernel("quad4", "standard")
    _HAVE_KERNELS, _WHY = True, ""
except Exception as exc:  # pragma: no cover - environment-dependent
    _HAVE_KERNELS, _WHY = False, str(exc)

pytestmark = pytest.mark.skipif(
    not _HAVE_KERNELS, reason=f"generated kernel did not compile (needs numpy.f2py + gfortran): {_WHY}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def distorted_block(element, nr=3, nz=3, R1=2.0, Z1=1.5, seed=1):
    """Straight-edged block on the axis with randomly moved interior corner nodes."""
    X, cells = mapped_block(element, np.linspace(0.0, R1, nr + 1), np.linspace(0.0, Z1, nz + 1))
    rng = np.random.default_rng(seed)
    nc = kernels.CORNERS[element]
    corners = np.unique(cells[:, :nc])
    inner = corners[(X[corners, 0] > 1e-12) & (X[corners, 0] < R1 - 1e-12)
                    & (X[corners, 1] > 1e-12) & (X[corners, 1] < Z1 - 1e-12)]
    X = X.copy()
    X[inner] += 0.15 * np.array([R1 / nr, Z1 / nz]) * rng.uniform(-1, 1, (len(inner), 2))
    if element in kernels.QUADRATIC:            # keep edges straight: re-centre mid nodes
        for c in cells:
            for loc in _EDGES[edge_family(element)]:
                X[c[loc[2]]] = 0.5 * (X[c[loc[0]]] + X[c[loc[1]]])
    return X, cells


def assembled(op, U, ndof):
    if isinstance(op, ElementGroup):
        op._rk_cache = None      # every probe is a new state
    R = np.zeros(ndof)
    r = op.residual(U, None, 1.0, 1.0)
    np.add.at(R, r.gdofs, r.values)
    return R


def homogeneous_state(X, dpn, a, b, props):
    """``u_r = a r``, ``u_z = b z`` and, for mixed layouts, ``p = K (J - 1)``."""
    U = np.zeros(dpn * len(X))
    U[0::dpn], U[1::dpn] = a * X[:, 0], b * X[:, 1]
    if dpn == 3:
        U[2::dpn] = props[3] * ((1 + a) ** 2 * (1 + b) - 1.0)
    return U


def uniaxial_stretch(props, lam_z):
    """Lateral stretch and nominal axial stress for stress-free sides."""
    lr = brentq(lambda s: principal_pk1(props, s, lam_z, s)[0], 0.3, 3.0, xtol=1e-15, rtol=1e-15)
    return lr, principal_pk1(props, lr, lam_z, lr)[1]


def neo_hookean(G, K):
    return (0.5 * G, 0.0, 0.0, K)


# ---------------------------------------------------------------------------
# generation and kernels
# ---------------------------------------------------------------------------

def test_axisymmetric_generation_guards(tmp_path):
    from coupfe.codegen.generators.uel_gen import generate_element
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(NotImplementedError):
            generate_element(kernels.weak_form("quad8", "standard"), str(tmp_path / "a.for"),
                             element="quad8_axi", formulation="standard", backend="abaqus_uel")
        with pytest.raises(NotImplementedError):
            generate_element(kernels.weak_form("quad4", "standard"), str(tmp_path / "b.for"),
                             element="quad4_axi", formulation="local_pressure")
    with pytest.raises(ValueError):
        kernels.weak_form("quad4", "mixed")       # equal-order mixed pairs are not offered


@pytest.mark.parametrize("element,formulation", CASES)
def test_generated_kernel_homogeneous_patch(element, formulation):
    X, cells = distorted_block(element)
    grp = kernels.element_group(X, cells, element, formulation, PROPS)
    dpn = kernels.layout(element, formulation)
    a, b = 0.07, -0.12
    U = homogeneous_state(X, dpn, a, b, PROPS)
    R = assembled(grp, U, U.size)
    interior = np.setdiff1d(np.arange(len(X)), np.unique(boundary_edges(cells, edge_family(element))))
    rows = np.concatenate([dpn * interior, dpn * interior + 1])
    if dpn == 3:   # every pressure equation vanishes at p = K (J - 1)
        corners = np.unique(cells[:, :kernels.CORNERS[element]])
        rows = np.concatenate([rows, 3 * corners + 2])
    assert np.abs(R[rows]).max() < 1e-11
    top = np.nonzero(np.isclose(X[:, 1], 1.5))[0]
    Pzz = principal_pk1(PROPS, 1 + a, 1 + b, 1 + a)[1]
    assert np.isclose(R[dpn * top + 1].sum(), Pzz * np.pi * 2.0 ** 2, rtol=1e-11)


@pytest.mark.parametrize("element,formulation", CASES)
def test_generated_tangent_matches_finite_difference(element, formulation):
    X, cells = distorted_block(element, nr=2, nz=2)
    grp = kernels.element_group(X, cells, element, formulation, PROPS)
    dpn = kernels.layout(element, formulation)
    rng = np.random.default_rng(3)
    U = homogeneous_state(X, dpn, 0.02, -0.03, PROPS) + 0.02 * rng.standard_normal(dpn * len(X))
    U[0::dpn][X[:, 0] == 0.0] = 0.0
    written = np.unique(grp.gm)
    grp._rk_cache = None
    K = assemble_tangent([grp], U, None, 1.0, 1.0, U.size).toarray()[np.ix_(written, written)]
    Kfd = np.empty_like(K)
    h = 1e-6
    for jj, j in enumerate(written):
        e = np.zeros(U.size)
        e[j] = h
        Kfd[:, jj] = (assembled(grp, U + e, U.size) - assembled(grp, U - e, U.size))[written] / (2 * h)
    assert np.abs(K - Kfd).max() < 1e-7 * np.abs(K).max()
    asym = np.abs(K - K.T).max() / np.abs(K).max()
    assert (asym > 1e-3) if formulation == "fbar" else (asym < 1e-12)


def test_plane_strain_kernel_fails_axisymmetric_patch():
    # Broken control: the same declaration generated without axisymmetry. A
    # homogeneous state with F_rr = F_tt is in equilibrium under both
    # kinematics, so the discriminating invariant is the reaction: the ring
    # integral gives P_zz * pi R^2, plane strain P_zz * R per unit thickness.
    from coupfe.codegen.generators.uel_gen import generate_element
    from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel
    X, cells = distorted_block("quad4")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "plane.for")
        with contextlib.redirect_stdout(io.StringIO()):
            generate_element(kernels.weak_form("quad4", "standard"), path, element="quad4",
                             formulation="standard", backend="native")
        module = build_element_kernel(path, "axi_broken_control_plane_q4", workdir=tmp)
        plane = ElementGroup(CompiledElement(module, props=np.array(PROPS), dof_per_node=2, mcrd=2,
                                             n_elem=len(cells)), X, cells, 2)
        axi = kernels.element_group(X, cells, "quad4", "standard", PROPS)
        a, b = 0.07, -0.12
        U = homogeneous_state(X, 2, a, b, PROPS)
        top = np.nonzero(np.isclose(X[:, 1], 1.5))[0]
        exact = principal_pk1(PROPS, 1 + a, 1 + b, 1 + a)[1] * np.pi * 2.0 ** 2
        good = assembled(axi, U, U.size).reshape(-1, 2)[top, 1].sum() / exact - 1.0
        bad = assembled(plane, U, U.size).reshape(-1, 2)[top, 1].sum() / exact - 1.0
    assert abs(good) < 1e-11 < 0.1 < abs(bad)


@pytest.mark.parametrize("element,formulation", CASES)
def test_uniaxial_compression_solve(element, formulation):
    X, cells = distorted_block(element)
    props = (0.5, 0.02, 0.0, 100.0)
    grp = kernels.element_group(X, cells, element, formulation, props)
    dpn = kernels.layout(element, formulation)
    ndof = dpn * len(X)
    lam = 0.7
    bc = {dpn * k: 0.0 for k in np.nonzero(X[:, 0] == 0.0)[0]}
    bc.update({dpn * k + 1: 0.0 for k in np.nonzero(np.isclose(X[:, 1], 0.0))[0]})
    bc.update({g: 0.0 for g in kernels.unused_pressure_dofs(cells, element, formulation, len(X))})
    top = np.nonzero(np.isclose(X[:, 1], 1.5))[0]
    U = np.zeros(ndof)
    for s in range(1, 7):  # 5% increments; the tangent predictor keeps Newton in its basin
        d = dict(bc)
        d.update({dpn * k + 1: (lam - 1) * 1.5 * s / 6 for k in top})
        U, _, nit = newton_solve([grp], U, None, ndof, d, rtol=1e-11, atol=1e-12,
                                 line_search="admissible", predictor="tangent")
        assert nit <= 10
    lr, Pzz = uniaxial_stretch(props, lam)
    R = assembled(grp, U, ndof)
    assert np.isclose(R[dpn * top + 1].sum(), Pzz * np.pi * 2.0 ** 2, rtol=1e-9)
    assert np.allclose(U[0::dpn], (lr - 1) * X[:, 0], atol=1e-10)


# ---------------------------------------------------------------------------
# thick-shell inflation (independent references)
# ---------------------------------------------------------------------------

def test_sphere_bvp_reference_matches_incompressible_limit():
    G, A, B, p = 1.0, 1.0, 2.0, 0.5
    a = bvp_inner_radius(neo_hookean(G, 1e3 * G), A, B, p)
    # O(G/K) compressibility correction: about 1e-3 at K/G = 1000
    assert np.isclose(incompressible_pressure(G, A, B, a), p, rtol=3e-3)


@pytest.mark.parametrize("element,formulation", [("quad4", "fbar"), ("quad8", "standard"),
                                                 ("quad8", "mixed"), ("tri6", "standard"),
                                                 ("tri6", "mixed")])
def test_sphere_inflation_matches_bvp(element, formulation):
    props = neo_hookean(1.0, 1000.0)
    a_ref = bvp_inner_radius(props, 1.0, 2.0, 0.6)
    out = solve_inflation(element, formulation, props, 0.6)
    assert abs((out["a"] - 1.0) / (a_ref - 1.0) - 1.0) < 0.02
    assert out["cavity"].volume(out["U"]) > out["cavity"].reference_volume


def test_sphere_inflation_converges_with_refinement():
    props = neo_hookean(1.0, 1000.0)
    a_ref = bvp_inner_radius(props, 1.0, 2.0, 0.4)
    errors = [abs(solve_inflation("quad4", "fbar", props, 0.4, n_rho=n, n_theta=n, steps=6)["a"] - a_ref)
              for n in (3, 6, 12)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] / errors[2] > 2.5


def test_standard_linear_elements_lock_and_fbar_and_mixed_do_not():
    props = neo_hookean(1.0, 1.0e4)
    a_ref = brentq(lambda a: incompressible_pressure(1.0, 1.0, 2.0, a) - 0.6, 1.0 + 1e-9, 1.8)
    for element in ("quad4", "tri3"):
        assert solve_inflation(element, "standard", props, 0.6)["a"] - 1.0 < 0.3 * (a_ref - 1.0)
    for element, formulation in (("quad4", "fbar"), ("quad8", "mixed"), ("tri6", "mixed")):
        a = solve_inflation(element, formulation, props, 0.6)["a"]
        assert abs((a - 1.0) / (a_ref - 1.0) - 1.0) < 0.03


# ---------------------------------------------------------------------------
# boundary helpers and cavity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("element", ["quad4", "quad8", "tri3", "tri6"])
def test_ring_areas_and_boundary_edges(element):
    X, cells = mapped_block(element, np.linspace(0.5, 2.0, 4), np.linspace(0.0, 1.5, 3))
    edges = boundary_edges(cells, element)
    A = ring_areas(X, edges)
    exact = 2 * np.pi * (0.5 + 2.0) * 1.5 + 2 * np.pi * (2.0 ** 2 - 0.5 ** 2)
    assert np.isclose(A.sum(), exact, rtol=1e-12)
    # counterclockwise: the outward normal (edge rotated by -90 deg) points away from the block
    mid = X[edges[:, :2]].mean(axis=1)
    e = X[edges[:, 1]] - X[edges[:, 0]]
    n = np.stack([e[:, 1], -e[:, 0]], axis=1)
    assert ((mid - X.mean(axis=0)) * n).sum(1).min() > 0


def test_cavity_volume_gradient_and_hessian():
    X, cells, cavity = shell_mesh("quad8", 1.0, 2.0, 2, 3)
    cav = AxisymmetricCavity(X, cavity, 2 * len(X))
    assert np.isclose(cav.reference_volume, 2 / 3 * np.pi, rtol=1e-3)   # 3 quadratic arcs
    rng = np.random.default_rng(2)
    U = np.concatenate([0.05 * rng.standard_normal(2 * len(X)), [0.3]])
    rows, g = cav.volume_gradient(U)
    grad = np.zeros(U.size)
    np.add.at(grad, rows, g)
    h = 1e-6
    for j in np.unique(rows)[:8]:
        e = np.zeros(U.size)
        e[j] = h
        assert np.isclose((cav.volume(U + e) - cav.volume(U - e)) / (2 * h), grad[j], rtol=1e-7, atol=1e-12)
    cav.set_fluid(FluidLaw(cav.volume(U), 0.2, compressibility=0.05))
    K = assemble_tangent([cav], U, None, 1.0, 1.0, U.size).toarray()
    Kfd = np.empty_like(K)
    for j in range(U.size):
        e = np.zeros(U.size)
        e[j] = h
        Kfd[:, j] = (assembled(cav, U + e, U.size) - assembled(cav, U - e, U.size)) / (2 * h)
    assert np.abs(K - Kfd).max() < 1e-7 * np.abs(K).max()


def test_incompressible_fill_conserves_volume_and_raises_pressure():
    X, cells, cavity = shell_mesh("quad4", 1.0, 1.6, 3, 6)
    nn = len(X)
    pdof, ndof = 2 * nn, 2 * nn + 1
    solid = kernels.element_group(X, cells, "quad4", "fbar", neo_hookean(1.0, 200.0))
    cav = AxisymmetricCavity(X, cavity, pdof, scale=10.0)
    bc = {2 * k: 0.0 for k in np.nonzero(X[:, 0] == 0.0)[0]}
    bc.update({2 * k + 1: 0.0 for k in np.nonzero(X[:, 1] == 0.0)[0]})
    kw = dict(rtol=1e-11, atol=1e-13, line_search="admissible", predictor="tangent")
    U, _, _ = newton_solve([solid, cav], np.zeros(ndof), None, ndof, {**bc, pdof: 0.2}, **kw)
    V1 = cav.volume(U)
    cav.set_fluid(FluidLaw(V1, 0.2, compressibility=0.0))
    pole = np.nonzero((X[:, 0] == 0.0) & np.isclose(X[:, 1], 1.6))[0]
    for s in range(1, 5):   # push the outer pole down while the fill is sealed
        d = dict(bc)
        d.update({2 * k + 1: -0.03 * s for k in pole})
        U, _, _ = newton_solve([solid, cav], U, None, ndof, d, **kw)
    assert np.isclose(cav.volume(U), V1, rtol=1e-10)
    assert U[pdof] > 0.2


# ---------------------------------------------------------------------------
# contact
# ---------------------------------------------------------------------------

def _two_cylinders(element, nr_a=3, nr_b=3):
    Xa, ca = mapped_block(element, np.linspace(0, 1, nr_a + 1), np.linspace(0, 1, 4))
    Xb, cb = mapped_block(element, np.linspace(0, 1, nr_b + 1), np.linspace(1, 2, 3))
    X = np.vstack([Xa, Xb])
    cb = cb + len(Xa)
    ea, eb = boundary_edges(ca, element), boundary_edges(cb, element)
    top_a = ea[np.isclose(X[ea[:, :2], 1], 1.0).all(1)]
    bot_b = eb[np.isclose(X[eb[:, :2], 1], 1.0).all(1)]
    return X, ca, cb, top_a, bot_b


def _compress_stack(element, nr_b, augment_steps, freeze=False):
    X, ca, cb, top_a, bot_b = _two_cylinders(element, nr_b=nr_b)
    props = (0.5, 0.0, 0.0, 50.0)
    formulation = "mixed" if element == "quad8" else "fbar"
    dpn = kernels.layout(element, formulation)
    ops = [kernels.element_group(X, ca, element, formulation, props),
           kernels.element_group(X, cb, element, formulation, props)]
    contact = AxisymmetricContact(X, top_a, bot_b, penalty=200.0, dof_per_node=dpn)
    ops.append(contact)
    ndof = dpn * len(X)
    bc = {dpn * k: 0.0 for k in np.nonzero(X[:, 0] == 0.0)[0]}
    bc.update({dpn * k + 1: 0.0 for k in np.nonzero(X[:, 1] == 0.0)[0]})
    bc.update({g: 0.0 for g in kernels.unused_pressure_dofs(np.vstack([ca, cb]), element, formulation,
                                                             len(X))})
    top = np.nonzero(np.isclose(X[:, 1], 2.0))[0]
    lam = 0.85
    U = np.zeros(ndof)
    kw = dict(rtol=1e-12, atol=1e-13, line_search="admissible", predictor="tangent")
    for s in range(1, 5):
        d = dict(bc)
        d.update({dpn * k + 1: 2.0 * (lam - 1) * s / 4 for k in top})
        if freeze:
            contact.freeze(U)
        U, _, _ = newton_solve(ops, U, None, ndof, d, **kw)
    forces = [assembled(ops[1], U, ndof)[dpn * top + 1].sum()]
    for _ in range(augment_steps):                      # augmented-Lagrangian iterations
        contact.augment(U)
        U, _, _ = newton_solve(ops, U, None, ndof, d, **kw)
        forces.append(assembled(ops[1], U, ndof)[dpn * top + 1].sum())
    _, Pzz = uniaxial_stretch(props, lam)
    return np.array(forces), Pzz * np.pi, contact, U, ndof, dpn


@pytest.mark.parametrize("element", ["quad4", "quad8"])
def test_stacked_cylinders_contact_reproduces_uniaxial_state(element):
    # conforming interface: one-pass node-to-segment passes the contact patch test
    forces, exact, contact, U, ndof, dpn = _compress_stack(element, nr_b=3, augment_steps=10)
    _, g = contact.gaps(U)
    assert abs(forces[-1] / exact - 1) < 1e-8 < abs(forces[0] / exact - 1)
    assert -g.min() < 1e-8
    Rc = assembled(contact, U, ndof)
    assert abs(Rc[1::dpn].sum()) < 1e-12 * abs(exact)       # equal and opposite


def test_nonmatching_interface_keeps_equilibrium_with_frozen_pairing():
    # non-matching meshes: node-to-segment does not transmit a uniform pressure
    # exactly (Taylor & Papadopoulos 1991); equilibrium and near-exact force remain
    forces, exact, contact, U, ndof, dpn = _compress_stack("quad4", nr_b=2, augment_steps=10, freeze=True)
    Rc = assembled(contact, U, ndof)
    assert abs(Rc[1::dpn].sum()) < 1e-12 * abs(exact)
    assert abs(forces[-1] / exact - 1) < 2e-2


def test_frozen_pairing_hysteresis():
    X, ca, cb, top_a, bot_b = _two_cylinders("quad4", nr_b=2)
    contact = AxisymmetricContact(X, top_a, bot_b, penalty=1.0, switch_margin=0.1)
    U = np.zeros(2 * len(X))
    first = contact.freeze(U)
    assert first == len(contact.secondary)                      # first freeze pairs every node
    seg = contact.frozen.copy()
    # move the shared primary vertex slightly: projections stay within the margin
    V = U.copy()
    shared = contact.segments[0, 1]
    V[2 * shared] += 0.02
    assert contact.freeze(V) == 0 and np.array_equal(contact.frozen, seg)
    contact.release()
    assert contact.frozen is None



def test_smoothed_contact_is_c1_and_exact_beyond_its_width():
    X, ca, cb, top_a, bot_b = _two_cylinders("quad4", nr_b=2)
    delta = 1e-3
    exact = AxisymmetricContact(X, top_a, bot_b, penalty=100.0)
    smooth = AxisymmetricContact(X, top_a, bot_b, penalty=100.0, smoothing=delta)
    lower = np.unique(ca)                                      # body A moves up into B
    ndof = 2 * len(X)

    def state(pen):
        U = np.zeros(ndof)
        U[2 * lower + 1] = pen
        return U

    def axial(op, pen):
        return assembled(op, state(pen), ndof).reshape(-1, 2)[:, 1].sum()

    # identical once every node penetrates by more than delta
    for pen in (1.5 * delta, 4 * delta):
        assert np.allclose(assembled(smooth, state(pen), ndof), assembled(exact, state(pen), ndof), rtol=1e-12, atol=0)
    # separated beyond delta: no force; inside the band: between zero and the exact law, and monotone
    assert smooth.residual(state(-1.5 * delta), None, 1, 1).gdofs.size == 0
    pens = np.linspace(-delta, delta, 9)
    top = np.unique(top_a)
    forces = [abs(assembled(smooth, state(p), ndof).reshape(-1, 2)[top, 1].sum()) for p in pens]
    assert forces[0] == 0.0 and np.all(np.diff(forces) > 0)
    assert np.isclose(forces[-1], abs(assembled(exact, state(delta), ndof).reshape(-1, 2)[top, 1].sum()), rtol=1e-12)
    # C1: the stiffness is continuous at both ends of the band, and the tangent is consistent inside it
    h = 1e-7
    k = lambda p: (axial(smooth, p + h) - axial(smooth, p - h)) / (2 * h)  # noqa: E731
    assert abs(k(-delta + 2 * h)) < 1e-2 * abs(k(delta + 2 * h))
    assert np.isclose(k(delta - 2 * h), k(delta + 2 * h), rtol=1e-3)
    U = state(0.3 * delta)
    smooth.freeze(U)       # conforming nodes sit on primary vertices; frozen pairing projects without clamping
    t = smooth.tangent(U, None, 1, 1)
    K = np.zeros((ndof, ndof))
    np.add.at(K, (t.rows, t.cols), t.values)
    dU = np.random.default_rng(3).standard_normal(ndof) * 1e-9
    fd = assembled(smooth, U + dU, ndof) - assembled(smooth, U - dU, ndof)
    assert np.allclose(K @ (2 * dU), fd, rtol=1e-5, atol=1e-12 * np.abs(fd).max())

def test_rigid_plate_contact_and_separation():
    X, ca = mapped_block("quad4", np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    plate = np.array([[0.0, 1.1], [3.0, 1.1]])          # CCW around the plate above the body
    Xall = np.vstack([X, plate])
    ip = np.array([len(X), len(X) + 1])
    ea = boundary_edges(ca, "quad4")
    top_a = ea[np.isclose(Xall[ea, 1], 1.0).all(1)]
    props = (0.5, 0.0, 0.0, 50.0)
    solid = kernels.element_group(Xall, ca, "quad4", "fbar", props)
    contact = AxisymmetricContact(Xall, top_a, ip[None, :], penalty=500.0)
    ndof = 2 * len(Xall)
    U = np.zeros(ndof)
    assert contact.residual(U, None, 1, 1).gdofs.size == 0      # separated: no force
    bc = {2 * k: 0.0 for k in np.nonzero(Xall[:len(X), 0] == 0.0)[0]}
    bc.update({2 * k + 1: 0.0 for k in np.nonzero(Xall[:len(X), 1] == 0.0)[0]})
    lam = 0.8
    kw = dict(rtol=1e-12, atol=1e-13, line_search="admissible", predictor="tangent")
    for s in range(1, 5):
        d = dict(bc)
        d.update({2 * k: 0.0 for k in ip})
        d.update({2 * k + 1: (-0.1 + (lam - 1)) * s / 4 for k in ip})
        U, _, _ = newton_solve([solid, contact], U, None, ndof, d, **kw)
    for _ in range(12):
        contact.augment(U)
        U, _, _ = newton_solve([solid, contact], U, None, ndof, d, **kw)
    _, Pzz = uniaxial_stretch(props, lam)
    reaction = assembled(contact, U, ndof).reshape(-1, 2)[ip, 1].sum()
    assert np.isclose(reaction, Pzz * np.pi, rtol=1e-6)      # the plate carries the body's axial load
