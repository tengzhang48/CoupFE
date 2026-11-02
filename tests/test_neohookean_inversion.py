"""CPU inversion regressions; builds use the runtime's gfortran/f2py route."""
from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path
import re
import shutil

import numpy as np
import pytest

from coupfe.codegen import Material, WeakForm, VectorField, ScalarField, LocalScalar, tensor
from coupfe.codegen.generators.uel_gen import generate_element
from coupfe.runtime.compiled_element import CompiledElement, build_element_kernel

ROOT = Path(__file__).resolve().parents[1]
Q4 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
HEX8 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                 [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
TET4 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)


class Neo(Material):
    props = dict(G=0.5, K=50.0)

    def stress_PK1(self, F):
        return self.G * (F - tensor.inv(F).T) + self.K * tensor.log(tensor.det(F)) * tensor.inv(F).T


class PowerNeo(Material):
    props = dict(G=0.5, K=50.0)

    def stress_PK1(self, F):
        J = tensor.det(F)
        finv_t = tensor.inv(F).T
        I1 = tensor.trace(F.T @ F)
        return (self.G * J**(-2.0 / 3.0) * (F - I1 / 3.0 * finv_t)
                + self.K * (J - 1.0) * J * finv_t)


class Mechanical(WeakForm):
    material = Neo
    ndim = 2

    def define_fields(self):
        self.u = VectorField('u', degree=1)

    def momentum_equation(self, v, F):
        return self.material.stress_PK1(F)


class LocalNeo(Neo):
    def stress_PK1(self, F, p):
        return self.G * (F - tensor.inv(F).T) + p * tensor.inv(F).T

    def pressure_resid(self, F, p):
        return p - self.K * tensor.log(tensor.det(F))


class LocalMechanical(Mechanical):
    material = LocalNeo

    def define_fields(self):
        self.u = VectorField('u', degree=1)
        self.p = LocalScalar('p', storage='SVARS', condensed=True, initial=0.0)

    def momentum_equation(self, v, F, p):
        return self.material.stress_PK1(F, p)

    def pressure_equation(self, q, F, p):
        return self.material.pressure_resid(F, p)


class CoupledNeo(LocalNeo):
    props = dict(G=0.5, K=50.0, D=0.1)

    def stress_PK1(self, F, p, mu):
        return self.G * (F - tensor.inv(F).T) + p * tensor.inv(F).T

    def pressure_resid(self, F, p, mu):
        return p - self.K * tensor.log(tensor.det(F))

    def solvent_flux(self, F, p, mu, grad_mu):
        return -self.D * grad_mu

    def solvent_storage(self, F, F_old, p, p_old, dt):
        return (tensor.det(F) - tensor.det(F_old)) / dt


class Coupled(Mechanical):
    material = CoupledNeo

    def define_fields(self):
        self.u = VectorField('u', degree=1)
        self.p = ScalarField('p', degree=1)
        self.mu = ScalarField('mu', degree=1)

    def momentum_equation(self, v, F, p, mu):
        return self.material.stress_PK1(F, p, mu)

    def pressure_equation(self, q, F, p, mu):
        return self.material.pressure_resid(F, p, mu)

    def transport_equation(self, w, F, p, mu, grad_mu, F_old, p_old, dt):
        return (self.material.solvent_storage(F, F_old, p, p_old, dt),
                self.material.solvent_flux(F, p, mu, grad_mu))


class MagneticNeo(Neo):
    props = dict(G=0.5, K=50.0, mu0=1.0)

    def stress_PK1(self, F, B):
        return self.G * (F - tensor.inv(F).T) + self.K * tensor.log(tensor.det(F)) * tensor.inv(F).T

    def h_field(self, F, B):
        return B / self.mu0


class MagneticNeoSRI(MagneticNeo):
    def stress_PK1(self, F, B):
        return self.G * tensor.det(F)**(-2.0 / 3.0) * F

    def volumetric_PK1(self, F):
        return self.K * tensor.log(tensor.det(F)) * tensor.inv(F).T


def load_example(relative):
    spec = importlib.util.spec_from_file_location('inversion_' + relative.replace('/', '_'),
                                                 ROOT / 'examples' / (relative if relative.endswith('.py') else relative + '/build.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# One test per emitter/backend, including every native residual-only twin.
CASES = [f'{form}_{backend}' for form in ('standard', 'fbar', 'coupled')
         for backend in ('abaqus_uel', 'native')]
CASES += ['local_quad4', 'local_hex8', 'coupled_fbar',
          'magneto', 'magneto_sri', 'tet4_native', 'power_native']
CASES += ['example_' + name for name in ('neo_std_q4', 'neo_fbar_q4', 'neo_up_native_q4')]


def make_source(case, path):
    """Return source, nodes, props, DOFs/node, SVARS count."""
    nodes, props, dpn, ns = Q4, (0.5, 50.0), 2, 1
    if case.startswith('example_'):
        source = ROOT / 'examples/compression_cylinders' / (case[8:] + '.for')
        path.write_text(source.read_text())
        return nodes, props, dpn, ns
    if case.startswith('magneto'):
        from coupfe.codegen.generators.uel_magneto import generate_uel_magneto
        generate_uel_magneto(MagneticNeoSRI() if case.endswith('sri') else MagneticNeo(),
                             str(path), n_current_props=0,
                             sri_volumetric=case.endswith('sri'))
        return HEX8, (0.5, 50.0, 1.0), 6, 1
    if case.startswith('local_'):
        problem = LocalMechanical()
        element = 'Hex8' if case == 'local_hex8' else 'Quad4'
        if element == 'Hex8':
            problem.ndim = 3
            nodes, dpn = HEX8, 3
        generate_element(problem, str(path), element=element, formulation='local_pressure',
                         backend='abaqus_uel', mat_prefix='invtest')
        return nodes, props, dpn, ns
    if case.startswith('coupled'):
        ex = load_example('scalar_diffusion_uel')
        problem, props, dpn = ex.HeatDiffusionProblem(), ex.DEFAULT_PROPS, 3
    else:
        problem = Mechanical()
        if case == 'power_native':
            class PowerMechanical(Mechanical):
                material = PowerNeo
            problem = PowerMechanical()
    if case == 'coupled_fbar':
        problem, props, dpn = Coupled(), (0.5, 50.0, 0.1), 4
    backend = 'native' if case.endswith('native') else 'abaqus_uel'
    element = 'Tet4' if case.startswith('tet4') else 'Quad4'
    if element == 'Tet4':
        nodes, dpn = TET4, 3
    generate_element(problem, str(path), element=element, mat_prefix='invtest', backend=backend,
                     formulation='fbar_mechanics' if case.startswith('fbar') else 'standard')
    if case == 'coupled_fbar':
        from coupfe.codegen.generators.uel_gen import _generate_uel, _wrap_lines
        from coupfe.codegen.generators.uel_fbar_coupled import _generate_uel_fbar_coupled
        from coupfe.codegen.generators.element_config import ELEMENT_CONFIGS
        cfg = ELEMENT_CONFIGS['quad4']
        old = '\n'.join(_wrap_lines(_generate_uel(problem, 'invtest', cfg).splitlines()))
        new = '\n'.join(_wrap_lines(_generate_uel_fbar_coupled(problem, 'invtest', cfg).splitlines()))
        source = path.read_text()
        assert old in source
        path.write_text(source.replace(old, new))
    return nodes, props, dpn, ns


def build(path, name, work, props, dpn, ns, nodes):
    module = build_element_kernel(str(path), name, workdir=str(work))
    element = CompiledElement(module, props, dpn, n_svars=ns, n_elem=1, mcrd=nodes.shape[1])
    return module, element


def outputs(module, element, nodes, u):
    result = list(element.element_rk(0, nodes, u, u))
    if 'SUBROUTINE coupfe_element_r(' in Path(element_source(module)).read_text():
        # Call the actual residual-only symbol from the same normal runtime build.
        arrays = [np.asfortranarray(nodes.T), np.array(u), np.array(u), element.props,
                  np.zeros(element.svars_size), np.empty(u.size),
                  np.empty(element.svars_size), np.array([1., 1.]), np.array(1.)]
        fn = ctypes.CDLL(module.__file__).coupfe_element_r_
        fn(*(a.ctypes.data_as(ctypes.c_void_p) for a in arrays))
        result.append(arrays[5])
    return result


def element_source(module):
    return next(Path(module.__file__).parent.glob('*.for'))


def states(nodes, dpn):
    ndim = nodes.shape[1]
    for amount in (-0.03, 0.04, 0.15):
        u = np.zeros((len(nodes), dpn))
        u[:, :ndim] = amount * nodes
        u[2, 0] += amount / 3
        yield u.ravel()


def _inverted_node(nodes):
    return 2 if nodes.shape[1] == 2 else (6 if len(nodes) == 8 else 3)


@pytest.mark.parametrize('case', CASES)
def test_inversion_fails_loud_and_positive_states_unchanged(case, tmp_path):
    if any(shutil.which(tool) is None for tool in ('gfortran', 'meson', 'ninja')):
        pytest.skip('gfortran/f2py toolchain unavailable')
    path = tmp_path / 'kernel.for'
    nodes, props, dpn, ns = make_source(case, path)
    module, element = build(path, 'inv_' + case, tmp_path / 'new', props, dpn, ns, nodes)

    # Remove only the new marked guards for a literal old-arithmetic control.
    control = tmp_path / 'control.for'
    old_source = re.sub(r'^C     BEGIN positive-J guard\n.*?^C     END positive-J guard\n',
                        '', path.read_text(), flags=re.M | re.S)
    control.write_text(old_source)
    if old_source == path.read_text():
        old_module, old_element = module, element
    else:
        old_module, old_element = build(control, 'inv_control_' + case, tmp_path / 'old',
                                        props, dpn, ns, nodes)
    for u in states(nodes, dpn):
        for actual, before in zip(outputs(module, element, nodes, u),
                                  outputs(old_module, old_element, nodes, u)):
            assert np.isfinite(actual).all()
            assert actual.tobytes() == before.tobytes()

    # Push one node through the element. Quad: Jmin=-0.57735, Jbar=0;
    # Hex: top corner crosses the bottom face; Tet: J=-1.
    for amount in (-2.0, -3.0):
        u = np.zeros((len(nodes), dpn))
        u[_inverted_node(nodes), nodes.shape[1] - 1] = amount
        for value in outputs(module, element, nodes, u.ravel()):
            assert np.isnan(value).all(), f'{case}: inverted state returned finite entries'
    if nodes.shape[1] == 2:
        # All Gauss-point J and Jbar negative, so F-bar's Jbar/J stays positive.
        u = np.zeros((len(nodes), dpn))
        u[2, :2] = -3.0
        for value in outputs(module, element, nodes, u.ravel()):
            assert np.isnan(value).all()
    # Exactly singular F: affine collapse, all J = Jbar = 0 on the square/tet.
    u = np.zeros((len(nodes), dpn))
    u[:, nodes.shape[1] - 1] = -nodes[:, -1]
    for value in outputs(module, element, nodes, u.ravel()):
        assert np.isnan(value).all(), f'{case}: singular state must return all NaNs'
