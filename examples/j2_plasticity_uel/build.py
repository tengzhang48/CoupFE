"""Small-strain J2 plasticity Quad4 UEL proof element.

This element stores plastic strain (``epsp``, a 3x3 tensor) and accumulated
equivalent plastic strain (``alpha``, a scalar) per Gauss point.  It is
intentionally a proof-of-state-schema element, **not** a research-grade
finite-strain J2 model: the returned PK1 stress is approximated as the
small-strain Cauchy stress, which is consistent for infinitesimal strains.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import coupfe.codegen as au
from coupfe.codegen.core import tensor
from coupfe.codegen.generators.uel_gen import generate_element, generate_uel
from coupfe.runtime.compiled_element import build_element_kernel


class J2SmallStrainPlasticity(au.Material):
    """Small-strain J2 plasticity with linear isotropic hardening.

    State variables:
        epsp  -- plastic strain (small-strain, 3x3 tensor)
        alpha -- accumulated equivalent plastic strain (scalar)
    """

    props = dict(E=200e3, nu=0.3, sigma_y=250.0, H=1000.0)
    state_vars = dict(
        epsp=np.zeros((3, 3)),   # plastic strain (small-strain)
        alpha=0.0,                # accumulated equivalent plastic strain
    )

    def stress_PK1(self, F, epsp_old, alpha_old, dt):
        # Small-strain engineering strain
        eps = 0.5 * (F + tensor.transpose(F)) - tensor.eye(3)
        eps_e = eps - epsp_old
        # Lamé parameters
        mu = self.E / (2.0 * (1.0 + self.nu))
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        # Trial stress (Cauchy-like, returned as PK1 approximation)
        trace_eps = eps_e[0, 0] + eps_e[1, 1] + eps_e[2, 2]
        sigma_trial = lam * trace_eps * tensor.eye(3) + 2.0 * mu * eps_e
        # Deviatoric part
        tr = (sigma_trial[0, 0] + sigma_trial[1, 1] + sigma_trial[2, 2]) / 3.0
        s_trial = sigma_trial - tr * tensor.eye(3)
        # von Mises equivalent stress
        seq = tensor.sqrt(1.5 * tensor.sum(s_trial * s_trial))
        # Yield surface (phi is used instead of f to avoid a Fortran
        # name collision with the deformation-gradient argument F)
        phi = seq - (self.sigma_y + self.H * alpha_old)
        # Plastic multiplier
        dgamma = phi / (3.0 * mu + self.H)
        dgamma = tensor.where(phi.real > 0.0, dgamma, 0.0)
        # Associated flow direction d(seq)/d(sigma) = 3 s / (2 seq)
        n = 1.5 * s_trial / (seq + 1e-30)
        # Updated stress
        sigma = sigma_trial - 2.0 * mu * dgamma * n
        # Updated state
        epsp_new = epsp_old + tensor.where(phi.real > 0.0,
                                           dgamma * n,
                                           tensor.zeros((3, 3)))
        alpha_new = alpha_old + tensor.where(phi.real > 0.0,
                                             dgamma,
                                             0.0)
        # Return PK1 = sigma (approximation valid for small strains)
        return sigma, {'epsp': epsp_new, 'alpha': alpha_new}


class J2Quad4(au.WeakForm):
    """Single-field displacement Quad4 with small-strain J2 plasticity."""

    material = J2SmallStrainPlasticity
    ndim = 2

    def define_fields(self):
        self.u = au.VectorField('u', degree=1)

    def momentum_equation(self, v, F, epsp_old, alpha_old, dt):
        sigma, _state = self._mat.stress_PK1(F, epsp_old, alpha_old, dt)
        return sigma


DEFAULT_PROPS = (200e3, 0.3, 250.0, 1000.0)


def verification_state():
    """A non-trivial material-point state for verification/tests."""
    F = np.array([[1.02, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    epsp = np.array([[0.005, 0.0, 0.0],
                     [0.0, -0.0025, 0.0],
                     [0.0, 0.0, -0.0025]])
    alpha = 0.01
    return dict(F=F, epsp=epsp, alpha=alpha, dt=0.1)


def _lame_parameters(E: float, nu: float):
    """Shear modulus ``mu`` and first Lamé parameter ``lam``."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def analytic_state_update(eps, epsp_old, alpha_old, E, nu, sigma_y, H):
    """NumPy oracle for the conventional associative radial-return update."""
    eps = np.asarray(eps, dtype=float)
    epsp_old = np.asarray(epsp_old, dtype=float)
    alpha_old = float(alpha_old)
    eps_e = eps - epsp_old
    mu, lam = _lame_parameters(E, nu)
    trace_eps = eps_e[0, 0] + eps_e[1, 1] + eps_e[2, 2]
    sigma_trial = lam * trace_eps * np.eye(3) + 2.0 * mu * eps_e
    tr = (sigma_trial[0, 0] + sigma_trial[1, 1] + sigma_trial[2, 2]) / 3.0
    s_trial = sigma_trial - tr * np.eye(3)
    seq = np.sqrt(1.5 * np.sum(s_trial * s_trial))
    f = seq - (sigma_y + H * alpha_old)
    if f > 0.0:
        dgamma = f / (3.0 * mu + H)
        n = 1.5 * s_trial / (seq + 1e-30)
        sigma = sigma_trial - 2.0 * mu * dgamma * n
        epsp_new = epsp_old + dgamma * n
        alpha_new = alpha_old + dgamma
    else:
        sigma = sigma_trial
        epsp_new = epsp_old.copy()
        alpha_new = alpha_old
    return sigma, epsp_new, alpha_new


def analytic_uniaxial(stretch, E=200e3, nu=0.3, sigma_y=250.0, H=1000.0):
    """Analytic axial stress and committed ``alpha`` for uniaxial strain.

    The 2D Quad4 element is plane-strain; this oracle assumes the
    out-of-plane and lateral strains are zero (constrained uniaxial
    strain).  The returned ``sigma_xx`` is the small-strain Cauchy stress.
    """
    e = stretch - 1.0
    eps = np.diag([e, 0.0, 0.0])
    sigma, _, alpha = analytic_state_update(
        eps, np.zeros((3, 3)), 0.0, E, nu, sigma_y, H
    )
    return float(sigma[0, 0]), float(alpha)


# ---------------------------------------------------------------------------
# State packing helpers for the compiled element
# ---------------------------------------------------------------------------

_PER_GP = 10  # 9 for epsp + 1 for alpha


def pack_element_state(svars_1d: np.ndarray, epsp, alpha) -> None:
    """Write the same ``epsp``/``alpha`` into every Gauss point of ``svars_1d``."""
    epsp = np.asarray(epsp, dtype=float)
    n_gp = svars_1d.size // _PER_GP
    for gp in range(n_gp):
        base = gp * _PER_GP
        svars_1d[base:base + 9] = epsp.ravel('F')
        svars_1d[base + 9] = float(alpha)


def unpack_gp_state(svars_1d: np.ndarray, gp: int = 0):
    """Return ``(epsp, alpha)`` for Gauss point ``gp``."""
    base = gp * _PER_GP
    epsp = svars_1d[base:base + 9].reshape((3, 3), order='F')
    alpha = svars_1d[base + 9]
    return epsp, alpha


def set_element_state(elem, epsp, alpha) -> None:
    """Pack ``epsp``/``alpha`` into every GP of a single-element CompiledElement."""
    if elem.svars is None:
        raise ValueError('CompiledElement has no state')
    pack_element_state(elem.svars[0], epsp, alpha)
    if elem.svars_trial is not None:
        pack_element_state(elem.svars_trial[0], epsp, alpha)


def get_gp_state(elem, gp: int = 0, ei: int = 0):
    """Read ``(epsp, alpha)`` from a CompiledElement after commit/trial."""
    return unpack_gp_state(elem.svars[ei], gp)


# ---------------------------------------------------------------------------
# Kernel generation
# ---------------------------------------------------------------------------

def build_kernel(
    tmpdir: Optional[str] = None,
    backend: str = 'native',
    module_name: str = 'j2_q4_module',
) -> Tuple[Path, object]:
    """Generate and compile the J2 Quad4 kernel.

    Returns ``(for_path, module)``.  ``backend`` may be ``'native'`` or
    ``'abaqus_uel'``.
    """
    problem = J2Quad4()
    problem.verify(state=verification_state(), verbose=False)

    here = Path(__file__).resolve().parent
    out = Path(tmpdir) if tmpdir else here
    out.mkdir(parents=True, exist_ok=True)
    for_path = out / 'j2_plasticity_q4.for'

    if backend == 'native':
        generate_element(
            problem, str(for_path), element='Quad4',
            formulation='standard', backend='native',
        )
    elif backend == 'abaqus_uel':
        generate_uel(
            problem, str(for_path), element='Quad4', formulation='standard'
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'")

    mod = build_element_kernel(str(for_path), module_name, workdir=str(out))
    return for_path, mod


if __name__ == '__main__':
    path, _ = build_kernel()
    print(f'Generated {path}')
