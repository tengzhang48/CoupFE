"""
Magneto-mechanical (u-A) hex8 UEL generator — Phase 3 of the MRE plan.

Generates a self-contained Abaqus UEL for the coupled finite-strain
magneto-mechanical problem of Dorn, Bodelot & Danas (JAM 88:071004,
2021): displacement u and magnetic vector potential A as nodal fields
(6 DOF/node, node-major), B = Curl A, total-Lagrangian.

Variational structure (their Eq. 1):

    P(u,A) = int W(F,B) dV - int (J.A) dV + 1/(2 mu0 xi) int (Div A)^2 dV

Element residual/tangent per Gauss point, with S = dW/dF, H = dW/dB
supplied by translator-generated material subroutines and the four
tangent blocks (dS/dF, dS/dB, dH/dF, dH/dB) by complex-step:

    r_u(a,i) = S(i,J) dsh(a,J)
    r_A(a,i) = H_k CSH(k,a,i),   CSH(k,b,m) = eps_kpm dsh(b,p) = dB_k/dA_(b,m)
    K_uu     = dsh . dS/dF . dsh         K_uA = dsh . dS/dB . CSH
    K_Au     = CSH . dH/dF . dsh         K_AA = CSH . dH/dB . CSH

plus the Coulomb-gauge penalty, evaluated by SELECTIVE REDUCED
INTEGRATION at the element centroid only (single Gauss point, as in the
paper — full integration would over-constrain A):

    r_A += (1/(mu0 xi)) (Div A) dsh0(a,i),  K_AA += (1/(mu0 xi)) dsh0 x dsh0

and an optional prescribed constant referential current density J
(PROPS-supplied) contributing only to r_A.

Material contract: a stateless Material with `stress_PK1(self, F, B)`
and `h_field(self, F, B)` written in the tensor DSL, and a `mu0` prop
(used by the gauge term). With `sri_volumetric=True` the material
additionally defines `volumetric_PK1(self, F)` carrying the Gp(J-1)^2
part (stress_PK1 then holds only deviatoric+magnetic+vacuum), and that
term is integrated at the centroid only — matching the paper's
quasi-incompressible MRE treatment. Hypergeometric (general-k)
magnetization is supported through @au.fortran_helper sidecars
(see examples/MRE/hyp2f1z_pair.for) passed via `extra_fortran_files`.

Scope guards (this generator raises rather than mis-generating —
lesson from the 2026-06-10 audit's prototype-generator findings):
hex8 only, stateless materials only, exactly the methods above.

Translator pitfall worth knowing: Fortran is case-insensitive, so DSL
material locals must not collide with dummy args up to case (e.g. a
local `f` collides with the `F(3,3)` argument — gfortran rejects the
generated code loudly).
"""

import os

from .uel_gen import (_generate_material_subroutine, _wrap_lines)
from .element_config import ELEMENT_CONFIGS

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')

CS_H = 1.0e-10


def _read_template(name):
    with open(os.path.join(_TEMPLATE_DIR, name)) as f:
        return f.read()


def generate_uel_magneto(material, output_path=None, mat_prefix='mag',
                         xi_gauge=1.0e-6, n_current_props=3,
                         jtype_comment='U1', sri_volumetric=False,
                         extra_fortran_files=None):
    """Generate the coupled u-A hex8 magneto UEL.

    Args:
        material: stateless Material with stress_PK1(F, B), h_field(F, B)
                  and a 'mu0' prop.
        output_path: if given, write the .for file there.
        mat_prefix: prefix for generated subroutine names.
        xi_gauge: Coulomb-gauge penalty parameter (paper: 1e-6).
        n_current_props: number of PROPS entries appended after the
                  material props holding the constant referential current
                  density vector (3, or 0 to omit the source term).
        sri_volumetric: selective reduced integration of the volumetric
                  penalty. Requires the material to define
                  volumetric_PK1(F) holding the Gp(J-1)^2 part, with
                  stress_PK1 carrying ONLY the deviatoric+magnetic+vacuum
                  parts (total stress = sum of the two). The volumetric
                  term is then integrated at the centroid only, matching
                  Dorn-Bodelot-Danas.
        extra_fortran_files: hand-written sidecar sources (e.g. the
                  hyp2f1z_pair.for required by @fortran_helper methods),
                  appended verbatim.

    Returns: the Fortran source as a string.
    """
    # ---- scope guards ----
    if material.state_vars:
        raise NotImplementedError(
            'generate_uel_magneto supports stateless materials only')
    methods = set(material.defined_methods())
    if not {'stress_PK1', 'h_field'} <= methods:
        raise ValueError(
            'magneto material must define stress_PK1(F, B) and '
            f'h_field(F, B); found {sorted(methods)}')
    for mname in ('stress_PK1', 'h_field'):
        params = material._methods[mname]['all_params']
        if params != ['F', 'B']:
            raise NotImplementedError(
                f'{mname} must take exactly (F, B); got {params}')
    has_vol = 'volumetric_PK1' in methods
    if sri_volumetric and not has_vol:
        raise ValueError(
            'sri_volumetric=True requires the material to define '
            'volumetric_PK1(F)')
    if has_vol and not sri_volumetric:
        raise ValueError(
            "material defines volumetric_PK1 but sri_volumetric=False — "
            'ambiguous total stress; enable SRI or fold the volumetric '
            'term into stress_PK1')
    if sri_volumetric:
        vparams = material._methods['volumetric_PK1']['all_params']
        if vparams != ['F']:
            raise NotImplementedError(
                f'volumetric_PK1 must take exactly (F); got {vparams}')
    if 'mu0' not in material.props_names:
        raise ValueError("material must carry a 'mu0' prop "
                         '(used by the gauge penalty term)')
    if n_current_props not in (0, 3):
        raise ValueError('n_current_props must be 0 or 3')

    cfg = ELEMENT_CONFIGS['hex8']
    nprops_mat = len(material.props_names)
    imu0 = material.props_names.index('mu0') + 1

    # ---- material + CS engine subroutines ----
    helper_ctx = {'material': material, 'helper_sources': [],
                  'helper_cache': {}, 'helper_method_kinds': {}}
    gen_methods = ['stress_PK1', 'h_field']
    if sri_volumetric:
        gen_methods.append('volumetric_PK1')
    mat_src = []
    for mname in gen_methods:
        raw = _generate_material_subroutine(
            material, mname, mat_prefix, helper_ctx=helper_ctx)
        mat_src.append('\n'.join(_wrap_lines(raw.splitlines())))
        mat_src.append('')
    for src in helper_ctx['helper_sources']:
        mat_src.append('\n'.join(_wrap_lines(src.splitlines())))
        mat_src.append('')

    cs_src = _generate_cs_blocks(mat_prefix, nprops_mat)
    if sri_volumetric:
        cs_src += '\n\n' + _generate_cs_vol(mat_prefix, nprops_mat)
    uel_src = _generate_uel_body(mat_prefix, nprops_mat, imu0,
                                 xi_gauge, n_current_props,
                                 jtype_comment, sri_volumetric)

    templates = [
        _read_template('tensor_ops.for'),
        _read_template('isoparametric.for'),
        _read_template('gauss_hex.for'),
        _read_template('shape_hex8.for'),
    ]

    header = [
        'C =====================================================',
        'C  Coupled magneto-mechanical (u-A) hex8 UEL',
        'C  Generated by coupfe.codegen.generators.uel_magneto',
        'C',
        'C  DOF layout (node-major): u1,u2,u3,A1,A2,A3 per node',
        'C  NDOFEL = 48, NNODE = 8, full 2x2x2 integration;',
        'C  Coulomb-gauge penalty reduced-integrated at centroid',
        f'C  with xi = {xi_gauge:.1e} and mu0 = PROPS({imu0}).',
    ]
    if n_current_props:
        header += [
            'C  Constant referential current density J =',
            f'C  PROPS({nprops_mat + 1}..{nprops_mat + 3}) '
            '(contributes to RHS only).',
        ]
    if sri_volumetric:
        header += [
            'C  Volumetric penalty (volumetric_PK1) reduced-',
            'C  integrated at the centroid (SRI).',
        ]
    header += [
        f'C  Material props: PROPS(1..{nprops_mat}) = '
        + ', '.join(material.props_names),
        'C =====================================================',
        '',
    ]

    extra_src = []
    for path in (extra_fortran_files or []):
        with open(path) as fh:
            extra_src.append('C     --- User Fortran helper sidecar: '
                             + os.path.basename(str(path)) + ' ---')
            extra_src.append(fh.read())

    full = '\n'.join(header) + uel_src + '\n\n' + cs_src + '\n\n' \
        + '\n'.join(mat_src) + '\n\n' + '\n\n'.join(templates)
    if extra_src:
        full += '\n\n' + '\n'.join(extra_src)

    if output_path is not None:
        with open(output_path, 'w') as f:
            f.write(full)
    return full


# =====================================================================
# CS four-block engine
# =====================================================================

def _generate_cs_blocks(p, nprops):
    """12-perturbation CS engine: KFF(3,3,3,3), KFB(3,3,3), KBF(3,3,3),
    KBB(3,3). Full complex-state reset before every perturbation."""
    L = []
    L.append(f'      SUBROUTINE {p}_cs_blocks(F, B, PROPS, KFF, KFB,')
    L.append('     &    KBF, KBB)')
    L.append('      IMPLICIT NONE')
    L.append('      DOUBLE PRECISION, INTENT(IN) :: F(3,3), B(3)')
    L.append(f'      DOUBLE PRECISION, INTENT(IN) :: PROPS({nprops})')
    L.append('      DOUBLE PRECISION, INTENT(OUT) :: KFF(3,3,3,3)')
    L.append('      DOUBLE PRECISION, INTENT(OUT) :: KFB(3,3,3)')
    L.append('      DOUBLE PRECISION, INTENT(OUT) :: KBF(3,3,3)')
    L.append('      DOUBLE PRECISION, INTENT(OUT) :: KBB(3,3)')
    L.append('      DOUBLE COMPLEX :: Fz(3,3), Bz(3), Pz(3,3), Hz(3)')
    L.append('      DOUBLE PRECISION, PARAMETER :: CS_H = 1.0d-10')
    L.append('      INTEGER :: i, j, k, l, m')
    L.append('')
    L.append('C     Perturb F(k,l): columns of dS/dF and dH/dF')
    L.append('      DO l = 1, 3')
    L.append('        DO k = 1, 3')
    L.append('          DO j = 1, 3')
    L.append('            DO i = 1, 3')
    L.append('              Fz(i,j) = DCMPLX(F(i,j), 0.0d0)')
    L.append('            END DO')
    L.append('            Bz(j) = DCMPLX(B(j), 0.0d0)')
    L.append('          END DO')
    L.append('          Fz(k,l) = Fz(k,l) + DCMPLX(0.0d0, CS_H)')
    L.append(f'          CALL {p}_stress_PK1(Fz, Bz, PROPS, Pz)')
    L.append(f'          CALL {p}_h_field(Fz, Bz, PROPS, Hz)')
    L.append('          DO j = 1, 3')
    L.append('            DO i = 1, 3')
    L.append('              KFF(i,j,k,l) = AIMAG(Pz(i,j)) / CS_H')
    L.append('            END DO')
    L.append('            KBF(j,k,l) = AIMAG(Hz(j)) / CS_H')
    L.append('          END DO')
    L.append('        END DO')
    L.append('      END DO')
    L.append('')
    L.append('C     Perturb B(m): columns of dS/dB and dH/dB')
    L.append('      DO m = 1, 3')
    L.append('        DO j = 1, 3')
    L.append('          DO i = 1, 3')
    L.append('            Fz(i,j) = DCMPLX(F(i,j), 0.0d0)')
    L.append('          END DO')
    L.append('          Bz(j) = DCMPLX(B(j), 0.0d0)')
    L.append('        END DO')
    L.append('        Bz(m) = Bz(m) + DCMPLX(0.0d0, CS_H)')
    L.append(f'        CALL {p}_stress_PK1(Fz, Bz, PROPS, Pz)')
    L.append(f'        CALL {p}_h_field(Fz, Bz, PROPS, Hz)')
    L.append('        DO j = 1, 3')
    L.append('          DO i = 1, 3')
    L.append('            KFB(i,j,m) = AIMAG(Pz(i,j)) / CS_H')
    L.append('          END DO')
    L.append('          KBB(j,m) = AIMAG(Hz(j)) / CS_H')
    L.append('        END DO')
    L.append('      END DO')
    L.append('')
    L.append('      RETURN')
    L.append(f'      END SUBROUTINE {p}_cs_blocks')
    return '\n'.join(_wrap_lines(L))


def _generate_cs_vol(p, nprops):
    """9-perturbation CS engine for the volumetric SRI part:
    KVV(3,3,3,3) = d(volumetric_PK1)/dF."""
    L = []
    L.append(f'      SUBROUTINE {p}_cs_vol(F, PROPS, KVV)')
    L.append('      IMPLICIT NONE')
    L.append('      DOUBLE PRECISION, INTENT(IN) :: F(3,3)')
    L.append(f'      DOUBLE PRECISION, INTENT(IN) :: PROPS({nprops})')
    L.append('      DOUBLE PRECISION, INTENT(OUT) :: KVV(3,3,3,3)')
    L.append('      DOUBLE COMPLEX :: Fz(3,3), Pz(3,3)')
    L.append('      DOUBLE PRECISION, PARAMETER :: CS_H = 1.0d-10')
    L.append('      INTEGER :: i, j, k, l')
    L.append('')
    L.append('      DO l = 1, 3')
    L.append('        DO k = 1, 3')
    L.append('          DO j = 1, 3')
    L.append('            DO i = 1, 3')
    L.append('              Fz(i,j) = DCMPLX(F(i,j), 0.0d0)')
    L.append('            END DO')
    L.append('          END DO')
    L.append('          Fz(k,l) = Fz(k,l) + DCMPLX(0.0d0, CS_H)')
    L.append(f'          CALL {p}_volumetric_PK1(Fz, PROPS, Pz)')
    L.append('          DO j = 1, 3')
    L.append('            DO i = 1, 3')
    L.append('              KVV(i,j,k,l) = AIMAG(Pz(i,j)) / CS_H')
    L.append('            END DO')
    L.append('          END DO')
    L.append('        END DO')
    L.append('      END DO')
    L.append('')
    L.append('      RETURN')
    L.append(f'      END SUBROUTINE {p}_cs_vol')
    return '\n'.join(_wrap_lines(L))


# =====================================================================
# UEL body
# =====================================================================

def _generate_uel_body(p, nprops_mat, imu0, xi_gauge, n_current_props,
                       jtype_comment, sri_volumetric=False):
    L = []
    L.append('      SUBROUTINE UEL(RHS, AMATRX, SVARS, ENERGY, NDOFEL,')
    L.append('     &  NRHS, NSVARS, PROPS, NPROPS, COORDS, MCRD, NNODE,')
    L.append('     &  U, DU, V, A, JTYPE, TIME, DTIME, KSTEP, KINC,')
    L.append('     &  JELEM, PARAMS, NDLOAD, JDLTYP, ADLMAG, PREDEF,')
    L.append('     &  NPREDF, LFLAGS, MLVARX, DDLMAG, MDLOAD, PNEWDT,')
    L.append('     &  JPROPS, NJPROP, PERIOD)')
    L.append('')
    L.append('      IMPLICIT NONE')
    L.append('')
    L.append('C     --- Abaqus UEL interface ---')
    L.append('      INTEGER NDOFEL, NRHS, NSVARS, NPROPS, MCRD, NNODE')
    L.append('      INTEGER JTYPE, KSTEP, KINC, JELEM, NDLOAD, NPREDF')
    L.append('      INTEGER MLVARX, MDLOAD, NJPROP')
    L.append('      INTEGER JDLTYP(MDLOAD,*), LFLAGS(*), JPROPS(*)')
    L.append('      DOUBLE PRECISION RHS(MLVARX,*), AMATRX(NDOFEL,NDOFEL)')
    L.append('      DOUBLE PRECISION SVARS(NSVARS), ENERGY(8)')
    L.append('      DOUBLE PRECISION PROPS(NPROPS), COORDS(MCRD,NNODE)')
    L.append('      DOUBLE PRECISION U(NDOFEL), DU(MLVARX,*), V(NDOFEL)')
    L.append('      DOUBLE PRECISION A(NDOFEL)')
    L.append('      DOUBLE PRECISION TIME(2), DTIME, PARAMS(3)')
    L.append('      DOUBLE PRECISION ADLMAG(MDLOAD,*)')
    L.append('      DOUBLE PRECISION PREDEF(2,NPREDF,NNODE)')
    L.append('      DOUBLE PRECISION DDLMAG(MDLOAD,*), PNEWDT, PERIOD')
    L.append('')
    L.append('C     --- Local parameters ---')
    L.append('      INTEGER, PARAMETER :: ndim = 3')
    L.append('      INTEGER, PARAMETER :: NNODE_E = 8')
    L.append('      INTEGER, PARAMETER :: NGP = 8')
    L.append('')
    L.append('C     --- Nodal arrays / DOF maps (node-major u,A) ---')
    L.append('      DOUBLE PRECISION :: u_node(3, NNODE_E)')
    L.append('      DOUBLE PRECISION :: a_node(3, NNODE_E)')
    L.append('      INTEGER :: edof_u(3, NNODE_E), edof_a(3, NNODE_E)')
    L.append('')
    L.append('C     --- Shape functions and mapping ---')
    L.append('      DOUBLE PRECISION :: sh8(8), dshxi8(8,3), dsh8(8,3)')
    L.append('      DOUBLE PRECISION :: xi_gp(8,3), w_gp(8)')
    L.append('      DOUBLE PRECISION :: coords_3d(3, NNODE_E)')
    L.append('      DOUBLE PRECISION :: detJxi, Jinv_xi(3,3)')
    L.append('      INTEGER :: ngp_out, stat')
    L.append('')
    L.append('C     --- Gauss-point fields ---')
    L.append('      DOUBLE PRECISION :: F(3,3), Bvec(3), detF')
    L.append('      DOUBLE PRECISION :: CSH(3, NNODE_E, 3)')
    L.append('      DOUBLE PRECISION :: S_r(3,3), H_r(3)')
    L.append('      DOUBLE PRECISION :: KFF(3,3,3,3), KFB(3,3,3)')
    L.append('      DOUBLE PRECISION :: KBF(3,3,3), KBB(3,3)')
    if sri_volumetric:
        L.append('      DOUBLE PRECISION :: KVV(3,3,3,3), Svol(3,3)')
    L.append('      DOUBLE COMPLEX :: Fz(3,3), Bz(3), Pz(3,3), Hz(3)')
    L.append('      DOUBLE PRECISION :: wdetJ, gfac, divA')
    L.append('      DOUBLE PRECISION :: det33d')
    L.append('      INTEGER :: idx, a_n, b_n, i, j, k, l, m, q, r, kk')
    L.append('      INTEGER :: row, col')
    L.append('')
    L.append('C     Extract coordinates')
    L.append('      DO a_n = 1, NNODE_E')
    L.append('        coords_3d(1, a_n) = COORDS(1, a_n)')
    L.append('        coords_3d(2, a_n) = COORDS(2, a_n)')
    L.append('        coords_3d(3, a_n) = COORDS(3, a_n)')
    L.append('      END DO')
    L.append('')
    L.append('C     Zero RHS and AMATRX')
    L.append('      DO i = 1, NDOFEL')
    L.append('        RHS(i, 1) = 0.0d0')
    L.append('        DO j = 1, NDOFEL')
    L.append('          AMATRX(i, j) = 0.0d0')
    L.append('        END DO')
    L.append('      END DO')
    L.append('')
    L.append('C     Parse DOFs (node-major: u1,u2,u3,A1,A2,A3)')
    L.append('      idx = 0')
    L.append('      DO a_n = 1, NNODE_E')
    L.append('        DO i = 1, 3')
    L.append('          idx = idx + 1')
    L.append('          u_node(i, a_n) = U(idx)')
    L.append('          edof_u(i, a_n) = idx')
    L.append('        END DO')
    L.append('        DO i = 1, 3')
    L.append('          idx = idx + 1')
    L.append('          a_node(i, a_n) = U(idx)')
    L.append('          edof_a(i, a_n) = idx')
    L.append('        END DO')
    L.append('      END DO')
    L.append('')
    L.append('C     ==========================================')
    L.append('C     Full 2x2x2 pass: W-terms (S, H, CS blocks)')
    L.append('C     ==========================================')
    L.append('      CALL gauss_hex8(xi_gp, w_gp, ngp_out)')
    L.append('')
    L.append('      DO kk = 1, NGP')
    L.append('        CALL shape_hex8(xi_gp(kk,1), xi_gp(kk,2),')
    L.append('     &    xi_gp(kk,3), sh8, dshxi8)')
    L.append('        CALL map_grad_3d(dshxi8, coords_3d, 8, dsh8,')
    L.append('     &    detJxi, Jinv_xi, stat)')
    L.append('        IF (stat .EQ. 0) THEN')
    L.append('          PNEWDT = 0.25d0')
    L.append('          RETURN')
    L.append('        END IF')
    L.append('        wdetJ = detJxi * w_gp(kk)')
    L.append('')
    L.append('C       F = I + sum u_a x dsh_a')
    L.append('        DO i = 1, 3')
    L.append('          DO j = 1, 3')
    L.append('            F(i,j) = 0.0d0')
    L.append('            IF (i .EQ. j) F(i,j) = 1.0d0')
    L.append('            DO a_n = 1, NNODE_E')
    L.append('              F(i,j) = F(i,j) + u_node(i,a_n)*dsh8(a_n,j)')
    L.append('            END DO')
    L.append('          END DO')
    L.append('        END DO')
    L.append('')
    L.append('C       Cutback on inverted configuration')
    L.append('        detF = det33d(F)')
    L.append('        IF (detF .LE. 0.0d0) THEN')
    L.append('          PNEWDT = 0.25d0')
    L.append('          RETURN')
    L.append('        END IF')
    L.append('')
    L.append('C       Curl-shape operator CSH(k,b,m) = dB_k/dA_(b,m)')
    L.append('C       = eps_kpm dsh(b,p)  (columns: dsh_b x e_m)')
    L.append('        DO b_n = 1, NNODE_E')
    L.append('          CSH(1,b_n,1) = 0.0d0')
    L.append('          CSH(2,b_n,1) = dsh8(b_n,3)')
    L.append('          CSH(3,b_n,1) = -dsh8(b_n,2)')
    L.append('          CSH(1,b_n,2) = -dsh8(b_n,3)')
    L.append('          CSH(2,b_n,2) = 0.0d0')
    L.append('          CSH(3,b_n,2) = dsh8(b_n,1)')
    L.append('          CSH(1,b_n,3) = dsh8(b_n,2)')
    L.append('          CSH(2,b_n,3) = -dsh8(b_n,1)')
    L.append('          CSH(3,b_n,3) = 0.0d0')
    L.append('        END DO')
    L.append('')
    L.append('C       B = Curl A via the same operator (consistency)')
    L.append('        DO k = 1, 3')
    L.append('          Bvec(k) = 0.0d0')
    L.append('          DO b_n = 1, NNODE_E')
    L.append('            DO m = 1, 3')
    L.append('              Bvec(k) = Bvec(k)')
    L.append('     &          + CSH(k,b_n,m)*a_node(m,b_n)')
    L.append('            END DO')
    L.append('          END DO')
    L.append('        END DO')
    L.append('')
    L.append('C       Real evaluation of S and H')
    L.append('        DO i = 1, 3')
    L.append('          DO j = 1, 3')
    L.append('            Fz(i,j) = DCMPLX(F(i,j), 0.0d0)')
    L.append('          END DO')
    L.append('          Bz(i) = DCMPLX(Bvec(i), 0.0d0)')
    L.append('        END DO')
    L.append(f'        CALL {p}_stress_PK1(Fz, Bz, PROPS, Pz)')
    L.append(f'        CALL {p}_h_field(Fz, Bz, PROPS, Hz)')
    L.append('        DO i = 1, 3')
    L.append('          DO j = 1, 3')
    L.append('            S_r(i,j) = DBLE(Pz(i,j))')
    L.append('          END DO')
    L.append('          H_r(i) = DBLE(Hz(i))')
    L.append('        END DO')
    L.append('')
    L.append('C       CS four-block tangent at this GP')
    L.append(f'        CALL {p}_cs_blocks(F, Bvec, PROPS, KFF, KFB,')
    L.append('     &    KBF, KBB)')
    L.append('')
    L.append('C       Residual: RHS = -r')
    L.append('        DO a_n = 1, NNODE_E')
    L.append('          DO i = 1, 3')
    L.append('            row = edof_u(i, a_n)')
    L.append('            DO j = 1, 3')
    L.append('              RHS(row,1) = RHS(row,1)')
    L.append('     &          - S_r(i,j)*dsh8(a_n,j)*wdetJ')
    L.append('            END DO')
    L.append('            row = edof_a(i, a_n)')
    L.append('            DO k = 1, 3')
    L.append('              RHS(row,1) = RHS(row,1)')
    L.append('     &          - H_r(k)*CSH(k,a_n,i)*wdetJ')
    L.append('            END DO')
    L.append('          END DO')
    L.append('        END DO')
    L.append('')
    L.append('C       Tangent: AMATRX = +dr/dU')
    L.append('        DO a_n = 1, NNODE_E')
    L.append('        DO i = 1, 3')
    L.append('          DO b_n = 1, NNODE_E')
    L.append('          DO k = 1, 3')
    L.append('C           K_uu')
    L.append('            row = edof_u(i, a_n)')
    L.append('            col = edof_u(k, b_n)')
    L.append('            DO j = 1, 3')
    L.append('              DO l = 1, 3')
    L.append('                AMATRX(row,col) = AMATRX(row,col)')
    L.append('     &            + dsh8(a_n,j)*KFF(i,j,k,l)')
    L.append('     &            * dsh8(b_n,l)*wdetJ')
    L.append('              END DO')
    L.append('            END DO')
    L.append('C           K_uA')
    L.append('            col = edof_a(k, b_n)')
    L.append('            DO j = 1, 3')
    L.append('              DO q = 1, 3')
    L.append('                AMATRX(row,col) = AMATRX(row,col)')
    L.append('     &            + dsh8(a_n,j)*KFB(i,j,q)')
    L.append('     &            * CSH(q,b_n,k)*wdetJ')
    L.append('              END DO')
    L.append('            END DO')
    L.append('C           K_Au')
    L.append('            row = edof_a(i, a_n)')
    L.append('            col = edof_u(k, b_n)')
    L.append('            DO q = 1, 3')
    L.append('              DO l = 1, 3')
    L.append('                AMATRX(row,col) = AMATRX(row,col)')
    L.append('     &            + CSH(q,a_n,i)*KBF(q,k,l)')
    L.append('     &            * dsh8(b_n,l)*wdetJ')
    L.append('              END DO')
    L.append('            END DO')
    L.append('C           K_AA')
    L.append('            col = edof_a(k, b_n)')
    L.append('            DO q = 1, 3')
    L.append('              DO r = 1, 3')
    L.append('                AMATRX(row,col) = AMATRX(row,col)')
    L.append('     &            + CSH(q,a_n,i)*KBB(q,r)')
    L.append('     &            * CSH(r,b_n,k)*wdetJ')
    L.append('              END DO')
    L.append('            END DO')
    L.append('          END DO')
    L.append('          END DO')
    L.append('        END DO')
    L.append('        END DO')

    if n_current_props:
        L.append('')
        L.append('C       Current source J.A: RHS only (J prescribed)')
        L.append('        DO a_n = 1, NNODE_E')
        L.append('          DO i = 1, 3')
        L.append('            row = edof_a(i, a_n)')
        L.append('            RHS(row,1) = RHS(row,1)')
        L.append(f'     &        + sh8(a_n)*PROPS({nprops_mat}+i)*wdetJ')
        L.append('          END DO')
        L.append('        END DO')

    L.append('      END DO')
    L.append('')
    L.append('C     ==========================================')
    L.append('C     Reduced (centroid) pass: Coulomb gauge')
    L.append('C     ==========================================')
    L.append('      CALL shape_hex8(0.0d0, 0.0d0, 0.0d0, sh8, dshxi8)')
    L.append('      CALL map_grad_3d(dshxi8, coords_3d, 8, dsh8,')
    L.append('     &  detJxi, Jinv_xi, stat)')
    L.append('      IF (stat .EQ. 0) THEN')
    L.append('        PNEWDT = 0.25d0')
    L.append('        RETURN')
    L.append('      END IF')
    L.append('C     Single-point rule: weight 8 in the parent domain')
    L.append('      wdetJ = detJxi * 8.0d0')
    xi_lit = f'{xi_gauge:.6e}'.replace('e', 'd')
    L.append(f'      gfac = 1.0d0 / (PROPS({imu0}) * {xi_lit})')
    L.append('')
    L.append('      divA = 0.0d0')
    L.append('      DO a_n = 1, NNODE_E')
    L.append('        DO i = 1, 3')
    L.append('          divA = divA + a_node(i,a_n)*dsh8(a_n,i)')
    L.append('        END DO')
    L.append('      END DO')
    L.append('')
    L.append('      DO a_n = 1, NNODE_E')
    L.append('        DO i = 1, 3')
    L.append('          row = edof_a(i, a_n)')
    L.append('          RHS(row,1) = RHS(row,1)')
    L.append('     &      - gfac*divA*dsh8(a_n,i)*wdetJ')
    L.append('          DO b_n = 1, NNODE_E')
    L.append('            DO m = 1, 3')
    L.append('              col = edof_a(m, b_n)')
    L.append('              AMATRX(row,col) = AMATRX(row,col)')
    L.append('     &          + gfac*dsh8(a_n,i)*dsh8(b_n,m)*wdetJ')
    L.append('            END DO')
    L.append('          END DO')
    L.append('        END DO')
    L.append('      END DO')
    if sri_volumetric:
        L.append('')
        L.append('C     Volumetric penalty at the centroid (SRI);')
        L.append('C     reuses the centroid dsh8/wdetJ from the gauge')
        L.append('C     pass above.')
        L.append('        DO i = 1, 3')
        L.append('          DO j = 1, 3')
        L.append('            F(i,j) = 0.0d0')
        L.append('            IF (i .EQ. j) F(i,j) = 1.0d0')
        L.append('            DO a_n = 1, NNODE_E')
        L.append('              F(i,j) = F(i,j) + u_node(i,a_n)*dsh8(a_n,j)')
        L.append('            END DO')
        L.append('          END DO')
        L.append('        END DO')
        L.append('      detF = det33d(F)')
        L.append('      IF (detF .LE. 0.0d0) THEN')
        L.append('        PNEWDT = 0.25d0')
        L.append('        RETURN')
        L.append('      END IF')
        L.append('      DO i = 1, 3')
        L.append('        DO j = 1, 3')
        L.append('          Fz(i,j) = DCMPLX(F(i,j), 0.0d0)')
        L.append('        END DO')
        L.append('      END DO')
        L.append(f'      CALL {p}_volumetric_PK1(Fz, PROPS, Pz)')
        L.append('      DO i = 1, 3')
        L.append('        DO j = 1, 3')
        L.append('          Svol(i,j) = DBLE(Pz(i,j))')
        L.append('        END DO')
        L.append('      END DO')
        L.append(f'      CALL {p}_cs_vol(F, PROPS, KVV)')
        L.append('')
        L.append('      DO a_n = 1, NNODE_E')
        L.append('        DO i = 1, 3')
        L.append('          row = edof_u(i, a_n)')
        L.append('          DO j = 1, 3')
        L.append('            RHS(row,1) = RHS(row,1)')
        L.append('     &        - Svol(i,j)*dsh8(a_n,j)*wdetJ')
        L.append('          END DO')
        L.append('          DO b_n = 1, NNODE_E')
        L.append('          DO k = 1, 3')
        L.append('            col = edof_u(k, b_n)')
        L.append('            DO j = 1, 3')
        L.append('              DO l = 1, 3')
        L.append('                AMATRX(row,col) = AMATRX(row,col)')
        L.append('     &            + dsh8(a_n,j)*KVV(i,j,k,l)')
        L.append('     &            * dsh8(b_n,l)*wdetJ')
        L.append('              END DO')
        L.append('            END DO')
        L.append('          END DO')
        L.append('          END DO')
        L.append('        END DO')
        L.append('      END DO')

    L.append('')
    L.append('      RETURN')
    L.append('      END SUBROUTINE UEL')
    return '\n'.join(_wrap_lines(L))
