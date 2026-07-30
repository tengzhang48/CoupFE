"""Prototype UEL generator for materials with element-local condensed pressure.

Supports three modes:

1. **Pure-mechanical u-p** (global displacement only):
   - Global fields: displacement ``u``
   - Local field: one constant pressure ``p_elem`` per element
   - Element: Quad4 or Hex8
   - The local pressure is stored in ``SVARS(1)`` and statically condensed from
     the element tangent before returning AMATRX/RHS.
   - The pressure equation ``p - K*lnJ = 0`` makes ``p`` the volume-average
     volumetric stress (L2 projection), the variationally-consistent alternative
     to F-bar.

2. **Mechanics with inelastic volume change** (global displacement only):
   - The material exposes ``inelastic_jacobian(F)`` returning ``J_inel``.
   - The generator forms the elastic Jacobian ``J_e = J / J_inel`` and passes
     ``J_inel`` to any material method that requests it (``stress_PK1``,
     ``pressure_resid``).  Setting ``J_inel = 1`` recovers the pure-mechanical
     path.

3. **Gel u-mu-p** (global displacement + chemical potential):
   - Global fields: displacement ``u`` and chemical potential ``mu``
   - Local field: one constant pressure ``p_elem`` per element
   - The gel transport (solvent flux/storage) is appended to the residual/tangent.

The local pressure is stored in ``SVARS(1)`` and statically condensed from the
element tangent before returning AMATRX/RHS to Abaqus/feacheap.
"""

from .element_config import ELEMENT_CONFIGS
from .uel_gen import (
    _generate_all_material_subs,
    _read_template,
    _wrap_lines,
)


def _mat_call_args(prefix, method, F_arg='Fz', p_arg='p_scalar_z',
                   mu_arg='muz', grad_arg='grad_mu_z',
                   J_inel_arg='J_inel_z',
                   Fold_arg='F_old_z', pold_arg='p_old_z',
                   dt_arg='dt', out_arg='out'):
    args = []
    for p in method['all_params']:
        if p == 'F':
            args.append(F_arg)
        elif p == 'p':
            args.append(p_arg)
        elif p == 'mu':
            args.append(mu_arg)
        elif p == 'grad_mu':
            args.append(grad_arg)
        elif p == 'J_inel':
            args.append(J_inel_arg)
        elif p == 'F_old':
            args.append(Fold_arg)
        elif p == 'p_old':
            args.append(pold_arg)
        elif p == 'dt':
            args.append(dt_arg)
        else:
            args.append(p)
    args.append('PROPS')
    args.append(out_arg)
    return f'CALL {prefix}({", ".join(args)})'


def _has_mu_transport(weakform):
    """Return True if the weak form has a global mu field + solvent methods."""
    mat = weakform._mat
    has_mu_field = 'mu' in getattr(weakform, 'field_names', [])
    has_methods = (
        'solvent_flux' in mat._methods and
        'solvent_storage' in mat._methods
    )
    return has_mu_field and has_methods


def _has_inelastic_jacobian(weakform):
    """Return True if the material exposes an inelastic Jacobian hook."""
    return 'inelastic_jacobian' in weakform._mat._methods


def _generate_local_pressure_cs(weakform, mat_prefix):
    mat = weakform._mat
    nprops = len(mat.props_names)
    has_mu = _has_mu_transport(weakform)

    stress = mat._methods['stress_PK1']
    press = mat._methods['pressure_resid']
    has_Jin = _has_inelastic_jacobian(weakform)
    J_inel_method = mat._methods.get('inelastic_jacobian')

    lines = []
    if has_mu:
        lines.append('      SUBROUTINE localp_eval_tangents(')
        lines.append('     &  F, p, mu, grad_mu, F_old, p_old, PROPS, dt,')
        lines.append('     &  P_real, rp_real, jR_real, cdot_real,')
        lines.append('     &  dP_dF, dP_dp, drp_dF, drp_dp, drp_dmu,')
        lines.append('     &  dflux_dF, dflux_dp, dflux_dmu, dflux_dgrad_mu,')
        lines.append('     &  dstorage_dF, dstorage_dp)')
        lines.append('      IMPLICIT NONE')
        lines.append(f'      DOUBLE PRECISION, INTENT(IN) :: F(3,3), p, mu')
        lines.append('      DOUBLE PRECISION, INTENT(IN) :: grad_mu(3)')
    else:
        lines.append('      SUBROUTINE localp_eval_tangents(')
        lines.append('     &  F, p, F_old, p_old, PROPS, dt,')
        lines.append('     &  P_real, rp_real,')
        lines.append('     &  dP_dF, dP_dp, drp_dF, drp_dp)')
        lines.append('      IMPLICIT NONE')
        lines.append(f'      DOUBLE PRECISION, INTENT(IN) :: F(3,3), p')

    lines.append('      DOUBLE PRECISION, INTENT(IN) :: F_old(3,3), p_old')
    lines.append(f'      DOUBLE PRECISION, INTENT(IN) :: PROPS({nprops}), dt')
    lines.append('      DOUBLE PRECISION, INTENT(OUT) :: P_real(3,3)')
    lines.append('      DOUBLE PRECISION, INTENT(OUT) :: rp_real')
    if has_mu:
        lines.append('      DOUBLE PRECISION, INTENT(OUT) :: jR_real(3)')
        lines.append('      DOUBLE PRECISION, INTENT(OUT) :: cdot_real')
    lines.append('      DOUBLE PRECISION, INTENT(OUT) :: dP_dF(3,3,3,3)')
    lines.append('      DOUBLE PRECISION, INTENT(OUT) :: dP_dp(3,3)')
    lines.append('      DOUBLE PRECISION, INTENT(OUT) :: drp_dF(3,3)')
    lines.append('      DOUBLE PRECISION, INTENT(OUT) :: drp_dp')
    if has_mu:
        lines.append('      DOUBLE PRECISION, INTENT(OUT) :: drp_dmu')
        lines.append('      DOUBLE PRECISION, INTENT(OUT) :: dflux_dF(3,3,3)')
        lines.append('      DOUBLE PRECISION, INTENT(OUT) :: dflux_dp(3)')
        lines.append('      DOUBLE PRECISION, INTENT(OUT) :: dflux_dmu(3)')
        lines.append('      DOUBLE PRECISION, INTENT(OUT) :: dflux_dgrad_mu(3,3)')
        lines.append('      DOUBLE PRECISION, INTENT(OUT) :: dstorage_dF(3,3)')
        lines.append('      DOUBLE PRECISION, INTENT(OUT) :: dstorage_dp')
    lines.append('      DOUBLE PRECISION, PARAMETER :: CS_H = 1.0d-10')
    lines.append('      DOUBLE COMPLEX :: Fz(3,3), F_old_z(3,3)')
    lines.append('      DOUBLE COMPLEX :: p_scalar_z, p_old_z')
    if has_mu:
        lines.append('      DOUBLE COMPLEX :: muz, grad_mu_z(3)')
    lines.append('      DOUBLE COMPLEX :: stress_z(3,3), rp_z')
    lines.append('      DOUBLE COMPLEX :: J_inel_z')
    if has_mu:
        lines.append('      DOUBLE COMPLEX :: flux_z(3), cdot_z')
    lines.append('      INTEGER :: i, j, k, l')
    lines.append('')

    def _set_J_inel(F_arg='Fz'):
        if has_Jin:
            return (f'      {_mat_call_args(mat_prefix + "_inelastic_jacobian", J_inel_method, F_arg=F_arg, out_arg="J_inel_z")}')
        return '      J_inel_z = DCMPLX(1.0d0, 0.0d0)'

    lines.append('C     Real outputs')
    lines.append('      CALL real2complex33(F, Fz)')
    lines.append('      CALL real2complex33(F_old, F_old_z)')
    lines.append('      p_scalar_z = DCMPLX(p, 0.0d0)')
    if has_mu:
        lines.append('      muz = DCMPLX(mu, 0.0d0)')
    lines.append('      p_old_z = DCMPLX(p_old, 0.0d0)')
    if has_mu:
        lines.append('      CALL real2complex3(grad_mu, grad_mu_z)')
    lines.append(f'      {_set_J_inel()}')
    lines.append(f'      {_mat_call_args(mat_prefix + "_stress_PK1", stress, out_arg="stress_z")}')
    lines.append(f'      {_mat_call_args(mat_prefix + "_pressure_resid", press, out_arg="rp_z")}')
    if has_mu:
        flux = mat._methods['solvent_flux']
        storage = mat._methods['solvent_storage']
        lines.append(f'      {_mat_call_args(mat_prefix + "_solvent_flux", flux, out_arg="flux_z")}')
        lines.append(f'      {_mat_call_args(mat_prefix + "_solvent_storage", storage, out_arg="cdot_z")}')
    lines.append('      DO i = 1, 3')
    lines.append('        DO j = 1, 3')
    lines.append('          P_real(i,j) = DBLE(stress_z(i,j))')
    lines.append('        END DO')
    lines.append('      END DO')
    lines.append('      rp_real = DBLE(rp_z)')
    if has_mu:
        lines.append('      DO i = 1, 3')
        lines.append('        jR_real(i) = DBLE(flux_z(i))')
        lines.append('      END DO')
        lines.append('      cdot_real = DBLE(cdot_z)')
    lines.append('')

    lines.append('C     F perturbations')
    lines.append('      DO k = 1, 3')
    lines.append('        DO l = 1, 3')
    lines.append('          CALL real2complex33(F, Fz)')
    lines.append('          CALL real2complex33(F_old, F_old_z)')
    lines.append('          p_scalar_z = DCMPLX(p, 0.0d0)')
    if has_mu:
        lines.append('          muz = DCMPLX(mu, 0.0d0)')
    lines.append('          p_old_z = DCMPLX(p_old, 0.0d0)')
    if has_mu:
        lines.append('          CALL real2complex3(grad_mu, grad_mu_z)')
    lines.append('          Fz(k,l) = Fz(k,l) + DCMPLX(0.0d0, CS_H)')
    lines.append(f'          {_set_J_inel()}')
    lines.append(f'          {_mat_call_args(mat_prefix + "_stress_PK1", stress, out_arg="stress_z")}')
    lines.append(f'          {_mat_call_args(mat_prefix + "_pressure_resid", press, out_arg="rp_z")}')
    if has_mu:
        lines.append(f'          {_mat_call_args(mat_prefix + "_solvent_flux", flux, out_arg="flux_z")}')
        lines.append(f'          {_mat_call_args(mat_prefix + "_solvent_storage", storage, out_arg="cdot_z")}')
    lines.append('          DO i = 1, 3')
    lines.append('            DO j = 1, 3')
    lines.append('              dP_dF(i,j,k,l) = AIMAG(stress_z(i,j)) / CS_H')
    lines.append('            END DO')
    lines.append('          END DO')
    lines.append('          drp_dF(k,l) = AIMAG(rp_z) / CS_H')
    if has_mu:
        lines.append('          DO i = 1, 3')
        lines.append('            dflux_dF(i,k,l) = AIMAG(flux_z(i)) / CS_H')
        lines.append('          END DO')
        lines.append('          dstorage_dF(k,l) = AIMAG(cdot_z) / CS_H')
    lines.append('        END DO')
    lines.append('      END DO')
    lines.append('')

    lines.append('C     p perturbation')
    lines.append('      CALL real2complex33(F, Fz)')
    lines.append('      CALL real2complex33(F_old, F_old_z)')
    lines.append('      p_scalar_z = DCMPLX(p, CS_H)')
    if has_mu:
        lines.append('      muz = DCMPLX(mu, 0.0d0)')
    lines.append('      p_old_z = DCMPLX(p_old, 0.0d0)')
    if has_mu:
        lines.append('      CALL real2complex3(grad_mu, grad_mu_z)')
    lines.append(f'      {_set_J_inel()}')
    lines.append(f'      {_mat_call_args(mat_prefix + "_stress_PK1", stress, out_arg="stress_z")}')
    lines.append(f'      {_mat_call_args(mat_prefix + "_pressure_resid", press, out_arg="rp_z")}')
    if has_mu:
        lines.append(f'      {_mat_call_args(mat_prefix + "_solvent_flux", flux, out_arg="flux_z")}')
        lines.append(f'      {_mat_call_args(mat_prefix + "_solvent_storage", storage, out_arg="cdot_z")}')
    lines.append('      DO i = 1, 3')
    lines.append('        DO j = 1, 3')
    lines.append('          dP_dp(i,j) = AIMAG(stress_z(i,j)) / CS_H')
    lines.append('        END DO')
    lines.append('      END DO')
    lines.append('      drp_dp = AIMAG(rp_z) / CS_H')
    if has_mu:
        lines.append('      DO i = 1, 3')
        lines.append('        dflux_dp(i) = AIMAG(flux_z(i)) / CS_H')
        lines.append('      END DO')
        lines.append('      dstorage_dp = AIMAG(cdot_z) / CS_H')
    lines.append('')

    if has_mu:
        lines.append('C     mu perturbation')
        lines.append('      CALL real2complex33(F, Fz)')
        lines.append('      CALL real2complex33(F_old, F_old_z)')
        lines.append('      p_scalar_z = DCMPLX(p, 0.0d0)')
        lines.append('      muz = DCMPLX(mu, CS_H)')
        lines.append('      p_old_z = DCMPLX(p_old, 0.0d0)')
        lines.append('      CALL real2complex3(grad_mu, grad_mu_z)')
        lines.append(f'      {_set_J_inel()}')
        lines.append(f'      {_mat_call_args(mat_prefix + "_pressure_resid", press, out_arg="rp_z")}')
        lines.append(f'      {_mat_call_args(mat_prefix + "_solvent_flux", flux, out_arg="flux_z")}')
        lines.append('      drp_dmu = AIMAG(rp_z) / CS_H')
        lines.append('      DO i = 1, 3')
        lines.append('        dflux_dmu(i) = AIMAG(flux_z(i)) / CS_H')
        lines.append('      END DO')
        lines.append('')

        lines.append('C     grad(mu) perturbations')
        lines.append('      DO k = 1, 3')
        lines.append('        CALL real2complex33(F, Fz)')
        lines.append('        p_scalar_z = DCMPLX(p, 0.0d0)')
        lines.append('        muz = DCMPLX(mu, 0.0d0)')
        lines.append('        CALL real2complex3(grad_mu, grad_mu_z)')
        lines.append('        grad_mu_z(k) = grad_mu_z(k)')
        lines.append('     &    + DCMPLX(0.0d0, CS_H)')
        lines.append(f'        {_mat_call_args(mat_prefix + "_solvent_flux", flux, out_arg="flux_z")}')
        lines.append('        DO i = 1, 3')
        lines.append('          dflux_dgrad_mu(i,k) = AIMAG(flux_z(i)) / CS_H')
        lines.append('        END DO')
        lines.append('      END DO')
        lines.append('')

    lines.append('      RETURN')
    lines.append('      END SUBROUTINE localp_eval_tangents')
    return '\n'.join(lines)


def _generate_uel_local_pressure(weakform, mat_prefix, cfg):
    if cfg.name not in ('quad4', 'hex8'):
        raise ValueError("local_pressure formulation currently supports "
                         "Quad4 and Hex8")
    has_mu = _has_mu_transport(weakform)
    nprops = len(weakform._mat.props_names)
    ndofel = cfg.n_nodes * (cfg.ndim + 1) if has_mu else cfg.n_nodes * cfg.ndim

    lines = []
    lines.append('      SUBROUTINE UEL(RHS, AMATRX, SVARS, ENERGY, NDOFEL,')
    lines.append('     &  NRHS, NSVARS, PROPS, NPROPS, COORDS, MCRD, NNODE,')
    lines.append('     &  U, DU, V, A, JTYPE, TIME, DTIME, KSTEP, KINC,')
    lines.append('     &  JELEM, PARAMS, NDLOAD, JDLTYP, ADLMAG, PREDEF,')
    lines.append('     &  NPREDF, LFLAGS, MLVARX, DDLMAG, MDLOAD, PNEWDT,')
    lines.append('     &  JPROPS, NJPROP, PERIOD)')
    lines.append('      USE localp_uvarm_bridge')
    lines.append('      IMPLICIT NONE')
    lines.append('      INTEGER NDOFEL, NRHS, NSVARS, NPROPS, MCRD, NNODE')
    lines.append('      INTEGER JTYPE, KSTEP, KINC, JELEM, NDLOAD, NPREDF')
    lines.append('      INTEGER MLVARX, MDLOAD, NJPROP')
    lines.append('      INTEGER JDLTYP(MDLOAD,*), LFLAGS(*), JPROPS(*)')
    lines.append('      DOUBLE PRECISION RHS(MLVARX,*), AMATRX(NDOFEL,NDOFEL)')
    lines.append('      DOUBLE PRECISION SVARS(NSVARS), ENERGY(8)')
    lines.append('      DOUBLE PRECISION PROPS(NPROPS), COORDS(MCRD,NNODE)')
    lines.append('      DOUBLE PRECISION U(NDOFEL), DU(MLVARX,*), V(NDOFEL), A(NDOFEL)')
    lines.append('      DOUBLE PRECISION TIME(2), DTIME, PARAMS(3)')
    lines.append('      DOUBLE PRECISION ADLMAG(MDLOAD,*), PREDEF(2,NPREDF,NNODE)')
    lines.append('      DOUBLE PRECISION DDLMAG(MDLOAD,*), PNEWDT, PERIOD')
    lines.append(f'      INTEGER, PARAMETER :: ndim = {cfg.ndim},'
                 f' NNODE_E = {cfg.n_nodes}, NGP = {cfg.n_gauss_points}')
    lines.append('      DOUBLE PRECISION :: u_node(ndim,NNODE_E), u_old(ndim,NNODE_E)')
    if has_mu:
        lines.append('      DOUBLE PRECISION :: mu_node(NNODE_E), mu_old(NNODE_E)')
        lines.append('      INTEGER :: edof_mu(NNODE_E)')
    lines.append('      INTEGER :: edof_u(ndim,NNODE_E)')
    lines.append('      DOUBLE PRECISION :: coords2(ndim,NNODE_E)')
    lines.append('      DOUBLE PRECISION :: sh4(NNODE_E), dshxi4(NNODE_E,ndim)')
    lines.append('      DOUBLE PRECISION :: dsh4(NNODE_E,ndim), xi_gp(NGP,ndim), w_gp(NGP)')
    lines.append('      DOUBLE PRECISION :: detJ, Jinv(ndim,ndim), wdetJ, dt_safe')
    lines.append('      DOUBLE PRECISION :: F(3,3), F_old(3,3)')
    if has_mu:
        lines.append('      DOUBLE PRECISION :: grad_mu(3), mu_gp')
    lines.append('      DOUBLE PRECISION :: p_elem, p_old, R_p, K_pp, dp')
    lines.append('      DOUBLE PRECISION :: detF_diag, Je_diag, phi_diag, det33d')
    lines.append('      DOUBLE PRECISION :: P_real(3,3), rp_real')
    if has_mu:
        lines.append('      DOUBLE PRECISION :: jR_real(3), cdot_real')
        lines.append('      DOUBLE PRECISION :: dflux_dF(3,3,3), dflux_dp(3)')
        lines.append('      DOUBLE PRECISION :: dflux_dmu(3), dflux_dgrad_mu(3,3)')
        lines.append('      DOUBLE PRECISION :: dstorage_dF(3,3), dstorage_dp')
    lines.append('      DOUBLE PRECISION :: dP_dF(3,3,3,3), dP_dp(3,3)')
    lines.append('      DOUBLE PRECISION :: drp_dF(3,3), drp_dp')
    if has_mu:
        lines.append('      DOUBLE PRECISION :: drp_dmu')
    lines.append('      DOUBLE PRECISION :: Kxp(NDOFEL), Kpx(NDOFEL), Rtmp(NDOFEL)')
    lines.append('      DOUBLE PRECISION :: Kxx(NDOFEL,NDOFEL), Rp_final')
    lines.append('      INTEGER :: ngp_out, stat, idx, ii_v, jj_v')
    lines.append('      INTEGER :: i, j, k, l, kk, iter_p, row, col')
    lines.append('')
    lines.append('      DO ii_v = 1, NNODE_E')
    lines.append('        DO i = 1, ndim')
    lines.append('          coords2(i,ii_v) = COORDS(i,ii_v)')
    lines.append('        END DO')
    lines.append('      END DO')
    lines.append('      DO i = 1, NDOFEL')
    lines.append('        RHS(i,1) = 0.0d0')
    lines.append('        Rtmp(i) = 0.0d0')
    lines.append('        Kxp(i) = 0.0d0')
    lines.append('        Kpx(i) = 0.0d0')
    lines.append('        DO j = 1, NDOFEL')
    lines.append('          AMATRX(i,j) = 0.0d0')
    lines.append('          Kxx(i,j) = 0.0d0')
    lines.append('        END DO')
    lines.append('      END DO')
    lines.append('      dt_safe = DTIME')
    lines.append('      IF (dt_safe .LT. 1.0d-14) dt_safe = 1.0d-14')
    lines.append('')
    lines.append('      idx = 0')
    lines.append('      DO ii_v = 1, NNODE_E')
    lines.append('        DO i = 1, ndim')
    lines.append('          idx = idx + 1')
    lines.append('          u_node(i,ii_v) = U(idx)')
    lines.append('          u_old(i,ii_v) = U(idx) - DU(idx,1)')
    lines.append('          edof_u(i,ii_v) = idx')
    lines.append('        END DO')
    if has_mu:
        lines.append('        idx = idx + 1')
        lines.append('        mu_node(ii_v) = U(idx)')
        lines.append('        mu_old(ii_v) = U(idx) - DU(idx,1)')
        lines.append('        edof_mu(ii_v) = idx')
    lines.append('      END DO')
    lines.append('')
    lines.append('      p_old = 0.0d0')
    lines.append('      IF (NSVARS .GE. 1) p_old = SVARS(1)')
    lines.append('      p_elem = p_old')
    lines.append(f'      CALL {cfg.gauss_subroutine}(xi_gp, w_gp, ngp_out)')
    lines.append('')
    lines.append('C     Local Newton for element pressure')
    lines.append('      DO iter_p = 1, 25')
    lines.append('        R_p = 0.0d0')
    lines.append('        K_pp = 0.0d0')
    lines.append('        DO kk = 1, ngp_out')
    lines.append(f'          CALL {cfg.shape_subroutine}({cfg.gauss_xi_args},')
    lines.append('     &      sh4, dshxi4)')
    lines.append(f'          CALL {cfg.jac_subroutine}(dshxi4, coords2, NNODE_E, dsh4,')
    lines.append('     &      detJ, Jinv, stat)')
    lines.append('          IF (stat .EQ. 0) THEN')
    lines.append('            PNEWDT = 0.25d0')
    lines.append('            RETURN')
    lines.append('          END IF')
    lines.append('          wdetJ = detJ * w_gp(kk)')
    lines.append('          CALL eye33d(F)')
    lines.append('          CALL eye33d(F_old)')
    if has_mu:
        lines.append('          mu_gp = 0.0d0')
        lines.append('          grad_mu(1) = 0.0d0')
        lines.append('          grad_mu(2) = 0.0d0')
        lines.append('          grad_mu(3) = 0.0d0')
    lines.append('          DO ii_v = 1, NNODE_E')
    if has_mu:
        lines.append('            mu_gp = mu_gp + mu_node(ii_v)*sh4(ii_v)')
        lines.append('            DO i = 1, ndim')
        lines.append('              grad_mu(i) = grad_mu(i)')
        lines.append('     &          + mu_node(ii_v)*dsh4(ii_v,i)')
        lines.append('              DO j = 1, ndim')
        lines.append('                F(i,j) = F(i,j)')
        lines.append('     &            + u_node(i,ii_v)*dsh4(ii_v,j)')
        lines.append('                F_old(i,j) = F_old(i,j)')
        lines.append('     &            + u_old(i,ii_v)*dsh4(ii_v,j)')
        lines.append('              END DO')
        lines.append('            END DO')
    else:
        lines.append('            DO i = 1, ndim')
        lines.append('              DO j = 1, ndim')
        lines.append('                F(i,j) = F(i,j)')
        lines.append('     &            + u_node(i,ii_v)*dsh4(ii_v,j)')
        lines.append('                F_old(i,j) = F_old(i,j)')
        lines.append('     &            + u_old(i,ii_v)*dsh4(ii_v,j)')
        lines.append('              END DO')
        lines.append('            END DO')
    lines.append('          END DO')

    if has_mu:
        lines.append('          CALL localp_eval_tangents(F, p_elem, mu_gp,')
        lines.append('     &      grad_mu, F_old, p_old, PROPS, dt_safe,')
        lines.append('     &      P_real, rp_real, jR_real, cdot_real,')
        lines.append('     &      dP_dF, dP_dp, drp_dF, drp_dp, drp_dmu,')
        lines.append('     &      dflux_dF, dflux_dp, dflux_dmu,')
        lines.append('     &      dflux_dgrad_mu, dstorage_dF, dstorage_dp)')
    else:
        lines.append('          CALL localp_eval_tangents(F, p_elem,')
        lines.append('     &      F_old, p_old, PROPS, dt_safe,')
        lines.append('     &      P_real, rp_real,')
        lines.append('     &      dP_dF, dP_dp, drp_dF, drp_dp)')
    lines.append('          R_p = R_p + rp_real*wdetJ')
    lines.append('          K_pp = K_pp + drp_dp*wdetJ')
    lines.append('        END DO')
    lines.append('        IF (DABS(K_pp) .LT. 1.0d-30) THEN')
    lines.append('          PNEWDT = 0.25d0')
    lines.append('          RETURN')
    lines.append('        END IF')
    lines.append('        dp = -R_p / K_pp')
    lines.append('        p_elem = p_elem + dp')
    lines.append('        IF (DABS(R_p) .LT. 1.0d-10) EXIT')
    lines.append('        IF (DABS(dp) .LT. 1.0d-10*(1.0d0+DABS(p_elem))) EXIT')
    lines.append('      END DO')
    lines.append('')
    lines.append('C     Assemble full x-p blocks, x = [u] (mechanical) or [u, mu] (gel)')
    lines.append('      R_p = 0.0d0')
    lines.append('      K_pp = 0.0d0')
    lines.append('      DO kk = 1, ngp_out')
    lines.append(f'        CALL {cfg.shape_subroutine}({cfg.gauss_xi_args}, sh4, dshxi4)')
    lines.append(f'        CALL {cfg.jac_subroutine}(dshxi4, coords2, NNODE_E, dsh4,')
    lines.append('     &    detJ, Jinv, stat)')
    lines.append('        IF (stat .EQ. 0) THEN')
    lines.append('          PNEWDT = 0.25d0')
    lines.append('          RETURN')
    lines.append('        END IF')
    lines.append('        wdetJ = detJ * w_gp(kk)')
    lines.append('        CALL eye33d(F)')
    lines.append('        CALL eye33d(F_old)')
    if has_mu:
        lines.append('        mu_gp = 0.0d0')
        lines.append('        grad_mu(1) = 0.0d0')
        lines.append('        grad_mu(2) = 0.0d0')
        lines.append('        grad_mu(3) = 0.0d0')
    lines.append('        DO ii_v = 1, NNODE_E')
    if has_mu:
        lines.append('          mu_gp = mu_gp + mu_node(ii_v)*sh4(ii_v)')
        lines.append('          DO i = 1, ndim')
        lines.append('            grad_mu(i) = grad_mu(i)')
        lines.append('     &        + mu_node(ii_v)*dsh4(ii_v,i)')
        lines.append('            DO j = 1, ndim')
        lines.append('              F(i,j) = F(i,j)')
        lines.append('     &          + u_node(i,ii_v)*dsh4(ii_v,j)')
        lines.append('              F_old(i,j) = F_old(i,j)')
        lines.append('     &          + u_old(i,ii_v)*dsh4(ii_v,j)')
        lines.append('            END DO')
        lines.append('          END DO')
    else:
        lines.append('          DO i = 1, ndim')
        lines.append('            DO j = 1, ndim')
        lines.append('              F(i,j) = F(i,j)')
        lines.append('     &          + u_node(i,ii_v)*dsh4(ii_v,j)')
        lines.append('              F_old(i,j) = F_old(i,j)')
        lines.append('     &          + u_old(i,ii_v)*dsh4(ii_v,j)')
        lines.append('            END DO')
        lines.append('          END DO')
    lines.append('        END DO')

    if has_mu:
        lines.append('        CALL localp_eval_tangents(F, p_elem, mu_gp, grad_mu,')
        lines.append('     &    F_old, p_old, PROPS, dt_safe,')
        lines.append('     &    P_real, rp_real, jR_real, cdot_real,')
        lines.append('     &    dP_dF, dP_dp, drp_dF, drp_dp, drp_dmu,')
        lines.append('     &    dflux_dF, dflux_dp, dflux_dmu,')
        lines.append('     &    dflux_dgrad_mu, dstorage_dF, dstorage_dp)')
    else:
        lines.append('        CALL localp_eval_tangents(F, p_elem, F_old, p_old,')
        lines.append('     &    PROPS, dt_safe,')
        lines.append('     &    P_real, rp_real,')
        lines.append('     &    dP_dF, dP_dp, drp_dF, drp_dp)')
    lines.append('        detF_diag = det33d(F)')
    lines.append('        Je_diag = DEXP(p_elem / PROPS(2))')
    lines.append('        phi_diag = 0.0d0')
    lines.append('        IF (NPROPS .GE. 9 .AND. DABS(detF_diag) .GT. 1.0d-30)')
    lines.append('     &    phi_diag = PROPS(9) * Je_diag / detF_diag')
    lines.append('        CALL localp_store_uvarm(JELEM, kk, phi_diag, p_elem)')
    lines.append('        IF (NSVARS .GE. 1 + NGP) SVARS(1+kk) = phi_diag')
    lines.append('        IF (NSVARS .GE. 1 + 2*NGP) SVARS(1+NGP+kk) = p_elem')
    lines.append('        R_p = R_p + rp_real*wdetJ')
    lines.append('        K_pp = K_pp + drp_dp*wdetJ')
    lines.append('')
    # RHS u
    lines.append('        DO ii_v = 1, NNODE_E')
    lines.append('          DO i = 1, ndim')
    lines.append('            row = edof_u(i,ii_v)')
    lines.append('            DO j = 1, ndim')
    lines.append('              Rtmp(row) = Rtmp(row)')
    lines.append('     &          - P_real(i,j)*dsh4(ii_v,j)*wdetJ')
    lines.append('            END DO')
    lines.append('          END DO')
    if has_mu:
        lines.append('          row = edof_mu(ii_v)')
        lines.append('          Rtmp(row) = Rtmp(row) - cdot_real*sh4(ii_v)*wdetJ')
        lines.append('          DO j = 1, ndim')
        lines.append('            Rtmp(row) = Rtmp(row)')
        lines.append('     &        + jR_real(j)*dsh4(ii_v,j)*wdetJ')
        lines.append('          END DO')
    lines.append('        END DO')
    # Kxp and Kpx
    lines.append('        DO ii_v = 1, NNODE_E')
    lines.append('          DO i = 1, ndim')
    lines.append('            row = edof_u(i,ii_v)')
    lines.append('            DO j = 1, ndim')
    lines.append('              Kxp(row) = Kxp(row)')
    lines.append('     &          + dP_dp(i,j)*dsh4(ii_v,j)*wdetJ')
    lines.append('            END DO')
    lines.append('          END DO')
    if has_mu:
        lines.append('          row = edof_mu(ii_v)')
        lines.append('          Kxp(row) = Kxp(row) + dstorage_dp*sh4(ii_v)*wdetJ')
        lines.append('          DO j = 1, ndim')
        lines.append('            Kxp(row) = Kxp(row)')
        lines.append('     &        - dflux_dp(j)*dsh4(ii_v,j)*wdetJ')
        lines.append('          END DO')
    lines.append('          DO k = 1, ndim')
    lines.append('            col = edof_u(k,ii_v)')
    lines.append('            DO l = 1, ndim')
    lines.append('              Kpx(col) = Kpx(col)')
    lines.append('     &          + drp_dF(k,l)*dsh4(ii_v,l)*wdetJ')
    lines.append('            END DO')
    lines.append('          END DO')
    if has_mu:
        lines.append('          col = edof_mu(ii_v)')
        lines.append('          Kpx(col) = Kpx(col) + drp_dmu*sh4(ii_v)*wdetJ')
    lines.append('        END DO')
    # Kxx
    lines.append('        DO ii_v = 1, NNODE_E')
    lines.append('          DO jj_v = 1, NNODE_E')
    lines.append('            DO i = 1, ndim')
    lines.append('              row = edof_u(i,ii_v)')
    lines.append('              DO k = 1, ndim')
    lines.append('                col = edof_u(k,jj_v)')
    lines.append('                DO j = 1, ndim')
    lines.append('                  DO l = 1, ndim')
    lines.append('                    Kxx(row,col) = Kxx(row,col)')
    lines.append('     &                + dP_dF(i,j,k,l)')
    lines.append('     &                * dsh4(ii_v,j)*dsh4(jj_v,l)*wdetJ')
    lines.append('                  END DO')
    lines.append('                END DO')
    lines.append('              END DO')
    if has_mu:
        lines.append('              col = edof_mu(jj_v)')
        lines.append('C             dP/dmu = 0 for this gel')
        lines.append('            END DO')
    else:
        lines.append('            END DO')
    if has_mu:
        lines.append('            row = edof_mu(ii_v)')
        lines.append('            DO k = 1, ndim')
        lines.append('              col = edof_u(k,jj_v)')
        lines.append('              DO l = 1, ndim')
        lines.append('                Kxx(row,col) = Kxx(row,col)')
        lines.append('     &            + dstorage_dF(k,l)*sh4(ii_v)')
        lines.append('     &            * dsh4(jj_v,l)*wdetJ')
        lines.append('                DO j = 1, ndim')
        lines.append('                  Kxx(row,col) = Kxx(row,col)')
        lines.append('     &              - dflux_dF(j,k,l)*dsh4(ii_v,j)')
        lines.append('     &              * dsh4(jj_v,l)*wdetJ')
        lines.append('                END DO')
        lines.append('              END DO')
        lines.append('            END DO')
        lines.append('            col = edof_mu(jj_v)')
        lines.append('C           storage has no d/dmu term')
        lines.append('            DO j = 1, ndim')
        lines.append('              Kxx(row,col) = Kxx(row,col)')
        lines.append('     &          - dflux_dmu(j)*dsh4(ii_v,j)*sh4(jj_v)*wdetJ')
        lines.append('              DO l = 1, ndim')
        lines.append('                Kxx(row,col) = Kxx(row,col)')
        lines.append('     &            - dflux_dgrad_mu(j,l)*dsh4(ii_v,j)')
        lines.append('     &            * dsh4(jj_v,l)*wdetJ')
        lines.append('              END DO')
        lines.append('            END DO')
    lines.append('          END DO')
    lines.append('        END DO')
    lines.append('      END DO')
    lines.append('')
    lines.append('      IF (DABS(K_pp) .LT. 1.0d-30) THEN')
    lines.append('        PNEWDT = 0.25d0')
    lines.append('        RETURN')
    lines.append('      END IF')
    lines.append('      Rp_final = R_p')
    # Schur complement of K [dx; dp] = [-r_x; -r_p]: eliminating dp gives
    # (Kxx - Kxp Kpx/Kpp) dx = -r_x + Kxp r_p/Kpp. Rtmp = -r_x, R_p = +r_p,
    # so the coupling term ADDS (audit 2026-06-10 finding M10).
    lines.append('      DO i = 1, NDOFEL')
    lines.append('        RHS(i,1) = Rtmp(i) + Kxp(i)*Rp_final/K_pp')
    lines.append('        DO j = 1, NDOFEL')
    lines.append('          AMATRX(i,j) = Kxx(i,j) - Kxp(i)*Kpx(j)/K_pp')
    lines.append('        END DO')
    lines.append('      END DO')
    lines.append('      IF (NSVARS .GE. 1) SVARS(1) = p_elem')
    lines.append('C     Optional diagnostic layouts.')
    lines.append('C     Preserve solver state in SVARS(1).  If enough slots exist,')
    lines.append('C       store UVARM-like diagnostics by integration point:')
    lines.append('C       SVARS(1+i)       = phi at GP i')
    lines.append('C       SVARS(1+NGP+i)   = local pressure at GP i')
    lines.append('      IF (ndim .EQ. 2 .AND. NSVARS .GE. 32) THEN')
    lines.append('        SVARS(1) = p_elem')
    lines.append('        SVARS(9) = p_elem')
    lines.append('        SVARS(17) = p_elem')
    lines.append('        SVARS(25) = p_elem')
    lines.append('      END IF')
    lines.append('      RETURN')
    lines.append('      END SUBROUTINE UEL')
    return '\n'.join(lines), ndofel


def _generate_uvarm_bridge():
    """Generate an Abaqus UVARM bridge for local-pressure diagnostics.

    UVARM is not called by feacheap.  The same diagnostics are also written to
    SVARS by the generated UEL so feacheap can project them as U1/U2.
    """
    lines = []
    lines.append('      MODULE localp_uvarm_bridge')
    lines.append('      IMPLICIT NONE')
    lines.append('      INTEGER, PARAMETER :: LOCALP_UVARM_OFFSET = 10000')
    lines.append('      INTEGER, PARAMETER :: LOCALP_MAX_GP = 64')
    lines.append('      INTEGER, PARAMETER :: LOCALP_NUVAR = 2')
    lines.append('      DOUBLE PRECISION, ALLOCATABLE :: LOCALP_UVAR(:,:,:)')
    lines.append('      CONTAINS')
    lines.append('      SUBROUTINE localp_ensure_capacity(noel)')
    lines.append('      IMPLICIT NONE')
    lines.append('      INTEGER, INTENT(IN) :: noel')
    lines.append('      INTEGER :: oldn, newn')
    lines.append('      DOUBLE PRECISION, ALLOCATABLE :: tmp(:,:,:)')
    lines.append('      IF (noel .LE. 0) RETURN')
    lines.append('      IF (.NOT. ALLOCATED(LOCALP_UVAR)) THEN')
    lines.append('        newn = MAX(1024, noel)')
    lines.append('        ALLOCATE(LOCALP_UVAR(newn,LOCALP_MAX_GP,LOCALP_NUVAR))')
    lines.append('        LOCALP_UVAR = 0.0d0')
    lines.append('      ELSE IF (noel .GT. SIZE(LOCALP_UVAR,1)) THEN')
    lines.append('        oldn = SIZE(LOCALP_UVAR,1)')
    lines.append('        newn = MAX(noel, 2*oldn)')
    lines.append('        ALLOCATE(tmp(oldn,LOCALP_MAX_GP,LOCALP_NUVAR))')
    lines.append('        tmp = LOCALP_UVAR')
    lines.append('        DEALLOCATE(LOCALP_UVAR)')
    lines.append('        ALLOCATE(LOCALP_UVAR(newn,LOCALP_MAX_GP,LOCALP_NUVAR))')
    lines.append('        LOCALP_UVAR = 0.0d0')
    lines.append('        LOCALP_UVAR(1:oldn,:,:) = tmp')
    lines.append('        DEALLOCATE(tmp)')
    lines.append('      END IF')
    lines.append('      RETURN')
    lines.append('      END SUBROUTINE localp_ensure_capacity')
    lines.append('')
    lines.append('      SUBROUTINE localp_store_uvarm(noel, npt, phi, p)')
    lines.append('      IMPLICIT NONE')
    lines.append('      INTEGER, INTENT(IN) :: noel, npt')
    lines.append('      DOUBLE PRECISION, INTENT(IN) :: phi, p')
    lines.append('      IF (npt .LT. 1 .OR. npt .GT. LOCALP_MAX_GP) RETURN')
    lines.append('      CALL localp_ensure_capacity(noel)')
    lines.append('      LOCALP_UVAR(noel,npt,1) = phi')
    lines.append('      LOCALP_UVAR(noel,npt,2) = p')
    lines.append('      RETURN')
    lines.append('      END SUBROUTINE localp_store_uvarm')
    lines.append('')
    lines.append('      SUBROUTINE localp_fetch_uvarm(noel, npt, uvar, nuvar)')
    lines.append('      IMPLICIT NONE')
    lines.append('      INTEGER, INTENT(IN) :: noel, npt, nuvar')
    lines.append('      DOUBLE PRECISION, INTENT(OUT) :: uvar(nuvar)')
    lines.append('      INTEGER :: src, i')
    lines.append('      DO i = 1, nuvar')
    lines.append('        uvar(i) = 0.0d0')
    lines.append('      END DO')
    lines.append('      src = noel - LOCALP_UVARM_OFFSET')
    lines.append('      IF (.NOT. ALLOCATED(LOCALP_UVAR)) RETURN')
    lines.append('      IF (src .LT. 1 .OR. src .GT. SIZE(LOCALP_UVAR,1)) RETURN')
    lines.append('      IF (npt .LT. 1 .OR. npt .GT. LOCALP_MAX_GP) RETURN')
    lines.append('      DO i = 1, MIN(nuvar, LOCALP_NUVAR)')
    lines.append('        uvar(i) = LOCALP_UVAR(src,npt,i)')
    lines.append('      END DO')
    lines.append('      RETURN')
    lines.append('      END SUBROUTINE localp_fetch_uvarm')
    lines.append('      END MODULE localp_uvarm_bridge')
    lines.append('')
    lines.append('      SUBROUTINE UVARM(UVAR,DIRECT,T,TIME,DTIME,CMNAME,')
    lines.append('     &  ORNAME,NUVARM,NOEL,NPT,LAYER,KSPT,KSTEP,KINC,')
    lines.append('     &  NDI,NSHR,COORD,JMAC,JMATYP,MATLAYO,LACCFLA)')
    lines.append('      USE localp_uvarm_bridge')
    lines.append('      IMPLICIT NONE')
    lines.append('      INTEGER NUVARM, NOEL, NPT, LAYER, KSPT, KSTEP, KINC')
    lines.append('      INTEGER NDI, NSHR, MATLAYO, LACCFLA')
    lines.append('      INTEGER JMAC(*), JMATYP(*)')
    lines.append('      DOUBLE PRECISION UVAR(NUVARM), DIRECT(3,3), T(3,3)')
    lines.append('      DOUBLE PRECISION TIME(2), DTIME, COORD(*)')
    lines.append('      CHARACTER*80 CMNAME, ORNAME')
    lines.append('      CALL localp_fetch_uvarm(NOEL, NPT, UVAR, NUVARM)')
    lines.append('      RETURN')
    lines.append('      END SUBROUTINE UVARM')
    return '\n'.join(lines)


def generate_uel_local_pressure(weakform, output_path,
                                mat_prefix='localpressuregel',
                                element_config=None):
    """Generate a Quad4/Hex8 UEL with element-local condensed pressure.

    The global fields are determined from the weak form:
      - displacement ``u`` is always required.
      - chemical potential ``mu`` is included only if the weak form declares a
        global ``mu`` field *and* the material defines ``solvent_flux`` and
        ``solvent_storage``.
    """
    cfg = element_config or ELEMENT_CONFIGS['quad4']
    if cfg.name not in ('quad4', 'hex8'):
        raise ValueError("local_pressure formulation currently supports "
                         "Quad4 and Hex8")
    has_mu = _has_mu_transport(weakform)
    mat_src = _generate_all_material_subs(weakform, mat_prefix)
    uel_src, ndofel = _generate_uel_local_pressure(weakform, mat_prefix, cfg)
    cs_src = _generate_local_pressure_cs(weakform, mat_prefix)
    templates = [
        _read_template('tensor_ops.for'),
        _read_template('isoparametric.for'),
    ]
    if cfg.ndim == 2:
        templates.append(_read_template('gauss_rules.for'))
    templates.extend(_read_template(name) for name in cfg.template_files)
    field_note = 'u, mu' if has_mu else 'u'
    full_src = '\n'.join([
        'C======================================================================',
        'C     Generated by coupfe.codegen local-pressure prototype',
        f'C     Fields: global {field_note}; local p stored in SVARS(1)',
        f'C     Element: {cfg.name.title()}, NDOFEL = {ndofel}',
        'C======================================================================',
        _generate_uvarm_bridge(),
        '',
        'C======================================================================',
        'C     UEL residual and tangent',
        'C======================================================================',
        uel_src,
        '',
        'C======================================================================',
        'C     Material subroutines',
        'C======================================================================',
        mat_src,
        '',
        'C======================================================================',
        'C     Local-pressure tangent engine',
        'C======================================================================',
        cs_src,
        '',
        'C======================================================================',
        'C     Templates',
        'C======================================================================',
        '\n'.join(templates),
    ])
    full_src = '\n'.join(_wrap_lines(full_src.split('\n')))
    with open(output_path, 'w') as f:
        f.write(full_src)
    print(f'  Generated: {output_path}')
    print(f'  Element:   {cfg.name.title()} local pressure')
    print(f'  NDOFEL:    {ndofel}')
    print('  NSVARS:    1 required; 1+2*NGP enables phi/p diagnostics')
