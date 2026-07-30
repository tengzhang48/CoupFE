subroutine drive_native(R, K, svars_out, svars_in, coords, u, du, props, jprops, &
                        time, dtime, pnewdt, lflags, params, jtype, &
                        kstep, kinc, jelem, period, ndofel, nsvars, &
                        mcrd, nnode, nprops, njprop)
  implicit none
  integer, intent(in) :: ndofel, nsvars, mcrd, nnode, nprops, njprop
  integer, intent(in) :: jtype, kstep, kinc, jelem
  integer, intent(in) :: lflags(6)
  integer, intent(in) :: jprops(njprop)
  double precision, intent(out) :: R(ndofel)
  double precision, intent(out) :: K(ndofel, ndofel)
  double precision, intent(out) :: svars_out(nsvars)
  double precision, intent(in) :: svars_in(nsvars)
  double precision, intent(in) :: coords(mcrd, nnode)
  double precision, intent(in) :: u(ndofel), du(ndofel)
  double precision, intent(in) :: props(nprops)
  double precision, intent(in) :: time(2), dtime, params(3), period
  double precision, intent(inout) :: pnewdt

  external coupfe_element_rk

!f2py intent(out) :: R, K, svars_out
!f2py intent(in) :: svars_in, coords, u, du, props, jprops, time, dtime, lflags, params
!f2py intent(in) :: jtype, kstep, kinc, jelem, period
!f2py intent(in,out) :: pnewdt
!f2py integer intent(hide), depend(u) :: ndofel = shape(u,0)
!f2py integer intent(hide), depend(svars_in) :: nsvars = shape(svars_in,0)
!f2py integer intent(hide), depend(coords) :: mcrd = shape(coords,0)
!f2py integer intent(hide), depend(coords) :: nnode = shape(coords,1)
!f2py integer intent(hide), depend(props) :: nprops = shape(props,0)
!f2py integer intent(hide), depend(jprops) :: njprop = shape(jprops,0)

  R = 0.0d0
  K = 0.0d0
  svars_out = 0.0d0

  call coupfe_element_rk(coords, u, du, props, svars_in, R, K, svars_out, &
                         time, dtime)
end subroutine drive_native


subroutine drive_native_batch(R, K, svars_out, svars_in, coords, u, du, props, jprops, &
                              time, dtime, pnewdt, lflags, params, jtype, &
                              kstep, kinc, period, nelem, ndofel, nsvars, &
                              mcrd, nnode, nprops, njprop)
  implicit none
  integer, intent(in) :: nelem, ndofel, nsvars, mcrd, nnode, nprops, njprop
  integer, intent(in) :: jtype, kstep, kinc
  integer, intent(in) :: lflags(6), jprops(njprop)
  double precision, intent(out) :: R(ndofel, nelem)
  double precision, intent(out) :: K(ndofel, ndofel, nelem)
  double precision, intent(out) :: svars_out(nsvars, nelem)
  double precision, intent(in) :: svars_in(nsvars, nelem)
  double precision, intent(in) :: coords(mcrd, nnode, nelem)
  double precision, intent(in) :: u(ndofel, nelem), du(ndofel, nelem)
  double precision, intent(in) :: props(nprops)
  double precision, intent(in) :: time(2), dtime, params(3), period
  double precision, intent(inout) :: pnewdt

  integer :: ie, i, j
  double precision :: svloc(nsvars), uloc(ndofel), duloc(ndofel), crd(mcrd, nnode)
  double precision :: Rloc(ndofel), Kloc(ndofel, ndofel)

  external coupfe_element_rk

!f2py intent(out) :: R, K, svars_out
!f2py intent(in) :: svars_in, coords, u, du, props, jprops, time, dtime, lflags, params
!f2py intent(in) :: jtype, kstep, kinc, period
!f2py intent(in,out) :: pnewdt
!f2py integer intent(hide), depend(u) :: ndofel = shape(u,0)
!f2py integer intent(hide), depend(u) :: nelem = shape(u,1)
!f2py integer intent(hide), depend(svars_in) :: nsvars = shape(svars_in,0)
!f2py integer intent(hide), depend(coords) :: mcrd = shape(coords,0)
!f2py integer intent(hide), depend(coords) :: nnode = shape(coords,1)
!f2py integer intent(hide), depend(props) :: nprops = shape(props,0)
!f2py integer intent(hide), depend(jprops) :: njprop = shape(jprops,0)

  do ie = 1, nelem
    Rloc = 0.0d0
    Kloc = 0.0d0
    svloc = 0.0d0
    do i = 1, ndofel
      uloc(i) = u(i, ie)
      duloc(i) = du(i, ie)
    end do
    do i = 1, nsvars
      svloc(i) = svars_in(i, ie)
    end do
    do j = 1, nnode
      do i = 1, mcrd
        crd(i, j) = coords(i, j, ie)
      end do
    end do

    call coupfe_element_rk(crd, uloc, duloc, props, svloc, Rloc, Kloc, svloc, &
                           time, dtime)

    do i = 1, ndofel
      R(i, ie) = Rloc(i)
    end do
    do j = 1, ndofel
      do i = 1, ndofel
        K(i, j, ie) = Kloc(i, j)
      end do
    end do
    do i = 1, nsvars
      svars_out(i, ie) = svloc(i)
    end do
  end do
end subroutine drive_native_batch
