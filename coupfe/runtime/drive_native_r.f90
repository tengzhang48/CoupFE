subroutine drive_native_r(R, svars_out, svars_in, coords, u, du, props, &
                          time, dtime, ndofel, nsvars, mcrd, nnode, nprops)
  implicit none
  integer, intent(in) :: ndofel, nsvars, mcrd, nnode, nprops
  double precision, intent(out) :: R(ndofel)
  double precision, intent(out) :: svars_out(nsvars)
  double precision, intent(in) :: svars_in(nsvars)
  double precision, intent(in) :: coords(mcrd, nnode)
  double precision, intent(in) :: u(ndofel), du(ndofel)
  double precision, intent(in) :: props(nprops)
  double precision, intent(in) :: time(2), dtime

  external coupfe_element_r

!f2py intent(out) :: R, svars_out
!f2py intent(in) :: svars_in, coords, u, du, props, time, dtime
!f2py integer intent(hide), depend(u) :: ndofel = shape(u,0)
!f2py integer intent(hide), depend(svars_in) :: nsvars = shape(svars_in,0)
!f2py integer intent(hide), depend(coords) :: mcrd = shape(coords,0)
!f2py integer intent(hide), depend(coords) :: nnode = shape(coords,1)
!f2py integer intent(hide), depend(props) :: nprops = shape(props,0)

  R = 0.0d0
  svars_out = 0.0d0

  call coupfe_element_r(coords, u, du, props, svars_in, R, svars_out, &
                        time, dtime)
end subroutine drive_native_r


subroutine drive_native_batch_r(R, svars_out, svars_in, coords, u, du, props, &
                                time, dtime, nelem, ndofel, nsvars, mcrd, &
                                nnode, nprops)
  implicit none
  integer, intent(in) :: nelem, ndofel, nsvars, mcrd, nnode, nprops
  double precision, intent(out) :: R(ndofel, nelem)
  double precision, intent(out) :: svars_out(nsvars, nelem)
  double precision, intent(in) :: svars_in(nsvars, nelem)
  double precision, intent(in) :: coords(mcrd, nnode, nelem)
  double precision, intent(in) :: u(ndofel, nelem), du(ndofel, nelem)
  double precision, intent(in) :: props(nprops)
  double precision, intent(in) :: time(2), dtime

  integer :: ie, i, j
  double precision :: svinloc(nsvars), svoutloc(nsvars)
  double precision :: uloc(ndofel), duloc(ndofel), crd(mcrd, nnode)
  double precision :: Rloc(ndofel)

  external coupfe_element_r

!f2py intent(out) :: R, svars_out
!f2py intent(in) :: svars_in, coords, u, du, props, time, dtime
!f2py integer intent(hide), depend(u) :: ndofel = shape(u,0)
!f2py integer intent(hide), depend(u) :: nelem = shape(u,1)
!f2py integer intent(hide), depend(svars_in) :: nsvars = shape(svars_in,0)
!f2py integer intent(hide), depend(coords) :: mcrd = shape(coords,0)
!f2py integer intent(hide), depend(coords) :: nnode = shape(coords,1)
!f2py integer intent(hide), depend(props) :: nprops = shape(props,0)

  do ie = 1, nelem
    Rloc = 0.0d0
    svinloc = 0.0d0
    svoutloc = 0.0d0
    do i = 1, ndofel
      uloc(i) = u(i, ie)
      duloc(i) = du(i, ie)
    end do
    do i = 1, nsvars
      svinloc(i) = svars_in(i, ie)
    end do
    do j = 1, nnode
      do i = 1, mcrd
        crd(i, j) = coords(i, j, ie)
      end do
    end do

    call coupfe_element_r(crd, uloc, duloc, props, svinloc, Rloc, svoutloc, &
                          time, dtime)

    do i = 1, ndofel
      R(i, ie) = Rloc(i)
    end do
    do i = 1, nsvars
      svars_out(i, ie) = svoutloc(i)
    end do
  end do
end subroutine drive_native_batch_r
