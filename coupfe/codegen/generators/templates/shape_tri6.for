C======================================================================
C     shape_tri6.for — 6-node quadratic triangle shape functions (Tri6)
C
C     Node numbering (Gmsh order, parent coordinates xi, eta):
C
C        3
C        | \
C        6  5
C        |    \
C        1--4--2
C
C     Corners: 1(0,0), 2(1,0), 3(0,1)
C     Midsides: 4 (1-2), 5 (2-3), 6 (3-1)
C
C     degree=2 fields use all 6 nodes; degree=1 fields use corners 1-3
C     (see shape_tri3).
C======================================================================

C----------------------------------------------------------------------
C     shape_tri6: quadratic triangle shape functions and derivatives
C----------------------------------------------------------------------
      SUBROUTINE shape_tri6(xi, eta, sh, dshxi)
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(IN)  :: xi, eta
      DOUBLE PRECISION, INTENT(OUT) :: sh(6), dshxi(6,2)
      DOUBLE PRECISION :: L1

      L1 = 1.0d0 - xi - eta
      sh(1) = L1*(2.0d0*L1 - 1.0d0)
      sh(2) = xi*(2.0d0*xi - 1.0d0)
      sh(3) = eta*(2.0d0*eta - 1.0d0)
      sh(4) = 4.0d0*L1*xi
      sh(5) = 4.0d0*xi*eta
      sh(6) = 4.0d0*eta*L1

      dshxi(1,1) = 1.0d0 - 4.0d0*L1
      dshxi(1,2) = 1.0d0 - 4.0d0*L1
      dshxi(2,1) = 4.0d0*xi - 1.0d0
      dshxi(2,2) = 0.0d0
      dshxi(3,1) = 0.0d0
      dshxi(3,2) = 4.0d0*eta - 1.0d0
      dshxi(4,1) = 4.0d0*(L1 - xi)
      dshxi(4,2) = -4.0d0*xi
      dshxi(5,1) = 4.0d0*eta
      dshxi(5,2) = 4.0d0*xi
      dshxi(6,1) = -4.0d0*eta
      dshxi(6,2) = 4.0d0*(L1 - eta)

      RETURN
      END SUBROUTINE shape_tri6
