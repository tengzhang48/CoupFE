C======================================================================
C     shape_tri3.for — 3-node linear triangle shape functions (Tri3)
C
C     Node numbering (parent coordinates xi, eta):
C
C        3            eta
C        | \           |
C        |  \          +-----xi
C        1---2
C
C     Nodes: 1(0,0), 2(1,0), 3(0,1)
C======================================================================

C----------------------------------------------------------------------
C     shape_tri3: linear triangle shape functions and derivatives
C
C     sh(3)       = shape function values at (xi, eta)
C     dshxi(3,2)  = derivatives: dshxi(a,1)=dN_a/dxi, dshxi(a,2)=dN_a/deta
C----------------------------------------------------------------------
      SUBROUTINE shape_tri3(xi, eta, sh, dshxi)
      IMPLICIT NONE
      DOUBLE PRECISION, INTENT(IN)  :: xi, eta
      DOUBLE PRECISION, INTENT(OUT) :: sh(3), dshxi(3,2)

      sh(1) = 1.0d0 - xi - eta
      sh(2) = xi
      sh(3) = eta
      dshxi(1,1) = -1.0d0
      dshxi(1,2) = -1.0d0
      dshxi(2,1) =  1.0d0
      dshxi(2,2) =  0.0d0
      dshxi(3,1) =  0.0d0
      dshxi(3,2) =  1.0d0

      RETURN
      END SUBROUTINE shape_tri3
