C======================================================================
C     gauss_tri.for -- triangle quadrature rules
C
C     Rules on the parent triangle (0,0)-(1,0)-(0,1); the weights sum to
C     its area 1/2.
C       1. gauss_tri1:  1-point rule, exact to degree 1
C       2. gauss_tri3:  3-point rule, exact to degree 2
C       3. gauss_tri6:  6-point rule, exact to degree 4
C                       (D. A. Dunavant, IJNME 21, 1985)
C======================================================================

C----------------------------------------------------------------------
      SUBROUTINE gauss_tri1(xi_gp, w_gp, ngp)
      IMPLICIT NONE
      INTEGER, INTENT(OUT) :: ngp
      DOUBLE PRECISION, INTENT(OUT) :: xi_gp(1,2), w_gp(1)
      ngp = 1
      xi_gp(1,1) = 1.0d0/3.0d0
      xi_gp(1,2) = 1.0d0/3.0d0
      w_gp(1) = 0.5d0
      RETURN
      END SUBROUTINE gauss_tri1

      SUBROUTINE gauss_tri3(xi_gp, w_gp, ngp)
      IMPLICIT NONE
      INTEGER, INTENT(OUT) :: ngp
      DOUBLE PRECISION, INTENT(OUT) :: xi_gp(3,2), w_gp(3)
      DOUBLE PRECISION :: a, b
      ngp = 3
      a = 1.0d0/6.0d0
      b = 2.0d0/3.0d0
      xi_gp(1,1) = a
      xi_gp(1,2) = a
      xi_gp(2,1) = b
      xi_gp(2,2) = a
      xi_gp(3,1) = a
      xi_gp(3,2) = b
      w_gp(1) = 1.0d0/6.0d0
      w_gp(2) = 1.0d0/6.0d0
      w_gp(3) = 1.0d0/6.0d0
      RETURN
      END SUBROUTINE gauss_tri3

      SUBROUTINE gauss_tri6(xi_gp, w_gp, ngp)
      IMPLICIT NONE
      INTEGER, INTENT(OUT) :: ngp
      DOUBLE PRECISION, INTENT(OUT) :: xi_gp(6,2), w_gp(6)
      DOUBLE PRECISION :: a, b, wa, wb
      ngp = 6
      a = 0.445948490915965d0
      b = 0.091576213509771d0
      wa = 0.5d0*0.223381589678011d0
      wb = 0.5d0*0.109951743655322d0
      xi_gp(1,1) = a
      xi_gp(1,2) = a
      xi_gp(2,1) = 1.0d0 - 2.0d0*a
      xi_gp(2,2) = a
      xi_gp(3,1) = a
      xi_gp(3,2) = 1.0d0 - 2.0d0*a
      xi_gp(4,1) = b
      xi_gp(4,2) = b
      xi_gp(5,1) = 1.0d0 - 2.0d0*b
      xi_gp(5,2) = b
      xi_gp(6,1) = b
      xi_gp(6,2) = 1.0d0 - 2.0d0*b
      w_gp(1) = wa
      w_gp(2) = wa
      w_gp(3) = wa
      w_gp(4) = wb
      w_gp(5) = wb
      w_gp(6) = wb
      RETURN
      END SUBROUTINE gauss_tri6
