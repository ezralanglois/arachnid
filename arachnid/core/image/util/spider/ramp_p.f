C++*********************************************************************
C
C RAMP_P.F
C
C **********************************************************************
C=*                                                                    *
C=* This file is part of:   SPIDER - Modular Image Processing System.  *
C=* SPIDER System Authors:  Joachim Frank & ArDean Leith               *
C=* Copyright 1985-2010  Health Research Inc.,                         *
C=* Riverview Center, 150 Broadway, Suite 560, Menands, NY 12204.      *
C=* Email: spider@wadsworth.org                                        *
C=*                                                                    *
C=* SPIDER is free software; you can redistribute it and/or            *
C=* modify it under the terms of the GNU General Public License as     *
C=* published by the Free Software Foundation; either version 2 of the *
C=* License, or (at your option) any later version.                    *
C=*                                                                    *
C=* SPIDER is distributed in the hope that it will be useful,          *
C=* but WITHOUT ANY WARRANTY; without even the implied warranty of     *
C=* merchantability or fitness for a particular purpose.  See the GNU  *
C=* General Public License for more details.                           *
C=* You should have received a copy of the GNU General Public License  *
C=* along with this program. If not, see <http://www.gnu.org/licenses> *
C=*                                                                    *
C **********************************************************************
C
C  RAMP_P(LUN1,LUN2,NSAM,NROW,NOUT)
C
C IMAGE_PROCESSING_ROUTINE
C
C23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
C--*********************************************************************

         SUBROUTINE  RAMP_P(LUN1,LUN2,NSAM,NROW,NOUT)

         DIMENSION        X(NSAM)
         EXTERNAL         BETAI
         DOUBLE PRECISION BETAI
         DOUBLE PRECISION C,D,EPS,B1,B2,A,F,R2,DN1,DN2
         DOUBLE PRECISION Q(6),S(9),QYX1,QYX2,QX1X2
     &                       ,QX1,QX2,QY,SYX1,SYX2,SX1X2,SX1
     &                       ,SX2,SY,SX1Q,SX2Q,SYQ
         EQUIVALENCE (Q(1),QYX1),(Q(2),QYX2),(Q(3),QX1X2),(Q(4),QX1),
     &               (Q(5),QX2),(Q(6),QY)
         EQUIVALENCE (S(1),SYX1),(S(2),SYX2),(S(3),SX1X2),(S(4),SX1),
     &               (S(5),SX2),(S(6),SY),(S(7),SX1Q),
     &               (S(8),SX2Q),(S(9),SYQ)

         DATA  EPS/1.0D-5/

C        ZERO ARRAY S
         S   = 0
         N1  = NSAM / 2
         N2  = NROW / 2
         SX1 = FLOAT(N1) * FLOAT(NSAM + 1)
         IF(MOD(NSAM,2) .EQ. 1)   SX1 = SX1 + 1 + N1
         SX2 = FLOAT(N2) * FLOAT(NROW + 1)
         IF(MOD(NROW,2) .EQ. 1)   SX2 = SX2 + 1 + N2
         SX1   = SX1 * NROW
         SX2   = SX2 * NSAM
         SX1X2 = 0.0D0
         DO  J = 1, NROW
           CALL REDLIN(LUN1,X,NSAM,J)
           DO I = 1, NSAM
             SYX1 = SYX1 + X(I) * I
             SYX2 = SYX2 + X(I) * J
             SY   = SY   + X(I)
             SX1Q = SX1Q + I * I
             SX2Q = SX2Q + J * J
             SYQ  = SYQ  + X(I) * DBLE(X(I))
           END DO
         END DO
         DN    = FLOAT(NSAM) * FLOAT(NROW)
         QYX1  = SYX1 - SX1 * SY / DN
         QYX2  = SYX2 - SX2 * SY / DN
         QX1X2 = 0.0
         QX1   = SX1Q - SX1 * SX1 / DN
         QX2   = SX2Q - SX2 * SX2 / DN
         QY    = SYQ  - SY  * SY  / DN
         C     = QX1  * QX2 - QX1X2 * QX1X2
         IF (C .GT. EPS) THEN
           B1  = (QYX1 * QX2 - QYX2 * QX1X2) / C
           B2  = (QYX2 * QX1 - QYX1 * QX1X2) / C
           A   = (SY - B1 * SX1 - B2 * SX2)  / DN
           D   = B1 * QYX1 + B2 * QYX2
           R2  = D / QY
           DN1 = 2
           DN2 = DN - 3

           IF (DABS(QY - D) .LT. EPS / 100.0) THEN
              F = 0.0
              P = 0.0
           ELSE
              F = D * (DN - 3.0) / 2 /(QY - D)
              P = 2.0*BETAI(0.5D0 * DN2, 0.5D0 * DN1, DN2 / 
     &             (DN2 + DN1 * F)) 
              IF (P.GT.1.0)  P = 2.0 - P
C +
C     &    (1.0D0-BETAI(0.5D0 * DN1, 0.5D0 * DN2, DN1 / (DN1 + DN2 / F)))
           END IF

           WRITE(NOUT,2020)  A, B1, B2, DSQRT(R2), R2, F, DN2, P
2020       FORMAT(/,
     &     '    Ramp model:    Y = a +  b1 * x1  +  b2 * x2',/,
     &     '    a  = ',1pd12.5,/,
     &     '    b1 = ',1pd12.5,/,
     &     '    b2 = ',1pd12.5,/,
     &     '    Multiple correlation R = ',0pf10.8,/,
     &     '    R squared              = ',0pf10.8,/,
     &     '    F-statistics  F = ',1pd12.5,
     &     '    with n1=2 and n2=',0pf7.0,'  df',/,
     &     '    Significance  p = ',0pf10.8)

           D = A + B1 + B2
           DO I = 1, NROW
             QY = D
             CALL REDLIN(LUN1,X,NSAM,I)
             DO  K = 1, NSAM
                X(K) = X(K) - QY
                QY   = QY + B1
             END DO
             CALL WRTLIN(LUN2,X,NSAM,I)
             D = D + B2
           END DO
C          CALL SETPRM(LUN2,NSAM,NROW,0.,0.,0.,'R')

         ELSE
           WRITE(NOUT,3030)
3030       FORMAT(/,' No solution - image is not modified !')
           DO I = 1,NROW
              CALL REDLIN(LUN1,X,NSAM,I)
              CALL WRTLIN(LUN2,X,NSAM,I)
           END DO
         END IF
         END
