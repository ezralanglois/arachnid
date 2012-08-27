C++*********************************************************************
C SPIDER Functions wrapped for Python
C
C FROM: ARACHNID - Image processing package
C AUTHOR: Robert Langlois
C Copyright (C) 2012
C
C--*********************************************************************


C++*********************************************************************
C
C RAMP_P.F
C
C **********************************************************************
C=* FROM: SPIDER - MODULAR IMAGE PROCESSING SYSTEM.   AUTHOR: J.FRANK  *
C=* Copyright (C) 1985-2005  Health Research Inc.                      *
C=*                                                                    *
C=* HEALTH RESEARCH INCORPORATED (HRI),                                *
C=* ONE UNIVERSITY PLACE, RENSSELAER, NY 12144-3455.                   *
C=*                                                                    *
C=* Email:  spider@wadsworth.org                                       *
C=*                                                                    *
C=* This program is free software; you can redistribute it and/or      *
C=* modify it under the terms of the GNU General Public License as     *
C=* published by the Free Software Foundation; either version 2 of the *
C=* License, or (at your option) any later version.                    *
C=*                                                                    *
C=* This program is distributed in the hope that it will be useful,    *
C=* but WITHOUT ANY WARRANTY; without even the implied warranty of     *
C=* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU  *
C=* General Public License for more details.                           *
C=*                                                                    *
C=* You should have received a copy of the GNU General Public License  *
C=* along with this program; if not, write to the                      *
C=* Free Software Foundation, Inc.,                                    *
C=* 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.      *
C=*                                                                    *
C **********************************************************************
C
C  RAMP_P(LUN1,LUN2,NSAM,NROW,NOUT)
C
C IMAGE_PROCESSING_ROUTINE
C
C23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
C--*********************************************************************

         SUBROUTINE  RAMP(IMG,NSAM,NROW,RETVAL)

         DOUBLE PRECISION IMG(NSAM,NROW)
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

cf2py intent(in) :: NSAM,NROW
cf2py intent(inout) :: IMG
cf2py intent(out) :: RETVAL
cf2py intent(hide) :: NSAM,NROW

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
           DO I = 1, NSAM
             SYX1 = SYX1 + IMG(I, J) * I
             SYX2 = SYX2 + IMG(I, J) * J
             SY   = SY   + IMG(I, J)
             SX1Q = SX1Q + I * I
             SX2Q = SX2Q + J * J
             SYQ  = SYQ  + IMG(I, J) * DBLE(IMG(I, J))
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

           D = A + B1 + B2
           DO I = 1, NROW
             QY = D
             DO  K = 1, NSAM
                IMG(I, K) = IMG(I, K) - QY
                QY   = QY + B1
             END DO
             D = D + B2
           END DO
           RETVAL = 0
         ELSE
           RETVAL = 1
         END IF
         END

C++*********************************************************************
C
C HISTE.F                               USED OPFILE NOV 00 ARDEAN LEITH
C                        REWORKED MEMORY ALLOCATION OCT 05 ARDEAN LEITH
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
C  HISTE(UNUSED)
C
C  PURPOSE: Finds the linear transformation (applied to pixels)
C           which fits the histogram of the image file to the
C           histogram of the reference file.
C
C--*********************************************************************

         SUBROUTINE  HISTC2(QK2,NSR1,QK6,NSR2,QK1,N,LENH,ITRMAX)

         DOUBLE PRECISION       ::  QK1(N),QK2(NSR1)
         LOGICAL   ::  QK6(NSR2)

         REAL      ::  QK4(3*LENH),QK5(3*LENH)

         REAL      ::  AK(2),P(3,2),Y(3)
         REAL      ::  PR(2),PRR(2),PBAR(2)

         EXTERNAL  FHT2


cf2py intent(in) :: NSR1,NSR2,N,LENH,ITRMAX,QK6,QK1
cf2py intent(inout) :: QK2
cf2py intent(hide) :: NSR1,NSR2,N

         XRMI = QK1(1)
         XRMA = XRMI
         AVR  = XRMI
         SR   = XRMI**2
         DO I=2,N
            AVR  = AVR + QK1(I)
            SR   = SR  + QK1(I) ** 2
            XRMI = AMIN1(XRMI,QK1(I))
            XRMA = AMAX1(XRMA,QK1(I))
	 	 ENDDO

C        ximi=amin1(ximi,xi(i))
C1       xima=amax1(xima,xi(i))

         NT1  = 0
         XIMI = 1.E23
         XIMA = -XIMI
         AVI  = 0.0
         SI   = 0.0

         DO I=1,NSR1
            IF (QK6(I))  THEN
               NT1 = NT1 + 1
               AVI = AVI + QK2(I)
               SI  = SI  + QK2(I)**2
            ENDIF
	 ENDDO

         RXR = XRMA - XRMI
C        rxi = xima - ximi
         AVR = AVR / N
         AVI = AVI / NT1
         SR  = SQRT((SR - N   * AVR * AVR) / (N-1))
         SI  = SQRT((SI - NT1 * AVI * AVI) / (NT1-1))

         DO I=1,3*LENH
            QK4(I) = 0
	 ENDDO

         DO I=1,N
            L = INT((QK1(I) - XRMI) / RXR * (LENH-1) + LENH+1)
            QK4(L) = QK4(L) + 1
	 ENDDO

         DO I=1,3*LENH
            QK4(I) = QK4(I) * FLOAT(NT1) / FLOAT(N)
	 ENDDO

         A      = SR/SI
         P(1,1) = 0.9*A
         P(2,1) = A
         P(3,1) = 1.1*A
         B      =  AVR-A*AVI
         P(1,2) = -B
         P(2,2) = B
         P(3,2) = 3*B

         DO I=1,3
            AK(1) = P(I,1)
            AK(2) = P(I,2)
            Y(I)  = FHT2(AK,NSR1,LENH,QK2,QK4,QK5,QK6,RXR,XRMI)
	 ENDDO

         N2  = 2
         EPS = 0.0001

         CALL AMOEBA2(P,Y,N2,EPS,FHT2,ITER,ITRMAX,PR,PRR,PBAR,
     &               NSR1,LENH,QK2,QK4,QK5,QK6,RXR,XRMI)

C        do  6  i=1,4
C6          print  203,(p(i,j),j=1,3)
C203     format(3(3x,e12.5))

         DO I=1,NSR1
            QK2(I) = QK2(I) * P(2,1) + P(2,2)
	 	 ENDDO

         END

