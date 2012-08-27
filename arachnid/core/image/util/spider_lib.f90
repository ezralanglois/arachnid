C++*********************************************************************
C SPIDER Utilities required by wrapped functions in spider_util.f90
C
C FROM: ARACHNID - Image processing package
C AUTHOR: Robert Langlois
C Copyright (C) 2012
C
C--*********************************************************************


C++*********************************************************************
C
C BETAI.FOR
C
C
C **********************************************************************
C *	AUTHOR: MAHIEDDINE LADJADJ     6/16/93                             *
C *                                                                         *
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
C  BETAI.FOR
C
C	RETURNS THE INCOMPLETE BETA FUNCTION  / (A,B)
C                                            /X
C	       "NUMERICAL RECIPES"
C	       BY      WILLIAM H PRESS ET ALL.
C       COPIED FROM PAGE 167 OF "NUMERICAL RECIPES" BOOK
C
C
C IMAGE_PROCESSING_ROUTINE
C23456789012345678901234567890123456789012345678901234567890123456789012
C--*********************************************************************

         DOUBLE PRECISION FUNCTION BETAI(A,B,X)

        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C        INCLUDE 'CMBLOCK.INC'

	 IF ( X .LT. 0.0D0 .OR. X .GT. 1.0D0) THEN
           WRITE(NOUT,*) '*** Bad argument X in BETAI'
         ELSEIF ( X .EQ. 0.0D0 .OR. X .EQ. 1.0D0) THEN
           BT =  0.0D0
         ELSE

C          FACTORS IN FRONT OF THE CONTINUED FRACTION.
           Y = X
           BT = EXP( GAMMLN(A + B) - GAMMLN(A) - GAMMLN(B) +
     &               A * DLOG(Y) + B * DLOG(1. - Y))

         END IF
         IF ( X .LT. ((A + 1.) / (A + B + 2.0))) THEN

C	   USE CONTINUED FRACTION DIRECTLY.
           BETAI = BT * BETACF (A, B, X) / A
         ELSE

C          USE CONTINUED FRACTION AFTER MAKING THE SYMMETRY TRANSFORMATION
           BETAI = 1.0 - (BT * BETACF(B, A, 1.0 - X) / B)
         END IF

         END

C++*********************************************************************
C
C GAMMLN.F
C
C **********************************************************************
C *                                                                        *
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
C	RETURNS THE VALUE ln(XX) FOR XX > 0. FULL ACCURACY IS
C	OBTAINED FOR  0 < XX < 1, THE REFLECTION FORMULA CAN BE USED
C       FIRST.
C
C IMAGE_PROCESSING_ROUTINE
C
C--*********************************************************************

        DOUBLE PRECISION FUNCTION GAMMLN(XX)

        DOUBLE PRECISION COF(6), STP, HALF, ONE, FPF, TMP, SER
        DOUBLE PRECISION X, XX
        INTEGER J


        DATA COF, STP/76.18009173D0, -86.50532033D0, 24.01409822D0,
     &          -1.231739516D0, .120858003D-2, -.536382D-5,
     &          2.50662827465D0/
        DATA HALF, ONE, FPF/0.5D0, 1.0D0, 5.5D0/

        X = XX - ONE
        TMP = X + FPF
        TMP = (X + HALF) * DLOG(TMP) - TMP
        SER = ONE
        DO J = 1, 6
          X = X + ONE
          SER = SER + COF(J) / X
        END DO
        GAMMLN = TMP + DLOG(STP * SER)
        END







C++*********************************************************************
C
C BETACF.FOR
C
C
C **********************************************************************
C *                                                                        *
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
C
C IMAGE_PROCESSING_ROUTINE
C23456789012345678901234567890123456789012345678901234567890123456789012
C--*********************************************************************

        DOUBLE PRECISION FUNCTION BETACF(A, B, X)

        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C        INCLUDE 'CMBLOCK.INC'

        PARAMETER (ITMAX = 100, EPS = 3. E-7)

        AM = 1.0
        BM = 1.0
        AZ = 1.0

        QAB = A + B
        QAP = A + 1
        QAM = A - 1
        BZ = 1.0 - (QAB * X / QAP)

C	CONTINUED FRACTION EVALUATION BY THE RECURENCE METHOD
C       EQUATION 5.2.5  IN BOOK

        DO M = 1, ITMAX
          EM = M
          TEM = EM + EM
          D = EM * (B - M) * X / ((QAM + TEM) * (A + TEM))

C         ONE STEP (THE EVEN ONE) OF THE RECURENCE
          AP = AZ + D * AM
          BP = BZ + D * BM
          D = - (A + EM) * (QAB + EM) * X / ((A + TEM) * (QAP + TEM))

C         NEXT STEP OF THE RECURRENCE (THE ODD ONE)
          APP = AP + D * AZ
          BPP = BP + D * BZ

C         SAVE THE OLD ANSWER
          AOLD = AZ

C         RENORMALIZE TO PREVENT OVERFLOW
          AM = AP / BPP
          BM = BP / BPP
          AZ = APP / BPP
          BZ = 1.0

C         ARE WE DONE ?
          IF(DABS(AZ - AOLD) .LT. EPS * DABS(AZ)) GOTO 1
        END DO

        WRITE(NOUT,*)
     &        '***  IN BETACF, A or B too big, or ITMAX too small '

1       BETACF = AZ

        END


C--*********************************************************************

         FUNCTION FHT2(AK,N,LENH,XI,H1,H2,IFP,RXR,XRMI)

         REAL    ::  AK(2),XI(N),H1(3*LENH),H2(3*LENH)
         LOGICAL ::  IFP(N)

         DO I=1,3*LENH
            H2(I) = 0
         ENDDO

         DO  I=1,N
            IF (IFP(I))  THEN
               XN    = XI(I) * AK(1) + AK(2)
               L     = INT((XN - XRMI) / RXR * (LENH - 1) + LENH + 1)
               IF (L.GE.1 .AND. L .LE. 3*LENH)   H2(L) = H2(L) + 1
            ENDIF
         ENDDO

         FHT2 = 0.0

         DO I=1,3*LENH
            FHT2 = FHT2 + (H1(I) - H2(I)) ** 2
         ENDDO

         END

C--*********************************************************************

         SUBROUTINE AMOEBA2(P,Y,NDIM,FTOL,FUNK,ITER,ITMAX,PR,PRR,PBAR,
     &                      NSR1,LENH,QK2,QK4,QK5,QK6,RXR,XRMI)

C         INCLUDE 'CMBLOCK.INC'

         PARAMETER (ALPHA=1.0,BETA=0.5,GAMMA=2.0)
         DIMENSION P(NDIM+1,NDIM),Y(NDIM+1),PR(NDIM)
         DIMENSION PRR(NDIM),PBAR(NDIM)

         EXTERNAL  FUNK

         MPTS = NDIM+1
         ITER = 0

1        ILO = 1
         IF (Y(1) .GT. Y(2)) THEN
            IHI  = 1
            INHI = 2
         ELSE
            IHI  = 2
            INHI = 1
         ENDIF

         DO I=1,MPTS
            IF (Y(I) .LT. Y(ILO)) ILO=I
            IF (Y(I) .GT. Y(IHI)) THEN
               INHI=IHI
               IHI=I
            ELSE IF (Y(I) .GT. Y(INHI)) THEN
               IF ( I.NE. IHI) INHI=I
            ENDIF
         ENDDO

         RTOL = 2.*ABS(Y(IHI)-Y(ILO))/(ABS(Y(IHI))+ABS(Y(ILO)))
         IF (RTOL .LT. FTOL) RETURN
         IF (ITER .EQ. ITMAX) THEN
            WRITE(NOUT,*) ' Amoeba exceeding maximum iterations.'
            RETURN
         ENDIF
         ITER = ITER + 1
         DO J=1,NDIM
            PBAR(J) = 0.
         ENDDO

         DO I=1,MPTS
            IF (I .NE. IHI) THEN
               DO  J=1,NDIM
                  PBAR(J) = PBAR(J)+P(I,J)
               ENDDO
            ENDIF
         ENDDO

         DO J=1,NDIM
            PBAR(J) = PBAR(J) / NDIM
            PR(J)   = (1. + ALPHA) * PBAR(J) - ALPHA * P(IHI,J)
         ENDDO

         YPR = FUNK(PR,NSR1,LENH,QK2,QK4,QK5,QK6,RXR,XRMI)
C              FHT2(AK,NSR1,LENH,QK2,QK4,QK5,QK6,RXR,XRMI)

         IF (YPR .LE. Y(ILO)) THEN
            DO  J=1,NDIM
               PRR(J) = GAMMA*PR(J)+(1. - GAMMA) * PBAR(J)
            ENDDO

            YPRR = FUNK(PRR,NSR1,LENH,QK2,QK4,QK5,QK6,RXR,XRMI)
C                  FHT2(AK, NSR1,LENH,QK2,QK4,QK5,QK6,RXR,XRMI)

            IF (YPRR .LT. Y(ILO)) THEN
               DO J=1,NDIM
                  P(IHI,J) = PRR(J)
               ENDDO

               Y(IHI)=YPRR
            ELSE
               DO  J=1,NDIM
                  P(IHI,J) = PR(J)
               ENDDO

               Y(IHI) = YPR
            ENDIF

         ELSE IF (YPR .GE. Y(INHI)) THEN
            IF (YPR .LT. Y(IHI)) THEN
               DO  J=1,NDIM
                  P(IHI,J) = PR(J)
               ENDDO

               Y(IHI)=YPR
            ENDIF
            DO J=1,NDIM
               PRR(J) = BETA*P(IHI,J) + (1.-BETA)*PBAR(J)
            ENDDO

            YPRR = FUNK(PRR,NSR1,LENH,QK2,QK4,QK5,QK6,RXR,XRMI)
C                   FHT2(AK,NSR1,LENH,QK2,QK4,QK5,QK6,RXR,XRMI)

            IF (YPRR .LT. Y(IHI)) THEN
               DO J=1,NDIM
                  P(IHI,J) = PRR(J)
               ENDDO

               Y(IHI)=YPRR
            ELSE
               DO  I=1,MPTS
                  IF (I.NE.ILO) THEN
                     DO  J=1,NDIM
                        PR(J)  = 0.5*(P(I,J)+P(ILO,J))
                        P(I,J) = PR(J)
                     ENDDO

                     Y(I) = FUNK(PR,NSR1,LENH,QK2,QK4,QK5,QK6,RXR,XRMI)
                  ENDIF
               ENDDO
            ENDIF
         ELSE
            DO  J=1,NDIM
               P(IHI,J) = PR(J)
            ENDDO

            Y(IHI) = YPR
         ENDIF

         GO TO 1
         END

