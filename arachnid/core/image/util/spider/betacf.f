C++*********************************************************************
C
C BETACF.FOR
C
C
C **********************************************************************
C *                                                                        *
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
C          
C IMAGE_PROCESSING_ROUTINE
C23456789012345678901234567890123456789012345678901234567890123456789012
C--*********************************************************************

        DOUBLE PRECISION FUNCTION BETACF(A, B, X)

        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INCLUDE 'CMBLOCK.INC'       

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

