C++*********************************************************************
C
C GAMMLN.F
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







