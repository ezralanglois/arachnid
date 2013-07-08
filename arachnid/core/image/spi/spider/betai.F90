C -*-fortran-*-
C++*********************************************************************
C
C BETAI.FOR
C               
C               
C **********************************************************************
C *	AUTHOR: MAHIEDDINE LADJADJ     6/16/93                             *
C *                                                                         *
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
        INCLUDE 'CMBLOCK.INC' 

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


