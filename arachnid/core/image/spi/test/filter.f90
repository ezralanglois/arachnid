    SUBROUTINE FQ_Q(LUN,LUNO, B, LSD,N2S,N2R, NX,NY,IOPT)

	INCLUDE 'CMBLOCK.INC'

	INTEGER          :: LUN,LUNO
	REAL             :: B(LSD,N2R)
	INTEGER          :: LSD,N2S,N2R
	INTEGER          :: NX,NY,IOPT

	DOUBLE PRECISION :: AVE
	REAL             :: BFPS(4)

        REAL, PARAMETER  :: PI = 3.14159265358979323846

C       TO SET THEM TO SOMETHING.
	PARM1 = 0.0
	PARM2 = 0.0

C       READ  IMAGE
	DO I=1,NY
 	   CALL  REDLIN(LUN,B(1,I),NX,I)
	ENDDO

C       BORDER PADDING
	IF (N2S .NE. NX .AND. N2R .NE. NY)  THEN
	   AVE = (SUM(B(1:NX,1))   + SUM(B(1:NX,NY)) +
     &	          SUM(B(1,2:NY-1)) + SUM(B(NX,2:NY-1)) ) /
     &		  REAL(2*(NX+NY)-4)

c$omp      parallel do private(i,j)
	   DO J=1,N2R
	      DO I=NX+1,N2S
	         B(I,J) = AVE
	      ENDDO
	   ENDDO

c$omp      parallel do private(i,j)
	   DO J=NY+1,N2R
	      DO I=1,NX
	         B(I,J) = AVE
	      ENDDO
	   ENDDO
	ENDIF

C       FORWARD FFT
	INV=1
	CALL FMRS_2(B,N2S,N2R,INV)
	IF (INV == 0) THEN
	   IOPT = -1
	   RETURN
	ENDIF

C       BUTTERWORTH FILTER ***********************************

	IF (IOPT == 7 .OR. IOPT == 8 .OR.
     &      IOPT == 9 .OR. IOPT == 10)  THEN

	   NMAX = 4
	   BFPS = 0.0
	   CALL RDPRA('PASS-BAND FREQUENCY & STOP-BAND FREQUENCY',
     &        NMAX,0,.FALSE.,BFPS,NGOT,IRTFLG)

	   EPS = 0.882
	   AA  = 10.624
	   ORD = 2.0 * ALOG10(EPS / SQRT(AA**2-1.0) )

	   IF (BFPS(3) == 0.0 .AND. BFPS(4) == 0.0) THEN
	      ORD   = ORD / ALOG10(BFPS(1) / BFPS(2))
	      PARM1 = BFPS(1) / (EPS)**(2./ORD)
	   ELSE
C             BUTTERWORTH FILTER ELLIPTIC FILTER:
C             LOW-PASS  IOPT=11,  HIGH-PASS IOPT=12
	      IOPT = IOPT + 4
           ENDIF

	ELSE
  	   CALL RDPRM2(PARM1,PARM2,NOT_USED,'FILTER RADIUS')

	   IF (PARM1 <  0.0 .OR. PARM1 > 0.5) PARM1 = 0.5*PARM1/(NX/2)
	   IF (PARM2 == 0.0)                  PARM2 = PARM1
	   IF (PARM2 <  0.0 .OR. PARM2 > 0.5) PARM1 = 0.5*PARM2/(NY/2)

	   IF (IOPT == 5 .OR. IOPT == 6)  THEN

C             FERMI DISTRIBUTION FILTER ********************
	      CALL RDPRM(TEMP,NOT_USED,'TEMPERATURE(0=CUTOFF)')

C             EXPONENTIAL FOR HIGH-PASS OPTION
	      IF (IOPT == 6) TEMP = -TEMP
	   ENDIF
	ENDIF

	NR2    = N2R / 2
	X1     = FLOAT(N2S/2)**2
	Y1     = FLOAT(NR2)  **2
	PARM   = PARM1**2
	PARM22 = PARM2**2

C       KEEP ZERO TERM FOR HIGH PASS OPTIONS
	AVG = B(1,1)

c$omp   parallel do private(i,j,ix,iy,f,fpe,fse,ordt,parmt,f2)
	DO J=1,N2R
	   IY = (J-1)
	   IF (IY > NR2) IY = IY-N2R

	   DO I=1,LSD,2
	      IX = (I-1)/2

	      IF (IOPT == 1) THEN
C                LOWPASS *************************************
                 IF (0.25*(FLOAT(IX*IX)/X1/PARM +
     &                      FLOAT(IY*IY)/Y1/PARM22) > 1.0) THEN
	            B(I,J)   = 0.0
	            B(I+1,J) = 0.0
	         ENDIF

	      ELSEIF (IOPT == 2) THEN
C                HIGH PASS ***********************************
	         IF ( (IX.NE.0 .OR. IY.NE.0) .AND.
     &                0.25*(FLOAT(IX*IX)/X1/PARM +
     &                      FLOAT(IY*IY)/Y1/PARM22) <= 1.0) THEN
	            B(I,J)   = 0.0
	            B(I+1,J) = 0.0
	         ENDIF

	      ELSEIF(IOPT == 3)  THEN
C                GAUSSIAN LOW PASS ***************************
	         F = 0.125*(FLOAT(IX*IX)/X1/PARM +
     &                      FLOAT(IY*IY)/Y1/PARM22)
	         IF (F < 16.0)  THEN
	            F        = EXP(-F)
                    B(I,J)   = B(I,J)  *F
                    B(I+1,J) = B(I+1,J)*F
	         ELSE
                    B(I,J)   = 0.0
                    B(I+1,J) = 0.0
	         ENDIF

	      ELSEIF (IOPT==4)  THEN
C                GAUSSIAN HIGH PASS **************************

	         IF (IX .NE. 0 .OR. IY .NE. 0)  THEN
	            F=0.125*(FLOAT(IX*IX)/X1/PARM +
     &                       FLOAT(IY*IY)/Y1/PARM22)
	            IF (F < 16.0)  THEN
	               F        = 1.0 - EXP(-F)
                       B(I,J)   = B(I,J)  *F
                       B(I+1,J) = B(I+1,J)*F
	            ENDIF
	         ENDIF

	      ELSEIF (IOPT == 5 .OR. IOPT == 6)  THEN
C                FERMI DISTRIBUTION FILTER *******************

	         F = (0.5*SQRT(FLOAT(IX*IX)/X1 +
     &                         FLOAT(IY*IY)/Y1)-PARM1) / TEMP
	         F        = AMIN1(AMAX1(F,-10.0), 10.0)
                 F        = (1.0/(1.0+EXP(F)))

                 B(I,J)   = B(I,J)  *F
                 B(I+1,J) = B(I+1,J)*F

 	      ELSEIF (IOPT == 7) THEN
C                BUTTERWORTH LOWPASS FILTER ******************

 	         F        = 0.5*SQRT(FLOAT(IX*IX)/X1 +
     &                               FLOAT(IY*IY)/Y1)

 	         F        = SQRT(1.0/(1.0+(F/PARM1)**ORD))
                 B(I,J)   = B(I,J)  *F
                 B(I+1,J) = B(I+1,J)*F

 	      ELSEIF (IOPT == 8) THEN
C                BUTTERWORTH HIGHPASS FILTER *****************

                 IF (IX.NE.0 .OR. IY.NE. 0) THEN
 	            F = 0.5*SQRT(FLOAT(IX*IX)/X1 +
     &                           FLOAT(IY*IY)/Y1)
 	            F = (1.0-SQRT(1.0/(1.0+(F/PARM1)**ORD)))

                    B(I,J)   = B(I,J)*F
                    B(I+1,J) = B(I+1,J)*F
 	         ENDIF


 	      ELSEIF (IOPT == 9) THEN
C                RAISED COSINE LOWPASS FILTER ******************

	         F = 0.5*SQRT(FLOAT(IX*IX)/X1 +
     &                        FLOAT(IY*IY)/Y1)
	         F = (F-BFPS(1)) / (BFPS(2)-BFPS(1))
                 IF (F < 0) THEN
	            F2 = 1
                 ELSEIF (F > 1) THEN
	            F2 = 0
                 ELSE
	            F2 = 0.5 * (COS(PI*F)+1)
	         ENDIF
                 B(I,J)   = B(I,J)  *F2
                 B(I+1,J) = B(I+1,J)*F2

	      ELSEIF (IOPT == 10) THEN
C                RAISED COSINE HIGHPASS FILTER ******************

	         F = 0.5*SQRT(FLOAT(IX*IX)/X1 +
     &                        FLOAT(IY*IY)/Y1)
	         F = (F-BFPS(1)) / (BFPS(2)-BFPS(1))

                 IF (F < 0) THEN
                    F2 = 0
                 ELSEIF (F > 1) THEN
	            F2 = 1
                 ELSE
	            F2 = 0.5 * (-COS(PI*F)+1)
	         ENDIF
                 B(I,J)   = B(I,J)  *F2
                 B(I+1,J) = B(I+1,J)*F2

	      ELSEIF (IOPT == 11) THEN
C                BUTTERWORTH ELLIPTIC LOWPASS FILTER *********
C                CALCULATE EFFECTIVE FP AND FS IN A GIVEN
C                DIRECTION ON THE PLANE

                 IF (IX.NE.0 .OR. IY.NE.0) THEN
	            FPE = ATAN2(BFPS(1)*SQRT(FLOAT(IY*IY)/Y1),
     &                          BFPS(3)*SQRT(FLOAT(IX*IX)/X1))
                    FPE = SQRT((BFPS(1)*COS(FPE))**2 +
     &                         (BFPS(3)*SIN(FPE))**2)

	            FSE = ATAN2(BFPS(2)*SQRT(FLOAT(IY*IY)/Y1),
     &                          BFPS(4)*SQRT(FLOAT(IX*IX)/X1))
                    FSE = SQRT((BFPS(2)*COS(FSE))**2 +
     &                         (BFPS(4)*SIN(FSE))**2)

	            ORDT     = ORD/ALOG10(FPE/FSE)
	            PARMT    = FPE/(EPS)**(2./ORDT)
	            F        = 0.5*SQRT(FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1)
	            F        = SQRT(1.0/(1.0+(F/PARMT)**ORDT))
                    B(I,J)   = B(I,J)  *F
                    B(I+1,J) = B(I+1,J)*F
	         ENDIF

	      ELSEIF (IOPT == 12) THEN
C                BUTTERWORTH ELLIPTIC HIGHPASS FILTER *********

                 IF (IX .NE. 0 .OR. IY.NE. 0) THEN
	            FPE = ATAN2(BFPS(1)*SQRT(FLOAT(IY*IY)/Y1),
     &                          BFPS(3)*SQRT(FLOAT(IX*IX)/X1))
                    FPE = SQRT((BFPS(1)*COS(FPE))**2 +
     &                         (BFPS(3)*SIN(FPE))**2)

	            FSE = ATAN2(BFPS(2)*SQRT(FLOAT(IY*IY)/Y1),
     &                          BFPS(4)*SQRT(FLOAT(IX*IX)/X1))
                    FSE = SQRT((BFPS(2)*COS(FSE))**2 +
     &                         (BFPS(4)*SIN(FSE))**2)

	            ORDT     = ORD / ALOG10(FPE/FSE)
	            PARMT    = FPE / (EPS)**(2./ORDT)
	            F        = 0.5*SQRT(FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1)
	            F        = (1.0-SQRT(1.0/(1.0+(F/PARMT)**ORDT)))
                    B(I,J)   = B(I,J)  *F
                    B(I+1,J) = B(I+1,J)*F
	         ENDIF
              ENDIF
	   ENDDO
	ENDDO

C       RESTORE ZERO TERM FOR HIGH PASS OPTIONS
	IF (IOPT == 2 .OR. IOPT == 4 .OR. IOPT == 6 .OR. IOPT == 8)
     &     B(1,1) = AVG

C       REVERSE FFT AND WRITE  IMAGE
	INV = -1
	CALL FMRS_2(B,N2S,N2R,INV)

	DO I=1,NY
 	   CALL  WRTLIN(LUNO,B(1,I),NX,I)
	ENDDO

        END
