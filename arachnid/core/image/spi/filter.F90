

C--*********************************************************************
		SUBROUTINE FILTER_2D(BUFI,IOPTT,BFPS,PARM1,PARM2,TEMP
     &						 B, LSD,N2X,N2Y, NX,NY,IRTFLG)

		REAL             		:: BFPS(4)
		REAL             		:: PARM1,PARM2,TEMP
        REAL             		:: B(LSD,N2Y)
		INTEGER          		:: LSD,N2X,N2Y,NX,NY

cf2py threadsafe
cf2py intent(inplace) :: B
cf2py intent(in) :: IOPTT,LSD,N2X,N2Y,NX,NY
cf2py intent(hide) :: LSD,N2Y
cf2py intent(out) :: IRTFLG

              CALL FQ_BUF(IOPTT,BFPS,PARM1T,PARM2T,TEMPT,
     &                    B, LSD,N2X,N2Y, NX,NY, IRTFLG)


		END

C--*********************************************************************
	SUBROUTINE FILTER_3D(B,LSD,N2S,N2R,N2L,NX,NY,NZ,IOPT)

	REAL             :: B(LSD,N2R,N2L)
	DOUBLE PRECISION :: AVE
	REAL             :: F,F2
	REAL             :: FP, FS
	REAL             :: FP2, FS2

C       BORDER PADDING

	IF (N2S.NE.NX .AND. N2R.NE.NY .AND. N2L.NE.NZ)  THEN

          AVE = (SUM(B(1:NX,1:NY,1))+SUM(B(1:NX,1:NY,NZ)) +
     &           SUM(B(1:NX,1,2:NZ-1))+SUM(B(1:NX,NY,2:NZ-1)) +
     &           SUM(B(1,2:NY-1,2:NZ-1))+SUM(B(NX,2:NY-1,2:NZ-1)))
     &		/REAL(4*(NX+NY+NZ)-16)

c$omp      parallel do private(i,j,k),reduction(+:ave)
	   DO K=1,NZ
	      DO J=1,NY
	         DO I=1,NX
	            AVE = AVE + B(I,J,K)
	         ENDDO
	      ENDDO
	   ENDDO

	   AVE = AVE/FLOAT(NX)/FLOAT(NY)/FLOAT(NZ)

c$omp      parallel do private(i,j,k)
	   DO K=1,NZ
	      DO J=1,N2R
	         DO I=NX+1,N2S
	            B(I,J,K) = AVE
	         ENDDO
	      ENDDO
	      DO J=NY+1,N2R
	         DO I=1,NX
	            B(I,J,K) = AVE
	         ENDDO
	      ENDDO
	   ENDDO

c$omp      parallel do private(i,j,k)
	   DO K=NZ+1,N2L
	      DO J=1,N2R
	         DO I=1,N2S
	            B(I,J,K) = AVE
	         ENDDO
	      ENDDO
	   ENDDO
	ENDIF

	INV = 1
	CALL FMRS_3(B,N2S,N2R,N2L,INV)
	IF (INV == 0) THEN
	   IOPT = -1
	   RETURN
	ENDIF

	IF (IOPT==7 .OR. IOPT==8 .OR. IOPT==9 .OR. IOPT==10) THEN
C          BUTTERWORTH FILTER OR  RAISED COSINE FILTER **************
	   EPS   =  0.882
	   AA    = 10.624

           CALL RDPRM2S(FP,FS,NOT_USED,
     &        'LOWER & UPPER LIMITING FREQ. (IN FREQ OR PIXEL UNITS)',
     &         IRTFLG)
           IF (IRTFLG .NE. 0) RETURN

	   IF (FP > 0.5) THEN
              FP2 = FP / NX
           ELSE
              FP2 = FP
	   ENDIF
           IF (FS > 0.5) THEN
              FS2 = FS / NX
           ELSE
              FS2 = FS
           ENDIF

	   ORD   = 2. * ALOG10(EPS / SQRT(AA**2-1.0))
	   ORD   = ORD/ALOG10(FP2 / FS2)
	   PARM1 = FP2/(EPS)**(2. / ORD)

	ELSE

           PARM1 = 0.25
	   CALL RDPRM1S(PARM1,NOT_USED,
     &         'FILTER RADIUS (IN FREQUENCY OR PIXEL UNITS)',IRTFLG)
           IF (IRTFLG .NE. 0) RETURN

           IF (PARM1<0.0 .OR. PARM1>0.5) PARM1 = 0.5 * PARM1 / (NX/2)


	   IF (IOPT==5 .OR. IOPT==6)  THEN
C             FERMI DISTRIBUTION FILTER ********************

	      CALL RDPRM1S(TEMP,NOT_USED,
     &                    'TEMPERATURE (0=CUTOFF)',IRTFLG)
              IF (IRTFLG .NE. 0) RETURN

C             EXPONENTIAL FOR HIGH-PASS OPTION
	      IF(IOPT == 6) TEMP = -TEMP
	   ENDIF
        ENDIF

	NR2  = N2R / 2
	NL2  = N2L / 2
	X1   = FLOAT(N2S / 2)**2
	Y1   = FLOAT(NR2)**2
	Z1   = FLOAT(NL2)**2
	PARM = PARM1**2

C       KEEP ZERO TERM FOR HIGH PASS OPTIONS
	AVG = B(1,1,1)

c$omp   parallel do private(i,j,k,ix,iy,iz,f)
	DO K=1,N2L
	   IZ = K-1
	   IF (IZ > NL2)  IZ = IZ-N2L

	   DO J=1,N2R
	      IY = J-1
	      IF (IY > NR2)  IY = IY-N2R

	      DO  I=1,LSD,2
	         IX = (I-1) / 2

                 SELECT CASE(IOPT)

                 CASE (1)    ! LOWPASS *****************************
                 IF (0.25*(FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1 +
     &               FLOAT(IZ*IZ)/Z1) > PARM)  THEN
	            B(I,J,K)   = 0.0
	            B(I+1,J,K) = 0.0
	         ENDIF


                 CASE (2)    !  HIGH PASS **************************

                IF ( 0.25*(FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1+
     &              FLOAT(IZ*IZ)/Z1) <= PARM)  THEN
	            B(I,J,K)   = 0.0
	            B(I+1,J,K) = 0.0
	         ENDIF


                 CASE (3)    !  GAUSSIAN LOW PASS ******************

 	         F = 0.125*(FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1+
     &                      FLOAT(IZ*IZ)/Z1)/PARM

	         IF (F < 16.0)  THEN
	            F          = EXP(-F)
	            B(I,J,K)   = B(I,J,K)*F
	            B(I+1,J,K) = B(I+1,J,K)*F
	         ELSE
                    B(I,J,K)   = 0.0
                    B(I+1,J,K) = 0.0
	         ENDIF


                 CASE (4)    !  GAUSSIAN HIGH PASS *****************

                 F = 0.125* (FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1+
     &                       FLOAT(IZ*IZ)/Z1)/PARM

	         IF (F < 16.0)  THEN
	            F          = (1.0-EXP(-F))
	            B(I,J,K)   = B(I,J,K)*F
	            B(I+1,J,K) = B(I+1,J,K)*F
	         ENDIF


                 CASE (5,6)  !  FERMI DISTRIBUTION FILTER **********

                 F = (0.5*SQRT(FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1+
     &                FLOAT(IZ*IZ)/Z1)-PARM1)/TEMP

	         F          = AMIN1(AMAX1(F,-10.0),10.0)
	         F          = (1.0/(1.0+EXP(F)))
	         B(I,J,K)   = B(I,J,K)*F
	         B(I+1,J,K) = B(I+1,J,K)*F


                 CASE (7)    !  BUTTERWORTH  LOWPASS FILTER ********

                 F = 0.5*SQRT(
     &	            FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1+FLOAT(IZ*IZ)/Z1)
	         F          = SQRT(1.0/(1.0+(F/PARM1)**ORD))

	         B(I,J,K)   = B(I,J,K)   * F
	         B(I+1,J,K) = B(I+1,J,K) * F


                 CASE (8)    !  BUTTERWORTH HIGIPASS FILTER *********

                 F = 0.5*SQRT(
     &	            FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1+FLOAT(IZ*IZ)/Z1)

	         F = (1.0-SQRT(1.0/(1.0+(F/PARM1)**ORD)))
	         B(I,J,K)   = B(I,J,K)   * F
	         B(I+1,J,K) = B(I+1,J,K) * F


                 CASE (9)    !  RAISED COSINE LOWPASS FILTER *******

                 F = 0.5*SQRT(
     &               FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1+FLOAT(IZ*IZ)/Z1)

	         IF (FP > 0.5) THEN
                    FP2 = FP/NX
                 ELSE
	            FP2 = FP
	         ENDIF
	         IF (FS > 0.5) THEN
                    FS2 = FS/NX
                 ELSE
	            FS2 = FS
	         ENDIF

	         F = (F-FP2) / (FS2-FP2)
                 IF (F < 0) THEN
	            F2 = 1
                 ELSEIF (F > 1) THEN
	            F2 = 0
                 ELSE
	            F2 = 0.5 * (COS(PI*F)+1)
	         ENDIF

                 B(I,J,K)   = B(I,J,K)  *F2
                 B(I+1,J,K) = B(I+1,J,K)*F2


                 CASE (10)    !  RAISED COSINE HIGHPASS FILTER *******

                 F = 0.5*SQRT(
     &               FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1+FLOAT(IZ*IZ)/Z1)

	         IF (FP > 0.5) THEN
                    FP2 = FP/NX
                 ELSE
	            FP2 = FP
	         ENDIF
	         IF (FS > 0.5) THEN
                    FS2 = FS/NX
                 ELSE
	            FS2 = FS
	         ENDIF

	         F = (F-FP2) / (FS2-FP2)
                 IF (F < 0) THEN
                    F2 = 0
                 ELSEIF (F > 1) THEN
	            F2 = 1
                 ELSE
	            F2 = 0.5 * (-COS(PI*F)+1)
	         ENDIF
	         B(I,J,K)   = B(I,J,K)   * F2
	         B(I+1,J,K) = B(I+1,J,K) * F2
                 END SELECT
              ENDDO
	   ENDDO
	ENDDO

C       RESTORE ZERO TERM FOR HIGH PASS OPTIONS
	IF (IOPT == 2 .OR. IOPT == 4 .OR.
     &      IOPT == 6 .OR. IOPT == 8)
     &      B(1,1,1) = AVG

C       REVERSE FFT
	INV = -1
	CALL  FMRS_3(B,N2S,N2R,N2L,INV)

	IF (INV == 0) THEN
	   CALL ERRT(38,'FQ',NE)
	   RETURN
	ENDIF

	END
