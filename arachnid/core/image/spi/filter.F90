!--*********************************************************************
! 2D filters
!--*********************************************************************
		SUBROUTINE GAUSSIAN_LP_2D(B,LSD,N2R,SIGMA,N2S,NX,NY,IRTFLG)
		REAL             		:: SIGMA
        REAL             		:: B(LSD,N2R)
		INTEGER          		:: LSD,N2S,N2R,NX,NY,IRTFLG

!f2py threadsafe
!f2py intent(inplace) :: B
!f2py intent(in) :: LSD,N2S,N2R,NX,NY,SIGMA
!f2py intent(hide) :: LSD,N2R
!f2py intent(out) :: IRTFLG
	   PARM1=REAL(SIGMA)
	   PARM2 = 0.0
	   IF (PARM1 <  0.0 .OR. PARM1 > 0.5) PARM1 = 0.5*PARM1/(NX/2)
	   IF (PARM2 == 0.0)                  PARM2 = PARM1
	   IF (PARM2 <  0.0 .OR. PARM2 > 0.5) PARM1 = 0.5*PARM2/(NY/2)
	   PARM   = PARM1**2
	   PARM22 = PARM2**2
	   NR2    = N2R / 2
	   X1     = FLOAT(N2S/2)**2
	   Y1     = FLOAT(NR2)  **2

	IRTFLG=0
	IF (N2S .NE. NX .AND. N2R .NE. NY)  THEN
	   AVE = (SUM(B(1:NX,1))   + SUM(B(1:NX,NY)) + &
     	          SUM(B(1,2:NY-1)) + SUM(B(NX,2:NY-1)) ) / &
     		  REAL(2*(NX+NY)-4)

!$omp      parallel do private(i,j)
	   DO J=1,N2R
	      DO I=NX+1,N2S
	         B(I,J) = AVE
	      ENDDO
	   ENDDO

!$omp      parallel do private(i,j)
	   DO J=NY+1,N2R
	      DO I=1,NX
	         B(I,J) = AVE
	      ENDDO
	   ENDDO
	ENDIF

	!       FORWARD FFT
	INV=1
	CALL FMRS_2(B,N2S,N2R,INV)
	IF (INV == 0) THEN
	   IRTFLG = -1
	   RETURN
	ENDIF

	! APPLY FILTER

	!$omp   parallel do private(i,j,ix,iy,f,fpe,fse,ordt,parmt,f2)
	DO J=1,N2R
	   IY = (J-1)
	   IF (IY > NR2) IY = IY-N2R

	   DO I=1,LSD,2
	      IX = (I-1)/2

	      F = 0.125*(FLOAT(IX*IX)/X1/PARM + &
                           FLOAT(IY*IY)/Y1/PARM22)

	      IF (F < 16.0)  THEN
	          F        = EXP(-F)
              B(I,J)   = B(I,J)  *F
              B(I+1,J) = B(I+1,J)*F
	      ELSE
              B(I,J)   = 0.0
              B(I+1,J) = 0.0
	      ENDIF
	   ENDDO
	ENDDO

	INV = -1
	CALL FMRS_2(B,N2S,N2R,INV)

	END


!--*********************************************************************
		 SUBROUTINE GAUSSIAN_HP_2D(B,LSD,N2R,SIGMA,N2S,NX,NY,IRTFLG)
		 REAL             		:: SIGMA
         REAL             		:: B(LSD,N2R)
		 INTEGER          		:: LSD,N2S,N2R,NX,NY,IRTFLG

!f2py threadsafe
!f2py intent(inplace) :: B
!f2py intent(in) :: LSD,N2S,N2R,NX,NY,SIGMA
!f2py intent(hide) :: LSD,N2R
!f2py intent(out) :: IRTFLG
	    PARM1=REAL(SIGMA)
	    PARM2 = 0.0
	    IF (PARM1 <  0.0 .OR. PARM1 > 0.5) PARM1 = 0.5*PARM1/(NX/2)
	    IF (PARM2 == 0.0)                  PARM2 = PARM1
	    IF (PARM2 <  0.0 .OR. PARM2 > 0.5) PARM1 = 0.5*PARM2/(NY/2)
	    PARM   = PARM1**2
	    PARM22 = PARM2**2
	    NR2    = N2R / 2
	    X1     = FLOAT(N2S/2)**2
	    Y1     = FLOAT(NR2)  **2

	   	!omega = params["cutoff_abs"];
		!omega = 0.5f/omega/omega;
		!argx = argy + float(jx*jx)*dx2;
		!1.0f-exp(-argx*omega);
		!dx = 1.0f/float(nxp);
		!float dx2 = dx*dx

		IRTFLG=0
		IF (N2S .NE. NX .AND. N2R .NE. NY)  THEN
	   		AVE = (SUM(B(1:NX,1))   + SUM(B(1:NX,NY)) + &
     	          SUM(B(1,2:NY-1)) + SUM(B(NX,2:NY-1)) ) / &
     		  REAL(2*(NX+NY)-4)

!$omp      parallel do private(i,j)
			DO J=1,N2R
				DO I=NX+1,N2S
					B(I,J) = AVE
				ENDDO
			ENDDO

!$omp      parallel do private(i,j)
		   	DO J=NY+1,N2R
				DO I=1,NX
					B(I,J) = AVE
				ENDDO
			ENDDO
		ENDIF

	!       FORWARD FFT
	INV=1
	CALL FMRS_2(B,N2S,N2R,INV)
	IF (INV == 0) THEN
	   IRTFLG = -1
	   RETURN
	ENDIF

	! APPLY FILTER
	AVG = B(1,1)
	!$omp   parallel do private(i,j,ix,iy,f,fpe,fse,ordt,parmt,f2)
	DO J=1,N2R
	   IY = (J-1)
	   IF (IY > NR2) IY = IY-N2R

	   DO I=1,LSD,2
	      IX = (I-1)/2

	        IF (IX .NE. 0 .OR. IY .NE. 0)  THEN
	            F=0.125*(FLOAT(IX*IX)/X1/PARM + &
                            FLOAT(IY*IY)/Y1/PARM22)
	            IF (F < 16.0)  THEN
	                   F        = 1.0 - EXP(-F)
                       B(I,J)   = B(I,J)  *F
                       B(I+1,J) = B(I+1,J)*F
	            ENDIF
	         ENDIF
	   ENDDO
	ENDDO

	B(1,1) = AVG
	INV = -1
	CALL FMRS_2(B,N2S,N2R,INV)

	END

	!--*********************************************************************
		 SUBROUTINE BUTTER_LP_2D(B,LSD,N2R,LCUT,HCUT,N2S,NX,NY,IRTFLG)
		 REAL             		:: LCUT,HCUT
         REAL             		:: B(LSD,N2R)
		 INTEGER          		:: LSD,N2S,N2R,NX,NY,IRTFLG

!f2py threadsafe
!f2py intent(inplace) :: B
!f2py intent(in) :: LSD,N2S,N2R,NX,NY,LCUT,HCUT
!f2py intent(hide) :: LSD,N2R
!f2py intent(out) :: IRTFLG
	    EPS = 0.882
	    AA  = 10.624
	    ORD = 2.0 * ALOG10(EPS / SQRT(AA**2-1.0) )
	    ORD   = ORD / ALOG10(LCUT / HCUT)
	    PARM1 = LCUT / (EPS)**(2./ORD)
	    PARM2 = 0.0

	    PARM   = PARM1**2
	    PARM22 = PARM2**2
	    NR2    = N2R / 2
	    X1     = FLOAT(N2S/2)**2
	    Y1     = FLOAT(NR2)  **2

		IRTFLG=0
		IF (N2S .NE. NX .AND. N2R .NE. NY)  THEN
	   		AVE = (SUM(B(1:NX,1))   + SUM(B(1:NX,NY)) + &
     	          SUM(B(1,2:NY-1)) + SUM(B(NX,2:NY-1)) ) / &
     		  REAL(2*(NX+NY)-4)

!$omp      parallel do private(i,j)
			DO J=1,N2R
				DO I=NX+1,N2S
					B(I,J) = AVE
				ENDDO
			ENDDO

!$omp      parallel do private(i,j)
		   	DO J=NY+1,N2R
				DO I=1,NX
					B(I,J) = AVE
				ENDDO
			ENDDO
		ENDIF

	!       FORWARD FFT
	INV=1
	CALL FMRS_2(B,N2S,N2R,INV)
	IF (INV == 0) THEN
	   IRTFLG = -1
	   RETURN
	ENDIF

	! APPLY FILTER
	!$omp   parallel do private(i,j,ix,iy,f,fpe,fse,ordt,parmt,f2)
	DO J=1,N2R
	   IY = (J-1)
	   IF (IY > NR2) IY = IY-N2R

	   DO I=1,LSD,2
	      IX = (I-1)/2
 	         F        = 0.5*SQRT(FLOAT(IX*IX)/X1 + &
                                    FLOAT(IY*IY)/Y1)

 	         F        = SQRT(1.0/(1.0+(F/PARM1)**ORD))
                 B(I,J)   = B(I,J)  *F
                 B(I+1,J) = B(I+1,J)*F
	   ENDDO
	ENDDO

	INV = -1
	CALL FMRS_2(B,N2S,N2R,INV)

	END

	!--*********************************************************************
		 SUBROUTINE BUTTER_HP_2D(B,LSD,N2R,LCUT,HCUT,N2S,NX,NY,IRTFLG)
		 REAL             		:: LCUT,HCUT
         REAL             		:: B(LSD,N2R)
		 INTEGER          		:: LSD,N2S,N2R,NX,NY,IRTFLG

!f2py threadsafe
!f2py intent(inplace) :: B
!f2py intent(in) :: LSD,N2S,N2R,NX,NY,LCUT,HCUT
!f2py intent(hide) :: LSD,N2R
!f2py intent(out) :: IRTFLG
	    EPS = 0.882
	    AA  = 10.624
	    ORD = 2.0 * ALOG10(EPS / SQRT(AA**2-1.0) )
	    ORD   = ORD / ALOG10(LCUT / HCUT)
	    PARM1 = LCUT / (EPS)**(2./ORD)
	    PARM2 = 0.0
	    PARM   = PARM1**2
	    PARM22 = PARM2**2
	    NR2    = N2R / 2
	    X1     = FLOAT(N2S/2)**2
	    Y1     = FLOAT(NR2)  **2

		IRTFLG=0
		IF (N2S .NE. NX .AND. N2R .NE. NY)  THEN
	   		AVE = (SUM(B(1:NX,1))   + SUM(B(1:NX,NY)) + &
     	          SUM(B(1,2:NY-1)) + SUM(B(NX,2:NY-1)) ) / &
     		  REAL(2*(NX+NY)-4)

!$omp      parallel do private(i,j)
			DO J=1,N2R
				DO I=NX+1,N2S
					B(I,J) = AVE
				ENDDO
			ENDDO

!$omp      parallel do private(i,j)
		   	DO J=NY+1,N2R
				DO I=1,NX
					B(I,J) = AVE
				ENDDO
			ENDDO
		ENDIF

	!       FORWARD FFT
	INV=1
	CALL FMRS_2(B,N2S,N2R,INV)
	IF (INV == 0) THEN
	   IRTFLG = -1
	   RETURN
	ENDIF

	! APPLY FILTER
	AVG = B(1,1)
	!$omp   parallel do private(i,j,ix,iy,f,fpe,fse,ordt,parmt,f2)
	DO J=1,N2R
	   IY = (J-1)
	   IF (IY > NR2) IY = IY-N2R

	   DO I=1,LSD,2
	      IX = (I-1)/2

            IF (IX.NE.0 .OR. IY.NE. 0) THEN
 	            F = 0.5*SQRT(FLOAT(IX*IX)/X1 + &
                                FLOAT(IY*IY)/Y1)
 	            F = (1.0-SQRT(1.0/(1.0+(F/PARM1)**ORD)))

                    B(I,J)   = B(I,J)*F
                    B(I+1,J) = B(I+1,J)*F
 	         ENDIF
	   ENDDO
	ENDDO

	B(1,1) = AVG
	INV = -1
	CALL FMRS_2(B,N2S,N2R,INV)

	END

!--*********************************************************************
		 SUBROUTINE RCOS_LP_2D(B,LSD,N2R,LCUT,HCUT,N2S,NX,NY,IRTFLG)
		 REAL             		:: LCUT,HCUT
         REAL             		:: B(LSD,N2R)
		 INTEGER          		:: LSD,N2S,N2R,NX,NY,IRTFLG
         REAL,PARAMETER  		:: PI = 3.14159265358979323846

!f2py threadsafe
!f2py intent(inplace) :: B
!f2py intent(in) :: LSD,N2S,N2R,NX,NY,LCUT,HCUT
!f2py intent(hide) :: LSD,N2R
!f2py intent(out) :: IRTFLG
	    X1     = FLOAT(N2S/2)**2
	    Y1     = FLOAT(N2R / 2)  **2
	    NR2    = N2R / 2

		IRTFLG=0
		IF (N2S .NE. NX .AND. N2R .NE. NY)  THEN
	   		AVE = (SUM(B(1:NX,1))   + SUM(B(1:NX,NY)) + &
     	          SUM(B(1,2:NY-1)) + SUM(B(NX,2:NY-1)) ) / &
     		  REAL(2*(NX+NY)-4)

!$omp      parallel do private(i,j)
			DO J=1,N2R
				DO I=NX+1,N2S
					B(I,J) = AVE
				ENDDO
			ENDDO

!$omp      parallel do private(i,j)
		   	DO J=NY+1,N2R
				DO I=1,NX
					B(I,J) = AVE
				ENDDO
			ENDDO
		ENDIF

	!       FORWARD FFT
	INV=1
	CALL FMRS_2(B,N2S,N2R,INV)
	IF (INV == 0) THEN
	   IRTFLG = -1
	   RETURN
	ENDIF

	! APPLY FILTER
	!$omp   parallel do private(i,j,ix,iy,f,fpe,fse,f2)
	DO J=1,N2R
	   IY = (J-1)
	   IF (IY > NR2) IY = IY-N2R

	   DO I=1,LSD,2
	      IX = (I-1)/2
	      F = 0.5*SQRT(FLOAT(IX*IX)/X1 + FLOAT(IY*IY)/Y1)
	         F = (F-LCUT) / (HCUT-LCUT)
                 IF (F < 0) THEN
	            F2 = 1
                 ELSEIF (F > 1) THEN
	            F2 = 0
                 ELSE
	            F2 = 0.5 * (COS(PI*F)+1)
	         ENDIF
                 B(I,J)   = B(I,J)  *F2
                 B(I+1,J) = B(I+1,J)*F2
	   ENDDO
	ENDDO

	INV = -1
	CALL FMRS_2(B,N2S,N2R,INV)

	END

!--*********************************************************************
		 SUBROUTINE RCOS_HP_2D(B,LSD,N2R,LCUT,HCUT,N2S,NX,NY,IRTFLG)
		 REAL             		:: LCUT,HCUT
         REAL             		:: B(LSD,N2R)
		 INTEGER          		:: LSD,N2S,N2R,NX,NY,IRTFLG
		 DOUBLE PRECISION 		:: AVE
         REAL,PARAMETER  		:: PI = 3.14159265358979323846

!f2py threadsafe
!f2py intent(inplace) :: B
!f2py intent(in) :: LSD,N2S,N2R,NX,NY,LCUT,HCUT
!f2py intent(hide) :: LSD,N2R
!f2py intent(out) :: IRTFLG
	    X1     = FLOAT(N2S/2)**2
	    Y1     = FLOAT(N2R / 2)  **2
	    NR2    = N2R / 2

		IRTFLG=0
		IF (N2S .NE. NX .AND. N2R .NE. NY)  THEN
	   		AVE = (SUM(B(1:NX,1))   + SUM(B(1:NX,NY)) + &
     	          SUM(B(1,2:NY-1)) + SUM(B(NX,2:NY-1)) ) / &
     		  REAL(2*(NX+NY)-4)

!$omp      parallel do private(i,j)
			DO J=1,N2R
				DO I=NX+1,N2S
					B(I,J) = AVE
				ENDDO
			ENDDO

!$omp      parallel do private(i,j)
		   	DO J=NY+1,N2R
				DO I=1,NX
					B(I,J) = AVE
				ENDDO
			ENDDO
		ENDIF

	!       FORWARD FFT
	INV=1
	CALL FMRS_2(B,N2S,N2R,INV)
	IF (INV == 0) THEN
	   IRTFLG = -1
	   RETURN
	ENDIF

	! APPLY FILTER
	AVG = B(1,1)
	!$omp   parallel do private(i,j,ix,iy,f,fpe,fse,f2)
	DO J=1,N2R
	   IY = (J-1)
	   IF (IY > NR2) IY = IY-N2R

	   DO I=1,LSD,2
	      IX = (I-1)/2

	         F = 0.5*SQRT(FLOAT(IX*IX)/X1 + &
                             FLOAT(IY*IY)/Y1)
	         F = (F-LCUT) / (HCUT-LCUT)

                 IF (F < 0) THEN
                    F2 = 0
                 ELSEIF (F > 1) THEN
	            F2 = 1
                 ELSE
	            F2 = 0.5 * (-COS(PI*F)+1)
	         ENDIF
                 B(I,J)   = B(I,J)  *F2
                 B(I+1,J) = B(I+1,J)*F2
	   ENDDO
	ENDDO

	B(1,1) = AVG
	INV = -1
	CALL FMRS_2(B,N2S,N2R,INV)
	IF (INV == 0) THEN
	   IRTFLG = -1
	   RETURN
	ENDIF

	END

!--*********************************************************************
! 3D filters
!--*********************************************************************

!--*********************************************************************
	SUBROUTINE GAUSSIAN_LP_3D(B,LSD,N2R,N2L,SIGMA,N2S,NX,NY,NZ,IRTFLG)

	REAL             :: B(LSD,N2R,N2L)
	DOUBLE PRECISION :: AVE
	REAL             :: F,F2
	REAL             :: FP, FS
	REAL             :: FP2, FS2,SIGMA
	INTEGER			 :: LSD,N2R,N2L,N2S,NX,NY,NZ,IRTFLG

!f2py threadsafe
!f2py intent(inplace) :: B
!f2py intent(in) :: LSD,N2S,N2R,N2L,NX,NY,NZ,SIGMA
!f2py intent(hide) :: LSD,N2R,N2L
!f2py intent(out) :: IRTFLG
!       BORDER PADDING

	IF (N2S.NE.NX .AND. N2R.NE.NY .AND. N2L.NE.NZ)  THEN

          AVE = (SUM(B(1:NX,1:NY,1))+SUM(B(1:NX,1:NY,NZ)) + &
                SUM(B(1:NX,1,2:NZ-1))+SUM(B(1:NX,NY,2:NZ-1)) + &
                SUM(B(1,2:NY-1,2:NZ-1))+SUM(B(NX,2:NY-1,2:NZ-1))) &
     		/REAL(4*(NX+NY+NZ)-16)

!$omp      parallel do private(i,j,k),reduction(+:ave)
	   DO K=1,NZ
	      DO J=1,NY
	         DO I=1,NX
	            AVE = AVE + B(I,J,K)
	         ENDDO
	      ENDDO
	   ENDDO

	   AVE = AVE/FLOAT(NX)/FLOAT(NY)/FLOAT(NZ)

!$omp      parallel do private(i,j,k)
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

!$omp      parallel do private(i,j,k)
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
	   IRTFLG = -1
	   RETURN
	ENDIF

	PARM1=SIGMA
    IF (PARM1<0.0 .OR. PARM1>0.5) PARM1 = 0.5 * PARM1 / (NX/2)
	NR2  = N2R / 2
	NL2  = N2L / 2
	X1   = FLOAT(N2S / 2)**2
	Y1   = FLOAT(NR2)**2
	Z1   = FLOAT(NL2)**2
	PARM = PARM1**2

!$omp   parallel do private(i,j,k,ix,iy,iz,f)
	DO K=1,N2L
	   IZ = K-1
	   IF (IZ > NL2)  IZ = IZ-N2L

	   DO J=1,N2R
	      IY = J-1
	      IF (IY > NR2)  IY = IY-N2R

	      DO  I=1,LSD,2
	         IX = (I-1) / 2

	 	     F = 0.125*(FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1+ &
                           FLOAT(IZ*IZ)/Z1)/PARM

	         IF (F < 16.0)  THEN
	            F          = EXP(-F)
	            B(I,J,K)   = B(I,J,K)*F
	            B(I+1,J,K) = B(I+1,J,K)*F
	         ELSE
                    B(I,J,K)   = 0.0
                    B(I+1,J,K) = 0.0
	         ENDIF
              ENDDO
	   ENDDO
	ENDDO
	INV = -1
	CALL  FMRS_3(B,N2S,N2R,N2L,INV)
	IF (INV == 0) THEN
	   IRTFLG = -1
	   RETURN
	ENDIF
	END

	!--*********************************************************************
	SUBROUTINE GAUSSIAN_HP_3D(B,LSD,N2R,N2L,SIGMA,N2S,NX,NY,NZ,IRTFLG)

	REAL             :: B(LSD,N2R,N2L)
	DOUBLE PRECISION :: AVE
	REAL             :: F,F2
	REAL             :: FP, FS
	REAL             :: FP2, FS2,SIGMA
	INTEGER			 :: LSD,N2R,N2L,N2S,NX,NY,NZ,IRTFLG

!f2py threadsafe
!f2py intent(inplace) :: B
!f2py intent(in) :: LSD,N2S,N2R,N2L,NX,NY,NZ,SIGMA
!f2py intent(hide) :: LSD,N2R,N2L
!f2py intent(out) :: IRTFLG
!       BORDER PADDING

	IF (N2S.NE.NX .AND. N2R.NE.NY .AND. N2L.NE.NZ)  THEN

          AVE = (SUM(B(1:NX,1:NY,1))+SUM(B(1:NX,1:NY,NZ)) + &
                SUM(B(1:NX,1,2:NZ-1))+SUM(B(1:NX,NY,2:NZ-1)) + &
                SUM(B(1,2:NY-1,2:NZ-1))+SUM(B(NX,2:NY-1,2:NZ-1))) &
     		/REAL(4*(NX+NY+NZ)-16)

!$omp      parallel do private(i,j,k),reduction(+:ave)
	   DO K=1,NZ
	      DO J=1,NY
	         DO I=1,NX
	            AVE = AVE + B(I,J,K)
	         ENDDO
	      ENDDO
	   ENDDO

	   AVE = AVE/FLOAT(NX)/FLOAT(NY)/FLOAT(NZ)

!$omp      parallel do private(i,j,k)
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

!$omp      parallel do private(i,j,k)
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
	   IRTFLG = -1
	   RETURN
	ENDIF

	PARM1=SIGMA
    IF (PARM1<0.0 .OR. PARM1>0.5) PARM1 = 0.5 * PARM1 / (NX/2)
	NR2  = N2R / 2
	NL2  = N2L / 2
	X1   = FLOAT(N2S / 2)**2
	Y1   = FLOAT(NR2)**2
	Z1   = FLOAT(NL2)**2
	PARM = PARM1**2

	AVG = B(1,1,1)
!$omp   parallel do private(i,j,k,ix,iy,iz,f)
	DO K=1,N2L
	   IZ = K-1
	   IF (IZ > NL2)  IZ = IZ-N2L

	   DO J=1,N2R
	      IY = J-1
	      IF (IY > NR2)  IY = IY-N2R

	      DO  I=1,LSD,2
	         IX = (I-1) / 2

             F = 0.125* (FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1+ &
                            FLOAT(IZ*IZ)/Z1)/PARM

	         IF (F < 16.0)  THEN
	            F          = (1.0-EXP(-F))
	            B(I,J,K)   = B(I,J,K)*F
	            B(I+1,J,K) = B(I+1,J,K)*F
	         ENDIF
              ENDDO
	   ENDDO
	ENDDO
	B(1,1,1) = AVG
	INV = -1
	CALL  FMRS_3(B,N2S,N2R,N2L,INV)
	IF (INV == 0) THEN
	   IRTFLG = -1
	   RETURN
	ENDIF
	END

	!--*********************************************************************
	SUBROUTINE BUTTER_LP_3D(B,LSD,N2R,N2L,LCUT,HCUT,N2S,NX,NY,NZ,IRTFLG)

	REAL             :: B(LSD,N2R,N2L)
	DOUBLE PRECISION :: AVE
	REAL             :: F,F2
	REAL             :: FP, FS
	REAL             :: FP2, FS2
	REAL             :: LCUT,HCUT
	INTEGER			 :: LSD,N2R,N2L,N2S,NX,NY,NZ,IRTFLG

!f2py threadsafe
!f2py intent(inplace) :: B
!f2py intent(in) :: LSD,N2S,N2R,N2L,NX,NY,NZ,LCUT,HCUT
!f2py intent(hide) :: LSD,N2R,N2L
!f2py intent(out) :: IRTFLG
!       BORDER PADDING

	IF (N2S.NE.NX .AND. N2R.NE.NY .AND. N2L.NE.NZ)  THEN

          AVE = (SUM(B(1:NX,1:NY,1))+SUM(B(1:NX,1:NY,NZ)) + &
                SUM(B(1:NX,1,2:NZ-1))+SUM(B(1:NX,NY,2:NZ-1)) + &
                SUM(B(1,2:NY-1,2:NZ-1))+SUM(B(NX,2:NY-1,2:NZ-1))) &
     		/REAL(4*(NX+NY+NZ)-16)

!$omp      parallel do private(i,j,k),reduction(+:ave)
	   DO K=1,NZ
	      DO J=1,NY
	         DO I=1,NX
	            AVE = AVE + B(I,J,K)
	         ENDDO
	      ENDDO
	   ENDDO

	   AVE = AVE/FLOAT(NX)/FLOAT(NY)/FLOAT(NZ)

!$omp      parallel do private(i,j,k)
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

!$omp      parallel do private(i,j,k)
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
	   IRTFLG = -1
	   RETURN
	ENDIF

	EPS   =  0.882
	AA    = 10.624
	FP=LCUT
	FS=HCUT
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

	NR2  = N2R / 2
	NL2  = N2L / 2
	X1   = FLOAT(N2S / 2)**2
	Y1   = FLOAT(NR2)**2
	Z1   = FLOAT(NL2)**2
	PARM = PARM1**2

!$omp   parallel do private(i,j,k,ix,iy,iz,f)
	DO K=1,N2L
	   IZ = K-1
	   IF (IZ > NL2)  IZ = IZ-N2L

	   DO J=1,N2R
	      IY = J-1
	      IF (IY > NR2)  IY = IY-N2R

	      DO  I=1,LSD,2
	         IX = (I-1) / 2

             F = 0.5*SQRT( &
     	            FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1+FLOAT(IZ*IZ)/Z1)
	         F          = SQRT(1.0/(1.0+(F/PARM1)**ORD))

	         B(I,J,K)   = B(I,J,K)   * F
	         B(I+1,J,K) = B(I+1,J,K) * F
              ENDDO
	   ENDDO
	ENDDO
	INV = -1
	CALL  FMRS_3(B,N2S,N2R,N2L,INV)
	IF (INV == 0) THEN
	   IRTFLG = -1
	   RETURN
	ENDIF
	END

	!--*********************************************************************
	SUBROUTINE BUTTER_HP_3D(B,LSD,N2R,N2L,LCUT,HCUT,N2S,NX,NY,NZ,IRTFLG)

	REAL             :: B(LSD,N2R,N2L)
	DOUBLE PRECISION :: AVE
	REAL             :: F,F2
	REAL             :: FP, FS
	REAL             :: FP2, FS2
	REAL             :: LCUT, HCUT
	INTEGER			 :: LSD,N2R,N2L,N2S,NX,NY,NZ,IRTFLG

!f2py threadsafe
!f2py intent(inplace) :: B
!f2py intent(in) :: LSD,N2S,N2R,N2L,NX,NY,NZ,LCUT,HCUT
!f2py intent(hide) :: LSD,N2R,N2L
!f2py intent(out) :: IRTFLG
!       BORDER PADDING

	IF (N2S.NE.NX .AND. N2R.NE.NY .AND. N2L.NE.NZ)  THEN

          AVE = (SUM(B(1:NX,1:NY,1))+SUM(B(1:NX,1:NY,NZ)) + &
                SUM(B(1:NX,1,2:NZ-1))+SUM(B(1:NX,NY,2:NZ-1)) + &
                SUM(B(1,2:NY-1,2:NZ-1))+SUM(B(NX,2:NY-1,2:NZ-1))) &
     		/REAL(4*(NX+NY+NZ)-16)

!$omp      parallel do private(i,j,k),reduction(+:ave)
	   DO K=1,NZ
	      DO J=1,NY
	         DO I=1,NX
	            AVE = AVE + B(I,J,K)
	         ENDDO
	      ENDDO
	   ENDDO

	   AVE = AVE/FLOAT(NX)/FLOAT(NY)/FLOAT(NZ)

!$omp      parallel do private(i,j,k)
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

!$omp      parallel do private(i,j,k)
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
	   IRTFLG = -1
	   RETURN
	ENDIF

	EPS   =  0.882
	AA    = 10.624
	FP=LCUT
	FS=HCUT
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
	NR2  = N2R / 2
	NL2  = N2L / 2
	X1   = FLOAT(N2S / 2)**2
	Y1   = FLOAT(NR2)**2
	Z1   = FLOAT(NL2)**2
	PARM = PARM1**2

	AVG = B(1,1,1)
!$omp   parallel do private(i,j,k,ix,iy,iz,f)
	DO K=1,N2L
	   IZ = K-1
	   IF (IZ > NL2)  IZ = IZ-N2L

	   DO J=1,N2R
	      IY = J-1
	      IF (IY > NR2)  IY = IY-N2R

	      DO  I=1,LSD,2
	         IX = (I-1) / 2

             F = 0.5*SQRT( &
     	            FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1+FLOAT(IZ*IZ)/Z1)

	         F = (1.0-SQRT(1.0/(1.0+(F/PARM1)**ORD)))
	         B(I,J,K)   = B(I,J,K)   * F
	         B(I+1,J,K) = B(I+1,J,K) * F
              ENDDO
	   ENDDO
	ENDDO
	B(1,1,1) = AVG
	INV = -1
	CALL  FMRS_3(B,N2S,N2R,N2L,INV)
	IF (INV == 0) THEN
	   IRTFLG = -1
	   RETURN
	ENDIF
	END


		!--*********************************************************************
	SUBROUTINE RCOS_LP_3D(B,LSD,N2R,N2L,LCUT,HCUT,N2S,NX,NY,NZ,IRTFLG)

	REAL             :: B(LSD,N2R,N2L)
	DOUBLE PRECISION :: AVE
	REAL             :: F,F2
	REAL             :: FP, FS
	REAL             :: FP2,FS2,LCUT,HCUT
	INTEGER			 :: LSD,N2R,N2L,N2S,NX,NY,NZ,IRTFLG
    REAL, PARAMETER	 :: PI = 3.14159265358979323846

!f2py threadsafe
!f2py intent(inplace) :: B
!f2py intent(in) :: LSD,N2S,N2R,N2L,NX,NY,NZ,LCUT,HCUT
!f2py intent(hide) :: LSD,N2R,N2L
!f2py intent(out) :: IRTFLG
!       BORDER PADDING

	IF (N2S.NE.NX .AND. N2R.NE.NY .AND. N2L.NE.NZ)  THEN

          AVE = (SUM(B(1:NX,1:NY,1))+SUM(B(1:NX,1:NY,NZ)) + &
                SUM(B(1:NX,1,2:NZ-1))+SUM(B(1:NX,NY,2:NZ-1)) + &
                SUM(B(1,2:NY-1,2:NZ-1))+SUM(B(NX,2:NY-1,2:NZ-1))) &
     		/REAL(4*(NX+NY+NZ)-16)

!$omp      parallel do private(i,j,k),reduction(+:ave)
	   DO K=1,NZ
	      DO J=1,NY
	         DO I=1,NX
	            AVE = AVE + B(I,J,K)
	         ENDDO
	      ENDDO
	   ENDDO

	   AVE = AVE/FLOAT(NX)/FLOAT(NY)/FLOAT(NZ)

!$omp      parallel do private(i,j,k)
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

!$omp      parallel do private(i,j,k)
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
	   IRTFLG = -1
	   RETURN
	ENDIF

	EPS   =  0.882
	AA    = 10.624
	FP=LCUT
	FS=HCUT
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

	NR2  = N2R / 2
	NL2  = N2L / 2
	X1   = FLOAT(N2S / 2)**2
	Y1   = FLOAT(NR2)**2
	Z1   = FLOAT(NL2)**2
	PARM = PARM1**2

!$omp   parallel do private(i,j,k,ix,iy,iz,f)
	DO K=1,N2L
	   IZ = K-1
	   IF (IZ > NL2)  IZ = IZ-N2L

	   DO J=1,N2R
	      IY = J-1
	      IF (IY > NR2)  IY = IY-N2R

	      DO  I=1,LSD,2
	         IX = (I-1) / 2

             F = 0.5*SQRT( &
                    FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1+FLOAT(IZ*IZ)/Z1)

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
              ENDDO
	   ENDDO
	ENDDO
	INV = -1
	CALL  FMRS_3(B,N2S,N2R,N2L,INV)
	IF (INV == 0) THEN
	   IRTFLG = -1
	   RETURN
	ENDIF
	END

	!--*********************************************************************
	SUBROUTINE RCOS_HP_3D(B,LSD,N2R,N2L,LCUT,HCUT,N2S,NX,NY,NZ,IRTFLG)

	REAL             :: B(LSD,N2R,N2L)
	DOUBLE PRECISION :: AVE
	REAL             :: F,F2
	REAL             :: FP, FS
	REAL             :: FP2, FS2,LCUT,HCUT
	INTEGER			 :: LSD,N2R,N2L,N2S,NX,NY,NZ,IRTFLG
    REAL, PARAMETER  :: PI = 3.14159265358979323846

!f2py threadsafe
!f2py intent(inplace) :: B
!f2py intent(in) :: LSD,N2S,N2R,N2L,NX,NY,NZ,LCUT,HCUT
!f2py intent(hide) :: LSD,N2R,N2L
!f2py intent(out) :: IRTFLG
!       BORDER PADDING

	IF (N2S.NE.NX .AND. N2R.NE.NY .AND. N2L.NE.NZ)  THEN

          AVE = (SUM(B(1:NX,1:NY,1))+SUM(B(1:NX,1:NY,NZ)) + &
                SUM(B(1:NX,1,2:NZ-1))+SUM(B(1:NX,NY,2:NZ-1)) + &
                SUM(B(1,2:NY-1,2:NZ-1))+SUM(B(NX,2:NY-1,2:NZ-1))) &
     		/REAL(4*(NX+NY+NZ)-16)

!$omp      parallel do private(i,j,k),reduction(+:ave)
	   DO K=1,NZ
	      DO J=1,NY
	         DO I=1,NX
	            AVE = AVE + B(I,J,K)
	         ENDDO
	      ENDDO
	   ENDDO

	   AVE = AVE/FLOAT(NX)/FLOAT(NY)/FLOAT(NZ)

!$omp      parallel do private(i,j,k)
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

!$omp      parallel do private(i,j,k)
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
	   IRTFLG = -1
	   RETURN
	ENDIF

	EPS   =  0.882
	AA    = 10.624
	FP=LCUT
	FS=HCUT
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
	NR2  = N2R / 2
	NL2  = N2L / 2
	X1   = FLOAT(N2S / 2)**2
	Y1   = FLOAT(NR2)**2
	Z1   = FLOAT(NL2)**2
	PARM = PARM1**2

	AVG = B(1,1,1)
!$omp   parallel do private(i,j,k,ix,iy,iz,f)
	DO K=1,N2L
	   IZ = K-1
	   IF (IZ > NL2)  IZ = IZ-N2L

	   DO J=1,N2R
	      IY = J-1
	      IF (IY > NR2)  IY = IY-N2R

	      DO  I=1,LSD,2
	         IX = (I-1) / 2

             F = 0.5*SQRT( &
                    FLOAT(IX*IX)/X1+FLOAT(IY*IY)/Y1+FLOAT(IZ*IZ)/Z1)

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
              ENDDO
	   ENDDO
	ENDDO
	B(1,1,1) = AVG
	INV = -1
	CALL  FMRS_3(B,N2S,N2R,N2L,INV)
	IF (INV == 0) THEN
	   IRTFLG = -1
	   RETURN
	ENDIF
	END



	!--*********************************************************************


         SUBROUTINE  HISTEQ(QK2,NSR1,QK6,NSR2,QK1,N,LENH,ITRMAX)

         REAL       ::  QK1(N),QK2(NSR1)
         LOGICAL    ::  QK6(NSR2)
         INTEGER    ::	NSR1,NSR2,N,LENH,ITRMAX

!f2py intent(in) :: NSR1,NSR2,N,LENH,ITRMAX
!f2py intent(inout) :: QK2, QK2, QK1
!f2py intent(hide) :: NSR1,NSR2,N
		CALL HISTC2(QK1,QK2,QK6,N,  NSR1,LENH,ITRMAX,NOUT)


         END




! **********************************************************************
!
!  RAMP(X,NX,NY,NOUT)
!
!--*********************************************************************

         SUBROUTINE RAMP(X,NX,NY,NOUT)

         IMPLICIT NONE

         INTEGER          :: NX,NY,NOUT
         REAL             :: X(NX,NY)

         EXTERNAL         :: BETAI
         DOUBLE PRECISION :: BETAI
         INTEGER          :: N1,N2,J,I,K,IX,IY
         REAL             :: DN

         DOUBLE PRECISION :: C,D,B1,B2,A,R2,DN1,DN2
         DOUBLE PRECISION :: Q(6),S(9),QYX1,QYX2,QX1X2 &
                                ,QX1,QX2,QY,SYX1,SYX2,SX1X2,SX1 &
                                ,SX2,SY,SX1Q,SX2Q,SYQ

         EQUIVALENCE (Q(1),QYX1),(Q(2),QYX2),(Q(3),QX1X2),(Q(4),QX1), &
                    (Q(5),QX2),(Q(6),QY)
         EQUIVALENCE (S(1),SYX1),(S(2),SYX2),(S(3),SX1X2),(S(4),SX1), &
                    (S(5),SX2),(S(6),SY),(S(7),SX1Q), &
                    (S(8),SX2Q),(S(9),SYQ)

         DOUBLE PRECISION, PARAMETER  :: EPS = 1.0D-5

!f2py intent(in) :: NX,NY
!f2py intent(inplace) :: X
!f2py intent(out) :: NOUT
!f2py intent(hide) :: NX,NY

!        ZERO ARRAY S
         S   = 0

         N1  = NX / 2
         N2  = NY / 2

         SX1 = FLOAT(N1) * FLOAT(NX + 1)
         IF (MOD(NX,2) ==  1)   SX1 = SX1 + 1 + N1

         SX2 = FLOAT(N2) * FLOAT(NY + 1)
         IF (MOD(NY,2) ==  1)   SX2 = SX2 + 1 + N2

         SX1   = SX1 * NY
         SX2   = SX2 * NX
         SX1X2 = 0.0D0

         DO  J = 1, NY
           DO I = 1, NX
             SYX1 = SYX1 + X(I,J) * I
             SYX2 = SYX2 + X(I,J) * J
             SY   = SY   + X(I,J)
             SX1Q = SX1Q + I * I
             SX2Q = SX2Q + J * J
             SYQ  = SYQ  + X(I,J) * DBLE(X(I,J))
           ENDDO
         ENDDO

         DN    = FLOAT(NX) * FLOAT(NY)
         QYX1  = SYX1 - SX1 * SY / DN
         QYX2  = SYX2 - SX2 * SY / DN
         QX1X2 = 0.0
         QX1   = SX1Q - SX1 * SX1 / DN
         QX2   = SX2Q - SX2 * SX2 / DN
         QY    = SYQ  - SY  * SY  / DN
         C     = QX1  * QX2 - QX1X2 * QX1X2
		 NOUT=0
         IF (C <= EPS) THEN
         	NOUT=1
            RETURN
         ENDIF

         B1  = (QYX1 * QX2 - QYX2 * QX1X2) / C
         B2  = (QYX2 * QX1 - QYX1 * QX1X2) / C
         A   = (SY - B1 * SX1 - B2 * SX2)  / DN
         D   = B1 * QYX1 + B2 * QYX2
         R2  = D / QY
         DN1 = 2
         DN2 = DN - 3

         D = A + B1 + B2

         DO IY = 1, NY
           QY = D

           DO IX = 1,NX
              X(IX,IY) = X(IX,IY) - QY
              QY     = QY + B1
           ENDDO

           D = D + B2
         ENDDO

         END
































