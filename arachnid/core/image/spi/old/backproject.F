C++*********************************************************************
C--*********************************************************************
		SUBROUTINE FINALIZE(X, NX, N2, W)

		REAL          			    W(0:NX, N2,N2)
		COMPLEX                     X(0:NX, N2,N2)

		EXTERNAL  SYMPLANE0
		EXTERNAL  NRMW2

cf2py intent(inout) :: X, W
cf2py intent(in) :: NX, N2
cf2py intent(hide) :: NX, N2

        CALL SYMPLANE0(X,W,NX,N2)
        CALL NRMW2(X,W,NX,N2)


		END

C ---------------------------------------------------------------------------

		SUBROUTINE WINDOW(X, NX, N2)

		REAL          			    W(0:NX, N2,N2)
		COMPLEX                     X(0:NX, N2,N2)

cf2py intent(in) :: NX,N2,ALPHA,AAAA,NN
cf2py intent(inout) :: X
        LN    = 5                           ! ALWAYS=5
        LN2   = LN / 2                      ! ALWAYS=2
		V     = REAL(LN-1) / 2.0 / REAL(N2) ! ALWAYS=4*N2
		ALPHA = 6.5                         ! ALWAYS=6.5
		AAAA  = 0.9*V                       ! ALWAYS=.9*4*N2
		NNN   = 3                           ! ALWAYS=2
		LSD     = N2 + 2 - MOD(N2,2)
        CALL WINDKB2A(X,X,NX,NX+1,N2,ALPHA,AAAA,NNN)
C        CALL WINDKB2A(X,X,NX,LSD,N2,ALPHA,AAAA,NNN)

		END


C ---------------------------------------------------------------------------

		SUBROUTINE BACKPROJECT(BI, DMS,X,W,TABI,LTAB,NX,N2)

C		REAL                        :: SM(3,3,MAXSYM)
		REAL                       DMS(3,3)
		COMPLEX        			   BI(0:NX,N2)
		REAL          			   W(0:NX, N2,N2)
		COMPLEX                    X(0:NX, N2,N2)
		REAL      	  			   TABI(LTAB)

		LN1 = 6
		LN2 = 2
		FLTB = REAL(LTAB) / REAL(LN2+1)

cf2py intent(in) :: LTAB
cf2py intent(inout) :: BI, DMS, X, W, TABI
cf2py intent(hide) :: LTAB
cNX, N2,

c$omp      parallel do private(i,j)
           DO J=1,N2
              DO I=0,NX
                 BI(I,J) = BI(I,J)*(-1)**(I+J+1)
              ENDDO
           ENDDO
C ---------------------
C Symetries disabled
C		   MAXSYM=1
C ---------------------
C           DO ISYM=1,MAXSYM
C             IF (MAXSYM .GT. 1)  THEN
C               SYMMETRIES, MULTIPLY MATRICES
C                DMS = MATMUL(SM(:,:,ISYM),DM(:,:))
C             ELSE
C                DMS = DM(:,:)
C             ENDIF
             DO J=-NX+1,NX
               CALL ONELINE(J,N2,NX,X,W,BI,DMS,LN2,FLTB,LTAB,TABI)
             ENDDO

C ---------------------
C Symetries disabled
C           ENDDO
C ---------------------

		END

		SUBROUTINE FILLBESSIL(TABI,LTAB,N2)

        REAL      TABI(0:LTAB)

cf2py intent(in) ::  LTAB, N2
cf2py intent(inout) :: TABI


C       GENERALIZED KAISER-BESSEL WINDOW ACCORDING TO LEWITT
        LN    = 5                           ! ALWAYS=5
        LN2   = LN / 2                      ! ALWAYS=2
		V     = REAL(LN-1) / 2.0 / REAL(N2) ! ALWAYS=4*N2
		ALPHA = 6.5                         ! ALWAYS=6.5
		AAAA  = 0.9*V                       ! ALWAYS=.9*4*N2
		NNN   = 3                           ! ALWAYS=2

C       GENERATE TABLE WITH INTERPOLANTS
 		B0   = SQRT(ALPHA) * BESI1(ALPHA)

        FLTB = REAL(LTAB) / REAL(LN2+1)

cc$omp  parallel do private(i,s,xt)
        DO I=0,LTAB
	   S = REAL(I) / FLTB / N2
	   IF (S .LE. AAAA)  THEN
	      XT      = SQRT(1.0 - (S/AAAA)**2)
	      TABI(I) = SQRT(ALPHA*XT) * BESI1(ALPHA*XT) / B0
	   ELSE
	      TABI(I) = 0.0
	   ENDIF
        ENDDO

        END

        SUBROUTINE CANG(PHI,THETA,PSI,DOSS,SS,DM)

        REAL              PHI,THETA,PSI
        LOGICAL           DOSS
        DOUBLE PRECISION  CPHI,SPHI,CTHE,STHE,CPSI,SPSI
        REAL              DM(9),SS(6)

		DOUBLE PRECISION  QUADPI, DGR_TO_RAD
		PARAMETER (QUADPI = 3.141592653589793238462643383279502884197)
		PARAMETER (DGR_TO_RAD = (QUADPI/180))

cf2py intent(in) ::  PHI,THETA,PSI,DOSS
cf2py intent(inout) :: SS,DM

        CPHI = DCOS(DBLE(PHI)   * DGR_TO_RAD)
        SPHI = DSIN(DBLE(PHI)   * DGR_TO_RAD)
        CTHE = DCOS(DBLE(THETA) * DGR_TO_RAD)
        STHE = DSIN(DBLE(THETA) * DGR_TO_RAD)
        CPSI = DCOS(DBLE(PSI)   * DGR_TO_RAD)
        SPSI = DSIN(DBLE(PSI)   * DGR_TO_RAD)

        IF (DOSS) THEN
C          WANT TO RETURN SS
	   SS(1) = SNGL(CPHI)
	   SS(2) = SNGL(SPHI)
	   SS(3) = SNGL(CTHE)
	   SS(4) = SNGL(STHE)
	   SS(5) = SNGL(CPSI)
	   SS(6) = SNGL(SPSI)
        ENDIF

        DM(1) =  CPHI*CTHE*CPSI - SPHI*SPSI
        DM(2) =  SPHI*CTHE*CPSI + CPHI*SPSI
        DM(3) = -STHE*CPSI
        DM(4) = -CPHI*CTHE*SPSI - SPHI*CPSI
        DM(5) = -SPHI*CTHE*SPSI + CPHI*CPSI
        DM(6) =  STHE*SPSI
        DM(7) =  STHE*CPHI
        DM(8) =  STHE*SPHI
        DM(9) =  CTHE

        END

		SUBROUTINE PADD(PROJ,BI,L,N,N2,LSD)

C       PADS: PROJ OF SIZE: L  INTO: BI  WITH SIZE: N

        DIMENSION  PROJ(L,L),BI(N2,N)

cf2py intent(inout) :: PROJ, BI
cf2py intent(in) :: L, N, N2, LSD

		CALL PADD2(PROJ,L,BI,N,N2)

        END
