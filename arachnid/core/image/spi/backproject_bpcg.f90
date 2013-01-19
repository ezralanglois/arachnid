C ---------------------------------------------------------------------------

		SUBROUTINE SETUP_BP3F(TABI,LTAB,N2)


cf2py intent(in) ::  LTAB, N2
cf2py intent(inout) :: TABI


C		N = NSAM
C		LDP   = N / 2 + 1
C		LDPNM = LDP
C		NXLD = N + 2 - MOD(N,2)

C		CALL PREPCUB_S(N,NN,IDUM,RI,.FALSE.,LDP)
C		CALL PREPCUB_S(N,NN,ICUBE,RI,.TRUE.,LDP)   ! RETURNS: IPCUBE
C       NXLD = N + 2 - MOD(N,2)



        END

C ---------------------------------------------------------------------------

		SUBROUTINE FINALIZE_BP3F(X, NR, V, N2, N, NS)

		REAL          		       :: NR(0:N2,N,N)
        COMPLEX                    :: X(0:N2,N,N)
        REAL                       :: V(NS, NS, NS)

cf2py intent(inplace) :: X, NR, V
cf2py intent(in) :: N, N2, NS
cf2py intent(hide) :: N, N2, NS


		END

C ---------------------------------------------------------------------------

		SUBROUTINE BACKPROJECT_BP3F(PROJ,X,NR,TABI,NS,N,N2,L,PSI,THE,PHI)

		REAL        			   :: PROJ(NS,NS)
		REAL          		   	   :: NR(0:N2,N,N)
		COMPLEX                    :: X(0:N2,N,N)
		REAL                       :: DMS(3,3)
        REAL                  	   :: SS(6)
		REAL      	  			   :: TABI(L)

        COMPLEX, ALLOCATABLE, DIMENSION(:,:)   :: BI

c    (inout) :: PROJ,X,NR
cf2py intent(inplace) :: X,NR,TABI
cf2py intent(in) :: PROJ, PSI,THE,PHI
cf2py intent(hide) :: NS,N,N2,L



		END


