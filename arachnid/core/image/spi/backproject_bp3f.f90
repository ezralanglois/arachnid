C ---------------------------------------------------------------------------

		SUBROUTINE SETUP_BP3F(TABI,LTAB,N2)

        REAL      TABI(0:LTAB)
cf2py threadsafe
cf2py intent(in) ::  LTAB, N2
cf2py intent(inout) :: TABI

		print *, 'SETUP_BP3F-1'

C       GENERALIZED KAISER-BESSEL WINDOW ACCORDING TO LEWITT
        LN    = 5                           ! ALWAYS=5
        LN2   = LN / 2                      ! ALWAYS=2
		V     = REAL(LN-1) / 2.0 / REAL(N2) ! ALWAYS=4*N2
		ALPHA = 6.5                         ! ALWAYS=6.5
		AAAA  = 0.9*V                       ! ALWAYS=.9*4*N2
		NNN   = 3                           ! ALWAYS=2

		print *, 'SETUP_BP3F-2'
C       GENERATE TABLE WITH INTERPOLANTS
 		B0   = SQRT(ALPHA) * BESI1(ALPHA)

        FLTB = REAL(LTAB) / REAL(LN2+1)
		print *, 'SETUP_BP3F-3'

cc  parallel do private(i,s,xt)
        DO I=0,LTAB
		print *, 'LTAB:Ê', I, LTAB
	   S = REAL(I) / FLTB / N2
	   IF (S .LE. AAAA)  THEN
	      XT      = SQRT(1.0 - (S/AAAA)**2)
	      TABI(I) = SQRT(ALPHA*XT) * BESI1(ALPHA*XT) / B0
	   ELSE
	      TABI(I) = 0.0
	   ENDIF
        ENDDO
		print *, 'SETUP_BP3F-4'
        END

C ---------------------------------------------------------------------------

		SUBROUTINE FINALIZE_BP3F(X, NR, V, N2, N, NS)

		REAL          		       :: NR(0:N2,N,N)
        COMPLEX                    :: X(0:N2,N,N)
        REAL                       :: V(NS, NS, NS)
cf2py threadsafe
cf2py intent(inplace) :: X, NR, V
cf2py intent(in) :: N, N2, NS
cf2py intent(hide) :: N, N2, NS

		print *, 'FINALIZE_BP3F-f90-1'
		LSD    = N+2-MOD(N,2)

C       SYMMETRIZE PLANE: 0
		print *, 'FINALIZE_BP3F-f90-2-SYMPLANE0'
        CALL SYMPLANE0(X,NR,N2,N)

		print *, 'FINALIZE_BP3F-f90-3-NRMW2'
		print *, "N2=", N2
		print *, "NS=", NS
C       CALCULATE REAL SPACE VOLUME
        CALL NRMW2(X,NR,N2,N)

        LN    = 5                           ! ALWAYS=5
	    LN2   = LN / 2                      ! ALWAYS=2
		V1     = REAL(LN-1) / 2.0 / REAL(N) ! ALWAYS=4*N2
		ALPHA = 6.5                         ! ALWAYS=6.5
		AAAA  = 0.9*V1                       ! ALWAYS=.9*4*N2
		NNN   = 3

		print *, 'FINALIZE_BP3F-f90-4-WINDKB2A '
C       WINDOW?
        CALL WINDKB2A(X,V,NS,LSD,N,ALPHA,AAAA,NNN)

		print *, 'FINALIZE_BP3F-f90-5-FMRS_DEPLAN'
        CALL FMRS_DEPLAN(IRTFLG)
		print *, 'FINALIZE_BP3F-f90-5-FMRS_DEPLAN-Done'

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
cf2py threadsafe
c    (inout) :: PROJ,X,NR
cf2py intent(inplace) :: X,NR,TABI
cf2py intent(in) :: PROJ, PSI,THE,PHI
cf2py intent(hide) :: NS,N,N2,L

		LN1 = 6
		LN2 = 2
		FLTB = REAL(L) / REAL(LN2+1)

		CALL CANG(PHI,THE,PSI,.FALSE.,SS,DMS)

		ALLOCATE(BI(0:N2,N), STAT=IRTFLG)
        IF (IRTFLG .NE. 0) THEN
C           MWANT = NS*NS + (N2+1)*N
C           CALL ERRT(46,'BP NF, PROJ, BI',MWANT)
           GOTO 999
        ENDIF

		LSD    = N+2-MOD(N,2)
        CALL PADD2(PROJ,NS,BI,LSD,N)
        INV = +1
        CALL FMRS_2(BI,N,N,INV)

c$omp      parallel do private(i,j)
           DO J=1,N
              DO I=0,N2
                 BI(I,J) = BI(I,J) * (-1)**(I+J+1)
              ENDDO
           ENDDO

C           DO ISYM=1,MAXSYM
C              IF (MAXSYM .GT. 1)  THEN
C                SYMMETRIES, MULTIPLY MATRICES
C                 DMS = MATMUL(SM(:,:,ISYM),DM)
C              ELSE
C                 DMS = DM
C              ENDIF
             DO J=-N2+1,N2
               CALL ONELINE(J,N,N2,X,NR,BI,DMS,LN2,FLTB,L,TABI)
             ENDDO
C           ENDDO   ! END OF SYMMETRIES LOOP


999     IF (ALLOCATED(BI))   DEALLOCATE (BI)

		END


