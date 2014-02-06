! ---------------------------------------------------------------------------

		SUBROUTINE SETUP_BP3F(TABI,LTAB,N2)

        REAL      :: TABI(0:LTAB)
        INTEGER   :: LTAB, N2
!f2py threadsafe
!f2py intent(in) ::  LTAB, N2
!f2py intent(inout) :: TABI
!f2py intent(hide) :: LTAB

!       GENERALIZED KAISER-BESSEL WINDOW ACCORDING TO LEWITT
        LN    = 5                           ! ALWAYS=5
        LN2   = LN / 2                      ! ALWAYS=2
		V     = REAL(LN-1) / 2.0 / REAL(N2) ! ALWAYS=4*N2
		ALPHA = 6.5                         ! ALWAYS=6.5
		AAAA  = 0.9*V                       ! ALWAYS=.9*4*N2
		NNN   = 3                           ! ALWAYS=2

!       GENERATE TABLE WITH INTERPOLANTS
 		B0   = SQRT(ALPHA) * BESI1(ALPHA)

        FLTB = REAL(LTAB) / REAL(LN2+1)

!c  parallel do private(i,s,xt)
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

! ---------------------------------------------------------------------------

		SUBROUTINE FINALIZE_BP3F(X, NR, V, N2, N, NS)

		REAL          		       :: NR(0:N2,N,N)
        COMPLEX                    :: X(0:N2,N,N)
        REAL                       :: V(NS, NS, NS)
        INTEGER   				   :: N2, N, NS
!f2py threadsafe
!f2py intent(inplace) :: X, NR, V
!f2py intent(in) :: N, N2, NS
!f2py intent(hide) :: N, N2, NS

		LSD    = N+2-MOD(N,2)

!       SYMMETRIZE PLANE: 0
        CALL SYMPLANE0(X,NR,N2,N)

!       CALCULATE REAL SPACE VOLUME
        CALL NRMW2(X,NR,N2,N)

        LN    = 5                           ! ALWAYS=5
	    LN2   = LN / 2                      ! ALWAYS=2
		V1     = REAL(LN-1) / 2.0 / REAL(N) ! ALWAYS=4*N2
		ALPHA = 6.5                         ! ALWAYS=6.5
		AAAA  = 0.9*V1                       ! ALWAYS=.9*4*N2
		NNN   = 3

!       WINDOW?
        CALL WINDKB2A(X,V,NS,LSD,N,ALPHA,AAAA,NNN)

		END

! ---------------------------------------------------------------------------

		SUBROUTINE CLEANUP_BP3F()


        CALL FMRS_DEPLAN(IRTFLG)

		END

! ---------------------------------------------------------------------------


		SUBROUTINE BACKPROJECT_BP3F(PROJ,X,NR,TABI,NS,N,N2,L,PSI,THE,PHI)

		REAL        			   :: PROJ(NS,NS)
		REAL          		   	   :: NR(0:N2,N,N)
		COMPLEX                    :: X(0:N2,N,N)
		REAL                       :: DMS(3,3)
        REAL                  	   :: SS(6)
		REAL      	  			   :: TABI(L)
		REAL					   :: PSI,THE,PHI
		INTEGER					   :: NS,N,N2,L

        COMPLEX, ALLOCATABLE, DIMENSION(:,:)   :: BI
!f2py threadsafe
!    (inout) :: PROJ,X,NR
!f2py intent(inplace) :: X,NR,TABI
!f2py intent(in) :: PROJ, PSI,THE,PHI
!f2py intent(hide) :: NS,N,N2,L

		LN1 = 6
		LN2 = 2
		FLTB = REAL(L) / REAL(LN2+1)

		CALL CANG(PHI,THE,PSI,.FALSE.,SS,DMS)

		ALLOCATE(BI(0:N2,N), STAT=IRTFLG)
        IF (IRTFLG .NE. 0) THEN
!           MWANT = NS*NS + (N2+1)*N
!           CALL ERRT(46,'BP NF, PROJ, BI',MWANT)
           GOTO 999
        ENDIF

		LSD    = N+2-MOD(N,2)
        CALL PADD2(PROJ,NS,BI,LSD,N)
        INV = +1
        CALL FMRS_2(BI,N,N,INV)

!$omp      parallel do private(i,j)
           DO J=1,N
              DO I=0,N2
                 BI(I,J) = BI(I,J) * (-1)**(I+J+1)
              ENDDO
           ENDDO

!           DO ISYM=1,MAXSYM
!              IF (MAXSYM .GT. 1)  THEN
!                SYMMETRIES, MULTIPLY MATRICES
!                 DMS = MATMUL(SM(:,:,ISYM),DM)
!              ELSE
!                 DMS = DM
!              ENDIF
             DO J=-N2+1,N2
               CALL ONELINE(J,N,N2,X,NR,BI,DMS,LN2,FLTB,L,TABI)
             ENDDO
!           ENDDO   ! END OF SYMMETRIES LOOP


999     IF (ALLOCATED(BI))   DEALLOCATE (BI)

		END


