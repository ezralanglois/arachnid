C ---------------------------------------------------------------------------

		SUBROUTINE FINALIZE_NN4F(X, NR, V, N2, N, NS)

		INTEGER          		   :: NR(0:N2,N,N)
        COMPLEX                    :: X(0:N2,N,N)
        REAL                       :: V(NS, NS, NS)

cf2py threadsafe
cf2py intent(inplace) :: X, NR, V
cf2py intent(in) :: NX, N2, NS
cf2py intent(hide) :: NX, N2, NS

		LSD    = N+2-MOD(N,2)
C		print *, 'LSD_last=',LSD
C		print *, 'NS_last=',NS
C       SYMMETRIZE PLANE: 0
        CALL SYMPLANEI(X,NR,N2,N)

C       CALCULATE REAL SPACE VOLUME
        CALL NORMN4(X,NR,N2,N)

C       WINDOW?
        CALL WINDUM(X,V,NS,LSD,N)



		END

C ---------------------------------------------------------------------------

		SUBROUTINE FINALIZE_NN4(X, NR, N2, N, NS)

		INTEGER          		   :: NR(0:N2,N,N)
        COMPLEX                    :: X(0:N2,N,N)

cf2py threadsafe
cf2py intent(inplace) :: X, NR
cf2py intent(in) :: NX, N2, NS
cf2py intent(hide) :: NX, N2

		LSD    = N+2-MOD(N,2)
C		print *, 'LSD_last=',LSD
C		print *, 'NS_last=',NS
C       SYMMETRIZE PLANE: 0
        CALL SYMPLANEI(X,NR,N2,N)

C       CALCULATE REAL SPACE VOLUME
        CALL NORMN4(X,NR,N2,N)

C       WINDOW?
        CALL WINDUM(X,X,NS,LSD,N)
		CALL FMRS_DEPLAN(IRTFLG)


		END


C ---------------------------------------------------------------------------

		SUBROUTINE BACKPROJECT_NN4F(PROJ,X,NR,NS,N,N2,PSI,THETA,PHI)

C		REAL                  	   :: DM(3,3)
		REAL        			   :: PROJ(NS,NS)
		INTEGER          		   :: NR(0:N2,N,N)
		COMPLEX                    :: X(0:N2,N,N)
		REAL                       :: DMS(3,3)
C        REAL                  	   :: SM(3,3,MAXSYM)
        REAL                  	   :: SS(6)

        COMPLEX, ALLOCATABLE, DIMENSION(:,:)   :: BI
cf2py threadsafe
c    (inout) :: PROJ,X,NR
cf2py intent(inplace) :: X,NR
cf2py intent(in) :: PROJ, PSI,THETA,PHI

		CALL CANG(PHI,THETA,PSI,.FALSE.,SS,DMS)

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
C,schedule(static)
c$omp         parallel do private(j),shared(N,N2,JT,X,NR,BI,DMS)
              DO J=-N2+1,N2
                 CALL ONELINENN(J,N,N2,X,NR,BI,DMS)
              ENDDO
C           ENDDO   ! END OF SYMMETRIES LOOP


999     IF (ALLOCATED(BI))   DEALLOCATE (BI)

		END




