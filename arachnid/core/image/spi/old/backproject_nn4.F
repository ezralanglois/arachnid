C++*********************************************************************
C--*********************************************************************
		SUBROUTINE FINALIZE_NN4(X, NR, N2, N)

		REAL          			   NR(0:N2,N,N)
		COMPLEX                    X(0:N2,N,N)

cf2py intent(inout) :: X, W
cf2py intent(in) :: NX, N2
cf2py intent(hide) :: NX, N2

        CALL SYMPLANEI(X,NR,N2,N)
        CALL NORMN4(X,NR,N2,N)



		END

C ---------------------------------------------------------------------------

		SUBROUTINE WINDOW_NN4(X, N, N2, NS)

		COMPLEX                    X(0:N2,N,N)

cf2py intent(in) :: NS
cf2py intent(inout) :: X

		LSD    = N2+2-MOD(N2,2)
        CALL WINDUM(X,X,NS,LSD,N)

		END


C ---------------------------------------------------------------------------

		SUBROUTINE BACKPROJECT_NN4(BI,X,NR,N,N2,DMS,SM,MAXSYM)

		REAL                       DM(3,3)
		COMPLEX        			   BI(0:N2,N)
		REAL          			   NR(0:N2,N,N)
		COMPLEX                    X(0:N2,N,N)
		REAL                       DMS(3,3)
        REAL                  	   SM(3,3,MAXSYM)

cf2py intent(inout) :: BI,X,NR,DMS,SM

c$omp      parallel do private(i,j)
           DO J=1,N
              DO I=0,N2
                 BI(I,J) = BI(I,J) * (-1)**(I+J+1)
              ENDDO
           ENDDO

           DO ISYM=1,MAXSYM
              IF (MAXSYM .GT. 1)  THEN
C                SYMMETRIES, MULTIPLY MATRICES
C                 DMS = MATMUL(SM(:,:,ISYM),DM)
              ELSE
C                 DMS = DM
              ENDIF
C,schedule(static)
c$omp         parallel do private(j),shared(N,N2,JT,X,NR,BI,DMS)
              DO J=-N2+1,N2
                 CALL ONELINENN(J,N,N2,X,NR,BI,DMS)
              ENDDO
           ENDDO   ! END OF SYMMETRIES LOOP

		END


