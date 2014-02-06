C*************************  ASTASQ  ********************************

        SUBROUTINE  ASTASQ(X,N,RI,ABA,KLP,SUS,SSQ,KLS)

        USE TYPE_KINDS

        REAL              :: X(N,N)
        DOUBLE PRECISION  :: ABA,SUS,SSQ
        INTEGER(KIND=I_8) :: KLP,KLS

C       ESTIMATE AVERAGE OUTSIDE THE CIRCLE.
C       RETURNS: ABA,KLP,SUS,SSQ,KLS

        R  = RI * RI
        NC = N/2+1

        DO J=1,N
           T  = J - NC
           XX = T*T

           DO I=1,N
              T=I - NC
              IF (XX + T*T > R)    THEN
C                OUTSIDE THE CIRCLE MASK.
                 ABA = ABA + DBLE(X(I,J))
                 KLP = KLP + 1
              ELSE
C                INSIDE THE CIRCLE MASK.
                 SSQ = SSQ + X(I,J) * DBLE(X(I,J))
                 SUS = SUS + X(I,J)
                 KLS = KLS + 1
              ENDIF
           ENDDO
        ENDDO
        END

C*************************  FIXEDGE1  ********************************

        SUBROUTINE  FIXEDGE1(BCKP,NNN,BCK3,N,IPCUBE,NN)

        REAL    :: BCKP(NNN),BCK3(N,N,N)
        INTEGER :: IPCUBE(5,NN)

C       PUT ZEROS OUTSIDE
        NT = 1
        DO I=1,NNN
           IF (NT .GT. NN)  THEN
              BCKP(I) = 0.0
           ELSEIF (I .LT. IPCUBE(1,NT))  THEN
              BCKP(I)= 0.0
           ELSEIF(I .EQ. IPCUBE(2,NT))  THEN
              NT = NT+1
           ENDIF
        ENDDO

C       ADD PIXELS ON THE EDGE
C       FIX THE EDGES IN BCKP
        DO  K=1,N
           DO J=1,N
              DO  I=1,N-1
                 IF (BCK3(I+1,J,K) .NE. 0.0) THEN
                    BCK3(I,J,K) = BCK3(I+1,J,K)
                    EXIT
                 ENDIF
              ENDDO

              DO  I=N,2,-1
                 IF (BCK3(I-1,J,K) .NE. 0.0) THEN
                    BCK3(I,J,K) =BCK3(I-1,J,K)
                    EXIT
                 ENDIF
              ENDDO
           ENDDO
        ENDDO
        DO  K=1,N
           DO I=1,N
              DO  J=1,N-1
                 IF (BCK3(I,J+1,K) .NE. 0.0) THEN
                    BCK3(I,J,K) = BCK3(I,J+1,K)
                    EXIT
                 ENDIF
              ENDDO

              DO  J=N,2,-1
                 IF (BCK3(I,J-1,K) .NE. 0.0) THEN
                    BCK3(I,J,K) = BCK3(I,J-1,K)
                    EXIT
                 ENDIF
              ENDDO
           ENDDO
        ENDDO
        DO  J=1,N
           DO I=1,N
              DO  K=1,N-1
                 IF (BCK3(I,J,K+1) .NE. 0.0) THEN
                    BCK3(I,J,K) = BCK3(I,J,K+1)
                    EXIT
                 ENDIF
              ENDDO

              DO  K=N,2,-1
                 IF (BCK3(I,J,K-1) .NE. 0.0) THEN
                    BCK3(I,J,K) = BCK3(I,J,K-1)
                    EXIT
                 ENDIF
              ENDDO
           ENDDO
        ENDDO
        END

C************************  FIXEDGE2 *********************************

        SUBROUTINE  FIXEDGE2(BCKP,NNN,BCK3,N,IPCUBE,NN)

        REAL    :: BCKP(NNN),BCK3(N,N,N)
        INTEGER :: IPCUBE(5,NN)

C       PUT ZEROS OUTSIDE

        NT = 1
        DO    I=1,NNN
           IF (NT .GT. NN)  THEN
              BCKP(I) = 0.0
           ELSEIF (I .LT. IPCUBE(1,NT))  THEN
              BCKP(I) = 0.0
           ELSEIF (I .EQ. IPCUBE(2,NT))  THEN
              NT = NT+1
           ENDIF
        ENDDO

C       ADD PIXELS ON THE EDGE
C       FIX THE EDGES IN BCKP
        DO  K=1,N
           DO J=1,N
              DO  I=2,N-1
                 IF (BCK3(I+1,J,K) .NE. 0.0) THEN
                     BCK3(I,J,K)   = BCK3(I+1,J,K)
                     BCK3(I-1,J,K) = BCK3(I+1,J,K)
                     EXIT
                 ENDIF
              ENDDO

              DO  I=N-1,2,-1
                 IF (BCK3(I-1,J,K) .NE. 0.0) THEN
                     BCK3(I,J,K)   = BCK3(I-1,J,K)
                     BCK3(I+1,J,K) = BCK3(I-1,J,K)
                     EXIT
                 ENDIF
             ENDDO
           ENDDO
        ENDDO
        DO  K=1,N
           DO I=1,N
              DO  J=2,N-1
                 IF(BCK3(I,J+1,K).NE.0.0) THEN
                    BCK3(I,J,K)=BCK3(I,J+1,K)
                    BCK3(I,J-1,K)=BCK3(I,J+1,K)
                    EXIT
                 ENDIF
              ENDDO

              DO  J=N-1,2,-1
                 IF(BCK3(I,J-1,K).NE.0.0) THEN
                    BCK3(I,J,K)   = BCK3(I,J-1,K)
                    BCK3(I,J+1,K) = BCK3(I,J-1,K)
                    EXIT
                 ENDIF
              ENDDO
           ENDDO
        ENDDO
        DO  J=1,N
           DO I=1,N
              DO  K=2,N-1
                 IF(BCK3(I,J,K+1).NE.0.0) THEN
                    BCK3(I,J,K)   = BCK3(I,J,K+1)
                    BCK3(I,J,K-1) = BCK3(I,J,K+1)
                    EXIT
                 ENDIF
              ENDDO

              DO  K=N-1,2,-1
                 IF(BCK3(I,J,K-1).NE.0.0) THEN
                    BCK3(I,J,K)   = BCK3(I,J,K-1)
                    BCK3(I,J,K+1) = BCK3(I,J,K-1)
                   EXIT
                 ENDIF
              ENDDO
           ENDDO
        ENDDO
        END

C************************  FIXEDGE3 ********************************

        SUBROUTINE  FIXEDGE3(BCKP,NNN,BCK3,N,IPCUBE,NN)

        REAL    :: BCKP(NNN),BCK3(N,N,N)
        INTEGER :: IPCUBE(5,NN)

C       PUT ZEROS OUTSIDE

        NT = 1
        DO I=1,NNN
           IF (NT.GT.NN)  THEN
              BCKP(I) = 0.0
           ELSEIF (I .LT. IPCUBE(1,NT))  THEN
              BCKP(I) = 0.0
           ELSEIF (I .EQ. IPCUBE(2,NT))  THEN
              N T= NT+1
           ENDIF
        ENDDO

C       ADD PIXELS ON THE EDGE
C       FIX THE EDGES IN BCKP
        DO  K=1,N
           DO J=1,N
              DO  I=3,N-1
                 IF(BCK3(I+1,J,K) .NE. 0.0) THEN
                    BCK3(I,J,K)   = BCK3(I+1,J,K)
                    BCK3(I-1,J,K) = BCK3(I+1,J,K)
                    BCK3(I-2,J,K) = BCK3(I+1,J,K)
                    EXIT
                 ENDIF
              ENDDO

              DO  I=N-2,2,-1
                 IF(BCK3(I-1,J,K) .NE. 0.0) THEN
                    BCK3(I,J,K)   = BCK3(I-1,J,K)
                    BCK3(I+1,J,K) = BCK3(I-1,J,K)
                    BCK3(I+2,J,K) = BCK3(I-1,J,K)
                    EXIT
                 ENDIF
              ENDDO
           ENDDO
        ENDDO

        DO  K=1,N
           DO I=1,N
              DO  J=3,N-1
                 IF(BCK3(I,J+1,K) .NE. 0.0) THEN
                    BCK3(I,J,K)   = BCK3(I,J+1,K)
                    BCK3(I,J-1,K) = BCK3(I,J+1,K)
                    BCK3(I,J-2,K) = BCK3(I,J+1,K)
                    EXIT
                 ENDIF
              ENDDO

              DO  J=N-2,2,-1
                 IF(BCK3(I,J-1,K) .NE. 0.0) THEN
                    BCK3(I,J,K)   = BCK3(I,J-1,K)
                    BCK3(I,J+1,K) = BCK3(I,J-1,K)
                    BCK3(I,J+2,K) = BCK3(I,J-1,K)
                   EXIT
                 ENDIF
              ENDDO
           ENDDO
        ENDDO
        DO  J=1,N
           DO I=1,N
              DO  K=3,N-1
                 IF(BCK3(I,J,K+1) .NE. 0.0) THEN
                    BCK3(I,J,K)   = BCK3(I,J,K+1)
                    BCK3(I,J,K-1) = BCK3(I,J,K+1)
                    BCK3(I,J,K-2) = BCK3(I,J,K+1)
                    EXIT
                 ENDIF
              ENDDO

              DO  K=N-2,2,-1
                 IF(BCK3(I,J,K-1) .NE. 0.0) THEN
                    BCK3(I,J,K)   = BCK3(I,J,K-1)
                    BCK3(I,J,K+1) = BCK3(I,J,K-1)
                    BCK3(I,J,K+2) = BCK3(I,J,K-1)
                    EXIT
                 ENDIF
              ENDDO
           ENDDO
        ENDDO
        END

C       ********************  BFIRSTS   ******************************

        SUBROUTINE  BFIRSTS(BCKE,BCKP,N,ANOISE,IPCUBE,NN)

        REAL      ::  BCKE(N,N,N),BCKP(N,N,N)
        INTEGER   ::  IPCUBE(5,NN)

c$omp   parallel do private(kn,j,k,i)
        DO KN=1,NN
           J = IPCUBE(4,KN)
           K = IPCUBE(5,KN)

           DO I=IPCUBE(3,KN),IPCUBE(3,KN)+IPCUBE(2,KN)-IPCUBE(1,KN)

              BCKE(I,J,K) = BCKE(I,J,K) + ANOISE*(6*BCKP(I,J,K)-(
     &           + BCKP(I+1,J,K) + BCKP(I,J+1,K) + BCKP(I,J,K+1)
     &           + BCKP(I-1,J,K) + BCKP(I,J-1,K) + BCKP(I,J,K-1)))
           ENDDO
        ENDDO
        END

C****************************  BSECOND ******************************

        SUBROUTINE  BSECOND(BCKE,BCKP,N,ANOISE,IPCUBE,NN)

        REAL      ::  BCKE(N,N,N),BCKP(N,N,N)
        INTEGER   ::  IPCUBE(5,NN)

c$omp   parallel do private(kn,j,k,i)
        DO KN=1,NN
           J = IPCUBE(4,KN)
           K = IPCUBE(5,KN)
           DO I=IPCUBE(3,KN),IPCUBE(3,KN)+IPCUBE(2,KN)-IPCUBE(1,KN)

               BCKE(I,J,K) = BCKE(I,J,K) + ANOISE*(18*BCKP(I,J,K)
     &          - 4.0*BCKP(I+1,J,K)+BCKP(I+2,J,K)
     &          - 4.0*BCKP(I-1,J,K)+BCKP(I-2,J,K)
     &          - 4.0*BCKP(I,J+1,K)+BCKP(I,J+2,K)
     &          - 4.0*BCKP(I,J-1,K)+BCKP(I,J-2,K)
     &          - 4.0*BCKP(I,J,K+1)+BCKP(I,J,K+2)
     &          - 4.0*BCKP(I,J,K-1)+BCKP(I,J,K-2)
     &          )
           ENDDO
        ENDDO
        END

C       ********************** BTHIRD *******************************

        SUBROUTINE  BTHIRD(BCKE,BCKP,N,ANOISE,IPCUBE,NN)

        REAL      :: BCKE(N,N,N),BCKP(N,N,N)
        INTEGER   :: IPCUBE(5,NN)

c$omp   parallel do private(kn,j,k,i)
        DO KN=1,NN
           J = IPCUBE(4,KN)
           K = IPCUBE(5,KN)
           DO I=IPCUBE(3,KN),IPCUBE(3,KN)+IPCUBE(2,KN)-IPCUBE(1,KN)

                 BCKE(I,J,K) = BCKE(I,J,K) + ANOISE*(60*BCKP(I,J,K)
     &          - 15.0*BCKP(I+1,J,K)+6.0*BCKP(I+2,J,K)-BCKP(I+3,J,K)
     &          - 15.0*BCKP(I-1,J,K)+6.0*BCKP(I-2,J,K)-BCKP(I-3,J,K)
     &          - 15.0*BCKP(I,J+1,K)+6.0*BCKP(I,J+2,K)-BCKP(I,J+3,K)
     &          - 15.0*BCKP(I,J-1,K)+6.0*BCKP(I,J-2,K)-BCKP(I,J-3,K)
     &          - 15.0*BCKP(I,J,K+1)+6.0*BCKP(I,J,K+2)-BCKP(I,J,K+3)
     &          - 15.0*BCKP(I,J,K-1)+6.0*BCKP(I,J,K-2)-BCKP(I,J,K-3)
     &          )
           ENDDO
        ENDDO
        END
