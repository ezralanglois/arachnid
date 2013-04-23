
C       ------------------- NN4 -------------------------------

C       --------------------- ONELINENN ---------------------------------

        SUBROUTINE  ONELINENN(J,N,N2,X,NR,BI,DM)

        DIMENSION      :: NR(0:N2,N,N)
        COMPLEX        :: BI(0:N2,N),X(0:N2,N,N),BTQ
        DIMENSION      :: DM(6)

        IF (J .GE. 0) THEN
           JP = J+1
        ELSE
           JP = N+J+1
        ENDIF

        DO  I=0,N2
           IF ((I*I+J*J.LT.N*N/4) .AND. .NOT.(I.EQ.0.AND.J.LT.0)) THEN
              XNEW = I*DM(1)+J*DM(4)
              YNEW = I*DM(2)+J*DM(5)
              ZNEW = I*DM(3)+J*DM(6)
              IF (XNEW .LT. 0.0)  THEN
                 XNEW = -XNEW
                 YNEW = -YNEW
                 ZNEW = -ZNEW
                 BTQ  = CONJG(BI(I,JP))
              ELSE
                 BTQ = BI(I,JP)
              ENDIF


C              IF (BTQ.NE.BTQ) THEN
C              	print *, 'I=',I, N2+1
C              	print *, 'JP=',JP, N
C              ENDIF

              IXN = IFIX(XNEW+0.5+N) - N
              IYN = IFIX(YNEW+0.5+N) - N
              IZN = IFIX(ZNEW+0.5+N) - N
              IF (IXN.LE.N2 .AND.
     &            IYN.GE.-N2.AND.IYN.LE.N2 .AND.
     &            IZN.GE.-N2.AND.IZN.LE.N2) THEN
                 IF (IXN .GE. 0) THEN
                    IF (IZN .GE. 0) THEN
                       IZA = IZN+1
                    ELSE
                       IZA = N+IZN+1
                    ENDIF

                    IF (IYN .GE. 0) THEN
                       IYA = IYN+1
                    ELSE
                       IYA = N+IYN+1
                    ENDIF

                    X(IXN,IYA,IZA)  = X(IXN,IYA,IZA)+BTQ
                    NR(IXN,IYA,IZA) = NR(IXN,IYA,IZA)+1
                 ELSE
                    IF (IZN .GT. 0)  THEN
                       IZT = N-IZN+1
                    ELSE
                       IZT = -IZN+1
                    ENDIF

                    IF (IYN .GT. 0) THEN
                       IYT = N- IYN + 1
                    ELSE
                       IYT = -IYN + 1
                    ENDIF

                    X(-IXN,IYT,IZT)  = X(-IXN,IYT,IZT)+CONJG(BTQ)
                    NR(-IXN,IYT,IZT) = NR(-IXN,IYT,IZT)+1
                 ENDIF
              ENDIF
           ENDIF
        ENDDO   !  END J-I LOOP

        END

C       ------------------- WINDUM -------------------------------

        SUBROUTINE WINDUM(BI,R,L,LSD,N)

        DIMENSION  R(L,L,L),BI(LSD,N,N)


        IP = (N-L) / 2 + MOD(L,2)
        DO K=1,L
           DO J=1,L
              DO I=1,L
                 R(I,J,K) = BI(IP+I,IP+J,IP+K)
              ENDDO
           ENDDO
        ENDDO

        L2  = (L/2)**2
        L2P = (L/2-1)**2
        IP  = L / 2 + 1
        TNR = 0.0
        M   = 0

        DO K=1,L
           DO J=1,L
              DO I=1,L
                 LR = (K-IP)**2+(J-IP)**2+(I-IP)**2
                 IF (LR .LE. L2) THEN
                    IF (LR.GE.L2P .AND. LR.LE.L2) THEN
                       TNR = TNR + R(I,J,K)
                       M   = M+1
                    ENDIF
                 ENDIF
              ENDDO
           ENDDO
        ENDDO

        TNR = TNR/REAL(M)
c$omp   parallel do private(i,j,k,lr)
        DO  K=1,L
           DO  J=1,L
              DO  I=1,L
                 LR = (K-IP)**2 + (J-IP)**2 + (I-IP)**2
                 IF (LR .LE. L2) THEN
                    R(I,J,K) = R(I,J,K) - TNR
                 ELSE
                    R(I,J,K) = 0.0
                 ENDIF
              ENDDO
           ENDDO
        ENDDO

        END


C       ------------------- NORMN4 -------------------------------

        SUBROUTINE  NORMN4(X,NR,N2,N)

        DIMENSION  :: NR(0:N2,N,N)
        COMPLEX    :: X(0:N2,N,N)

c$omp   parallel do private(i,j,k)
        DO K=1,N
           DO J=1,N
              DO I=0,N2
                 IF (NR(I,J,K).GT.0)
     &               X(I,J,K) = X(I,J,K) * (-1)**(I+J+K) / NR(I,J,K)
              ENDDO
           ENDDO
        ENDDO

C       FOURIER BACK TRANSFORM
        INV = -1
        CALL FMRS_3(X,N,N,N,INV)

        END

C       ----------------SYMPLANEI ---------------------------------------

        SUBROUTINE  SYMPLANEI(X,W,N2,N)

C       POSSIBLE PURPOSE??: CaLCULATE WIENER SUMMATION FROM THE
C       INSERTED 2D SLICE PUT THE SUMMATION INTO 3D GRIDS USING
C       NEAREST NEIGHBOUR APPROXIMATION

        INTEGER  :: W(0:N2,N,N)
        COMPLEX  :: X(0:N2,N,N)

C       SYMMETRIZE PLANE 0
        DO  IZA=2,N2
           DO  IYA=2,N2
              X(0,IYA,IZA) = X(0,IYA,IZA)+CONJG(X(0,N-IYA+2,N-IZA+2))
              W(0,IYA,IZA) = W(0,IYA,IZA)+W(0,N-IYA+2,N-IZA+2)
              X(0,N-IYA+2,N-IZA+2) = CONJG(X(0,IYA,IZA))
              W(0,N-IYA+2,N-IZA+2) = W(0,IYA,IZA)
              X(0,N-IYA+2,IZA)=X(0,N-IYA+2,IZA)+CONJG(X(0,IYA,N-IZA+2))
              W(0,N-IYA+2,IZA) = W(0,N-IYA+2,IZA)+W(0,IYA,N-IZA+2)
              X(0,IYA,N-IZA+2) = CONJG(X(0,N-IYA+2,IZA))
              W(0,IYA,N-IZA+2) = W(0,N-IYA+2,IZA)
           ENDDO
        ENDDO

        DO  IYA=2,N2
           X(0,IYA,1)     = X(0,IYA,1)+CONJG(X(0,N-IYA+2,1))
           W(0,IYA,1)     = W(0,IYA,1)+W(0,N-IYA+2,1)
           X(0,N-IYA+2,1) = CONJG(X(0,IYA,1))
           W(0,N-IYA+2,1) = W(0,IYA,1)
        ENDDO

        DO  IZA=2,N2
           X(0,1,IZA)     = X(0,1,IZA)+CONJG(X(0,1,N-IZA+2))
           W(0,1,IZA)     = W(0,1,IZA)+W(0,1,N-IZA+2)
           X(0,1,N-IZA+2) = CONJG(X(0,1,IZA))
           W(0,1,N-IZA+2) = W(0,1,IZA)
        ENDDO

        END

C       ----------------ASTA ---------------------------------------

        SUBROUTINE  ASTA(X,N,RI,ABA,KLP)

        DIMENSION   X(N,N)
        DOUBLE PRECISION  ABA

C       ESTIMATE AVERAGE OUTSIDE THE CIRCLE
        R  =RI*RI
        NC =N/2+1
        DO   J=1,N
           T=J-NC
           XX=T*T
           DO   I=1,N
              T=I-NC
              IF (XX+T*T.GT.R)    THEN
                 ABA=ABA+DBLE(X(I,J))
                 KLP=KLP+1
              ENDIF
           ENDDO
        ENDDO
        RETURN
        END

C       ------------------- PADD2 -------------------------------

        SUBROUTINE PADD2(PROJ,L,BI,LSD,N)

C       PADS: PROJ OF SIZE: L  INTO: BI  WITH SIZE: N

        DIMENSION  PROJ(L,L),BI(LSD,N)
        DOUBLE     PRECISION QS

        KLP = 0
        R   = L/2
        QS  = 0.0D0

C       ESTIMATE AVERAGE OUTSIDE THE CIRCLE
        CALL ASTA(PROJ,L,R,QS,KLP)
        QS = QS/REAL(KLP)

C       ZEROS ALL OF: BI
c$omp   parallel do private(i,j)
        DO J=1,N
           DO I=1,N
              BI(I,J) = 0.0
           ENDDO
        ENDDO

C       FOR L ODD ADD ONE.  N IS ALWAYS EVEN
        IP = (N-L)/2+MOD(L,2)

c$omp   parallel do private(i,j)
        DO J=1,L
           DO I=1,L
              BI(IP+I,IP+J) = PROJ(I,J) - QS
           ENDDO
        ENDDO

        END


	  SUBROUTINE ERRT(IERRT,PROG,NE)

      INCLUDE 'CMBLOCK.INC'

      CHARACTER (LEN=*)   :: PROG
      CHARACTER (LEN=78)  :: MESG
      CHARACTER (LEN=180) :: CSTRING
      CHARACTER (LEN=1)   :: NUL

      END

C ---------------------------------------------------------------------------

        SUBROUTINE CANG(PHI,THETA,PSI,DOSS,SS,DM)

        REAL              :: PHI,THETA,PSI
        LOGICAL           :: DOSS
        DOUBLE PRECISION  :: CPHI,SPHI,CTHE,STHE,CPSI,SPSI
        REAL              :: DM(9),SS(6)

	DOUBLE PRECISION  :: QUADPI, DGR_TO_RAD
	PARAMETER (QUADPI = 3.141592653589793238462643383279502884197)
	PARAMETER (DGR_TO_RAD = (QUADPI/180))

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

C ---------------------------------------------------------------------------

        SUBROUTINE NRMW2(R,W,N2,N)

        DIMENSION  W(0:N2,N,N)
        COMPLEX    R(0:N2,N,N)

c   parallel do private(i,j,k)
        DO K=1,N
           DO J=1,N
              DO I=0,N2
                IF (W(I,J,K) .GT. 0.1)  THEN
		   R(I,J,K) = R(I,J,K) * (-1)**(I+J+K)/W(I,J,K)
		ELSE
		   R(I,J,K) = (0.0,0.0)
		ENDIF
              ENDDO
           ENDDO
        ENDDO

        INV = -1
        CALL FMRS_3(R,N,N,N,INV)

        END


C       ----------------SYMPLANE0 ---------------------------------------

        SUBROUTINE  SYMPLANE0(X,W,N2,N)

        DIMENSION  W(0:N2,N,N)
        COMPLEX  X(0:N2,N,N)

C       SYMMETRIZE PLANE 0
        DO  IZA=2,N2
           DO  IYA=2,N2
              X(0,IYA,IZA)=X(0,IYA,IZA)+CONJG(X(0,N-IYA+2,N-IZA+2))
              W(0,IYA,IZA)=W(0,IYA,IZA)+W(0,N-IYA+2,N-IZA+2)
              X(0,N-IYA+2,N-IZA+2)=CONJG(X(0,IYA,IZA))
              W(0,N-IYA+2,N-IZA+2)=W(0,IYA,IZA)
              X(0,N-IYA+2,IZA)=X(0,N-IYA+2,IZA)+CONJG(X(0,IYA,N-IZA+2))
              W(0,N-IYA+2,IZA)=W(0,N-IYA+2,IZA)+W(0,IYA,N-IZA+2)
              X(0,IYA,N-IZA+2)=CONJG(X(0,N-IYA+2,IZA))
              W(0,IYA,N-IZA+2)=W(0,N-IYA+2,IZA)
           ENDDO
        ENDDO

        DO  IYA=2,N2
           X(0,IYA,1)=X(0,IYA,1)+CONJG(X(0,N-IYA+2,1))
           W(0,IYA,1)=W(0,IYA,1)+W(0,N-IYA+2,1)
           X(0,N-IYA+2,1)=CONJG(X(0,IYA,1))
           W(0,N-IYA+2,1)=W(0,IYA,1)
        ENDDO

        DO  IZA=2,N2
           X(0,1,IZA)=X(0,1,IZA)+CONJG(X(0,1,N-IZA+2))
           W(0,1,IZA)=W(0,1,IZA)+W(0,1,N-IZA+2)
           X(0,1,N-IZA+2)=CONJG(X(0,1,IZA))
           W(0,1,N-IZA+2)=W(0,1,IZA)
        ENDDO

        END

C       ------------------- WINDKB2A -------------------------------

        SUBROUTINE WINDKB2A(BI,R,L,LSD,N,ALPHA,AAAA,NNN)

        DIMENSION  R(L,L,L), BI(LSD,N,N)

        PARAMETER (QUADPI = 3.14159265358979323846)
        PARAMETER (TWOPI = 2*QUADPI)

        IP = (N-L ) / 2 + MOD(L,2)
        DO K=1,L
           DO J=1,L
              DO I=1,L
                 R(I,J,K) = BI(IP+I,IP+J,IP+K)
              ENDDO
           ENDDO
        ENDDO

        L2   = (L/2)**2
        L2P  = (L/2-1)**2
        IP   = L / 2+1
        XNU  = REAL(NNN) / 2.

        RI   = RIBSL(ALPHA,XNU)

        WKB0 = ALPHA**XNU / RI
        QRT  = (TWOPI*AAAA)**2
        TNR  = 0.0
        M    = 0
        DO K=1,L
           DO J=1,L
              DO I=1,L
                 LR = (K-IP)**2+(J-IP)**2+(I-IP)**2
                 IF (LR.LE.L2) THEN
                 SIGMA = QRT*LR-ALPHA*ALPHA
                  IF (ABS(SIGMA).LT.1.0E-7) THEN
                     WKB=1.0

                  ELSEIF(SIGMA.GT.0.0) THEN
C                    2PI A R > ALPHA
                     ART = SQRT(SIGMA)
                     RI  = RJBSL(ART, XNU)
                     WKB = WKB0*RI/ART**XNU

                  ELSE
C                    2PI A R < ALPHA
                     ART = SQRT(ABS(SIGMA))
                     RI  = RIBSL(ART,XNU)
                     WKB = WKB0*RI/ART**XNU
                  ENDIF

                  R(I,J,K) = R(I,J,K) / ABS(WKB)
                  IF (LR .GE. L2P .AND. LR .LE. L2) THEN
                      TNR = TNR+R(I,J,K)
                      M   = M+1
                  ENDIF
                 ENDIF
              ENDDO
           ENDDO
        ENDDO
        TNR = TNR / REAL(M)
c$omp   parallel do private(i,j,k,lr)
        DO K=1,L
           DO J=1,L
              DO I=1,L
                 LR = (K-IP)**2 + (J-IP)**2 + (I-IP)**2
                 IF (LR .LE. L2) THEN
                    R(I,J,K) = R(I,J,K) - TNR
                 ELSE
                    R(I,J,K) = 0.0
                 ENDIF
              ENDDO
           ENDDO
        ENDDO

        END

C       ------------------- WINDKB2 -------------------------------

        SUBROUTINE WINDKB2(BI,R,L,LSD,N)

        DIMENSION  R(L,L,L),BI(LSD,N,N)
        COMMON  /BESSEL_PARAM/  ALPHA,AAAA,NNN

        PARAMETER (QUADPI = 3.14159265358979323846)
        PARAMETER (TWOPI = 2*QUADPI)

        IP = (N-L)/2+MOD(L,2)
        DO K=1,L
           DO J=1,L
              DO I=1,L
                 R(I,J,K) = BI(IP+I,IP+J,IP+K)
              ENDDO
           ENDDO
        ENDDO

        L2  = (L/2)**2
        L2P = (L/2-1)**2
        IP  = L/2+1
        XNU = REAL(NNN)/2.

        RI = RIBSL(ALPHA,XNU)
C       IF (ABS(RI-RIN).GT.1.E-5)  PRINT  *,'BESSIK'

        WKB0 = ALPHA**XNU/RI
        QRT  = (TWOPI*AAAA)**2
        TNR  = 0.0
        M    = 0
        DO K=1,L
           DO J=1,L
              DO I=1,L
                 LR = (K-IP)**2+(J-IP)**2+(I-IP)**2
                 IF (LR.LE.L2) THEN
                 SIGMA=QRT*LR-ALPHA*ALPHA
                  IF (ABS(SIGMA).LT.1.0E-7)  THEN
                     WKB=1.0
                  ELSEIF(SIGMA.GT.0.0)  THEN
C                    2PI A R > ALPHA
                     ART = SQRT(SIGMA)
                     RI = RJBSL(ART, XNU)
C       if(abs(ri-rin)/rin.gt.1.e-5)  print  *,'bessjy',i,j,k
                     WKB=WKB0*RI/ART**XNU
                  ELSE
C                    2PI A R < ALPHA
                     ART = SQRT(ABS(SIGMA))
                     RI = RIBSL(ART,XNU)
C       if(abs(ri-rin)/rin.gt.1.e-5)  print  *,'bessik',i,j,k,ri,rin
                     WKB=WKB0*RI/ART**XNU
                  ENDIF
                  R(I,J,K) = R(I,J,K)/ABS(WKB)
                  IF (LR.GE.L2P .AND. LR.LE.L2) THEN
                      TNR=TNR+R(I,J,K)
                      M=M+1
                  ENDIF
                 ENDIF
              ENDDO
           ENDDO
        ENDDO
        TNR = TNR/REAL(M)
c$omp   parallel do private(i,j,k,lr)
        DO K=1,L
           DO J=1,L
              DO I=1,L
                 LR=(K-IP)**2+(J-IP)**2+(I-IP)**2
                 IF (LR.LE.L2) THEN
                    R(I,J,K)=R(I,J,K)-TNR
                 ELSE
                    R(I,J,K)=0.0
                 ENDIF
              ENDDO
           ENDDO
        ENDDO

        END


C       ------------------- ONELINE -------------------------------

		SUBROUTINE  ONELINE(J,N,N2,X,W,BI,DM,LN2,FLTB,LTAB,TABI)

        DIMENSION      W(0:N2,N,N)
        COMPLEX        BI(0:N2,N),X(0:N2,N,N),BTQ
        DIMENSION      DM(6)
        REAL      	  TABI(0:LTAB)

C        PARAMETER      (LTAB=4999)
C        COMMON  /TABS/ LN2,FLTB,TABI(0:LTAB)

        IF (J .GE. 0)  THEN
           JP = J+1
        ELSE
           JP = N+J+1
        ENDIF

        DO  I=0,N2
           IF (((I*I+J*J) .LT.  (N*N/4)) .AND..NOT.
     &          (I.EQ. 0  .AND. J.LT.0)) THEN
              XNEW = I * DM(1) + J * DM(4)
              YNEW = I * DM(2) + J * DM(5)
              ZNEW = I * DM(3) + J * DM(6)

              IF (XNEW .LT. 0.0)  THEN
                 XNEW = -XNEW
                 YNEW = -YNEW
                 ZNEW = -ZNEW
                 BTQ  = CONJG(BI(I,JP))
              ELSE
                 BTQ  = BI(I,JP)
              ENDIF

              IXN = IFIX(XNEW+0.5+N) - N
              IYN = IFIX(YNEW+0.5+N) - N
              IZN = IFIX(ZNEW+0.5+N) - N

              IF (IXN .LE. (N2-LN2-1)  .AND.
     &            IYN .GE. (-N2+2+LN2) .AND. IYN .LE. (N2-LN2-1) .AND.
     &            IZN .GE. (-N2+2+LN2) .AND. IZN .LE. (N2-LN2-1)) THEN

                 IF (IXN .GE. 0) THEN
C                   MAKE SURE THAT LOWER LIMIT FOR X DOES NOT GO BELOW 0
                    LB = -MIN0(IXN,LN2)
                    DO LZ=-LN2,LN2
                       IZP = IZN + LZ
                       IF(IZP .GE. 0) THEN
                          IZA = IZP + 1
                       ELSE
                          IZA = N + IZP + 1
                       ENDIF

                       TZ  = TABI(NINT(ABS(ZNEW-IZP) * FLTB))

                       IF (TZ .NE. 0.0)  THEN
                          DO  LY=-LN2,LN2
                             IYP = IYN + LY
                             IF (IYP .GE .0) THEN
                                IYA = IYP + 1
                             ELSE
                                IYA = N + IYP + 1
                             ENDIF

                             TY  = TABI(NINT(ABS(YNEW-IYP) * FLTB)) * TZ
                             IF (TY .NE. 0.0)  THEN
                                DO  IXP=LB+IXN,LN2+IXN

C                                  GET THE WEIGHT
                                   WG=TABI(NINT(ABS(XNEW-IXP)*FLTB))*TY
                                   IF (WG .NE. 0.0) THEN

                                      X(IXP,IYA,IZA) =
     &                                    X(IXP,IYA,IZA) + BTQ * WG
                                      W(IXP,IYA,IZA) =
     &                                    W(IXP,IYA,IZA) + WG
                                   ENDIF
                                ENDDO
                             ENDIF
                          ENDDO
                       ENDIF
                   ENDDO
                ENDIF

C               ADD REFLECTED POINTS
                IF (IXN .LT. LN2) THEN
                   DO  LZ=-LN2,LN2
                      IZP = IZN + LZ
                      IZT =  - IZP + 1
                      IF (IZP .GT. 0)  IZT = N + IZT

                      TZ = TABI(NINT(ABS(ZNEW-IZP) * FLTB))

                      IF (TZ .NE. 0.0)  THEN
                         DO  LY=-LN2,LN2
                            IYP = IYN + LY
                            IYT = -IYP + 1
                            IF (IYP .GT. 0) IYT = IYT + N

                            TY = TABI(NINT(ABS(YNEW-IYP) * FLTB)) * TZ
                            IF (TY .NE. 0.0)  THEN
                               DO  IXP=IXN-LN2,-1

C                                 GET THE WEIGHT
                                  WG = TABI(NINT(ABS(XNEW-IXP)*FLTB))*TY

                                  IF (WG .NE. 0.0)  THEN
                                     X(-IXP,IYT,IZT) =
     &                                   X(-IXP,IYT,IZT) + CONJG(BTQ)*WG
                                     W(-IXP,IYT,IZT) =
     &                                   W(-IXP,IYT,IZT) + WG
                                  ENDIF
                               ENDDO
                            ENDIF
                         ENDDO
                      ENDIF
                   ENDDO
                ENDIF
              ENDIF
           ENDIF
C          END J-I LOOP
        ENDDO

        END
