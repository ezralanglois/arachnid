C++*********************************************************************
C--*********************************************************************
		SUBROUTINE FINALIZE(X, NX, N2, W)

		REAL          			   :: W(0:NX, N2,N2)
		COMPLEX                     :: X(0:NX, N2,N2)

		EXTERNAL  SYMPLANE0
		EXTERNAL  NRMW2

cf2py intent(inout) :: X, W
cf2py intent(in) :: NX, N2
cf2py intent(hide) :: NX, N2

		LSD     = N2 + 2 - MOD(N2,2)
C       SYMMETRIZE PLANE 0
        CALL SYMPLANE0(X,W,NX,N2)

C       WEIGHT AND FOURIER TRANSFORM
        CALL NRMW2(X,W,NX,N2)


		END

C ---------------------------------------------------------------------------

		SUBROUTINE WINDOW(X, NX, N2,ALPHA,AAAA,NNN)

		REAL          			   :: W(0:NX, N2,N2)
		COMPLEX                     :: X(0:NX, N2,N2)

cf2py intent(in) :: NX,N2,ALPHA,AAAA,NN
cf2py intent(inout) :: X

		LSD     = N2 + 2 - MOD(N2,2)
C       WINDOW
        CALL WINDKB2A(X,X,NX,LSD,N2,ALPHA,AAAA,NNN)

		END

C        CALL FILLBESSIL(N2,LN2,LTAB,FLTB,TABI,ALPHA,AAAA,NNN)
C  		 COMMON  /TABS/ LN2,FLTB,TABI(0:LTAB)
		SUBROUTINE BACKPROJECT(NX, DMS, BI, N2, X, W, LN2, FLTB, LTAB, TABI)

		REAL                        :: SM(3,3,MAXSYM)
		REAL                        :: DMS(3,3)
C          REAL                        :: DM(3,3)
		COMPLEX        			   :: BI(0:NX,N2)
		REAL          			   :: W(0:NX, N2,N2)
		COMPLEX                     :: X(0:NX, N2,N2)
		REAL      	  			   :: TABI(0:LTAB)

cf2py intent(in) :: NX, N2, LN2, FLTB, LTAB
cf2py intent(inout) :: BI, DMS, X, W, TABI

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

#ifdef _OPENMP
	     DO JT=1,LN1
c$omp           parallel do private(j)
                DO J=-NX+JT,NX,LN1
                   CALL ONELINE(J,N2,NX,X,W,BI,DMS, LN2, FLTB, LTAB, TABI)
                ENDDO
             ENDDO
#else
             DO J=-NX+1,NX
               CALL ONELINE(J,N2,NX,X,W,BI,DMS, LN2, FLTB, LTAB, TABI)
             ENDDO
#endif
           ENDDO

		END
C ---------------------------------------------------------------------------
C       J  - voxel offset
C       N  - padded projection size
C       NX - actual projection size
C		X - volume
C		W - weight
C		BI - fft of padded image
C		DM - rotation matrix
		SUBROUTINE  ONELINE(J,N,N2,X,W,BI,DM,LN2,FLTB,LTAB,TABI)

        DIMENSION      W(0:N2,N,N)
        COMPLEX        BI(0:N2,N),X(0:N2,N,N),BTQ
        DIMENSION      DM(6)
        REAL      	  :: TABI(0:LTAB)

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

C ---------------------------------------------------------------------------
C ---------------------------------------------------------------------------
       SUBROUTINE FILLBESSIL(N2,LN2,LTAB,FLTB,TABI,ALPHA,AAAA,NNN)

C        !! COMMON  /TABS/ LN2,FLTB,TABI(0:LTAB)

        REAL      :: TABI(0:LTAB)

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

C ---------------------------------------------------------------------------
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


C ---------------------------------------------------------------------------
        SUBROUTINE NRMW2(R,W,N2,N)

        DIMENSION  W(0:N2,N,N)
        COMPLEX    R(0:N2,N,N)

c$omp   parallel do private(i,j,k)
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

C        INV = -1
C        CALL FMRS_3(R,N,N,N,INV)

        END





CCALL PADD2(PROJ,NX,BI,LSD,N2)
		SUBROUTINE PADD2(PROJ,L,BI,N)

C       PADS: PROJ OF SIZE: L  INTO: BI  WITH SIZE: N

        DIMENSION  PROJ(L,L),BI(LSD,N)
        DOUBLE     PRECISION QS

cf2py intent(in) ::  L, N
cf2py intent(inout) :: PROJ, BI

		LSD     = N + 2 - MOD(N,2)
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
