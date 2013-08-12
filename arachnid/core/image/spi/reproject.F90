

		SUBROUTINE SETUP_REPROJECT_3Q(VOL, RADIUS, NX, NY, NZ, NN)

		REAL          		   		:: VOL(NX,NY,NZ)
		INTEGER						:: RADIUS, NX, NY, NZ, NN

cf2py threadsafe
cf2py intent(inplace) :: VOL
cf2py intent(in) :: NX,NY,NZ,RADIUS
cf2py intent(hide) :: NX,NY,NZ
cf2py intent(out) :: NN

		IF (1 .EQ. 0) THEN ! avoid warning
		VOL(1,1,1)=0
		ENDIF
        NN   = 1
c        MD   =
		LDPX = NX  /2+1
        LDPY = NY  /2+1
        LDPZ = NZ/2+1
		CALL PREPCUB(NX,NY,NZ,NN,IDUM,RADIUS,.FALSE.,LDPX,LDPY,LDPZ)


		END


C ---------------------------------------------------------------------------
		SUBROUTINE REPROJECT_3Q(VOL,PRJ,VI,PSI,THE,PHI,RI,NX,NY,NZ,NN)

		REAL          		   		:: VOL(NX,NY,NZ)
		REAL                  		:: PRJ(NX,NY)
		INTEGER                  	:: VI(5,NN)
		INTEGER						:: NX,NY,NZ,NN
		REAL						:: PSI,THE,PHI,RI

cf2py threadsafe
cf2py intent(inplace) :: VOL, PRJ, VI
cf2py intent(in) :: NX,NY,NZ,PSI,THE,PHI,RI,NN
cf2py intent(hide) :: NX,NY,NZ,NN

		 LDPX = NX  /2+1
         LDPY = NY  /2+1
         LDPZ = NZ/2+1
         NVOX = NX*NY*NZ

		CALL WPRO_N(PRJ,
     &               NX,NY,NZ,VOL,
     &               NVOX,VI,NN,
     &               PHI,THE,PSI,
     &               RI,LDPX,LDPY,LDPZ)



		END

C ---------------------------------------------------------------------------

		SUBROUTINE REPROJECT_3Q_OMP(VOL,PRJ,ANG,RI,NANG,NX,NY,NZ)

		REAL          		   		:: VOL(NX,NY,NZ)
		REAL                  		:: PRJ(NX,NY,NANG)
		REAL                  		:: ANG(3,NANG)
		INTEGER, ALLOCATABLE        :: VI(:,:)
		INTEGER						:: NX,NY,NZ,NANG
		REAL						:: RI

cf2py threadsafe
cf2py intent(inplace) :: VOL, PRJ, ANG
cf2py intent(in) :: NX,NY,NZ,PSI,THE,PHI,RI,NANG
cf2py intent(hide) :: NX,NY,NZ,NANG

         NN   = 1
c         MD   = .FALSE.
		 LDPX = NX  /2+1
         LDPY = NY  /2+1
         LDPZ = NZ/2+1
         NVOX = NX*NY*NZ
		 CALL PREPCUB(NX,NY,NZ,NN,IDUM,RI,.FALSE.,LDPX,LDPY,LDPZ)
         ALLOCATE(VI(5,NN),STAT=IRTFLG)
		 CALL PREPCUB(NX,NY,NZ,NN,VI,RI,.TRUE.,LDPX,LDPY,LDPZ)

c$omp          parallel do private(i)
               DO I=1,NANG
                  CALL WPRO_N(PRJ(1,1,I),
     &               NX,NY,NZ,VOL,
     &               NVOX,VI,NN,
     &               ANG(3,I),ANG(2,I),ANG(1,I),
     &               RI, LDPX,LDPY,LDPZ)
               ENDDO

		END


C ---------------------------------------------------------------------------

		SUBROUTINE PROJECT_POLAR(PRJ,OUT,NX,NY,NXP,NYP,MR,NR)

		REAL                  		:: PRJ(NX,NY)
		REAL                  		:: OUT(NXP,NYP)
		INTEGER						:: NX,NY,MR,NR,NXP,NYP

cf2py threadsafe
cf2py intent(inplace) :: PRJ,OUT
cf2py intent(in) :: NX,NY,MR,NR
cf2py intent(hide) :: NX,NY,NXP,NYP

		IXC = NX/2+1
		IYC = NY/2+1

		DO  J=MR,NR
		  DO I=1,NXP
		     FI     = (I-1) * DFI
		     XS     = COS(FI) * J
		     YS     = SIN(FI) * J
		     OUT(I,J-MR+1) = QUADRI(XS+IXC,YS+IYC,NX,NY,X)
     &        * SQRT(REAL(J))
		  ENDDO
		ENDDO

		END

