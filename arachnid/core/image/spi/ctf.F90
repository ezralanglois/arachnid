

! ---------------------------------------------------------------------------
		SUBROUTINE FFT2_IMAGE(IMG, NSAM3, NROW, NSAM)
		REAL 						:: IMG(NSAM3,NROW)
!f2py threadsafe
!f2py intent(inout) :: IMG
!f2py intent(in) :: NSAM, NROW, NSAM3
!f2py intent(hide) :: NSAM3, NROW

		INV=1
		CALL FMRS_2(IMG,NSAM,NROW,INV)

		END

! ---------------------------------------------------------------------------
		SUBROUTINE CORRECT_IMAGE_FOURIER(IMG, B, NSAM3, NROW, NSAM2, NSAM)

		COMPLEX 					:: B(NSAM2,NROW)
		REAL 						:: IMG(NSAM3,NROW)
!f2py threadsafe
!f2py intent(inout) :: IMG, B
!f2py intent(in) :: NSAM, NROW, NSAM2,NSAM3
!f2py intent(hide) :: NSAM3, NROW, NSAM2

		!print *, 'nsam3=', NSAM3
		!CALL FLUSH()
		INV=1
		CALL FMRS_2(IMG,NSAM,NROW,INV)


        CALL MULT_CTF(IMG, B, NSAM2, NROW)



		END

! ---------------------------------------------------------------------------
		SUBROUTINE CORRECT_IMAGE(IMG, B, NSAM3, NROW, NSAM2, NSAM)

		COMPLEX 					:: B(NSAM2,NROW)
		REAL 						:: IMG(NSAM3,NROW)
!f2py threadsafe
!f2py intent(inout) :: IMG, B
!f2py intent(in) :: NSAM, NROW, NSAM2,NSAM3
!f2py intent(hide) :: NSAM3, NROW, NSAM2

		!print *, 'nsam3=', NSAM3
		!CALL FLUSH()
		INV=1
		CALL FMRS_2(IMG,NSAM,NROW,INV)


        CALL MULT_CTF(IMG, B, NSAM2, NROW)


		INV = -1
		CALL FMRS_2(IMG, NSAM, NROW, INV)



		END

! ---------------------------------------------------------------------------

		SUBROUTINE MULT_CTF(IMG, B, NSAM, NROW)

		COMPLEX 					:: B(NSAM,NROW)
		COMPLEX 				 	:: IMG(NSAM,NROW)

           DO J=1,NROW
              DO I=1,NSAM
                 IMG(I,J) = IMG(I,J) * B(I,J)
              ENDDO
           ENDDO

		END





! ---------------------------------------------------------------------------
		SUBROUTINE TRANSFER_FUNCTION_PHASE_FLIP_2D(B,ISAM,NDUM,NSAM,CS,DZZ, &
		KM,LAMBDA,Q,DS,DZA,AZZ,WGH,ENV,SIGN)

		COMPLEX 					:: B(ISAM,NDUM)
		REAL						:: CS,DZZ,LAMBDA,Q,DS,DZA,AZZ,WGH,ENV
        INTEGER			  			:: NSAM,NDUM,IFORM,SIGN
        LOGICAL       				WANT_CT

!f2py threadsafe
!f2py intent(inout) :: B
!f2py intent(in) :: ISAM,NSAM,NDUM,IFORM,SIGN,CS,DZZ,LAMBDA,Q,DS,DZA,AZZ,WGH,ENV
!f2py intent(hide) :: ISAM,NDUM
		WANT_CT=.FALSE.
        IF (ENV .NE. 0.0) THEN
           ENV = 1./ENV**2
           WANT_CT=.TRUE.
        ENDIF

         IF (MOD(NSAM,2) .EQ. 0)  THEN
            IFORM = -12
            LSM   = NSAM+2
         ELSE
            IFORM = -11
            LSM   = NSAM+1
         ENDIF

         IXC    = NSAM/2+1
         IF (NDUM.EQ.0)  THEN
            NROW   = NSAM
            IYC    = IXC
         ELSE
            NROW   = NDUM
            IYC    = NROW/2+1
         ENDIF

!        SC=KM/FLOAT(NSAM/2)
         SCX = 2.0 / NSAM
         SCY = 2.0 / NROW

         IE =0
!        IE=0 SELECTS TRANSFER FUNCTION OPTION IN SUBROUTINE TFD
         WGH = ATAN(WGH/(1.0-WGH))
         CS  = CS*1.E7
         DO  K=1,NROW
            KY = K-1
            IF (K.GT.IYC) KY = KY-NROW
            DO  I=1,LSM,2
               KX = (I-1)/2

!              Changed AK to handle rectangular images
!              AK = SQRT(FLOAT(KY)**2 + FLOAT(KX)**2)*SC
               AK = KM * SQRT((KX*SCX)**2 + (KY*SCY)**2)

!              AZ = QUADPI/2.
	       	   IF (KX.NE.0) THEN
                  AZ = ATAN2(FLOAT(KY),FLOAT(KX)) + QUADPI/2.
	       	   ELSE
	              AZ =  QUADPI/2.
               ENDIF



               AZR = AZZ*(QUADPI/180.)
               DZZ = DZ+DZA/2*SIN(2*(AZ-AZR))

               CALL TFD(TF,CS,DZZ,LAMBDA,Q,DS,IE,AK,WGH,ENV)

               !IF (KX.GT.ISAM) THEN
               !print *, "out of bounds: ", KX+1, " ", ISAM
               !CALL EXIT(1)
               !ENDIF

               IF (WANT_CT) THEN
                  IF (TF .GE. 0.0) THEN
                     B(K, KX+1) = CMPLX(1.0, 0.0) * SIGN
                  ELSE
                     B(K, KX+1) = CMPLX(-1.0, 0.0) * SIGN
                  ENDIF
               ELSE
                  B(K, KX+1) = CMPLX(TF*SIGN, 0.0)
               ENDIF
            ENDDO
         ENDDO

		END

