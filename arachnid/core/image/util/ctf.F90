C**************************************************************************
      REAL FUNCTION EVALCTF_TEST(CS,WL,WGH1,WGH2,DFMID1,DFMID2,ANGAST,
     +			    THETATR,HW,AIN,MD,NXYZ,RMIN2,RMAX2,DAST)
C**************************************************************************
C     This part of the code was made more efficient by
C
C     Robert Sinkovits
C     Department of Chemistry and Biochemistry
C     San Diego Supercomputer Center
C
C**************************************************************************
C
      IMPLICIT NONE
C
      INTEGER L,LL,M,MM,NXYZ(3),ID,IS
      REAL CS,WL,WGH1,WGH2,DFMID1,DFMID2,ANGAST,THETATR,SUM1
      REAL SUM,AIN(*),MD(*),RES2,RMIN2,RMAX2,CTF,CTFV,HW,SUM2,DAST
      real rad2, hangle2, angspt, c1, c2, angdif, ccos, df, chi
      real expv, twopi_wli, ctfv2, dsum, ddif, rpart1, rpart2
      real half_thetatrsq, recip_nxyz1, recip_nxyz2
      real :: twopi=6.2831853071796

cf2py threadsafe
cf2py intent(inout) :: AIN,NXYZ,MD
cf2py intent(in) :: CS,WL,WGH1,WGH2,DFMID1,DFMID2,ANGAST,THETATR,HW,RMIN2,RMAX2,DAS

      twopi_wli  = twopi/wl
      dsum = dfmid1 + dfmid2
      ddif = dfmid1 - dfmid2
      half_thetatrsq = 0.5*thetatr*thetatr
      recip_nxyz1 = 1.0/nxyz(1)
      recip_nxyz2 = 1.0/nxyz(2)

      SUM  = 0.0
      SUM1 = 0.0
      SUM2 = 0.0
      IS   = 0

      DO M=1,NXYZ(2)
         MM=M-1
         IF (MM > NXYZ(2)/2) MM=MM-NXYZ(2)
         rpart2 = real(mm) * recip_nxyz2

         DO L=1,NXYZ(1)/2
            LL=L-1
            rpart1 = real(ll) * recip_nxyz1
            RES2 = rpart1*rpart1 + rpart2*rpart2

            IF (RES2 <= RMAX2 .AND. RES2 > RMIN2) THEN
               RAD2 = LL*LL + MM*MM
               IF (RAD2.NE.0.0) THEN
                  ANGSPT = ATAN2(REAL(MM), REAL(LL))
                  ANGDIF = ANGSPT - ANGAST
                  CCOS   = COS(2.0*ANGDIF)
                  DF     = 0.5*(DSUM + CCOS*DDIF)

                  HANGLE2 = rad2 * half_thetatrsq
                  C1      = twopi_wli*HANGLE2
                  C2      = -C1*CS*HANGLE2
                  CHI     = C1*DF + C2
                  CTFV    = -WGH1*SIN(CHI)-WGH2*COS(CHI)
               ELSE
                  CTFV    = -WGH2
               ENDIF

               ctfv2 = ctfv*ctfv
               ID   = L+NXYZ(1)/2*(M-1)
               IS   = IS + 1

               if(hw == 0.0) then
                  SUM  = SUM  + AIN(ID)*ctfv2
                  SUM1 = SUM1 + ctfv2*ctfv2
                  SUM2 = SUM2 + AIN(ID)*AIN(ID)
               else
                  expv = exp(hw*res2)
                  SUM  = SUM  + AIN(ID)*ctfv2*expv
                  SUM1 = SUM1 + ctfv2*ctfv2
                  SUM2 = SUM2 + AIN(ID)*AIN(ID)*expv*expv
               endif
               MD(ID)=ctfv2*ctfv2

            ENDIF

         enddo
      enddo

      IF (IS.NE.0) THEN
        SUM=SUM/SQRT(SUM1*SUM2)
        IF (DAST.GT.0.0) SUM=SUM-DDIF**2/2.0/DAST**2/IS
      ENDIF
      EVALCTF_TEST=SUM
      RETURN
      END
C**************************************************************************
      REAL FUNCTION EVALCTF(CS,WL,WGH1,WGH2,DFMID1,DFMID2,ANGAST,
     +			    THETATR,HW,AIN,NXYZ,RMIN2,RMAX2,DAST)
C**************************************************************************
C     This part of the code was made more efficient by
C
C     Robert Sinkovits
C     Department of Chemistry and Biochemistry
C     San Diego Supercomputer Center
C
C**************************************************************************
C
      IMPLICIT NONE
C
      INTEGER L,LL,M,MM,NXYZ(3),ID,IS
      REAL CS,WL,WGH1,WGH2,DFMID1,DFMID2,ANGAST,THETATR,SUM1
      REAL SUM,AIN(*),RES2,RMIN2,RMAX2,CTF,CTFV,HW,SUM2,DAST
      real rad2, hangle2, angspt, c1, c2, angdif, ccos, df, chi
      real expv, twopi_wli, ctfv2, dsum, ddif, rpart1, rpart2
      real half_thetatrsq, recip_nxyz1, recip_nxyz2
      real :: twopi=6.2831853071796

cf2py threadsafe
cf2py intent(inout) :: AIN,NXYZ
cf2py intent(in) :: CS,WL,WGH1,WGH2,DFMID1,DFMID2,ANGAST,THETATR,HW,RMIN2,RMAX2,DAS

      twopi_wli  = twopi/wl
      dsum = dfmid1 + dfmid2
      ddif = dfmid1 - dfmid2
      half_thetatrsq = 0.5*thetatr*thetatr
      recip_nxyz1 = 1.0/nxyz(1)
      recip_nxyz2 = 1.0/nxyz(2)

      SUM  = 0.0
      SUM1 = 0.0
      SUM2 = 0.0
      IS   = 0

      DO M=1,NXYZ(2)
         MM=M-1
         IF (MM > NXYZ(2)/2) MM=MM-NXYZ(2)
         rpart2 = real(mm) * recip_nxyz2

         DO L=1,NXYZ(1)/2
            LL=L-1
            rpart1 = real(ll) * recip_nxyz1
            RES2 = rpart1*rpart1 + rpart2*rpart2

            IF (RES2 <= RMAX2 .AND. RES2 > RMIN2) THEN
               RAD2 = LL*LL + MM*MM
               IF (RAD2.NE.0.0) THEN
                  ANGSPT = ATAN2(REAL(MM), REAL(LL))
                  ANGDIF = ANGSPT - ANGAST
                  CCOS   = COS(2.0*ANGDIF)
                  DF     = 0.5*(DSUM + CCOS*DDIF)

                  HANGLE2 = rad2 * half_thetatrsq
                  C1      = twopi_wli*HANGLE2
                  C2      = -C1*CS*HANGLE2
                  CHI     = C1*DF + C2
                  CTFV    = -WGH1*SIN(CHI)-WGH2*COS(CHI)
               ELSE
                  CTFV    = -WGH2
               ENDIF

               ctfv2 = ctfv*ctfv
               ID   = L+NXYZ(1)/2*(M-1)
               IS   = IS + 1

               if(hw == 0.0) then
                  SUM  = SUM  + AIN(ID)*ctfv2
                  SUM1 = SUM1 + ctfv2*ctfv2
                  SUM2 = SUM2 + AIN(ID)*AIN(ID)
               else
                  expv = exp(hw*res2)
                  SUM  = SUM  + AIN(ID)*ctfv2*expv
                  SUM1 = SUM1 + ctfv2*ctfv2
                  SUM2 = SUM2 + AIN(ID)*AIN(ID)*expv*expv
               endif

            ENDIF

         enddo
      enddo

      IF (IS.NE.0) THEN
        SUM=SUM/SQRT(SUM1*SUM2)
        IF (DAST.GT.0.0) SUM=SUM-DDIF**2/2.0/DAST**2/IS
      ENDIF
      EVALCTF=SUM
      RETURN
      END

C**************************************************************************
      SUBROUTINE MSMOOTH(ABOX,NXYZ,NW,BUF)
C**************************************************************************
C       Calculates a smooth background in the power spectrum
C       in ABOX using a box convolution with box size 2NW+1 x 2NW+1.
C       Replaces input with background-subtracted power spectrum.
C**************************************************************************
      IMPLICIT NONE
C
      INTEGER NXYZ(*),NW,I,J,K,L,IX,IY,ID,CNT
      REAL ABOX(*),SUM,BUF(*)

cf2py threadsafe
cf2py intent(inplace) :: ABOX,BUF
cf2py intent(in) :: NXYZ, NW

C
C       loop over X and Y
C
      DO 10 I=1,NXYZ(1)
        DO 10 J=1,NXYZ(2)
          SUM=0.0
          CNT=0
C
C       loop over box to average
C
          DO 20 K=-NW,NW
            DO 20 L=-NW,NW
              IX=I+K
              IY=J+L
C
C       here reset IX to wrap around spectrum
C
              IF (IX.GT.NXYZ(1)) IX=IX-2*NXYZ(1)
              IF (IX.LT.1) THEN
                IX=1-IX
                IY=1-IY
              ENDIF
C
C       here reset IY to wrap around spectrum
C
              IF (IY.GT.NXYZ(2)) IY=IY-NXYZ(2)
              IF (IY.LE.-NXYZ(2)) IY=IY+NXYZ(2)
              IF (IY.LT.1) IY=1-IY
              ID=IX+NXYZ(1)*(IY-1)
C              IF (ID.NE.1) THEN
              IF ((IX.GT.1).AND.(IY.GT.1)) THEN
                SUM=SUM+ABOX(ID)
                CNT=CNT+1
              ENDIF
20        CONTINUE
          SUM=SUM/CNT
          ID=I+NXYZ(1)*(J-1)
          IF (ID.NE.1) THEN
            BUF(ID)=SUM
          ELSE
            BUF(ID)=ABOX(ID)
          ENDIF
10    CONTINUE
C
C       replace input with background-subtracted spectrum
C
      DO 40 I=1,NXYZ(1)*NXYZ(2)
        ABOX(I)=ABOX(I)**2-BUF(I)**2
40    CONTINUE
C
      RETURN
      END
C

C**************************************************************************
      REAL FUNCTION CTF(CS,WL,WGH1,WGH2,DFMID1,DFMID2,ANGAST,
     +                  THETATR,IX,IY)
C**************************************************************************
C
      PARAMETER (TWOPI=6.2831853071796)

cf2py threadsafe
cf2py intent(in) :: CS,WL,WGH1,WGH2,DFMID1,DFMID2,ANGAST, THETATR,IX,IY
C
      RAD=IX**2+IY**2
      IF (RAD.NE.0.0) THEN
        RAD=SQRT(RAD)
        ANGLE=RAD*THETATR
        ANGSPT=ATAN2(REAL(IY),REAL(IX))
        C1=TWOPI*ANGLE*ANGLE/(2.0*WL)
        C2=-C1*CS*ANGLE*ANGLE/2.0
        ANGDIF=ANGSPT-ANGAST
        CCOS=COS(2.0*ANGDIF)
        DF=0.5*(DFMID1+DFMID2+CCOS*(DFMID1-DFMID2))
        CHI=C1*DF+C2
        CTF=-WGH1*SIN(CHI)-WGH2*COS(CHI)
      ELSE
        CTF=-WGH2
      ENDIF
C
      RETURN
      END
