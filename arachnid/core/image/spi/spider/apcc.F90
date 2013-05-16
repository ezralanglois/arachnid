
C++*********************************************************************
C
C APCC.F         NEW FROM APSHIFT FOR SPEEDUP       FEB 08 ARDEAN LEITH
C                CCRS INOUT                         APR 09 ARDEAN LEITH
C                MAX RANGE     BUG                  AUG 11 ARDEAN LEITH
C                PEAKV SCALING BUG                  AUG 11 ARDEAN LEITH
C **********************************************************************
C=*                                                                    *
C=* This file is part of:   SPIDER - Modular Image Processing System.  *
C=* SPIDER System Authors:  Joachim Frank & ArDean Leith               *
C=* Copyright 1985-2011  Health Research Inc.,                         *
C=* Riverview Center, 150 Broadway, Suite 560, Menands, NY 12204.      *
C=* Email: spider@wadsworth.org                                        *
C=*                                                                    *
C=* SPIDER is free software; you can redistribute it and/or            *
C=* modify it under the terms of the GNU General Public License as     *
C=* published by the Free Software Foundation; either version 2 of the *
C=* License, or (at your option) any later version.                    *
C=*                                                                    *
C=* SPIDER is distributed in the hope that it will be useful,          *
C=* but WITHOUT ANY WARRANTY; without even the implied warranty of     *
C=* merchantability or fitness for a particular purpose.  See the GNU  *
C=* General Public License for more details.                           *
C=* You should have received a copy of the GNU General Public License  *
C=* along with this program. If not, see <http://www.gnu.org/licenses> *
C=*                                                                    *
C **********************************************************************
C
C  APCC(LSC,NSAM,NROW,NSLICE, BUFI,BUFR,  
C       SKIP_PEAK,NORMIT,SPIDER_SIGN,
C       ISHRANGEX,ISHRANGEY,ISHRANGEZ,
C       XSHNEW,YSHNEW,ZSHNEW,PEAKV,IRTFLG)                                   
C
C  PURPOSE: CROSS-CORRELATE WITH REFERENCE IMAGE, OPTIONALLY FIND 
C           CC PEAK LOCATION.
C
C  PARAMETERS: LSC           PADDED INPUT X DIMENSION          SENT
C              NSAM...       INPUT IMAGE DIMENSIONS            SENT
C              BUFI          SAMPLE IMAGE --> PEAK IMAGE       SENT/RET.  
C              BUFR          REFERENCE IMAGE                   SENT/RET.
C                            ALTERED IF FFT IS DONE
C              DO_FFT_R      FFT THE BUFR CONTENTS             SENT
C              SKIP_PEAK     SKIP PEAK LOCATION FLAG           SENT
C              NORMIT        NORM LOCATION 0 FLAG              SENT
C              SPIDER_SIGN   USE SPIDERS FFT SIGN CONVENTION   SENT
C              ISHRANGEX..   POSSIBLE IMAGE SHIFT              SENT
C              XSHNEW..      SUB PIXEL IMAGE SHIFT             RET.
C              PEAKV         PEAK HEIGHT                       RET.
C                            (ONLY RETURNED FOR 2D & 3D!!)
C              IRTFLG        ERROR FLAG                        RET.  
C
C  NOTE:   BUFI and BUFR SHOULD CONTAIN IMAGES INSIDE A ROW 
C          LENGTH: NSAM+2-MOD(NSAM,2)).   FOR BEST ACCURACY IMAGES
C          SHOULD BE PADDED TO 2X ORIGINAL SIZE, THEN ROW LENGTH WILL
C          BE: 2*NSAM+2 (WHERE NSAM IS ORIGINAL IMAGE ROW LENGTH)
C
C--*********************************************************************

        SUBROUTINE APCC_NEW(LSC, NSAM,NROW,NSLICE, BUFI,BUFR,
     &                  DO_FFT_I,DO_FFT_R,
     &                  SKIP_PEAK,NORMIT,SPIDER_SIGN,
     &                  ISHRANGEX,ISHRANGEY,ISHRANGEZ,
     &                  XSHNEW,YSHNEW,ZSHNEW,PEAKV,IRTFLG)

        IMPLICIT NONE

        INCLUDE 'CMBLOCK.INC'

        INTEGER, INTENT(IN)     :: LSC,NSAM,NROW,NSLICE
        REAL                    :: BUFI(LSC,NROW,NSLICE)
        REAL                    :: BUFR(LSC,NROW,NSLICE)

        LOGICAL, INTENT(IN)     :: DO_FFT_I,DO_FFT_R
        LOGICAL, INTENT(IN)     :: SKIP_PEAK,NORMIT,SPIDER_SIGN
        INTEGER, INTENT(IN)     :: ISHRANGEX,ISHRANGEY,ISHRANGEZ
        REAL,    INTENT(OUT)    :: XSHNEW,YSHNEW,ZSHNEW,PEAKV
        INTEGER, INTENT(OUT)    :: IRTFLG

        INTEGER                 :: INV,IGOX,IENDX,IGOY,IENDY
        INTEGER                 :: IGOZ,IENDZ,IX,IY,IZ
        INTEGER                 :: IXSH,IYSH,IZSH,NE,NT2,I,J
        REAL                    :: QMAX,RX,RY,RZ

        INTEGER * 8             :: IPLAN = 0     !STRUCTURE POINTER 
        INTEGER                 :: ILOCS(3)
        LOGICAL                 :: SPIDER_SCALE = .FALSE.

	REAL                    :: Z(-1:1,-1:1)
        REAL                    :: FNPIX_INV 
        REAL                    :: PEAKPAR

        IRTFLG       = 0

C       CROSS CORRELATION --------------------------------------- CC

        IF (DO_FFT_I) THEN
           INV = +1
           CALL FMRS(BUFI, NSAM,NROW,NSLICE, IPLAN,
     &               SPIDER_SIGN,SPIDER_SCALE, INV,IRTFLG)
           IF (IRTFLG .NE. 0) THEN
              CALL ERRT(101,'APCC FFT ERROR ON BUFI',NE)
              RETURN
           ENDIF
        ENDIF

        IF (DO_FFT_R) THEN
           INV = +1
           CALL FMRS(BUFR, NSAM,NROW,NSLICE, IPLAN,
     &               SPIDER_SIGN, SPIDER_SCALE, INV,IRTFLG)
           IF (IRTFLG .NE. 0) THEN
              CALL ERRT(101,'APCC FFT ERROR ON BUFR',NE)
              RETURN
           ENDIF
        ENDIF

        !!call chkminloc('apcc exp',bufi,lsc*nrow)
        !call chkmaxloc('apcc exp',bufi,lsc*nrow)
        !call chkminloc('apcc ref',bufr,lsc*nrow)
        !call chkmaxloc('apcc ref',bufr,lsc*nrow)

        IF (NORMIT) THEN          ! WARNING: BUFI IS RETURNED!!
           BUFI(1,1,1) = 0.0
           BUFI(2,1,1) = 0.0
        ENDIF

        CALL CCRS(BUFI,BUFR, LSC,NSAM,NROW,NSLICE, 
     &            SPIDER_SIGN,SPIDER_SCALE, IRTFLG)
        IF (IRTFLG .NE. 0) THEN
           CALL ERRT(101,'CCRS CROSS CORRELATION ERROR',NE)
           RETURN
        ENDIF

        IF (SKIP_PEAK) RETURN    ! DO NOT WANT TO FIND PEAK LOCATION


C       PEAK SEARCH ON THE WINDOWED CC IMAGE -------------------- PEAK

C       WINDOW SIZE & WINDOW CORNER
        IGOX   = NSAM/2   - ISHRANGEX +1 
        IENDX  = IGOX + 2 * ISHRANGEX
        
        IGOY   = NROW/2   - ISHRANGEY +1  
        IENDY  = IGOY + 2 * ISHRANGEY 
        
        IGOZ   = NSLICE/2 - ISHRANGEZ +1  
        IENDZ  = IGOZ + 2 * ISHRANGEZ 

C       PEAK SEARCH WITHIN SEARCH RANGE WINDOW
        ILOCS = MAXLOC(BUFI(IGOX:IENDX, IGOY:IENDY, IGOZ:IENDZ))

        IX     = IGOX + ILOCS(1) - 1
        IY     = IGOY + ILOCS(2) - 1
        IZ     = IGOZ + ILOCS(3) - 1

        PEAKV  = BUFI(IX,IY,IZ)

C       FFT FASTER IF SCALING SKIPPED THERE, SO MUST REVALUE OUTPUT
        FNPIX_INV = 1.0 / FLOAT(NSAM * NROW * NSLICE) 
        PEAKV     = PEAKV * FNPIX_INV 

        IXSH   = -(ILOCS(1) - ISHRANGEX - 1)  
        IYSH   = -(ILOCS(2) - ISHRANGEY - 1)  
        IZSH   = -(ILOCS(3) - ISHRANGEZ - 1)  

        XSHNEW = IXSH    ! SET HERE IN CASE OF BOUNDARY RETURN
        YSHNEW = IYSH
        ZSHNEW = IZSH

        !write(6,*) ' Ishrangex         :',ishrangex
        !write(6,*) ' Ix,iy         :',ix,iy,iz,peakv!!,BUFI(IX,IY,IZ)
        !write(6,*) ' Integer shifts:',ixsh,iysh
        !write(6,*) ' ilocs:             ',ilocs

        IF (ISHRANGEZ .GT. 0) THEN   !-------------------------- VOLUME
C          DO NOT INTERPOLATE IF THE MAX IS ON THE EDGE

           IF (IX .LE. 1 .OR. IX .GE. NSAM .OR.
     &         IY .LE. 1 .OR. IY .GE. NROW .OR.
     &         IZ .LE. 1 .OR. IZ .GE. NSLICE) RETURN

           IF (ABS(IXSH) .GE. ISHRANGEX .OR. 
     &         ABS(IYSH) .GE. ISHRANGEY .OR.
     &         ABS(IZSH) .GE. ISHRANGEZ) RETURN

C          BINARY INTERPOLATION TO FIND SUB-PIXEL PEAK
           QMAX = BUFI(IX,IY,IZ)     ! CENTRAL ELEMENT

C          X, Y & Z SUB-PIXEL SHIFT
           RX = 0
           IF (ISHRANGEX .GT. 0) THEN 
             CALL PKSR3_SUB(QMAX,BUFI(IX-1,IY,IZ),BUFI(IX+1,IY,IZ),1,RX)
           ENDIF
           RY = 0
           IF (ISHRANGEY .GT. 0) THEN 
             CALL PKSR3_SUB(QMAX,BUFI(IX,IY-1,IZ),BUFI(IX,IY+1,IZ),1,RY)
           ENDIF
           CALL PKSR3_SUB  (QMAX,BUFI(IX,IY,IZ-1),BUFI(IX,IY,IZ+1),1,RZ)
          
           XSHNEW  = IXSH - RX 
           YSHNEW  = IYSH - RY
           ZSHNEW  = IZSH - RZ

        ELSEIF (ISHRANGEX .GT. 0 .AND. ISHRANGEY .GT. 0)THEN !--- IMAGE

C          DO NOT INTERPOLATE IF THE MAX IS ON THE EDGE
           IF (IX .LE. 1 .OR. IX .GE. NSAM .OR.
     &         IY .LE. 1 .OR. IY .GE. NROW) RETURN

           IF (ABS(IXSH) .GE. ISHRANGEX .OR. 
     &         ABS(IYSH) .GE. ISHRANGEY) RETURN

           NT2 = 1
           IF (NSLICE .GT. 1) NT2 = 2
           DO J=-1,1
              DO I=-1,1
                 Z(I,J) = BUFI(IX+I, IY+J, NT2)  ! 2 for nslice is 2x
              ENDDO
           ENDDO

C          2D PARABOLIC FIT TO 3X3 NEIGHBORHOOD OF PEAK
           CALL PARABL(Z, RX,RY, PEAKPAR)

           XSHNEW = IXSH - RX
           YSHNEW = IYSH - RY
           ZSHNEW = 0.0

           !write(6,*) ' '
           !write(6,*) ' 2d buf,peak: ',bufi(ix,iy,iz),peakv
           !write(6,*) ' Shifts:      ',ixsh,iysh, xshnew,yshnew
           !write(6,*) ' Shifts:      ',ixsh,iysh, rx,ry
           !write(6,"('  z:',1p3g12.4)") z 

        ELSEIF (ISHRANGEX .GT. 0) THEN ! -------------------------- ROW

           IF (IX .LE. 1 .OR. IX .GE. NSAM) RETURN

           IF (ABS(IXSH) .GE. ISHRANGEX) RETURN

C          BINARY INTERPOLATION TO FIND SUB-PIXEL PEAK
           QMAX = BUFI(IX,IY,IZ)         ! CENTRAL ELEMENT
           CALL PKSR3_SUB(QMAX,BUFI(IX-1,IY,IZ),BUFI(IX+1,IY,IZ),1,RX)

           XSHNEW = IXSH - RX 
           YSHNEW = 0.0
           ZSHNEW = 0.0

        ELSEIF (ISHRANGEY .GT. 0) THEN ! ----------------------- COLUMN

           IF (IY .LE. 1 .OR. IY .GE. NROW) RETURN

           IF (ABS(IYSH) .GE. ISHRANGEY)    RETURN

C          BINARY INTERPOLATION TO FIND SUB-PIXEL PEAK
           QMAX = BUFI(IX,IY,IZ)         ! CENTRAL ELEMENT
           CALL PKSR3_SUB(QMAX,BUFI(IX,IY-1,IZ),BUFI(IX,IY+1,IZ),1,RY)

           XSHNEW = 0.0 
           YSHNEW = IYSH - RY
           ZSHNEW = 0.0

        ENDIF
        
C       NORMALIZATION DONE IN CALLER IF DESIRED

        IRTFLG = 0

        END





        SUBROUTINE APCC(LSC, NSAM,NROW,NSLICE, BUFI,BUFR,
     &                  SKIP_PEAK,NORMIT,SPIDER_SIGN,
     &                  ISHRANGEX,ISHRANGEY,ISHRANGEZ,
     &                  XSHNEW,YSHNEW,ZSHNEW,PEAKV,IRTFLG)

        USE TYPE_KINDS
        IMPLICIT NONE
        INTEGER(KIND=I_8)       :: IPLAN = 0     !STRUCTURE POINTER 

        INCLUDE 'CMBLOCK.INC'

        INTEGER, INTENT(IN)     :: LSC,NSAM,NROW,NSLICE
        REAL                    :: BUFI(LSC,NROW,NSLICE)
        REAL                    :: BUFR(LSC,NROW,NSLICE)

        LOGICAL, INTENT(IN)     :: SKIP_PEAK,NORMIT,SPIDER_SIGN
        INTEGER, INTENT(IN)     :: ISHRANGEX,ISHRANGEY,ISHRANGEZ
        REAL,    INTENT(OUT)    :: XSHNEW,YSHNEW,ZSHNEW,PEAKV
        INTEGER, INTENT(OUT)    :: IRTFLG

        INTEGER                 :: INV,IGOX,IENDX,IGOY,IENDY
        INTEGER                 :: IGOZ,IENDZ,IX,IY,IZ
        INTEGER                 :: IXSH,IYSH,IZSH,NE,NT2,I,J
        REAL                    :: QMAX,RX,RY,RZ

        INTEGER                 :: ILOCS(3)
        LOGICAL                 :: SPIDER_SCALE = .FALSE.

	REAL                    :: Z(-1:1,-1:1)
        REAL                    :: FNPIX_INV,PEAKPAR 

        IRTFLG       = 0

C       CROSS CORRELATION --------------------------------------- CC

        INV = +1
        CALL FMRS(BUFI, NSAM,NROW,NSLICE, IPLAN,
     &            SPIDER_SIGN,SPIDER_SCALE, INV,IRTFLG)
       IF (IRTFLG .NE. 0) THEN
           CALL ERRT(101,'APCC FFT ERROR ON BUFI',NE)
           RETURN
        ENDIF

        INV = +1
        CALL FMRS(BUFR, NSAM,NROW,NSLICE, IPLAN,
     &            SPIDER_SIGN, SPIDER_SCALE, INV,IRTFLG)
        IF (IRTFLG .NE. 0) THEN
           CALL ERRT(101,'APCC FFT ERROR ON BUFR',NE)
           RETURN
        ENDIF

        IF (NORMIT) THEN          ! WARNING: BUFI IS RETURNED!!
           BUFI(1,1,1) = 0.0
           BUFI(2,1,1) = 0.0
        ENDIF

        CALL CCRS(BUFI,BUFR, LSC,NSAM,NROW,NSLICE, 
     &            SPIDER_SIGN,SPIDER_SCALE, IRTFLG)
        IF (IRTFLG .NE. 0) THEN
           CALL ERRT(101,'CCRS CROSS CORRELATION ERROR',NE)
           RETURN
        ENDIF

        IF (SKIP_PEAK) RETURN    ! DO NOT WANT TO FIND PEAK LOCATION


C       PEAK SEARCH ON THE WINDOWED CC IMAGE -------------------- PEAK

C       WINDOW SIZE & WINDOW CORNER
        IGOX   = NSAM/2   - ISHRANGEX +1 
        IENDX  = IGOX + 2 * ISHRANGEX
        
        IGOY   = NROW/2   - ISHRANGEY +1  
        IENDY  = IGOY + 2 * ISHRANGEY 
        
        IGOZ   = NSLICE/2 - ISHRANGEZ +1  
        IENDZ  = IGOZ + 2 * ISHRANGEZ 

C       PEAK SEARCH WITHIN SEARCH RANGE WINDOW
        ILOCS = MAXLOC(BUFI(IGOX:IENDX, IGOY:IENDY, IGOZ:IENDZ))

        IX     = IGOX + ILOCS(1) - 1
        IY     = IGOY + ILOCS(2) - 1
        IZ     = IGOZ + ILOCS(3) - 1

        PEAKV  = BUFI(IX,IY,IZ)

C       FFT FASTER IF SCALING SKIPPED THERE, SO MUST REVALUE OUTPUT
        FNPIX_INV = 1.0 / FLOAT(NSAM * NROW * NSLICE) 
        PEAKV     = PEAKV * FNPIX_INV 

        IXSH   = -(ILOCS(1) - ISHRANGEX - 1)  
        IYSH   = -(ILOCS(2) - ISHRANGEY - 1)  
        IZSH   = -(ILOCS(3) - ISHRANGEZ - 1)  

        XSHNEW = IXSH    ! SET HERE IN CASE OF BOUNDARY RETURN
        YSHNEW = IYSH
        ZSHNEW = IZSH

        !write(6,*) ' Ishrangex         :',ishrangex
        !write(6,*) ' Ix,iy         :',ix,iy,iz,peakv!!,BUFI(IX,IY,IZ)
        !write(6,*) ' Integer shifts:',ixsh,iysh
        !write(6,*) ' ilocs:             ',ilocs

        IF (ISHRANGEZ .GT. 0) THEN   !-------------------------- VOLUME
C          DO NOT INTERPOLATE IF THE MAX IS ON THE EDGE

           IF (IX .LE. 1 .OR. IX .GE. NSAM .OR.
     &         IY .LE. 1 .OR. IY .GE. NROW .OR.
     &         IZ .LE. 1 .OR. IZ .GE. NSLICE) RETURN

           IF (ABS(IXSH) .GE. ISHRANGEX .OR. 
     &         ABS(IYSH) .GE. ISHRANGEY .OR.
     &         ABS(IZSH) .GE. ISHRANGEZ) RETURN

C          BINARY INTERPOLATION TO FIND SUB-PIXEL PEAK
           QMAX = BUFI(IX,IY,IZ)     ! CENTRAL ELEMENT

C          X, Y & Z SUB-PIXEL SHIFT
           RX = 0
           IF (ISHRANGEX .GT. 0) THEN 
             CALL PKSR3_SUB(QMAX,BUFI(IX-1,IY,IZ),BUFI(IX+1,IY,IZ),1,RX)
           ENDIF
           RY = 0
           IF (ISHRANGEY .GT. 0) THEN 
             CALL PKSR3_SUB(QMAX,BUFI(IX,IY-1,IZ),BUFI(IX,IY+1,IZ),1,RY)
           ENDIF
           CALL PKSR3_SUB  (QMAX,BUFI(IX,IY,IZ-1),BUFI(IX,IY,IZ+1),1,RZ)
          
           XSHNEW  = IXSH - RX 
           YSHNEW  = IYSH - RY
           ZSHNEW  = IZSH - RZ

        ELSEIF (ISHRANGEX .GT. 0 .AND. ISHRANGEY .GT. 0)THEN !--- IMAGE

C          DO NOT INTERPOLATE IF THE MAX IS ON THE EDGE
           IF (IX .LE. 1 .OR. IX .GE. NSAM .OR.
     &         IY .LE. 1 .OR. IY .GE. NROW) RETURN

           IF (ABS(IXSH) .GE. ISHRANGEX .OR. 
     &         ABS(IYSH) .GE. ISHRANGEY) RETURN

           NT2 = 1
           IF (NSLICE .GT. 1) NT2 = 2
           DO J=-1,1
              DO I=-1,1
                 Z(I,J) = BUFI(IX+I, IY+J, NT2)  ! 2 for nslice is 2x
              ENDDO
           ENDDO

C          2D PARABOLIC FIT TO 3X3 NEIGHBORHOOD OF PEAK
           CALL PARABL(Z, RX,RY, PEAKPAR)

           XSHNEW = IXSH - RX
           YSHNEW = IYSH - RY
           ZSHNEW = 0.0

           !write(6,*) ' '
           !write(6,*) ' 2d buf,peak: ',bufi(ix,iy,iz),peakv
           !write(6,*) ' Shifts:      ',ixsh,iysh, xshnew,yshnew
           !write(6,*) ' Shifts:      ',ixsh,iysh, rx,ry
           !write(6,"('  z:',1p3g12.4)") z 
   

        ELSEIF (ISHRANGEX .GT. 0) THEN ! -------------------------- ROW

           IF (IX .LE. 1 .OR. IX .GE. NSAM) RETURN

           IF (ABS(IXSH) .GE. ISHRANGEX) RETURN

C          BINARY INTERPOLATION TO FIND SUB-PIXEL PEAK
           QMAX = BUFI(IX,IY,IZ)         ! CENTRAL ELEMENT
           CALL PKSR3_SUB(QMAX,BUFI(IX-1,IY,IZ),BUFI(IX+1,IY,IZ),1,RX)

           XSHNEW = IXSH - RX 
           YSHNEW = 0.0
           ZSHNEW = 0.0

        ELSEIF (ISHRANGEY .GT. 0) THEN ! ----------------------- COLUMN

           IF (IY .LE. 1 .OR. IY .GE. NROW) RETURN

           IF (ABS(IYSH) .GE. ISHRANGEY)    RETURN

C          BINARY INTERPOLATION TO FIND SUB-PIXEL PEAK
           QMAX = BUFI(IX,IY,IZ)         ! CENTRAL ELEMENT
           CALL PKSR3_SUB(QMAX,BUFI(IX,IY-1,IZ),BUFI(IX,IY+1,IZ),1,RY)

           XSHNEW = 0.0 
           YSHNEW = IYSH - RY
           ZSHNEW = 0.0

        ENDIF
        
C       NORMALIZATION DONE IN CALLER IF DESIRED

        IRTFLG = 0

        END




C ------------------------- UNUSED BELOW HERE -------------------------

#ifdef NEVER
#ifdef DEBUG
           write(6,*)'  '
           write(6,*)'IGOX,IENDX,IGOY,IENDY ',IGOX,IENDX,IGOY,IENDY
           write(6,*)'  '
           write(6,*)'ILOCS:        ',ilocs
           write(6,*)'IX,IY,IZ:     ',ix,iy
           write(6,*)'IXSH,IYSH:    ',ixsh,iysh 
           write(6,*)'XSHNEW,YSHNEW:',xshnew,yshnew
           write(6,*)'PEAKV:        ',peakv
#endif
#ifdef DEBUG
           write(6,*)'  '
           write(6,*)'XSHNEW,YSHNEW:',xshnew,yshnew
#endif

           write(6,*)'  '
           write(6,*)'IGOX,IENDX,IGOY,IENDY,IGOZ,IENDZ '
           write(6,*) IGOX,IENDX,IGOY,IENDY,IGOZ,IENDZ

           write(6,*)'  '
           write(6,*)'ILOCS:          ',ilocs
           write(6,*)'IX,IY,IZ:       ',ix,iy,iz
           write(6,*)'IXSH,IYSH,IZSH: ',ixsh,iysh,izsh 

           write(6,*)'  '
           write(6,*)'RX,RY,RZ:',rx,ry,rz
           write(6,*)'XSHNEW,YSHNEW,ZSHNEW:',xshnew,yshnew,zshnew

           write(6,*)'  '
           write(6,*)'peakv: ',peakv

           nnn = nsam*nrow*nslice
           lnn = lsc*nrow*nslice
           write(6,*)'nsam*nrow*nslice:',nnn
           write(6,*)'lsc*nrow*nslice:',lnn
           pdnnn = peakv / float(nnn)
           pdlnn = peakv / float(lnn)
           write(6,*)'pdnnn,pdlnn:',pdnnn,pdlnn
#endif

#ifdef NEVER
        call opfilec(0,.false.,'jnk002',98,'O',itype,
     &               nsam,nrow,nslice,maxim,' ',.true.,irtflg)
        call redvol(98,nsam,nrow, 1,nslice, bufs,irtflg)
        close(98)
        write(6,*) ' bufs filled with jnk002 '

        maxim = 0
        itype = -21
        if (mod(nsam,2).eq.0) itype = -22
        call opfilec(0,.false.,'jnkapsccffti',98,'U',itype,
     &               lsc,nrow,nslice,maxim,' ',.true.,irtflg)
        call wrtvol(98,lsc,nrow, 1,nslice, bufi,irtflg)
        close(98)

        itype = -21
        if (mod(nsam,2).eq.0) itype = -22
        call opfilec(0,.false.,'jnkapsccfftr',98,'U',itype,
     &                lsc,nrow,nslice,maxim,' ',.true.,irtflg)
        call wrtvol(98,lsc,nrow,1,nslice, bufr,irtflg)
        close(98)

        maxim = 0
        itype = 3
        call opfilec(0,.false.,'jnknewpkpad',98,'U',itype,
     &                nsam,nrow,nslice,maxim,' ',.false.,irtflg)

c       this only writes first lsc values from each row
        call writev(98,bufi,lsc,nrow, nsam,nrow,nslice)
        close(98)
#endif


