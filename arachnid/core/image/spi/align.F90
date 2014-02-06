

C--*********************************************************************
		SUBROUTINE ALIGN_T(BUFI,BUFR,LSE,NSAM,NROW,NSLICE,
     &						ISHRANGEX,ISHRANGEY,ISHRANGEZ,
     &						XSHNEW,YSHNEW,ZSHNEW,PEAKV,
     &						IRTFLG)

        REAL                    :: BUFI(LSE,NROW,NSLICE)
        REAL                    :: BUFR(LSE,NROW,NSLICE)
        INTEGER 			    :: LSE,NSAM,NROW,NSLICE
        INTEGER  			   	:: ISHRANGEX,ISHRANGEY,ISHRANGEZ
        REAL   				 	:: XSHNEW,YSHNEW,ZSHNEW,PEAKV
        INTEGER    				:: IRTFLG

C        INTEGER, INTENT(IN)     :: LSE,NSAM,NROW,NSLICE
C        INTEGER, INTENT(IN)     :: ISHRANGEX,ISHRANGEY,ISHRANGEZ
C        REAL,    INTENT(OUT)    :: XSHNEW,YSHNEW,ZSHNEW,PEAKV
C        INTEGER, INTENT(OUT)    :: IRTFLG

cf2py threadsafe
cf2py intent(inplace) :: BUFI, BUFR
cf2py intent(in) :: LSE,NSAM,NROW,NSLICE,ISHRANGEX,ISHRANGEY,ISHRANGEZ
cf2py intent(hide) :: LSE,NROW,NSLICE
cf2py intent(out) :: XSHNEW,YSHNEW,ZSHNEW,PEAKV,IRTFLG

              CALL APCC_NEW(LSE,NSAM,NROW,NSLICE,BUFI,BUFR,
     &                  .TRUE.,.TRUE.,
     &                  .FALSE.,.FALSE.,.FALSE.,
     &                  ISHRANGEX,ISHRANGEY,ISHRANGEZ,
     &                  XSHNEW,YSHNEW,ZSHNEW,PEAKV,IRTFLG)


		END


