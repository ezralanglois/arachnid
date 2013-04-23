         SUBROUTINE ROTATE_IMAGE(XIMG,BUFOUT, NX,NY, NXP,NYP,
     &                     THETA,SCLI,SHXI,SHYI)

         REAL            :: XIMG(NX,NY)
         INTEGER         :: NX,NY
         REAL            :: BUFOUT(NXP,NYP)
         INTEGER         :: NXP,NYP
         REAL            :: THETA,SCLI,SHXI,SHYI
         INTEGER         :: IRTFLG

cf2py threadsafe
cf2py intent(inplace) :: XIMG,BUFOUT
cf2py intent(in) :: NX,NY, NXP,NYP,THETA,SCLI,SHXI,SHYI
cf2py intent(hide) :: NX,NY, NXP,NYP

         CALL RTSQ(XIMG, BUFOUT,NX,NY,NXP,NYP,THETA,
     &             SCLI,SHXI,SHYI,IRTFLG)
		END

C ---------------------------------------------------------------------------

         SUBROUTINE  ROTATE_EULER(FI1,FI2,FIO)
         REAL            :: FI1(3),FI2(3),FIO(3)

cf2py intent(inout) :: FI1,FI2,FIO
		CALL CALD(FI1, FI2, FIO)
		END
