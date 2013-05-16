#include <math.h>

static void incrementRadon(double *pr, double pixel, double r)
{
    int r1;
    double delta;
    r1 = (int) r;
    delta = r - r1;
    pr[r1] += pixel * (1.0 - delta);
    pr[r1+1] += pixel * delta;
}

void radon(double *pPtr, double *iPtr, double *thetaPtr, double *xCosTable, double *ySinTable, int M, int N,
           int xOrigin, int yOrigin, int numAngles, int rFirst, int rSize)
{
    int k, m, n;              /* loop counters */
    double angle;             /* radian angle value */
    double cosine, sine;      /* cosine and sine of current angle */
    double *pr;               /* points inside output array */
    double *pixelPtr;         /* points inside input array */
    double pixel;             /* current pixel value */
    double x,y;
    double r, delta;
    int r1;


    for (k = 0; k < numAngles; k++) {
        angle = thetaPtr[k];
        pr = pPtr + k*rSize;  /* pointer to the top of the output column */
        cosine = cos(angle);
        sine = sin(angle);

        /* Radon impulse response locus:  R = X*cos(angle) + Y*sin(angle) */
        /* Fill the X*cos table and the Y*sin table.                      */
        /* x- and y-coordinates are offset from pixel locations by 0.25 */
        /* spaced by intervals of 0.5. */
        for (n = 0; n < N; n++)
        {
            x = n - xOrigin;
            xCosTable[2*n]   = (x - 0.25)*cosine;
            xCosTable[2*n+1] = (x + 0.25)*cosine;
        }
        for (m = 0; m < M; m++)
        {
            y = yOrigin - m;
            ySinTable[2*m] = (y - 0.25)*sine;
            ySinTable[2*m+1] = (y + 0.25)*sine;
        }

        pixelPtr = iPtr;
        for (n = 0; n < N; n++)
        {
            for (m = 0; m < M; m++)
            {
                pixel = *pixelPtr++;
                if (pixel != 0.0)
                {
                    pixel *= 0.25;

                    r = xCosTable[2*n] + ySinTable[2*m] - rFirst;
		    //printf("\nr1=%f ", (float)r);fflush(stdout);
                    incrementRadon(pr, pixel, r);

                    r = xCosTable[2*n+1] + ySinTable[2*m] - rFirst;
                    //printf("\nr2=%f ", (float)r);fflush(stdout);
                    incrementRadon(pr, pixel, r);

                    r = xCosTable[2*n] + ySinTable[2*m+1] - rFirst;
                    //printf("\nr3=%f ", (float)r);fflush(stdout);
                    incrementRadon(pr, pixel, r);

                    r = xCosTable[2*n+1] + ySinTable[2*m+1] - rFirst;
                    //printf("\nr4=%f ", (float)r);fflush(stdout);
                    incrementRadon(pr, pixel, r);
                }
            }
        }
    }
}

