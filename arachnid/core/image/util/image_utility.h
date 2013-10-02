//#include <complex>

template<class T>
inline void sine_psd(std::complex<T>* img, int nix, int niy, T* out, int nox, int noy, int klim, int npad)
{
	int nf = nix/npad;
	int np2 = nix;
	T ck = 1.0/T(klim*klim);
	for(int mx=0;mx<nf;++mx)
	{
		int mx2 = mx*2;
		for(int my=0;my<nf;++my)
		{
			int my2 = my*2;
			for(int kx=0;kx<klim;++kx)
			{
				int jx1 = (mx2+np2-(kx+1))%np2;
				int jx2 = (mx2+(kx+1))%np2;
				for(int ky=0;ky<klim;++ky)
				{
					int jy1 = (my2+np2-(ky+1))%np2;
					int jy2 = (my2+(ky+1))%np2;
					std::complex<T> zz = img[jx1+jx2*nix]-img[jy1+jy2*nix];
					T wk = (1.0 - ck*T(kx)*T(kx))* (1.0 - ck*T(ky)*T(ky));
					out[mx+my*nix] += ( std::real(zz)*std::real(zz) + std::imag(zz)*std::imag(zz) )*wk;
				}
			}
			out[mx+my*nix] *= (6.0*T(klim)/T(4*klim*klim+3*klim-1));
		}
	}
}

template<class T>
inline void sine_psd_1D(std::complex<T>* roo, int nrx, T* out1, int nox1, int klim, int npad)
{
	int nf = nrx/npad;
	int np2 = nrx;
	T ck = 1.0/T(klim*klim);
	for(int m=0;m<nf;++m)
	{
		int m2 = m*2;
		for(int k=0;k<klim;++k)
		{
			int j1 = (m2+np2-(k+1))%np2;
			int j2 = (m2+(k+1))%np2;
			std::complex<T> zz = roo[j1]-roo[j2];
			T wk = (1.0 - ck*T(k)*T(k));
			out1[m] += ( std::real(zz)*std::real(zz) + std::imag(zz)*std::imag(zz) )*wk;
		}
		out1[m] *= (6.0*T(klim)/T(4*klim*klim+3*klim-1));
	}
}


/** Calculate a radon transform matrix.
 *
 *
 * @param dist 1D representation of a 2D matrix
 * @param sdist sparse radon transform matrix data
 * @param ns number of elements in sparse radon transform
 * @param Dr sparse row indices for radon transform
 * @param nr number of sparse row indices for radon transform
 * @param Dc sparse column indices for radon transform
 * @param nc number of sparse row indices for radon transform
 * @param irow number of rows
 * @param icol number of columns
 * @param nang number of angles to sample from 1 to 180
 * param nthreads number of threads
 */
template<class I, class T>
inline int radon_transform(T * sdist, int ns, I* Dr, int nr, I* Dc, int nc, int nang, int irow, int icol)
{
	int ind;
	int sindex = 0;
    int xOrigin = (icol-1) / 2;
    int yOrigin = (irow-1) / 2;
    int temp1 = irow - 1 - yOrigin;
    int temp2 = icol - 1 - xOrigin;
    int rLast, rFirst, rSize;
    int area = irow*icol;
    double *image, *angles, *rad;
    double *ySinTable, *xCosTable;
    double deg2rad = 3.14159265358979 / 180.0;

    rLast = (int)ceil(sqrt((double) (temp1*temp1+temp2*temp2))) + 1;
    rFirst = -rLast;
    rSize = rLast - rFirst + 1;

    if( (xCosTable = (double *) malloc(2*icol*sizeof(double))) == 0 ) return -1;
    if( (ySinTable = (double *) malloc(2*irow* sizeof(double))) == 0 ) return -1;
    if( (image=(double*)malloc(area*sizeof(double))) == 0 ) return -1;
	if( (rad=(double*)malloc(rSize*nang*sizeof(double))) == 0 ) return -1;
	if( (angles=(double*)malloc(nang*sizeof(double))) == 0 ) return -1;
	for(int i=0;i<nang;++i) angles[i] = (i/((double)nang)*180.0)*deg2rad;

	for(int n=0;n<icol;n++)
	{
		for(int m=0;m<irow;m++)
	    {
			ind=n*irow+m;
			memset(image, 0, area*sizeof(double));
			memset(rad, 0, rSize*nang*sizeof(double));
			image[ind] = 1.0;
			radon(rad, image, angles, xCosTable, ySinTable, irow, icol, xOrigin, yOrigin, nang, rFirst, rSize);
			for(int k=0, kn=rSize*nang; k<kn; k++)
			{
				if( rad[k] <= 1e-10 ) continue;
				if(sindex >= ns)
				{
					fprintf(stderr, "Error: input sparse data array too short -- %d >= %d\n", sindex, ns);
					return -1;
				}
				sdist[sindex] = rad[k];
				Dr[sindex] = k;
				Dc[sindex] = ind;
				sindex++;
			}
	    }
	}
	free(image);
	free(rad);
	free(angles);
	free(xCosTable);
	free(ySinTable);
	return sindex;
}

/** Calculate a radon transform matrix.
 *
 *
 * @param dist 1D representation of a 2D matrix
 * @param sdist sparse radon transform matrix data
 * @param ns number of elements in sparse radon transform
 * @param Dr sparse row indices for radon transform
 * @param nr number of sparse row indices for radon transform
 * @param Dc sparse column indices for radon transform
 * @param nc number of sparse row indices for radon transform
 * @param irow number of rows
 * @param icol number of columns
 * @param nang number of angles to sample from 1 to 180
 * param nthreads number of threads
 */
inline int radon_count(int nang, int irow, int icol)
{
	int ind;
	int sindex = 0;
    int xOrigin = (icol-1) / 2;
    int yOrigin = (irow-1) / 2;
    int temp1 = irow - 1 - yOrigin;
    int temp2 = icol - 1 - xOrigin;
    int rLast, rFirst, rSize;
    int area = irow*icol;
    double *image, *angles, *rad;
    double *ySinTable, *xCosTable;
    double deg2rad = 3.14159265358979 / 180.0;

    rLast = (int)ceil(sqrt((double) (temp1*temp1+temp2*temp2))) + 1;
    rFirst = -rLast;
    rSize = rLast - rFirst + 1;

    if( (xCosTable = (double *) malloc(2*icol*sizeof(double))) == 0 ) return -1;
    if( (ySinTable = (double *) malloc(2*irow* sizeof(double))) == 0 ) return -1;
    if( (image=(double*)malloc(area*sizeof(double))) == 0 ) return -1;
	if( (rad=(double*)malloc(rSize*nang*sizeof(double))) == 0 ) return -1;
	if( (angles=(double*)malloc(nang*sizeof(double))) == 0 ) return -1;
	for(int i=0;i<nang;++i) angles[i] = (i/((double)nang)*180.0)*deg2rad;

	for(int n=0;n<icol;n++)
	{
		for(int m=0;m<irow;m++)
	    {
			ind=n*irow+m;
			memset(image, 0, area*sizeof(double));
			memset(rad, 0, rSize*nang*sizeof(double));
			image[ind] = 1.0;
			radon(rad, image, angles, xCosTable, ySinTable, irow, icol, xOrigin, yOrigin, nang, rFirst, rSize);
			for(int k=0, kn=rSize*nang; k<kn; k++)
			{
				if( rad[k] <= 1e-10 ) continue;
				sindex++;
			}
	    }
	}
	free(image);
	free(rad);
	free(angles);
	free(xCosTable);
	free(ySinTable);
	return sindex;
}

template<class T>
void rotavg(T* out, int onx, int ony, int onz, T* avg, int na, int rmax)
{
	T padded = 0.0;
	int padcnt = 0;

#	if defined(_OPENMP_3_0)
#	pragma omp parallel for reduction(+: padded, padcnt) collapse(3)
#	elif defined(_OPENMP)
#	pragma omp parallel for reduction(+: padded, padcnt)
#	endif
	for(int z=-rmax;z<=rmax;++z)
	{
		for(int y=-rmax;y<=rmax;++y)
		{
			for(int x=-rmax;x<=rmax;++x)
			{
				T r = std::sqrt( T(z*z) + T(y*y) + T(x*x) );
				int ir = int(r);
				if( (rmax-2) <= ir && ir <= rmax )
				{
					if( ir >= na ) fprintf(stderr, "error here1\n");
					padded += avg[ir];
					padcnt += 1;
				}
			}
		}
	}
	padded /= padcnt;

	int lz=(onz/2+onz%2), ly=(ony/2+ony%2), lx=(onx/2+onx%2);
	int nz = onx*ony;
	int tot = nz*onz;
	if( onz == 1 ) nz = 0;
#	if defined(_OPENMP_3_0)
#	pragma omp parallel for collapse(3)
#	elif defined(_OPENMP)
#	pragma omp parallel for
#	endif
	for(int z=-onz/2;z<lz;++z)
	{
		for(int y=-ony/2;y<ly;++y)
		{
			for(int x=-onx/2;x<lx;++x)
			{
				T r = std::sqrt( T(z*z) + T(y*y) + T(x*x) );
				int ir = int(r);
				int idx = x+onx/2 + (y+ony/2)*onx + (z+onz/2)*nz;
				if( ir >= rmax )
				{
					if( idx >= tot || idx < 0 ) fprintf(stderr, "error here3\n");
					out[idx] = padded;
				}
				else
				{
					if( idx >= tot || idx < 0 ) fprintf(stderr, "error here4\n");
					if( (ir+1) >= na ) fprintf(stderr, "error here2\n");
					out[idx] = avg[ir] + ( avg[ir+1] - avg[ir] ) * ( r - T(ir) );
				}
			}
		}
	}
}
