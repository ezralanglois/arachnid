

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
					padded += avg[ir];
					padcnt += 1;
				}
			}
		}
	}
	padded /= padcnt;

	int lz=(onz/2+onz%2), ly=(ony/2+ony%2), lx=(onx/2+onx%2);
#	if defined(_OPENMP_3_0)
#	pragma omp parallel for reduction(+: padded, padcnt) collapse(3)
#	elif defined(_OPENMP)
#	pragma omp parallel for reduction(+: padded, padcnt)
#	endif
	for(int z=-onz/2;z<lz;++z)
	{
		for(int y=-ony/2;y<ly;++y)
		{
			for(int x=-onx/2;x<lx;++x)
			{
				T r = std::sqrt( T(z*z) + T(y*y) + T(x*x) );
				int ir = int(r);
				if( ir >= rmax )
				{
					out[x+y*onx+z*onx*ony] = padded;
				}
				else
				{
					out[x+y*onx+z*onx*ony] = avg[ir] + ( avg[ir+1] - avg[ir] ) * ( r - T(ir) );
				}
			}
		}
	}

}
