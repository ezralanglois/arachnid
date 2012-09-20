

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
