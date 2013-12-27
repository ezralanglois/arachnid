
typedef long dsize_type;
#include <cmath>
//#include <cstdlib>

template<class T>
T integrate(T* img_int, dsize_type r0, dsize_type c0, dsize_type r1, dsize_type c1)
{
    T sum=0.0;

    sum += img_int[r1, c1];

    if ( (r0 - 1 >= 0) && (c0 - 1 >= 0) )
        sum += img_int[r0 - 1, c0 - 1];

    if ( (r0 - 1) >= 0)
        sum -= img_int[r0 - 1, c1];

    if ( (c0 - 1) >= 0)
    	sum -= img_int[r1, c0 - 1];
    return sum;
}

template<class T>
void normalize_correlation(T* ccmap, dsize_type row, dsize_type col, T* img_int, dsize_type irow, dsize_type icol, T* img_int_sq, dsize_type i2row, dsize_type i2col, dsize_type trow, dsize_type tcol, T ref_ssd)
{

	//fprintf(stderr, "vals: %f, %f\n", ccmap[0], ccmap[row/2*col+col/2]);
	T inv_area = 1.0 / (trow * tcol);
//#	if defined(_OPENMP)
//#	pragma omp parallel for
//#	endif
	for(dsize_type r=0;r<row;++r)
	{
		dsize_type r_end = r + trow - 1;
		for(dsize_type c=0;c<col;++c)
		{
			dsize_type c_end = c + tcol - 1;
			T win_sum = integrate(img_int, r, c, r_end, c_end);
			T win_sum_sq = integrate(img_int_sq, r, c, r_end, c_end);
			/*if (r >=irow || r_end >= irow)
			{
				fprintf(stderr, "row: %d > %d or %d > %d\n", r, irow, r_end, irow);
				std::exit(1);
			}
			if (c >=icol || c_end >= icol)
			{
				fprintf(stderr, "col: %d > %d or %d > %d\n", c, icol, c_end, icol);
				std::exit(1);
			}*/
			T win_mean_sq = win_sum*win_sum*inv_area;
			if( win_sum_sq <= win_mean_sq )
			{
				ccmap[r,c]=0;
				continue;
			}
			T den = std::sqrt((win_sum_sq-win_mean_sq)*ref_ssd);
			T val = ccmap[r,c];
			/*if (r == 0 && c==0)
			{
				if(std::isinf(val))fprintf(stderr, "here-inf, %f - %f * %f == (%d,%d) - %f/%f\n", win_sum_sq, win_mean_sq, ref_ssd, r, c, val, den);
			}*/
			ccmap[r,c]/=den;
		}
	}
}
