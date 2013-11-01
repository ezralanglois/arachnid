typedef long dsize_type;


template<class T> inline T wrap_kernel(T* kernel, T k, T fltb)
{
	return kernel[dsize_type((k<0.0) ? -k*fltb+0.5 : k*fltb+0.5)];
}

/**
 * Adopted from SPARX
 */
template<class T>
void downsample(T* img, dsize_type nx, dsize_type ny, T* out, dsize_type ox, dsize_type oy, T* kernel, dsize_type ksize, T scale)
{

	ksize-=3;
	dsize_type msize = (dsize_type)kernel[0];
	T fltb = kernel[1];
	dsize_type kbmin = -msize/2;
	dsize_type kbmax = -kbmin;
	dsize_type kbc   = kbmax+1;
	kernel+=2;

	for(dsize_type yd=0;yd<oy;yd++)
	{
		T ys = T(yd)/scale;
		dsize_type y_orig = dsize_type(ys + 0.5);
		for(dsize_type xd = 0;xd<ox;xd++)
		{
			T val = 0.0;
			T w = 0.0;
			T xs = T(xd)/scale;
			dsize_type x_orig = dsize_type(xs + 0.5);
			if( x_orig<=kbc || x_orig>=(nx-kbc-2) || y_orig <= kbc || y_orig>=(ny-kbc-2) )
			{
				for(dsize_type my = kbmin;my<=kbmax;my++)
				{
					T k = ys-y_orig-my;
					T t = wrap_kernel(kernel, k, fltb);
					for(dsize_type mx = kbmin;mx<=kbmax;mx++)
					{
						T k = xs - x_orig-mx;
						T q = wrap_kernel(kernel, k, fltb)*t;
						val += img[ ((x_orig+mx+nx)%nx) + ((y_orig+my+ny)%ny)*nx ]*q;
						w += q;
					}
				}
			}
			else
			{
				for(dsize_type my = kbmin;my<=kbmax;my++)
				{
					T k = ys-y_orig-my;
					T t = wrap_kernel(kernel, k, fltb);
					for(dsize_type mx = kbmin;mx<=kbmax;mx++)
					{
						T k = xs - x_orig-mx;
						T q = wrap_kernel(kernel, k, fltb)*t;
						val += img[ (x_orig+mx) + (y_orig+my)*nx ]*q;
						w += q;
					}
				}
			}
			out[xd+yd*ox] = val/(w+1e-9);
		}
	}
}

const long double pi = 3.141592653589793238462643383279502884197L;
const long double twopi = 2*pi;

/**
 * Adopted from SPARX
 */
template<class T>
void sinc_blackman_kernel(T* kernel, dsize_type ksize, int m, float freq)
{
	ksize-=3;
	dsize_type ltab = dsize_type(ksize/1.25 + 0.5);
	dsize_type mhalf = (dsize_type)m/2;
	T fltb = T(ltab)/mhalf;
	T x = 1.0e-7;
	kernel[0]=m;
	kernel[1]=fltb;
	kernel+=2;

	kernel[0] = (T)(std::sin(twopi*freq*x)/x*(0.52-0.5*std::cos(twopi*(x-mhalf)/m)+0.08*std::cos(2*twopi*(x-mhalf)/m)));
	//kernel[0] = (T)(std::sin(twopi*freq*x)/x*(0.42-0.5*std::cos(twopi*(x-mhalf)/m)+0.08*std::cos(2*twopi*(x-mhalf)/m)));
	for(dsize_type i=1;i<ksize;i++) kernel[i]=T(0.0);
	for(dsize_type i=1;i<=ltab;++i)
	{
		x = T(i)/fltb;
		//kernel[i] = (T)(std::sin(twopi*freq*x)/x*(0.42-0.5*std::cos(twopi*(x-mhalf)/m)+0.08*std::cos(2*twopi*(x-mhalf)/m)));
		kernel[i] = (T)(std::sin(twopi*freq*x)/x*(0.52-0.5*std::cos(twopi*(x-mhalf)/m)+0.08*std::cos(2*twopi*(x-mhalf)/m)));
	}
}
