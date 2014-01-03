typedef long dsize_type;

// future swig example http://mdanalysis.googlecode.com/svn/trunk/src/KDTree/KDTree.i

/**
 * todo: remove fshifts by changing index
 */
template<class T>
void resample_fft_center(T* img, dsize_type img_r, dsize_type img_c, T* out, dsize_type out_r, dsize_type out_c)
{
	dsize_type rend=std::min(img_r, out_r);
	dsize_type cend=std::min(img_c, out_c);
	dsize_type img_rbeg;
	dsize_type img_cbeg;
	dsize_type out_rbeg;
	dsize_type out_cbeg;
	if( ((img_r%2) == 0 && out_r>img_r) || ((img_r%2)!= 0 &&  out_r < img_r) )
	{
		out_rbeg = std::max(std::floor((out_r-img_r)/2.0), 0.0);
		img_rbeg = std::max(std::floor((img_r-out_r)/2.0), 0.0);
	}
	else
	{
		out_rbeg = std::max(std::ceil((out_r-img_r)/2.0), 0.0);
		img_rbeg = std::max(std::ceil((img_r-out_r)/2.0), 0.0);
	}
	if( ((img_c%2) == 0 && out_c>img_c) || ((img_c%2)!= 0 &&  out_c < img_c) )
	{
		out_cbeg = std::max(std::floor((out_c-img_c)/2.0), 0.0);
		img_cbeg = std::max(std::floor((img_c-out_c)/2.0), 0.0);
	}
	else
	{
		out_cbeg = std::max(std::ceil((out_c-img_c)/2.0), 0.0);
		img_cbeg = std::max(std::ceil((img_c-out_c)/2.0), 0.0);
	}
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(dsize_type r = 0;r<rend;++r)
	{
		for(dsize_type c = 0;c<cend;++c)
		{
			out[out_cbeg+c+((out_rbeg+r)*out_c)] = img[img_cbeg+c+((img_rbeg+r)*img_c)];
		}
	}
}

template<class T> inline T wrap_kernel(T* kernel, T k, T fltb)
{
	return kernel[dsize_type((k<0.0) ? -k*fltb+0.5 : k*fltb+0.5)];
}

/**
 * Adopted from SPARX
 */
template<class T> // rows, cols
void downsample(T* img, dsize_type irow, dsize_type icol, T* out, dsize_type orow, dsize_type ocol, T* kernel, dsize_type ksize, T scale)
{

	ksize-=3;
	dsize_type msize = (dsize_type)kernel[0];
	T fltb = kernel[1];
	dsize_type kbmin = -msize/2;
	dsize_type kbmax = -kbmin;
	dsize_type kbc   = kbmax+1;
	kernel+=2;

	for(dsize_type dr = 0;dr<orow;dr++)
	{
		T sr = T(dr)/scale;
		dsize_type dr_orig = dsize_type(sr + 0.5);
		dsize_type drow = dr*ocol;
		for(dsize_type dc=0;dc<ocol;dc++)
		{
			T val = 0.0;
			T w = 0.0;
			T sc = T(dc)/scale;
			dsize_type dc_orig = dsize_type(sc + 0.5);
			if( dr_orig<=kbc || dr_orig>=(irow-kbc-2) || dc_orig <= kbc || dc_orig>=(icol-kbc-2) )
			{
				for(dsize_type mr = kbmin;mr<=kbmax;mr++)
				{
					T k = sr - dr_orig-mr;
					T t = wrap_kernel(kernel, k, fltb);
					dsize_type row = ((dr_orig+mr+irow)%irow)*icol;
					for(dsize_type mc = kbmin;mc<=kbmax;mc++)
					{
						T k = sc-dc_orig-mc;
						T q = wrap_kernel(kernel, k, fltb)*t;
						val += img[ row  + ((dc_orig+mc+icol)%icol) ]*q;
						w += q;
					}
				}
			}
			else
			{
				for(dsize_type mr = kbmin;mr<=kbmax;mr++)
				{
					T k = sr - dr_orig-mr;
					T t = wrap_kernel(kernel, k, fltb);
					dsize_type row = (dr_orig+mr)*icol;
					for(dsize_type mc = kbmin;mc<=kbmax;mc++)
					{
						T k = sc-dc_orig-mc;
						T q = wrap_kernel(kernel, k, fltb)*t;
						val += img[ row + (dc_orig+mc) ]*q;
						w += q;
					}
				}
			}
			out[drow+dc] = val/(w);//+1e-9);
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
