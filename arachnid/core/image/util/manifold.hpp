typedef long size_type;

#define USE_BLAS
#ifdef USE_BLAS
inline void x_gemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
{
	cblas_sgemm(order, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline void x_gemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
{
	cblas_dgemm(order, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<class T>
void gemm(T* samp1, int n1, int m1, T* samp2, int n2, int m2, T* distm, int n3, int m3, double alpha, double beta)
{
	//x_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,      n1, n2, m1, T(alpha), samp1, m1, samp2, m1, T(beta), distm, n2);
	x_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, n1, n2, m1, T(alpha), samp1, m1, samp2, m2, T(beta), distm, m3);
}

/*
 SUBROUTINE xGEMM ( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )
 where TRANSA and TRANSB determines
 if the matrices A and B are to be transposed.
 M is the number of rows in matrix C and, depending on TRANSA, the number of rows in the original matrix A or its transpose.
 N is the number of columns in matrix C and, depending on TRANSB, the number of columns in the matrix B or its transpose.
 K is the number of columns in matrix A (or its transpose) and rows in matrix B (or its transpose).
 LDA, LDB and LDC specifies the size of the first dimension of the matrices, as laid out in memory; meaning the memory distance between the start of each row/column, depending on the memory structure
 */

template<class T>
void gemm_t1(T* samp1, int n1, int m1, T* samp2, int n2, int m2, T* distm, int n3, int m3, double alpha, double beta)
{
	x_gemm(CblasRowMajor, CblasTrans, CblasNoTrans, m1, m2, n1, T(alpha), samp1, m1, samp2, m2, T(beta), distm, m3);
}
#endif
/*
 * inline void x_gemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
{
	cblas_dgemm(order, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
			x_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, batchr, batchc, cn, alpha, samp+br*m, cn, samp+bc*m, cn, beta, pdist, batchc);
 */

template<class I, class T>
void knn_restricted_dist_mask(T* data, size_type nd, I* col_ind, size_type nc, I* row_ind, size_type nr, T* samp1, int n1, int m1, I* maskidx, size_type mn)
{

#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(size_type i=0;i<nr;++i)
	{
		if( row_ind[i] == col_ind[i] )
		{
			data[i]=0;
			continue;
		}
		T d = 0;
		unsigned long r = row_ind[i];
		r *= m1;
		T* sampr = samp1+r;
		r = col_ind[i];
		r *= m1;
		T* sampc = samp1+r;
		for(size_type j=0;j<mn;++j)
		{
			size_type k=maskidx[j];
			d += (sampr[k]-sampc[k])*(sampr[k]-sampc[k]);
		}
		data[i]=d;
	}
}

template<class I>
void knn_offset(I* row_ind, size_type nr, I* offsets, size_type on)
{
	for(size_type i=0;i<nr;++i)
	{
		offsets[row_ind[i]]++;
	}
}

template<class I, class T>
void knn_restricted_dist(T* data, size_type nd, I* col_ind, size_type nc, I* row_ind, size_type nr, T* samp1, int n1, int m1)
{

#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(size_type i=0;i<nr;++i)
	{
		if( row_ind[i] == col_ind[i] )
		{
			data[i]=0;
			continue;
		}
		T d = 0;
		unsigned long r = row_ind[i];
		r *= m1;
		T* sampr = samp1+r;
		r = col_ind[i];
		r *= m1;
		T* sampc = samp1+r;
		for(size_type j=0;j<m1;++j)
			d += (sampr[j]-sampc[j])*(sampr[j]-sampc[j]);
		data[i]=d;
	}
}

template<class I, class T>
I knn_reduce_eps_cmp(T* data, size_type nd, I* col_ind, size_type nc, I* row_ind, size_type nr, T* sdata, size_type snd, I* scol_ind, size_type snc, I* srow_ind, size_type snr, T* cdata, size_type cnd, float eps)
{
	I j=0;
	for(size_type r=0;r<cnd;++r)
	{
		//if( r < 20 ) fprintf(stderr, "data[%d]=%f < %f\n", r, data[r], eps);
		if( cdata[r] < eps )
		{
			sdata[j]=data[r];
			scol_ind[j]=col_ind[r];
			srow_ind[j]=row_ind[r];
			++j;
		}
	}
	return j;
}


template<class I, class T>
I knn_reduce_eps(T* data, size_type nd, I* col_ind, size_type nc, I* row_ind, size_type nr, T* sdata, size_type snd, I* scol_ind, size_type snc, I* srow_ind, size_type snr, float eps)
{
	I j=0;
	for(size_type r=0;r<nd;++r)
	{
		//if( r < 20 ) fprintf(stderr, "data[%d]=%f < %f\n", r, data[r], eps);
		if( data[r] < eps )
		{
			sdata[j]=data[r];
			scol_ind[j]=col_ind[r];
			srow_ind[j]=row_ind[r];
			++j;
		}
	}
	return j;
}

/*
template<class I, class T>
void knn_reduce_csr(T* data, size_type nd, I* col_ind, size_type nc, I* row_ind, size_type nr, T* sdata, size_type snd, I* scol_ind, size_type snc, I* srow_ind, size_type snr, int d, int k)
{
	size_type j=0;
	for(size_type r=0;r<snr;++j)
	{
		assert(r<snd);
		if(j>=nd)
			{
			fprintf(stderr, "big error\n");
			exit(1);
			}
		assert(j<nd);
		sdata[r]=data[j];
		scol_ind[r]=col_ind[j];
		srow_ind[r]=row_ind[j];
		++r;
		if( (r%k)==0 ) j+=size_type(d);
	}
	assert(j==nd);
}
*/

template<class I, class T>
void knn_reduce(T* data, size_type nd, I* col_ind, size_type nc, I* row_ind, size_type nr, T* sdata, size_type snd, I* scol_ind, size_type snc, I* srow_ind, size_type snr, int d, int k)
{
	/*if( snr > 0 && nd > 0 )
	{
		sdata[0]=data[0];
		scol_ind[0]=col_ind[0];
		srow_ind[0]=row_ind[0];
	}
	size_type j=1;
	for(size_type r=1;r<snr;++r,++j)*/
	size_type j=0;
	for(size_type r=0;r<snr;++j)
	{
		assert(r<snd);
		if(j>=nd)
			{
			fprintf(stderr, "big error\n");
			exit(1);
			}
		assert(j<nd);
		sdata[r]=data[j];
		scol_ind[r]=col_ind[j];
		srow_ind[r]=row_ind[j];
		++r;
		if( (r%k)==0 ) j+=size_type(d);
	}
	assert(j==nd);
}

template<class I>
I* find_mutual(I* b, I* e, I v)
{
	for(;b < e;++b) if( (*b) == v ) return b;
	return 0;
}

template<class I, class T>
I knn_mutual(T* data, size_type nd, I* col_ind, size_type nc, I* row_ind, size_type nr, int k)
{
	I j=0;
	for(size_type r=0;r<nr;++r)
	{
		if( col_ind[r] > row_ind[r] ) // Test if mutual exists
		{
			I c = col_ind[r];
			assert(c>0);
			//if (c < 0) fprintf(stderr, "bug in c\n");
			//if( c < 0 ) c = I(-c);
			I* mc = find_mutual(col_ind+c*k, col_ind+(c+1)*k, row_ind[r]);
			if( mc != 0 )
			{
				(*mc) = -((*mc)+1);
				data[j] = data[r];
				col_ind[j] = col_ind[r];
				row_ind[j] = row_ind[r];
				j++;
			}
		}
		else if( col_ind[r] < row_ind[r] ) // Test if mutual already found
		{
			if( col_ind[r] < 0 )
			{
				data[j] = data[r];
				col_ind[j] = -(col_ind[r]+1);
				row_ind[j] = row_ind[r];
				j++;
			}
		}
		else
		{
			data[j] = 0.0;
			col_ind[j] = col_ind[r];
			row_ind[j] = row_ind[r];
			j++;
		}

	}
	return j;
}

template<class I, class T>
I knn_mutual_alt(T* data, size_type nd, I* col_ind, size_type nc, I* row_ind, size_type nr, int k)
{
	I j=0;
	for(size_type r=0;r<nr;++r)
	{
		if( col_ind[r] > row_ind[r] ) // Test if mutual exists
		{
			I c = col_ind[r];
			I* mc = find_mutual(col_ind+c*k, col_ind+(c+1)*k, row_ind[r]);
			if( mc != 0 )
			{
				data[j] = data[r];
				col_ind[j] = col_ind[r];
				row_ind[j] = row_ind[r];
				++j;
			}
		}
		else if( col_ind[r] == row_ind[r] ) // Test if mutual already found
		{
			data[j] = 0.0;
			col_ind[j] = col_ind[r];
			row_ind[j] = row_ind[r];
			j++;
		}
	}
	//remove?
	for(size_type r=0, nr=j;r<nr;++r)
	{
		if( col_ind[r] != row_ind[r] )
		{
			data[j] = data[r];
			col_ind[j] = col_ind[r];
			row_ind[j] = row_ind[r];
			++j;
		}
	}
	return j;
}


template<class I, class T>
void push_to_heap(T* dist2, size_type n, size_type m, T* data, size_type nd, I* col_ind, size_type nc, size_type offset, size_type k)
{
	typedef std::pair<T,I> index_dist;
	typedef std::vector< index_dist > index_vector;
#ifdef _OPENMP
	index_vector vheap(omp_get_max_threads()*k);
#else
	index_vector vheap(k);
#endif

	size_type m1=m;
	size_type n1=n;
	//size_type t=m1*n1;

#	if defined(_OPENMP)
#	pragma omp parallel for
#	endif
	for(size_type r=0;r<n1;++r)
	{
		size_type rm = r*m;
		size_type rk = r*k;
#ifdef _OPENMP
		typename index_vector::iterator hbeg = vheap.begin()+omp_get_thread_num()*k, hcur=hbeg, hend=hbeg+k;
#else
		typename index_vector::iterator hbeg = vheap.begin(), hcur=hbeg, hend=hbeg+k;
#endif
		T* data_rk = data+rk;
		I* col_rk = col_ind+rk;
		//fprintf(stderr, "r: %d | data: %p - col: %p - hbeg: %p - dist2: %p\n", r, data_rk, col_rk, &(*hbeg), dist2);
		size_type c=0;
		//fprintf(stderr, "here-1 %ld -- %ld, %ld < %ld -- %ld, %ld < %ld\n", r, rm, rk, n*m1, k, offset, std::distance(hcur, hend));
		for(size_type l=std::min(k, offset);c<l;++c, ++hcur)
		{
			*hcur = index_dist(data_rk[c], col_rk[c]);
		}
		c=0;
		assert(hcur<=hend);
		for(;hcur != hend && c<m1;++c, ++hcur)
		{
			*hcur = index_dist(dist2[rm+c], offset+c);
		}
		assert(c==m || hcur == hend);
		if( hcur == hend ) std::make_heap(hbeg, hend);
		//fprintf(stderr, "here-2 %d\n", r);
		/*if ( r == 0)
		{
			if( std::min_element(hbeg, hend)->first > 0 )
				fprintf(stderr, "heap invalid-1: r(%d): %f (%d, %d) - offset: %d\n", r, std::min_element(hbeg, hend)->first, hcur == hend, c==m, offset);
		}*/
		for(;c<m1;++c)
		{
			assert((rm+c)<t);
			T d = dist2[rm+c];
			if( d < hbeg->first )
			{
				*hbeg = index_dist(d, offset+c);
				std::make_heap(hbeg, hend);
			}
		}
		//fprintf(stderr, "here-3 %d\n", r);
		/*if ( r == 0)
		{
			if( std::min_element(hbeg, hend)->first > 0 )
				fprintf(stderr, "heap invalid-2: r(%d): %f (%d, %d) - offset: %d\n", r, std::min_element(hbeg, hend)->first, hcur == hend, c==m, offset);
		}*/
		hcur = hbeg;
		for(c=0;c<k;++c, ++hcur)
		{
			assert(c<nc);
			data_rk[c] = hcur->first;
			col_rk[c] = hcur->second;
		}
		//fprintf(stderr, "here-4 %d\n", r);
	}
}

template<class I, class T>
void finalize_heap(T* data, size_type nd, I* col_ind, size_type nc, size_type offset, size_type k)
{
	typedef std::pair<T,I> index_dist;
	typedef std::vector< index_dist > index_vector;
#ifdef _OPENMP
	index_vector vheap(omp_get_max_threads()*k);
#else
	index_vector vheap(k);
#endif

	size_type e = size_type(T(nd)/k);

#	if defined(_OPENMP)
#	pragma omp parallel for
#	endif
	for(size_type r=0;r<e;++r)
	{
#ifdef _OPENMP
		typename index_vector::iterator hbeg = vheap.begin()+omp_get_thread_num()*k, hcur=hbeg, hend=hbeg+k;
#else
		typename index_vector::iterator hbeg = vheap.begin(), hcur=hbeg, hend=hbeg+k;
#endif
		T* data_rk = data+r*k;
		I* col_rk = col_ind+r*k;
		for(size_type c=0;c<k;++c, ++hcur) *hcur = index_dist(data_rk[c], col_rk[c]);
		std::sort_heap(hbeg, hbeg+k);
		hcur = hbeg;
		size_type c=0;
		if (hcur->second != I(r+offset)) // Ensure that the first neighbor is itself
		{
			assert(c<nc);
			data_rk[c] = 0;
			col_rk[c] = r+offset;
			c++;
		}
		for(;hcur != hend;++hcur)
		{
			if(hcur->second != I(r+offset) || c == 0)
			{
				assert(c<nc);
				data_rk[c] = hcur->first;
				col_rk[c] = hcur->second;
				++c;
				if( c == k ) break;
			}
		}
		if( c != k )
		{
			fprintf(stderr, "Bug for row: %ld -- %ld == %ld\n", r+offset, c, k);
			for(hcur=hbeg;hcur != hend;++hcur) fprintf(stderr, "%f - %ld\n", hcur->first, hcur->second);
			exit(1);
		}
	}
}

template<class I, class T>
I select_subset_csr(T* data, size_type nd, I* col_ind, size_type nc, I* row_ptr, size_type nr, I* selected, size_type scnt)
{
	nr-=1;
	I cnt = 0, rc=1;
	I* index_map = new I[nr];
	for(size_type i=0;i<nr;++i) index_map[i]=I(-1);
	for(size_type i=0;i<scnt;++i) index_map[selected[i]]=I(i);

	for(size_type s = 0;s<scnt;++s)
	{
		I r = selected[s];
		for(I j=row_ptr[r];j<row_ptr[r+1];++j)
		{
			if( index_map[col_ind[j]] != I(-1) )
			{
				data[cnt] = data[j];
				col_ind[cnt] = index_map[col_ind[j]];
				cnt ++;
			}
		}
		row_ptr[rc]=cnt;
		rc++;
	}
	delete[] index_map;
	return cnt;
}



template<class T>
void gaussian_kernel_range(T* dist, size_type nd, T* sigma_cum, size_type ns1, size_type ns2)
{
	T* sigma = sigma_cum;
	T* sum = sigma_cum+ns2;

#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(int i=0;i<nd;i++)
	{
		T val = dist[i];
		if( val < 0.0 ) val = -val;
		for(size_type j=0;j<ns2;++j)
			sum[j] += std::exp(-val / sigma[j]);
	}
}

template<class T>
void gaussian_kernel(T* dist, size_type nd, T* sdist, size_type ns, double sigma)
{
	T sum = T(0.0);
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(int i=0;i<nd;i++)
	{
		T val = dist[i];
		if( val < 0.0 ) val = -val;
		sdist[i] = std::exp(-val / sigma);
	}
}

template<class I, class T>
void self_tuning_gaussian_kernel_csr(T* sdist, size_type ns, T* data, size_type nd, I* col_ind, size_type nc, I* row_ptr, size_type nr)
{
	nr-=1;
	T* ndist = new T[nr];
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(size_type i=0;i<nr;i++)
	{
		ndist[i] = 0;
	}
	for(size_type i=0;i<nc;i++)
	{
		if ( ndist[col_ind[i]] < data[i] )
			ndist[col_ind[i]] = data[i];
	}
	I* row_ind = new I[nc];
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(size_type r=0;r<nr;++r)
	{
		for(I j=row_ptr[r];j<row_ptr[r+1];++j)
		{
			row_ind[j]=r;
		}
	}
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(size_type i=0;i<nc;i++)
	{
		double den = 1.0;
		double val = double(ndist[row_ind[i]]);
		if(val != 0) den *= std::sqrt(val);
		val = double(ndist[col_ind[i]]);
		if(val != 0) den *= std::sqrt(val);
		if( den != 0.0 ) sdist[i] = std::exp( -data[i] / T(den) );//+1e-12
		else sdist[i] = std::exp( -data[i] );
	}
	delete[] ndist;
	delete[] row_ind;
}

template<class I, class T>
void normalize_csr(T* sdist, size_type ns, T* data, size_type nd, I* col_ind, size_type nc, I* row_ptr, size_type nr)
{
	nr-=1;
	T* ndist = new T[nr];
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(size_type i=0;i<nr;i++)
	{
		ndist[i] = 0;
	}
	for(size_type i=0;i<nc;i++)
	{
		ndist[col_ind[i]] += data[i];
	}
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(size_type i=0;i<nr;i++)
	{
		if(ndist[i] == 0) ndist[i]=1.0;
		else ndist[i] = T(1.0) / ndist[i];//(ndist[i]+1e-12);
	}
	I* row_ind = new I[nc];
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(size_type r=0;r<nr;++r)
	{
		for(I j=row_ptr[r];j<row_ptr[r+1];++j) row_ind[j]=r;
	}
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(size_type i=0;i<nc;i++)
	{
		sdist[i] = data[i]*ndist[row_ind[i]]*ndist[col_ind[i]];
	}
	delete[] ndist;
	delete[] row_ind;
}
