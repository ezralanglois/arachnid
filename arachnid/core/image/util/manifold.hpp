

/*
inline void x_gemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
{
	cblas_sgemm(order, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline void x_gemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
{
	cblas_dgemm(order, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<class T>
void gemm(T* samp1, int n1, int m1, T* samp2, int n2, int m2, T* dist2, int n, int m, T alpha, T beta)
{
	x_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, n1, n2, cn, alpha, samp1, cn, samp2, cn, beta, dist2, n2);
}
*/

template<class I, class T>
void knn_reduce(T* data, int nd, I* col_ind, int nc, I* row_ind, int nr, T* sdata, int snd, I* scol_ind, int snc, I* srow_ind, int snr, int d, int k)
{
	if( snr > 0 )
	{
		sdata[0]=data[0];
		scol_ind[0]=col_ind[0];
		srow_ind[0]=row_ind[0];
	}
	I j=1;
	for(I r=1;r<snr;++r,++j)
	{
		if( r%k==0 ) j+=d;
		sdata[r]=data[j];
		scol_ind[r]=col_ind[j];
		srow_ind[r]=row_ind[j];
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
I knn_mutual(T* data, int nd, I* col_ind, int nc, I* row_ind, int nr, int k)
{
	I j=0;
	for(I r=0;r<nr;++r)
	{
		if( long(col_ind[r]) > long(row_ind[r]) )
		{
			I c = col_ind[r];
			if( long(c) < 0 ) c = -c;
			I* mc = find_mutual(col_ind+c*k, col_ind+(c+1)*k, row_ind[r]);
			if( mc != 0 )
			{
				(*mc) = -((*mc)+1);
				data[j] = data[r];
				col_ind[j] = col_ind[r];
				row_ind[j] = row_ind[r];
				++j;
			}
		}
		else if( long(col_ind[r]) < long(row_ind[r]) )
		{
			if( long(col_ind[r]) < 0 )
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
void push_to_heap(T* dist2, int n, int m, T* data, int nd, I* col_ind, int nc, int offset, int k)
{
	typedef std::pair<T,I> index_dist;
	typedef std::vector< index_dist > index_vector;
#ifdef _OPENMP
	index_vector vheap(omp_get_max_threads()*k);
#else
	index_vector vheap(k);
#endif

	unsigned long m1=m;

#	if defined(_OPENMP)
#	pragma omp parallel for
#	endif
	for(int r=0;r<n;++r)
	{
		unsigned long rm = r*m;
		unsigned long rk = r*k;
#ifdef _OPENMP
		typename index_vector::iterator hbeg = vheap.begin()+omp_get_thread_num()*k, hcur=hbeg, hend=hbeg+k;
#else
		typename index_vector::iterator hbeg = vheap.begin(), hcur=hbeg, hend=hbeg+k;
#endif
		T* data_rk = data+rk;
		I* col_rk = col_ind+rk;
		//fprintf(stderr, "r: %d | data: %p - col: %p - hbeg: %p - dist2: %p\n", r, data_rk, col_rk, &(*hbeg), dist2);
		unsigned long c=0;
		if( offset > 0 )
		{
			for(unsigned long l=std::min(k, offset);c<l;++c, ++hcur) *hcur = index_dist(data_rk[c], col_rk[c]);
		}
		for(c=0;hcur != hend && c<m1;++c, ++hcur) *hcur = index_dist(dist2[rm+c], offset+c);
		assert(c==m || hcur == hend);
		if( hcur == hend ) std::make_heap(hbeg, hend);
		for(;c<m1;++c)
		{
			T d = dist2[rm+c];
			if( d < hbeg->first )
			{
				*hbeg = index_dist(d, offset+c);
				std::make_heap(hbeg, hend);
			}
		}
		hcur = hbeg;
		for(c=0;c<k;++c, ++hcur)
		{
			data_rk[c] = hcur->first;
			col_rk[c] = hcur->second;
		}
	}
}

template<class I, class T>
void finalize_heap(T* data, int nd, I* col_ind, int nc, int k)
{
	typedef std::pair<T,I> index_dist;
	typedef std::vector< index_dist > index_vector;
#ifdef _OPENMP
	index_vector vheap(omp_get_max_threads()*k);
#else
	index_vector vheap(k);
#endif

	I e = I(T(nd)/k);

#	if defined(_OPENMP)
#	pragma omp parallel for
#	endif
	for(int r=0;r<e;++r)
	{
#ifdef _OPENMP
		typename index_vector::iterator hbeg = vheap.begin()+omp_get_thread_num()*k, hcur=hbeg;
#else
		typename index_vector::iterator hbeg = vheap.begin(), hcur=hbeg;
#endif
		T* data_rk = data+r*k;
		I* col_rk = col_ind+r*k;
		for(int c=0;c<k;++c, ++hcur) *hcur = index_dist(data_rk[c], col_rk[c]);
		std::sort_heap(hbeg, hbeg+k);
		hcur = hbeg;
		for(int c=0;c<k;++c, ++hcur)
		{
			data_rk[c] = hcur->first;
			col_rk[c] = hcur->second;
		}
	}
}

template<class I, class T>
I select_subset_csr(T* data, int nd, I* col_ind, int nc, I* row_ptr, int nr, I* selected, int scnt)
{
	nr-=1;
	I cnt = 0, rc=1;
	I* index_map = new I[nr];
	for(I i=0;i<nr;++i) index_map[i]=-1;
	for(I i=0;i<scnt;++i) index_map[selected[i]]=i;

	for(I s = 0;s<scnt;++s)
	{
		I r = selected[s];
		for(int j=row_ptr[r];j<row_ptr[r+1];++j)
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

template<class I, class T>
void self_tuning_gaussian_kernel_csr(T* sdist, int ns, T* data, int nd, I* col_ind, int nc, I* row_ptr, int nr)
{
	nr-=1;
	T* ndist = new T[nr];
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(int i=0;i<nr;i++)
	{
		ndist[i] = 0;
	}
	for(int i=0;i<nr;i++)
	{
		if ( ndist[col_ind[i]] < data[i] )
			ndist[col_ind[i]] = data[i];
	}
	I* row_ind = new I[nc];
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(int r=0;r<nr;++r)
	{
		for(int j=row_ptr[r];j<row_ptr[r+1];++j)
		{
			row_ind[j]=r;
		}
	}
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(int i=0;i<nc;i++)
	{
		double den = 1.0;
		if( ndist[row_ind[i]] != 0.0 ) den *= std::sqrt(double(ndist[row_ind[i]]));
		if( ndist[col_ind[i]] != 0.0 ) den *= std::sqrt(double(ndist[col_ind[i]]));
		if( den != 0.0 ) sdist[i] = std::exp( -data[i] / T(den) );
		else sdist[i] = std::exp( -data[i] );
	}
	delete[] ndist, row_ind;
}

template<class I, class T>
void normalize_csr(T* sdist, int ns, T* data, int nd, I* col_ind, int nc, I* row_ptr, int nr)
{
	nr-=1;
	T* ndist = new T[nr];
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(int i=0;i<nr;i++)
	{
		ndist[i] = 0;
	}
	for(int i=0;i<nr;i++)
	{
		ndist[col_ind[i]] += data[i];
	}
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(int i=0;i<nr;i++)
	{
		if( ndist[i] == 0.0 ) ndist[i] = 1.0;
		else ndist[i] = T(1.0) / ndist[i];
	}
	I* row_ind = new I[nc];
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(int r=0;r<nr;++r)
	{
		for(int j=row_ptr[r];j<row_ptr[r+1];++j) row_ind[j]=r;
	}
#	ifdef _OPENMP
#	pragma omp parallel for
#	endif
	for(int i=0;i<nc;i++)
	{
		sdist[i] = data[i]*ndist[row_ind[i]]*ndist[col_ind[i]];
	}
	delete[] ndist, row_ind;
}
