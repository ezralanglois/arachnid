
template<class T>
void rotate_avg(float *ref, int nn, int rnr, int rnc, T *psi, int pn, float* avg, int an)
{
	typedef float float_type;
	int n = rnr*rnc;
	float_type* buf = new float_type[n];
	float_type scale = 1.0, tx = 0.0, ty=0.0;
	int ret=0;

	for(int j=0;j<n;++j) avg[j] = 0;
	for(int i=0;i<nn;++i)
	{
		float_type psi1=float_type(psi[i]);
		rtsq_(ref+i*n, buf, &rnr, &rnc, &rnr, &rnc, &psi1, &scale, &tx, &ty, &ret);
		for(int j=0;j<n;++j) avg[j] += buf[j];
	}
	for(int j=0;j<n;++j) avg[j] /= float_type(nn);

	delete[] buf;
}

template<class I, class T>
T rotate_error_mask(float *ref, int nn, int rnr, int rnc, T *psi, int pn, I* maskidx, int mn)
{
	typedef float float_type;

	int n = rnr*rnc;
	//fprintf(stderr, "here1: %d*%d=%d\n", rnr, rnc, n);
	float_type* avg = new float_type[n];
	float_type* buf = new float_type[n];
	float_type scale = 1.0, tx = 0.0, ty=0.0;
	int ret=0;

	for(int j=0;j<n;++j) avg[j] = 0;
	for(int i=0;i<nn;++i)
	{
		float_type psi1=float_type(psi[i]);
		rtsq_(ref+i*n, buf, &rnr, &rnc, &rnr, &rnc, &psi1, &scale, &tx, &ty, &ret);
		for(int j=0;j<n;++j) avg[j] += buf[j];
	}
	//fprintf(stderr, "done1\n");
	for(int j=0;j<n;++j) avg[j] /= float_type(nn);

	T err = 0.0;
	for(int i=0;i<nn;++i)
	{
		float_type psi1=float_type(psi[i]);
		rtsq_(ref+i*n, buf, &rnr, &rnc, &rnr, &rnc, &psi1, &scale, &tx, &ty, &ret);
		for(int j=0;j<mn;++j)
		{
			I k=maskidx[j];
			T v = buf[k]-avg[k];
			err += v*v;
		}
	}

	delete[] avg;
	delete[] buf;
	return err/(n-1);
}

template<class T>
T rotate_error(float *ref, int nn, int rnr, int rnc, T *psi, int pn)
{
	typedef float float_type;

	int n = rnr*rnc;
	float_type* avg = new float_type[n];
	float_type* buf = new float_type[n];
	float_type scale = 1.0, tx = 0.0, ty=0.0;
	int ret=0;

	for(int j=0;j<n;++j) avg[j] = 0;
	for(int i=0;i<nn;++i)
	{
		float_type psi1=float_type(psi[i]);
		rtsq_(ref+i*n, buf, &rnr, &rnc, &rnr, &rnc, &psi1, &scale, &tx, &ty, &ret);
		for(int j=0;j<n;++j) avg[j] += buf[j];
	}
	for(int j=0;j<n;++j) avg[j] /= float_type(nn);

	T err = 0.0;
	for(int i=0;i<nn;++i)
	{
		float_type psi1=float_type(psi[i]);
		rtsq_(ref+i*n, buf, &rnr, &rnc, &rnr, &rnc, &psi1, &scale, &tx, &ty, &ret);
		for(int j=0;j<n;++j)
		{
			T v = buf[j]-avg[j];
			err += v*v;
		}
	}

	delete[] avg;
	delete[] buf;
	return err/(n-1);
}

template<class I, class T>
void rotate_distance_mask(float *ref, int nn, int rnr, int rnc, I* Dr, int rn, I* Dc, int cn, T *psi, int pn, T* dist, int dn, I* maskidx, int mn)
{
	typedef float float_type;
	/*
	extern"C" {
	void rtsq_(T*, T*, I*, I*, I*, I*, T*, T*, T*, T*);
	}*/
	int n = rnr*rnc;
#ifdef _OPENMP
	float_type* tbuf = new float_type[omp_get_max_threads()*n];
#else
	float_type* buf = new float_type[n];
	float_type* tbuf=buf;
#endif
	float_type scale = 1.0, tx = 0.0, ty=0.0;
	int ret=0;

#	if defined(_OPENMP)
#	pragma omp parallel for
#	endif
	for(int i=0;i<dn;++i)
	{
#ifdef _OPENMP
		float_type* buf = tbuf+omp_get_thread_num()*n;
#endif
		float_type psi1=float_type(psi[i]);
		float_type* cref = ref+n*Dc[i];
		float_type* rref = ref+n*Dr[i];
		rtsq_(cref, buf, &rnr, &rnc, &rnr, &rnc, &psi1, &scale, &tx, &ty, &ret);
		T d=T(0.0);
		for(int k=0, j;k<mn;++k)
		{
			j=maskidx[k];
			//d += (buf[j]-cref[j])*(buf[j]-cref[j]);
			d += (buf[j]-rref[j])*(buf[j]-rref[j]);
		}
		dist[i] = d;
	}
	delete[] tbuf;
}

template<class I, class T>
void rotate_distance_array_mask(float* img, int nr, int nc, float *ref, int nn, int rnr, int rnc, T *psi, int pn, T* dist, int dn, I* maskidx, int mn)
{
	typedef float float_type;
	/*
	extern"C" {
	void rtsq_(T*, T*, I*, I*, I*, I*, T*, T*, T*, T*);
	}*/
	int n = nr*nc;
#ifdef _OPENMP
	float_type* tbuf = new float_type[omp_get_max_threads()*n];
#else
	float_type* buf = new float_type[n];
	float_type* tbuf=buf;
#endif
	float_type scale = 1.0, tx = 0.0, ty=0.0;
	int ret=0;

#	if defined(_OPENMP)
#	pragma omp parallel for
#	endif
	for(int i=0;i<nn;++i)
	{
#ifdef _OPENMP
		float_type* buf = tbuf+omp_get_thread_num()*n;
#endif
		float_type psi1=float_type(psi[i]);
		float_type* cref = ref+n*i;
		rtsq_(cref, buf, &nr, &nc, &nr, &nc, &psi1, &scale, &tx, &ty, &ret);
		T d=T(0.0);
		for(int k=0, j;k<mn;++k)
		{
			j=maskidx[k];
			//d += (buf[j]-cref[j])*(buf[j]-cref[j]);
			d += (buf[j]-img[j])*(buf[j]-img[j]);
		}
		dist[i] = d;
	}
	delete[] tbuf;
}

template<class T>
void rotate_distance_array(float* img, int nr, int nc, float *ref, int nn, int rnr, int rnc, T *psi, int pn, T* dist, int dn)
{
	typedef float float_type;
	/*
	extern"C" {
	void rtsq_(T*, T*, I*, I*, I*, I*, T*, T*, T*, T*);
	}*/
	int n = nr*nc;
#ifdef _OPENMP
	float_type* tbuf = new float_type[omp_get_max_threads()*n];
#else
	float_type* buf = new float_type[n];
	float_type* tbuf=buf;
#endif
	float_type scale = 1.0, tx = 0.0, ty=0.0;
	int ret=0;

#	if defined(_OPENMP)
#	pragma omp parallel for
#	endif
	for(int i=0;i<nn;++i)
	{
#ifdef _OPENMP
		float_type* buf = tbuf+omp_get_thread_num()*n;
#endif
		float_type psi1=float_type(psi[i]);
		float_type* cref = ref+n*i;
		rtsq_(cref, buf, &nr, &nc, &nr, &nc, &psi1, &scale, &tx, &ty, &ret);
		T d=T(0.0);
		for(int j=0;j<n;++j)
		{
			//d += (buf[j]-cref[j])*(buf[j]-cref[j]);
			d += (buf[j]-img[j])*(buf[j]-img[j]);
		}
		dist[i] = d;
	}
	delete[] tbuf;
}


template<class T>
void rotate_image(T* img, int nr, int nc, T *rimg, int rnr, int rnc, T ang)
{
	T scale = 1.0, tx = 0.0, ty=0.0;
	int ret=0;
	rtsq_(img, rimg, &nr, &nc, &nr, &nc, &ang, &scale, &tx, &ty, &ret);
}
