typedef long size_type;

#define EPSILON 8.8817841970012523e-016
const double pi=3.141592653589793238462643383279502884197;

enum{R11, R12, R13, R21, R22, R23, R31, R32, R33};

template<class T>
T det3x3(T* m)
{	// R11      R22     R33														  R11    R32     R23          R21*R12*R33
	//row0[0]*row1[1]*row2[2]												    row0[0]*row2[1]*row1[2]  row1[0]*row0[1]*row2[2]
	return m[R11]*m[R22]*m[R33] + m[R12]*m[R23]*m[R31] + m[R13]*m[R21]*m[R32] - m[R11]*m[R23]*m[R32] - m[R12]*m[R21]*m[R33] - m[R13]*m[R22]*m[R31];
}

template<class T>
T frobenius_norm_squared3x3(T* m)
{
	return m[R11]*m[R11] + m[R12]*m[R12] + m[R13]*m[R13] + m[R21]*m[R21] + m[R22]*m[R22] + m[R23]*m[R23] + m[R31]*m[R31] + m[R32]*m[R32] + m[R33]*m[R33];
}

template<class T>
int inverse_transpose3x3(T* m, T* out)
{
	T det = det3x3(m);
	if( det == 0 ) return 1;
	det = T(1) / det;
	out[0] = (m[R22]*m[R33]-m[R23]*m[R32])*det;
	out[3] = (m[R32]*m[R13]-m[R33]*m[R12])*det;
	out[6] = (m[R12]*m[R23]-m[R13]*m[R22])*det;
	out[1] = (m[R23]*m[R31]-m[R21]*m[R33])*det;
	out[4] = (m[R11]*m[R33]-m[R13]*m[R31])*det;
	out[7] = (m[R21]*m[R13]-m[R23]*m[R11])*det;
	out[2] = (m[R21]*m[R32]-m[R22]*m[R31])*det;
	out[5] = (m[R12]*m[R31]-m[R11]*m[R32])*det;
	out[8] = (m[R11]*m[R22]-m[R12]*m[R21])*det;
	return 0;
}

template<class T>
T dot(T* v1, T* v2, size_type len)
{
	T sum = 0.0f;
	for(T* ve = v1+len;v1 != ve;++v1, ++v2)
		sum += (*v1)*(*v2);
	return sum;
}

template<class T>
int polar_decomposition3x3(T* prot, T* rot, int iter=100, float eps=1e-7)
{
	T pq[9];
	T diff[9];
	for(int i=0;i<9;++i) pq[i] = prot[i];
	for(int i=0;i<iter;++i)
	{
		if(inverse_transpose3x3(pq, rot)) return 1;
		for(int j=0;j<9;++j)
		{
			rot[j] = 0.5*(pq[j] + rot[j]);
			diff[j] = rot[j] - pq[j];
		}
		if( std::sqrt(frobenius_norm_squared3x3(diff)) < eps ) break;
		for(int j=0;j<9;++j) pq[j] = rot[j];
	}
	return 0;
}

