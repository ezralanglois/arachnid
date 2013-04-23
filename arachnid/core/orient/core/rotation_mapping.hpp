typedef long size_type;



template<class T>
void rotation_cost_function(T* map, size_type msize, T* cost, size_type csize, T* eigs, size_type erow, size_type ecol)
{

#ifdef _OPENMP
#	pragma omp parallel for
#endif
	for(size_type i=0;i<csize;++i)
	{
		T* ev = eigs+i*ecol;
		T* mf = map;
		T r11 = dot(mf, ev, ecol); mf += ecol;
		T r12 = dot(mf, ev, ecol); mf += ecol;
		T r13 = dot(mf, ev, ecol); mf += ecol;
		T r21 = dot(mf, ev, ecol); mf += ecol;
		T r22 = dot(mf, ev, ecol); mf += ecol;
		T r23 = dot(mf, ev, ecol); mf += ecol;
		T r31 = dot(mf, ev, ecol); mf += ecol;
		T r32 = dot(mf, ev, ecol); mf += ecol;
		T r33 = dot(mf, ev, ecol);
		T det = r11*r22*r33 + r12*r23*r31 + r13*r21*r32 - r11*r23*r32 - r12*r21*r33 - r13*r22*r31 - 1;
		det *= det;
		T sum = 0.0f, val;


	    //r11 r12 r13     r11 r21 r31	r11 r12 r13
	    //r21 r22 r23     r12 r22 r32	r21 r22 r23
	    //r31 r32 r33     r13 r23 r33	r31 r32 r33

		val = r11*r11 + r21*r21 + r31*r31 - 1; val *= val; sum += val;//r11
		val = r11*r12 + r21*r22 + r31*r32 - 0; val *= val; sum += val;//r12
		val = r11*r13 + r21*r23 + r31*r33 - 0; val *= val; sum += val;//r13
		val = r12*r11 + r22*r21 + r32*r31 - 0; val *= val; sum += val;//r21
		val = r12*r12 + r22*r22 + r32*r32 - 1; val *= val; sum += val;//r22
		val = r12*r13 + r22*r23 + r32*r33 - 0; val *= val; sum += val;//r23
		val = r13*r11 + r23*r21 + r33*r31 - 0; val *= val; sum += val;//r31
		val = r13*r12 + r23*r22 + r33*r32 - 0; val *= val; sum += val;//r32
		val = r13*r13 + r23*r23 + r33*r33 - 1; val *= val; sum += val;//r33
		cost[i] = std::sqrt(sum + det);//sum+det;//
	}
}

template<class T>
void map_rotation(T* eigen, size_type esize, T* map, size_type msize, T* rot, size_type rn)
{
	for(int j=0;j<9;++j, map+=esize)
	{
		rot[j] = dot(map, eigen, esize);
	}
}

template<class T>
int map_orthogonal_rotation(T* eigen, size_type esize, T* map, size_type msize, T* rot, size_type rn, int iter, float eps)
{
	map_rotation(eigen, esize, map, msize, rot, rn);
	return polar_decomposition3x3(rot, rot, iter, eps);
}

template<class T>
void map_rotations(T* eigs, size_type erow, size_type ecol, T* map, size_type msize, T* rots, size_type rn, size_type cn)
{
#ifdef _OPENMP
#	pragma omp parallel for
#endif
	for(size_type i=0;i<rn;++i)
	{
		map_rotation(eigs+ecol*i, ecol, map, msize, rots+cn*i, cn);
	}
}

template<class T>
void map_orthogonal_rotations(T* eigs, size_type erow, size_type ecol, T* map, size_type msize, T* rots, size_type rn, size_type cn, int iter, float eps)
{
#ifdef _OPENMP
#	pragma omp parallel for
#endif
	for(size_type i=0;i<rn;++i)
	{
		map_orthogonal_rotation(eigs+ecol*i, ecol, map, msize, rots+cn*i, cn, iter, eps);
	}
}

