
typedef long size_type;

size_type index(size_type i, size_type j, size_type k, size_type ni, size_type nj, size_type nk)
{
//	return i*(nj*nk)+j+k*nj;
	return i*(nj*nk)+j*nk+k;
//	return i+j*ni+k*(nj*ni);
}
size_type index(size_type i, size_type j, size_type ni, size_type nj)
{
	return i*nj+j;
//	return i+j*ni;
}

/*
size_type index(size_type i, size_type j, size_type k, size_type nk, size_type njk)
{
	return i*njk+j*nk+k;
}
size_type index(size_type i, size_type j, size_type nj)
{
	return i*nj+j;
}
*/

template<class C, class T>
void oneline_nn(C* fvol, size_type nv1, size_type nv2, T* wvol, C* fpimg, T* rot, size_type j)
{
	C btq;
	size_type jp = (j>=0) ? j : nv2+j;
	T xn, yn, zn;
	size_type xi, yi, zi;
	size_type ayi, azi,nv3=nv2*nv2;

	//debug
	size_type nvol=nv2*nv2*nv1;
	size_type nimg=nv2*nv1;

	for(size_type i=0;i<nv1;++i)
	{
		if( (i*i+j*j)<nv2*nv2/4 && !(i==0 && j<0) )
		{
			xn = i*rot[0]+j*rot[3];
			yn = i*rot[1]+j*rot[4];
			zn = i*rot[2]+j*rot[5];
			size_type idx = index(i, jp, nv1, nv2);
			//idx=i*nv2+jp;
			if(idx>=nimg) fprintf(stderr, "Index out of bounds: %d >= %d\n", idx, nimg);
			assert(idx<nimg);
			if( xn < 0  )
			{
				xn = -xn;
				yn = -yn;
				zn = -zn;
				btq = std::conj(fpimg[idx]);
			}
			else btq = fpimg[idx];

			xi = size_type(xn+0.5+nv2) - nv2;
			yi = size_type(yn+0.5+nv2) - nv2;
			zi = size_type(zn+0.5+nv2) - nv2;

			//azi = (zi>=0) ? zi+1 :nv2+zi+1;
			//ayi = (yi>=0) ? yi+1 : nv2+yi+1;

			azi = (zi>=0) ? zi : nv2+zi;
			ayi = (yi>=0) ? yi : nv2+yi;

			//idx = xi*nv3+ayi*nv2+azi;
			idx = index(xi, ayi, azi, nv1, nv2, nv2);
			if(idx>=nvol) fprintf(stderr, "Index out of bounds2: %d >= %d\n", idx, nvol);
			assert(idx<nvol);
			fvol[idx]+=btq;
			wvol[idx]+=1;
			/*
			if(xi<=nv1 && yi>=-nv1 && yi<=nv1 && zi>=-nv1 && zi<=nv1)
			{
				if(xi>=0)
				{
					azi = (zi>=0) ? zi+1 : nv2+zi+1;
					ayi = (yi>=0) ? yi+1 : nv2+yi+1;
				}
				else
				{
					azi = (zi>0) ? nv2-zi+1 : -zi+1;
					ayi = (yi>0) ? nv2-yi+1 : -yi+1;
					xi=-xi;
					btq=conj(btq);
				}
				idx = xi*nv3+ayi*nv2+azi;
				//idx = xi+ayi*nv1+azi*nv3;
				fvol[idx]+=btq;
				wvol[idx]+=1;
			}
			*/
		}
	}
}

template<class C, class T>
void backproject_nn(C* fvol, size_type nv1, size_type nv2, size_type nv3, T* wvol, size_type nw1, size_type nw2, size_type nw3, C* fpimg, size_type fn1, size_type fn2, T* rot, size_type rn1, size_type rn2)
{
	for(size_type j=0;j<fn2;++j)
	{
		for(size_type i=0;i<fn1;++i)
		{
			if(((i+j)%2) != 0)
			{
				size_type idx = index(i, j, fn1, fn2);
				fpimg[idx] *= -1;
				//fpimg[i*fn2+j] *= -1;
			}
		}
	}
	//add sym
	for(size_type j=-nv1+1;j<=nv1;++j)
		oneline_nn(fvol, nv1, nv2, wvol, fpimg, rot, j);
}

template<class C>
void window(C* fvol, size_type nv1, size_type nv2, size_type nv3, C* rvol, size_type nr1, size_type nr2, size_type nr3)
{
	size_type ns = nv1-1;

	size_type ip = (nv2-ns)/2+ns%2;
	size_type nv=nv3*nv2;
	size_type nvt=nv*nv1;
	size_type ns2 = ns*ns;
	for(size_type x=0;x<ns;++x)
	{
		for(size_type y=0;y<ns;++y)
		{
			for(size_type z=0;z<ns;++z)
			{
				size_type idx1 = index(x, y, z, nv1, nv2, nv3);
				//size_type idx1 = x*ns2+y*ns+z;
				size_type idx2 = index(ip+x, ip+y, ip+z, nv1, nv2, nv3);
				//size_type idx2 = (ip+x)*ns2+(ip+y)*ns+(ip+z);
				assert(idx1<nvt);
				assert(idx1>=0);
				assert(idx2<nvt);
				assert(idx2>=0);
				rvol[idx1]=fvol[idx2];
			}
		}
	}

	size_type l2 = (ns/2)*(ns/2);
	size_type l2p = (ns/2-1)*(ns/2-1);
	ip = ns/2+1;
	C tnr = 0.0;
	size_type m = 0;
	for(size_type x=0;x<ns;++x)
	{
		for(size_type y=0;y<ns;++y)
		{
			for(size_type z=0;z<ns;++z)
			{
				size_type lr = (x-ip)*(x-ip)+(y-ip)*(y-ip)+(z-ip)*(z-ip);
				if(lr<=l2 && lr>=l2p)
				{
					size_type idx1 = index(x, y, z, nv1, nv2, nv3);
					//size_type idx1 = x*nv+y*nv2+z;
					assert(idx1<nv);
					assert(idx1>=0);
					tnr += rvol[idx1];
					m+=1;
				}
			}
		}
	}
	tnr /= m;
	for(size_type x=0;x<ns;++x)
	{
		for(size_type y=0;y<ns;++y)
		{
			for(size_type z=0;z<ns;++z)
			{
				size_type lr = (x-ip)*(x-ip)+(y-ip)*(y-ip)+(z-ip)*(z-ip);
				size_type idx1 = index(x, y, z, nv1, nv2, nv3);
				//size_type idx1 = x*nv+y*nv2+z;
				assert(idx1<nv);
				assert(idx1>=0);
				rvol[idx1] = (lr<=l2) ? rvol[idx1]-tnr : 0.0;
			}
		}
	}
}

template<class C, class T>
void finalize(C* fvol, size_type nv1, size_type nv2, size_type nv3, T* wvol, size_type nw1, size_type nw2, size_type nw3)
{

	//debug
	size_type nvol = nv1*nv2*nv3;
	for(size_type y=1;y<nv1;++y)
	{
		for(size_type z=1;z<nv1;++z)
		{
			size_type idx1 = index(0, y, z, nv1, nv2, nv3);
			size_type idx2 = index(0, nv2-y+1, nv2-z+1, nv1, nv2, nv3);
			//idx1 = y*nv2+z;
			//idx2 =(nv2-y+1)*nv2 + (nv2-z+1);
			if(idx1>=nvol) fprintf(stderr, "Index out of bounds3a: %d >= %d\n", idx1, nvol);
			if(idx2>=nvol) fprintf(stderr, "Index out of bounds4a: %d >= %d\n", idx2, nvol);
			if(idx1<0) fprintf(stderr, "Index out of bounds5a: %d\n", idx1);
			if(idx2<0) fprintf(stderr, "Index out of bounds6a: %d\n", idx2);
			assert(idx1<nvol);
			assert(idx2<nvol);
			assert(idx1>=0);
			assert(idx2>=0);
			fvol[idx1] += std::conj(fvol[idx2]);
			wvol[idx1] += wvol[idx2];
			fvol[idx2] = std::conj(fvol[idx1]);
			wvol[idx2] = wvol[idx1];

			idx1 = index(0, nv2-y+1, z-1, nv1, nv2, nv3);
			idx2 = index(0, y-1, nv2-z+1, nv1, nv2, nv3);
			if(idx1>=nvol) fprintf(stderr, "Index out of bounds3b: %d >= %d\n", idx1, nvol);
			if(idx2>=nvol) fprintf(stderr, "Index out of bounds4b: %d >= %d\n", idx2, nvol);
			if(idx1<0) fprintf(stderr, "Index out of bounds5b: %d\n", idx1);
			if(idx2<0) fprintf(stderr, "Index out of bounds6b: %d\n", idx2);
			//idx1 =(nv2-y+1)*nv2 + (z-1);
			//idx2 =(y-1)*nv2 + (nv2-z+1);
			assert(idx1<nvol);
			assert(idx2<nvol);
			assert(idx1>=0);
			assert(idx2>=0);
			fvol[idx1]+=std::conj(fvol[idx2]);
			wvol[idx1]+=wvol[idx2];
			fvol[idx2]=std::conj(fvol[idx1]);
			wvol[idx2]=wvol[idx1];
		}
	}
	for(size_type y=1;y<nv1;++y)
	{
		size_type idx1 = index(0, y-1, 0, nv1, nv2, nv3);
		size_type idx2 = index(0, nv2-y+1, 0, nv1, nv2, nv3);
		//idx1=(y-1)*nv2;
		//idx2=(nv2-y+1)*nv2;
		assert(idx1<nvol);
		assert(idx2<nvol);
		assert(idx1>=0);
		assert(idx2>=0);
		fvol[idx1]+=std::conj(fvol[idx2]);
		wvol[idx1]+=wvol[idx2];
		fvol[idx2]=std::conj(fvol[idx1]);
		wvol[idx2]=wvol[idx1];
	}
	for(size_type z=1;z<nv1;++z)
	{
		size_type idx1 = index(0, 0, z-1, nv1, nv2, nv3);
		size_type idx2 = index(0, 0, nv2-z+1, nv1, nv2, nv3);
		//idx1=z-1;
		//idx2=(nv2-z+1);
		assert(idx1<nvol);
		assert(idx2<nvol);
		assert(idx1>=0);
		assert(idx2>=0);
		fvol[idx1]+=std::conj(fvol[idx2]);
		wvol[idx1]+=wvol[idx2];
		fvol[idx2]=std::conj(fvol[idx1]);
		wvol[idx2]=wvol[idx1];
	}
	for(size_type i=0, x=0;x<nv1;++x)
	{
		for(size_type y=0;y<nv2;++y)
		{
			for(size_type z=0;z<nv3;++z,++i)
			{
				if(wvol[i]==0) continue;
				if((x+y+z)%2!=0)
				{
					fvol[i] *= -1.0/wvol[i];
				}
				else
				{
					fvol[i] *= 1.0/wvol[i];
				}
			}
		}
	}
}






















