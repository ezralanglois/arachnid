void EMData::nn(EMData* wptr, EMData* myfft, const Transform& tf, int mult)
{
	ENTERFUNC;
	int nxc = attr_dict["nxc"]; // # of complex elements along x
	// let's treat nr, bi, and local data as matrices
	vector<int> saved_offsets = get_array_offsets();
	vector<int> myfft_saved_offsets = myfft->get_array_offsets();
	set_array_offsets(0,1,1);
	myfft->set_array_offsets(0,1);
	// loop over frequencies in y
	//for(int i = 0; i <= 2; i++){{for(int l = 0; l <= 2; l++) std::cout<<"  "<<tf[l][i];}std::cout<<std::endl;};std::cout<<std::endl;
	//Dict tt = tf.get_rotation("spider");
	//std::cout << static_cast<float>(tt["phi"]) << " " << static_cast<float>(tt["theta"]) << " " << static_cast<float>(tt["psi"]) << std::endl;
	if( mult == 1 ) {
		for (int iy = -ny/2 + 1; iy <= ny/2; iy++) onelinenn(iy, ny, nxc, wptr, myfft, tf);
	} else {
		for (int iy = -ny/2 + 1; iy <= ny/2; iy++) onelinenn_mult(iy, ny, nxc, wptr, myfft, tf, mult);
        }

        set_array_offsets(saved_offsets);
	myfft->set_array_offsets(myfft_saved_offsets);
	EXITFUNC;
}



void EMData::onelinenn_mult(int j, int n, int n2, EMData* wptr, EMData* bi, const Transform& tf, int mult)
{
        //std::cout<<"   onelinenn  "<<j<<"  "<<n<<"  "<<n2<<"  "<<std::endl;
	int jp = (j >= 0) ? j+1 : n+j+1;
	//for(int i = 0; i <= 1; i++){for(int l = 0; l <= 2; l++){std::cout<<"  "<<tf[i][l]<<"  "<<std::endl;}}
	// loop over x
	for (int i = 0; i <= n2; i++) {
        if (((i*i+j*j) < n*n/4) && !((0 == i) && (j < 0))) {
//        if ( !((0 == i) && (j < 0))) {
			float xnew = i*tf[0][0] + j*tf[1][0];
			float ynew = i*tf[0][1] + j*tf[1][1];
			float znew = i*tf[0][2] + j*tf[1][2];
			std::complex<float> btq;
			if (xnew < 0.) {
				xnew = -xnew;
				ynew = -ynew;
				znew = -znew;
				btq = conj(bi->cmplx(i,jp));
			} else {
				btq = bi->cmplx(i,jp);
			}
			int ixn = int(xnew + 0.5 + n) - n;
			int iyn = int(ynew + 0.5 + n) - n;
			int izn = int(znew + 0.5 + n) - n;


			int iza, iya;
			if (izn >= 0)  iza = izn + 1;
			else	       iza = n + izn + 1;

			if (iyn >= 0) iya = iyn + 1;
			else	      iya = n + iyn + 1;

			cmplx(ixn,iya,iza) += btq*float(mult);
			(*wptr)(ixn,iya,iza)+=float(mult);

			/*if ((ixn <= n2) && (iyn >= -n2) && (iyn <= n2)  && (izn >= -n2) && (izn <= n2)) {
				if (ixn >= 0) {
					int iza, iya;
					if (izn >= 0)  iza = izn + 1;
					else	       iza = n + izn + 1;

					if (iyn >= 0) iya = iyn + 1;
					else	      iya = n + iyn + 1;

					cmplx(ixn,iya,iza) += btq*float(mult);
					(*wptr)(ixn,iya,iza)+=float(mult);
				} else {
					int izt, iyt;
					if (izn > 0) izt = n - izn + 1;
					else	     izt = -izn + 1;

					if (iyn > 0) iyt = n - iyn + 1;
					else	     iyt = -iyn + 1;

					cmplx(-ixn,iyt,izt) += conj(btq)*float(mult);
					(*wptr)(-ixn,iyt,izt)+=float(mult);
				}
			}*/
		}
	}
}


void nn4Reconstructor::setup()
{
	int size = params["size"];
	int npad = params["npad"];


	string symmetry;
	if( params.has_key("symmetry") )  symmetry = params["symmetry"].to_str();
	else                               symmetry = "c1";

	if( params.has_key("ndim") )  m_ndim = params["ndim"];
	else                          m_ndim = 3;

	if( params.has_key( "snr" ) )  m_osnr = 1.0f/float( params["snr"] );
	else                           m_osnr = 0.0;

	setup( symmetry, size, npad );
}

void nn4Reconstructor::setup( const string& symmetry, int size, int npad )
{
	m_weighting = ESTIMATE;
	m_wghta = 0.2f;

	m_symmetry = symmetry;
	m_npad = npad;
	m_nsym = Transform::get_nsym(m_symmetry);

	m_vnx = size;
	m_vny = size;
	m_vnz = (m_ndim==3) ? size : 1;

	m_vnxp = size*npad;
	m_vnyp = size*npad;
	m_vnzp = (m_ndim==3) ? size*npad : 1;

	m_vnxc = m_vnxp/2;
	m_vnyc = m_vnyp/2;
	m_vnzc = (m_ndim==3) ? m_vnzp/2 : 1;

	buildFFTVolume();
	buildNormVolume();

}


void nn4Reconstructor::buildFFTVolume() {
	int offset = 2 - m_vnxp%2;

	m_volume = params["fftvol"];

	if( m_volume->get_xsize() != m_vnxp+offset && m_volume->get_ysize() != m_vnyp && m_volume->get_zsize() != m_vnzp ) {
		m_volume->set_size(m_vnxp+offset,m_vnyp,m_vnzp);
		m_volume->to_zero();
	}
	// ----------------------------------------------------------------
	// Added by Zhengfan Yang on 03/15/07
	// Original author: please check whether my revision is correct and
	// other Reconstructor need similiar revision.
	if ( m_vnxp % 2 == 0 )  m_volume->set_fftodd(0);
	else                    m_volume->set_fftodd(1);
	// ----------------------------------------------------------------

	m_volume->set_nxc(m_vnxp/2);
	m_volume->set_complex(true);
	m_volume->set_ri(true);
	m_volume->set_fftpad(true);
	m_volume->set_attr("npad", m_npad);
	m_volume->set_array_offsets(0,1,1);
}

void nn4Reconstructor::buildNormVolume() {

	m_wptr = params["weight"];

	if( m_wptr->get_xsize() != m_vnxc+1 &&
		m_wptr->get_ysize() != m_vnyp &&
		m_wptr->get_zsize() != m_vnzp ) {
		m_wptr->set_size(m_vnxc+1,m_vnyp,m_vnzp);
		m_wptr->to_zero();
	}
	m_wptr->set_array_offsets(0,1,1);
}


int nn4Reconstructor::insert_slice(const EMData* const slice, const Transform& t, const float) {
	// sanity checks
	if (!slice) {
		LOGERR("try to insert NULL slice");
		return 1;
	}

        int padffted= slice->get_attr_default( "padffted", 0 );
        if( m_ndim==3 ) {
		if ( padffted==0 && (slice->get_xsize()!=slice->get_ysize() || slice->get_xsize()!=m_vnx)  ) {
			// FIXME: Why doesn't this throw an exception?
			LOGERR("Tried to insert a slice that is the wrong size.");
			return 1;
		}
        } else {
		Assert( m_ndim==2 );
		if( slice->get_ysize() !=1 ) {
			LOGERR( "for 2D reconstruction, a line is excepted" );
        		return 1;
		}
	}

	EMData* padfft = NULL;

	if( padffted != 0 ) padfft = new EMData(*slice);
	else                padfft = padfft_slice( slice, t,  m_npad );

	int mult= slice->get_attr_default( "mult", 1 );
	Assert( mult > 0 );

        if( m_ndim==3 ) {
		insert_padfft_slice( padfft, t, mult );
	} else {
		float alpha = padfft->get_attr( "alpha" );
		alpha = alpha/180.0f*M_PI;
		for(int i=0; i < m_vnxc+1; ++i ) {
			float xnew = i*cos(alpha);
			float ynew = -i*sin(alpha);
			float btqr = padfft->get_value_at( 2*i, 0, 0 );
			float btqi = padfft->get_value_at( 2*i+1, 0, 0 );
			if( xnew < 0.0 ) {
				xnew *= -1;
				ynew *= -1;
				btqi *= -1;
			}

			int ixn = int(xnew+0.5+m_vnxp) - m_vnxp;
			int iyn = int(ynew+0.5+m_vnyp) - m_vnyp;

			if(iyn < 0 ) iyn += m_vnyp;

			(*m_volume)( 2*ixn, iyn+1, 1 ) += btqr *float(mult);
			(*m_volume)( 2*ixn+1, iyn+1, 1 ) += btqi * float(mult);
			(*m_wptr)(ixn,iyn+1, 1) += float(mult);
		}

	}
	checked_delete( padfft );
	return 0;
}

int nn4Reconstructor::insert_padfft_slice( EMData* padfft, const Transform& t, int mult )
{
	Assert( padfft != NULL );

	vector<Transform> tsym = t.get_sym_proj(m_symmetry);
	for (unsigned int isym=0; isym < tsym.size(); isym++) {
		m_volume->nn( m_wptr, padfft, tsym[isym], mult);
        }

	/*for (int isym=0; isym < m_nsym; isym++) {
		Transform tsym = t.get_sym(m_symmetry, isym);
		m_volume->nn( m_wptr, padfft, tsym, mult);
        }*/
	return 0;
}


#define  tw(i,j,k)      tw[ i-1 + (j-1+(k-1)*iy)*ix ]
void circumf_rect( EMData* win , int npad)
{
	float *tw = win->get_data();
	//  correct for the fall-off
	//  mask and subtract circumference average
	int ix = win->get_xsize();
	int iy = win->get_ysize();
	int iz = win->get_zsize();

	int IP = ix/2+1;
	int JP = iy/2+1;
	int KP = iz/2+1;

	//  sinc functions tabulated for fall-off
	float* sincx = new float[IP+1];
	float* sincy = new float[JP+1];
	float* sincz = new float[KP+1];

	sincx[0] = 1.0f;
	sincy[0] = 1.0f;
	sincz[0] = 1.0f;

	float cdf = M_PI/float(npad*2*ix);
	for (int i = 1; i <= IP; ++i)  sincx[i] = sin(i*cdf)/(i*cdf);
	cdf = M_PI/float(npad*2*iy);
	for (int i = 1; i <= JP; ++i)  sincy[i] = sin(i*cdf)/(i*cdf);
	cdf = M_PI/float(npad*2*iz);
	for (int i = 1; i <= KP; ++i)  sincz[i] = sin(i*cdf)/(i*cdf);
	for (int k = 1; k <= iz; ++k) {
		int kkp = abs(k-KP);
		for (int j = 1; j <= iy; ++j) {
			cdf = sincy[abs(j- JP)]*sincz[kkp];
			for (int i = 1; i <= ix; ++i)  tw(i,j,k) /= (sincx[abs(i-IP)]*cdf);
		}
	}

	delete[] sincx;
	delete[] sincy;
	delete[] sincz;



	float dxx = 1.0f/float(0.25*ix*ix);
	float dyy = 1.0f/float(0.25*iy*iy);



	float LR2=(float(ix)/2-1)*(float(ix)/2-1)*dxx;

	float  TNR = 0.0f;
	size_t m = 0;
	for (int k = 1; k <= iz; ++k) {
		for (int j = 1; j <= iy; ++j) {
			for (int i = 1; i <= ix; ++i) {
				float LR = (j-JP)*(j-JP)*dyy+(i-IP)*(i-IP)*dxx;
				if (LR<=1.0f && LR >= LR2) {
					TNR += tw(i,j,k);
					++m;
				}
			}
		}
	}

	TNR /=float(m);


	for (int k = 1; k <= iz; ++k) {
		for (int j = 1; j <= iy; ++j) {
			for (int i = 1; i <= ix; ++i) {
				float LR = (j-JP)*(j-JP)*dyy+(i-IP)*(i-IP)*dxx;
				if (LR<=1.0f)  tw(i,j,k)=tw(i,j,k)-TNR;
			 	else 		tw(i,j,k) = 0.0f;
			}
		}
	}

}
#undef tw
