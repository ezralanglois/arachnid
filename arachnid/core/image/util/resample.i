%define DOCSTRING
"This C/C++ Python extension defines an optimized set of utilities for images.
"
%enddef

%module resample

/* why does SWIG complain about int arrays? a typecheck is provided */
#pragma SWIG nowarn=467

%{
#define SWIG_FILE_WITH_INIT
#include "Python.h"
#include "numpy/arrayobject.h"
#include <limits>
#include <cmath>
#include <complex>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "resample.hpp"
%}

%include "resample.hpp"

%feature("autodoc", "0");

%include "numpy.i"
%include <std_complex.i>

%init %{
    import_array();
%}

%exception {
    Py_BEGIN_ALLOW_THREADS
    try {
    $action
    } catch(...) {
    PyEval_RestoreThread(_save);
    PyErr_SetString(PyExc_StandardError,"Unknown exception thrown");
    return NULL;
    }
    Py_END_ALLOW_THREADS
}

%inline %{
typedef long dsize_type;
%}

%numpy_typemaps( std::complex<float> , NPY_CFLOAT , int)
%numpy_typemaps( std::complex<double> , NPY_CDOUBLE, int)


/** Declare the numpy array data types 
 */
%define DECLARE_DATA_TYPE( dtype )
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* img, dsize_type irow, dsize_type icol)};
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* out, dsize_type orow, dsize_type ocol)};
%apply (dtype* INPLACE_ARRAY1, int DIM1) {(dtype* kernel, dsize_type ksize)};
%enddef


%apply (std::complex<float>* INPLACE_ARRAY2, int DIM1, int DIM2) {(std::complex<float>* img, dsize_type img_r, dsize_type img_c)};
%apply (std::complex<float>* INPLACE_ARRAY2, int DIM1, int DIM2) {(std::complex<float>* out, dsize_type out_r, dsize_type out_c)};
%apply (std::complex<double>* INPLACE_ARRAY2, int DIM1, int DIM2) {(std::complex<double>* img, dsize_type img_r, dsize_type img_c)};
%apply (std::complex<double>* INPLACE_ARRAY2, int DIM1, int DIM2) {(std::complex<double>* out, dsize_type out_r, dsize_type out_c)};
%apply (std::complex<double>* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(std::complex<double>* vol, dsize_type vol_r, dsize_type vol_c, dsize_type vol_d)};
%apply (std::complex<double>* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(std::complex<double>* vout, dsize_type vout_r, dsize_type vout_c, dsize_type vout_d)};


DECLARE_DATA_TYPE(float)
DECLARE_DATA_TYPE(double)
DECLARE_DATA_TYPE(long double)


%define INSTANTIATE_DATA( f_name )
%template(f_name)   f_name<float>;
%template(f_name)   f_name<double>;
%template(f_name)   f_name<long double>;
%enddef

%define INSTANTIATE_DATA_MORE( f_name )
%template(f_name)   f_name<float>;
%template(f_name)   f_name<double>;
%template(f_name)   f_name<long double>;
%template(f_name)   f_name< std::complex<float> >;
%template(f_name)   f_name< std::complex<double> >;
%template(f_name)   f_name< std::complex<long double> >;
%enddef

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function resample an image from
		the Fourier Transform.
		
		Assumes the Fourier transform is shifted to the center
		using fftshift.

		:Parameters:

		img : array
			  Input 2D complex array
		out : array
			  Resampled output 2D complex array
		");
INSTANTIATE_DATA_MORE(resample_fft_center);

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function downsamples an image
		
		Adopted from SPARX

		:Parameters:

		img : array
			  Input 2D array
		out : array
			  Downsampled output 2D array
		kernel : array
			 	 Filter kernel
		");
INSTANTIATE_DATA(downsample);

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function creates a 2D kernel for filtering an image
		
		Adopted from SPARX

		:Parameters:

		kernel : array
			 	 Filter kernel
		m : int
			Kernel size
		freq : float
			   Frequency response
		");
INSTANTIATE_DATA(sinc_blackman_kernel);

