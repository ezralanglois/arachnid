%define DOCSTRING
"This C/C++ Python extension defines routines for backprojection.
"
%enddef

%module local_reconstruct

/* why does SWIG complain about int arrays? a typecheck is provided */
#pragma SWIG nowarn=467

%{
#define SWIG_FILE_WITH_INIT
#include "Python.h"
#include "numpy/arrayobject.h"
//#include <complex.h>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#endif
#include "back_project.hpp"
%}

%include "back_project.hpp"

%feature("autodoc", "0");

%include "complex.i"
%include "numpy.i"

%init %{
    import_array();
%}

%numpy_typemaps( std::complex<float> , NPY_CFLOAT , int)
%numpy_typemaps( std::complex<double> , NPY_CDOUBLE, int)
//%numpy_typemaps(complex double, NPY_CDOUBLE, int)
//%numpy_typemaps(complex long double, NPY_CLONGDOUBLE, int)

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
typedef long size_type;
%}




/** Declare the numpy array data types
 */
 
%define DECLARE_DATA_TYPE2( dtype, itype )
%apply (itype* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(itype* fvol, size_type nv1, size_type nv2, size_type nv3)};
%apply (itype* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(itype* rvol, size_type nr1, size_type nr2, size_type nr3)};
%apply (dtype* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(dtype* wvol, size_type nw1, size_type nw2, size_type nw3)};
%apply (itype* INPLACE_ARRAY2, int DIM1, int DIM2) {(itype* fpimg, size_type fn1, size_type fn2)};
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* rot, size_type rn1, size_type rn2)};
%enddef


%define DECLARE_DATA_TYPE( itype )
DECLARE_DATA_TYPE2(int, itype)
DECLARE_DATA_TYPE2(float, itype)
DECLARE_DATA_TYPE2(double, itype)
%enddef

DECLARE_DATA_TYPE(std::complex<float> )
DECLARE_DATA_TYPE(std::complex<double> )
 

/** Create a set of concrete functions from the templates
 */
%define INSTANTIATE_ALL_DATA( f_name, itype )
%template(f_name)   f_name<itype,float>;
%template(f_name)   f_name<itype,double>;
%template(f_name)   f_name<itype,long double>; // bug in version 11 of pgi compiler fixed in 11.6
%enddef

%define INSTANTIATE_ALL( f_name )
INSTANTIATE_ALL_DATA(f_name, std::complex<float> )
INSTANTIATE_ALL_DATA(f_name, std::complex<double> )
%enddef

%define INSTANTIATE_VOL( f_name )
%template(f_name)   f_name< float >;
%template(f_name)   f_name< double >;
%template(f_name)   f_name< std::complex<float> >;
%template(f_name)   f_name< std::complex<double> >;
%enddef

%feature("autodoc", "");
%feature("docstring",
		" Backproject using nearest neighbor interpolation

		:Parameters:

		fvol : complex array
			   Fourier volume
		wvol :array
			  Weight volume
		fpimg : array
			   	Fourier projection
		rot : array
			  Rotation matrix
		");
INSTANTIATE_ALL(backproject_nn)


%feature("autodoc", "");
%feature("docstring",
		" Backproject using nearest neighbor interpolation

		:Parameters:

		fvol : complex array
			   Fourier volume
		wvol :array
			  Weight volume
		");
INSTANTIATE_ALL(finalize)

%feature("autodoc", "");
%feature("docstring",
		" Window out the volume from the padded space

		:Parameters:

		fvol : complex array
			   Fourier volume
		rvol :array
			  Real volume
		");
INSTANTIATE_VOL(window)

