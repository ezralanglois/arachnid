%define DOCSTRING
"This C/C++ Python extension defines an optimized set of utilities for fast matrix
operations.
"
%enddef

%module fastdot

/* why does SWIG complain about int arrays? a typecheck is provided */
#pragma SWIG nowarn=467

%{
#define SWIG_FILE_WITH_INIT
#include "Python.h"
#include "numpy/arrayobject.h"
#include <complex>
extern "C" {
#include "cblas.h"
}

#ifdef _OPENMP
#include <omp.h>
#endif
#include "fastdot.hpp"
%}

%include "fastdot.hpp"
%include <std_complex.i>

%feature("autodoc", "0");

%include "numpy.i"

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
typedef long size_type;
%}


%numpy_typemaps( std::complex<float> , NPY_CFLOAT , int)
%numpy_typemaps( std::complex<double> , NPY_CDOUBLE, int)


/** Declare the numpy array data types
 */
 
%define DECLARE_DATA_TYPE2( dtype, itype )
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* samp1, int n1, int m1)};
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* samp2, int n2, int m2)};
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* distm, int n3, int m3)};
%enddef


%apply (std::complex<float>* INPLACE_ARRAY2, int DIM1, int DIM2) {(std::complex<float>* samp1, int n1, int m1)};
%apply (std::complex<float>* INPLACE_ARRAY2, int DIM1, int DIM2) {(std::complex<float>* samp2, int n2, int m2)};
%apply (std::complex<float>* INPLACE_ARRAY2, int DIM1, int DIM2) {(std::complex<float>* distm, int n3, int m3)};

%apply (std::complex<double>* INPLACE_ARRAY2, int DIM1, int DIM2) {(std::complex<double>* samp1, int n1, int m1)};
%apply (std::complex<double>* INPLACE_ARRAY2, int DIM1, int DIM2) {(std::complex<double>* samp2, int n2, int m2)};
%apply (std::complex<double>* INPLACE_ARRAY2, int DIM1, int DIM2) {(std::complex<double>* distm, int n3, int m3)};


%define DECLARE_DATA_TYPE( itype )
DECLARE_DATA_TYPE2(int, itype)
DECLARE_DATA_TYPE2(float, itype)
DECLARE_DATA_TYPE2(double, itype)
%enddef

DECLARE_DATA_TYPE(int)
DECLARE_DATA_TYPE(long)
DECLARE_DATA_TYPE(long long)
DECLARE_DATA_TYPE(unsigned int)
DECLARE_DATA_TYPE(unsigned long)

%define INSTANTIATE_DATA( f_name )
%template(f_name)   f_name<float>;
%template(f_name)   f_name<double>;
%enddef


%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function selects a subset of rows 
		(and columns) from a CSR sparse matrix.

		:Parameters:

		samp1 : array
			   In/out 1D array of values
		samp2 :array
			   In/out 1D array of values
		dist2 : array
			   	Output matrix
		alpha : float
				Value to mulitply by result
		beta : float
			   Value to add to result
		");
INSTANTIATE_DATA(gemm)

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function selects a subset of rows 
		(and columns) from a CSR sparse matrix.

		:Parameters:

		samp1 : array
			   In/out 1D array of values
		samp2 :array
			   In/out 1D array of values
		dist2 : array
			   	Output matrix
		alpha : float
				Value to mulitply by result
		beta : float
			   Value to add to result
		");
INSTANTIATE_DATA(gemm_t1)



