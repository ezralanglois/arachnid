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
#ifdef _OPENMP
#include <omp.h>
#endif

#include "resample.hpp"
%}

%include "resample.hpp"

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
typedef long dsize_type;
%}


/** Declare the numpy array data types 
 */
%define DECLARE_DATA_TYPE( dtype )
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* img, dsize_type nx, dsize_type ny)};
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* out, dsize_type ox, dsize_type oy)};
%apply (dtype* INPLACE_ARRAY1, int DIM1) {(dtype* kernel, dsize_type ksize)};
%enddef


DECLARE_DATA_TYPE(float)
DECLARE_DATA_TYPE(double)
DECLARE_DATA_TYPE(long double)


%define INSTANTIATE_DATA( f_name )
%template(f_name)   f_name<float>;
%template(f_name)   f_name<double>;
%template(f_name)   f_name<long double>;
%enddef

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

