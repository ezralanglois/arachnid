%define DOCSTRING
"This C/C++ Python extension defines an optimized set of utilities for images.
"
%enddef

%module transform

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

#include "transform.hpp"
%}

%include "transform.hpp"

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

%define T_INPLACE_ARRAY2( ctype )
%apply ctype * INPLACE_ARRAY2 {
  ctype Mx [ ]
};
%enddef

/** Declare the numpy array data types 
 */
%define DECLARE_DATA_TYPE( dtype )
//%apply (dtype* INPLACE_ARRAY2[ANY][ANY]) {(dtype* img_int)};
//%apply (dtype* INPLACE_ARRAY2[ANY][ANY]) {(dtype* img_int_sq)};
//%apply dtype * INPLACE_ARRAY2 {dtype img_int [] };
//%apply dtype * INPLACE_ARRAY2 {dtype img_int_sq [] };
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* img_int, dsize_type irow, dsize_type icol)};
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* img_int_sq, dsize_type i2row, dsize_type i2col)};
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* ccmap, dsize_type row, dsize_type col)};
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
		
		.. note::
			
			Adopted from scikit-image

		:Parameters:

		img : array
			  Integral image as a 2D array
    	r0, c0 : int
        		 Top-left corner of block to be summed.
    	r1, c1 : int
        		 Bottom-right corner of block to be summed.

    	:Returns:
		
    	sum : int
        	  Sum over the given window.
		");
INSTANTIATE_DATA(integrate);

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function normalizes a cross-correlation
		map using summed area tables aka integtral image.
		
		.. note::
			
			Adopted from scikit-image

		:Parameters:

		ccmap : array
			 	2D array cross-correlation map
		img_int : array
				  Integral image
		img_int_sq : array
			   		 Integral image squared
		trow, tcol : int
					 Rows and columns of the template
		ref_ssd : float
				  Sum of squared for reference image
		inv_area : float
				   Inverse of the area for the reference image
		");
INSTANTIATE_DATA(normalize_correlation);

