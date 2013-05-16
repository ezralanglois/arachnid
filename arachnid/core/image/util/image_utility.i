%define DOCSTRING
"This C/C++ Python extension defines an optimized set of utilities for images.
"
%enddef

%module image_utility

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
extern "C" {
void radon(double *pPtr, double *iPtr, double *thetaPtr, double *xCosTable, double *ySinTable, int M, int N,
           int xOrigin, int yOrigin, int numAngles, int rFirst, int rSize);
           }
#include "image_utility.h"
%}

%include "image_utility.h"

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


/** Declare the numpy array data types 
 */
%define DECLARE_DATA_TYPE2( dtype, itype )
%apply (dtype* INPLACE_ARRAY1, int DIM1) {(dtype* avg, int na)};
%apply (dtype* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(dtype* out, int onx, int ony, int onz)};
%apply (itype* INPLACE_ARRAY1, int DIM1) {(itype* Dr, int nr)};
%apply (itype* INPLACE_ARRAY1, int DIM1) {(itype* Dc, int nc)};
%apply (dtype* INPLACE_ARRAY1, int DIM1) {(dtype* sdist, int ns)};
%enddef


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
DECLARE_DATA_TYPE(unsigned long long)


/** Create a set of concrete functions from the templates
 */
%define INSTANTIATE_ALL_DATA( f_name, itype )
//%template(f_name)   f_name<itype,short>;
//%template(f_name)   f_name<itype,int>;
%template(f_name)   f_name<itype,float>;
%template(f_name)   f_name<itype,double>;
%template(f_name)   f_name<itype,long double>; // bug in version 11 of pgi compiler fixed in 11.6
%enddef

%define INSTANTIATE_ALL( f_name )
INSTANTIATE_ALL_DATA(f_name, int)
INSTANTIATE_ALL_DATA(f_name, long)
INSTANTIATE_ALL_DATA(f_name, long long)
INSTANTIATE_ALL_DATA(f_name, unsigned int)
INSTANTIATE_ALL_DATA(f_name, unsigned long)
INSTANTIATE_ALL_DATA(f_name, unsigned long long)
%enddef

%define INSTANTIATE_DATA( f_name )
//%template(f_name)   f_name<int>;
%template(f_name)   f_name<float>;
%template(f_name)   f_name<double>;
%template(f_name)   f_name<long double>;
%enddef

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function creates a 2D rotational average from a 1D average.

		:Parameters:

		out : array
			  Output 3D rotational average
		avg :array
			 1D array rotational average
		rmax : int
			   Maximum radius
		");
INSTANTIATE_DATA(rotavg)

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function creates a 2D rotational average from a 1D average.
		");
INSTANTIATE_ALL(radon_transform)

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function creates a 2D rotational average from a 1D average.
		");
int radon_count(int nang, int irow, int icol);


