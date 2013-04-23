%define DOCSTRING
"This C/C++ Python extension defines an optimized set of utilities for manifold-learning.
"
%enddef

%module rotation_mapping

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
#include "linalg.hpp"
#include "rotation_mapping.hpp"
%}

%include "rotation_mapping.hpp"

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


/** Declare the numpy array data types
 */
 
%define DECLARE_DATA_TYPE2( dtype, itype )
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* eigs, size_type erow, size_type ecol)};
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* rots, size_type rn, size_type cn)};
%apply (dtype* INPLACE_ARRAY1, int DIM1) {(dtype* map, size_type msize)};
%apply (dtype* INPLACE_ARRAY1, int DIM1) {(dtype* cost, size_type csize)};
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
//DECLARE_DATA_TYPE(unsigned long long)


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
INSTANTIATE_ALL_DATA(f_name, short)
INSTANTIATE_ALL_DATA(f_name, int)
INSTANTIATE_ALL_DATA(f_name, long)
INSTANTIATE_ALL_DATA(f_name, long long)
//INSTANTIATE_ALL_DATA(f_name, unsigned int)
//INSTANTIATE_ALL_DATA(f_name, unsigned long)
//INSTANTIATE_ALL_DATA(f_name, unsigned long long)
%enddef

%define INSTANTIATE_DATA( f_name )
//%template(f_name)   f_name<int>;
%template(f_name)   f_name<float>;
%template(f_name)   f_name<double>;
%template(f_name)   f_name<long double>;
%enddef

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function defines a cost function to map a manifold to a rotation.

		:Parameters:

		map : array
			   In/out 1D array of values (81 =9x9)
		cost :array
			   Output 1D array of values (nx1)
		eigs : array
			   	2D array of manifold values (nx9)
		");
INSTANTIATE_DATA(rotation_cost_function)

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function defines a cost function to map a manifold to a rotation.

		:Parameters:

		eigs : array
			   	2D array of manifold values (nx9)
		map : array
			   In/out 1D array of values (81 =9x9)
		out :array
			   Output 2D array of rotations (nx9)
		");
INSTANTIATE_DATA(map_rotations)

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function defines a cost function to map a manifold to a rotation.

		:Parameters:

		eigs : array
			   	2D array of manifold values (nx9)
		map : array
			   In/out 1D array of values (81 =9x9)
		out :array
			   Output 2D array of rotations (nx9)
		");
INSTANTIATE_DATA(map_orthogonal_rotations)

