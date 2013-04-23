%define DOCSTRING
"This C/C++ Python extension defines an optimized set of utilities for manifold-learning.
"
%enddef

%module manifold

/* why does SWIG complain about int arrays? a typecheck is provided */
#pragma SWIG nowarn=467

%{
#define SWIG_FILE_WITH_INIT
#include "Python.h"
#include "numpy/arrayobject.h"
#include <limits>
#include <cmath>
#include <vector>
#include <algorithm>
#define USE_BLAS
#ifdef USE_BLAS
extern "C" {
#include "cblas.h"
}
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include "manifold.hpp"
%}

%include "manifold.hpp"

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
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* dist2, size_type n, size_type m)};
%apply (dtype* INPLACE_ARRAY1, int DIM1) {(dtype* data, size_type nd)};
%apply (dtype* INPLACE_ARRAY1, int DIM1) {(dtype* sdata, size_type snd)};
%apply (dtype* INPLACE_ARRAY1, int DIM1) {(dtype* cdata, size_type cnd)};
%apply (dtype* INPLACE_ARRAY1, int DIM1) {(dtype* sdist, size_type ns)};
%apply (itype* INPLACE_ARRAY1, int DIM1) {(itype* col_ind, size_type nc)};
%apply (itype* INPLACE_ARRAY1, int DIM1) {(itype* row_ptr, size_type nr)};
%apply (itype* INPLACE_ARRAY1, int DIM1) {(itype* row_ind, size_type nr)};
%apply (itype* INPLACE_ARRAY1, int DIM1) {(itype* srow_ind, size_type snr)};
%apply (itype* INPLACE_ARRAY1, int DIM1) {(itype* scol_ind, size_type snc)};
%apply (itype* INPLACE_ARRAY1, int DIM1) {(itype* selected, size_type scnt)};
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* samp1, int n1, int m1)};
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* samp2, int n2, int m2)};
%apply (dtype* INPLACE_ARRAY2, int DIM1, int DIM2) {(dtype* distm, int n3, int m3)};
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

%define INSTANTIATE_DATA2( f_name )
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
INSTANTIATE_DATA2(gemm)

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function selects a subset of rows 
		(and columns) from a CSR sparse matrix.

		:Parameters:

		data : array
			   In/out 1D array of distances
		col_ind :array
			 	 In/out 1D array column indicies
		row_ptr : array
			   	  In/out 1D array row pointers
		selected : array
				   Input 1D array of selected rows
		
		:Returns:
		
		n : int
			Number of sparse elements
		");
INSTANTIATE_ALL(select_subset_csr)

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function calculates a self-tuning gaussin kernel over
		a sparse matrix in CSR format.

		:Parameters:

		sdist : array
			    Output 1D array of distances
		data : array
			   Input 1D array of distances
		col_ind :array
			 	 Input 1D array column indicies
		row_ptr : array
			   	  Input 1D array row pointers
		");
INSTANTIATE_ALL(self_tuning_gaussian_kernel_csr)

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function normalizes a sparse matrix in CSR format.

		:Parameters:

		sdist : array
			    Output 1D array of distances
		data : array
			   Input 1D array of distances
		col_ind :array
			 	 Input 1D array column indicies
		row_ptr : array
			   	  Input 1D array row pointers
		");
INSTANTIATE_ALL(normalize_csr)

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function heaps sorts a partial distance matrix.

		:Parameters:

		dist2 : array
			   Input 2D array of distances
		data : array
			   Output 1D array of distances
		col_ind :array
			 	 Output 1D array column indicies
		offset : int
				 Offset for the column index
		k : int
			Number of neighbors
		");
INSTANTIATE_ALL(push_to_heap)

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function creates a sparse matrix from a heap sorted
		distance matrix.

		:Parameters:

		data : array
			   Output 1D array of distances
		col_ind :array
			 	 Output 1D array column indicies
		k : int
			Number of neighbors
		");
INSTANTIATE_ALL(finalize_heap)

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function calculates a self-tuning gaussin kernel over
		a sparse matrix in CSR format.

		:Parameters:
		
		data : array
			   Input 1D array of distances
		col_ind :array
			 	 Input 1D array column indicies
		row_ind : array
			   	  Input 1D array row indicies
		sdata : array
			   Output 1D array of distances
		scol_ind :array
			 	 Output 1D array column indicies
		srow_ind : array
			   	  Output 1D array row indicies
		d : int
			Difference between old and new number of neighbors
		k : int
			New number of neighbors
		");
INSTANTIATE_ALL(knn_reduce)

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function calculates a self-tuning gaussin kernel over
		a sparse matrix in CSR format.

		:Parameters:
		
		data : array
			   Input 1D array of distances
		col_ind :array
			 	 Input 1D array column indicies
		row_ind : array
			   	  Input 1D array row indicies
		sdata : array
			   Output 1D array of distances
		scol_ind :array
			 	 Output 1D array column indicies
		srow_ind : array
			   	  Output 1D array row indicies
		eps : float
			  Maximum allowed distance between neighbors
		");
INSTANTIATE_ALL(knn_reduce_eps)

%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function calculates a self-tuning gaussin kernel over
		a sparse matrix in CSR format.

		:Parameters:
		
		data : array
			   Input 1D array of distances
		col_ind :array
			 	 Input 1D array column indicies
		row_ind : array
			   	  Input 1D array row indicies
		sdata : array
			   Output 1D array of distances
		scol_ind :array
			 	 Output 1D array column indicies
		srow_ind : array
			   	  Output 1D array row indicies
		cdata : array
			    Input 1D array of distances used for comparison
		eps : float
			  Maximum allowed distance between neighbors
		");
INSTANTIATE_ALL(knn_reduce_eps_cmp)



%feature("autodoc", "");
%feature("docstring",
		" This SWIG wrapper function calculates a self-tuning gaussin kernel over
		a sparse matrix in CSR format.

		:Parameters:
		
		data : array
			   In/out 1D array of distances
		col_ind :array
			 	 In/out 1D array column indicies
		row_ind : array
			   	  In/out 1D array row indicies
		k : int
			New number of neighbors
		
		:Returns:
		
		n : int
			Number of sparse values
		");
INSTANTIATE_ALL(knn_mutual)
