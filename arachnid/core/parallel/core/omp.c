#define _VERSION_ "2012.09.05"
#include "Python.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _SCIPY_MKL_H
#include "mkl_service.h"
#endif

char py_set_num_threads_doc[] =
    "Set the number of threads for OpenMP to use";
static PyObject* py_set_num_threads(PyObject *obj, PyObject *args)
{
    int thread_count;
    if (!PyArg_ParseTuple(args, "i", &thread_count)) return NULL;
#	ifdef _OPENMP
    if (thread_count < 1) thread_count = 1;
	omp_set_num_threads(thread_count);
#		ifdef _SCIPY_MKL_H
		fprintf(stderr, 'called\n');
		mkl_set_num_threads(thread_count);
#		endif
#	endif
	Py_RETURN_NONE;
}

char py_get_max_threads_doc[] =
    "Get maximum number of available threads for OpenMP to use";
static PyObject* py_get_max_threads(PyObject *obj)
{
#	ifdef _OPENMP
    return PyInt_FromLong(omp_get_max_threads());
#   else
    return PyInt_FromLong(0);
#	endif
}

char py_get_num_procs_doc[] =
    "Get maximum number of available processors";
static PyObject* py_get_num_procs(PyObject *obj)
{
#	ifdef _OPENMP
    return PyInt_FromLong(omp_get_num_procs());
#   else
    return PyInt_FromLong(0);
#	endif
}




/*****************************************************************************/
/* Create Python module */

char module_doc[] =
    "pyOpenMP Library.\n\n"
    "Authors:\n  Robert Langlois <code.google.com/p/arachnid>\n"
    "\n\nVersion: %s\n";

static PyMethodDef module_methods[] = {
    {"set_num_threads", (PyCFunction)py_set_num_threads, METH_VARARGS, py_set_num_threads_doc},
    {"get_max_threads", (PyCFunction)py_get_max_threads, 0, py_get_max_threads_doc},
    {"get_num_procs", (PyCFunction)py_get_num_procs, 0, py_get_num_procs_doc},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int module_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int module_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_omp",
        NULL,
        sizeof(struct module_state),
        module_methods,
        NULL,
        module_traverse,
        module_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit__transformations(void)

#else

#define INITERROR return

PyMODINIT_FUNC
init_omp(void)
#endif
{
    PyObject *module;

    char *doc = (char *)PyMem_Malloc(sizeof(module_doc) + sizeof(_VERSION_));
    sprintf(doc, module_doc, _VERSION_);

#if PY_MAJOR_VERSION >= 3
    moduledef.m_doc = doc;
    module = PyModule_Create(&moduledef);
#else
    module = Py_InitModule3("_omp", module_methods, doc);
#endif

    PyMem_Free(doc);

    if (module == NULL)
        INITERROR;

    /*if (_import_array() < 0) {
        Py_DECREF(module);
        INITERROR;
    }*/

    {
#if PY_MAJOR_VERSION < 3
    PyObject *s = PyString_FromString(_VERSION_);
#else
    PyObject *s = PyUnicode_FromString(_VERSION_);
#endif
    PyObject *dict = PyModule_GetDict(module);
    PyDict_SetItemString(dict, "__version__", s);
    Py_DECREF(s);
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
