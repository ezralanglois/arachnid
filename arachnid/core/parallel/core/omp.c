#define _VERSION_ "2012.09.05"
#include "Python.h"

#ifdef _OPENMP
#include <omp.h>
#endif


char py_set_num_threads_doc[] =
    "Set the number of threads for OpenMP to use";

static PyObject *
py_set_num_threads(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    int thread_count;
    static char *kwlist[] = {"thread_count", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O", kwlist,
        &thread_count)) goto _fail;
#	ifdef _OPENMP
    if (thread_count < 1) thread_count = 1;
	omp_set_num_threads(thread_count);
#	endif

 _fail:
    return NULL;
}


/*****************************************************************************/
/* Create Python module */

char module_doc[] =
    "pyOpenMP Library.\n\n"
    "Authors:\n  Robert Langlois <code.google.com/p/arachnid>\n"
    "\n\nVersion: %s\n";

static PyMethodDef module_methods[] = {
    {"set_num_threads",
        (PyCFunction)py_set_num_threads,
        METH_VARARGS|METH_KEYWORDS, py_set_num_threads_doc},
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
init_healpix(void)
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

    if (_import_array() < 0) {
        Py_DECREF(module);
        INITERROR;
    }

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
