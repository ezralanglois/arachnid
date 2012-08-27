#include "healpix/chealpix.h"
#define WIN32_LEAN_AND_MEAN
#define _VERSION_ "2012.08.17"

#include "Python.h"

#include "structmember.h"
#include "math.h"
#include "float.h"
#include "string.h"
#include "numpy/arrayobject.h"

static int
PyConverter_DoubleVector3OrNone(
    PyObject *object,
    PyObject **address)
{
    if ((object == NULL) || (object == Py_None)) {
        *address = NULL;
    } else {
        PyArrayObject *obj;
        *address = PyArray_FROM_OTF(object, NPY_DOUBLE, NPY_IN_ARRAY);
        if (*address == NULL) {
            PyErr_Format(PyExc_ValueError, "can not convert to array");
            return NPY_FAIL;
        }
        obj = (PyArrayObject *) *address;
        if ((PyArray_NDIM(obj) != 1) || (PyArray_DIM(obj, 0) < 3)
            || PyArray_ISCOMPLEX(obj)) {
            PyErr_Format(PyExc_ValueError, "not a vector3");
            Py_DECREF(*address);
            *address = NULL;
            return NPY_FAIL;
        }
    }
    return NPY_SUCCEED;
}

char py_pix2ang_nest_doc[] =
    "Return Euler angles for specified pixel in the nest scheme";

static PyObject *
py_pix2ang_nest(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    PyObject *euler = NULL;
    PyObject *result = NULL;
    Py_ssize_t dims[] = {2};
    long ipix;
    long order;
    static char *kwlist[] = {"order", "ipix", "euler", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O", kwlist,
        &order, &ipix, PyConverter_DoubleVector3OrNone, &euler)) goto _fail;

    if( euler == NULL )
    {
    	result = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
		if (euler == NULL)
		{
			PyErr_Format(PyExc_MemoryError, "unable to allocate result");
			goto _fail;
		}
    }
    else result = euler;

    {
    	double *ang = (double *)PyArray_DATA(result);
    	pix2ang_nest(order, ipix, ang, ang+1);
    }

    if( euler != NULL ) Py_XDECREF(euler);
    return PyArray_Return(result);

  _fail:
    if( euler != NULL ) Py_XDECREF(euler);
    return NULL;
}

char py_pix2ang_ring_doc[] =
    "Return Euler angles for specified pixel in the nest scheme";

static PyObject *
py_pix2ang_ring(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    PyObject *euler = NULL;
    PyObject *result = NULL;
    Py_ssize_t dims[] = {2};
    long ipix;
    long order;
    static char *kwlist[] = {"order", "ipix", "euler", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O", kwlist,
        &order, &ipix, PyConverter_DoubleVector3OrNone, &euler)) goto _fail;

    if( euler == NULL )
    {
    	result = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
		if (euler == NULL)
		{
			PyErr_Format(PyExc_MemoryError, "unable to allocate result");
			goto _fail;
		}
    }
    else result = euler;

    {
    	double *ang = (double *)PyArray_DATA(result);
    	pix2ang_ring(order, ipix, ang, ang+1);
    }

    if( euler != NULL ) Py_XDECREF(euler);
    return PyArray_Return(result);

  _fail:
    if( euler != NULL ) Py_XDECREF(euler);
    return NULL;
}

char py_ang2pix_nest_doc[] =
    "Return Pixel in nested scheme for specified Euler angles";

static PyObject *
py_ang2pix_nest(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    long order;
    double theta;
    double phi;
	long ipix;
    static char *kwlist[] = {"order", "theta", "phi", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O", kwlist,
        &order, &theta, &phi)) goto _fail;

    {
    	ang2pix_nest(order, theta, phi, &ipix);
    }

    return PyArray_Return(ipix);

  _fail:
    return NULL;
}

char py_ang2pix_ring_doc[] =
    "Return Pixel in ring scheme for specified Euler angles";

static PyObject *
py_ang2pix_ring(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    long order;
    double theta;
    double phi;
	long ipix;
    static char *kwlist[] = {"order", "theta", "phi", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O", kwlist,
        &order, &theta, &phi)) goto _fail;

    {
    	ang2pix_ring(order, theta, phi, &ipix);
    }

    return PyArray_Return(ipix);

  _fail:
    return NULL;
}

char py_nest2ring_doc[] =
    "Return Pixel in ring scheme for pixel in nest scheme";

static PyObject *
py_nest2ring(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    long order;
	long ipix;
	long ipixr;
    static char *kwlist[] = {"order", "ipix", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O", kwlist,
        &order, &ipix)) goto _fail;

    {
    	nest2ring(order, ipix, &ipixr);
    }

    return PyArray_Return(ipixr);

  _fail:
    return NULL;
}

char py_ring2nest_doc[] =
    "Return Pixel in nest scheme for pixel in ring scheme";

static PyObject *
py_ring2nest(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    long order;
	long ipix;
	long ipixn;
    static char *kwlist[] = {"order", "ipix", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O", kwlist,
        &order, &ipix)) goto _fail;

    {
    	ring2nest(order, ipix, &ipixn);
    }

    return PyArray_Return(ipixn);

  _fail:
    return NULL;
}

char py_nside2npix_doc[] =
    "Return number of pixels for specified order";

static PyObject *
py_nside2npix(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    long order;
    long npix;
    static char *kwlist[] = {"order", "ipix", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O", kwlist,
        &order)) goto _fail;

    {
    	npix = nside2npix(order);
    }

    return PyArray_Return(npix);

  _fail:
    return NULL;
}

char py_npix2nside_doc[] =
    "Return order for number of pixels";

static PyObject *
py_npix2nside(
    PyObject *obj,
    PyObject *args,
    PyObject *kwds)
{
    long order;
    long npix;
    static char *kwlist[] = {"order", "ipix", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O", kwlist,
        &npix)) goto _fail;

    {
    	order = npix2nside(npix);
    }

    return PyArray_Return(order);

  _fail:
    return NULL;
}


/*****************************************************************************/
/* Create Python module */

char module_doc[] =
    "PyHEALPix Library.\n\n"
    "Authors:\n  Robert Langlois <code.google.com/p/arachnid>\n"
    "\n\nVersion: %s\n";

static PyMethodDef module_methods[] = {
    {"pix2ang_nest",
        (PyCFunction)py_pix2ang_nest,
        METH_VARARGS|METH_KEYWORDS, py_pix2ang_nest_doc},
    {"pix2ang_ring",
        (PyCFunction)py_pix2ang_ring,
        METH_VARARGS|METH_KEYWORDS, py_pix2ang_ring_doc},
	{"ang2pix_nest",
			(PyCFunction)py_ang2pix_nest,
			METH_VARARGS|METH_KEYWORDS, py_ang2pix_nest_doc},
	{"ang2pix_ring",
			(PyCFunction)py_ang2pix_ring,
			METH_VARARGS|METH_KEYWORDS, py_ang2pix_ring_doc},
	{"ring2nest",
			(PyCFunction)py_ring2nest,
			METH_VARARGS|METH_KEYWORDS, py_ring2nest_doc},
	{"nest2ring",
			(PyCFunction)py_nest2ring,
			METH_VARARGS|METH_KEYWORDS, py_nest2ring_doc},
	{"nside2npix",
			(PyCFunction)py_nside2npix,
			METH_VARARGS|METH_KEYWORDS, py_nside2npix_doc},
	{"npix2nside",
			(PyCFunction)py_npix2nside,
			METH_VARARGS|METH_KEYWORDS, py_npix2nside_doc},

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
        "_healpix",
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
    module = Py_InitModule3("_healpix", module_methods, doc);
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
