''' Bridge to EMAN2 with asorted utilities

This module handles testing for and converting EMAN2 objects to
numpy arrays.

.. Created on Sep 28, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import logging
try:
    import EMAN2
    import utilities
    import fundamentals
    EMAN2;
except:
    logging.error("Cannot import EMAN2 libaries, ensure they are proplery installed and availabe on the PYTHONPATH")
    EMAN2 = None
import numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def fshift(img, x, y, z=0, out=None):
    ''' Shift an image
    
    :Parameters:
    
    img : array
          Image data
    x : float
        Translation in the x-direction
    y : float
        Translation in the y-direction
    z : float
        Translation in the z-direction
    out : array
          Output array
        
    :Returns:
    
    img : array
          Transformed image
    '''
    if out is None: out = img.copy()
    emdata = numpy2em(img)
    emdata = fundamentals.fshift(emdata, x, y, z)
    out[:, :] = em2numpy(emdata)
    return out

def mirror(img, out=None):
    ''' Mirror an image about the x-axis
    
    :Parameters:
    
    img : array
          Image data
    out : array
          Output array
        
    :Returns:
    
    img : array
          Transformed image
    '''
    if out is None: out = img.copy()
    emdata = numpy2em(img)
    emdata.process_inplace("mirror", {"axis":'x'})
    out[:, :] = em2numpy(emdata)
    return out

def rot_shift2D(img, psi, tx=None, ty=None, m=None, out=None):
    ''' Rotate and shift an image in 2D
    
    :Parameters:
    
    img : array
          Image data
    psi : float
          Inplane rotation
    tx : int
         Translation in the x-direction
    ty : int
         Translation in the y-direction
    m : bool
        True if image should be mirrored around x-axis
    out : array
          Output array
        
    :Returns:
    
    img : array
          Transformed image
    '''
    if tx is None:
        #m = psi[1] > 179.9
        tx = psi[6]
        ty = psi[7]
        psi = psi[5]
    if out is None: out = img.copy()
    emdata = numpy2em(img)
    emdata = fundamentals.rot_shift2D(emdata, psi, tx, ty, m, interpolation_method="gridding")
    out[:, :] = em2numpy(emdata)
    return out

def model_circle(rad, x, y):
    ''' Create a model circle
    
    :Parameters:
    
    rad : int
          Radius
    x : int
        Width of the image
    y : int
        Height of the image
        
    :Returns:
    
    model : array
            Image with disk of radius `rad`
    '''
    
    emdata =  utilities.model_circle(rad, x, y)
    return em2numpy(emdata).copy()

def em2numpy2em(fn):
    ''' Convert the first argument from EMData to ndarray and convert result
    from ndarray to EMData only if the input is EMData.
    
    :Parameters:
    
    fn : function
         Name of the function to decorate
    
    :Returns:
    
    new : function
          Decorator
    '''
    
    def new(img, *args, **kwargs):
        '''Convert the first argument from EMData to ndarray and convert result
        from ndarray to EMData only if the input is EMData.
        
        :Parameters:
            
        img : EMData or ndarray
              Image
        args : list
               Other positional arguments
        kwargs : dict
                 Other keyword arguments
        
        :Returns:
        
        img : EMData or ndarray (depending on input)
              Image
        '''
        
        orig = img
        if is_em(img): img = em2numpy(img)
        res = fn(img, *args, **kwargs)
        if is_em(orig): res = numpy2em(res)
        return res
    return new

def em2numpy2res(fn):
    ''' Convert the first argument from EMData to ndarray and convert result
    from ndarray to EMData only if the input is EMData.
    
    :Parameters:
    
    fn : function
         Name of the function to decorate
    
    :Returns:
    
    new : function
          Decorator
    '''
    
    def new(img, *args, **kwargs):
        '''Convert the first argument from EMData to ndarray and convert result
        from ndarray to EMData only if the input is EMData.
        
        :Parameters:
            
        img : EMData or ndarray
              Image
        args : list
               Other positional arguments
        kwargs : dict
                 Other keyword arguments
        
        :Returns:
        
        img : EMData or ndarray (depending on input)
              Image
        '''
        
        orig = img
        orig;
        if is_em(img): img = em2numpy(img)
        res = fn(img, *args, **kwargs)
        return res
    return new

def is_em(im):
    '''Test if image is an EMAN2 image (EMData)
    
    This convenience method tests if the image is an EMAN2 image (EMAN2.EMData)
    
    .. sourcecode:: py
        
        >>> from core.image.eman2_utility import *
        >>> is_em(numpy.zeros((20,20)))
        False
        >>> is_em(EMData())
        True
    
    :Parameters:

    img : image-like object
         A object holding an image possibly EMAN2.EMData
        
    :Returns:
        
    return_val : boolean
                True if it is an EMAN2 image (EMAN2.EMData)
    '''
    
    return isinstance(im, EMAN2.EMData)

def is_numpy(im):
    '''Test if image is a NumPy array
    
    This convenience method tests if the image is a numpy.ndarray.
    
    .. sourcecode:: py
    
        >>> from core.image.eman2_utility import *
        >>> is_em(numpy.zeros((20,20)))
        True
        >>> is_em(EMData())
        False
    
    :Parameters:

    img : image-like object
         A object holding an image possibly numpy.ndarray
        
    :Returns:
        
    return_val : boolean
                True if is a numpy.ndarray
    '''
    
    return isinstance(im, numpy.ndarray)

def em2numpy(im):
    '''Convert EMAN2 image object to a NumPy array
    
    This convenience method converts an EMAN2.EMData object into a numpy.ndarray.
    
    .. sourcecode:: py
    
        >>> from core.image.eman2_utility import *
        >>> e = EMData()
        >>> e.set_size(2, 2, 1)
        >>> e.to_zero()
        >>> em2numpy(e)
        array([[ 0.,  0.],
               [ 0.,  0.]], dtype=float32)
    
    :Parameters:

    img : EMAN2.EMData
         An image object
        
    :Returns:
        
    return_val : numpy.ndarray
                An numpy.ndarray holding image data
    '''
    
    return EMAN2.EMNumPy.em2numpy(im)

def numpy2em(im):
    '''Convert NumPy array to an EMAN2 image object
    
    This convenience method converts a numpy.ndarray object into an EMAN2.EMData
    
    .. sourcecode:: py
    
        >>> from core.image.eman2_utility import *
        >>> ar = numpy.zeros((2,2))
        >>> numpy2em(ar)
        <libpyEMData2.EMData object at 0xdd61b0>
    
    :Parameters:

    img : numpy.ndarray
          A numpy array
        
    :Returns:
        
    return_val : EMAN2.EMData
                An EMAN2 image object
    '''
        
    try:
        e = EMAN2.EMData()
        EMAN2.EMNumPy.numpy2em(im, e)
        return e
    except:
        return EMAN2.EMNumPy.numpy2em(im)

def fsc(img1, img2, complex=False):
    ''' Estimate the Fourier shell correlation between two images
    
    :Parameters:
    
    img1 : array
           Image
    img2 : array
           Image
    complex : bool
              Set true if images are complex
    
    :Returns:
    
    fsc : array
          Fourier shell correlation curve: (0) spatial frequency (1) FSC
    '''
    
    if not is_em(img1): img1 = numpy2em(img1)
    if not is_em(img2): img2 = numpy2em(img2)
    if complex:
        img1.set_attr('is_complex', 1)
        img2.set_attr('is_complex', 1)
    res = img1.calc_fourier_shell_correlation(img2, 1.0)
    res = numpy.asarray(res).reshape((3, len(res)/3)).T
    #sel = numpy.abs(res[:, 1]-0.5)
    #sp = res[sel.argmin(), 0]
    #resolution = apix*bin_factor/sp
    return res

def ramp(img, inplace=True):
    '''Remove change in illumination across an image
    
    :Parameters:

    img : EMAN2.EMData
          Input Image
    inplace : bool
              If True, process the image in place.
    
    :Returns:
    
    img : EMAN2.EMData
          Ramped Image
    '''
    
    if inplace: img.process_inplace("filter.ramp")
    else: img = img.process("filter.ramp")
    return img

def histfit(img, mask, noise):
    '''Contrast enhancement using histogram fitting (ce_fit in Spider).
    
    :Parameters:
        
    img : EMAN2.EMData
          Input Image
    mask : EMAN2.EMData
           Image mask
    noise : EMAN2.EMData
            Noise image
    
    :Returns:

    out : EMAN2.EMData
          Enhanced image
    '''
    
    return utilities.ce_fit(img, noise, mask)[2]
    
def decimate(img, bin_factor=0, force_even=False, **extra):
    '''Decimate the image
    
    :Parameters:

    img : EMAN2.EMData
          Image to decimate
    bin_factor: int
                Factor to decimate
    force_even: bool
                Ensure decimated image has even dimensions
    extra : dict
            Unused extra keyword arguments
    
    :Returns:

    val : EMAN2.EMData
          Decimated image
    '''
    orig = img
    if not is_em(img): img = numpy2em(img)
    
    if bin_factor == 0.0: return img
    bin_factor = 1.0/bin_factor
    
    if force_even:
        n = int( float(img.get_xsize()) * bin_factor )
        if (n % 2) > 0:
            n += 1
            bin_factor = float(n) / float(img.get_xsize())
    
    frequency_cutoff = 0.5*bin_factor
    template_min = 15
    sb = EMAN2.Util.sincBlackman(template_min, frequency_cutoff, 1999) # 1999 taken directly from util_sparx.h
    img = img.downsample(sb, bin_factor)
    if not is_em(orig): 
        orig = img
        img = em2numpy(orig).copy()
    return img

def gaussian_high_pass(img, ghp_sigma=0.1, pad=False, **extra):
    ''' Filter an image with the Gaussian high pass filter
    
    :Parameters:

    img : EMAN2.EMData
          Image requiring filtering
    ghp_sigma : float
                Frequency range
    pad : bool
          Pad image
    extra : dict
            Unused extra keyword arguments
    
    :Returns:

    val : EMAN2.EMData
          Filtered image
    '''
    
    if ghp_sigma == 0.0: return img
    orig = img
    if not is_em(img): img = numpy2em(img)
    img = EMAN2.Processor.EMFourierFilter(img, {"filter_type" : EMAN2.Processor.fourier_filter_types.GAUSS_HIGH_PASS,   "cutoff_abs": ghp_sigma, "dopad" : pad})
    if not is_em(orig): img = em2numpy(img).copy()
    return img

def gaussian_low_pass(img, glp_sigma=0.1, pad=False, **extra):
    ''' Filter an image with the Gaussian low pass filter
    
    :Parameters:

    img : EMAN2.EMData
          Image requiring filtering
    glp_sigma : float
                Frequency range
    pad : bool
          Pad image
    extra : dict
            Unused extra keyword arguments
    
    :Returns:

    val : EMAN2.EMData
          Filtered image
    '''
    
    if glp_sigma == 0.0: return img
    orig = img
    if not is_em(img): img = numpy2em(img)
    img = EMAN2.Processor.EMFourierFilter(img, {"filter_type" : EMAN2.Processor.fourier_filter_types.GAUSS_LOW_PASS,    "cutoff_abs": glp_sigma, "dopad" : pad})
    if not is_em(orig): img = em2numpy(img).copy()
    return img

def setup_nn4(image_size, npad=2, sym='c1', weighting=1):
    ''' Initalize a reconstruction object
    
    :Parameters:
    
    image_size : int
                 Size of the input image and output volume
    npad : int, optional
           Number of times to pad the input image, default: 2
    sym : str, optional
          Type of symmetry, default: 'c1'
    weighting : int
                Amount of weight to give projections
    
    :Returns:
    
    recon : tuple
            Reconstructor, Fourier volume, Weight Volume, and numpy versions
    '''
    
    fftvol = EMAN2.EMData()
    weight = EMAN2.EMData()
    param = {"size":image_size, "npad":npad, "symmetry":sym, "weighting":weighting, "fftvol": fftvol, "weight": weight}
    r = EMAN2.Reconstructors.get("nn4", param)
    r.setup()
    return (r, fftvol, weight), em2numpy(fftvol), em2numpy(weight)

def backproject_nn4(img, align=None, recon=None, **extra):
    ''' Add the given image and alignment or generator of image/alignment pairs
    to the current reconstruction
    
    :Parameters:
    
    img : array or EMData
          Image of projection to backproject into reconstruction
    align : array, optional
            Array of alignment parameters (not required if img is generator of images and alignments)
    recon : tuple
            Reconstructor, Fourier volume, Weight Volume, and numpy versions
    extra : dict
            Keyword arguments to be passed to setup_nn4 if recon is None
    
    :Returns:
    
    recon : tuple
            Reconstructor, Fourier volume, Weight Volume, and numpy versions
    '''
    
    npad, sym, weighting = extra.get('npad', 2), extra.get('sym', 'c1'), extra.get('weighting', 1)
    if not hasattr(img, 'ndim'):
        for i, val in enumerate(img):
            if isinstance(val, tuple): val, a = val
            else: a = align[i]
            xform_proj = EMAN2.Transform({"type":"spider","phi":a[2],"theta":a[1],"psi":a[0]})
            if not is_em(val): val = numpy2em(val)
            if recon is None: recon = setup_nn4(val.get_xsize(), npad, sym, weighting)
            recon[0][0].insert_slice(val, xform_proj)
    else:
        xform_proj = EMAN2.Transform({"type":"spider","phi":align[2],"theta":align[1],"psi":align[0]})
        if not is_em(img):img = numpy2em(img)
        if recon is None: recon = setup_nn4(val.get_xsize(), npad, sym, weighting)
        recon[0][0].insert_slice(img, xform_proj)
    return recon

def finalize_nn4(recon):
    ''' Inverse Fourier transform the Fourier volume
    
    :Parameters:
    
    recon : tuple
            Reconstructor, Fourier volume, Weight Volume, and numpy versions
    
    :Returns:
    
    vol : array
          Volume as a numpy array
    '''
    
    return em2numpy(recon[0][0].finish()).copy()

