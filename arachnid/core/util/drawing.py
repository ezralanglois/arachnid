''' Utilities for drawing on images

.. Created on Nov 4, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
import logging
import scipy.misc

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

try: 
    from PIL import ImageDraw  #@UnresolvedImport
    ImageDraw;
except: 
    try:
        import ImageDraw
        ImageDraw;
    except: 
        tracing.log_import_error('Failed to load PIL - certain features will be disabled', _logger)
        ImageDraw = None
        
def is_available():
    ''' Test if PIL is available
    
    :Returns:
    
    flag : bool
           True if PIL is available
    '''
    
    return ImageDraw is not None

def mark(img):
    ''' Draw an X through an image
    
    :Parameters:
    
    img : array
          Image
    
    :Returns:
    
    out : array
          Returns the resulting RGB image as an array (nxnx3)
    '''
    
    if ImageDraw is None: return None
    if hasattr(img, 'ndim'):
        img = scipy.misc.toimage(img).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.line((0, 0)+img.size, fill=128, width=5)
    draw.line((0, img.size[1], img.size[0], 0), fill=128, width=5)
    return scipy.misc.fromimage(img)

def draw_particle_boxes(mic, coords, window, bin_factor=1.0, outline="#ff4040", **extra):
    ''' Draw boxes around particles on the micrograph using the given coordinates, window size
    and down sampling factor.
    
    :Parameters:
    
    mic : array
          Micrograph image
    coords : list
             List of particle coordinates
    window : int
             Size of window in pixels
    bin_factor : float
                 Image downsampling factor
    outline : str
              Color of box outline
    extra : dict
            Unused key word arguments
            
    :Returns:
    
    out : PIL.Image
          Returns the resulting RGB image
    '''
    
    if ImageDraw is None: return
    
    offset = window/2
    if hasattr(mic, 'ndim'):
        mic = scipy.misc.toimage(mic).convert("RGB")
    draw = ImageDraw.Draw(mic)
    
    for box in coords:
        x = box.x / bin_factor
        y = box.y / bin_factor
        draw.rectangle((x+offset, y+offset, x-offset, y-offset), fill=None, outline=outline)
    return mic

def draw_particle_boxes_to_file(mic, coords, window, bin_factor=1.0, box_image="", **extra):
    ''' Write out an image file to given filename with boxes drawn around particles on the micrograph 
    using the given coordinates, window size and down sampling factor.
    
    :Parameters:
    
    mic : array
          Micrograph image
    coords : list
             List of particle coordinates
    window : int
             Size of window in pixels
    bin_factor : float
                 Image downsampling factor
    box_image : str
                Output filename
    extra : dict
            Unused key word arguments
    '''
    
    if ImageDraw is None: return
    mic = draw_particle_boxes(mic, coords, window, bin_factor, **extra)
    mic.save(box_image)

def draw_particle_boxes_to_array(mic, coords, window, bin_factor=1.0, **extra):
    ''' Draw boxes around particles on the micrograph using the given coordinates, window size
    and down sampling factor. Return image as an array.
    
    :Parameters:
    
    mic : array
          Micrograph image
    coords : list
             List of particle coordinates
    window : int
             Size of window in pixels
    bin_factor : float
                 Image downsampling factor
    extra : dict
            Unused key word arguments
            
    :Returns:
    
    out : array
          Returns the resulting RGB image as an array (nxnx3)
    '''
    
    if ImageDraw is None: return
    mic = draw_particle_boxes(mic, coords, window, bin_factor, **extra)
    return scipy.misc.fromimage(mic)


