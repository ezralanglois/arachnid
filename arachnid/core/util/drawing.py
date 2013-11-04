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
    from PIL import ImageDraw 
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

def draw_particle_boxes(mic, coords, window, bin_factor=1.0, outline="#ff4040", ret_draw=False, **extra):
    ''' Write out an image with the particles boxed
    
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
    '''
    
    if ImageDraw is None: return
    
    offset = window/2
    if hasattr(mic, 'ndim'):
        mic = scipy.misc.toimage(mic).convert("RGB")
        draw = ImageDraw.Draw(mic)
    else:
        draw = mic
    
    for box in coords:
        x = box.x / bin_factor
        y = box.y / bin_factor
        draw.rectangle((x+offset, y+offset, x-offset, y-offset), fill=None, outline=outline)
    return mic if not ret_draw else draw

def draw_particle_boxes_to_file(mic, coords, window, bin_factor=1.0, box_image="", **extra):
    ''' Write out an image with the particles boxed
    
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
    ''' Write out an image with the particles boxed
    
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
    '''
    
    if ImageDraw is None: return
    mic = draw_particle_boxes(mic, coords, window, bin_factor, **extra)
    return scipy.misc.fromimage(mic)


