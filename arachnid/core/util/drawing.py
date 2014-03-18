''' Utilities for drawing on images

.. Created on Nov 4, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..app import tracing
import logging
import scipy.misc
import numpy

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

def draw_particle_boxes(mic, coords, window, bin_factor=1.0, outline="#ff4040", width=1, **extra):
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
        draw.rectangle((x+offset, y+offset, x-offset, y-offset), fill=None, outline=outline, width=width)
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

def draw_arrow(draw, x0, y0, x1, y1, width=5, fill=None):
    '''
    '''
    
    assert((x1-x0)!=0)
    assert((y1-y0)!=0)
    vec = numpy.array([[x1 - x0,], [y1 - y0,]])
    
    vec_anti = numpy.dot(numpy.array([[0.0, -1.0], [1.0, 0.0]]), vec)
    vec_clock = numpy.dot(numpy.array([[0.0, 1.0], [-1.0, 0.0]]), vec)
    print vec_anti, vec_clock
    vec_anti *= 0.5*width/((vec_anti**2).sum())**0.5
    vec_clock *= 0.5*width/((vec_clock**2).sum())**0.5
    
    perp_st = (x1 + float(vec_anti[0]), y1 + float(vec_anti[1]))
    perp_end = (x1 + float(vec_clock[0]), y1 + float(vec_clock[1]))
    
    slope = (y1-y0)/(x1-x0)
    angle = numpy.arctan(slope)
    dirc_pt = (x1 + width*numpy.cos(angle), y1 + width*numpy.sin(angle))
    draw.polygon((perp_st[0], perp_st[1], perp_end[0], perp_end[1], dirc_pt[0], dirc_pt[1]), fill=fill)
    

def draw_path(img, waypoints, color="#ff4040", width=10, out=None):
    ''' Draw a path on an image using the given waypoints.
    
    :Parameters:
    
    mic : array
          An image
    waypoints : array
                Waypoints for the path
    color : str
            Color code for the line segments
    out : str or array, optional
    
    '''
    
    if ImageDraw is None or len(waypoints) == 0: return img
    
    if hasattr(img, 'ndim'): 
        img = img - img.min()
        img /= img.max()
        img *= 255
        img = scipy.misc.toimage(img).convert("RGB")
    draw = ImageDraw.Draw(img)
    origin = waypoints[0]
    for point in waypoints[1:]:
        draw.line((origin[0], origin[1], point[0], point[1]), fill=color, width=width)
        if (origin[1]-point[1]) != 0 and (origin[0]-point[0]) != 0:
            draw_arrow(draw, origin[0], origin[1], point[0], point[1], width*2, 'blue')
        
        #draw.ellipse((origin[0]-width, origin[1]-width, width, width), fill=color)
        origin=point
    
    if out is not None:
        try:out+""
        except:
            if hasattr(out, 'ndim'):
                img -= img.min()
                img /= img.max()
                img *= 255
                out[:]=scipy.misc.fromimage(img)
            else: raise ValueError, "`out` must be str or array"
        else:
            img.save(out)
    else: out = img
    return out

