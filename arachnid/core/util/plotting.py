''' Utility functions for plotting

.. Created on Oct 16, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
try:
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
except:
    print "Cannot import offset, upgrade matplotlib"
import matplotlib.cm as cm
import matplotlib._pylab_helpers
from ..image import analysis
import numpy, pylab, logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def save_as_image(fig):
    '''
    '''
    
    fig.canvas.draw()
    data = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return rgb2gray(data)

def rgb2gray(rgb):
    '''
    '''
    
    r, g, b = numpy.rollaxis(rgb[...,:3], axis = -1)
    return 0.299 * r + 0.587 * g + 0.114 * b

def plot_embedding(x, y, selected=None, group=None, dpi=80, **extra):
    ''' Plot an embedding
    
    :Parameters:
    
    x : array
        X coordindates
    y : array
        Y coordindates
    selected : array, optional
               Plot selected points
    dpi : int
          Figure resolution
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    fig : Figure
          Matplotlib figure
    ax : Axes
          Matplotlib axes
    '''
    
    fig = pylab.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    if group is not None:
        refs = numpy.unique(group)
        beg, inc = 0.0, 1.0/len(refs)
        for r in refs:
            sel = r == group
            color = cm.spectral(beg)
            ax.plot(x[sel], y[sel], 'o', ls='.', markersize=3, c=color, **extra)
            beg += inc
    else:
        ax.plot(x, y, 'ro', ls='.', markersize=3, **extra)
    if selected is not None:
        ax.plot(x[selected], y[selected], 'k+', ls='.', markersize=2, **extra)
    return fig, ax

def figure_list():
    ''' Get a list of figures
    
    :Returns:
    
    figs : list
           List of figures
    '''
    
    return [manager.canvas.figure for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]

def nonoverlapping_subset(ax, x, y, radius, n):
    ''' Find a non-overlapping subset of points on the given axes
    
    :Parameters:
    
    ax : Axes
         Current axes
    x : array
        Corresponding x coordinate
    y : array
        Corresponding y coordinate
    radius : float
             Radius of exclusion
    n : int
        Maximum number of points
    
    :Returns:
    
    index : array
            Selected indicies
    '''
    
    if x.ndim == 1: x=x.reshape((x.shape[0], 1))
    if y.ndim == 1: y=y.reshape((y.shape[0], 1))
    return analysis.subset_no_overlap(ax.transData.transform(numpy.hstack((x, y))), numpy.hypot(radius, radius)*2, n)

def plot_images(fig, img_iter, x, y, zoom, radius):
    ''' Plot images on the specified figure
    
    :Parameters:
    
    fig : Figure
          Current figure object
    img_iter : iterable
               Image iterable object
    x : array
        Corresponding x coordinate
    y : array
        Corresponding y coordinate
    zoom : float
                 Zoom level of the image, radius
    radius : float
             Offset from the data point
    '''
    
    for i, img in enumerate(img_iter):
        im = OffsetImage(img, zoom=zoom, cmap=cm.Greys_r)
        try:
            ab = AnnotationBbox(im, (x[i], y[i]), xycoords='data', xybox=(radius, 0.), boxcoords="offset points", frameon=False)
        except:
            _logger.error("%d < %d"%(i, len(x)))
            raise
        fig.gca().add_artist(ab)

