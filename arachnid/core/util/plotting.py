''' Utility functions for plotting

.. Created on Oct 16, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.cm as cm
import matplotlib._pylab_helpers
from ..image import analysis
import numpy

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
        ab = AnnotationBbox(im, (x[i], y[i]), xycoords='data', xybox=(radius, 0.), boxcoords="offset points", frameon=False)
        fig.gca().add_artist(ab)
