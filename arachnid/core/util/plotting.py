''' Utility functions for plotting

.. Created on Oct 16, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''


from matplotlib_nogui import pylab, matplotlib
try:
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
except:
    print "Cannot import offset, upgrade matplotlib"
from ..image import analysis
from ..metadata import format_utility
import matplotlib.cm as cm
import matplotlib._pylab_helpers
import numpy, logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def is_plotting_disabled(): return pylab is None

def plot_scatter(output, x, x_label, y, y_label, color=None, dpi=72):
    ''' Plot a scatter plot
    '''
    
    if pylab is None: return
    
    fig = pylab.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    if color is not None:
        sc = ax.scatter(x, y, cmap=pylab.cm.cool, c=color)
        v1 = min(numpy.min(x),numpy.min(y))
        v2 = max(numpy.max(x),numpy.max(y))
        ax.plot(numpy.asarray([v1, v2]), numpy.asarray([v1, v2]))
        pylab.colorbar(sc)
    else:
        ax.scatter(x, y)
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    
    fig.savefig(format_utility.new_filename(output, x_label.lower().replace(' ', '_')+"_"+y_label.lower().replace(' ', '_')+"_summary_", ext="png"), dpi=dpi)

def plot_histogram_cum(output, vals, x_label, y_label, th=None, dpi=72):
    ''' Plot a histogram of the distribution
    '''
    
    if pylab is None: return
    
    fig = pylab.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    vals = ax.hist(-vals, bins=numpy.sqrt(len(vals)), cumulative=True)
    if th is not None:
        h = pylab.gca().get_ylim()[1]
        pylab.plot((th, th), (0, h))
    pylab.xlabel(x_label)
    pylab.ylabel('Cumulative '+y_label)
    
    fig.savefig(format_utility.new_filename(output, x_label.lower().replace(' ', '_')+"_"+y_label.lower().replace(' ', '_')+"_summary_", ext="png"), dpi=dpi)

def trim():
    '''
    '''
    
    pylab.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    
def draw_image(img, label=None, dpi=72, facecolor='white', cmap=cm.gray, output_filename=None, **extra):
    '''
    '''
    
    img = img.copy()
    fig = pylab.figure(0, dpi=dpi, facecolor=facecolor)
    img -= img.min()
    img /= img.max()
    
    #fig.set_size_inches(10, 10)
    #ax = pylab.Axes(fig, [0., 0., 10., 10.])
    #ax.set_axis_off()
    #fig.add_axes(ax)
    
    ax = pylab.axes(frameon=False)
    ax.set_axis_off()
    ax.imshow(img, cmap=cmap)#, aspect = 'normal')
    if label is not None:
        ax.text(10, 20, label, color='black', backgroundcolor='white', fontsize=8)
    if output_filename is not None:
        fig.savefig(format_utility.new_filename(output_filename, ext='png'), dpi=dpi)
    return fig, ax

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def save_as_image(fig):
    '''
    '''
    
    pylab.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
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

