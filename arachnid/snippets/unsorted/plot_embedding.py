''' Plots an embedding with images

See http://matplotlib.org/ for more details on customizing plotting.

Download to edit and run: :download:`plot_embedding.py <../../arachnid/snippets/plot_embedding.py>`

To run:

.. sourcecode:: sh
    
    $ python plot_embedding.py

.. note::
    
    You must have Arachnid and Matplotlib installed to run this script

.. literalinclude:: ../../arachnid/snippets/plot_embedding.py
   :language: python
   :lines: 22-
   :linenos:
'''
from arachnid.core.metadata import format, format_utility, spider_utility
from arachnid.core.image import ndimage_file, ndimage_utility
from arachnid.core.util import plotting
#from arachnid.core.util import numpy_ext
import numpy
import pylab, os, itertools

def plot_embedding(x, y, selection=None, best=None, markersize=3, dpi=80):
    ''' Plot the embedding with selection and predictions if available
    
    :Parameters:
    
    x : array
        X-values
    y : array
        Y-values
    selection : array
                Selections
    best : array
           Predictions
    markersize : int
                 Size of the marker
    dpi : int
          Resolution of the image
    
    :Returns:
    
    ax : axes
         Plotting axes
    '''
    
    fig = pylab.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    if selection is not None:
        usel = numpy.unique(selection)
        markers=['o', '+', '*']
        colors=['y', 'g', 'b']
        for i in xrange(len(usel)):
            sel = usel[i] == selection
            print sel.shape, x.shape
            ax.plot(x[sel], y[sel], marker=markers[i], color=colors[i], ls='.', markersize=markersize)
        if best is not None:
            ax.plot(x[best], y[best], 'k.', markersize=markersize)
    else:
        ax.plot(x, y, marker="o", ls='.')
    return ax



if __name__ == '__main__':
    
    path = os.path.expanduser("~/Desktop/")
    input_file = os.path.join(path, "embed_clean_002.csv")
    select_file = "" #os.path.join(path, "select", "select1.prj")
    stack_file = os.path.join(path, "pos_clean_002.sdo") if 1 == 1 else ""
    output_file = os.path.join(path, "example.png")
    #output_file=""
    id_len=0
    image_size=0.4
    radius=20
    dpi=200
    markersize=3
    
    data = format.read(input_file, numeric=True)
    selection = format.read(select_file, numeric=True) if select_file else None
    if selection is not None:
        selection, header = format_utility.tuple2numpy(selection)
        selection = selection[:, header.index('select')]
    
    try:
        label = spider_utility.tuple2id(data, input_file, id_len)
    except: label=None
    
    data, header = format_utility.tuple2numpy(data)
    data = data.squeeze()
    c0 = header.index('c0')
    c1 = header.index('c1')
    try:
        best = header.index('best')
    except: best = None
    
    ax1=plot_embedding(data[:, c0], data[:, c1], selection, data[:, best] < 0.5, markersize, dpi)
    #ax2=plot_embedding(data[:, c0], data[:, c1], selection, best, markersize, dpi)
    #ax3=plot_embedding(data[:, c0], data[:, c1], selection, best, markersize, dpi)
    
    try:
        fig = pylab.figure(dpi=dpi)#, dpi=160)#figsize=(8, 6)
        ax = fig.add_subplot(111)
        
        if selection is not None:
            usel = numpy.unique(selection)
            markers=['o', '+', '*']
            colors=['y', 'g', 'b']
            for i in xrange(len(usel)):
                sel = usel[i] == selection
                ax.plot(data[sel, c0], data[sel, c1], marker=markers[i], color=colors[i], ls='.', markersize=markersize)
            if best is not None:
                sel = data[:, best] < 0.5
                ax.plot(data[sel, c0], data[sel, c1], 'k.', markersize=markersize)
            sel = numpy.argwhere(numpy.logical_and(data[:, best] > 0.5, selection < 0.0))
                
        else:
            sel = numpy.argwhere(data[:, best] > 0.5)
            ax.plot(data[:, c0], data[:, c1], marker="o", ls='.')
        if stack_file != "" and label is not None:
            index = sel[plotting.nonoverlapping_subset(ax, data[sel, c0], data[sel, c1], radius, 100)]
            label2 = label[index].squeeze()
            label2[:, 1]-=1
            iter_single_images = itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(stack_file, label2))
            plotting.plot_images(fig, iter_single_images, data[index, c0], data[index, c1], image_size, radius)
            for i in index: ax.plot([data[i, c0]], [data[i, c1]], 'r.', markersize=markersize)
        
        fig = pylab.figure(dpi=dpi)#, dpi=160)#figsize=(8, 6)
        
        ax = fig.add_subplot(111)
        
        if selection is not None:
            usel = numpy.unique(selection)
            markers=['o', '+', '*']
            colors=['y', 'g', 'b']
            for i in xrange(len(usel)):
                sel = usel[i] == selection
                ax.plot(data[sel, c0], data[sel, c1], marker=markers[i], color=colors[i], ls='.', markersize=markersize)
            if best is not None:
                sel = data[:, best] < 0.5
                ax.plot(data[sel, c0], data[sel, c1], 'k.', markersize=markersize)
            sel = numpy.argwhere(numpy.logical_and(data[:, best] < 0.5, selection < 0.0)) # True Negatives
                
        else:
            sel = numpy.argwhere(data[:, best] < 0.5)
            ax.plot(data[:, c0], data[:, c1], marker="o", ls='.')
        
        if stack_file != "" and label is not None:
            index = sel[plotting.nonoverlapping_subset(ax, data[sel, c0], data[sel, c1], radius, 100)]
            label2 = label[index].squeeze()
            label2[:, 1]-=1
            
            iter_single_images = itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(stack_file, label2))
            plotting.plot_images(fig, iter_single_images, data[index, c0], data[index, c1], image_size, radius)
            for i in index: ax.plot([data[i, c0]], [data[i, c1]], 'r.', markersize=markersize)
    except: pass
    
    if output_file:
        for i, fig in enumerate(plotting.figure_list()):
            fig.savefig(format_utility.add_prefix(output_file, "fig_%d"%(i+1)), dpi=dpi)
    else:
        pylab.show()
