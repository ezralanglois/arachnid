''' Plots an FSC that can be customized for publication

See http://matplotlib.org/ for more details on customizing plotting.

Download to edit and run: :download:`plot_cc.py <../../arachnid/snippets/plot_cc.py>`

To run:

.. sourcecode:: sh
    
    $ python plot_cc.py

.. note::
    
    You must have Arachnid and Matplotlib installed to run this script

.. literalinclude:: ../../arachnid/snippets/plot_cc.py
   :language: python
   :lines: 22-
   :linenos:
'''
from arachnid.core.metadata import format
from arachnid.core.learn import unary_classification
from arachnid.core.orient import healpix
import numpy
import pylab, os

if __name__ == '__main__':
    
    alignment_file = os.path.expanduser("")
    resolution = 1
    output_path=""
    output_files=[]
    
    align, header = format.read_alignment(alignment_file, ndarray=True)
    cc_rot = header.index('cc_rot')
    
    output_files.append(os.path.join(output_path, "ccrot_histogram.png"))
    ax = pylab.figure(0)
    th = unary_classification.otsu(align[:, cc_rot], numpy.sqrt(align.shape[0]))
    n = ax.hist(align[:, cc_rot], bins=numpy.sqrt(align.shape[0]))[0]
    max_val = sorted(n)[-1]
    ax.plot((th, th), (0, max_val))
    
    if resolution > 0:
        ax = pylab.figure(1)
        print healpix.res2npix(resolution)
        view = healpix.ang2pix(resolution, numpy.deg2rad(align[:, 1:3]))
        for i in xrange(healpix.res2npix(resolution)):
            sel = view == i
            ax.hist(align[sel, cc_rot], bins=numpy.sqrt(align.shape[0]))
    
    
    if output_path != "":
        for i, filename in enumerate(output_files):
            pylab.figure(i)
            pylab.savefig(filename)
    else:
        pylab.show()
    
    