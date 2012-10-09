''' Plots an FSC that can be customized for publication

See http://matplotlib.org/ for more details on customizing plotting.

Download to edit and run: :download:`plot_angles.py <../../arachnid/snippets/plot_angles.py>`

To run:

.. sourcecode:: sh
    
    $ python plot_angles.py

.. note::
    
    You must have Arachnid and Matplotlib installed to run this script

.. literalinclude:: ../../arachnid/snippets/plot_angles.py
   :language: python
   :lines: 22-
   :linenos:
'''
from arachnid.core.metadata import format, format_utility
from arachnid.core.orient import healpix
import matplotlib, numpy, scipy
matplotlib;

from mpl_toolkits.mplot3d import Axes3D
import pylab, os

if __name__ == '__main__':

    # Parameters
    
    alignment_file = os.path.expanduser("")
    resolution = 2
    output=""
    
    align = format.read(alignment_file, numeric=True)
    align,header = format_utility.tuple2numpy(align)
    
    ax = Axes3D(pylab.figure(0))
    
    align = numpy.deg2rad(align[:, 1:3])
    if resolution > 0:
        healpix.coarse(resolution, align, out=align)
    
    theta = align[:, 0]
    phi = align[:, 1]
    x = numpy.sin(theta)*numpy.cos(phi)
    y = numpy.sin(theta)*numpy.sin(phi)
    z = numpy.cos(theta)
    
    ax.scatter3D(x, y, z) #, c=c, cmap=cm.cool)
    
    if output != "":
        base, ext = os.path.splitext(output)
        for elev in scipy.linspace(0.0, numpy.pi, 3):
            for azim in scipy.linspace(0.0, 2*numpy.pi, 3):
                ax.view_init(elev, azim)
                pylab.savefig(base+("_%0.2f_%0.2f"%(elev, azim))+ext)
    else:pylab.show()
    
