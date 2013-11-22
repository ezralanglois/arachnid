''' Plots an FSC that can be customized for publication

See http://matplotlib.org/ for more details on customizing plotting.

Download to edit and run: :download:`plot_angles.py <../../arachnid/snippets/plot_angles.py>`

To run:

.. sourcecode:: sh
    
    $ python plot_angles.py

.. note::
    
    You must have Arachnid and Matplotlib installed to run this script

2D Map Plot: http://matplotlib.github.com/basemap/users/mapsetup.html

    - aeqd    Azimuthal Equidistant
    - apoly    Polyconic
    - agnom    Gnomonic
    - amoll    Mollweide
    - atmerc    Transverse Mercator
    - anplaea    North-Polar Lambert Azimuthal
    - agall    Gall Stereographic Cylindrical
    - amill    Miller Cylindrical
    - amerc    Mercator
    - astere    Stereographic
    - anpstere    North-Polar Stereographic
    - ahammer    Hammer
    - ageos    Geostationary
    - ansper    Near-Sided Perspective
    - avandg    van der Grinten
    - alaea    Lambert Azimuthal Equal Area
    - ambtfpq    McBryde-Thomas Flat-Polar Quartic
    - asinu    Sinusoidal
    - aspstere    South-Polar Stereographic
    - alcc    Lambert Conformal
    - anpaeqd    North-Polar Azimuthal Equidistant
    - aeqdc    Equidistant Conic
    - acyl    Cylindrical Equidistant
    - aomerc    Oblique Mercator
    - aaea    Albers Equal Area
    - aspaeqd    South-Polar Azimuthal Equidistant
    - aortho    Orthographic
    - acass    Cassini-Soldner
    - asplaea    South-Polar Lambert Azimuthal
    - arobin    Robinson

.. literalinclude:: ../../arachnid/snippets/plot_angles.py
   :language: python
   :lines: 55-
   :linenos:
'''
from arachnid.core.metadata import format, format_utility
from arachnid.core.orient import healpix
import numpy, scipy
#import matplotlib
#matplotlib.use('Qt4Agg')

import matplotlib.cm as cm
from mpl_toolkits import mplot3d
from mpl_toolkits import basemap
from matplotlib import pyplot
import os

if __name__ == '__main__':

    # Parameters
    
    alignment_file = os.path.expanduser("~/Desktop/refine_005b.tls")
    resolution = 2
    output=""
    use_3d=False
    area_mult = 1.0
    
    align,header = format.read(alignment_file, ndarray=True)
    
    if use_3d:
        ax = mplot3d.Axes3D(pyplot.figure(0))
        
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
                    pyplot.savefig(base+("_%0.2f_%0.2f"%(elev, azim))+ext)
        else:pyplot.show()
    else:
        # Count number of projections per view
        ref = align[:, 3].astype(numpy.int)
        hist=numpy.histogram(ref, len(numpy.unique(ref)))[0]
        print hist.shape, hist.dtype
        hist = hist.ravel()
        s = numpy.sqrt(hist)*area_mult
        
        sel = align[:, 1] >= 180.0
        align[sel, 1] -= 180.0
        sel = align[:, 1] >= 90.0
        align[sel, 1] = 180.0 - align[sel, 1]
        
        # Create figure
        fig = pyplot.figure(0)
        # Create figure axis
        ax = fig.add_axes([0.05,0.05,0.9,0.9])
        
        # Create mapping object
        param = {} #dict(llcrnrlon=ll_lon, llcrnrlat=ll_lat, urcrnrlon=ur_lon, urcrnrlat=ur_lat)
        m = basemap.Basemap(projection="ortho", lat_0=0.0, lon_0=0.0)#, **param)
        
        # Properly scale angles
        x, y = m(align[:, 1], 90.0-align[:, 0])
        
        # Plot angular histogram
        im = m.scatter(x, y, s=s, marker="o", cmap=cm.cool, c=hist.astype(numpy.dtype), alpha=0.5)#, norm=matplotlib.colors.Normalize(vmin=0, vmax=1) )
        m.drawparallels(numpy.arange(-90.,120.,30.))
        m.drawmeridians(numpy.arange(0.,420.,60.))
        
        # Draw color bar
        #cb = m.colorbar(im, "right", size="3%", pad='1%')
        pyplot.show()
    
