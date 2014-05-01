''' Plot angular coverage

This script (`ara-coverage`) generates angular histogram tabulating the number
of projections within a discrete area on a spherical surface. This angular
histogram can be displayed in a number of ways:

 - 2D map projection (default)
 - 3D scatter plot
 - Cylinders surrounding a volume in Chimera
 
The number of projections is represented by both the color and size of circle/cylinder in each
representation. An 'x' is used to denote a view with no projections in the 2D project plot.

Examples
========

.. sourcecode :: sh

    $ ara-coverage relion_data.star -o plot.png
    
    $ ara-coverage relion_data.star -o plot.bild --chimera
    
    $ ara-coverage relion_data.star -o plot3d.png --projection 3d

Critical Options
================

.. program:: ara-coverage

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of input filenames
    If you use the parameters `-i` or `--inputfiles` the filenames may be comma or 
    space separated on the command line; they must be comma seperated in a configuration 
    file. Note, these flags are optional for input files; the filenames must be separated 
    by spaces. For a very large number of files (>5000) use `-i "filename*"`

.. option:: -o <FILENAME>, --output <FILENAME>
    
    Output template for the enumerated filename (e.g. mic_0000.mrc)

.. option:: -s <FILENAME>, --selection-file <FILENAME>
    
    Selection file for projections in alignment file

Plot Options
============

.. program:: ara-coverage

.. option:: -p, --projection <CHOICE>

    Map projection type. See below for more details on 
    available map projections.

.. option:: -d, --dpi <int>

    Resolution of the image in dots per inch

.. option:: --count-mode <Shape|Color|Both>
    
    Mode to measure the number of projections per view (Currently only works for map projection output)

.. option:: --hide-zero-marker <BOOL>
    
    Hide the zero markers (Currently does not work for Chimera BILD output)

.. option:: --color-map <CHOICE>
    
    Set the color map. See below for more details on 
    available color maps.

Histogram Options
=================

.. option:: --view-resolution <INT>
    
    Group views into a coarse grid: (2) 15 deg, (3) 7.5 deg ...

.. option:: --disable-mirror <BOOL>
    
    Disable mirroring over the equator for counting

Chimera Options
===============

.. option:: --chimera <BOOL>
    
    Write out Chimera bild file

.. option:: --particle-radius <FLOAT>
    
    Radius from center for ball projections

.. option:: --particle-center <FLOAT>
    
    Offset from center for ball projections (zero means use radius)

Projection Options
==================

.. option:: --area-mult <BOOL>
    
    Circle area multiplier

.. option:: --alpha <FLOAT>
    
    Transparency of the marker (1.0 = solid, 0.0 = no color)

.. option:: --label-view <LIST>
    
    List of views to label with number and Euler Angles (theta,phi)

.. option:: --use-scale <BOOL>
    
    Display scale and color instead of color bar

Layout Options
==============

.. option:: --lon-zero <FLOAT>
    
    Longitude for axis zero (empty for default values determined by projection)

.. option:: --lat-zero <FLOAT>
    
    Latitude for axis zero (empty for default values determined by projection)

.. option:: --ll-lat <FLOAT>
    
    Latitude of lower left hand corner of the desired map domain (degrees)

.. option:: --ll-lon <FLOAT>
    
    Longitude of lower left hand corner of the desired map domain (degrees)

.. option:: -ur-lat <FLOAT>
    
    Latitude of upper right hand corner of the desired map domain (degrees)

.. option:: --ur-lon <FLOAT>
    
    Longitude of upper right hand corner of the desired map domain (degrees)

.. option:: --boundinglat <FLOAT>
    
    Bounding latitude for npstere,spstere,nplaea,splaea,npaeqd,spaeqd

.. option:: --proj-width <INT>
    
    Width of desired map domain in projection coordinates (meters)

.. option:: --proj-height <INT>
    
    Height of desired map domain in projection coordinates (meters)

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`

Additional Option Information
=============================

Support color maps
------------------

Here is an limited selection of the available color maps:

.. image:: http://matplotlib.org/_images/colormaps_reference_00.png

.. note::
    
    A full description of the color maps can be found at:
    http://matplotlib.org/examples/color/colormaps_reference.html
          
Supported Projection Plots
---------------------------

The following are supported projections for the `:option:-p` option.

======    =================================
Option    Description
======    =================================
aeqd      Azimuthal Equidistant
poly      Polyconic
gnom      Gnomonic
moll      Mollweide
tmerc     Transverse Mercator
nplaea    North-Polar Lambert Azimuthal
gall      Gall Stereographic Cylindrical
mill      Miller Cylindrical
merc      Mercator
stere     Stereographic
npstere   North-Polar Stereographic
hammer    Hammer
geos      Geostationary
nsper     Near-Sided Perspective
vandg     van der Grinten
laea      Lambert Azimuthal Equal Area
mbtfpq    McBryde-Thomas Flat-Polar Quartic
sinu      Sinusoidal
spstere   South-Polar Stereographic
lcc       Lambert Conformal
npaeqd    North-Polar Azimuthal Equidistant
eqdc      Equidistant Conic
cyl       Cylindrical Equidistant
omerc     Oblique Mercator
aea       Albers Equal Area
spaeqd    South-Polar Azimuthal Equidistant
ortho     Orthographic
cass      Cassini-Soldner
splaea    South-Polar Lambert Azimuthal
robin     Robinson
======    =================================

.. note::
    
     For more information concerning the map projections, 
     see http://matplotlib.github.com/basemap/users/mapsetup.html

.. Created on Aug 27, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.util.matplotlib_nogui import pylab
from ..core.app import program
from ..core.metadata import format
from ..core.metadata import format_utility
from ..core.orient import healpix
from ..core.orient import spider_transforms
from mpl_toolkits import basemap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.cm
import matplotlib.lines
import matplotlib.font_manager
import scipy.io
import logging
import numpy
import os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, dpi, chimera=False, **extra):
    ''' Generate an angular histogram for the given alignment files
        
    :Parameters:
    
        files : list
                List of input alignment files
        output : str
                 Output filename for angular histogram
        dpi : float
              Dots per inch for plots
        chimera : bool
                  Output in chimera bild format
    '''
    
    outputn=output
    mapargs = projection_args(**extra) if extra['projection'].lower() != '3d' else None
    if chimera or mapargs is None: del extra['use_mirror']
    i=0
    for filename in files:
        if len(files) > 1:
            outputn = format_utility.new_filename(output, suffix=os.path.splitext(os.path.basename(filename))[0], ext='.png')
        angs = read_angles(filename, **extra)
        angs,cnt = angular_histogram(angs, **extra)
        if chimera:
            chimera_bild(angs, cnt, outputn, **extra)
        elif mapargs is None:
            fig = pylab.figure(i, dpi=dpi)
            scatterEuler3d(fig, angs, cnt, **extra)
            fig.savefig(outputn, dpi=dpi)
        else:
            _logger.info("%s has %d missing views"%(os.path.basename(filename), numpy.sum(cnt < 1)))
            fig = pylab.figure(i, dpi=dpi)
            fig.add_axes([0.05,0.05,0.9,0.9])
            plot_angles(angs, cnt, mapargs, **extra)
            fig.savefig(outputn, dpi=dpi)
    
    _logger.info("Completed")
    
def scatterEuler3d(fig, angs, cnt, color_map='cool', hide_zero_marker=False, **extra):
    ''' Plot the angular histogram using a 3D scatter plot
    
    :Parameters:
        
        fig : Figure
              Matplotlib figure handle
        angs : array
               Array of view angles
        cnt : array
              Histogram for each view angle 
        color_map : str
                    Name of color map
        hide_zero_marker : bool
                           If true, hide the zero projection count marker
        extra : dict
                Unused keyword arguments
    '''
    
    cmap = getattr(cm, color_map)
    cnt = cnt.astype(numpy.float)
    nhist = cnt.copy()
    if nhist.min() != nhist.max():
        nhist-=nhist.min()
        nhist/=nhist.max()
    
    ax = Axes3D(fig)
    data = numpy.zeros((len(angs), 3))
    for i in xrange(len(angs)):
        if i == 0: print angs[i, :]
        data[i, :] = spider_transforms.euler_to_vector(*angs[i, :])
    nonzero = numpy.nonzero(cnt)
    ax.scatter3D(data[nonzero, 0].ravel(), data[nonzero, 1].ravel(), data[nonzero, 2].ravel(), c=nhist, cmap=cmap)
    if not hide_zero_marker:
        ax.scatter3D(data[nonzero, 0].ravel(), data[nonzero, 1].ravel(), data[nonzero, 2].ravel(), color=cm.gray(0.5), marker='x') # @UndefinedVariable
        
def chimera_bild(angs, cnt, output, particle_radius=60.0, particle_center=0.0, radius_frac=0.3, width_frac=0.5, color_map='cool', view_resolution=3, **extra):
    '''Write out angular histogram has a Chimera BILD file
    
    :Parameters:
        
        angs : array
               Array of view angles
        cnt : array
              Histogram for each view angle 
        output : str
                 Output filename
        particle_radius : float
                          Radius of paritlce in angstroms
        particle_center : float
                          Ceneter of particle in angstroms
        radius_frac : float
                      Radius scaling factor
        width_frac : float
                     Cylinder width scaling factor
        color_map : str
                    Name of color map
        view_resolution : int
                          HealPix resolution
        extra : dict
                Unused keyword arguments
    '''
    
    #double offset = ori_size * pixel_size / 2.;
    cmap = getattr(cm, color_map)
    output = os.path.splitext(output)[0]+'.bild'
    fout = open(output, 'w')
    maxcnt = cnt.max()
    
    width = width_frac * numpy.pi*particle_radius/healpix.sampling(view_resolution)
    try:
        for i in xrange(len(angs)):
            val = cnt[i]/float(maxcnt)
            r, g, b = cmap(val)[:3]
            fout.write('.color %f %f %f\n'%(r, g, b))
            
            v1,v2,v3 = spider_transforms.euler_to_vector(*angs[i, :])
            length = particle_radius + radius_frac * particle_radius * val;
            diff = particle_radius-length
            if abs(diff*v1) < 0.01 and abs(diff*v2) < 0.01 and abs(diff*v3) < 0.01: continue
            fout.write('.cylinder %f %f %f %f %f %f %d\n'%(particle_radius*v1+particle_center, particle_radius*v2+particle_center, particle_radius*v3+particle_center,
                                               length*v1+particle_center, length*v2+particle_center, length*v3+particle_center,
                                               width))
    finally:
        fout.close()

def plot_angles(angs, hist, mapargs, color_map='cool', area_mult=1.0, alpha=0.9, hide_zero_marker=False, use_scale=False, label_view=[], **extra):
    ''' Plot the angular histogram using a map projection from basemap
    
    .. note::
         
        Basemap uses longitude latitude conventions, but the given angles are in
        colatitude, longitude convention.
    
    :Parameters:
        
        angs : array
               Array of view angles
        cnt : array
              Histogram for each view angle 
        mapargs : dict
                  Arguments specific to a map projection in basemap
        color_map : str
                    Name of color map
        area_mult : float
                    Scaling factor for size display
        alpha : float
                Transparency factor
        hide_zero_marker : bool
                           If true, hide the zero projection count marker
        use_scale : bool
                    If true, then display scale for size and color
        label_view : list
                     Label each view with text
        extra : dict
                Unused keyword arguments 
    '''
    
    cmap = getattr(cm, color_map)
    m = basemap.Basemap(**mapargs)
    
    # Y -> latitude
    # Z -> longitude
    
    #longitude, latitude = 90-colatitude
    x, y = m(angs[:, 1], 90.0-angs[:, 0])
    sel = hist < 1
    hist = hist.astype(numpy.float)
    s = numpy.sqrt(hist)*area_mult
    nhist = hist.copy()
    nhist-=nhist.min()
    nhist/=nhist.max()
    m.drawparallels(numpy.arange(-90.,120.,30.))
    m.drawmeridians(numpy.arange(0.,420.,60.))
    im = m.scatter(x, y, s=s, marker="o", c=cmap(nhist), alpha=alpha, edgecolors='none')
    
    font_tiny=matplotlib.font_manager.FontProperties()
    font_tiny.set_size('xx-small')
    
    if len(label_view) > 0:
        for i in label_view:
            if i > len(angs): 
                _logger.warn("Cannot label view: %d when there are only %d views (skipping)"%(i, len(angs)))
                continue
            ytext = -15 if i%2==0 else 15
            pylab.annotate('%d: %.1f,%.1f'%(i+1, angs[i, 0], angs[i, 1]), xy=(x[i],y[i]),  xycoords='data',
                                    xytext=(-15, ytext), textcoords='offset points',
                                    arrowprops=dict(arrowstyle="->"), fontproperties =font_tiny
                                    )
    
    if numpy.sum(sel) > 0 and not hide_zero_marker:
        im = m.scatter(x[sel], y[sel], numpy.max(s), marker="x", c=cm.gray(0.5))#@UndefinedVariable
    
    if not use_scale:
        im = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize())
        im.set_array(hist)
        m.colorbar(im, "right", size="3%", pad='1%')
    else:
        fontP=matplotlib.font_manager.FontProperties()
        fontP.set_size('small')
        inc = len(hist)/10
        lines = []
        labels = []
        idx = numpy.argsort(hist)[::-1]
        for j in xrange(0, len(hist), inc):
            i=idx[j]
            labels.append("%s"%hist[i])
            lines.append(matplotlib.lines.Line2D(range(1), range(1), color=cmap(nhist[i]), marker='o', markersize=s[i]/2, linestyle='none', markeredgecolor='white'))
        if numpy.sum(sel) > 0 and not hide_zero_marker:
            i=idx[len(idx)-1]
            labels.append("%s"%hist[i])
            lines.append(matplotlib.lines.Line2D(range(1), range(1), color=cm.gray(0.5), marker='x', markersize=numpy.max(s)/5, linestyle='none')) #@UndefinedVariable
        pylab.legend(tuple(lines),tuple(labels), numpoints=1, frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)
        #l.get_lines()[0]._legmarker.set_ms(numpy.max(s)) 

def projection_args(projection, lat_zero, lon_zero, ll_lon, ll_lat, ur_lon, ur_lat, proj_width=0, proj_height=0, boundinglat=0, **extra):
    ''' Get default values for various projection types
    
    :Parameters:
    
        projection : str
                     Name of the map projection mode
        lat_zero : float
                   Latitude for axis zero
        lon_zero : float
                   Longitude for axis zero
        ll_lon : float
                 Longitude of lower left hand corner of the desired map domain (degrees)
        ll_lat : float
                 Latitude of lower left hand corner of the desired map domain (degrees)
        ur_lon : float
                 Longitude of upper right hand corner of the desired map domain (degrees)
        ur_lat : float
                 Latitude of upper right hand corner of the desired map domain (degrees)
        proj_width : int
                     Width of desired map domain in projection coordinates (meters)
        proj_height : int
                      Height of desired map domain in projection coordinates (meters)
        boundinglat : int
                     Bounding latitude for npstere,spstere,nplaea,splaea,npaeqd,spaeqd
        extra : dict
                Unused keyword arguments 
        
    :Returns:
    
    args : dict
           Keyword dictionary for each parameter value pair
    '''
    
    if projection == 'hammer':
        lat_zero = -90.0 if not lat_zero else float(lat_zero)
        lon_zero = 90.0 if not lon_zero else float(lon_zero)
        boundinglat = 0 if not boundinglat else float(boundinglat)
    elif projection == 'ortho':
        lat_zero = 90.0 if not lat_zero else float(lat_zero)
        lon_zero = 0.0 if not lon_zero else float(lon_zero)
        boundinglat = 0 if not boundinglat else float(boundinglat)
    elif projection == 'npstere':
        lat_zero = 90.0 if not lat_zero else float(lat_zero)
        lon_zero = 1.0 if not lon_zero else float(lon_zero)
        boundinglat = 0.000000001 if not boundinglat else float(boundinglat)
    else:
        if not lat_zero or not lon_zero:
            _logger.warning("No default for %s projection - setting to zeros"%projection)
        lat_zero = 0.0 if not lat_zero else float(lat_zero)
        lon_zero = 0.0 if not lon_zero else float(lon_zero)
    _logger.info("Map Projection: %s"%projection)
    _logger.info("Longitude 0: %f - Latitude 0: %f - Bounding Lat: %f"%(lon_zero, lat_zero, boundinglat))
    param = dict(projection=projection, lat_0=lat_zero, lon_0=lon_zero, llcrnrlon=ll_lon, llcrnrlat=ll_lat, urcrnrlon=ur_lon, urcrnrlat=ur_lat, celestial=False)
    if proj_width > 0: param['width']=proj_width
    if proj_height > 0: param['height']=proj_height
    if boundinglat > 0: param['boundinglat'] = boundinglat
    return param
    
def angular_histogram(angs, view_resolution=3, disable_mirror=False, use_mirror=True, **extra):
    ''' Discretize the angles using healpix and tabulate an angular histogram
    
    .. todo:: add mode to scale by number of STDs
        
    :Parameters:
        
        angs : array
               Array of angles (theta, phi) in degrees
        view_resolution : int
                          Healpix resolution where (2) 15 deg, (3) 7.5 deg ...
        disable_mirror : bool
                         Use the full sphere for counting, not half sphere
        use_mirror : bool
                     Display views on both hemispheres
        extra : dict
                Unused keyword arguments 
    
    :Returns:
        
        angs : array
               Discretized angles
        count : array
                Number of projections for each angle
    '''
    
    if view_resolution == 0: return angs, numpy.ones(len(angs))
    total = healpix.res2npix(view_resolution, not disable_mirror)
    _logger.info("Healpix order %d gives %d views"%(view_resolution, total))
    total = healpix.res2npix(view_resolution)
    
    
    pix = healpix.ang2pix(view_resolution, numpy.deg2rad(angs))#,  half=not disable_mirror)
    count1 = numpy.bincount(pix)
    
    count = numpy.zeros(total, dtype=numpy.int)
    sel = numpy.nonzero(count1)[0]
    count[sel] = count1[sel]
    pix = numpy.arange(total, dtype=numpy.int)
    
    if not disable_mirror:
        mpix = healpix.pix2mirror(view_resolution, pix)
        for i in xrange(len(pix)):
            if i == mpix[i]: continue
            count[i] += count[mpix[i]]
            count[mpix[i]]=count[i]
    if not use_mirror:
        total = healpix.res2npix(view_resolution, True, True)
        pix = pix[:total]
        count=count[:total]
    
    #pix = numpy.nonzero(count)[0]
    #count = count[pix].copy().squeeze()
    
    angs = numpy.rad2deg(healpix.pix2ang(view_resolution, pix))
    return angs, count
    
def read_angles(filename, header=None, selection_file="", **extra):
    ''' Read in an alignment file and apply optional selection file
        
    :Parameters:
        
        filename : str
                   Filename for alignment file
        header : int
                 Header for alignment file
        selection_file : str
                         Selection file for projections in alignment file
        extra : dict
                Unused keyword arguments 
    
    :Returns:
        
        angles : array
                 Array of angles for each projection
    '''
    
    selection_file, sheader = format_utility.parse_header(selection_file)
    select = None
    if selection_file != "":
        if os.path.splitext(selection_file)[1]=='.mat':
            select = scipy.io.loadmat(selection_file)
            select = select[sheader[0]]
        else:
            select,sheader = format.read(selection_file, ndarray=True)
            select=select[:, sheader.index('id')]
    if format.get_format(filename) == format.star:
        align = format.read(filename, numeric=True)
        
        if select is None: 
            select = xrange(len(align))
            angles = numpy.zeros((len(align), 2))
        else: 
            select -= 1
            angles = numpy.zeros((len(select), 2))
        for j, i in enumerate(select):
            angles[j] = (align[i].rlnAngleTilt, align[i].rlnAngleRot)
    else:
        if select is not None:
            align = format.read_alignment(filename, header=header, map_ids='id')
            angles = numpy.zeros((len(select), 2))
            for j, i in enumerate(select):
                angles[j] = (align[i].theta, align[i].phi)
        else:
            align = format.read_alignment(filename, header=header)
            angles = numpy.zeros((len(align), 2))
            for i in xrange(len(align)):
                angles[i] = (align[i].theta, align[i].phi)
  
    return angles

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Plot", "Options to control creation of the coverage plot",  id=__name__)
    group.add_option("-p", projection="npstere",    help="Map projection type")
    group.add_option("-d", dpi=300,                 help="Resolution of the image in dots per inch")
    group.add_option("",   count_mode=('Shape', 'Color', 'Both'), help="Mode to measure the number of projections per view", default=2)
    group.add_option("",   hide_zero_marker=False,  help="Hide the zero markers")
    group.add_option("",   color_map='cool',        help="Set the color map")
    pgroup.add_option_group(group)
    
    group = OptionGroup(parser, "Histogram", "Options to control angular histogram")
    group.add_option("-r", view_resolution=3,       help="Group views into a coarse grid: (2) 15 deg, (3) 7.5 deg ...")
    group.add_option("",   disable_mirror=False,    help="Disable mirroring over the equator for counting")
    pgroup.add_option_group(group)
    
    group = OptionGroup(parser, "Chimera", "Options to control chimera bild output")
    group.add_option("", chimera=False,             help="Write out Chimera bild file")
    group.add_option("", particle_radius=320.0,     help="Radius from center for ball projections")
    group.add_option("", particle_center=0.0,       help="Offset from center for ball projections")
    pgroup.add_option_group(group)
    
    group = OptionGroup(parser, "Projection", "Options to control map projection output")
    group.add_option("-a", area_mult=1.0,           help="Circle area multiplier")
    group.add_option("",   alpha=0.9,               help="Transparency of the marker (1.0 = solid, 0.0 = no color)")
    group.add_option("",   label_view=[],           help="List of views to label with number and Euler Angles (theta,phi)")
    group.add_option("",   use_scale=False,         help="Display scale and color instead of color bar")
    group.add_option("",   use_mirror=False,        help="Display projections on both hemispheres")
    pgroup.add_option_group(group)
    
    group = OptionGroup(parser, "Layout", "Options to control the projection layout")
    group.add_option("",   lon_zero="",             help="Longitude for axis zero (empty for default values determined by projection)")
    group.add_option("",   lat_zero="",             help="Latitude for axis zero (empty for default values determined by projection)")
    group.add_option("",   ll_lat=-90.0,            help="Latitude of lower left hand corner of the desired map domain (degrees)")
    group.add_option("",   ll_lon=-180.0,           help="Longitude of lower left hand corner of the desired map domain (degrees)")
    group.add_option("",   ur_lat=90.0,             help="Latitude of upper right hand corner of the desired map domain (degrees)")
    group.add_option("",   ur_lon=180.0,            help="Longitude of upper right hand corner of the desired map domain (degrees)")
    group.add_option("",   boundinglat="",          help="Bounding latitude for npstere,spstere,nplaea,splaea,npaeqd,spaeqd")
    group.add_option("",   proj_width=0,            help="Width of desired map domain in projection coordinates (meters)")
    group.add_option("",   proj_height=0,           help="Height of desired map domain in projection coordinates (meters)")
    pgroup.add_option_group(group)
    
    if main_option:
        pgroup.add_option("-i", input_files=[],     help="List of filenames for the input stacks or selection file", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",          help="Output filename for the relion selection file", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-s", selection_file="",  help="Selection file", gui=dict(filetype="open"))
        parser.change_default(log_level=3)

def check_options(options, main_option=False):
    #Check if the option values are valid
    
    from ..core.app.settings import OptionValueError
    if options.view_resolution < 1: raise OptionValueError, "--view-resolution must have a value greater than 0, found: %d"%options.view_resolution
    
    if options.chimera:
        if options.particle_radius == 0: raise OptionValueError, "--particle-radius must be greater than 0"
        if options.particle_center == 0:
            options.particle_center = options.particle_radius
    if options.boundinglat:
        try:float(options.boundinglat)
        except: raise OptionValueError, "--boundinglat must be a floating point number"
    if options.lon_zero:
        try:float(options.lon_zero)
        except: raise OptionValueError, "--lon-zero must be a floating point number"
    if options.lat_zero:
        try:float(options.lat_zero)
        except: raise OptionValueError, "--lat-zero must be a floating point number"
    
    maps=[m for m in matplotlib.cm.datad if not m.endswith("_r")]
    if options.color_map not in set(maps):
        raise OptionValueError, "%s is not a supported color map for --color-map, supported colormaps are\n%s"%(options.color_map, "\n".join(maps))
    if options.projection not in set(basemap._projnames.keys()) and options.projection != '3d': 
        raise OptionValueError, "%s is not a supported projection for --projection, supported projections are\n %s"%(options.projection, basemap.supported_projections)
    '''
    projection_params = {'cyl'      : 'corners only (no width/height)',
             'merc'     : 'corners plus lat_ts (no width/height)',
             'tmerc'    : 'lon_0,lat_0',
             'omerc'    : 'lon_0,lat_0,lat_1,lat_2,lon_1,lon_2,no_rot',
             'mill'     : 'corners only (no width/height)',
             'gall'     : 'corners only (no width/height)',
             'lcc'      : 'lon_0,lat_0,lat_1,lat_2',
             'laea'     : 'lon_0,lat_0',
             'nplaea'   : 'bounding_lat,lon_0,lat_0,no corners or width/height',
             'splaea'   : 'bounding_lat,lon_0,lat_0,no corners or width/height',
             'eqdc'     : 'lon_0,lat_0,lat_1,lat_2',
             'aeqd'     : 'lon_0,lat_0',
             'npaeqd'   : 'bounding_lat,lon_0,lat_0,no corners or width/height',
             'spaeqd'   : 'bounding_lat,lon_0,lat_0,no corners or width/height',
             'aea'      : 'lon_0,lat_0,lat_1',
             'stere'    : 'lon_0,lat_0,lat_ts',
             'npstere'  : 'bounding_lat,lon_0,lat_0,no corners or width/height',
             'spstere'  : 'bounding_lat,lon_0,lat_0,no corners or width/height',
             'cass'     : 'lon_0,lat_0',
             'poly'     : 'lon_0,lat_0',
             'ortho'    : 'lon_0,lat_0,llcrnrx,llcrnry,urcrnrx,urcrnry,no width/height',
             'geos'     : 'lon_0,satellite_height,llcrnrx,llcrnry,urcrnrx,urcrnry,no width/height',
             'nsper'    : 'lon_0,satellite_height,llcrnrx,llcrnry,urcrnrx,urcrnry,no width/height',
             'sinu'     : 'lon_0,lat_0,no corners or width/height',
             'moll'     : 'lon_0,lat_0,no corners or width/height',
             'hammer'   : 'lon_0,lat_0,no corners or width/height',
             'robin'    : 'lon_0,lat_0,no corners or width/height',
             'vandg'    : 'lon_0,lat_0,no corners or width/height',
             'mbtfpq'   : 'lon_0,lat_0,no corners or width/height',
             'gnom'     : 'lon_0,lat_0',
             }
    '''
    

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = ''' Plot angular coverage on a map projection
                         
                         Example: Map a relion star file
                         $ ara-coverage data.star -o plot.png
                         
                         Example: Chimera Bild file
                         $ ara-coverage data.star -o plot.bild --chimera
                         
                         Example: 3D Scatter
                         $ ara-coverage data.star -o plot.png --projection 3d
                      ''',
        supports_MPI = False,
        use_version = False,
    )
def dependents(): return []
if __name__ == "__main__": main()


