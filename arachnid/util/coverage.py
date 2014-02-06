''' Plot angular coverage

This script (`ara-selrelion`) generates angular histogram mapped to a 2D projection of a 3D sphere and writes it out
as a color image. It both the size and color of a circle indicate the number of projections in a view and uses an 'x'
to denote a view with no projections.

todo add 3d scatter in plot_angles

>>> reset
>>> turn y theta coordinatesystem #0 # Rotating frame like SPIDER
>>> turn z phi coordinatesystem #0
>>> turn x 180

#open #0 spider:~/Desktop/autopick/enh_25_r7_05.ter

SPIDER:  ZYZ rotating frame
CHIMERA: XYZ rotating frame

#http://plato.cgl.ucsf.edu/pipermail/chimera-users/attachments/20080429/aaf842f9/attachment.ksh

.. Created on Aug 27, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.util.matplotlib_nogui import pylab
from ..core.app import program
from ..core.metadata import format
from ..core.metadata import format_utility
from ..core.orient import healpix
from ..core.orient import spider_transforms
import scipy.io

from mpl_toolkits import basemap
import matplotlib.cm as cm
import matplotlib.cm
import matplotlib.lines
import matplotlib.font_manager


import logging
import numpy
import os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, dpi, chimera=False, **extra):
    '''
    '''
    
    outputn=output
    mapargs = projection_args(**extra)
    i=0
    for filename in files:
        if len(files) > 1:
            outputn = format_utility.new_filename(output, suffix=os.path.splitext(os.path.basename(filename))[0], ext='.png')
        angs = read_angles(filename, **extra)
        if chimera:
            chimera_balls(angs, outputn, **extra)
        else:
            angs,cnt = count_angles(angs, **extra)
            _logger.info("%s has %d missing views"%(os.path.basename(filename), numpy.sum(cnt < 1)))
            fig = pylab.figure(i, dpi=dpi)
            fig.add_axes([0.05,0.05,0.9,0.9])
            plot_angles(angs, cnt, mapargs, **extra)
            fig.savefig(outputn, dpi=dpi)
    
    _logger.info("Completed")
    
def chimera_balls(angs, output, view_resolution=3, disable_mirror=False, ball_radius=60, ball_center=0, ball_size=1.0, mirror=False, count_mode=2, color_map='cool', **extra):
    '''
    '''
    
    pix = healpix.ang2pix(view_resolution, numpy.deg2rad(angs),  half=not disable_mirror)
    total = healpix.ang2pix(view_resolution, numpy.deg2rad(healpix.angles(view_resolution))[:, 1:], half=not disable_mirror).max()+1
    count = numpy.histogram(pix, total)[0]
    maxcnt = count.max()
    pix = numpy.arange(total, dtype=numpy.int)
    angs = numpy.rad2deg(healpix.pix2ang(view_resolution, pix))
    _logger.info("Number of angles %d for resolution %d"%(len(angs), view_resolution))
    fout = open(output, 'w')
    
    # Todo empty shape
    # Fix orientation
    
    cmap = getattr(cm, color_map)
    if count_mode == 0:
        fout.write('.color 1 0 0\n')
    for i in xrange(len(angs)):
        
        v1,v2,v3 = spider_transforms.euler_to_vector(angs[i, 1], angs[i, 0])
        ncnt = count[i]/float(maxcnt)
        if count_mode != 0:
            r, g, b = cmap(ncnt)[:3]
            fout.write('.color %f %f %f\n'%(r, g, b))
            if count_mode == 1: ncnt=1.0
            
        fout.write('.sphere %f %f %f %f\n'%(v1*ball_radius+ball_center, v2*ball_radius+ball_center, v3*ball_radius+ball_center, ncnt*ball_size))
        if mirror:
            v1,v2,v3 = spider_transforms.euler_to_vector(angs[i, 1], 180+angs[i, 0])
            fout.write('.sphere %f %f %f %f\n'%(v1*ball_radius+ball_center, v2*ball_radius+ball_center, v3*ball_radius+ball_center, ncnt*ball_size))
            
    fout.close()

def plot_angles(angs, hist, mapargs, color_map='cool', area_mult=1.0, alpha=0.9, hide_zero_marker=False, use_scale=False, label_view=[], **extra):
    '''
    '''
    
    cmap = getattr(cm, color_map)
    m = basemap.Basemap(**mapargs)
    
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
    param = dict(projection=projection, lat_0=lat_zero, lon_0=lon_zero, llcrnrlon=ll_lon, llcrnrlat=ll_lat, urcrnrlon=ur_lon, urcrnrlat=ur_lat)
    if proj_width > 0: param['width']=proj_width
    if proj_height > 0: param['height']=proj_height
    if boundinglat > 0: param['boundinglat'] = boundinglat
    return param
    
def count_angles(angs, view_resolution=3, disable_mirror=False, **extra):
    '''
    '''
    
    pix = healpix.ang2pix(view_resolution, numpy.deg2rad(angs),  half=not disable_mirror)
    total = healpix.ang2pix(view_resolution, numpy.deg2rad(healpix.angles(view_resolution))[:, 1:], half=not disable_mirror).max()+1
    _logger.info("Healpix order %d gives %d views"%(view_resolution, total))
    count = numpy.histogram(pix, total)[0]
    pix = numpy.arange(total, dtype=numpy.int)
    angs = numpy.rad2deg(healpix.pix2ang(view_resolution, pix))
    return angs, count
    
def read_angles(filename, header=None, select_file="", **extra):
    '''
    '''
    
    select_file, header = format_utility.parse_header(select_file)
    select = None
    if select_file != "":
        if os.path.splitext(select_file)[1]=='.mat':
            select = scipy.io.loadmat(select_file)
            select = select[header[0]]
        else:
            select,header = format.read(select_file, ndarray=True)
            select=select[:, header.index('id')]
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
    group = OptionGroup(parser, "Coverage", "Options to control creation of the coverage plot",  id=__name__)
    group.add_option("-p", projection="npstere",    help="Map projection type")
    group.add_option("-d", dpi=300,                 help="Resolution of the image in dots per inch")
    group.add_option("-r", view_resolution=3,       help="Group views into a coarse grid: (2) 15 deg, (3) 7.5 deg ...")
    group.add_option("-a", area_mult=1.0,           help="Cirle area multiplier")
    group.add_option("", disable_mirror=False,      help="Disable mirroring over the equator for counting")
    group.add_option("", mirror=False,              help="Mirroring over the equator for visualization")
    group.add_option("", count_mode=('Shape', 'Color', 'Both'),              help="Mirroring over the equator for visualization", default=2)
    group.add_option("", hide_zero_marker=False,    help="Hide the zero markers")
    group.add_option("", color_map='cool',          help="Set the color map")
    group.add_option("", alpha=0.9,                 help="Transparency of the marker (1.0 = solid, 0.0 = no color)")
    group.add_option("", use_scale=False,           help="Display scale and color instead of color bar")
    group.add_option("", label_view=[],             help="List of views to label with number and Euler Angles (theta,phi)")
    group.add_option("", chimera=False,             help="Write out Chimera bild file")
    group.add_option("", ball_radius=60,            help="Radius from center for ball projections")
    group.add_option("", ball_size=1.0,             help="Size of largest ball projection")
    group.add_option("", ball_center=0,             help="Offset from center for ball projections")
    group.add_option("", select_file="",            help="Selection file", gui=dict(filetype="open"))
    
    pgroup.add_option_group(group)
    
    group = OptionGroup(parser, "Projection", "Options to control the projection",  id=__name__)
    group.add_option("",   lon_zero="",             help="Longitude for axis zero (empty for default values determined by projection)")
    group.add_option("",   lat_zero="",             help="Latitude for axis zero (empty for default values determined by projection)")
    group.add_option("",   ll_lat=-90.0,            help="Latitude of lower left hand corner of the desired map domain (degrees)")
    group.add_option("",   ll_lon=-180.0,           help="Longitude of lower left hand corner of the desired map domain (degrees)")
    group.add_option("",   ur_lat=90.0,             help="Longitude of upper right hand corner of the desired map domain (degrees)")
    group.add_option("",   ur_lon=180.0,            help="Latitude of upper right hand corner of the desired map domain (degrees)")
    group.add_option("",   boundinglat="",          help="Bounding latitude for npstere,spstere,nplaea,splaea,npaeqd,spaeqd")
    group.add_option("",   proj_width=0,            help="Width of desired map domain in projection coordinates (meters)")
    group.add_option("",   proj_height=0,           help="Height of desired map domain in projection coordinates (meters)")
    pgroup.add_option_group(group)
    
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input stacks or selection file", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the relion selection file", gui=dict(filetype="save"), required_file=True)
        parser.change_default(log_level=3)

def check_options(options, main_option=False):
    #Check if the option values are valid
    
    from ..core.app.settings import OptionValueError
    if options.view_resolution < 1: raise OptionValueError, "--view-resolution must have a value greater than 0, found: %d"%options.view_resolution
    
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
    if options.projection not in set(basemap._projnames.keys()): 
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
        
                         http://
                         
                         Example: Map a relion star file
                         
                         $ ara-coverage data.star -o plot.png
                         
                        For more information:
                        
                        http://matplotlib.github.com/basemap/users/mapsetup.html
                        
                        Supported Projection Plots
                        ---------------------------
                        aeqd    Azimuthal Equidistant
                        poly    Polyconic
                        gnom    Gnomonic
                        moll    Mollweide
                        tmerc    Transverse Mercator
                        nplaea    North-Polar Lambert Azimuthal
                        gall    Gall Stereographic Cylindrical
                        mill    Miller Cylindrical
                        merc    Mercator
                        stere    Stereographic
                        npstere    North-Polar Stereographic
                        hammer    Hammer
                        geos    Geostationary
                        nsper    Near-Sided Perspective
                        vandg    van der Grinten
                        laea    Lambert Azimuthal Equal Area
                        mbtfpq    McBryde-Thomas Flat-Polar Quartic
                        sinu    Sinusoidal
                        spstere    South-Polar Stereographic
                        lcc    Lambert Conformal
                        npaeqd    North-Polar Azimuthal Equidistant
                        eqdc    Equidistant Conic
                        cyl    Cylindrical Equidistant
                        omerc    Oblique Mercator
                        aea    Albers Equal Area
                        spaeqd    South-Polar Azimuthal Equidistant
                        ortho    Orthographic
                        cass    Cassini-Soldner
                        splaea    South-Polar Lambert Azimuthal
                        robin    Robinson
                      ''',
        supports_MPI = False,
        use_version = False,
    )
def dependents(): return []
if __name__ == "__main__": main()


