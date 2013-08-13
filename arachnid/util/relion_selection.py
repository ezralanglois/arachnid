''' Generate selection files to interface SPIDER and RELION

This script (`ara-selrelion`) can generate a RELION selection file from a list of stacks, a defocus
file and a SPIDER params file. It can also take a RELION selection file and a SPIDER selection file
and generate a new RELION selection file. Finally, it can generate a set of SPIDER selection files (by micrograph)
from a RELION selection file.

Tips
====

 #. Input filenames: To specifiy multiple files, either use a selection file `--selection-doc sel01.spi` with a single input file mic_00001.spi or use a single star mic_*.spi to
    replace the number.
 
 #. All other filenames: Must follow the SPIDER format with a number before the extension, e.g. mic_00001.spi. Output files just require the number of digits: `--output sndc_0000.spi`
 
 #. A stack that is not found in the defocus file is skipped. If your RELION selection file is empty, then you are likly extracting the micrograph ID from the wrong 
    column, use `--defocus-header` to select the proper column.

 #. The input file determines how the script will work, if the input is a stack or set of stacks, then it will generate a relion selection file from those stacks. If
    the input file is a relion selection file, then it will generate a new set of selection files based on the extension of the output file.

 #. An output file with the star extension, will write out a RELION star file, all other extensions will write out a SPIDER selection file

Running Script
===============

.. sourcecode :: sh
    
    # Source AutoPart - FrankLab only
    
    $ source /guam.raid.cluster.software/arachnid/arachnid.rc
    
    # Example usage - Generate a ReLion selection file from a set of stacks
    
    $ ara-selrelion win/win_*.spi --defocus-file def_avg.spi --param-file params.spi -o relion_selection.star
    
    # Example usage - Generate a ReLion selection file from a set of stacks with a stack (or micrograph or power spectra) selection file (first column holds the ID key)
    
    $ ara-selrelion win/win_*.spi --defocus-file def_avg.spi --param-file params.spi -o relion_selection.star -s select_file.spi=id:0
    
    # Example usage - Generate a ReLion selection file from a set of stacks with manual selection
    
    $ ara-selrelion win/win_*.spi --defocus-file def_avg.spi --param-file params.spi -o relion_selection.star --good good_00000.spi
    
    # Example usage - Generate a ReLion selection file from a set of stacks with manual selection where selection was produced by spider and does not have a header
    
    $ ara-selrelion win/win_*.spi --defocus-file def_avg.spi --param-file params.spi -o relion_selection.star --good good_00000.spi=id,select
    
    # Example usage - Generate a ReLion selection file from a SPIDER selection file and a Relion selection file
    
    $ ara-selrelion relion_selection_full.star --select good_classavg01.spi -o relion_selection_subset.star
    
    # Example usage - Generate SPIDER selection files by micrograph from a Relion and SPIDER selection file
    
    $ ara-selrelion relion_selection_full.star --select good_classavg01.spi -o good_000001.spi


Critical Options
================

.. program:: ara-selrelion

.. option:: -i <filename1,filename2>, --input-files <filename1,filename2>, filename1 filename
    
    List of filenames for the input micrographs.
    If you use the parameters `-i` or `--inputfiles` they must be comma separated 
    (no spaces). If you do not use a flag, then separate by spaces. For a 
    very large number of files (>5000) use `-i "filename*"`

.. option:: -o <str>, --output <str>
    
    Output filename for the relion selection file

.. option:: -p <str>, --param-file  <str>
    
    SPIDER parameters file (Only required when the input is a stack)
    
.. option:: -d <str>, --defocus-file <str>
    
    SPIDER defocus file (Only required when the input is a stack)
    
.. option:: -s <str>, --select <str>
    
    SPIDER micrograph or class selection file - if select file does not have proper header, then use `--select filename=id` or `--select filename=id,select`
    
.. option:: -g <str>, --good <str>
    
    SPIDER particle selection file (used when creating a new relion selection file) - if select file does not have proper header, then use `--select filename=id` or `--select filename=id,select`
    
Useful Options
==============
    
.. option:: -l <str>, --defocus-header <str>
    
    Column location for micrograph id and defocus value (Default: id:0,defocus:)
    
    Example defocus file
    
    | ;             id     defocus   astig_ang   astig_mag cutoff_freq
    | 1  5       21792     29654.2     34.2511     409.274     0.20764
    | 2  5       21794     32612.5     41.0366     473.659    0.201649

.. option:: -m <INT>, --minimum-group <INT>
    
    Minimum number of particles per defocus group (regroups using the micrograph name)

.. option:: --stack-file <STR>

    Used to rename the stack portion of the image name (rlnImageName); ignored when creating a relion file

.. option:: --scale <FLOAT>
    
    Used to scale the translations in a relion file (Default: 1.0)

.. option:: --column <str>
    
    Column name in relion file for selection, e.g. rlnClassNumber to select classes (Default: rlnClassNumber)

Other Options
=============

This is not a complete list of options available to this script, for additional options see:

    #. :ref:`Options shared by all scripts ... <shared-options>`
    
.. Created on Nov 27, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..core.util.matplotlib_nogui import pylab
from ..core.app.program import run_hybrid_program
from ..core.metadata import spider_utility, format_utility, format, spider_params
from ..core.image import eman2_utility, ndimage_file #, ndimage_utility
from ..core.parallel import parallel_utility
from ..core.orient import healpix
import numpy, os, logging, operator, glob

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, refine_spi=False, views=0, frame="", **extra):
    '''Generate a relion selection file
    
    :Parameters:
    
    filename : tuple
               File index and input filename
    extra : dict
            Unused key word arguments
    '''
    
    if not format.is_readable(files[0]) and ndimage_file.is_readable(files[0]):
        _logger.info("Generating a relion selection file from a set of stacks")
        img = ndimage_file.read_image(files[0])
        #img = ndimage_file.read_image(files[0])
        generate_relion_selection_file(files, img, **extra)
        
        #relion_gui3dauto.settings
        #generate_settings(**extra)
        
    else:
        _logger.info("Transforming a relion selection file: %d"%len(files))
        vals = []
        for f in files:
            try:
                vals.append(format.read(f, numeric=True))
            except:
                raise ValueError, "Input not an image or a selection file"
        if len(vals) > 1:
            vals = format_utility.combine(vals)
        else: vals = vals[0]
        vals = select_good(vals, **extra)
        
        if frame != "":
            create_movie(vals, frame, **extra)
        elif views > 0:
            view_distrubtion(vals, views, **extra)
        elif refine_spi:
            create_refinement(vals, **extra)
        else:
            select_class_subset(vals, **extra)
    _logger.info("Completed")
    
def select_good(vals, good, **extra):
    '''
    '''
    
    if good == "": return vals
    
    last=None
    subset=[]
    for v in vals:
        mic,pid1 = spider_utility.relion_id(v.rlnImageName)
        if mic != last:
            try:
                select_vals = set([s.id for s in format.read(good, spiderid=mic, numeric=True)])
            except:select_vals=set()
            last=mic
        if pid1 in select_vals:
            subset.append(v)
    if len(subset) == 0: raise ValueError, "Nothing selected from %s"%good
    return subset
        
    
    
def create_movie(vals, frame, output, frame_limit=0, **extra):
    '''
    '''
    
    _logger.info("Creating movie mode relion selection file: %d"%frame_limit)
    frame_vals = []
    last = -1
    idlen=None
    consecutive=None
    last=-1
    for v in vals:
        mic,pid1 = spider_utility.relion_id(v.rlnImageName)
        if mic != last:
            frames = glob.glob(spider_utility.spider_filename(frame, mic))
            if consecutive is None:
                avg_count = ndimage_file.count_images(spider_utility.relion_file(v.rlnImageName, True))
                frm_count = ndimage_file.count_images(frames[0])
                consecutive = avg_count != frm_count
                if consecutive:
                    _logger.info("Detected fewer particles in stack %s - %d < %d"%(v.rlnImageName, frm_count, avg_count))
            last=mic
            if consecutive: pid=0
            if idlen is None:
                idlen = len(str(len(vals)*len(frames)))
        if consecutive: pid += 1
        else: pid=pid1
        frames = sorted(frames)
        if frame_limit == 0: frame_limit=len(frames)
        if len(frames) < frame_limit:
            _logger.warn("Skipping %s - too few frames: %d < %d"%(v.rlnImageName, len(frames), frame_limit))
            continue
        frames=frames[:frame_limit]
        for f in frames:
            frame_vals.append(v._replace(rlnImageName="%s@%s"%(str(pid).zfill(idlen), f))+(v.rlnImageName, ))
    header = list(vals[0]._fields)
    header.append('rlnParticleName')
    format.write(output, frame_vals, header=header)

    
def view_distrubtion(vals, views, output, column, select="", **extra):
    '''
    '''
    
    if select != "":
        subset=[]
        try: select=int(select)
        except:
            if select.find(",") != -1:
                select = set([int(v) for v in select.split(',')])
            else:
                select = format.read(select, numeric=True)
                select = set([s.id for s in select if s.select > 0])
        else: select=set([select])
        _logger.info("Selecting classes: %s"%str(select))
        for v in vals:
            id = getattr(v, column)
            try: id = int(id)
            except: id = spider_utility.spider_id(id)
            if id in select: subset.append(v)
        if len(subset) == 0: raise ValueError, "No classes selected"
        vals = subset
    
    try:
        grouping = numpy.asarray([getattr(v, column) for v in vals], dtype=numpy.int)
        groups = numpy.unique(grouping)
    except: groups=[]
    if len(groups) > 1:
        ref_hist = numpy.zeros((healpix.res2npix(views), len(groups)+1))
        ref=healpix.ang2pix(views, [(numpy.deg2rad(v.rlnAngleTilt), numpy.deg2rad(v.rlnAngleRot)) for v in vals])
        
        if 1 == 0:
            angmap = {}
            for i in xrange(len(vals)):
                v=vals[i]
                if v.rlnAngleTilt not in angmap: angmap[v.rlnAngleTilt]={}
                if v.rlnAngleRot not in angmap[v.rlnAngleTilt]: angmap[v.rlnAngleTilt][v.rlnAngleRot]=[]
                if len(angmap[v.rlnAngleTilt][v.rlnAngleRot]) > 0:
                    idx=angmap[v.rlnAngleTilt][v.rlnAngleRot][0]
                    top=ref[idx]
                    if ref[i] != top:
                        _logger.error("%d == %d | %f, %f -- %f,%f"%(ref[i], top, v.rlnAngleTilt, v.rlnAngleRot, vals[idx].rlnAngleTilt, vals[idx].rlnAngleRot))
                    assert(ref[i] == top)
                angmap[v.rlnAngleTilt][v.rlnAngleRot].append(i)
                
        
        ref_hist[:, 0] = numpy.histogram(ref, healpix.res2npix(views))[0]
        gref = [ref.astype(numpy.float)]
        for i in xrange(len(groups)):
            ref_hist[:, i+1] = numpy.histogram(ref[groups[i]==grouping], healpix.res2npix(views))[0]
            gref.append(ref[groups[i]==grouping].astype(numpy.float))
        
        for i in xrange(len(ref_hist)):
            _logger.info("%d: %s"%(i+1, ",".join([str(v) for v in ref_hist[i]])))
        format.write(output, ref, header=['ref'])
        header = [('c_%d'%(i+1)) for i in xrange(len(groups))]
        format.write(output, numpy.hstack((numpy.arange(1, len(ref_hist)+1)[:, numpy.newaxis], ref_hist)), header=['id', 'full']+header, prefix='hist_text_')
        
        if pylab is not None:
            '''
            n, bins, patches = P.hist(x, 10, normed=1, histtype='bar',
                                color=['crimson', 'burlywood', 'chartreuse'],
                                label=['Crimson', 'Burlywood', 'Chartreuse'])
            '''
            pylab.clf()
            pylab.figure(figsize=(10*len(gref), 6))
            bins=healpix.res2npix(views)
            #bins=10
            pylab.hist(gref, bins, label=['full']+header, histtype='bar', fill=True)#, color=color
            pylab.legend()
            pylab.savefig(format_utility.new_filename(output, ext="png"), dpi=300)
    else:
        ref=healpix.ang2pix(views, [(numpy.deg2rad(v.rlnAngleTilt), numpy.deg2rad(v.rlnAngleRot)) for v in vals])
        ref_hist = numpy.histogram(ref, healpix.res2npix(views))[0]
        for i in xrange(len(ref_hist)):
            _logger.info("%d: %d"%(i+1, ref_hist[i]))
        _logger.info("Number of unique: %d"%len(numpy.unique(ref)))
        _logger.info("Number of expected: %d"%healpix.res2npix(views))
        
        format.write(output, ref, header=['ref'])
        format.write(output, numpy.hstack((numpy.arange(1, len(ref_hist)+1)[:, numpy.newaxis], ref_hist[:, numpy.newaxis])), header=['id', 'count'], prefix='hist_text_')
        
        if pylab is not None:
            '''
            n, bins, patches = P.hist(x, 10, normed=1, histtype='bar',
                                color=['crimson', 'burlywood', 'chartreuse'],
                                label=['Crimson', 'Burlywood', 'Chartreuse'])
            '''
            pylab.clf()
            pylab.hist(ref, healpix.res2npix(views))
            pylab.savefig(format_utility.new_filename(output, ext="png"), dpi=200)

def create_refinement(vals, output, **extra):
    '''
    '''
    
    #spider_params.write(os.path.join(os.path.dirname(output), 'params'+os.path.splitext(output)[1]), 0.0, vals[0].rlnVoltage, rlnSphericalAberration, pixel_diameter, window_size=window_size, cs=vals[0].rlnAmplitudeContrast)
    align = numpy.zeros((len(vals), 18))
    align[:, 4] = numpy.arange(1, len(vals)+1)
    for i in xrange(len(vals)):
        mic,par = spider_utility.relion_id(vals[i].rlnImageName)
        align[i, 15] = mic
        align[i, 16] = par
        align[i, 17] = vals[i].rlnDefocusU
    format.write(output, align, header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,stack_id,defocus".split(','), format=format.spiderdoc) 
    
def select_class_subset(vals, select, output, column="rlnClassNumber", random_subset=0, view_resolution=0, **extra):
    ''' Select a subset of classes and write a new selection file
    
    :Parameter:
    
    vals : list
           List of entries from a selection file
    select : str
             Filename for good class selection file
    output : str
            Filename for output selection file
    extra : dict
            Unused key word arguments
    '''
    
    if len(vals) == 0: raise ValueError, "No values read"
    
    tmp = numpy.asarray(vals)
    header = list(vals[0]._fields)
    if column in set(header):
        tmp = numpy.asarray([getattr(v, column) for v in vals])
        clazzes = numpy.unique(tmp)
        for cl in clazzes:
            _logger.info("Class: %d has %d projections"%(cl, numpy.sum(cl==tmp)))
    
    if select != "" and not isinstance(select, list):
        subset=[]
        try: select=int(select)
        except:
            if select.find(",") != -1:
                select = set([int(v) for v in select.split(',')])
            else:
                select = format.read(select, numeric=True)
                select = set([s.id for s in select if s.select > 0])
        else: select=set([select])
        _logger.info("Selecting classes: %s"%str(select))
        for v in vals:
            id = getattr(v, column)
            try: id = int(id)
            except: id = spider_utility.spider_id(id)
            if id in select: subset.append(v)
        if len(subset) == 0: raise ValueError, "No classes selected"
    else: subset = vals
    
    if view_resolution > 0:
        n=healpix.res2npix(view_resolution)
        _logger.info("Culling %d views with resolution %d"%(n, view_resolution))
        ang = numpy.asarray([(v.rlnAngleTilt, v.rlnAngleRot) for v in subset])
        view = healpix.ang2pix(view_resolution, numpy.deg2rad(ang))
        vhist = numpy.histogram(view, n)[0]
        maximum_views = numpy.median(vhist)
        _logger.info("Maximum of %d projections allowed per view"%(maximum_views))
        assert(vhist[1] == numpy.sum(view==1))
        assert(vhist[20] == numpy.sum(view==20))
        assert(vhist[30] == numpy.sum(view==30))
        vals = subset
        subset = []
        idx = numpy.arange(len(vals), dtype=numpy.int)
        numpy.random.shuffle(idx)
        count = numpy.zeros(n)
        for i in idx:
            if count[view[i]] < maximum_views:
                subset.append(vals[i])
                count[view[i]] += 1
        _logger.info("Reduced projections from %d to %d"%(len(vals), len(subset)))
    
    defocus_dict = read_defocus(**extra)
    if len(defocus_dict) > 0:
        for i in xrange(len(subset)):
            mic,par = spider_utility.relion_id(subset[i].rlnImageName)
            subset[i] = subset[i]._replace(rlnDefocusU=defocus_dict[mic].defocus)
    
    if os.path.splitext(output)[1] == '.star':
        if random_subset > 1: 
            index = numpy.arange(len(subset), dtype=numpy.int)
            numpy.random.shuffle(index)
            index_sets = parallel_utility.partition_array(index, random_subset)
            for i, index in enumerate(index_sets):
                curr_subset=[subset[j].__class__(*subset[j]) for j in index]
                groupmap = regroup(build_group(curr_subset), **extra)
                update_parameters(curr_subset, list(curr_subset[0]._fields), groupmap, **extra)
                format.write(output, curr_subset, spiderid=(i+1))
        else:
            groupmap = regroup(build_group(subset), **extra)
            update_parameters(subset, list(subset[0]._fields), groupmap, **extra)
            format.write(output, subset)
    else:
        # @todo add defocus output to micrograph selection file
        micselect={}
        for v in subset:
            mic,par = spider_utility.relion_id(v.rlnImageName)
            if mic not in micselect: micselect[mic]=[]
            micselect[mic].append((par, 1))
        prefix=None
        if spider_utility.is_spider_filename(output):
            for mic,vals in micselect.iteritems():
                format.write(output, numpy.asarray(vals), spiderid=mic, header="id,select".split(','), format=format.spidersel)
            prefix='mic_'
        format.write(output, numpy.hstack((numpy.asarray(micselect.keys())[:, numpy.newaxis], numpy.ones(len(micselect.keys()))[:, numpy.newaxis])), header="id,select".split(','), format=format.spidersel, prefix=prefix)

def generate_relion_selection_file(files, img, output, param_file, select="", good="", test_all=False, **extra):
    ''' Generate a relion selection file for a list of stacks, defocus file and params file
    
    :Parameters:

    files : list
            List of stack files
    img : EMData
          Image used to query size information
    output : str
             Filename for output selection file
    param_file : str
                 Filename for input SPIDER Params file
    select : str
             Filename for input optional selection file (for good particles in each stack)
    extra : dict
            Unused key word arguments
    '''
    
    header = "rlnImageName,rlnMicrographName,rlnDefocusU,rlnVoltage,rlnSphericalAberration,rlnAmplitudeContrast,rlnGroupNumber".split(',')
    spider_params.read(param_file, extra)
    pixel_radius = int(extra['pixel_diameter']/2.0)
    if img.shape[0]%2 != 0: raise ValueError, "Relion requires even sized images"
    if img.shape[0] != img.shape[0]: raise ValueError, "Relion requires square images"
    if pixel_radius > 0:
        #mask = ndimage_utility.model_disk(pixel_radius, img.shape[0])*-1+1
        mask = eman2_utility.model_circle(pixel_radius, img.shape[0], img.shape[1])*-1+1
        avg = numpy.mean(img*mask)
        if numpy.allclose(0.0, avg):
            _logger.warn("Relion requires the background to be zero normalized, not %g"%avg)
    
    if test_all:
        mask = eman2_utility.model_circle(pixel_radius, img.shape[0], img.shape[1])*-1+1
        for img in ndimage_file.iter_images(files):
             avg = numpy.mean(img*mask)
             std = numpy.std(img*mask)
             if not numpy.allclose(0.0, avg): raise ValueError, "Image mean not correct: mean: %f, std: %f"%(avg, std)
             if not numpy.allclose(1.0, std): raise ValueError, "Image std not correct: mean: %f, std: %f"%(avg, std)
    
    defocus_dict = read_defocus(**extra)
    if select != "": 
        select = format.read(select, numeric=True)
        files = spider_utility.select_subset(files, select)
        old = defocus_dict
        defocus_dict = {}
        for s in select:
            if s.id in old: defocus_dict[s.id]=old[s.id]
    spider_params.read(param_file, extra)
    voltage, cs, ampcont=extra['voltage'], extra['cs'], extra['ampcont']
    idlen = len(str(ndimage_file.count_images(files)))
    
    tilt_pair = read_tilt_pair(**extra)
    if len(tilt_pair) > 0:
        format.write(output, generate_selection(files, header, tilt_pair[:, :2], defocus_dict, **extra), header=header, prefix="first_")
        format.write(output, generate_selection(files, header, tilt_pair[:, 2:4], defocus_dict, **extra), header=header, prefix="second_")
    else:
        label = []
        group = []
        for filename in files:
            mic = spider_utility.spider_id(filename)
            if good != "":
                if not os.path.exists(spider_utility.spider_filename(good, mic)): continue
                select_vals = format.read(good, spiderid=mic, numeric=True)
                if len(select_vals) > 0 and not hasattr(select_vals[0], 'id'):
                    raise ValueError, "Error with selection file (`--select`) missing `id` in header, return with `--select filename=id` or `--select filename=id,select` indicating which column has the id and select"
                if len(select_vals) > 0 and 'select' in select_vals[0]._fields:
                    select_vals = [s.id for s in select_vals if s.select > 0] if len(select_vals) > 0 and hasattr(select_vals[0], 'select') else [s.id for s in select_vals]
            else:
                select_vals = xrange(1, ndimage_file.count_images(filename)+1)
            if mic not in defocus_dict:
                _logger.warn("Micrograph not found in defocus file: %d -- skipping"%mic)
                continue
            if defocus_dict[mic].defocus < 1000: 
                #assert(False)
                _logger.warn("Micrograph %d defocus too small: %f -- skipping"%(mic, defocus_dict[mic].defocus))
                continue
            
            group.append((defocus_dict[mic].defocus, len(select_vals), len(label), mic))
            for pid in select_vals:
                #label.append( ["%s@%s"%(str(pid).zfill(idlen), filename), filename, defocus_dict[mic].defocus, voltage, cs, ampcont, mic] )
                label.append( ["%s@%s"%(str(pid).zfill(idlen), filename), filename, defocus_dict[mic].defocus, voltage, cs, ampcont, len(group)-1] )
        if len(group) == 0: raise ValueError, "No values to write out, try changing selection file"
        groupmap = regroup(group, **extra)
        update_parameters(label, header, groupmap, **extra)
        
        format.write(output, label, header=header)

#                    files, header, tilt_pair[:, :2], defocus_dict
def generate_selection(files, header, select, defocus_dict, voltage=0, cs=0, ampcont=0, id_len=0, **extra):
    '''
    '''
    
    if ampcont == 0 or voltage == 0 or cs == 0: raise ValueError, "Missing SPIDER params file: voltage: %f, cs: %f, ampcont: %f"%(voltage, cs, ampcont)
    idlen = len(str(select[:, 0].max()))
    label = []
    group = []
    i=0
    filename = files[0]
    while i < len(select):
        mic = int(select[i, 0])
        select_vals = select[mic==select[:, 0], 1]
        group.append((defocus_dict[mic].defocus, len(select_vals), len(label), mic))
        filename = spider_utility.spider_filename(filename, mic, id_len)
        for pid in select_vals:
            label.append( ["%s@%s"%(str(pid).zfill(idlen), filename), filename, defocus_dict[mic].defocus, voltage, cs, ampcont, len(group)-1] )
        i+=len(select_vals)
    if len(group) == 0: raise ValueError, "No values to write out, try changing selection file"
    groupmap = regroup(group, **extra)
    update_parameters(label, header, groupmap, **extra)
    return label
    
def read_tilt_pair(tilt_pair, **extra):
    '''
    '''
    
    if tilt_pair == "": return []
    #mic1          id        mic2         id2
    return numpy.asarray(format.read(tilt_pair, numeric=True)).astype(numpy.int)
    
def read_defocus(defocus_file, defocus_header, min_defocus, max_defocus, **extra):
    ''' Read a defocus file
    
    :Parameters:
    
    defocus_file : str
                   Filename for input defocus file
    defocus_header : str
                     Header for defocus file
    extra : dict
            Unused key word arguments
    '''
    
    if defocus_file == "": return {}
    defocus_dict = format.read(defocus_file, header=defocus_header, numeric=True)
    for i in xrange(len(defocus_dict)-1, 0, -1):
        if defocus_dict[i].defocus < min_defocus or defocus_dict[i].defocus > max_defocus:
            _logger.warn("Removing micrograph %d because defocus %f violates allowed range %f-%f "%(defocus_dict[i].id, defocus_dict[i].defocus, min_defocus, max_defocus))
            del defocus_dict[i]
    return format_utility.map_object_list(defocus_dict)
    
def build_group(data):
    ''' Build a grouping from a set of relion data
    
    :Parameters:
    
    data : list
           List of alignment parameters
    
    :Returns:
    
    group : list
            List of groups: defocus, size, offset
    '''
    
    mic_index = list(data[0]._fields).index('rlnMicrographName')
    def_index = list(data[0]._fields).index('rlnDefocusU')
    data = sorted(data, key=operator.itemgetter(mic_index))
    group = []
    defocus = data[0][def_index]
    last = spider_utility.spider_id(data[0][mic_index])
    total = 0
    selected = 0
    for d in data:
        id = spider_utility.spider_id(d[mic_index])
        if id != last:
            group.append((defocus, selected, total, last))
            defocus = data[0][def_index]
            total += selected
            selected = 0
            last = id
        selected += 1
    if selected > 0:
        group.append((defocus, selected, total, last))
    if len(group) == 1: raise ValueError, "--minimum-group may be too small for your selection file of size %d"%len(data)
    return group
    
def regroup(group, minimum_group, **extra):
    ''' Regroup micrographs by defocus
    
    :Parameters:
    
    group : list
            Micrograph grouping
    minimum_group : int
                    Minimum size of defocus group
    
    :Returns:
    
    group_map : dict
                Group mapping
    '''
    
    if minimum_group == 0: return {}
    group = numpy.asarray(group)
    assert(group.shape[1]==4)
    regroup = []
    offset = 0
    total = 0
    groupmap = {}
    try:
        idx = numpy.argsort(group[:, 0])
    except:
        _logger.error("group: %s"%str(group.shape))
        raise
    
    for i in idx:
        if total <= minimum_group:
            id = int(group[i, 3])
            groupmap[id]=offset
            regroup.append(id)
            total += group[i, 1]
        if total > minimum_group:
            offset += 1
            for id in regroup: groupmap[id]=offset
            total = 0
            regroup=[]
    _logger.info("Regrouping from %d to %d"%(len(groupmap), offset))
    return groupmap
    
def update_parameters(data, header, group_map=None, scale=1.0, stack_file="", **extra):
    ''' Update parameters in a relion selection file
    
    data : list
           List of alignment parameters
    header : list
             List of column names
    group_map : dict
                Group mapping
    scale : float
            Scale factor for translations
    stack_file : str
                 Filename for image stack
    extra : dict
            Unused key word arguments
    
    :Returns:
    
    data : list
           Updated data list
    '''
    
    if group_map is not None and len(group_map)==0: group_map=None
    group_col = header.index('rlnMicrographName')
    group_col2 = header.index('rlnGroupNumber')
    if scale != 1.0:
        try:x_col = header.index('rlnOriginX')
        except: x_col = -1
        try:y_col = header.index('rlnOriginY')
        except: y_col = -1
    else: x_col, y_col = -1, -1
    name_col = header.index('rlnImageName') if stack_file != "" else -1
    Tuple = data[0].__class__
    
    for i in xrange(len(data)):
        vals = data[i] if not isinstance(data[i], tuple) else list(data[i])
        if group_col >= 0 and group_map is not None:
            id = spider_utility.spider_id(vals[group_col])
            try:
                vals[group_col] = spider_utility.spider_filename(vals[group_col], group_map[id])
                vals[group_col2] = spider_utility.spider_id(group_map[id])
                
            except:
                _logger.error("keys: %s"%str(group_map.keys()))
                raise
        if x_col > -1: vals[x_col]*= scale
        if y_col > -1: vals[y_col]*= scale
        if name_col > -1: vals[name_col] = spider_utility.relion_filename(stack_file, vals[name_col])
        if hasattr(Tuple, '_make'): data[i] = Tuple._make(vals)
        else: data[i] = tuple(vals)
    
    return data

def generate_settings(**extra):
    '''
    '''
    
    if 'reference' not in extra: extra['reference']=""
    if 'diameter' not in extra: extra['diameter']=extra['pixel_diameter']*extra['apix']
    return """
    Input images: == %(output)s
    Reference map: == %(reference)s
    Particle mask diameter (A): == %(diameter)d
    Pixel size (A): == %(apix)d
    Additional arguments: == --max_memory 32
    """.format(**extra)

def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "Relion Selection", "Options to control creation of a relion selection file",  id=__name__)
    group.add_option("-s", select="",                       help="SPIDER micrograph or class selection file - if select file does not have proper header, then use `--select filename=id` or `--select filename=id,select`", gui=dict(filetype="open"))
    group.add_option("-g", good="",                         help="SPIDER particle selection file (used when creating a new relion selection file) - if select file does not have proper header, then use `--select filename=id` or `--select filename=id,select`", gui=dict(filetype="open"))
    group.add_option("-p", param_file="",                   help="SPIDER parameters file (Only required when the input is a stack)", gui=dict(filetype="open"))
    group.add_option("-d", defocus_file="",                 help="SPIDER defocus file (Only required when the input is a stack)", gui=dict(filetype="open"))
    group.add_option("-l", defocus_header="id:0,defocus:1", help="Column location for micrograph id and defocus value (Only required when the input is a stack)")
    group.add_option("-m", minimum_group=20,                help="Minimum number of particles per defocus group", gui=dict(minimum=0, singleStep=1))
    group.add_option("",   bin_factor=1.0,                  help="Number of times to decimate parameters") # to do set automatically
    group.add_option("",   stack_file="",                   help="Used to rename the stack portion of the image name (rlnImageName); ignored when creating a relion file")
    group.add_option("",   scale=1.0,                       help="Used to scale the translations in a relion file")
    group.add_option("",   column="rlnClassNumber",         help="Column name in relion file for selection, e.g. rlnClassNumber to select classes")
    group.add_option("",   test_all=False,                  help="Test the normalization of all the images")
    group.add_option("",   tilt_pair="",                    help="Selection file that defines pairs of particles (e.g. tilt pairs micrograph1, id1, micrograph2, id2) - outputs a tilted/untilted star files")
    group.add_option("",   min_defocus=10000,                help="Minimum allowed defocus")
    group.add_option("",   max_defocus=70000,                help="Maximum allowed defocus")
    group.add_option("",   random_subset=0,                 help="Split a relion selection file into `n` number of random subsets (0 disables)")
    group.add_option("",   refine_spi=False,                help="Convert a relion selection file to a set of SPIDER refinement files")
    group.add_option("",   views=0,                         help="Write out view distribution for given healpix order and relion star file")
    group.add_option("",   frame="",                        help="Frame stack used to build new relion star file")
    group.add_option("",   frame_limit=0,                   help="Limit number of frames to use (0 means no limit)")
    group.add_option("",   view_resolution=0,               help="Select a subset to ensure roughly even view distribution")
    
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input stacks or selection file", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the relion selection file", gui=dict(filetype="save"), required_file=True)
        parser.change_default(log_level=3)

def check_options(options, main_option=False):
    #Check if the option values are valid
    
    from ..core.app.settings import OptionValueError
    
    if not format.is_readable(options.input_files[0]) and ndimage_file.is_readable(options.input_files[0]):
        if options.defocus_file == "": raise OptionValueError, "No defocus file specified"
        if options.param_file == "": raise OptionValueError, "No parameter file specified"
    #elif main_option:
    #    if len(options.input_files) != 1: raise OptionValueError, "Only a single input file is supported"

def main():
    #Main entry point for this script
    
    run_hybrid_program(__name__,
        description = '''Generate a relion selection file from a set of stacks and a defocus file
        
                         http://
                         
                         Example: Generate a relion selection file from a set of stacks, defocus file and params file
                         
                         $ ara-selrelion win*.spi -d def_avg.spi -p params.spi -o relion_select.star
                         
                         Example: Select projects in a relion selection file based on the class column using a class selection file
                         
                         $ ara-selrelion relion_select.star -s good_classes.spi relion_select_good.star
                      ''',
        supports_MPI = False,
        use_version = False,
    )
def dependents(): return []
if __name__ == "__main__": main()


