'''

http://deeplearning.stanford.edu/wiki/index.php/Implementing_PCA/Whitening

.. Created on Sep 21, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import matplotlib
matplotlib.use("Agg")

from ..core.app.program import run_hybrid_program
from ..core.image import ndimage_file, eman2_utility, analysis, ndimage_utility, manifold #, reconstruct
from ..core.metadata import spider_utility, format, format_utility, spider_params
from ..core.parallel import mpi_utility
from arachnid.core.util import plotting #, fitting
import logging, numpy, os, scipy,itertools

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def process(input_vals, input_files, output, id_len=0, neig=1, nstd=1.5, **extra):
    '''Concatenate files and write to a single output file
        
    :Parameters:
    
    input_vals : list 
                 Tuple(view id, image labels and alignment parameters)
    input_files : list
                  List of input file stacks
    output : str
             Filename for output file
    id_len : int
             Max length of SPIDER ID
    extra : dict
            Unused key word arguments
                
    :Returns:
    
    filename : str
               Current filename
    '''
    
    image_size=0.4
    radius=40
    image_count=400
    label, align = rotational_sample(*input_vals[1:], **extra)
    mask = create_mask(input_files, **extra)
    data = ndimage_file.read_image_mat(input_files, label, image_transform, shared=False, mask=mask, cache_file=None, align=align, **extra)
    tst = data-data.mean(0)
    
    if 1 == 1:
        try:
            U, d, V = scipy.linalg.svd(tst, False)
        except:
            _logger.warn("SVD failed to converge for view %d"%input_vals[0])
            rsel = numpy.ones(len(input_vals[1]), dtype=numpy.bool)
            return input_vals, rsel
        feat = d*numpy.dot(V, tst.T).T
        feat = feat[:, :max(5, neig)]
    else:
        #from sklearn.neighbors import NearestNeighbors
        
        index = manifold.largest_connected(manifold.knn_reduce(manifold.knn(data, extra['nsamples']*2), extra['nsamples']*2, False))[1]
        
        from sklearn.manifold import locally_linear
        feat = locally_linear.locally_linear_embedding(data[index], extra['nsamples']*2+1, max(5, neig))[0]
    if 1 == 1:
        sel=None
        for i in xrange(neig):
            if 1 == 1:
                sel1 = analysis.robust_rejection(numpy.abs(feat[:, i]), nstd)
            else:
                sel1 = analysis.robust_rejection(feat[:, i], nstd)
                sel2 = analysis.robust_rejection(-feat[:, i], nstd)
                if numpy.sum(sel1) > numpy.sum(sel2): sel1=sel2
            sel = numpy.logical_and(sel1, sel) if sel is not None else sel1
        format.write_dataset(output, numpy.hstack((sel[:, numpy.newaxis], align[:, 0][:, numpy.newaxis], label[:, 1][:, numpy.newaxis], feat)), input_vals[0], label, header='select,rot,group', prefix='pca_')
    else:
        neigh = manifold.knn(feat[:, :2], extra['nsamples']/2)
        index = manifold.largest_connected(manifold.knn_reduce(neigh, 3, False))[1]
        sel = numpy.zeros(len(data), dtype=numpy.bool)
        sel[index]=1
        index = manifold.largest_connected(manifold.knn_reduce(neigh, 2, False))[1]
        sel2 = numpy.zeros(len(data), dtype=numpy.bool)
        sel2[index]=1
        format.write_dataset(output, numpy.hstack((sel[:, numpy.newaxis],sel2[:, numpy.newaxis], align[:, 0][:, numpy.newaxis], label[:, 1][:, numpy.newaxis], feat)), input_vals[0], label, header='select,select2,rot,group', prefix='pca_')
    image_size, radius, sel, image_count;
    #plot_embedded(feat[:, 0], feat[:, 1], "pca_%d"%input_vals[0], label, input_files[0], output, image_size, radius, sel, image_count)
    rsel = numpy.ones(input_vals[1].shape[0], dtype=numpy.bool)
    nsamples=extra['nsamples']
    for i in xrange(rsel.shape[0]):
        rsel[i] = numpy.alltrue(sel[i*nsamples:(i+1)*nsamples])
    return input_vals, rsel

def initialize(files, param):
    # Initialize global parameters for the script
    
    if mpi_utility.is_root(**param):
        if param['resolution'] > 0.0: _logger.info("Filter and decimate to resolution: %f"%param['resolution'])
        if param['use_rtsq']: _logger.info("Rotate and translate data stack")
    group = None
    if mpi_utility.is_root(**param): 
        param['sel_by_mic']={}
        group = group_by_reference(*read_alignment(files, **param))
        if param['single_view'] > 0:
            tmp=group
            group = [tmp[param['single_view']-1]]
    group = mpi_utility.broadcast(group, **param)
    return group

def reduce_all(val, sel_by_mic, output, id_len=0, **extra):
    # Process each input file in the main thread (for multi-threaded code)
    
    input, sel = val
    label = input[1]
    for i in numpy.argwhere(sel):
        sel_by_mic.setdefault(int(label[i, 0]), []).append(int(label[i, 1]+1))    
    
    tot=numpy.sum(sel)
    return "%d - Selected: %d -- Removed %d"%(input[0], tot, label.shape[0]-tot)

def finalize(files, output, sel_by_mic, **extra):
    # Finalize global parameters for the script
    
    for id, sel in sel_by_mic.iteritems():
        sel = numpy.asarray(sel)
        format.write(output, numpy.vstack((sel, numpy.ones(sel.shape[0]))).T, prefix="sel_", spiderid=id, header=['id', 'select'], default_format=format.spidersel)
    _logger.info("Completed")

def rotational_sample(label, align, nsamples, angle_range, **extra):
    '''
    '''
    
    if nsamples < 2:
        return label, align
    label2 = numpy.zeros((label.shape[0]*nsamples, label.shape[1]))
    align2 = numpy.zeros((align.shape[0]*nsamples, align.shape[1]))
    for i in xrange(len(label)):
        label2[i*nsamples:(i+1)*nsamples] = label[i]
        align2[i*nsamples:(i+1)*nsamples] = align[i]
        align2[i*nsamples:(i+1)*nsamples, 0]=scipy.linspace(-angle_range/2.0, angle_range/2.0, nsamples,True)
    return label2, align2

def create_mask(files, pixel_diameter, resolution, apix, **extra):
    '''
    '''
    
    img = ndimage_file.read_image(files[0])
    mask = eman2_utility.model_circle(int(pixel_diameter/2.0), img.shape[0], img.shape[1])
    bin_factor = max(1, min(8, resolution / (apix*2))) if resolution > (2*apix) else 1
    _logger.info("Decimation factor %f for resolution %f and pixel size %f"%(bin_factor, resolution, apix))
    if bin_factor > 1: mask = eman2_utility.decimate(mask, bin_factor)
    return mask

def image_transform(img, i, mask, resolution, apix, var_one=True, align=None, **extra):
    '''
    '''
    
    if align[i, 1] > 179.999: img = eman2_utility.mirror(img)
    if align[i, 0] != 0: img = eman2_utility.rot_shift2D(img, align[i, 0], 0, 0, 0)
    ndimage_utility.vst(img, img)
    bin_factor = max(1, min(8, resolution / (apix*2))) if resolution > (2*apix) else 1
    if bin_factor > 1: img = eman2_utility.decimate(img, bin_factor)
    ndimage_utility.normalize_standard(img, mask, var_one, img)
    if 1 == 0:
        img, freq = ndimage_utility.bispectrum(img, 0.5, 'gaussian', 1.0)
        freq;
        img = numpy.log10(numpy.sqrt(numpy.abs(img.real+1)))
    
    #img = ndimage_utility.compress_image(img, mask)
    return img
    
def plot_embedded(x, y, title, label, filename, output, image_size, radius, ref=None, dpi=72, select=None):
    '''
    '''
    
    xs = x[select] if select is not None else x
    ys = y[select] if select is not None else y
    fig, ax = plotting.plot_embedding(x, y, select, ref, dpi=dpi)
    vals = plotting.nonoverlapping_subset(ax, xs, ys, radius, 100)
    if len(vals) > 0:
        try:
            index = select[vals].ravel() if select is not None else vals
        except:
            _logger.error("%d-%d | %d"%(numpy.min(vals), numpy.max(vals), len(select)))
        else:
            if index.shape[0] > 1:
                iter_single_images = itertools.imap(ndimage_utility.normalize_min_max, ndimage_file.iter_images(filename, label[index].squeeze()))
                plotting.plot_images(fig, iter_single_images, x[index], y[index], image_size, radius)
    fig.savefig(format_utility.new_filename(output, title, ext="png"), dpi=dpi)

def group_by_reference(label, align, ref):
    '''
    '''
    
    group=[]
    refs = numpy.unique(ref)
    for r in refs:
        sel = r == ref
        group.append((r, label[sel], align[sel]))
    return group

def read_alignment(files, alignment="", **extra):
    ''' Read alignment parameters
    
    :Parameters:
    
    files : list
            List of input files containing particle stacks
    alignment : str
                Input filename containing alignment parameters
    
    :Returns:
    
    group : list
            List of tuples, one for each view containing: view id, image labels and alignment parameters
    '''
    
    if len(files) == 1:
        spiderid = files[0] if not os.path.exists(alignment) else None
        align = format.read_alignment(alignment, spiderid=spiderid)
        align, header = format_utility.tuple2numpy(align)
        refidx = header.index('ref_num')
        label = numpy.zeros((len(align), 2), dtype=numpy.int)
        label[:, 0] = spider_utility.spider_id(files[0])
        label[:, 1] = align[:, 4].astype(numpy.int)-1
        if numpy.max(label[:, 1]) >= ndimage_file.count_images(files[0]):
            label[:, 1] = numpy.arange(0, len(align))
    else:
        align = None
        refidx = None
        if os.path.exists(alignment):
            _logger.debug("Alignment exists")
            align = format.read_alignment(alignment)
            align, header = format_utility.tuple2numpy(align)
            refidx = header.index('ref_num')
            if len(align)>0 and 'stack_id' in set(header):
                align = numpy.asarray(align)
                label = align[:, 15:17].astype(numpy.int)
                label[:, 1]-=1
            else:
                align=None
        if align is None:
            alignvals = []
            total = 0 
            for f in files:
                aligncur = format.read_alignment(alignment, spiderid=f)
                aligncur, header = format_utility.tuple2numpy(aligncur)
                if refidx is None: refidx = header.index('ref_num')
                alignvals.append((f, aligncur))
                total += len(aligncur)
            label = numpy.zeros((total, 2), dtype=numpy.int)
            align = numpy.zeros((total, aligncur.shape[1]))
            total = 0
            for f, cur in alignvals:
                end = total+cur.shape[0]
                align[total:end, :] = cur
                label[total:end, 0] = spider_utility.spider_id(f)
                label[total:end, 1] = align[total:end, 4].astype(numpy.int)
                if numpy.max(label[total:end, 1]) > ndimage_file.count_images(f):
                    label[:, 1] = numpy.arange(0, len(align[total:end]))
                total = end
            align = numpy.asarray(alignvals)
    ref = align[:, refidx].astype(numpy.int)
    #refs = numpy.unique(ref)
    return label, align, ref


def setup_options(parser, pgroup=None, main_option=False):
    # Collection of options necessary to use functions in this script
    
    from ..core.app.settings import OptionGroup
    group = OptionGroup(parser, "AutoPick", "Options to control reference-free particle selection",  id=__name__)
    group.add_option("", nsamples=10,                 help="Number of rotational samples")
    group.add_option("", angle_range=3.0,              help="Angular search range")
    group.add_option("", resolution=15.0,             help="Filter to given resolution - requires apix to be set")
    #group.add_option("", resolution_hi=0.0,           help="High-pass filter to given resolution - requires apix to be set")
    group.add_option("", use_rtsq=False,              help="Use alignment parameters to rotate projections in 2D")
    group.add_option("", neig=1,                    help="Number of eigen vectors to use")
    group.add_option("", nstd=1.5,                    help="Number of deviations from the median")
    group.add_option("", single_view=0,                help="Test the algorithm on a specific view")
    
    pgroup.add_option_group(group)
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of filenames for the input micrographs", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Output filename for the coordinate file with correct number of digits (e.g. sndc_0000.spi)", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-a", alignment="",   help="Input file containing alignment parameters", required_file=True, gui=dict(filetype="open"))
#def check_options(options, main_option=False):
    #Check if the option values are valid
#    from ..core.app.settings import OptionValueError
    

def main():
    #Main entry point for this script
    run_hybrid_program(__name__,
        description = '''Clean a particle selection of any remaning bad windows
                        
                        http://
                        
                        Example:
                         
                        $ ara-autoclean input-stack.spi -o coords.dat -p params.spi
                      ''',
        use_version = True,
        supports_OMP=True,
    )

def dependents(): return [spider_params]
if __name__ == "__main__": main()

