''' Fast algorithm to find reference view

Download to edit and run: :download:`fast_alignment_registration.py <../../arachnid/snippets/fast_alignment_registration.py>`

To run:

.. sourcecode:: sh
    
    $ python fast_alignment_registration.py

.. note::
    
    You must have Arachnid and Matplotlib installed to run this script

.. literalinclude:: ../../arachnid/snippets/fast_alignment_registration.py
   :language: python
   :lines: 22-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')
from arachnid.core.spider import spider
from arachnid.core.image import ndimage_file, ndimage_utility, eman2_utility
from arachnid.core.metadata import spider_params, format #, spider_utility
from arachnid.core.orient import orient_utility
#import scipy.spatial.distance
#import scipy.optimize
from arachnid.pyspider import autorefine
import numpy

if __name__ == '__main__':

    # Parameters
    image_file = sys.argv[1]
    image_sel = sys.argv[2]
    reference = sys.argv[3]
    align_file = sys.argv[4]
    param_file = sys.argv[5]
    outputfile = sys.argv[6]
    resolution=30
    spider_path = "/guam.raid.cluster.software/spider.20.12/bin/spider_linux_mp_intel64"
    thread_count=2
    enable_results=False
    angle_cache='angles'
    reference_stack='ref_proj'
    trans_ref_stack='trans_proj.spi'
    trans_exp_stack='trans_exp.spi'
    trans_exp_align='align_exp.spi'
    tmp_img='tmp_proj'
    extra = spider_params.read(param_file)
    #trans_range = autorefine.ensure_translation_range(window, ring_last, 500)
    resolution = 30 #18.86910293062289057
    print "apix: ", extra['apix']
    print "window: ", extra['window']
    print "resolution: ", resolution
    bin_factor = autorefine.decimation_level(resolution, 30, min_bin_factor=8, **extra)
    extra.update(spider_params.update_params(bin_factor, **extra))
    theta_delta = autorefine.theta_delta_est(resolution, trans_range=1, theta_delta=15, ring_last=int(extra['pixel_diameter']/2.0), **extra)
    
    print 'theta_delta: ', theta_delta
    print 'bin_factor: ', bin_factor
    
    
    # Open SPIDER session
    spi = spider.open_session([image_file], spider_path, thread_count=thread_count, enable_results=enable_results)
    align = format.read(align_file, numeric=True)
    #sel = format.read(image_sel, numeric=True, header=['mic', 'id'])
    sel = numpy.loadtxt(image_sel, delimiter=",")
    alignmap = {}
    for al in align:
        alignmap.setdefault(al.micrograph, {})
        alignmap[al.micrograph][al.stack_id] = al
    

    
    if 1 == 0:
        # 1. Generate reference stack
        print "copy reference"
        sys.stdout.flush()
        reference = spider.copy_safe(spi, reference, **extra)
        print "generate angles"
        sys.stdout.flush()
        spi.de(angle_cache)
        angle_doc, angle_num = spi.vo_ea(theta_delta, outputfile=angle_cache)
        print "generate projections"
        sys.stdout.flush()
        spi.de(reference_stack)
        spi.pj_3q(reference, angle_doc, (1, angle_num), outputfile=reference_stack, **extra)
        
        # 2. Transform reference stack
        
    
    refs=None
    count = ndimage_file.count_images(spi.replace_ext(reference_stack))
    min_val = 1e20
    for i, ref in enumerate(ndimage_file.iter_images(spi.replace_ext(reference_stack))): 
        if refs is None:
            refs = numpy.zeros((count, ref.shape[0], ref.shape[1]), dtype=ref.dtype)
        refs[i, :] = ref
    
    values = []
    scale=1.0
    for i in xrange(5):
        # 3. Load and transform image
        val = sel[i]
        #print val
        al = alignmap[val[0]][val[1]]
        #print al
        values.append(al)
        spi.ip((image_file, i+1), (extra['window'], extra['window']), outputfile=tmp_img)
        img = ndimage_file.read_image(spi.replace_ext(tmp_img))
        best = (-1e20, None)
        for j, template in enumerate(refs):
            cc_map = ndimage_utility.cross_correlate(img, template)
            x, y = numpy.unravel_index(numpy.argmax(cc_map), cc_map.shape)[::-1]
            x -= cc_map.shape[0]/2
            y -= cc_map.shape[1]/2
            #pimg, log_base = ndimage_utility.logpolar(img)
            pimg = ndimage_utility.polar(img)
            cc_map = ndimage_utility.cross_correlate(pimg, ndimage_utility.logpolar(template)[0])
            rx, ry = numpy.unravel_index(numpy.argmax(cc_map), cc_map.shape)[::-1]
            angle = 180.0 * ry / cc_map.shape[0]
            #scale = log_base ** rx
            angle, x, y = orient_utility.align_param_3D_to_2D_simple(angle, x, y)
            if cc_map[ry,rx] > best[0]: best = (cc_map[ry,rx], angle, x, y, scale, j+1, 0)
            img = eman2_utility.mirror(img)
            cc_map = ndimage_utility.cross_correlate(img, template)
            x, y = numpy.unravel_index(numpy.argmax(cc_map), cc_map.shape)[::-1]
            x -= cc_map.shape[0]/2
            y -= cc_map.shape[1]/2
            #pimg, log_base = ndimage_utility.logpolar(img)
            pimg = ndimage_utility.polar(img)
            cc_map = ndimage_utility.cross_correlate(pimg, ndimage_utility.logpolar(template)[0])
            rx, ry = numpy.unravel_index(numpy.argmax(cc_map), cc_map.shape)[::-1]
            angle = 180.0 * ry / cc_map.shape[0]
            #scale = log_base ** rx
            angle, x, y = orient_utility.align_param_3D_to_2D_simple(angle, x, y)
            if cc_map[ry,rx] > best[0]: best = (cc_map[ry,rx], angle, x, y, scale, j+1, 1)
        print best[1], '==', al.psi, "|", best[2], '==', al.tx/extra['apix'], "|", best[3], '==', al.ty/extra['apix'], '|', best[4], "|", best[0], '==', al.cc_rot, "|", best[5], '==', al.ref_num, "|", best[6], '==', al.theta
            
            
    
    