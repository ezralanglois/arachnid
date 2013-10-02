''' Fast algorithm to find reference view

Download to edit and run: :download:`fast_projection_finding.py <../../arachnid/snippets/fast_projection_finding.py>`

To run:

.. sourcecode:: sh
    
    $ python fast_projection_finding.py

.. note::
    
    You must have Arachnid and Matplotlib installed to run this script

.. literalinclude:: ../../arachnid/snippets/fast_projection_finding.py
   :language: python
   :lines: 22-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')
from arachnid.core.spider import spider
from arachnid.core.image import ndimage_file, eman2_utility, rotate #, ndimage_utility
from arachnid.core.metadata import spider_params, format #, spider_utility
#from arachnid.core.orient import orient_utility
#import scipy.spatial.distance
#import scipy.optimize
from arachnid.pyspider import autorefine
import numpy #, functools

if __name__ == '__main__':

    # Parameters
    image_file = sys.argv[1]
    image_sel = sys.argv[2]
    reference = sys.argv[3]
    align_file = sys.argv[4]
    param_file = sys.argv[5]
    outputfile = sys.argv[6]
    align_last = sys.argv[7]
    
    align_last, align_file = align_file, align_last
    
    spider_path = "/guam.raid.cluster.software/spider.20.12/bin/spider_linux_mp_intel64"
    thread_count=2
    enable_results=False
    mode=0
    nangs=180
    angle_cache='angles'
    reference_stack='ref_proj'
    trans_ref_stack='trans_proj.spi'
    trans_exp_stack='trans_exp.spi'
    trans_exp_align='align_exp.spi'
    dala='dala.spi'
    tmp_img='tmp_proj'
    extra = spider_params.read(param_file)
    #trans_range = autorefine.ensure_translation_range(window, ring_last, 500)
    resolution = 4.714083
    print "apix: ", extra['apix']
    print "window: ", extra['window']
    print "resolution: ", resolution
    bin_factor = autorefine.decimation_level(resolution, 30, min_bin_factor=8, **extra)
    extra.update(spider_params.update_params(bin_factor, **extra))
    theta_delta = 2.04532082383 #autorefine.theta_delta_est(resolution, trans_range=1, theta_delta=15, ring_last=int(extra['pixel_diameter']/2.0), **extra)
    
    print "window: ", extra['window']
    print 'theta_delta: ', theta_delta
    print 'bin_factor: ', bin_factor
    
    spi = spider.open_session([image_file], spider_path, thread_count=thread_count, enable_results=enable_results)
    align = format.read(align_file, numeric=True)
    lalign = format.read(align_last, numeric=True)
    #sel = format.read(image_sel, numeric=True, header=['mic', 'id'])
    sel = numpy.loadtxt(image_sel, delimiter=",")
    alignmap = {}
    for al in zip(align,lalign):
        al1=al[0]
        alignmap.setdefault(al1.micrograph, {})
        alignmap[al1.micrograph][al1.stack_id] = al
        
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
        #image_file=spi.rt_sq(image_file, align_last, outputfile=dala)
        
        # 2. Transform reference stack
        
    #ndimage_file.process_images(spi.replace_ext(reference_stack), trans_ref_stack, image_transform)
    
    refs=None
    count = ndimage_file.count_images(spi.replace_ext(reference_stack))
    for i, ref in enumerate(ndimage_file.iter_images(spi.replace_ext(reference_stack))): 
        ref -= ref.mean()
        if refs is None:
            refs = numpy.zeros((count, ref.ravel().shape[0]), dtype=ref.dtype)
            print "Reference shape: ", ref.shape, count
        refs[i, :] = ref.ravel()
        
    count = ndimage_file.count_images(spi.replace_ext(image_file))
    dists = numpy.zeros(len(refs))
    for i in xrange(count):
        val = sel[i]
        #print val
        al,lal = alignmap[val[0]][val[1]]
        #print al
        ipimg=spi.ip((image_file, i+1), (extra['window'], extra['window']))
        #rimg=spider.rt_sq_single(spi,ipimg, (al.psi, al.tx/extra['apix'], al.ty/extra['apix']), outputfile=tmp_img)
        rimg=spider.rt_sq_single(spi,ipimg, (lal.psi, lal.tx/extra['apix'], lal.ty/extra['apix']), outputfile=tmp_img)
        
        img = ndimage_file.read_image(spi.replace_ext(tmp_img))
        if i == 0: print "Image shape: ", img.shape, extra['window']
        if al.theta > 179.99: img = eman2_utility.mirror(img)
        best=(1e20, None)
            
        img -= img.mean()
        for i in xrange(len(refs)):
            ref = refs[i].reshape((img.shape)).copy()
            psi = rotate.optimal_inplane(numpy.asarray((0, al.theta, al.phi)), numpy.asarray((0, lal.theta, lal.phi)))
            rimg = rotate.rotate_image(img, psi)
            
            #if i == 0:
            #    ndimage_file.write_image("sinogram01.spi", refs[i].reshape((img.shape)))
            #fref = numpy.abs(scipy.fftpack.fft(ref.ravel()))
            #fref = numpy.abs(scipy.fftpack.fft(numpy.asfortranarray(ref).ravel()))
            
            
            #cc_map = ndimage_utility.cross_correlate(img, ref)
            #dist=-numpy.max(cc_map, axis=1).mean()
            #x, y = numpy.unravel_index(numpy.argmax(cc_map), cc_map.shape)[::-1]
            
            dist = numpy.sum(numpy.square((rimg-ref)))
            #dist = numpy.sum(numpy.square((img-ref)))
            dists[i]=dist
            
            #dist=-scipy.spatial.distance.correlation(img.ravel(), ref.ravel())
            
            if (i+1) == al.ref_num:
                actual = dist
            if dist < best[0]: best = (dist, i+1)
        idx=numpy.argsort(dists).squeeze()
        rank=numpy.nonzero(idx == (al.ref_num-1))[0]
        print "best: ", best, '--', "act: ", (actual, al.ref_num), 'rank:', rank, al.sx/extra['apix'], al.sy/extra['apix'], al.spsi, lal.cc_rot
        
