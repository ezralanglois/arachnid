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
import scipy.spatial.distance
import scipy.optimize
from arachnid.pyspider import autorefine
import numpy, functools

def image_transform_transinv(img, bispec_mode=0):
    
    i, img = img
    if 1 == 1:
        img -= img.mean()
        img = ndimage_utility.fftamp(img)
        img *= eman2_utility.model_circle(3, img.shape[0], img.shape[1])*-1+1
        img *= eman2_utility.model_circle(15, img.shape[0], img.shape[1])
        #img = numpy.log(img+1)
        #img -= ndimage_utility.rotavg(img)
        return img
    if 1 == 0:
        img = ndimage_utility.fourier_mellin(img)
        return img
    if 1 == 0:
        tmp = eman2_utility.numpy2em(img)
        tmp = tmp.bispecRotTransInvDirect(0)
        img = eman2_utility.em2numpy(tmp).copy()
        return img
    
    if 1 == 1:
        #img -= img.mean()
        #img /= img.std()
        img, freq = ndimage_utility.bispectrum(img, img.shape[0]-1, 'gaussian')
        #eturn numpy.log(numpy.abs(img.real)+1)
        img.real = 1
        img = numpy.power(img, 2.0)
        #img = numpy.vstack((, ).T
        #img = numpy.arctan2(ndimage_utility.mean_azimuthal(img.imag)[1:], ndimage_utility.mean_azimuthal(img.real)[1:])
        #cut = 1/15
        #sel = freq < cut
        #img = img[sel, sel].copy()
        #if i ==0:
        #    print len(freq), numpy.sum(freq>cut)
        #print numpy.min(numpy.rad2deg(numpy.angle(img.imag))), numpy.max(numpy.rad2deg(numpy.angle(img.imag)))
        if bispec_mode == 1:
            img = numpy.arctan(img.imag/img.real)
        elif bispec_mode == 2:
            img = numpy.vstack((img.real, img.imag)).T
        elif bispec_mode == 3:
            img = numpy.vstack((numpy.log10(numpy.abs(img.real)+1), numpy.angle(img.imag))).T
    else:
        img = ndimage_utility.fftamp(img)
    return img

def image_transform_rotinv(img, radon):
    ''''none' does not compute a window
            'uniform' computes the uniform hexagonal window
            'sasaki' computes the sasaki window
            'priestley' computes the priestley window
            'parzen' computes the parzen window
            'hamming' computes the hamming window
            'gaussian' computes the gaussian distribution window
            'daniell' computes the daniell window
    '''
    
    i, img = img
    #img = ndimage_utility.logpolar(img)[0]
    #img = ndimage_utility.cross_correlate(img, img)*(eman2_utility.model_circle(10, img.shape[0], img.shape[1])+1*-1)
    img, freq = ndimage_utility.bispectrum(img, img.shape[0]-1, 'uniform')
    img = numpy.arctan(img.imag/img.real)
    
    #img = ndimage_utility.sinogram(img, radon)
    return img

def image_transform_none(img):
    
    i, img = img
    return img

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
    mode=0
    nangs=180
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
        
    #ndimage_file.process_images(spi.replace_ext(reference_stack), trans_ref_stack, image_transform)
    
    refs=None
    count = ndimage_file.count_images(spi.replace_ext(reference_stack))
    min_val = 1e20
    image_transform = None
    for i, ref in enumerate(ndimage_file.iter_images(spi.replace_ext(reference_stack))): 
        ref -= ref.mean()
        if image_transform is None:
            if mode == 0:
                #param["radon_transform"] = transform.radon_transform(img.get_xsize(), img.get_ysize(), param['radon_angle']).tocsr()
                image_transform = functools.partial(image_transform_rotinv, radon=None) #ndimage_utility.radon_transform(ref.shape[0], ref.shape[1], nangs))
            elif mode == 1:
                image_transform = image_transform_transinv
            else:
                image_transform = image_transform_none
        ref = image_transform((0,ref))
        if refs is None:
            refs = numpy.zeros((count, ref.ravel().shape[0]), dtype=ref.dtype)
        refs[i, :] = ref.ravel()
    

    
    if 1 == 0:    
        exp = numpy.zeros((5, ref.ravel().shape[0]))
        values = []
        for i in xrange(5):
            val = sel[i]
            al = alignmap[val[0]][val[1]]
            values.append(al)
            iimg=spi.ip((image_file, i+1), (extra['window'], extra['window']))
            if mode == 0:
                tx, ty = orient_utility.align_param_2D_to_3D_simple(al.psi, al.tx/extra['apix'], al.ty/extra['apix'])[1:]
                spi.rt_sq_single(iimg, (0, tx, ty), outputfile=tmp_img)
            elif mode == 1:
                #spi.rt_sq_single(iimg, (al.psi, 0, 0), outputfile=tmp_img)
                spi.rt_sq_single(iimg, (al.psi, al.tx/extra['apix'], al.ty/extra['apix']), outputfile=tmp_img)
            else:
                spi.rt_sq_single(iimg, (al.psi, al.tx/extra['apix'], al.ty/extra['apix']), outputfile=tmp_img)
            img = ndimage_file.read_image(spi.replace_ext(tmp_img))
            if al.theta > 179.99: img = eman2_utility.mirror(img)
            img = image_transform((0,img))
            exp[i, :]=img.ravel()
    
        x0 = numpy.ones(ref.ravel().shape[0])
        def error_func(param, exp, ref, values):
            '''
            '''
            
            mexp = exp * param.reshape((1, len(param)))
            num = 0
            den = 0
            for i in xrange(exp.shape[0]):
                num += numpy.sum(numpy.square(mexp[i]-ref[values[i].ref_num-1]))
                for j in xrange(ref.shape[0]):
                    den += numpy.sum(numpy.square(mexp[i]-ref[j]))
            num /= mexp.shape[0]
            den /= mexp.shape[0]*ref.shape[0]
            return num/den
        
        ret = scipy.optimize.anneal(error_func, x0, args=(exp, ref, values)) 
        print ret
    else:
        values = []
        dists = numpy.zeros(len(refs))
        for i in xrange(5):
            # 3. Load and transform image
            val = sel[i]
            #print val
            al = alignmap[val[0]][val[1]]
            #print al
            values.append(al)
            iimg=spi.ip((image_file, i+1), (extra['window'], extra['window']))
            if mode == 0 and 1 == 0:
                tx, ty = orient_utility.align_param_2D_to_3D_simple(al.psi, al.tx/extra['apix'], al.ty/extra['apix'])[1:]
                spi.sh_f(iimg, (tx, ty), outputfile=tmp_img)
                #spi.rt_sq_single(iimg, (0, tx, ty), outputfile=tmp_img)
            elif mode == 1 or 1 == 1:
                #spi.rt_sq_single(iimg, (al.psi, 0, 0), outputfile=tmp_img)
                spi.rt_sq_single(iimg, (al.psi, al.tx/extra['apix'], al.ty/extra['apix']), outputfile=tmp_img)
            else:
                spi.rt_sq_single(iimg, (al.psi, al.tx/extra['apix'], al.ty/extra['apix']), outputfile=tmp_img)
            img = ndimage_file.read_image(spi.replace_ext(tmp_img))
            if al.theta > 179.99: img = eman2_utility.mirror(img)
            #ndimage_file.write_image(trans_exp_stack, img, i)
            img -= img.mean()
            img = image_transform((0,img))
            
            #fimg = numpy.abs(scipy.fftpack.fft(img.ravel()))
            #fimg = numpy.abs(scipy.fftpack.fft(numpy.asfortranarray(img).ravel()))
            
            best=(1e20, None)
            
            for i in xrange(len(refs)):
                ref = refs[i].reshape((img.shape)).copy()
                #if i == 0:
                #    ndimage_file.write_image("sinogram01.spi", refs[i].reshape((img.shape)))
                #fref = numpy.abs(scipy.fftpack.fft(ref.ravel()))
                #fref = numpy.abs(scipy.fftpack.fft(numpy.asfortranarray(ref).ravel()))
                
                
                #cc_map = ndimage_utility.cross_correlate(img, ref)
                #dist=-numpy.max(cc_map, axis=1).mean()
                #x, y = numpy.unravel_index(numpy.argmax(cc_map), cc_map.shape)[::-1]
                
                dist = numpy.sum(numpy.square((img-ref)))
                dists[i]=dist
                
                #dist=-scipy.spatial.distance.correlation(img.ravel(), ref.ravel())
                
                if (i+1) == al.ref_num:
                    actual = dist
                if dist < best[0]: best = (dist, i+1)
            idx=numpy.argsort(dists).squeeze()
            rank=numpy.nonzero(idx == (al.ref_num-1))[0]
            print "best: ", best, '--', "act: ", (actual, al.ref_num), 'rank:', rank
        format.write(trans_exp_align, values)
    
    