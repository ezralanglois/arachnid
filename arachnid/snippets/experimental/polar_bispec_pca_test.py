''' Embed a set of rotationally invariance images

Download to edit and run: :download:`polar_bispec_pca_test.py <../../arachnid/snippets/polar_bispec_pca_test.py>`

    
Right click on link -> "Copy Link Address" or "Copy Link Location"
Download from the command line:

.. sourcecode:: sh
    
    $ wget <paste-url-here>

How to run command:

.. sourcecode:: sh
    
    $ python polar_bispec_pca_test.py data_000.spi align.spi 2 view_stack.spi

.. literalinclude:: ../../arachnid/snippets/polar_bispec_pca_test.py
   :language: python
   :lines: 23-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

from arachnid.core.metadata import format_alignment, format #format, 
from arachnid.core.image import ndimage_file,  ndimage_utility, manifold, ndimage_interpolate
from arachnid.core.learn import dimensionality_reduction
import numpy, logging

def image_transform(img, mask, bin_factor, align):
    '''
    '''
    
    #ndimage_utility.vst(img, img)
    #bin_factor = max(1, min(8, resolution / (apix*4))) if resolution > (4*apix) else 1
    if align[1] > 180.0: img = ndimage_utility.mirror(img)
    if bin_factor > 1: img = ndimage_interpolate.downsample(img, bin_factor)
    #ndimage_utility.normalize_standard_norm(img, mask, True, out=img)
    ndimage_utility.normalize_standard(img, mask, True, out=img)
    if 1 == 1:
        #img=ndimage_utility.polar(img)
        #img = ndimage_utility.polar_simple(img)
        img = ndimage_utility.bispectrum(img, int(img.shape[0]-1), 'uniform')[0]
        img = numpy.vstack((img.real, img.imag)).T
    #img = numpy.log10(numpy.abs(img.real)+1)
    return img

if __name__ == '__main__':

    # Parameters
    
    image_file = sys.argv[1]        # image_file = "data/dala_01.spi"
    align_file = sys.argv[2]        # align_file = "data/align_01.spi"
    output_file=sys.argv[3]         # output_file="stack01.spi"
    radius = 88
    bin_factor = 4
    
    logging.basicConfig(log_level=logging.INFO)
    files, align = format_alignment.read_alignment(align_file, image_file)
    ref = align[:, 3].astype(numpy.int)
    refs = numpy.unique(ref)
    print len(refs)
    sel = numpy.logical_or(ref == refs[2], ref == refs[1])
    print numpy.sum(sel), numpy.sum(ref == refs[2]), numpy.sum(ref == refs[1])
    align = align[sel]
    ref = ref[sel]
    label = numpy.argwhere(sel).squeeze()
    files = [files[i] for i in label]
    
    data = None
    mask = None
    for i, img in enumerate(ndimage_file.iter_images(files)):
        if (i%100)==0:print i+1, ' of ', len(files)
        if mask is None:
            mask = ndimage_utility.model_disk(radius, img.shape)
            if bin_factor > 1: mask = ndimage_interpolate.downsample(mask, bin_factor)
        print align[i, :3]
        img = image_transform(img, mask, bin_factor, align[i])
        if data is None: data = numpy.zeros((len(align), img.ravel().shape[0]), dtype=img.dtype)
        data[i, :] = img.ravel()
    if 1 == 0:
        data2 = data.copy()
        cnt=15
        for v in (refs[2], refs[1]):
            idx = numpy.argwhere(ref == v)
            for i in xrange(len(idx)):
                avg = data[idx[i]].copy()
                for j in xrange(i-cnt, i):
                    if j < 0: j = len(idx)+j
                    avg += data[idx[j]]
                data2[idx[i]]=avg/(cnt+1)
        data=data2
    else:
        data[ref == refs[2]]=manifold.local_neighbor_average(data[ref == refs[2]], manifold.knn(data[ref == refs[2]], 3, 10000))
        data[ref == refs[1]]=manifold.local_neighbor_average(data[ref == refs[1]], manifold.knn(data[ref == refs[1]], 3, 10000))
    
    good = ref==refs[2]
    if 1 == 0:
        data=manifold.local_neighbor_average(data, manifold.knn(data, 5, 10000))
        feat, eigv, index = manifold.diffusion_maps(data, 5, 20, True, 10000)
        if index is not None: 
            print index
            label = label[index]
            good = good[index]
    elif 1 == 0:
        eigv, feat = dimensionality_reduction.pca_fast(data, data, 5)[1:]
    else:
        eigv, feat = dimensionality_reduction.dhr_pca(data, data, 5, iter=5)
    print eigv[:10]
    format.write_dataset(output_file, feat, None, label, good)
    
    
    