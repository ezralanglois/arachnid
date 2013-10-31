''' Transform a set of images into a matrix for matlab

Download to edit and run: :download:`transform_image.py <../../arachnid/snippets/transform_image.py>`

To run:

.. sourcecode:: sh
    
    $ python transform_image.py

.. literalinclude:: ../../arachnid/snippets/transform_image.py
   :language: python
   :lines: 28-
   :linenos:
'''
from arachnid.core.metadata import format, spider_utility, format_utility
from arachnid.core.image import ndimage_file, ndimage_utility, ndimage_processor, ndimage_interpolate, rotate
import numpy, logging, scipy.io, os

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
            logging.debug("Alignment exists")
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


def rotational_sample(label, align, nsamples, angle_range, **extra):
    '''
    '''
    
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
    mask = ndimage_utility.model_disk(int(pixel_diameter/2.0), img.shape)
    bin_factor = max(1, min(8, resolution / (apix*2))) if resolution > (2*apix) else 1
    logging.info("Decimation factor %f for resolution %f and pixel size %f"%(bin_factor, resolution, apix))
    if bin_factor > 1: mask = ndimage_interpolate.decimate(mask, bin_factor)
    return mask

def image_transform(img, i, mask, resolution, apix, var_one=True, align=None, **extra):
    '''
    '''
    
    if align[i, 1] > 179.999: img = ndimage_utility.mirror(img)
    if align[i, 0] != 0: img = rotate.rotate_image(img, align[i, 0], 0, 0)
    ndimage_utility.vst(img, img)
    bin_factor = max(1, min(8, resolution / (apix*2))) if resolution > (2*apix) else 1
    if bin_factor > 1: img = ndimage_interpolate.decimate(img, bin_factor)
    ndimage_utility.normalize_standard(img, mask, var_one, img)
    if 1 == 0:
        img, freq = ndimage_utility.bispectrum(img, 0.5, 'gaussian', 1.0)
        freq;
        img = numpy.log10(numpy.sqrt(numpy.abs(img.real+1)))
    
    #img = ndimage_utility.compress_image(img, mask)
    return img

def group_by_reference(label, align, ref):
    '''
    '''
    
    group=[]
    refs = numpy.unique(ref)
    for r in refs:
        sel = r == ref
        group.append((r, label[sel], align[sel]))
    return group


if __name__ == '__main__':
    input_image_file = ""
    input_align_file =""
    output_mat_file="data.mat"
    nsamples=20
    angle_range=3
    pixel_diameter=220 
    resolution=40
    apix=1.5
    thread_count=8
    
    r, label, align = group_by_reference(*read_alignment([input_image_file], input_align_file))[0]
    label, align = rotational_sample(label, align, nsamples, angle_range)
    mask = create_mask([input_image_file], pixel_diameter, resolution, apix)
    data = ndimage_processor.read_image_mat([input_image_file], label, image_transform, shared=False, mask=mask, cache_file=None, align=align, resolution=resolution, apix=apix, thread_count=thread_count)
    scipy.io.savemat(output_mat_file, {'data': data})
    
