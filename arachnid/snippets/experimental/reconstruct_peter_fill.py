'''
.. Created on Aug 20, 2013
.. codeauthor:: robertlanglois
'''

import sys

from arachnid.core.app import tracing
from arachnid.core.metadata import format
from arachnid.core.metadata import format_utility
#from arachnid.core.metadata import spider_params, spider_utility
from arachnid.core.image import ndimage_file, ndimage_utility, reproject #, rotate, ndimage_utility
from arachnid.core.image import reconstruct
#from arachnid.core.image import ctf
from arachnid.core.orient import healpix #,orient_utility
import numpy, scipy.io

def process_image(img, data, **extra):
    '''
    '''
    
    return ndimage_utility.invert(img)

def iter_images(win_file, index, projections):
    '''
    '''
    
    count = ndimage_file.count_images(win_file)
    for img in ndimage_file.iter_images(win_file, index[index < count]):
        yield img
    for i in index[index >= count]:
        yield projections[i]

if __name__ == '__main__':
    win_file = sys.argv[1]
    align_file = sys.argv[2]
    select_file = sys.argv[3]
    vol_file = sys.argv[4]
    output = sys.argv[5]
    resolution = 2
    use_2d=True
    
    select = scipy.io.loadmat(select_file)
    idx = select['ind']-1
    print "Loaded selection file"
    print "# entries: ", idx.shape
    
    
    tracing.configure_logging()
    extra = {}
    print "Loaded param file"
    extra.update(thread_count=32)
    
    data, header = format.read(align_file, ndarray=True)
    print "Loaded alignment file"
    img = ndimage_file.read_image(win_file)
    align_cl = data[idx, :].copy().squeeze()
    align_cl[:, 0] = -align_cl[:, 5]
    if resolution>0:
        refvol = ndimage_file.read_image(vol_file)
        total = healpix.ang2pix(resolution, numpy.deg2rad(healpix.angles(resolution))[:, 1:], half=True).max()+1
        dpix = healpix.ang2pix(resolution, numpy.deg2rad(data[:, 1:3]), half=True)
        print "total number of views: ", total
        ang = healpix.angles(resolution)
        pix = healpix.ang2pix(resolution, numpy.deg2rad(align_cl[:, 1:3]), half=True)
        
        views = numpy.histogram(pix, bins=total)[0]
        print "Number of missing views: ", numpy.sum(views==0)
        print "Minimum # of proj per view: ", views[views>0].min(), ' max:', views[views>0].max()
        old_align=align_cl
        missing = []
        for view, cnt in enumerate(views):
            if cnt == 0:missing.append(view)
        projections = []
        align_cl = numpy.zeros((len(old_align)+len(missing), old_align.shape[1]))
        align_cl[:len(align_cl)] = old_align
        for i in missing:
            euler = ang[i]
            align_cl[i+len(old_align), :3]=ang[i]
            avg = reproject.reproject_3q_single(refvol, img.shape[0]/2, euler.reshape((1, 3)))[0]
            ndimage_utility.invert(avg,avg)
            projections.append(avg)
    even = numpy.arange(0, len(align_cl), 2, dtype=numpy.int)
    odd = numpy.arange(1, len(align_cl), 2, dtype=numpy.int)
    even_images = iter_images(win_file, even, projections)
    odd_images = iter_images(win_file, odd, projections)
    vol,vol_even,vol_odd = reconstruct.reconstruct3_bp3f_mp(img.shape[0], even_images, odd_images, align_cl[even], align_cl[odd], process_image=process_image, **extra)
    
    ndimage_file.write_image(output, vol)
    ndimage_file.write_image(format_utility.add_prefix(output, "even_"), vol_even)
    ndimage_file.write_image(format_utility.add_prefix(output, "odd_"), vol_odd)
    print 'Done'
    

