'''
.. Created on Aug 20, 2013
.. codeauthor:: robertlanglois
'''

import sys

from arachnid.core.app import tracing
from arachnid.core.metadata import format
from arachnid.core.metadata import format_utility
from arachnid.core.metadata import spider_params, spider_utility
from arachnid.core.image import ndimage_file, rotate, ndimage_utility
from arachnid.core.image import reconstruct
from arachnid.core.image import ctf
from arachnid.core.orient import healpix,orient_utility
import numpy, scipy.io

def process_image(img, data, **extra):
    '''
    '''
    
    img = rotate.rotate_image(img, data[5], data[6], data[7])
    ctfimg = ctf.phase_flip_transfer_function(img.shape, data[17], **extra)
    img = ctf.correct(img, ctfimg)
    return img

if __name__ == '__main__':
    win_file = sys.argv[1]
    align_file = sys.argv[2]
    param_file = sys.argv[3]
    select_file = sys.argv[4]
    output = sys.argv[5]
    resolution = 2
    use_2d=True
    
    select = scipy.io.loadmat(select_file)
    indices = select['indices']
    print "Loaded selection file"
    print "# entries: ", indices.shape
    print "# classes: ", len(numpy.unique(indices[:, 1]))
    
    
    tracing.configure_logging()
    extra = spider_params.read(param_file)
    print "Loaded param file"
    extra.update(thread_count=32)
    
    data, header = format.read(align_file, ndarray=True)
    print "Loaded alignment file"
    #print header[5], header[6], header[7], header[17]
    #print data[0, 5], data[0, 6], data[0, 7], data[0, 17]
    img = ndimage_file.read_image(win_file)
    
    total = healpix.ang2pix(resolution, numpy.deg2rad(healpix.angles(resolution))[:, 1:], half=True).max()+1
    dpix = healpix.ang2pix(resolution, numpy.deg2rad(data[:, 1:3]), half=True)
    print "total number of views: ", total
    ang = healpix.angles(resolution)
    
    for i in xrange(10):
        idx = indices[indices[:, 1]==i, 0]-1
        print 'Reconstructing class', i+1, 'with', len(idx), 'projections'
        align_cl = data[idx].copy().squeeze()
        align_cl = align_cl[numpy.argsort(align_cl[:, 15])].copy().squeeze()
        align_cl[:, 16]-=1
        output = spider_utility.spider_filename(output, i+1)
        if use_2d:
            
            pix = healpix.ang2pix(resolution, numpy.deg2rad(align_cl[:, 1:3]), half=True)
            view_avg = {}
            for i, al in enumerate(align_cl):
                theta = al[1]
                if al[1]  > 179.999: theta -= 180.0
                rang = rotate.rotate_euler(ang[pix[i]], (-al[5], theta, al[2]))
                rot = (rang[0]+rang[2])
                rt3d = orient_utility.align_param_2D_to_3D_simple(al[5], al[6], al[7])
                psi, tx, ty = orient_utility.align_param_2D_to_3D_simple(rot, rt3d[1], rt3d[2])
                img = ndimage_file.read_image(spider_utility.spider_filename(win_file, al[15]), al[16])
                img = rotate.rotate_image(img, psi, tx, ty)
                if al[1]  > 179.999: img = ndimage_utility.mirror(img)
                if pix[i] not in view_avg: view_avg[pix[i]]= img.copy()
                else: view_avg[pix[i]] +=img
            j=0
            for i, v in view_avg.iteritems():
                print j, i
                ndimage_file.write_image(output, v, j)
                j+=1
                
        else:
            if resolution>0:
                pix = healpix.ang2pix(resolution, numpy.deg2rad(align_cl[:, 1:3]), half=True)
                views = numpy.histogram(pix, bins=total)[0]
                print "Number of missing views: ", numpy.sum(views==0)
                print "Minimum # of proj per view: ", views[views>0].min(), ' max:', views[views>0].max()
                idx = idx.tolist()
                for view, cnt in enumerate(views):
                    if cnt == 0:
                        sel = numpy.argwhere(dpix == view)
                        if len(sel) > 0:
                            sel=sel.squeeze()
                            idx.append(sel[0])
                idx = numpy.asarray(idx)
                align_cl = data[idx].copy().squeeze()
                align_cl = align_cl[numpy.argsort(align_cl[:, 15])].copy().squeeze()
                print 'Filled views - reconstructing with', len(idx), 'projections'
            even = numpy.arange(0, len(align_cl), 2, dtype=numpy.int)
            odd = numpy.arange(1, len(align_cl), 2, dtype=numpy.int)
            even_images = ndimage_file.iter_images(win_file, align_cl[even, 15:17].astype(numpy.int))
            odd_images = ndimage_file.iter_images(win_file, align_cl[odd, 15:17].astype(numpy.int))
            vol,vol_even,vol_odd = reconstruct.reconstruct3_bp3f_mp(img.shape[0], even_images, odd_images, align_cl[even], align_cl[odd], process_image=process_image, **extra)
            
            ndimage_file.write_image(output, vol)
            ndimage_file.write_image(format_utility.add_prefix(output, "even_"), vol_even)
            ndimage_file.write_image(format_utility.add_prefix(output, "odd_"), vol_odd)
    print 'Done'
    

