''' 

Download to edit and run: :download:`particle_align_dm.py <../../arachnid/snippets/particle_align_dm.py>`

To run:

.. sourcecode:: sh
    
    $ python particle_align_dm.py

.. literalinclude:: ../../arachnid/snippets/particle_align_dm.py
   :language: python
   :lines: 17-
   :linenos:
'''
import sys, glob

from arachnid.core.metadata import format
from arachnid.core.image import ndimage_file, ndimage_utility, ndimage_interpolate, manifold
import numpy

def image_transform(img, bin_factor, mask=None, var_one=True):
    '''
    '''
    
    if bin_factor > 1: img=ndimage_interpolate.interpolate(img, bin_factor)
    ndimage_utility.normalize_standard_norm(img, mask, var_one, out=img)
    if mask is not None: return ndimage_utility.compress_image(img, mask)
    return img.ravel()
    
#python frame_align_dm.py "frame_win/*5133.dat" dm.csv 0
if __name__ == '__main__':
    frames = sys.argv[1]
    output = sys.argv[2]
    offset=0
    if len(sys.argv) > 3: offset=int(sys.argv[3])
    trans_file = sys.argv[4] if len(sys.argv) > 4 else None
    bin_factor=1.0
    radius=110
    mask=None
    neighbors=200
    dimension=10
    rng=2
    beg=0.0
    jitter=100
    frame_beg=3
    frame_end=24
    
    untrans=None 
    frames = glob.glob(frames)
    img = ndimage_file.read_image(frames[0], offset)
    if radius > 0: 
        mask = ndimage_utility.model_disk(radius, img.shape)
        if bin_factor > 1.0: mask=ndimage_interpolate.interpolate(mask, bin_factor)
    avg = img.copy()
    img = image_transform(img, bin_factor, mask)
    data = numpy.zeros((len(frames), img.shape[0]))
    avg[:]=0
    trans = None
    if trans_file is not None:
        untrans = format.read(trans_file, spiderid=frames[0], numeric=True)
    for i, frame in enumerate(frames):
        simg = ndimage_file.read_image(frame, offset)
        if trans is not None:
            j = i-frame_beg if len(trans) == (frame_end-frame_beg) else i
            print 'Untrans:', j, -trans[j].dx, -trans[j].dy
            simg=ndimage_utility.fourier_shift(simg, -trans[j].dx, -trans[j].dy) #cwrc
        if 1 == 0:
            data[i, :] = image_transform(simg, bin_factor, mask)
        avg += simg
   
    
    prefix = "untrans" if untrans is not None else "trans"
    ndimage_file.write_image(output+"_%s.%d.spi"%(prefix, offset), avg)#*mask)
    ndimage_file.write_image(output+"_%s.%d.m.spi"%(prefix, offset), avg*mask)
    cimg=numpy.sum(data, axis=0)
    img2 = numpy.zeros(mask.shape)
    img2.ravel()[mask.ravel() > 0.5]=cimg
    img2.ravel()[numpy.logical_not(mask.ravel()>0.5)]=numpy.mean(cimg)
    #ndimage_file.write_image(output+".%d.mask.spi"%offset, img2) 
    if 1 == 0:
        dist2 = manifold.knn(data, neighbors, 10000)
        dist2 = manifold.knn_reduce(dist2, neighbors, True)
        feat, eigv, index = manifold.diffusion_maps_dist(dist2, dimension)
        print eigv
        label = numpy.arange(len(frames))
        if index is not None: label = label[index]
        format.write_dataset(output, numpy.hstack((feat, label[:, numpy.newaxis])), None, label)
    
    trans = []
    off=rng/2
    for i in xrange(jitter):
        v1 = numpy.random.rand()*rng-off
        v1 = v1+beg if v1 > 0 else v1-beg
        v2 = numpy.random.rand()*rng-off
        v2 = v2+beg if v2 > 0 else v2-beg
        trans.append((v1,v2))
    trans = numpy.asarray(trans)
    trans[0, :] = (0,0)
    data2 = numpy.zeros((len(frames)*len(trans), img.shape[0]))
    label = numpy.zeros((len(data2), 2))
    ttrans = numpy.zeros((len(data2), 2))
    k=0
    for i, frame in enumerate(frames):
        img = ndimage_file.read_image(frame, offset)
        if untrans is not None:
            j = i-frame_beg if len(untrans) == (frame_end-frame_beg) else i
            print 'Untrans:', j, -untrans[j].dx, -untrans[j].dy
            img=ndimage_utility.fourier_shift(img, -untrans[j].dx, -untrans[j].dy) #cwrc
        for j in xrange(len(trans)):
            label[k, :] = (i+1, j+1)
            if j > 0:
                v1 = numpy.random.rand()*rng-off
                v1 = v1+beg if v1 > 0 else v1-beg
                v2 = numpy.random.rand()*rng-off
                v2 = v2+beg if v2 > 0 else v2-beg
            else: v1, v2 = (0,0)
            ttrans[k, :]=(v1, v2)
            simg = ndimage_utility.fourier_shift(img, v1, v2)
            data2[k, :] = image_transform(simg, bin_factor, mask)
            k += 1
    
    dist2 = manifold.knn(data2, neighbors, 10000)
    dist2 = manifold.knn_reduce(dist2, neighbors, True)
    feat, eigv, index = manifold.diffusion_maps_dist(dist2, dimension)
    print eigv
    if index is not None: 
        print 'index=', len(index)
        label = label[index]
        ttrans = ttrans[index]
        data3=data2[index]
    else: data3=data2
    neigh = manifold.knn(feat[:, :1].copy(), len(trans), 10000)
    ucol = neigh.col.reshape((len(feat), len(trans)+1))/len(trans)
    col = neigh.col.reshape((len(feat), len(trans)+1))
    dist = neigh.data.reshape((len(feat), len(trans)+1))
    cl = numpy.arange(len(data))
    min_dist = (1e20, 0, None)
    for i in xrange(len(feat)):
        tot = len(numpy.intersect1d(ucol[i], cl))
        dmap = {}
        for j in xrange(len(dist[i])): 
            if ucol[i, j] not in dmap: dmap[ucol[i, j]]=dist[i,j]
        d = numpy.sum(dmap.values())
        if tot > min_dist[1]: min_dist = (d, tot, i)
        elif d < min_dist[0]: min_dist = (d, tot, i)
    print "min dist: ", min_dist
    best = numpy.zeros(len(feat))
    dmap={}
    i=min_dist[1]
    if i is not None:
        for j in xrange(len(dist[i])):
            if ucol[i, j] not in dmap: dmap[ucol[i, j]]=col[i,j]
        #print len(dmap.values()), len(cl), len(data), dmap.values()
        best[numpy.asarray(dmap.values()).astype(numpy.int)]=1
        cimg=numpy.sum(data3[best.astype(numpy.bool)], axis=0)
        img = numpy.zeros(mask.shape)
        img.flat[mask.astype(numpy.bool).ravel()]=cimg
        img.flat[numpy.logical_not(mask.astype(numpy.bool).ravel())]=numpy.mean(cimg)
        ndimage_file.write_image(output+"_%s.%d.exp.spi"%(prefix, offset), img) #ndimage_utility.uncompress(img, mask))
    else: print "No best"
    prefix = "untrans_%d_%d_"%(len(trans), offset) if untrans is not None else "trans_%d_%d_"%(len(trans), offset) 
    print format.write_dataset(output, numpy.hstack((feat, label, ttrans,best[:, numpy.newaxis])), None, label, prefix=prefix)
