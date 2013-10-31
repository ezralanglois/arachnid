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
    #ndimage_utility.normalize_standard_norm(img, mask, var_one, out=img)
    if mask is not None: return ndimage_utility.compress_image(img, mask)
    return img.ravel()

def read_data(frames, bin_factor, radius, offset):
    
        
    img = ndimage_file.read_image(frames[0], offset)
    if radius > 0: 
        mask = ndimage_utility.model_disk(radius, img.shape)
        if bin_factor > 1.0: mask=ndimage_interpolate.interpolate(mask, bin_factor)
    img = image_transform(img, bin_factor, mask)
    data = numpy.zeros((len(frames), img.shape[0]))
    label = numpy.zeros((len(data), 2))
    for i, frame in enumerate(frames):
        simg = ndimage_file.read_image(frame, offset)
        data[i, :] = image_transform(simg, bin_factor, mask)
        label[i, :] = (offset, i+1)
    return data, label

def embed(data, neighbors, dimension):
    
    dist2 = manifold.knn(data, neighbors, 10000)
    dist2 = manifold.knn_reduce(dist2, neighbors, True)
    feat, eigv, index = manifold.diffusion_maps_dist(dist2, dimension)
    print eigv
    return feat, index
    
#python frame_align_dm.py "frame_win/*5133.dat" dm.csv 0
if __name__ == '__main__':
    frames = sys.argv[1]
    output = sys.argv[2]
    offset=0
    bin_factor=1.0
    neighbors=3
    if len(sys.argv) > 3: offset=int(sys.argv[3])
    if len(sys.argv) > 4: bin_factor=float(sys.argv[4])
    if len(sys.argv) > 5: neighbors=int(sys.argv[5])
    
    radius=110
    mask=None
    dimension=10
    
    frames = glob.glob(frames)
    print "Read data from %d frames and bin factor %f"%(len(frames), bin_factor)
    data, label = read_data(frames, bin_factor, radius, offset)
    print "Embed with %d neighbors"%neighbors
    feat, index = embed(data, neighbors, dimension)
    if index is not None:
        print 'Found %d connected'%len(index)
        label = label[index]
    prefix='trans_%d_%.1f_%d'%(offset, bin_factor, neighbors)
    print format.write_dataset(output, feat, None, label, prefix=prefix)
