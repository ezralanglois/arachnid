''' This script estimates the diameter of the particle from a reference

Download to edit and run: :download:`relion_abbas.py <../../arachnid/snippets/relion_abbas.py>`

To run:

.. sourcecode:: sh
    
    $ python relion_abbas.py

.. literalinclude:: ../../arachnid/snippets/relion_abbas.py
   :language: python
   :lines: 17-
   :linenos:
'''
import sys

from arachnid.core.image import ndimage_file, ndimage_utility, manifold
import numpy, scipy.ndimage, scipy.spatial.distance

if __name__ == '__main__':

    # Parameters
    input_vol = sys.argv[1]
    apix = float(sys.argv[2]) if len(sys.argv) > 2 else None
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else None
    
    header={}
    vol = ndimage_file.read_image(input_vol, header=header)
    print header
    if 'apix' in header: apix=header['apix']
    if apix is None: raise ValueError, "No pixel size specified"
    
    mask, thresh = ndimage_utility.tight_mask(vol, threshold, 0, 0)
    print "Threshold:", thresh
    mask2=scipy.ndimage.binary_dilation(mask, scipy.ndimage.generate_binary_structure(mask.ndim, 2), 1)
    print mask.sum(), mask2.sum()
    
    mask2 *= mask*-1+1
    mask=mask2
    #mask *= ndimage_utility.dialate_mask(mask, 1)*-1+1
    
    print len(numpy.nonzero(mask.ravel()))
    y,x,z = numpy.unravel_index(numpy.nonzero(mask.ravel()), mask.shape)
    coords = numpy.zeros((len(x.ravel()), 3))
    coords[:, 0]=x.ravel()
    coords[:, 1]=y.ravel()
    coords[:, 2]=z.ravel()
    print coords.shape
    dist = numpy.sqrt(manifold.max_dist(coords))-2
    #dist = scipy.spatial.distance.pdist(coords, metric='euclidean').max()-2
    print dist, dist*apix
    
    
    