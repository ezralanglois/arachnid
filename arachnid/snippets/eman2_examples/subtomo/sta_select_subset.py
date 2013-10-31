''' Reextract aligned particles from the tomograph

Download to edit and run: :download:`sta_reextract.py <../../arachnid/snippets/sta_reextract.py>`

To run:

.. sourcecode:: sh
    
    $ python sta_reextract.py

.. note::
    
    Requires EMAN2 2.1

.. literalinclude:: ../../arachnid/snippets/sta_reextract.py
   :language: python
   :lines: 20-
   :linenos:
'''
import sys

#from arachnid.core.image.eman2_utility 
import EMAN2
import numpy
#, scipy.spatial.distance

if __name__ == '__main__':

    # Parameters
    subtomo_stack = sys.argv[1]
    partcoords = sys.argv[2]
    output = sys.argv[3]
    
    coords = numpy.asarray([[int(val) for val in line.split()]for line in open(partcoords,'r')])
    
    j=0
    print "Selecting %d subtomograms from %d"%(len(coords), EMAN2.EMUtil.get_image_count(subtomo_stack))
    for i in xrange(EMAN2.EMUtil.get_image_count(subtomo_stack)):
        e = EMAN2.EMData()
        e.read_image(subtomo_stack, i)
        ptcl_coord=numpy.asarray(e['ptcl_source_coord'])
        #dist_cent = scipy.spatial.distance.cdist(coords, ptcl_coord.reshape((1, len(ptcl_coord))), metric='euclidean').ravel()
        dist_cent = [numpy.linalg.norm(coord-ptcl_coord) for coord in coords]
        dist_cent = numpy.asarray(dist_cent)
        idx = dist_cent.argmin()
        if dist_cent[idx] < 72:
            print 'Found: ', idx, dist_cent[idx]
            e.write_image(output, j)
            j+=1
        else: print 'Missing', dist_cent[idx]

