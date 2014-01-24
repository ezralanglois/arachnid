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

if __name__ == '__main__':

    # Parameters
    alignment = sys.argv[1]
    output = sys.argv[2]
    coords = [line.strip().split() for line in open(sys.argv[3]).readlines()]
    coords = [(float(c[0]), float(c[1]), float(c[2])) for c in coords]
    
    data = []
    
    for i in xrange(EMAN2.EMUtil.get_image_count(alignment)):
        t = EMAN2.EMData()
        t.read_image(alignment, i)
        trans = t['xform.align3d']
        
        trans.translate(coords[i][0], coords[i][1], coords[i][2])
        rot = trans.get_rotation('spider')
        psi, theta, phi = rot['psi'],rot['theta'],rot['phi']
        tx, ty, tz = trans.get_pre_trans()
        data.append([i+1, 7, i+1, psi, theta, phi, tx/8, ty/8, tz/8])
    numpy.savetxt(output, data, "%.4f", delimiter='\t')
        
        
        # norm, intervt, threshold
    
        
        #ptcl.process_inplace("xform",{"transform":ptcl_parms[0]["xform.align3d"]})
        
    