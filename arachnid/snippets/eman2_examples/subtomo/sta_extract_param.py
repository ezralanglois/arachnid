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
    
    data = []
    
    for i in xrange(EMAN2.EMUtil.get_image_count(alignment)):
        t = EMAN2.EMData()
        t.read_image(alignment, i)
        trans = t['xform.align3d']
        
        
        rot = trans.get_rotation('spider')
        psi, theta, phi = rot['psi'],rot['theta'],rot['phi']
        tx, ty, tz = trans.get_pre_trans()
        data.append([i+1, 7, i+1, psi, theta, phi, tx, ty, tz])
    numpy.savetxt(output, data, "%.4f", delimiter='\t')
        
        
        # norm, intervt, threshold
    
        
        #ptcl.process_inplace("xform",{"transform":ptcl_parms[0]["xform.align3d"]})
        
    