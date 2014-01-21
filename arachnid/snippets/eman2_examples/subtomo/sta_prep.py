''' Prepare stacks for MSA

Download to edit and run: :download:`sta_prep.py <../../arachnid/snippets/sta_prep.py>`

To run:

.. sourcecode:: sh
    
    $ python sta_prep.py

.. note::
    
    Requires EMAN2 2.1

.. literalinclude:: ../../arachnid/snippets/sta_prep.py
   :language: python
   :lines: 20-
   :linenos:
'''
import sys

#from arachnid.core.image.eman2_utility 
import EMAN2, utilities
#import numpy

if __name__ == '__main__':

    # Parameters
    subtomo_stack = sys.argv[1]
    output = sys.argv[2]
    keep = 1.0
    
    if keep < 1.0:
        thresh=[]
        for i in xrange(EMAN2.EMUtil.get_image_count(subtomo_stack)):
            e = EMAN2.EMData()
            e.read_image(subtomo_stack, i)
            thresh.append( e['spt_score'])
        thresh.sort()
        keep=thresh[int(keep*len(thresh))-1]
        print "Threshold", keep
    else: keep=None
    #spt_score
    mask = None
    #e.process_inplace(options.normproc[0],options.normproc[1])
    total=0
    print "Averaging %d subtomograms"%EMAN2.EMUtil.get_image_count(subtomo_stack)
    for i in xrange(EMAN2.EMUtil.get_image_count(subtomo_stack)):
        e = EMAN2.EMData()
        e.read_image(subtomo_stack, i)
        if keep is not None and keep < e['spt_score']: continue
        if mask is None: mask = utilities.model_circle(72, e["nx"],e["ny"],e["nz"])
        e.process_inplace('normalize.mask', dict(mask=mask))
        e.mult(mask)
        e.write_image(output, total)
        print i+1, e.get_attr('mean')
        total += 1
    mask.write_image('mask.hdf')
    print 'Total', total
