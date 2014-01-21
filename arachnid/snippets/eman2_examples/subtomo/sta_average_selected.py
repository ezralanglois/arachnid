''' Average an aligned stack of subtomograms

Download to edit and run: :download:`sta_average.py <../../arachnid/snippets/sta_average.py>`

To run:

.. sourcecode:: sh
    
    $ python sta_average.py

.. note::
    
    Requires EMAN2 2.1

.. literalinclude:: ../../arachnid/snippets/sta_average.py
   :language: python
   :lines: 20-
   :linenos:
'''
import sys

#from arachnid.core.image.eman2_utility 
import EMAN2
#import numpy

if __name__ == '__main__':

    # Parameters
    subtomo_stack = sys.argv[1]
    selected = sys.argv[2]
    output = sys.argv[3]
    
    lines = [line.strip() for line in open(selected, 'r')]
    select = []
    for line in lines[1:]:
        if 1 == 0:
            b = line.find(',')+1
            e = line.find(',', b)
            if e == -1: e = len(line)
            print b, e
        else:
            b = 0
            e = line.find(',')
        select.append(int(line[b:e]))
    
    #e.process_inplace(options.normproc[0],options.normproc[1])
    total=0
    avgr=EMAN2.Averagers.get('mean.tomo')
    print "Averaging %d subtomograms"%EMAN2.EMUtil.get_image_count(subtomo_stack)
    for i in select:
        e = EMAN2.EMData()
        e.read_image(subtomo_stack, i-1)
        #e*=-1
        #e.process_inplace('normalize', {})
        print i+1, e.get_attr('mean')
        avgr.add_image(e)
        total += 1
    print 'Total', total
    avg=avgr.finish()
    #mask=EMAN2.EMData(avg["nx"],avg["ny"],avg["nz"])
    #mask.to_one()
    #mask.process_inplace('mask.sharp', dict(outer_radius=72))
    #avg.mult(mask)
    avg.process_inplace('normalize', {})
    avg.process_inplace('filter.lowpass.gauss', dict(cutoff_freq=0.025))
    avg.write_image(output)






