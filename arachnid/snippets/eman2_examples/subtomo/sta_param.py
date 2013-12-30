''' Extract parameters from EMAN2 JSON format

Download to edit and run: :download:`sta_param.py <../../arachnid/snippets/sta_param.py>`

To run:

.. sourcecode:: sh
    
    $ python sta_param.py

.. note::
    
    Requires EMAN2 2.1

.. literalinclude:: ../../arachnid/snippets/sta_param.py
   :language: python
   :lines: 20-
   :linenos:
'''
import sys

import EMAN2
import numpy

#../../full_dataset/cheri/patches_bin8_coord_861.txt
if __name__ == '__main__':

    # Parameters
    input = sys.argv[1]
    coords = sys.argv[2]
    output = sys.argv[3]
    
    aligndict = EMAN2.js_open_dict(input)
    offsets = numpy.loadtxt(coords, numpy.int, '#', '\t')*8
    
    param = numpy.zeros((len(offsets), 3+3+3))
    param[:, :3]=offsets
    for key in aligndict.keys():
        print key, aligndict[key][0]
        off = key.find('_')+1
        id = int(key[off:])
        trans = aligndict[key][0]
        tx, ty, tz = trans.get_pre_trans()
        #x, y, z = param[id-1, :3]
        #x = x - tx
        #y = y - ty
        #z = z - tz
        euler = trans.get_rotation('spider')
        param[id-1, 3:] = (tx, ty, tz, euler['psi'], euler['theta'], euler['phi'])
    numpy.savetxt(output, param, '%.3f', '\t', '\n')#, 'boxer_x, boxer_y, boxer_z, tx, ty, tz, psi, theta, phi')
        
        