''' Place average in the position of each subtomogram in the full tomogram

Download to edit and run: :download:`sta_place_average.py <../../arachnid/snippets/sta_place_average.py>`

To run:

.. sourcecode:: sh
    
    $ python sta_place_average.py

.. note::
    
    Requires EMAN2 2.1

.. literalinclude:: ../../arachnid/snippets/sta_place_average.py
   :language: python
   :lines: 20-
   :linenos:
'''
import sys
import EMAN2
import EMNumPy
import numpy

if __name__ == '__main__':

    # Parameters
    average = sys.argv[1]
    tomogram = sys.argv[2]
    alignment = sys.argv[3]
    coords = sys.argv[4]
    output = sys.argv[5]
    
    coords = numpy.loadtxt(coords, numpy.int, '#', '\t')
    
    tomo = EMAN2.EMData()
    tomo.read_image(tomogram)
    tomo.to_zero()
    avg = EMAN2.EMData()
    avg.read_image(average)
    
    nx = avg.get_xsize()
    ny = nx
    nz = nx
    nptomo = EMNumPy.em2numpy(tomo)
    aligndict = EMAN2.js_open_dict(alignment)
    for key in aligndict.keys():
        off = key.find('_')+1
        id = int(key[off:])
        trans = aligndict[key][0]
        itrans = trans.inverse()
        cavg = avg.copy()
        cavg.transform(itrans)
        x, y, z = coords[id]
        nptomo[x:x+nx, y:y+ny, z:z+nz] = avg
    tomo.write_image(output)
        
        
        
    
    