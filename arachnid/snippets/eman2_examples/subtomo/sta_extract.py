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
    tomogram = sys.argv[1]
    partcoords = sys.argv[2]
    alignment = sys.argv[3]
    output = sys.argv[4]
    boxsize = int(sys.argv[5]) if len(sys.argv)>5 else 200
    cshrink=8
    invert=True
    norm=False
    
    
    coords = numpy.asarray([[int(val) for val in line.split()]for line in open(partcoords,'r')])
    
    #get_trans
    #get_rotation_transform
    
    for i in xrange(len(coords)):
        print i+1, len(coords)
        t = EMAN2.EMData()
        t.read_image(alignment, i)
        trans = t['xform.align3d']
        tx, ty, tz = trans.get_pre_trans()
        rx, ry, rz = tx-int(tx), ty-int(ty), tz-int(tz)
        x, y, z = coords[i]
        x=round(x*cshrink)
        y=round(y*cshrink)
        z=round(z*cshrink)
        x = x - int(tx)
        y = y - int(ty)
        z = z - int(tz)
        r = EMAN2.Region((2*x - boxsize)/2,(2*y - boxsize)/2, (2*z - boxsize)/2, boxsize, boxsize, boxsize)
        e = EMAN2.EMData()
        e.read_image(tomogram,0,False,r)
        if invert:
            e = e*-1
        if norm:
            e.process_inplace('normalize', {})
        #rot = EMAN2.Transform()
        #rot.set_rotation(trans.get_rotation())
        rot=EMAN2.Transform()
        rot=trans
        rot.set_scale(1.0)
        rot.set_mirror(False)
        rot.set_pre_trans((rx, ry, rz))
        #rot=trans.get_rotation_transform()
        e.process_inplace("xform",{"transform":rot})
        e['spt_score']=t['spt_score']
        e['xform.align3d']=rot
        e['origin_x'] = 0
        e['origin_y'] = 0
        e['origin_z'] = 0
        e.write_image(output, i)
        
        # norm, intervt, threshold
    
        
        #ptcl.process_inplace("xform",{"transform":ptcl_parms[0]["xform.align3d"]})
        
    