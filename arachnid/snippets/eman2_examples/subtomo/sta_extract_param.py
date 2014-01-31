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
    #coords = [line.strip().split() for line in open(sys.argv[3]).readlines()]
    #coords = [(float(c[0]), float(c[1]), float(c[2])) for c in coords]
    coords = numpy.loadtxt(sys.argv[3], numpy.int, '#', '\t')
    
    print alignment
    fout = open(output, 'w')
    if 1 == 1:
        param = numpy.zeros((len(coords), 3+3+3))
        aligndict = EMAN2.js_open_dict(alignment)
        for key in aligndict.keys():
            off = key.find('_')+1
            id = int(key[off:])
            print key, id
            i = id
            trans = aligndict[key][0]
            q = trans.get_rotation('quaternion')
            e0 = q['e0']
            e1 = -q['e1']
            e2 = -q['e2']
            e3 = -q['e3']
            
            q = numpy.asarray((e0, e1, e2, e3))
            q /= numpy.dot(q, q)
            e0, e1, e2, e3 = q
            trans.set_rotation({'type':'quaternion', 'e0':e0, 'e1': e1, 'e2':e2, 'e3':e3})
            rot = trans.get_rotation('spider')
            
            psi, theta, phi = rot['psi'],rot['theta'],rot['phi']
            #tx, ty, tz = trans.get_pre_trans()
            tx, ty, tz = trans.get_trans()
            if 1 == 1:
                tx = (coords[i][0] - tx)/8.0
                ty = (coords[i][1] - ty)/8.0
                tz = (coords[i][2] - tz)/8.0
            else:
                tx /= 8.0
                ty /= 8.0
                tz /= 8.0
            param[i, :] = (i+1, 7, -psi, -theta, -phi, tx, ty, tz, i+1)
        for i in xrange(len(param)):
            fout.write("%5d %d %11g %11g %11g %11g %11g %11g %11g\n"%tuple(param[i]))
    else:
        data = []
        for i in xrange(EMAN2.EMUtil.get_image_count(alignment)):
            t = EMAN2.EMData()
            t.read_image(alignment, i)
            trans = t['xform.align3d']
            
            #print coords[i][0], coords[i][1], coords[i][2]
            #trans.translate(coords[i][0], coords[i][1], coords[i][2])
            rot = trans.get_rotation('spider')
            psi, theta, phi = rot['psi'],rot['theta'],rot['phi']
            tx, ty, tz = trans.get_pre_trans()
            tx = (coords[i][0] - tx)/8.0
            ty = (coords[i][1] - ty)/8.0
            tz = (coords[i][2] - tz)/8.0
            #data.append([i+1, 7, i+1, psi, theta, phi, tx, ty, tz])
            data.append([i+1, 7, psi, theta, phi, tx, ty, tz, i+1])
            fout.write("%5d %d %11g %11g %11g %11g %11g %11g %11g\n"%(i+1, 7, psi, theta, phi, tx, ty, tz, i+1))
    fout.close()
    #numpy.savetxt(output, data, "%.4f", delimiter='\t')
        
        
        # norm, intervt, threshold
    
        
        #ptcl.process_inplace("xform",{"transform":ptcl_parms[0]["xform.align3d"]})
        
    