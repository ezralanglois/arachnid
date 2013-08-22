''' This script reindexs the image files for a relion star file

Download to edit and run: :download:`reindex_relion_star.py <../../arachnid/snippets/reindex_relion_star.py>`

To run:

.. sourcecode:: sh
    
    $ python reindex_relion_star.py

.. literalinclude:: ../../arachnid/snippets/reindex_relion_star.py
   :language: python
   :lines: 17-
   :linenos:
'''
import sys

from arachnid.core.metadata import format

if __name__ == '__main__':

    # Parameters
    selfile = sys.argv[1]
    newindexfile = sys.argv[2]
    oldindexfile = sys.argv[3]
    outputfile = sys.argv[4]
    
    
    newvals = format.read(newindexfile)
    oldvals = format.read(oldindexfile)
    valmap = {}
    for n,o in zip(newvals, oldvals): valmap[o.rlnImageName]=n.rlnImageName
    
    vals = format.read(selfile)
    newvals=[]
    for v in vals:
        newvals.append(v._replace(rlnImageName=valmap[v.rlnImageName]))
    
    
    format.write(outputfile, newvals)


