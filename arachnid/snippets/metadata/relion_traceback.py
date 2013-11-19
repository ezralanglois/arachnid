''' This script modifies the translations in an alignment file so the 
reconstruction can be done at a different interpolation level.

Download to edit and run: :download:`relion_traceback.py <../../arachnid/snippets/relion_traceback.py>`

To run:

.. sourcecode:: sh
    
    $ python relion_traceback.py

.. literalinclude:: ../../arachnid/snippets/relion_traceback.py
   :language: python
   :lines: 17-
   :linenos:
'''
from arachnid.core.metadata import format, relion_utility

if __name__ == '__main__':

    # Parameters
    
    before_preprocess_file = ""
    after_preprocess_file = ""
    output_file = ""
    mult=1.0
    
    
    # Read an alignment file
    before = format.read(before_preprocess_file, numeric=True)
    after = format.read(after_preprocess_file, numeric=True)
    
    mapped = []
    for i in xrange(len(after)):
        id = relion_utility.relion_id(after[i].rlnImageName)[1]
        tx = after[i].rlnOriginX
        ty = after[i].rlnOriginY
        mapped.append( after[i]._replace(rlnImageName=before[id-1].rlnImageName, rlnOriginX=tx*mult, rlnOriginY=ty*mult) )
    
    format.write(output_file, mapped)



