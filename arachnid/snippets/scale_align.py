''' This script modifies the translations in an alignment file so the 
reconstruction can be done at a different interpolation level.

Download to edit and run: :download:`scale_align.py <../../arachnid/snippets/scale_align.py>`

To run:

.. sourcecode:: sh
    
    $ python scale_align.py

.. literalinclude:: ../../arachnid/snippets/scale_align.py
   :language: python
   :lines: 17-
   :linenos:
'''
from arachnid.core.metadata import format, spider_utility

if 1 == 0:
    import logging
    format._logger.setLevel(logging.DEBUG)
    format.star._logger.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    format._logger.addHandler(logging.StreamHandler())
    format.star._logger.addHandler(logging.StreamHandler())
if __name__ == '__main__':

    # Parameters
    
    align_file = ""
    output_file = ""
    stack_file = ""
    mult=2
    
    
    # Read an alignment file
    align = format.read(align_file)
    
    if hasattr(align[0], 'tx'):
        for i in xrange(len(align)):
            tx = align[i].tx
            ty = align[i].ty
            align[i] = align[i]._replace(tx=tx*mult, ty=ty*mult)
    else:
        for i in xrange(len(align)):
            tx = align[i].rlnOriginX
            ty = align[i].rlnOriginY
            fid = align[i].rlnImageName
            align[i] = align[i]._replace(rlnOriginX=tx*mult, rlnOriginY=ty*mult, rlnImageName=spider_utility.relion_filename(stack_file, align[i].rlnImageName))
    
    format.write(output_file, align)