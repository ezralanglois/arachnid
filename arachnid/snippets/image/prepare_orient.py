''' Prepare for orientation recovery

Download to edit and run: :download:`prepare_orient.py <../../arachnid/snippets/prepare_orient.py>`

To run:

.. sourcecode:: sh
    
    $ python prepare_orient.py

.. note::
    
    You must have Arachnid and Matplotlib installed to run this script

.. literalinclude:: ../../arachnid/snippets/prepare_orient.py
   :language: python
   :lines: 22-
   :linenos:
'''
import sys
sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')
from arachnid.core.image import ndimage_file, eman2_utility
from arachnid.core.orient import orient_utility
from arachnid.core.metadata import format

if __name__ == '__main__':

    # Parameters
    input_file = sys.argv[1]
    align_file = sys.argv[2]
    outputfile = sys.argv[3]
    size=32
    
    align = format.read_alignment(align_file)
    for i, img in enumerate(ndimage_file.iter_images(input_file)):
        dx, dy = orient_utility.align_param_2D_to_3D(align[i].psi, align[i].tx, align[i].ty)
        img = eman2_utility.fshift(img, dx, dy)
        img = eman2_utility.decimate(img, float(img.shape[0])/size)
        ndimage_file.write_image(outputfile, img, i)



    