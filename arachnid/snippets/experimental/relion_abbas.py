''' This script converts a relion star file to a format required by Abbas

Download to edit and run: :download:`relion_abbas.py <../../arachnid/snippets/relion_abbas.py>`

To run:

.. sourcecode:: sh
    
    $ python relion_abbas.py

.. literalinclude:: ../../arachnid/snippets/relion_abbas.py
   :language: python
   :lines: 17-
   :linenos:
'''
import sys,os

from arachnid.core.metadata import format, spider_utility, relion_utility
from arachnid.core.image import ndimage_file, ndimage_utility

if __name__ == '__main__':

    # Parameters
    selfile = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "win_0000000.dat"
    align_file = os.path.join(os.path.dirname(output), 'align.csv')
    ctf_file = os.path.join(os.path.dirname(output), 'ctf.csv')
    
    data = format.read(selfile, numeric=True)
    
    angs=[]
    ctfs=[]
    for i, entry in enumerate(data):
        filename, pid = relion_utility.relion_file(entry.rlnImageName)
        img = ndimage_file.read_image(filename, pid-1)
        img = ndimage_utility.fourier_shift(img, entry.rlnOriginX, entry.rlnOriginY)
        ndimage_file.write_image(spider_utility.spider_filename(output, i+1), img)
        angs.append((entry.rlnAnglePsi, entry.rlnAngleTilt, entry.rlnAngleRot))
        ctfs.append((entry.rlnDefocusU,entry.rlnSphericalAberration,entry.rlnAmplitudeContrast,entry.rlnVoltage))
    format.write(align_file, angs, header="psi,theta,phi".split(','))
    format.write(ctf_file, ctfs, header="defocus,cs,ampcont,voltage".split(','))

    
    