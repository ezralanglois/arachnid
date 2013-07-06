'''  Convert FREALIGN stacks and alignment to relion star file

Download to edit and run: :download:`frealign2relion.py <../../arachnid/snippets/frealign2relion.py>`

To run:

.. sourcecode:: sh
    
    $ python frealign2relion.py vol.spi ref_stack.spi 2

.. literalinclude:: ../../arachnid/snippets/frealign2relion.py
   :language: python
   :lines: 22-
   :linenos:
'''
import sys, os
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')
from arachnid.core.image import ndimage_file, eman2_utility, ndimage_utility
from arachnid.core.metadata import format, spider_params, spider_utility
import glob

#python frealign2relion.py "data/stack040411_256x256_*.stk" data/stack040411_256x256_4_fixed.par ../cluster/data/params.dat relion_stack_01.spi

if __name__ == '__main__':

    # Parameters
    image_files = sys.argv[1]
    select_file = sys.argv[2]
    params_file = sys.argv[3]
    output = sys.argv[4]
    
    param = spider_params.read(params_file)
    voltage = param['voltage']
    cs = param['cs']
    ampcont = param['ampcont']
    header="rlnImageName,rlnMicrographName,rlnDefocusU,rlnDefocusV,rlnVoltage,rlnSphericalAberration,rlnAmplitudeContrast,rlnGroupNumber".split(',')
    pixel_radius = int(param['pixel_diameter']/2.0)
    vals = []
    for image_file in glob.glob(image_files):
        #tmpheader = ['id', 'psi', 'theta', 'phi', 'shx', 'shy', 'mag', 'film', 'defocusu', 'defocusv', 'unk1', 'unk2']
        select = format.read(select_file, spiderid=image_file, numeric=True)
        print "Selection: %s - %d"%(image_file, len(select))
        idlen = len(str(len(select)))
        output = spider_utility.spider_filename(output, image_file)
        for s in select:
            vals.append( tuple(["%s@%s"%(str(s.id).zfill(idlen), output), str(s.film), s.defocusu, s.defocusv, voltage, cs, ampcont, s.film]) )
    
        sys.stdout.flush()
        mask = None
        print output
        if 1 == 0:
            for i, img in enumerate(ndimage_file.mrc.iter_images(image_file)):
                if mask is None:
                    mask = eman2_utility.model_circle(pixel_radius, img.shape[0], img.shape[1])*-1+1
                ndimage_utility.normalize_standard(img, mask, out=img)
                ndimage_file.write_image(output, img, i)
    
    format.write(os.path.splitext(output)[0]+".star", vals, header=header)