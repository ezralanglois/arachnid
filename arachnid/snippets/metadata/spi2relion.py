'''  Convert SPIDER single-stack to relion

Download to edit and run: :download:`spi2relion.py <../../arachnid/snippets/spi2relion.py>`

To run:

.. sourcecode:: sh
    
    $ python spi2relion.py vol.spi ref_stack.spi 2

.. literalinclude:: ../../arachnid/snippets/spi2relion.py
   :language: python
   :lines: 22-
   :linenos:
'''
import sys, os
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')
from arachnid.core.image import ndimage_file, eman2_utility, ndimage_utility
from arachnid.core.metadata import format, format_utility, spider_params

if __name__ == '__main__':

    # Parameters
    image_file = sys.argv[1]
    select_file = sys.argv[2]
    defocus_file = sys.argv[3]
    params_file = sys.argv[4]
    output = sys.argv[5]
    
    header="rlnImageName,rlnMicrographName,rlnDefocusU,rlnVoltage,rlnSphericalAberration,rlnAmplitudeContrast,rlnGroupNumber".split(',')
    select = format.read(select_file, numeric=True, header='id,stack,group,micrograph'.split(','))
    print "Selection: %s - %d"%(select_file, len(select))
    defocus_dict = format_utility.map_object_list(format.read(defocus_file, numeric=True, header='id,defocus,group,avg'.split(',')))
    print "Defocus-new: %s - %d"%(defocus_file, len(defocus_dict))
    param = spider_params.read(params_file)
    
    voltage = param['voltage']
    cs = param['cs']
    ampcont = param['ampcont']
    vals = []
    idlen = len(str(len(select)))
    for s in select:
        vals.append( tuple(["%s@%s"%(str(s.id).zfill(idlen), image_file), image_file, defocus_dict[s.micrograph].defocus, voltage, cs, ampcont, s.micrograph]) )
    print len(vals), len(vals[0]), len(header)
    print header
    sys.stdout.flush()
    format.write(os.path.splitext(output)[0]+".star", vals, header=header)
    
    print "Star file created"
    sys.stdout.flush()
    pixel_radius = int(param['pixel_diameter']/2.0)
    mask = None
    for i, img in enumerate(ndimage_file.iter_images(image_file)):
        if mask is None:
            mask = eman2_utility.model_circle(pixel_radius, img.shape[0], img.shape[1])*-1+1
        ndimage_utility.normalize_standard(img, mask, out=img)
        ndimage_file.write_image(output, img, i)