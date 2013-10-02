''' Decimate images in a stack

Download to edit and run: :download:`decimate.py <../../arachnid/snippets/decimate.py>`

To run:

.. sourcecode:: sh
    
    $ python decimate.py

.. literalinclude:: ../../arachnid/snippets/decimate.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')


from arachnid.core.metadata import spider_params, spider_utility
#from arachnid.core.metadata import format, format_utility, spider_utility
from arachnid.core.image import ndimage_file, eman2_utility, ndimage_utility
import glob

if __name__ == '__main__':

    # Parameters
    
    input_files = sys.argv[1]
    output_file = sys.argv[2]
    params_file = sys.argv[3]
    bin_factor = int(sys.argv[4])
    
    input_files = glob.glob(input_files)
    print "processing %d files"%len(input_files)
    radius = spider_params.read(params_file)['pixel_diameter']/2
    img = ndimage_file.read_image(input_files[0])
    mask = ndimage_utility.model_disk(radius, img.shape)*-1+1
    mask = eman2_utility.decimate(mask, bin_factor)
    
    for input_file in input_files:
        print 'processing ', input_file
        output_file = spider_utility.spider_filename(output_file, input_file)
        for i, img in enumerate(ndimage_file.iter_images(input_file)):
            img = eman2_utility.decimate(img, bin_factor)
            img=ndimage_utility.normalize_standard(img, mask)
            ndimage_file.write_image(output_file, img, i)





