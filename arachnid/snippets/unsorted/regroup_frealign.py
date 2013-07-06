''' Regroup MRC stacks into frealign groups

Download to edit and run: :download:`polar_bispec_pca_test.py <../../arachnid/snippets/polar_bispec_pca_test.py>`

    
Right click on link -> "Copy Link Address" or "Copy Link Location"
Download from the command line:

.. sourcecode:: sh
    
    $ wget <paste-url-here>

How to run command:

.. sourcecode:: sh
    
    $ python polar_bispec_pca_test.py data_000.spi align.spi 2 view_stack.spi

.. literalinclude:: ../../arachnid/snippets/polar_bispec_pca_test.py
   :language: python
   :lines: 23-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

from arachnid.core.metadata import format, format_utility, spider_utility
from arachnid.core.image import ndimage_file #,  ndimage_utility, eman2_utility, analysis, manifold #, rotate, eman2_utility
from arachnid.core.parallel import parallel_utility
import numpy, glob #, logging


if __name__ == '__main__':

    # Parameters
    
    image_files = glob.glob(sys.argv[1])       # image_file = "data/dala_01.spi"
    align_files = glob.glob(sys.argv[2])       # align_file = "data/align_01.spi"
    output_file=sys.argv[3]         # output_file="stack01.spi"
    
    defocus = []
    for align_file in align_files:
        #PSI   THETA     PHI     SHX     SHY    MAG   FILM      DF1      DF2
        print align_file
        defocus.extend(format.read(align_file, numeric=True, header='id,psi,theta,phi,shx,shy,mag,film,df1,df2,astig,cc'.split(',')))
    defocus=format_utility.map_object_list(defocus, key='film')
    
    total = ndimage_file.count_images(image_files)
    sizes = parallel_utility.partition_size(total, 9)
    
    index = 0
    count = 0
    outputstack = spider_utility.spider_filename(output_file, index+1)
    print 'max=', numpy.max(sizes)
    values = numpy.zeros((numpy.max(sizes)+1, 4))
    for image_file in image_files:
        id = spider_utility.spider_id(image_file)
        for img in ndimage_file.iter_images(image_file):
            try:
                values[count, :] = (id, defocus[id].df1, defocus[id].df2, defocus[id].astig)
            except:
                print 'error: ', count, len(values)
                raise
            ndimage_file.mrc.write_image(outputstack, img, count)
            count += 1
            if count > sizes[index]:
                format.write(format_utility.add_prefix(outputstack, 'sel_'), values[:count], header='film,df1,df2,astig'.split(','))
                index += 1
                count = 0
                outputstack = spider_utility.spider_filename(output_file, index+1)
    