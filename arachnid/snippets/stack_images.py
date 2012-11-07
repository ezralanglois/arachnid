''' Stack a set of images into a set of stacks with no more than a certain number and write a selection file for each stack 

Download to edit and run: :download:`stack_images.py <../../arachnid/snippets/stack_images.py>`

To run:

.. sourcecode:: sh
    
    $ python stack_images.py

.. literalinclude:: ../../arachnid/snippets/stack_images.py
   :language: python
   :lines: 16-
   :linenos:
'''
from arachnid.core.metadata import format, format_utility, spider_utility
from arachnid.core.image import ndimage_file, eman2_utility, ndimage_utility
import glob, numpy, os


if __name__ == '__main__':

    # Parameters
    
    input_file = "mic_*.spi"
    output_file = "mic_stack_000.spi"
    stack_size = 500
    bin_factor = 0
    is_ccd=False
    
    selection = numpy.zeros((stack_size,2))
    index = 0
    stack_id = 1
    output_file = spider_utility.spider_filename(output_file, stack_id)
    for filename in glob.glob(input_file):
        img = ndimage_file.read_image(filename)
        if bin_factor > 1.0: img = eman2_utility.decimate(img, bin_factor)
        if is_ccd: img = ndimage_utility.invert(img)
        ndimage_file.write_image(output_file, img, index)
        selection[index, 0] = spider_utility.spider_id(filename)
        index += 1
        if index == stack_size:
            format.write(format_utility.add_prefix(output_file, 'sel'), selection, header=['oid', 'select'], default_format=format.spiderdoc)
            stack_id += 1
            index = 0
            output_file = spider_utility.spider_filename(output_file, stack_id)
    
    if index > 0:
        format.write(format_utility.add_prefix(output_file, 'sel'), selection[:index], header=['oid', 'select'], default_format=format.spiderdoc)
    
    outputfile=format_utility.add_prefix(output_file, 'selall')
    base, ext = os.path.splitext(outputfile)
    outputfile = base+"b"+ext
    format.write(outputfile, numpy.hstack((numpy.arange(1, stack_id+1)[:, numpy.newaxis], numpy.zeros(stack_id)[:, numpy.newaxis])), header=['id', 'select'], default_format=format.spiderdoc)
    