''' Create a histogram of a set of images with discreet pixels

Download to edit and run: :download:`histogram.py <../../arachnid/snippets/histogram.py>`

To run:

.. sourcecode:: sh
    
    $ python histogram.py

.. literalinclude:: ../../arachnid/snippets/histogram.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

from arachnid.core.image import ndimage_file
from arachnid.core.metadata import format_utility
import pylab, numpy, glob, os

if __name__ == "__main__":
    
    image_file = sys.argv[1]
    output_file = sys.argv[2]
    
    vals = numpy.zeros(5)
    vals[0] = 1e20
    vals[1] = -1e20
    img_list = glob.glob(image_file)
    for image_file in img_list:
        img=ndimage_file.read_image(image_file)
        
        vals[0] = min(img.min(), vals[0])
        vals[1] = max(img.max(), vals[0])
        vals[2] += numpy.sum(img==0.0)
        vals[3] += img.ravel().shape[0]
        vals[4] += (float(numpy.sum(img==0.0))/img.ravel().shape[0])/len(img_list)
        print os.path.basename(image_file), img.min(), img.max(), numpy.sum(img==0.0), img.ravel().shape[0], float(numpy.sum(img==0.0))/img.ravel().shape[0]
    print "overall: ", vals[0], vals[1], vals[2], vals[3], vals[4]
    hist = None
    for image_file in img_list:
        img=ndimage_file.read_image(image_file)
        if hist is None:
            hist, bins = numpy.histogram(img.ravel(), 4*int(img.max()-img.min()))
        else:
            hist+=numpy.histogram(img.ravel(), bins)
    
    width = 0.7*(bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    pylab.bar(center, hist, align = 'center', width = width)
    pylab.savefig(format_utility.new_filename(output_file, ext="png"), dpi=200)
    
    #pylab.clf()
    #pylab.hist(img.ravel()[img.ravel()>10], 4*int(img.max()-img.min()))
    #pylab.savefig(format_utility.new_filename(output_file, suffix='_cut', ext="png"), dpi=200)