''' Filtering with DCT

Download to edit and run: :download:`dct.py <../../arachnid/snippets/dct.py>`

To run:

.. sourcecode:: sh
    
    $ python dct.py

.. note::
    
    You must have Arachnid and Matplotlib installed to run this script

.. literalinclude:: ../../arachnid/snippets/dct.py
   :language: python
   :lines: 22-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')
from arachnid.core.image import ndimage_file
import scipy.linalg, scipy.fftpack, numpy

#format._logger.setLevel(logging.DEBUG)
#format.csv._logger.setLevel(logging.DEBUG)

#class2only_full.star  class6only_empty.star  ice_thickness.csv  remap.star  time_stamp.csv

if __name__ == '__main__':

    # Parameters
    input_file = sys.argv[1]
    outputfile = sys.argv[2]
    frac=0.9
    
    data=None
    total = ndimage_file.count_images(input_file)
    for i in xrange(total):
        img = ndimage_file.read_image(input_file)
        if data is None:
            data = numpy.zeros((total, img.ravel().shape[0]))
        data[i, :]=scipy.fftpack.dct(scipy.fftpack.dct(img, axis=-1, norm='ortho'), norm='ortho', axis=-2).ravel()
    U, d, V = scipy.linalg.svd(data, False)
    t = d**2/data.shape[0]
    t /= t.sum()
    print numpy.sum(t.cumsum()<frac)+1
        
    
   
