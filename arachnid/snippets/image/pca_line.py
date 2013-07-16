''' Looks for outlier lines in an image

Download to edit and run: :download:`pca_line.py <../../arachnid/snippets/pca_line.py>`

To run:

.. sourcecode:: sh
    
    $ python pca_line.py

.. note::
    
    You must have Arachnid and Matplotlib installed to run this script

.. literalinclude:: ../../arachnid/snippets/pca_line.py
   :language: python
   :lines: 22-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')
from arachnid.core.metadata import format #, spider_utility, format_utility
from arachnid.core.image import ndimage_file
import numpy, scipy.linalg #pylab, , os
#import logging

#format._logger.setLevel(logging.DEBUG)
#format.csv._logger.setLevel(logging.DEBUG)

#class2only_full.star  class6only_empty.star  ice_thickness.csv  remap.star  time_stamp.csv

if __name__ == '__main__':

    # Parameters
    print sys.argv[0]
    input_file = sys.argv[1]
    stack_file = sys.argv[2]
    outputfile = sys.argv[3]
    
    label = format.read(input_file, ndarray=True)[0][:, :2]
    label[:, 1]-=1
    img = ndimage_file.read_image(stack_file)
    data = numpy.zeros((len(label), img.ravel().shape[0]))
    for i,img in enumerate(ndimage_file.iter_images(stack_file, label)):
        data[i, :] = img.ravel()
    data = data.reshape((img.shape[0]*len(label), img.shape[1]))
    d, V = scipy.linalg.svd(data, False)[1:]
    val = d[:2]*numpy.dot(V[:2], data.T).T
    good = numpy.zeros((len(label), img.shape[0]))
    label_new = numpy.zeros((len(label), img.shape[0], 2))
    label[:, 1]+=1
    for i in xrange(len(good)):
        label_new[i, :, 0]= label[i, 0]
        label_new[i, :, 1]= label[i, 1]
        good[i, :]=i
    label_new = label_new.reshape((data.shape[0], 2))
    format.write_dataset(outputfile, val, None, label_new, good.ravel())


