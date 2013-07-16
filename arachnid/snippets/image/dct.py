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
sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')
from arachnid.core.image import ndimage_file
import scipy.fftpack, numpy
#import logging

#format._logger.setLevel(logging.DEBUG)
#format.csv._logger.setLevel(logging.DEBUG)

#class2only_full.star  class6only_empty.star  ice_thickness.csv  remap.star  time_stamp.csv

if __name__ == '__main__':

    # Parameters
    input_file = sys.argv[1]
    outputfile = sys.argv[2]
    
    img = ndimage_file.read_image(input_file, 8)
    ndimage_file.write_image(outputfile, img, 0)
    dimg=scipy.fftpack.dct(scipy.fftpack.dct(img, axis=-1, norm='ortho'), norm='ortho', axis=-2)
    
    print img.shape
    print dimg.shape
    print len(dimg.ravel())
    off = numpy.asarray([0, 10, 20, 30, 40]) #*dimg.shape[0]
    for i in xrange(len(off)):
        dimg1 = dimg.copy()
        if 1 == 0:
            if 1 == 0:
                idx = numpy.argsort(dimg1.ravel())
                dimg1.ravel()[idx[:off[i]]]=0
            else:
                 dimg1[::-1,::-1][:off[i],:off[i]]=0
                 #dimg1.ravel()[::-1][:off[i]]=0
            fimg=scipy.fftpack.idct(scipy.fftpack.idct(dimg1.T).T)
        else:
            fimg = numpy.zeros(dimg.shape)
            fimg[:fimg.shape[0]-off[i], :fimg.shape[1]-off[i]]=scipy.fftpack.idct(scipy.fftpack.idct(dimg1[:fimg.shape[0]-off[i], :fimg.shape[1]-off[i]], axis=-1, norm='ortho'), axis=-2, norm='ortho')
            #fimg[:fimg.shape[0]-off[i], :fimg.shape[1]-off[i]]=scipy.fftpack.idct(scipy.fftpack.idct(dimg1[off[i]:, off[i]:], axis=-1, norm='ortho'), axis=-2, norm='ortho')
        ndimage_file.write_image(outputfile, fimg, i+1)
    if 1 == 0:
        dimg1 = dimg.copy()
        dimg1.ravel()[::-1][:10]=0
        fimg=scipy.fftpack.idct(scipy.fftpack.idct(dimg1.T).T)
        ndimage_file.write_image(outputfile, fimg, 1)
        
        dimg1 = dimg.copy()
        dimg1.ravel()[::-1][:20]=0
        fimg=scipy.fftpack.idct(scipy.fftpack.idct(dimg1.T).T)
        ndimage_file.write_image(outputfile, fimg, 2)
        
        dimg1 = dimg.copy()
        dimg1[::-1][:30]=0
        fimg=scipy.fftpack.idct(scipy.fftpack.idct(dimg1.T).T)
        ndimage_file.write_image(outputfile, fimg, 3)
        dimg1 = dimg.copy()
        dimg1[::-1][:59]=0
        fimg=scipy.fftpack.idct(scipy.fftpack.idct(dimg1.T).T)
        ndimage_file.write_image(outputfile, fimg, 4)
    

