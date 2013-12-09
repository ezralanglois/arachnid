''' Converts Matlab data files to EMAN2 HDF

Download to edit and run: :download:`convert_matlab_to_hdf.py <../../arachnid/snippets/convert_matlab_to_hdf.py>`

To run:

.. sourcecode:: sh
    
    $ python convert_matlab_to_hdf.py data.mat images.hdf # Assumes matlab array is named data
    
    $ python convert_matlab_to_hdf.py data.mat images.hdf name_of_the_matlab_array

.. literalinclude:: ../../arachnid/snippets/convert_matlab_to_hdf.py
   :language: python
   :lines: 22-
   :linenos:
'''
import EMAN2
import sys
import scipy.io

inputfile = sys.argv[1]
outputfile = sys.argv[2]
varname = sys.argv[3] if len(sys.argv) > 3 else 'data' 

#LxLxn
mat = scipy.io.loadmat(inputfile)
data = mat[varname]

eimg = EMAN2.EMData()

for i in xrange(data.shape[2]):
    try:
        EMAN2.EMNumPy.numpy2em(data[:, :, i], eimg)
    except: eimg=EMAN2.EMNumPy.numpy2em(data[:, :, i])
    eimg.write_image(outputfile, i)


