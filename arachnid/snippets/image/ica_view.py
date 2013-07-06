'''
.. Created on Oct 22, 2012
.. codeauthor:: robertlanglois
'''
from arachnid.core.image import ndimage_file
from arachnid.app import autoclean
from sklearn import decomposition
import numpy

def dummy(img, *args, **kwargs):
    '''
    '''
    
    return img

if __name__ == '__main__':
    stack_file = ""
    align_file = ""
    output_file=""
    
    view, label, align = autoclean.read_alignment([stack_file], align_file)[0]
    
    data = ndimage_file.read_image_mat(stack_file, label, dummy)
    
    ica = decomposition.FastICA()
    comp = ica.fit(data).transform(data)
    
    n=int(numpy.sqrt(float(comp.shape[1])))
    for i in xrange(5):
        ndimage_file.write_image(output_file, comp[i].reshape((n,n)), i)