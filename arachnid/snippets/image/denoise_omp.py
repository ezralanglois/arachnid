'''  Denoise a micrograph

Download to edit and run: :download:`denoise_omp.py <../../arachnid/snippets/denoise_omp.py>`

To run:

.. sourcecode:: sh
    
    $ python denoise_omp.py vol.spi ref_stack.spi 2

.. literalinclude:: ../../arachnid/snippets/denoise_omp.py
   :language: python
   :lines: 22-
   :linenos:
'''
import sys, numpy
#sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')
from arachnid.core.image import ndimage_file
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

#python denoise_omp.py "data/stack040411_256x256_*.stk" data/stack040411_256x256_4_fixed.par ../cluster/data/params.dat relion_stack_01.spi

if __name__ == '__main__':

    # Parameters
    image_file = sys.argv[1]
    output = sys.argv[2]
    
    distorted = ndimage_file.read_image(image_file)
    height, width = distorted.shape
    patch_size = (7,7)
    data = extract_patches_2d(distorted[:, :], patch_size)
    data = data.reshape(data.shape[0], -1)
    intercept = numpy.mean(data, axis=0)
    data -= intercept
    
    dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
    V = dico.fit(data).components_
    
    dico.set_params(transform_algorithm='omp', **{'transform_n_nonzero_coefs': 2})
    code = dico.transform(data)
    patches = numpy.dot(code, V)
    patches += intercept
    patches = patches.reshape(len(data), *patch_size)
    
    img = reconstruct_from_patches_2d( patches, (width, height))
    ndimage_file.write_image(output, img)